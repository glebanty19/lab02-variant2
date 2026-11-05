#!/usr/bin/env python3
"""Build a small, event-separated classification set from an xBD train archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import secrets
import tarfile
from pathlib import Path

from PIL import Image, ImageDraw

CLASSES = ("destroyed", "major-damage", "minor-damage", "no-damage")
SPLITS = {
    "hurricane-harvey": "train",
    "hurricane-matthew": "train",
    "hurricane-michael": "val",
    "hurricane-florence": "test",
}
PATCH_SIZE = 256
PER_CLASS_PER_EVENT = 6
MANIFEST_FIELDS = [
    "tile_id", "scene", "source", "sensor", "gsd_m", "date", "x", "y",
    "coord_ref", "source_image", "footprint_width", "footprint_height", "license",
    "image_path", "label", "label_ann1", "label_ann2", "label_ann1_v2",
    "label_ann2_v2", "uncertain", "source_object_id", "split",
]


def polygon_bounds(wkt: str) -> tuple[float, float, float, float] | None:
    coords = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", wkt)
    if len(coords) < 8 or len(coords) % 2:
        return None
    points = [(float(coords[i]), float(coords[i + 1])) for i in range(0, len(coords), 2)]
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs), max(ys)


def polygon_points(wkt: str) -> list[tuple[float, float]]:
    coords = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", wkt)
    return [
        (float(coords[i]), float(coords[i + 1]))
        for i in range(0, len(coords), 2)
    ]


def crop_origin(bounds: tuple[float, float, float, float]) -> tuple[int, int] | None:
    x0, y0, x1, y1 = bounds
    width, height = x1 - x0, y1 - y0
    if width < 10 or height < 10 or width > 180 or height > 180:
        return None
    left = max(0, min(1024 - PATCH_SIZE, round((x0 + x1 - PATCH_SIZE) / 2)))
    top = max(0, min(1024 - PATCH_SIZE, round((y0 + y1 - PATCH_SIZE) / 2)))
    if x0 < left or y0 < top or x1 > left + PATCH_SIZE or y1 > top + PATCH_SIZE:
        return None
    return left, top


def collect_candidates(archive: Path) -> dict[str, list[dict]]:
    candidates: dict[str, dict[str, dict]] = {
        event: {label: {} for label in CLASSES} for event in SPLITS
    }
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            if "/images/" in member.name and member.isfile():
                break
            if not member.isfile() or not member.name.endswith("_post_disaster.json"):
                continue
            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue
            data = json.load(fileobj)
            metadata = data.get("metadata", {})
            event = metadata.get("disaster")
            if event not in candidates:
                continue
            for feature in data.get("features", {}).get("xy", []):
                properties = feature.get("properties", {})
                label = properties.get("subtype")
                if label not in candidates[event]:
                    continue
                bounds = polygon_bounds(feature.get("wkt", ""))
                if bounds is None:
                    continue
                origin = crop_origin(bounds)
                if origin is None:
                    continue
                image_name = metadata.get("img_name")
                uid = properties.get("uid")
                if not image_name or not uid:
                    continue
                score = hashlib.sha256(f"{image_name}:{uid}".encode()).hexdigest()
                candidate = {
                    "event": event,
                    "label": label,
                    "image_name": image_name,
                    "uid": uid,
                    "bounds": bounds,
                    "wkt": feature["wkt"],
                    "origin": origin,
                    "metadata": metadata,
                    "score": score,
                }
                previous = candidates[event][label].get(image_name)
                if previous is None or score < previous["score"]:
                    candidates[event][label][image_name] = candidate
    return {
        event: [entry for label in CLASSES for entry in candidates[event][label].values()]
        for event in SPLITS
    }


def select_candidates(candidates: dict[str, list[dict]]) -> list[dict]:
    selected: list[dict] = []
    for event in SPLITS:
        used_images: set[str] = set()
        for label in CLASSES:
            options = sorted(
                (item for item in candidates[event] if item["label"] == label),
                key=lambda item: item["score"],
            )
            chosen = []
            for item in options:
                if item["image_name"] in used_images:
                    continue
                chosen.append(item)
                used_images.add(item["image_name"])
                if len(chosen) == PER_CLASS_PER_EVENT:
                    break
            if len(chosen) != PER_CLASS_PER_EVENT:
                raise ValueError(
                    f"{event}/{label}: знайдено {len(chosen)} унікальних сцен "
                    f"з {PER_CLASS_PER_EVENT} потрібних."
                )
            selected.extend(chosen)
    return selected


def extract_patches(archive: Path, selected: list[dict], output_dir: Path) -> None:
    by_image = {item["image_name"]: item for item in selected}
    images_dir = output_dir / "images"
    annotation_images_dir = output_dir / "annotation_images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotation_images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    pending = set(by_image)
    rows = []
    source_annotations = []
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            image_name = member.name.rsplit("/", 1)[-1]
            if image_name not in pending or "/images/" not in member.name or not member.isfile():
                continue
            fileobj = tar.extractfile(member)
            if fileobj is None:
                continue
            source = by_image[image_name]
            image = Image.open(io.BytesIO(fileobj.read())).convert("RGB")
            if image.size != (1024, 1024):
                raise ValueError(f"Очікувався фрагмент 1024x1024: {member.name} ({image.size})")
            left, top = source["origin"]
            tile_id = (
                f"{source['event']}_{source['label'].replace('-', '_')}_"
                f"{source['uid'][:8]}"
            )
            relative_image = f"images/{tile_id}.png"
            image.crop((left, top, left + PATCH_SIZE, top + PATCH_SIZE)).save(
                output_dir / relative_image, format="PNG"
            )
            metadata = source["metadata"]
            rows.append(
                {
                    "tile_id": tile_id,
                    "scene": source["event"],
                    "source": "xBD Challenge train set",
                    "sensor": metadata.get("provider_asset_type", metadata.get("sensor", "")),
                    "gsd_m": metadata.get("gsd"),
                    "date": str(metadata.get("capture_date", ""))[:10],
                    "x": left,
                    "y": top,
                    "coord_ref": f"{source['image_name']} (source-image pixels)",
                    "source_image": source["image_name"],
                    "footprint_width": PATCH_SIZE,
                    "footprint_height": PATCH_SIZE,
                    "license": "xBD challenge terms; source images are not redistributed",
                    "image_path": relative_image,
                    "label": source["label"],
                    "label_ann1": "",
                    "label_ann2": "",
                    "label_ann1_v2": "",
                    "label_ann2_v2": "",
                    "uncertain": "",
                    "source_object_id": source["uid"],
                    "split": SPLITS[source["event"]],
                }
            )
            x0, y0, x1, y1 = source["bounds"]
            source_annotations.append(
                {
                    "tile_id": tile_id,
                    "source_image": source["image_name"],
                    "source_object_id": source["uid"],
                    "source_label": source["label"],
                    "source_polygon_wkt": source["wkt"],
                    "bbox_x0": x0,
                    "bbox_y0": y0,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "crop_x": left,
                    "crop_y": top,
                }
            )
            pending.remove(image_name)
            if not pending:
                break
    if pending:
        raise FileNotFoundError(f"Архів не містить зображень для {len(pending)} вибраних сцен.")
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (labels_dir / "source_annotations.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(source_annotations[0]))
        writer.writeheader()
        writer.writerows(source_annotations)
    labels_by_tile = {row["tile_id"]: row["label"] for row in rows}
    scenes_by_tile = {row["tile_id"]: row["scene"] for row in rows}
    queue_rows = []
    for label in CLASSES:
        for event in SPLITS:
            event_rows = [
                row for row in rows if row["label"] == label and row["scene"] == event
            ]
            for row in event_rows[:2]:
                queue_rows.append(
                    {
                        "sample_id": secrets.token_hex(6),
                        "tile_id": row["tile_id"],
                        "annotator_1_pre": "",
                        "annotator_2_pre": "",
                        "annotator_1_post": "",
                        "annotator_2_post": "",
                        "uncertain": "",
                        "repeat_after_guideline": 0,
                        "arbitrated_label": "",
                    }
                )
    labels_by_tile = {row["tile_id"]: row["label"] for row in rows}
    repeat_ids = set()
    queue_rows.sort(key=lambda row: row["sample_id"])
    for label in CLASSES:
        matching = [
            row["tile_id"]
            for row in queue_rows
            if labels_by_tile[row["tile_id"]] == label
        ]
        repeat_ids.update(matching[:2])
    for label in CLASSES[:2]:
        repeat_ids.add(
            next(
                row["tile_id"]
                for row in queue_rows
                if labels_by_tile[row["tile_id"]] == label
                and row["tile_id"] not in repeat_ids
            )
        )
    for row in queue_rows:
        row["repeat_after_guideline"] = int(row["tile_id"] in repeat_ids)
        row["image_path"] = f"annotation_images/{row['sample_id']}.png"
        source_row = next(item for item in rows if item["tile_id"] == row["tile_id"])
        annotation_image = Image.open(output_dir / source_row["image_path"]).convert("RGB")
        points = [
            (round(x - source_row["x"]), round(y - source_row["y"]))
            for x, y in polygon_points(
                next(item["source_polygon_wkt"] for item in source_annotations
                     if item["tile_id"] == row["tile_id"])
            )
        ]
        if len(points) >= 3:
            ImageDraw.Draw(annotation_image).line(
                points + [points[0]], fill=(0, 255, 255), width=2
            )
        annotation_image.save(output_dir / row["image_path"], format="PNG")
    with (labels_dir / "double_annotation_queue.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id",
                "image_path",
                "annotator_1_pre",
                "annotator_2_pre",
                "annotator_1_post",
                "annotator_2_post",
                "uncertain",
                "repeat_after_guideline",
                "arbitrated_label",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(queue_rows)
    with (labels_dir / "source_annotation_key.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sample_id", "tile_id", "source_label", "scene"],
        )
        writer.writeheader()
        writer.writerows(
            {
                "sample_id": queue_row["sample_id"],
                "tile_id": queue_row["tile_id"],
                "source_label": labels_by_tile[queue_row["tile_id"]],
                "scene": scenes_by_tile[queue_row["tile_id"]],
            }
            for queue_row in queue_rows
        )
    print(f"Збережено {len(rows)} фрагментів; архівні оригінали не розпаковувалися.")
    print(f"Класи/підмножини: {output_dir / 'manifest.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    selected = select_candidates(collect_candidates(args.archive))
    print(f"Відібрано {len(selected)} цільових будівель із {len(SPLITS)} різних подій.")
    extract_patches(args.archive, selected, args.output_dir)


if __name__ == "__main__":
    main()
