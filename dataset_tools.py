#!/usr/bin/env python3
"""Validate and summarize the variant 2 image-fragment dataset."""

from __future__ import annotations

import argparse
import hashlib
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import GroupShuffleSplit

CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
REQUIRED = [
    "tile_id", "scene", "source", "sensor", "gsd_m", "date", "x", "y",
    "coord_ref", "license", "image_path", "label",
]


def dhash(path: Path, size: int = 8) -> int:
    image = Image.open(path).convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(image)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(first: int, second: int) -> int:
    return (first ^ second).bit_count()


def report_agreement(
    df: pd.DataFrame, first: str, second: str, title: str, minimum: int = 30
) -> None:
    if first not in df or second not in df:
        return
    paired = df[[first, second]].dropna()
    paired = paired[(paired[first].astype(str) != "") & (paired[second].astype(str) != "")]
    if paired.empty:
        print(f"{title}: немає парних міток.")
        return
    print(
        f"{title}: n={len(paired)}, "
        f"kappa={cohen_kappa_score(paired[first], paired[second], labels=CLASSES):.4f}"
    )
    if len(paired) < minimum:
        print(f"ПОПЕРЕДЖЕННЯ: потрібно щонайменше {minimum} парних зразків; наявно {len(paired)}.")


def grouped_split(df: pd.DataFrame, seed: int) -> pd.Series:
    groups = df["scene"].astype(str)
    if groups.nunique() < 3:
        raise ValueError("Для трьох підмножин потрібні щонайменше 3 незалежні сцени.")
    first = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, remainder_idx = next(first.split(df, groups=groups))
    remainder = df.iloc[remainder_idx]
    second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_local, test_local = next(second.split(remainder, groups=groups.iloc[remainder_idx]))
    result = pd.Series(index=df.index, dtype="object")
    result.iloc[train_idx] = "train"
    result.iloc[remainder_idx[val_local]] = "val"
    result.iloc[remainder_idx[test_local]] = "test"
    return result


def same_scene_neighbor_rate(train: pd.DataFrame, other: pd.DataFrame) -> float:
    if other.empty:
        return 0.0
    return float(other["scene"].astype(str).isin(set(train["scene"].astype(str))).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("stats"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--annotations",
        type=Path,
        help="CSV з незалежними мітками двох учасників до/після уточнення інструкції",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.manifest, dtype={"tile_id": str, "scene": str})
    missing_columns = [column for column in REQUIRED if column not in df.columns]
    if missing_columns:
        raise ValueError(f"У маніфесті відсутні обов'язкові стовпці: {missing_columns}")
    if df.empty:
        raise ValueError("Маніфест не містить зразків; додайте фактичні дані.")
    if df["tile_id"].isna().any() or df["tile_id"].duplicated().any():
        raise ValueError("tile_id має бути заповненим і унікальним.")
    if df[["scene", "label"]].isna().any().any():
        raise ValueError("Поля scene і label мають бути заповненими для кожного зразка.")

    if not 80 <= len(df) <= 200:
        print(f"ПОПЕРЕДЖЕННЯ: потрібно 80–200 зразків; наявно {len(df)}.")
    if df["scene"].nunique() < 4:
        print(f"ПОПЕРЕДЖЕННЯ: потрібно щонайменше 4 сцени; наявно {df['scene'].nunique()}.")
    invalid_labels = sorted(set(df["label"].dropna()) - set(CLASSES))
    if invalid_labels:
        raise ValueError(f"Невідомі значення label: {invalid_labels}")

    paths = [(tile_id, args.manifest.parent / image_path) for tile_id, image_path in
             zip(df["tile_id"], df["image_path"].fillna(""))]
    absent = [(tile_id, path) for tile_id, path in paths if not path.is_file()]
    if absent:
        print(f"ПОПЕРЕДЖЕННЯ: відсутні файли зображень: {len(absent)}; перший: {absent[0]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, tuple[str, int]] = {}
    for tile_id, path in paths:
        if path.is_file():
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[tile_id] = (sha, dhash(path))

    exact_pairs = []
    near_pairs = []
    for first, second in itertools.combinations(hashes, 2):
        sha_a, hash_a = hashes[first]
        sha_b, hash_b = hashes[second]
        if sha_a == sha_b:
            exact_pairs.append((first, second))
        elif hamming(hash_a, hash_b) <= 6:
            near_pairs.append((first, second, hamming(hash_a, hash_b)))

    pd.DataFrame(exact_pairs, columns=["tile_id_a", "tile_id_b"]).to_csv(
        args.output_dir / "exact_duplicates.csv", index=False
    )
    pd.DataFrame(near_pairs, columns=["tile_id_a", "tile_id_b", "hamming_distance"]).to_csv(
        args.output_dir / "perceptual_candidates.csv", index=False
    )
    print(f"Точні дублікати: {len(exact_pairs)}; кандидати dHash (відстань <=6): {len(near_pairs)}.")

    report_agreement(df, "label_ann1", "label_ann2", "Узгодженість до уточнення")
    report_agreement(df, "label_ann1_v2", "label_ann2_v2", "Узгодженість після уточнення")
    if args.annotations:
        annotations = pd.read_csv(args.annotations)
        report_agreement(
            annotations, "annotator_1_pre", "annotator_2_pre",
            "Незалежне анотування до уточнення",
        )
        repeated = annotations[
            annotations["repeat_after_guideline"].fillna(0).astype(int) == 1
        ]
        report_agreement(
            repeated, "annotator_1_post", "annotator_2_post",
            "Повторне анотування після уточнення", minimum=10,
        )

    predefined = df["split"].dropna().astype(str) if "split" in df else pd.Series(dtype=str)
    if "split" in df and len(predefined) == len(df) and set(predefined).issubset(
        {"train", "val", "test"}
    ):
        split = df["split"].astype(str)
        scene_split_counts = df.assign(_split=split).groupby("scene")["_split"].nunique()
        if (scene_split_counts > 1).any():
            raise ValueError("Виявлено витік: одна сцена призначена різним підмножинам.")
        print("Використано заздалегідь задане групове розбиття з маніфесту.")
    else:
        split = grouped_split(df, args.seed)
    df["split"] = split
    split_scenes = df.groupby("split")["scene"].nunique().to_dict()
    print(f"Сцени за підмножинами: {split_scenes}")
    for name in ("train", "val", "test"):
        subset = df[df["split"] == name]
        represented = set(subset["label"])
        missing = sorted(set(CLASSES) - represented)
        if missing:
            print(f"ПОПЕРЕДЖЕННЯ: у {name} відсутні класи {missing}; потрібен перегляд добору сцен.")

    if "source_image" in df:
        source_splits = df.groupby("source_image")["split"].nunique()
        shared_sources = int((source_splits > 1).sum())
        print(
            f"Вихідних зображень, спільних між підмножинами: {shared_sources} "
            f"(унікальних у наборі: {df['source_image'].nunique()})."
        )
        if shared_sources:
            raise ValueError("Виявлено витік: вихідне зображення потрапило до кількох підмножин.")

    shuffled = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    n_train = int(len(shuffled) * 0.70)
    n_val = int(len(shuffled) * 0.15)
    random_train = shuffled.iloc[:n_train]
    random_val = shuffled.iloc[n_train:n_train + n_val]
    random_test = shuffled.iloc[n_train + n_val:]
    print(
        "Частка випадкових val/test зразків зі сценою у train: "
        f"val={same_scene_neighbor_rate(random_train, random_val):.3f}, "
        f"test={same_scene_neighbor_rate(random_train, random_test):.3f}"
    )
    if {"footprint_width", "footprint_height", "source_image"}.issubset(df.columns):
        overlaps = []
        valid = df.dropna(
            subset=["source_image", "x", "y", "footprint_width", "footprint_height"]
        )
        for source_image, scene_rows in valid.groupby("source_image"):
            for (_, a), (_, b) in itertools.combinations(scene_rows.iterrows(), 2):
                if a["split"] == b["split"] or a["coord_ref"] != b["coord_ref"]:
                    continue
                intersects = (
                    a["x"] < b["x"] + b["footprint_width"]
                    and a["x"] + a["footprint_width"] > b["x"]
                    and a["y"] < b["y"] + b["footprint_height"]
                    and a["y"] + a["footprint_height"] > b["y"]
                )
                if intersects:
                    overlaps.append((source_image, a["tile_id"], b["tile_id"]))
        pd.DataFrame(overlaps, columns=["source_image", "tile_id_a", "tile_id_b"]).to_csv(
            args.output_dir / "spatial_overlaps.csv", index=False
        )
        print(f"Перетини плиток з одного вихідного зображення між підмножинами: {len(overlaps)}.")
    else:
        print("Координатне перекриття не перевірено: додайте footprint_width/footprint_height.")

    df[["tile_id", "scene", "split"]].to_csv(args.manifest.parent / "split.csv", index=False)
    counts = df.groupby(["split", "label"]).size().unstack(fill_value=0).reindex(
        columns=CLASSES, fill_value=0
    )
    counts.to_csv(args.output_dir / "class_counts.csv")
    df.groupby(["split", "scene"]).size().rename("count").to_csv(
        args.output_dir / "scene_counts.csv"
    )
    checksum_rows = []
    for tile_id, path in paths:
        if path.is_file():
            checksum_rows.append(
                {
                    "tile_id": tile_id,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    pd.DataFrame(checksum_rows).to_csv(args.output_dir / "checksums.csv", index=False)
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    (args.output_dir / "manifest.sha256").write_text(
        f"{manifest_hash}  {args.manifest.name}\n", encoding="ascii"
    )
    print(f"Збережено {args.manifest.parent / 'split.csv'} та статистику в {args.output_dir}.")


if __name__ == "__main__":
    main()
