import argparse
from collections import Counter
import json
import random
import re
import unicodedata
from pathlib import Path

import matplotlib

from datasets.arrow_writer import ArrowWriter
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from srp.model.utils import count_syllables, extract_pyphen_text

TRAIN_HOURS_PER_LANG = 250
VAL_SAMPLES_PER_LANG = 100
VALID_PUNCTUATION = '\'",.?!;:。，、！？；：「」『』【】-'

def check_valid_chars(input_str: str, lang) -> bool:
    for c in input_str:
        if lang == "th":
            if unicodedata.category(c).startswith(('L','M')):
                continue
        else:
            if c.isalpha():
                continue
        if c in VALID_PUNCTUATION:
            continue
        if c.isspace():
            continue
        return False
    return True

def map_to_class(speed: float, delta: float = 0.25) -> float:
    return round(speed / delta) * delta


def read_all_metadata(input_dir: str):
    input_path = Path(input_dir) / "csvs"
    csv_files = list(input_path.glob("metadata_*_top_1000.0h.csv"))

    if not csv_files:
        print(f"No proper csv files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(csv_files)} metadata files: {[f.name for f in csv_files]}")
    all_tasks = []
    for csv_file in csv_files:
        lang_code = csv_file.stem.split("_")[1]
        print(f"lang_code: {lang_code}")
        all_tasks.append((lang_code, csv_file))
    return all_tasks


def read_csv_file(csv_path: Path):
    items = []
    with open(csv_path, "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) == 3:
                audio_path, duration, text = parts
            elif len(parts) == 4:
                audio_path, duration, text, _ = parts
            try:
                duration = float(duration)
            except ValueError:
                continue
            items.append((audio_path, text, duration))
    return items


def build_sample(audio_path: str, text: str, duration: float, lang_code: str):
    if duration <= 0 or not check_valid_chars(text, lang=lang_code):
        return None

    syllable_count = count_syllables(text, lang_code)
    speed_syllables = map_to_class(syllable_count / duration)
    if syllable_count <= 0 or speed_syllables > 8.0:
        return None

    return {
        "audio_path": audio_path,
        "text": text,
        "duration": duration,
        "speed_syllables": speed_syllables,
        "lang": lang_code,
    }


def select_split_items(raw_items, lang_code: str, rng: random.Random, inp_dir: Path):
    shuffled_items = list(raw_items)
    rng.shuffle(shuffled_items)

    train_results = []
    val_results = []
    train_durations = []
    val_durations = []
    train_duration_sum = 0.0
    last_print_duration = 0
    PRINT_INTERVAL_SEC = 10 * 3600  
    for audio_path, text, duration in shuffled_items:
        audio_dir = inp_dir / "wavs" / audio_path
        if not audio_dir.exists():
            print(f"{audio_dir} not exists")
            continue
        sample = build_sample(audio_path, text, duration, lang_code)
        
        if sample is None:
            continue

        if len(val_results) < VAL_SAMPLES_PER_LANG:
            val_results.append(sample)
            val_durations.append(duration)
            continue

        if train_duration_sum < TRAIN_HOURS_PER_LANG * 3600:
            train_results.append(sample)
            train_durations.append(duration)
            train_duration_sum += duration
            if train_duration_sum - last_print_duration >= PRINT_INTERVAL_SEC:
                collected_hours = train_duration_sum / 3600
                print(collected_hours)
                last_print_duration = train_duration_sum

        if len(val_results) >= VAL_SAMPLES_PER_LANG and train_duration_sum >= TRAIN_HOURS_PER_LANG * 3600:
            break

    return train_results, train_durations, val_results, val_durations


def write_arrow(output_path: Path, rows):
    with ArrowWriter(path=output_path.as_posix(), writer_batch_size=10000) as writer:
        for row in rows:
            writer.write(row)
        writer.finalize()


def write_duration_json(output_path: Path, durations):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"duration": durations}, f, ensure_ascii=False)


def write_speed_syllables_stats(out_dir: Path, split_name: str, rows, write_histogram: bool):
    counter = Counter(row["speed_syllables"] for row in rows)
    sorted_items = sorted(counter.items(), key=lambda x: x[0])
    counts = {str(k): v for k, v in sorted_items}

    with open(
        out_dir / f"speed_syllables_counts_{split_name}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(counts, f, ensure_ascii=False, indent=2)

    if write_histogram:
        plt.figure(figsize=(12, 5))
        plt.bar([str(k) for k, _ in sorted_items], [v for _, v in sorted_items])
        plt.xlabel("speed_syllables")
        plt.ylabel("count")
        plt.title(f"speed_syllables histogram ({split_name})")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_dir / f"speed_syllables_hist_{split_name}.png", dpi=200)
        plt.close()


def prepare_all(inp_dir: str, out_dir_root: str, dataset_name: str, seed: int = 42):
    inp_dir = Path(inp_dir)
    out_dir_root = Path(out_dir_root)
    out_dir = Path(f"{out_dir_root / dataset_name}_srp")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Will be saved to {out_dir}")

    tasks = read_all_metadata(inp_dir)

    train_rows = []
    val_rows = []
    train_durations = []
    val_durations = []

    for idx, (lang_code, csv_path) in enumerate(tasks):
        print(f"\nProcessing Language: {lang_code} (from {csv_path.name})")
        raw_items = read_csv_file(csv_path)
        print(f"Loaded {len(raw_items)} rows before filtering.")

        rng = random.Random(seed + idx)
        lang_train, lang_train_durations, lang_val, lang_val_durations = select_split_items(
            raw_items, lang_code, rng, inp_dir
        )

        train_rows.extend(lang_train)
        train_durations.extend(lang_train_durations)
        val_rows.extend(lang_val)
        val_durations.extend(lang_val_durations)

        print(
            f"Selected train={len(lang_train)} samples ({sum(lang_train_durations) / 3600:.2f}h), "
            f"val={len(lang_val)} samples"
        )

    write_arrow(out_dir / "raw.arrow", train_rows)
    write_arrow(out_dir / "raw_val.arrow", val_rows)
    write_duration_json(out_dir / "duration.json", train_durations)
    write_duration_json(out_dir / "duration_val.json", val_durations)
    write_speed_syllables_stats(out_dir, "train", train_rows, write_histogram=True)

    print("\n" + "=" * 50)
    print(f"Train samples: {len(train_rows)}")
    print(f"Train hours:   {sum(train_durations) / 3600:.2f}")
    print(f"Val samples:   {len(val_rows)}")
    print(f"Val hours:     {sum(val_durations) / 3600:.2f}")
    print(f"Saved to:      {out_dir}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inp_dir",
        type=str,
        default="/inspire/hdd/project/embodied-multimodality/chenxie-25019/rixixu/datasets",
        help="Root dir containing csv_train/metadata_*_full.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[4] / "data"),
        help="Output root dir for SRP data",
    )
    parser.add_argument("--dataset_name", type=str, default="multilingual_250_100")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_all(args.inp_dir, args.out_dir, args.dataset_name, args.seed)


if __name__ == "__main__":
    main()

# python src/train/datasets/prepare_multilingual_speed.py --dataset_name multilingual_250_100_v2
