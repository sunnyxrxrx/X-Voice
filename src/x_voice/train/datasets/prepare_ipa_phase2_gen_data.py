import os
import sys
import argparse
import json
import shutil
import multiprocessing
import re
import regex
from pathlib import Path
from typing import List, Union, Pattern
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import logging

import torchaudio
from tqdm import tqdm
from datasets.arrow_writer import ArrowWriter

from ipa_v3_tokenizer import PhonemizeTextTokenizer as PhonemizeTextTokenizer_v3
from ipa_v6_tokenizer import PhonemizeTextTokenizer as PhonemizeTextTokenizer_v6
from x_voice.model.utils import get_ipa_id
logger = logging.getLogger("phonemizer")
logger.setLevel(logging.ERROR)
logger.propagate = False

# import debugpy
# debugpy.listen(('localhost', 568))
# print("Waiting for debugger attach")
# debugpy.wait_for_client()

sys.path.append(os.getcwd())

TOKENIZERS = {}
def get_tokenizer(lang_code, tokenizer):
    espeak_code = get_ipa_id(lang_code)
    if espeak_code not in TOKENIZERS:
        try:
            if tokenizer == "ipa_v3":
                TOKENIZERS[espeak_code] = PhonemizeTextTokenizer_v3(language=espeak_code)
            elif tokenizer == "ipa_v6":
                TOKENIZERS[espeak_code] = PhonemizeTextTokenizer_v6(language=espeak_code, with_stress=True)
        except RuntimeError as e:
            print(f"Error initializing espeak for {espeak_code}: {e}")
            return None
    return TOKENIZERS[espeak_code]



def process_batch(batch_data, lang_code, tokenizer_str):
    """
    batch_data: list[(audio_path, ref_text, gen_text, duration)]
    """
    tokenizer = get_tokenizer(lang_code, tokenizer=tokenizer_str)
    if tokenizer is None:
        return []

    # Extract reference and generated texts separately.
    ref_texts = [item[1] for item in batch_data]
    gen_texts = [item[2] for item in batch_data]
    
    try:
        ref_ipa_texts = [tokenizer([text.strip()]) for text in ref_texts]
        gen_ipa_texts = [tokenizer([text.strip()]) for text in gen_texts]
    except Exception as e:
        print(f"IPA conversion failed for batch in {lang_code}: {e}")
        return []
    
    target_sample_rate = 24000
    hop_length = 256
    results = []
    for (audio_path, ref_text, gen_text, duration), ref_ipa, gen_ipa in zip(batch_data, ref_ipa_texts, gen_ipa_texts):
        if not ref_ipa.strip() or not gen_ipa.strip():
            continue

        ref_text_len = len(ref_text.encode("utf-8"))
        gen_text_len = len(gen_text.encode("utf-8"))
        # Avoid division-by-zero issues.
        if ref_text_len == 0: continue
        # Estimate the combined mel length.
        ref_mel_len = int(duration * target_sample_rate / hop_length)
        total_mel_len = ref_mel_len + int(ref_mel_len / ref_text_len * gen_text_len)
        # Keep the relative path without its suffix.
        rel_path = audio_path
        results.append({
            "ref_text": ref_text,
            "gen_text": gen_text,
            "ref_text_ipa": ref_ipa,
            "gen_text_ipa": gen_ipa,
            "duration": duration,
            "total_mel_len": total_mel_len,
            "language_id": lang_code,
            "rel_path": rel_path
        })
    return results

def read_all_metadata(input_dir):
    """
    Scan all metadata_*.csv files under the input directory.
    """
    input_path = Path(input_dir)/'csv_stage2_debug'
    # Match files such as metadata_zh.csv and metadata_en.csv.
    all_files = list(input_path.glob("metadata_*_test.csv"))
    csv_files = []
    for f in all_files:
        csv_files.append(f)
    
    if not csv_files:
        print(f"No proper csv files found in {input_dir}")
        sys.exit(1)
        
    print(f"Found {len(csv_files)} metadata files: {[f.name for f in csv_files]}")
    all_tasks = [] # (lang_code, file_path)
    
    for csv_file in csv_files:
        # Extract the language code from the file name, e.g. metadata_zh.csv -> zh.
        lang_code = csv_file.stem.split('_')[1]
        if True:
            print(f"lang_code:{lang_code}")
            all_tasks.append((lang_code, csv_file))
        
    return all_tasks


def read_csv_file(csv_path, target_duration=None):
    items = []
    all_duration = 0
    # Step 1: collect all valid rows with durations into a temporary list.
    temp_valid_lines = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip()  # Skip the header row.
        for line in f:
            parts = line.strip().split('|')
            if len(parts) in [3, 4]:  # Valid path|duration|text row, with optional DNSMOS field.
                # Store the raw path, text, and duration first.
                temp_valid_lines.append((parts[0], parts[2], float(parts[1])))
            elif len(parts) == 2:  # Row without a duration value.
                print("Warning: no duration. Check the metadata file")

                
    
    # Step 2: shuffle valid rows only when a target duration is provided.
    if target_duration is not None:
        random.shuffle(temp_valid_lines)  # Shuffle valid rows.
    
    # Step 3: accumulate duration until the target is reached.
    for path, text, duration in temp_valid_lines:
        # Stop once the requested duration budget has been exceeded.
        if target_duration is not None and all_duration > target_duration * 3600:
            break
        items.append((path, text, duration))
        all_duration += duration
    
    return items, all_duration/3600

def prepare_all(inp_dir, out_dir_root, tokenizer, dataset_name, num_workers=16, duration_map=None):
    inp_dir = Path(inp_dir)
    out_dir_root = Path(out_dir_root)
    out_dir = out_dir_root / f"{dataset_name}_{tokenizer}_gp"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Will be saved to {out_dir}")
    
    tasks = read_all_metadata(inp_dir) # List[(lang, csv_path)]
    
    raw_arrow_path = out_dir / "raw.arrow"
    writer = ArrowWriter(path=raw_arrow_path.as_posix(), writer_batch_size=10000)
    
    total_samples = 0
    duration_list = []
    mel_duration_list = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for lang_code, csv_path in tasks:
            print(f"\nProcessing Language: {lang_code} (from {csv_path.name})")
            lang_duration = None
            if duration_map is not None:
                lang_duration = duration_map.get(lang_code, None)
                if lang_duration is None:
                    lang_duration = duration_map.get("default", None)
                print(f"Will choose {lang_duration} hours for {lang_code}")
            
            import bisect
            raw_items, return_duration = read_csv_file(csv_path, lang_duration)
            if not raw_items: continue
            
            # 1. Build a sorted text pool as (text, byte_length) pairs.
            # Sorting lets us use binary search for the target range.
            text_pool = sorted(
                [(t, len(t.encode("utf-8"))) for _, t, _ in raw_items],
                key=lambda x: x[1]
            )
            pool_lengths = [x[1] for x in text_pool]  # Extract the length list for binary search.
            
            fixed_items = []
            chunck_items = 0
            for p, t, d in raw_items:
                ref_len = len(t.encode("utf-8"))
                if ref_len == 0: continue
                
                # Target interval: [min_b, max_b]
                min_b, max_b = ref_len * 0.1, ref_len * 0.4
                
                # 2. Locate the valid index range with binary search.
                start_idx = bisect.bisect_left(pool_lengths, min_b)
                end_idx = bisect.bisect_right(pool_lengths, max_b)
                
                gen_t = None
                if start_idx < end_idx:
                    # If the interval exists, sample one item randomly for diversity.
                    gen_t = text_pool[random.randint(start_idx, end_idx - 1)][0]
                else:
                    # 3. Fallback: truncate the nearest longer sentence if no short one exists.
                    # Find the first sentence longer than max_b.
                    chunck_items += 1
                    trunc_idx = bisect.bisect_left(pool_lengths, max_b)
                    if trunc_idx < len(text_pool):
                        rand_idx = random.randint(trunc_idx, len(text_pool) - 1)
                        long_cand = text_pool[rand_idx][0]
                        # Truncate by words for alphabetic languages and by characters for CJK-like text.
                        words = long_cand.split()
                        if lang_code not in ["zh", "ja", "ko"]:
                            tmp_words, curr_b = [], 0
                            for w in words:
                                w_b = len((w + " ").encode("utf-8"))
                                if curr_b + w_b > max_b: break
                                tmp_words.append(w); curr_b += w_b
                            gen_t = " ".join(tmp_words).strip()
                        else:
                            tmp_chars, curr_b = "", 0
                            for char in long_cand:
                                c_b = len(char.encode("utf-8"))
                                if curr_b + c_b > max_b: break
                                tmp_chars += char; curr_b += c_b
                            gen_t = tmp_chars
                
                # Keep samples only when a non-empty generated text is available.
                if gen_t and len(gen_t.encode("utf-8")) >= 1: # At least one byte.
                    fixed_items.append((p, t, gen_t, d))
          
            print(f"Loaded {len(fixed_items)} lines. Duration: {return_duration}. Starting G2P conversion.")
            
            batch_size = 1000 # Each worker processes 1000 rows at a time.
            batches = [fixed_items[i:i + batch_size] for i in range(0, len(fixed_items), batch_size)]
            
            futures = [executor.submit(process_batch, batch, lang_code, tokenizer) for batch in batches]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"  -> {lang_code}"):
                batch_results = future.result()
                
                for res in batch_results:
                    writer.write(res)
                    mel_duration_list.append(res['total_mel_len'])
                    duration_list.append(res["duration"])
                    total_samples += 1
    writer.finalize()
    
    total_duration = sum(duration_list)
    with open(out_dir / "duration.json", "w", encoding="utf-8") as f:
        json.dump({
            "duration": mel_duration_list, 
        }, f, ensure_ascii=False)
            
    print("\n" + "="*50)
    print(f"Total Samples: {total_samples}")
    print(f"Total Hours:   {total_duration/3600:.2f}")
    print(f"Saved to:      {out_dir}")
    print("="*50)

def main():
    parser = argparse.ArgumentParser()
    support_tokenizer = ["ipa_v3", "ipa_v6"]
    parser.add_argument("--inp_dir", type=str, default="/inspire/hdd/project/embodied-multimodality/chenxie-25019/qingyuliu/datasets",help="Root dir containing metadata_*.csv and wavs/")
    parser.add_argument("--out_dir", type=str, default="/inspire/hdd/project/embodied-multimodality/chenxie-25019/qingyuliu/github/XVtest/data",help="Output root dir for raw.arrow")
    parser.add_argument("--workers", type=int, default=16, help="Number of CPU workers")
    parser.add_argument("--tokenizer",type=str, choices=support_tokenizer, default="ipa_v6")
    parser.add_argument("--dataset_name",type=str, required=True)
    
    
    args = parser.parse_args()
    duration_map=None
    
    prepare_all(args.inp_dir, args.out_dir, args.tokenizer, args.dataset_name, args.workers, duration_map)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
    
# python src/x_voice/train/datasets/prepare_ipa_phase2_gen_data.py --tokenizer ipa_v6 --dataset_name multilingual_qyl_test
