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
# debugpy.listen(('localhost', 5698))
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

def process_batch(batch_data, lang_code, tokenizer_str, sft_gen_dir):
    """
    batch_data: list[(audio_path, text, duration)]
    """
    local_fail = 0
    tokenizer = get_tokenizer(lang_code, tokenizer=tokenizer_str)
    if tokenizer is None:
        return []

    texts = [item[1] for item in batch_data]
    
    try:
        ipa_texts = [tokenizer([text.strip()]) for text in texts]
    except Exception as e:
        print(f"IPA conversion failed for batch in {lang_code}: {e}")
        return []

    results =[]
    for (audio_path, _, duration), ipa_text in zip(batch_data, ipa_texts):
        if not ipa_text.strip():
            continue
        
        # Extract the relative path without its suffix (for example zh/xxx).
        rel_path_no_suffix = os.path.splitext(audio_path)[0]
        
        # Build the matching .pt and .json paths.
        pt_path = os.path.join(sft_gen_dir, f"{rel_path_no_suffix}.pt")
        json_path = os.path.join(sft_gen_dir, f"{rel_path_no_suffix}.json")
        
        if not (os.path.exists(pt_path) and os.path.exists(json_path)):
            local_fail += 1
            continue
            
        # Read gen_len from the JSON metadata.
        try:
            with open(json_path, "r", encoding="utf-8") as jf:
                meta = json.load(jf)
                gen_len = meta["gen_len"]
                prompt_text = meta["text_ipa"]
        except Exception as e:
            print(f"Error reading JSON {json_path}: {e}")
            continue

        results.append({
            "audio_path": audio_path,
            "text": ipa_text,     
            "total_text": prompt_text + "_ _" + ipa_text,
            "duration": duration,
            "language_id": lang_code,
            "prompt_path": os.path.abspath(pt_path),
            "prompt_frames": gen_len,
        })
    return results, local_fail

def read_all_metadata(input_dir):
    input_path = Path(input_dir)/'csv_stage2_debug'
    all_files = list(input_path.glob("metadata_*_test.csv"))
    csv_files =[]
    for f in all_files:
        csv_files.append(f)
    
    if not csv_files:
        print(f"No proper csv files found in {input_dir}")
        sys.exit(1)
        
    print(f"Found {len(csv_files)} metadata files: {[f.name for f in csv_files]}")
    all_tasks =[] 
    
    for csv_file in csv_files:
        lang_code = csv_file.stem.split('_')[1]
        if True: #lang_code in ["vi"]: 
            print(f"lang_code:{lang_code}")
            all_tasks.append((lang_code, csv_file))
        
    return all_tasks

def read_csv_file(csv_path, target_duration=None):
    items =[]
    all_duration = 0
    temp_valid_lines =[]
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip()  # Skip the header row.
        for line in f:
            parts = line.strip().split('|')
            if len(parts) in [3, 4]:  # Valid path|duration|text row, with optional DNSMOS field.
                # Store the raw path, text, and duration first.
                temp_valid_lines.append((parts[0], parts[2], float(parts[1])))
            elif len(parts) == 2:  # Row without a duration value.
                print("Warning: no duration. Check the metadata file")
     
    
    if target_duration is not None:
        random.shuffle(temp_valid_lines)  
    
    for path, text, duration in temp_valid_lines:
        if target_duration is not None and all_duration > target_duration * 3600:
            break
        items.append((path, text, duration))
        all_duration += duration
    
    return items, all_duration/3600

def prepare_all(inp_dir, out_dir_root, tokenizer, dataset_name, sft_gen_dir, num_workers=16, duration_map=None):
    fail_items = {}
    inp_dir = Path(inp_dir)
    out_dir_root = Path(out_dir_root)
    out_dir = out_dir_root / f"{dataset_name}_{tokenizer}_sft"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Will be saved to {out_dir}")
    
    tasks = read_all_metadata(inp_dir) 
    
    raw_arrow_path = out_dir / "raw.arrow"
    writer = ArrowWriter(path=raw_arrow_path.as_posix(), writer_batch_size=10000)
    
    total_samples = 0
    duration_list = []
    prompt_frames_list = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for lang_code, csv_path in tasks:
            fail_items[lang_code] = 0
            print(f"\nProcessing Language: {lang_code} (from {csv_path.name})")
            lang_duration = None
            if duration_map is not None:
                lang_duration = duration_map.get(lang_code, None)
                if lang_duration is None:
                    lang_duration = duration_map.get("default", None)
                print(f"Will choose {lang_duration} hours for {lang_code}")
            raw_items, return_duration = read_csv_file(csv_path, lang_duration)
            if not raw_items:
                continue
            fixed_items = []
            for p, t, d in raw_items:
                fixed_items.append((p, t, d))
                
            print(f"Loaded {len(fixed_items)} lines. Duration: {return_duration:.2f}h. Starting G2P conversion.")
            main_tokenizer = get_tokenizer(lang_code, tokenizer=tokenizer)
            
            batch_size = 1000 
            batches = [fixed_items[i:i + batch_size] for i in range(0, len(fixed_items), batch_size)]

            futures =[executor.submit(process_batch, batch, lang_code, tokenizer, sft_gen_dir) for batch in batches]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"  -> {lang_code}"):
                batch_results, local_fail = future.result()
                fail_items[lang_code] += local_fail
                for res in batch_results:
                    prompt_frames = res.get("prompt_frames")
                    prompt_frames_list.append(prompt_frames)
                    
                    writer.write(res)
                    duration_list.append(res['duration'])
                    total_samples += 1
    writer.finalize()
    
    total_duration = sum(duration_list)
    # Write duration.json.
    with open(out_dir / "duration.json", "w", encoding="utf-8") as f:
        json.dump({
            "duration": duration_list,              # Original audio duration for each sample.
            "prompt_frames": prompt_frames_list,    # Prompt length for each sample.
            "total_hours": total_duration / 3600,   # Sum of original audio duration in hours.
            "total_samples": total_samples          # Total number of samples.
        }, f, ensure_ascii=False)

    print("\n" + "="*50)
    print(f"Total Valid SFT Samples: {total_samples}")
    print(f"Total Orig Audio Hours:  {total_duration/3600:.2f}")
    print(f"Saved to:                {out_dir}")
    print(f"Failed items: {fail_items}")
    print("="*50)

def main():
    parser = argparse.ArgumentParser()
    support_tokenizer = ["ipa_v3", "ipa_v6"]
    parser.add_argument("--inp_dir", type=str, default="/inspire/hdd/project/embodied-multimodality/chenxie-25019/qingyuliu/datasets",help="Root dir containing metadata_*.csv and wavs/")
    parser.add_argument("--out_dir", type=str, default="/inspire/hdd/project/embodied-multimodality/chenxie-25019/qingyuliu/github/XVtest/data",help="Output root dir for raw.arrow")
    parser.add_argument("--sft_gen_dir", type=str, required=True, help="Root dir containing generated .pt and .json files (e.g. multilingual_sft_gen)")
    parser.add_argument("--workers", type=int, default=16, help="Number of CPU workers")
    parser.add_argument("--tokenizer",type=str, choices=support_tokenizer, default="ipa_v3")
    parser.add_argument("--dataset_name",type=str, required=True)
    
    args = parser.parse_args()
    duration_map=None
    
    prepare_all(args.inp_dir, args.out_dir, args.tokenizer, args.dataset_name, args.sft_gen_dir, args.workers, duration_map)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
    
# python src/x_voice/train/datasets/prepare_ipa_sft.py --tokenizer ipa_v6 --dataset_name multilingual_qyl_test --sft_gen_dir ./multilingual_qyl_test_gen
