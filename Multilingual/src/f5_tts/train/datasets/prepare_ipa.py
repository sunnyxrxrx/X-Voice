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

import torchaudio
from tqdm import tqdm
from datasets.arrow_writer import ArrowWriter

from ipa_v3_tokenizer import PhonemizeTextTokenizer as PhonemizeTextTokenizer_v3
from ipa_v5_tokenizer import PhonemizeTextTokenizer as PhonemizeTextTokenizer_v5
from ipa_v6_tokenizer import PhonemizeTextTokenizer as PhonemizeTextTokenizer_v6
import random
from f5_tts.model.utils import str_to_list_ipa_v3, str_to_list_ipa_v5, str_to_list_ipa_v6, get_ipa_id

# import debugpy
# debugpy.listen(('localhost', 5678))
# print("Waiting for debugger attach")
# debugpy.wait_for_client()

sys.path.append(os.getcwd())




# 全局缓存 Tokenizer 
TOKENIZERS = {}
def get_tokenizer(lang_code, tokenizer):
    espeak_code = get_ipa_id(lang_code)
    if espeak_code not in TOKENIZERS:
        try:
            if tokenizer == "ipa_v3":
                TOKENIZERS[espeak_code] = PhonemizeTextTokenizer_v3(language=espeak_code)
            elif tokenizer == "ipa_v5":
                TOKENIZERS[espeak_code] = PhonemizeTextTokenizer_v5(language=espeak_code)
            elif tokenizer == "ipa_v6":
                TOKENIZERS[espeak_code] = PhonemizeTextTokenizer_v6(language=espeak_code, with_stress=True)
        except RuntimeError as e:
            print(f"Error initializing espeak for {espeak_code}: {e}")
            return None
    return TOKENIZERS[espeak_code]



def process_batch(batch_data, lang_code, tokenizer_str):
    """
    batch_data: list[(audio_path, text, duration)]
    """
    tokenizer = get_tokenizer(lang_code, tokenizer=tokenizer_str)
    if tokenizer is None:
        return []

    texts = [item[1] for item in batch_data]
    
    try:
        ipa_texts = [tokenizer([text.strip()]) for text in texts]
        # print(ipa_texts)
    except Exception as e:
        print(f"IPA conversion failed for batch in {lang_code}: {e}")
        return []

    results = []
    for (audio_path, _, duration), ipa_text in zip(batch_data, ipa_texts):
        if not ipa_text.strip():
            continue
        results.append({
            "audio_path": audio_path,
            "text": ipa_text,     
            "duration": duration,
            "language_id": lang_code
        })
    return results

def read_all_metadata(input_dir):
    """
    扫描目录下所有的 metadata_*.csv 文件
    """
    input_path = Path(input_dir)/'csv_train'
    # 匹配 metadata_zh.csv, metadata_en.csv 等
    all_files = list(input_path.glob("metadata_*_full.csv"))
    csv_files = []
    for f in all_files:
        csv_files.append(f)
    
    if not csv_files:
        print(f"No proper csv files found in {input_dir}")
        sys.exit(1)
        
    print(f"Found {len(csv_files)} metadata files: {[f.name for f in csv_files]}")
    all_tasks = [] # (lang_code, file_path)
    
    for csv_file in csv_files:
        # 解析文件名获取语言代码，如 metadata_zh.csv -> zh
        lang_code = csv_file.stem.split('_')[1]
        if True: #lang_code in ["en"]:
            print(f"lang_code:{lang_code}")
            all_tasks.append((lang_code, csv_file))
        
    return all_tasks


def read_csv_file(csv_path, target_duration=None):
    items = []
    all_duration = 0
    # 第一步：先读取所有有效行（带duration的），存储到临时列表
    temp_valid_lines = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip()  # 跳过表头
        for line in f:
            parts = line.strip().split('|')
            if len(parts) == 3:  # path|duration|text 有效行
                # 先暂存原始数据（路径、文本、时长）
                temp_valid_lines.append((parts[0], parts[2], float(parts[1])))
            elif len(parts) == 2:  # 无duration的行
                print("Warning: no duration. Check the metadata file")
    
    # 第二步：如果指定了目标时长，先打乱有效行；否则直接使用原顺序
    if target_duration is not None:
        random.shuffle(temp_valid_lines)  # 随机打乱有效行
    
    # 第三步：遍历（打乱后的）有效行，累加时长直到达到目标
    for path, text, duration in temp_valid_lines:
        # 如果指定了目标时长且已超过，停止遍历
        if target_duration is not None and all_duration > target_duration * 3600:
            break
        items.append((path, text, duration))
        all_duration += duration
    
    return items, all_duration/3600

def prepare_all(inp_dir, out_dir_root, tokenizer, dataset_name, num_workers=16, duration_map=None):
    inp_dir = Path(inp_dir)
    out_dir_root = Path(out_dir_root)
    out_dir = out_dir_root / f"{dataset_name}_{tokenizer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Will be saved to {out_dir}")
    
    tasks = read_all_metadata(inp_dir) # List[(lang, csv_path)]
    
    raw_arrow_path = out_dir / "raw.arrow"
    writer = ArrowWriter(path=raw_arrow_path.as_posix(), writer_batch_size=10000)
    
    vocab_set = set()
    global_token_stats = {}
    total_samples = 0
    duration_list = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for lang_code, csv_path in tasks:
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
                
            print(f"Loaded {len(fixed_items)} lines. Duration: {return_duration}. Starting G2P conversion.")
            main_tokenizer = get_tokenizer(lang_code, tokenizer=tokenizer)
            
            batch_size = 1000 # 每个进程处理 1000 条
            batches = [fixed_items[i:i + batch_size] for i in range(0, len(fixed_items), batch_size)]
            
            futures = [executor.submit(process_batch, batch, lang_code, tokenizer) for batch in batches]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"  -> {lang_code}"):
                batch_results = future.result()
                
                for res in batch_results:
                    writer.write(res)
                    duration_list.append(res['duration'])
                    total_samples += 1
                    
                    # 更新 Vocab 
                    if tokenizer == "ipa_v3":
                        tokens = str_to_list_ipa_v3(res['text']) 
                    elif tokenizer == "ipa_v5":
                        tokens = str_to_list_ipa_v5(res['text'])
                    elif tokenizer == "ipa_v6":
                        tokens = str_to_list_ipa_v6(res['text'])
                    if random.random() <0.000001:
                        tqdm.write(res['text'])
                    
                    for token in tokens:
                        if token not in global_token_stats:
                            global_token_stats[token] = {"count": 0, "langs": set()}
                        global_token_stats[token]["count"] += 1
                        global_token_stats[token]["langs"].add(lang_code)
                    vocab_set.update(tokens)
    writer.finalize()
    
    total_duration = sum(duration_list)
    with open(out_dir / "duration.json", "w", encoding="utf-8") as f:
        json.dump({
            "duration": duration_list, 
            "total_hours": total_duration / 3600,
            "total_samples": total_samples
        }, f, ensure_ascii=False)
        
    # 6. 保存 Vocab
    stats_path = out_dir / "vocab_stats.txt"
    print(f"\nSaving detailed vocabulary statistics to {stats_path}...")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("Token\tTotalCount\tNumLangs\tLanguages\n")
        sorted_stats = sorted(global_token_stats.items(), key=lambda item: item[1]['count'], reverse=True)
        for token, stats in sorted_stats:
            lang_str = ",".join(sorted(list(stats['langs']))) # 排序让输出稳定
            f.write(f"{token}\t{stats['count']}\t{len(stats['langs'])}\t{lang_str}\n")
            
    vocab_path = out_dir / "vocab.txt"
    final_vocab = sorted(list(vocab_set))
    #if "_" not in final_vocab: final_vocab.append("_")
    
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write(" \n<pad>\n")
        for v in final_vocab:
            f.write(f"{v}\n")
            
    print("\n" + "="*50)
    print(f"Total Samples: {total_samples}")
    print(f"Total Hours:   {total_duration/3600:.2f}")
    print(f"Vocab Size:    {len(final_vocab)}")
    print(f"Saved to:      {out_dir}")
    print("="*50)

def main():
    parser = argparse.ArgumentParser()
    support_tokenizer = ["ipa_v3","ipa_v5", "ipa_v6"]
    parser.add_argument("--inp_dir", type=str, default="/inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/rixixu/datasets",help="Root dir containing metadata_*.csv and wavs/")
    parser.add_argument("--out_dir", type=str, default="/inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/rixixu/F5-TTS/data",help="Output root dir for raw.arrow")
    parser.add_argument("--workers", type=int, default=16, help="Number of CPU workers")
    parser.add_argument("--tokenizer",type=str, choices=support_tokenizer, default="ipa_v3")
    parser.add_argument("--dataset_name",type=str, required=True)
    
    
    args = parser.parse_args()
    duration_map=None
    
    prepare_all(args.inp_dir, args.out_dir, args.tokenizer, args.dataset_name, args.workers, duration_map)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
    
# python src/f5_tts/train/datasets/prepare_ipa.py --tokenizer ipa_v6 --dataset_name multilingual_stress_ko