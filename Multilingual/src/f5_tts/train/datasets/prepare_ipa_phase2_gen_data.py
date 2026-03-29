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
# debugpy.listen(('localhost', 568))
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
    batch_data: list[(audio_path, ref_text, gen_text, duration)]
    """
    tokenizer = get_tokenizer(lang_code, tokenizer=tokenizer_str)
    if tokenizer is None:
        return []

    # 分别提取参考文本和生成文本
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
        # 防止除零错误
        if ref_text_len == 0: continue
        # 计算 total_mel_len
        ref_mel_len = int(duration * target_sample_rate / hop_length)
        total_mel_len = ref_mel_len + int(ref_mel_len / ref_text_len * gen_text_len)
        # 提取不带后缀的相对路径
        rel_path = audio_path
        if random.random() < 0.0001:
            print(f"ref: {ref_text}\ngen: {gen_text}\nref_ipa: {ref_ipa}\ngen_ipa: {gen_ipa}\nduration: {duration}\nmel len: {total_mel_len}\npath: {rel_path}\n")
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
    扫描目录下所有的 metadata_*.csv 文件
    """
    input_path = Path(input_dir)/'csvs'
    # 匹配 metadata_zh.csv, metadata_en.csv 等
    all_files = list(input_path.glob("metadata_*_top_1000.0h.csv"))
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
        if lang_code in ["en","id","vi"]:
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
            if len(parts) in [3, 4]:  # path|duration|text 有效行，可选DNSMOS行
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
    out_dir = out_dir_root / f"{dataset_name}_{tokenizer}_gp"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Will be saved to {out_dir}")
    
    tasks = read_all_metadata(inp_dir) # List[(lang, csv_path)]
    
    raw_arrow_path = out_dir / "raw.arrow"
    writer = ArrowWriter(path=raw_arrow_path.as_posix(), writer_batch_size=10000)
    
    vocab_set = set()
    global_token_stats = {}
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
            
            # 1. 建立排序后的文本池 (格式: (文本, 字节数))
            # 排序是为了后续使用二分查找定位区间
            text_pool = sorted(
                [(t, len(t.encode("utf-8"))) for _, t, _ in raw_items],
                key=lambda x: x[1]
            )
            pool_lengths = [x[1] for x in text_pool]  # 提取纯长度列表用于二分
            
            fixed_items = []
            chunck_items = 0
            for p, t, d in raw_items:
                ref_len = len(t.encode("utf-8"))
                if ref_len == 0: continue
                
                # 目标区间：[min_b, max_b]
                min_b, max_b = ref_len * 0.1, ref_len * 0.4
                
                # 2. 使用二分查找快速定位符合条件的索引区间
                start_idx = bisect.bisect_left(pool_lengths, min_b)
                end_idx = bisect.bisect_right(pool_lengths, max_b)
                
                gen_t = None
                if start_idx < end_idx:
                    # 如果区间存在，随机选一个，保证数据多样性
                    gen_t = text_pool[random.randint(start_idx, end_idx - 1)][0]
                else:
                    # 3. 兜底逻辑：如果池子里确实没有这么短的句子，查找最接近 max_b 的句子进行截断
                    # 找到第一个长度大于 max_b 的句子
                    chunck_items += 1
                    trunc_idx = bisect.bisect_left(pool_lengths, max_b)
                    if trunc_idx < len(text_pool):
                        rand_idx = random.randint(trunc_idx, len(text_pool) - 1)
                        long_cand = text_pool[rand_idx][0]
                        # 执行智能截断 (欧美按词，中日泰按字)
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
                        if random.random() < 0.1:
                            print(f"{d}\n")
                            print(gen_t)
                
                # 只有成功获得 gen_text 且长度合理才加入
                if gen_t and len(gen_t.encode("utf-8")) >= 1: # 至少1个字节
                    fixed_items.append((p, t, gen_t, d))
          
            print(f"Loaded {len(fixed_items)} lines. Failed item {chunck_items}. Duration: {return_duration}. Starting G2P conversion.")
            
            batch_size = 1000 # 每个进程处理 1000 条
            batches = [fixed_items[i:i + batch_size] for i in range(0, len(fixed_items), batch_size)]
            
            futures = [executor.submit(process_batch, batch, lang_code, tokenizer) for batch in batches]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"  -> {lang_code}"):
                batch_results = future.result()
                
                for res in batch_results:
                    writer.write(res)
                    mel_duration_list.append(res['total_mel_len'])
                    duration_list.append(res["duration"])
                    total_samples += 1
                    
                    # 更新 Vocab 
                    if tokenizer == "ipa_v3":
                        tokens = str_to_list_ipa_v3(res['gen_text_ipa']) 
                    elif tokenizer == "ipa_v5":
                        tokens = str_to_list_ipa_v5(res['gen_text_ipa'])
                    elif tokenizer == "ipa_v6":
                        tokens = str_to_list_ipa_v6(res['gen_text_ipa'])
                    if random.random() <0.000001:
                        tqdm.write(res['gen_text_ipa'])
                    
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
            "duration": mel_duration_list, 
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
    parser.add_argument("--inp_dir", type=str, default="/inspire/hdd/project/embodied-multimodality/chenxie-25019/rixixu/datasets",help="Root dir containing metadata_*.csv and wavs/")
    parser.add_argument("--out_dir", type=str, default="/inspire/hdd/project/embodied-multimodality/chenxie-25019/rixixu/Multilingual_F5-TTS/F5-TTS/data",help="Output root dir for raw.arrow")
    parser.add_argument("--workers", type=int, default=16, help="Number of CPU workers")
    parser.add_argument("--tokenizer",type=str, choices=support_tokenizer, default="ipa_v6")
    parser.add_argument("--dataset_name",type=str, required=True)
    
    
    args = parser.parse_args()
    duration_map=None
    
    prepare_all(args.inp_dir, args.out_dir, args.tokenizer, args.dataset_name, args.workers, duration_map)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
    
# python src/f5_tts/train/datasets/prepare_ipa_phase2_gen_data.py --tokenizer ipa_v6 --dataset_name multilingual_sft_en_id_vi