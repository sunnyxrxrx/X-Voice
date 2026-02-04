import os
import shutil
import random
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import torchaudio
import re
from num2words import num2words


SAMPLES_PER_LANG = 500
MIN_DURATION = 1.0
MAX_DURATION = 16.0
MIN_RMS = 0.05  # 音量阈值 


def check_audio_valid(wav_path):
    """检查音频是否有效：时长和音量"""
    try:
        # 检查时长
        info = sf.info(wav_path)
        if not (MIN_DURATION <= info.duration <= MAX_DURATION):
            return False, f"Duration {info.duration:.2f}s out of range"
    
        data, sr = sf.read(wav_path)
        if len(data) == 0:
            return False, "Empty audio"
            
        # 计算 RMS
        rms = np.sqrt(np.mean(data**2))
        if rms < MIN_RMS:
            return False, f"Low volume (RMS={rms:.4f})"
            
        return True, "OK"
        
    except Exception as e:
        return False, f"Read Error: {e}"

def create_testset_from_local(dataset):
    if dataset == "lemas":
        TESTSET_RAW_ROOT = "/inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/rixixu/lemas-eval/eval"
        OUTPUT_ROOT = "./data/lemas_eval"
        TARGET_LANGS = {
        "es":"es",
        "de": "de",
        "en":"en",
        "it":"it",
        "de":"de",
        "pt":"pt",
        "vi":"vi",
        "fr":"fr",
        "zh": "cmn_hans_cn",
        }
    elif dataset == "cv3": 
        TESTSET_RAW_ROOT = "/inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/rixixu/CV3-testset"
        OUTPUT_ROOT = "./data/cv3_eval"
        TARGET_LANGS = {
        #"en": "en_us",
        "it":"it_it",
        #"zh": "cmn_hans_cn",
        }
    output_path = Path(OUTPUT_ROOT)
    tests_raw_path = Path(TESTSET_RAW_ROOT)
    print(f"Loading data from: {tests_raw_path.resolve()}")

    for lang_short, lang_folder in TARGET_LANGS.items():
        print(f"\n{'='*60}")
        print(f"Processing {lang_short} ({lang_folder})")
        
        jsonl_path = tests_raw_path / "metadata.jsonl" 
        audio_dir = tests_raw_path
        
        if not jsonl_path.exists():
            print(f"[SKIP] {lang_short}: {jsonl_path} not found.")
            continue

        try:
            # 读取JSONL并转为List[Dict]
            import json
            data = []
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            
            # 构造 DataFrame
            df = pd.DataFrame(data)
            if dataset == "lemas":
                # 根据 key 字段的前缀进行过滤
                df = df[df['key'].str.startswith(f"{lang_short}_")].copy()
                if len(df) == 0:
                    print(f"[SKIP] {lang_short}: No entries found with prefix {lang_short}_ in mixed metadata.")
                    continue
            df['full_path'] = df['file_name'].apply(lambda x: tests_raw_path / x)
            # 过滤不存在的文件
            df = df[df['full_path'].apply(lambda x: x.exists())]

            lang_out_dir = output_path / "zero_shot" / lang_short
            wav_out_dir = lang_out_dir / "waveform"
            wav_out_dir.mkdir(parents=True, exist_ok=True)
            
            f_p_txt = open(lang_out_dir / "prompt_text", 'w', encoding='utf-8')
            f_p_wav = open(lang_out_dir / "prompt_wav.scp", 'w', encoding='utf-8')
            f_text = open(lang_out_dir / "text", 'w', encoding='utf-8')
            
            # 打乱整个 DataFrame
            df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            valid_count = 0
            pbar = tqdm(total=SAMPLES_PER_LANG, desc=f"Selecting {lang_short}")
            
            # 取两个不一样的样本，一个当 Prompt，一个提供 Text
            # 这里的i是Prompt的索引，(i+1)是Target的索引
            
            total_candidates = len(df_shuffled)
            idx = 0
            
            while valid_count < SAMPLES_PER_LANG and idx < total_candidates - 1:
                # 候选 Prompt
                prompt_row = df_shuffled.iloc[idx]
                # 候选 Target Text (取下一个样本)
                target_row = df_shuffled.iloc[(idx + 1) % total_candidates]
                
                wav_path = prompt_row['full_path']
                
                # 检查音频质量
                is_valid, msg = check_audio_valid(wav_path)
                
                if is_valid:
                    utt_id = f"uttid_{valid_count}"
                    if dataset == "cv3":
                        raw_prompt_text = prompt_row['raw_text']
                        raw_target_text = target_row['raw_text']
                    elif dataset == "lemas":
                        raw_prompt_text = prompt_row['txt']
                        raw_target_text = target_row['txt']
                    else:
                        raw_prompt_text = ""
                        raw_target_text = ""

                    clean_p_text = raw_prompt_text
                    clean_t_text = raw_target_text
                    # 复制音频
                    if dataset == "cv3":
                        dest_name = f"prompt_{valid_count}.wav"
                        
                        f_p_txt.write(f"{utt_id} {clean_p_text}\n")
                    
                        rel_path = Path("zero_shot") / lang_short / "waveform" / dest_name
                        f_p_wav.write(f"{utt_id} {rel_path.as_posix()}\n")
                        
                        f_text.write(f"{utt_id} {clean_t_text}\n")
                        
                        shutil.copy(wav_path, wav_out_dir / dest_name)
                        
                    elif dataset == "lemas":
                        dest_name = f"prompt_{valid_count}.wav"
                        f_p_txt.write(f"{utt_id} {clean_p_text}\n")
                    
                        rel_path = Path("zero_shot") / lang_short / "waveform" / dest_name
                        f_p_wav.write(f"{utt_id} {rel_path.as_posix()}\n")
                        f_text.write(f"{utt_id} {clean_t_text}\n")
                        
                        audio, sr = torchaudio.load(wav_path)
                        torchaudio.save(wav_out_dir / f"prompt_{valid_count}.wav", audio, sr)
                    
                    valid_count += 1
                    pbar.update(1)
                else:
                    print(msg)
                # 无论是否合格，都看下一个
                idx += 1
                
            pbar.close()
            
            # 关闭文件
            f_p_txt.close()
            f_p_wav.close()
            f_text.close()
            
            print(f"  -> Finished. Valid samples: {valid_count}/{SAMPLES_PER_LANG}")
            if valid_count < SAMPLES_PER_LANG:
                print(f"  [WARN] Not enough valid samples! (Scanned {idx} files)")

        except Exception as e:
            print(f"  [ERROR] {lang_short}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nAll done! Output at {OUTPUT_ROOT}")

if __name__ == "__main__":
    create_testset_from_local("lemas")