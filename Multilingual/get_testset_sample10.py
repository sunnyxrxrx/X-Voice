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
REF_MIN_DURATION = 3.0
REF_MAX_DURATION = 8.0
REF_RMS = 0.

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
    
def check_audio_valid_ref(wav_path):
    """检查音频是否有效：时长和音量"""
    try:
        # 检查时长
        info = sf.info(wav_path)
        if not (REF_MIN_DURATION <= info.duration <= REF_MAX_DURATION):
            return False, f"Duration {info.duration:.2f}s out of range"
    
        data, sr = sf.read(wav_path)
        if len(data) == 0:
            return False, "Empty audio"
            
        # 计算 RMS
        rms = np.sqrt(np.mean(data**2))
        if rms < REF_RMS:
            return False, f"Low volume (RMS={rms:.4f})"       
        return True, "OK"  
     
    except Exception as e:
        return False, f"Read Error: {e}"

def create_testset_from_local(dataset):
    if dataset == "lemas":
        TESTSET_RAW_ROOT = "/inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/rixixu/lemas-eval/eval"
        OUTPUT_ROOT = "./data/lemas_eval_new"
        TARGET_LANGS = {
        "en":"en",
        "it":"it",
        "de":"de",
        "pt":"pt",
        "vi":"vi",
        "fr":"fr",
        "zh": "cmn_hans_cn",
        "es":"es",
        "id":"id"
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
    testset_raw_path = Path(TESTSET_RAW_ROOT)
    print(f"Loading data from: {testset_raw_path.resolve()}")

    for lang_short, lang_folder in TARGET_LANGS.items():
        print(f"\n{'='*60}")
        print(f"Processing {lang_short} ({lang_folder})")
        
        jsonl_path = testset_raw_path / "metadata.jsonl" 
        audio_dir = testset_raw_path
        
        if not jsonl_path.exists():
            print(f"[SKIP] {lang_short}: {jsonl_path} not found.")
            continue

        try:
            # 读取 JSONL并转为List[Dict]
            import json
            data = []
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            
            # 构造 DataFrame
            df = pd.DataFrame(data)
            if dataset == "lemas":
                # 根据key字段的前缀进行过滤
                df = df[df['key'].str.startswith(f"{lang_short}_")].copy()
                if len(df) == 0:
                    print(f"[SKIP] {lang_short}: No entries found with prefix {lang_short}_ in mixed metadata.")
                    continue
            df['full_path'] = df['file_name'].apply(lambda x: testset_raw_path / x)
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
            total_candidates = len(df_shuffled)
            # 先选出 10 条固定的参考音频
            NUM_PROMPTS = 10
            valid_prompts = []
            scan_idx = 0
            
            print(f"Finding {NUM_PROMPTS} reference audios for {lang_short}...")
            while len(valid_prompts) < NUM_PROMPTS and scan_idx < total_candidates:
                row = df_shuffled.iloc[scan_idx]
                is_valid, _ = check_audio_valid_ref(row['full_path'])
                if is_valid and re.search(r'\d', row['txt']) is None:
                    valid_prompts.append(row)
                scan_idx += 1
            
            if len(valid_prompts) < NUM_PROMPTS:
                print(f"[error]: Not enough suitable audios for references")
                continue
            
            # 循环500次生成测试对
            valid_count = 0
            pbar = tqdm(total=SAMPLES_PER_LANG, desc=f"Generating pairs for {lang_short}")
            
            # 为了保证文本不重复，从scan_idx之后开始取文本样本
            text_idx = scan_idx 
            
            while valid_count < SAMPLES_PER_LANG and text_idx < total_candidates:
                # 从 10 条固定音频中轮流取一条
                prompt_row = valid_prompts[valid_count % NUM_PROMPTS]
                # 从剩余的数据集中取一条作为Target Text
                target_row = df_shuffled.iloc[text_idx]
                
                utt_id = f"uttid_{valid_count}"
                
                # 获取文本逻辑
                if dataset == "cv3":
                    raw_prompt_text = prompt_row['raw_text']
                    raw_target_text = target_row['raw_text']
                else:
                    raw_prompt_text = prompt_row['txt']
                    raw_target_text = target_row['txt']

                clean_p_text = raw_prompt_text
                clean_t_text = raw_target_text

                # 写入文件和复制音频
                dest_name = f"prompt_{valid_count}.wav"
                
                # 写入 prompt_text 和 prompt_wav
                f_p_txt.write(f"{utt_id} {clean_p_text}\n")
                rel_path = Path("zero_shot") / lang_short / "waveform" / dest_name
                f_p_wav.write(f"{utt_id} {rel_path.as_posix()}\n")
                
                # 写入目标文本
                f_text.write(f"{utt_id} {clean_t_text}\n")
                
                # 保存音频
                audio, sr = torchaudio.load(prompt_row['full_path'])
                torchaudio.save(wav_out_dir / dest_name, audio, sr)
                
                valid_count += 1
                text_idx += 1
                pbar.update(1)
            
            print(f"Finished. Valid samples: {valid_count}/{SAMPLES_PER_LANG}")
            
        except Exception as e:
            print(f"[error]: {lang_short}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nAll done! Output at {OUTPUT_ROOT}")

if __name__ == "__main__":
    create_testset_from_local("lemas")