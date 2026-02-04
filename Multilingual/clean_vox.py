import os
import torch
import torchaudio
import pandas as pd
import re
import subprocess
from pathlib import Path
from tqdm import tqdm
import torch.multiprocessing as mp
from torchaudio.pipelines import MMS_FA as bundle
import numpy as np

import debugpy
debugpy.listen(('localhost', 5678))
print("Waiting for debugger attach")
debugpy.wait_for_client()

RAW_AUDIO_ROOT = Path("/inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/rixixu/datasets/wavs")
INPUT_METADATA = "/inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/rixixu/datasets/metadata_es_with_duration.csv" 
OUTPUT_BASE = Path("/inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/rixixu/datasets/")
NUM_GPUS = torch.cuda.device_count()
MAX_DUR = 10.0
UROMAN_PATH = "/inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/rixixu/uroman/bin/uroman.pl"
    

def uromanize(text):
    """
    调用 uroman.pl 将原始文本转义为拉丁字母
    """
    try:
        # 使用 subprocess 调用 perl 脚本
        process = subprocess.Popen(
            ['perl', UROMAN_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=text)
        return stdout.strip()
    except Exception as e:
        # print(f"Uroman error: {e}")
        return text

def clean_text_for_mms(text):
    # 1. 使用 uroman 转义 
    text = uromanize(text)
    
    # 2. 强制转小写
    text = text.lower()
    
    # 3. 只保留 a-z 和空格
    # MMS Tokenizer 的词表非常窄，任何标点、特殊符号、数字都会导致 KeyError
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 4. 压缩空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def worker(rank, lang_df, lang_short, output_wav_dir):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    model = bundle.get_model().to(device)
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    
    results = []
    
    for _, row in tqdm(lang_df.iterrows(), total=len(lang_df), desc=f"GPU {rank}"):
        wav_path = RAW_AUDIO_ROOT / row['file_path']
        if not wav_path.exists(): continue
        
        try:
            waveform, sr = torchaudio.load(wav_path)
            waveform_16k = torchaudio.functional.resample(waveform, sr, 16000).to(device)
            
            raw_text = row['text']
            # 使用 uroman 清洗后的文本
            cleaned_text = clean_text_for_mms(raw_text)
            words = cleaned_text.split()
            if not words: continue

            with torch.inference_mode():
                emission, _ = model(waveform_16k)
                
                # 尝试进行 Tokenize
                try:
                    tokens = tokenizer(words)
                except KeyError:
                    print(f"Key: {words}")
                    # 如果还是有 OOV 字符，逐个单词过滤掉坏字符
                    safe_words = []
                    for w in words:
                        try:
                            _ = tokenizer([w])
                            safe_words.append(w)
                        except:
                            continue
                    words = safe_words
                    if not words: continue
                    tokens = tokenizer(words)

                # 执行对齐
                point_segments, _ = aligner(emission[0], tokens)

            num_frames = emission.size(1)
            audio_dur = waveform.size(1) / sr
            def f2s(f): return f * audio_dur / num_frames

            # 切分逻辑
            curr_start_idx = 0
            raw_words = str(raw_text).split() # 原始文本用于保存，保留标点
            
            for i in range(len(point_segments)):
                t_start = f2s(point_segments[curr_start_idx][0].start)
                t_end = f2s(point_segments[i][-1].end)
                
                if (t_end - t_start) >= MAX_DUR or i == len(point_segments) - 1:
                    s_sample = int(t_start * sr)
                    e_sample = int(t_end * sr)
                    sub_wav = waveform[:, s_sample:e_sample]
                    
                    if sub_wav.shape[1] > 1600:
                        seg_name = f"{wav_path.stem}_r{rank}_s{len(results)}.wav"
                        save_path = output_wav_dir / seg_name
                        torchaudio.save(save_path, sub_wav, sr)
                        
                        # 截取对应的原始文本
                        actual_sub_text = " ".join(raw_words[curr_start_idx : i+1])
                        results.append([save_path.as_posix(), t_end - t_start, actual_sub_text])
                    curr_start_idx = i + 1
        except Exception as e:
            print(e)
            continue
            
    temp_df = pd.DataFrame(results, columns=['file_path', 'duration', 'text'])
    temp_df.to_csv(OUTPUT_BASE / f"temp_res_{lang_short}_{rank}.csv", index=False, sep="|")

def process_multilingual_parallel(lang_short, input_csv):
    """
    主控程序：分发任务给多个 GPU
    """
    output_wav_dir = OUTPUT_BASE / "seg_wavs" / lang_short
    output_wav_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(input_csv, sep="|")
    
    # 将任务平分成 NUM_GPUS 份
    df_chunks = np.array_split(df, NUM_GPUS)
    
    print(f"🔥 Starting Parallel Processing on {NUM_GPUS} GPUs...")
    processes = []
    for rank in range(NUM_GPUS):
        p = mp.Process(target=worker, args=(rank, df_chunks[rank], lang_short, output_wav_dir))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
        
    # 合并结果
    all_res = []
    for rank in range(NUM_GPUS):
        tmp_path = OUTPUT_BASE / f"temp_res_{lang_short}_{rank}.csv"
        if tmp_path.exists():
            all_res.append(pd.read_csv(tmp_path, sep="|"))
            tmp_path.unlink() # 删除临时文件
            
    final_df = pd.concat(all_res)
    final_df.to_csv(OUTPUT_BASE / f"metadata_{lang_short}_segmented.csv", index=False, sep="|")
    print(f"✅ Done! Total segments: {len(final_df)}")

if __name__ == "__main__":
    # 必须使用 spawn 模式来启动 CUDA 进程
    mp.set_start_method('spawn', force=True)
    
    # 示例：处理西语
    process_multilingual_parallel("es", INPUT_METADATA)