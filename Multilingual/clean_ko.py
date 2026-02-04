import os
import csv
import torch
import torchaudio
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor



ROOT_DIR = "/inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/rixixu/datasets/wavs"
METADATA_PATH = "/inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/rixixu/datasets/metadata_ko_with_duration.csv"
OUTPUT_CSV = "/inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/rixixu/datasets/metadata_ko_with_duration_clean.csv"
BAD_FILES_LOG = "bad_files_ko.log"

# 清洗阈值
MIN_DURATION = 2.0
MAX_DURATION = 30.0
ENERGY_THRESHOLD = 1e-4  # max_abs 峰值阈值
RMS_THRESHOLD = 1e-5     # rms 能量平均阈值
# ===========================================

def check_audio_file(item):
    """
    检查单条音频的合法性
    item: dict，包含 file_path, duration, text
    返回: (is_valid, item_data, error_msg)
    """
    rel_path = item["file_path"]
    
    # 路径拼接逻辑
    if os.path.isabs(rel_path):
        audio_path = rel_path
    else:
        audio_path = os.path.join(ROOT_DIR, rel_path)
    
    try:
        # 1. 快速检查元数据 
        if not os.path.exists(audio_path):
            return False, None, f"File not found: {audio_path}"
            
        info = torchaudio.info(audio_path)
        duration = info.num_frames / info.sample_rate
        #print(duration)
        if not (MIN_DURATION <= duration <= MAX_DURATION):
            return False, None, f"Duration mismatch: {duration:.2f}s (file_path: {rel_path})"

        # 2. 深度检查内容 
        audio, sr = torchaudio.load(audio_path)
        
        # 检查 NaN/Inf
        if torch.any(torch.isnan(audio)) or torch.any(torch.isinf(audio)):
            return False, None, f"Contains NaN/Inf: {rel_path}"
        
        # 检查静音/低能量
        max_val = audio.abs().max().item()
        rms_val = torch.sqrt(torch.mean(audio**2)).item()
        print(max_val)
        
        if max_val < ENERGY_THRESHOLD or rms_val < RMS_THRESHOLD:
            return False, None, f"Silence: max={max_val:.6f}, rms={rms_val:.6f} ({rel_path})"
        
        # 3. 检查时长一致性 (CSV 标注 vs 实际文件)
        try:
            meta_duration = float(item["duration"])
            if abs(duration - meta_duration) > 0.5:
                return False, None, f"Inconsistent: meta={meta_duration}, actual={duration:.2f}"
        except ValueError:
            pass # 如果标注时长不是数字，跳过此项检查

        return True, item, None

    except Exception as e:
        return False, None, f"Error processing {rel_path}: {str(e)}"

def main():
    # 1. 读取原始 CSV
    # 使用 DictReader 自动处理表头，delimiter='|'
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        # 自动识别列名。如果你的 CSV 没有表头，请手动指定 fieldnames=['file_path', 'duration', 'text']
        reader = csv.DictReader(f, delimiter='|')
        data = list(reader)
        fieldnames = reader.fieldnames

    print(f"Total entries to check: {len(data)}")
    
    bad_count = 0
    valid_count = 0

    # 2. 准备写入清洗后的结果
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as out_f, \
         open(BAD_FILES_LOG, 'w', encoding='utf-8') as log_f:
        
        writer = csv.DictWriter(out_f, fieldnames=fieldnames, delimiter='|')
        writer.writeheader()
        
        # 3. 多进程处理
        # 建议开启 CPU 核心数的 80%-90%
        # 使用 imap 以节省内存，适合处理千万级别的数据行
        with ProcessPoolExecutor(max_workers=64) as executor:
            # 这里的 chunksize 很重要，对于大量小任务，设置 chunksize=100 会快很多
            for is_valid, item, err in tqdm(executor.map(check_audio_file, data, chunksize=500), total=len(data), desc="Cleaning CSV Data"):
                if is_valid:
                    writer.writerow(item)
                    valid_count += 1
                else:
                    bad_count += 1
                    print(err)
                    log_f.write(f"{err}\n")

    print(f"\nProcessing Complete!")
    print(f"Total: {len(data)}")
    print(f"Valid (Saved to CSV): {valid_count}")
    print(f"Removed (Logged to file): {bad_count}")
    print(f"Cleaned metadata saved to: {OUTPUT_CSV}")
    print(f"Bad files details in: {BAD_FILES_LOG}")

if __name__ == "__main__":
    # 建议在使用 torch 的多进程时，如果是 Linux 系统，采用 spawn 模式更安全
    # 但对于 F5-TTS 的简单读取任务，默认 fork 也可以
    main()