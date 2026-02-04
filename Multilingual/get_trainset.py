import os
import sys
import random
import shutil
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import soundfile as sf
from tqdm import tqdm
import re 
import argparse
import json
import pyarrow.parquet as pq
#from langdetect import detect,detect_langs,LangDetectException
from functools import partial
from tqdm.contrib.concurrent import process_map 

# import debugpy
# debugpy.listen(('localhost', 12345))
# print("Waiting for debugger attach")
# debugpy.wait_for_client()

min_duration = 0.4 # 过滤不合适的音频长度
max_duration = 30.0


# 对于mls、voxpopuli、gigaspeech，先用soundfile获取时长，返回时长、文本、音频绝对路径的result
def get_duration_worker(utt_id, text, dataset, dataset_root):
    try:
        if dataset=="mls":
            speaker_id, chapter_id, _ = utt_id.split("_")
            audio_path = dataset_root / "audio" / speaker_id / chapter_id / f"{utt_id}.flac"
        elif dataset=="voxpopuli":
            speaker_id = utt_id[:4]
            audio_path = dataset_root / speaker_id / f"{utt_id}.ogg"
        else: #dataset=="gigaspeech2":
            speaker_id, chapter_id, _ = utt_id.split("-")
            audio_path = dataset_root / "train" / speaker_id / chapter_id / f"{utt_id}.wav"
            if not audio_path.exists(): # 越南语走此分支
                audio_path = dataset_root / "train" / "release1" / "train" / speaker_id / chapter_id / f"{utt_id}.wav"
        if audio_path.exists():
            duration = sf.info(str(audio_path)).duration
            
            if random.random()<0.0001:
                print(duration)
            return {
                "utt_id": utt_id, "speaker_id": speaker_id,
                "audio_path": audio_path, "text": text,
                "duration": duration,
            }
        else:
            print(f"path {audio_path} does not exists")
            return None
    except Exception:
        return None
    
# 专门开一个给koreaspeech
# 处理单个parquet文件，记录metadata同时，生成wav文件，放在zhikang的目录下，后面再软连接过去
def process_single_parquet(p_file):
    local_metadata = []
    try:
        # 从文件名提取speaker id
        filename = p_file.name
        match = re.search(r'train-(\d+)-of', filename) 
        speaker_id = match.group(1) if match else "unknown"
        
        # 在parquet同级目录下创建speaker_id文件夹
        source_dir = p_file.parent
        speaker_dir = source_dir / speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True) 
        table = pq.read_table(p_file, columns=['id', 'sentence', 'meta', 'audio'])
        rows = table.to_pylist() 
        for row in rows:
            utt_id = row['id']
            wav_filename = f"{utt_id}.wav"
            source_wav_path = speaker_dir / wav_filename
            # 写入wav
            if not source_wav_path.exists():
                audio_bytes = row['audio']['bytes']
                with open(source_wav_path, 'wb') as f:
                    f.write(audio_bytes)
            duration = float(row['meta'].get('length', 0.0))
            local_metadata.append({
                "utt_id": utt_id,
                "speaker_id": speaker_id,
                "audio_path": source_wav_path, 
                "text": row['sentence'],
                "duration": duration,
            })
            if random.random()<0.00001:
                print(source_wav_path,row['sentence'])
            
    except Exception as e:
        print(f"Error processing {p_file}: {e}")
    
    return local_metadata

# TODO. parquet解压+写wav经常会处理到一半中断，应该是内存的问题
def get_koreaspeech_metadata(dataset_root, in_language, num_workers=1):
    if num_workers!=1:
        print("Warning: num_workers=1 is recommended in get_koreaspeech_metaadata, otherwise it may cause OOM")
    metadata = []
    parquet_files = list(dataset_root.rglob("*00086.parquet"))
    print(f"Found {len(parquet_files)} parquet files. Extracting WAVs and metadata...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_parquet, p_file): p_file for p_file in parquet_files}
        for future in tqdm(as_completed(futures), total=len(parquet_files), desc="Extracting"):
            try:
                result = future.result()
                metadata.extend(result)
            except Exception as e:
                print(f"Worker failed: {e}")

    return metadata

def collect_metadata(dataset, dataset_root, language, in_language, workers, output_dir,transcripts_name,transcripts_root=None,force_rescan=False,interset=None): 
    # 缓存加载
    if dataset == "voxpopuli" and in_language in interset:
        cache_file = output_dir /f"metadata_{in_language}_vox_cache.pkl"
    elif dataset == "mls":
        cache_file = output_dir /f"metadata_{in_language}_mls_cache.pkl"
    elif dataset == "lemas":
        cache_file = output_dir /f"metadata_{in_language}_lemas_cache.pkl"
    else:
        cache_file = output_dir /f"metadata_{in_language}_cache.pkl"
    # if not cache_file.exists():# 前向版本是存全称的
    #     cache_file = output_dir /f"metadata_{language}_cache.pkl"
    
    if not force_rescan and cache_file.exists():
        print(f"Loading pre-scanned metadata from '{cache_file}'...")
        with open(cache_file, 'rb') as f:
            metadata = pickle.load(f)
        total_hours = sum([m['duration'] for m in metadata]) / 3600
        print("\n--- Diagnostic: Checking Top 10 Longest Files ---")
        # 按时长降序排列
        sorted_meta = sorted(metadata, key=lambda x: x['duration'], reverse=True)
        for i, m in enumerate(sorted_meta[:10]):
            print(f"Rank {i+1}: Duration={m['duration']:.2f}s ({m['duration']/3600:.2f}h) | TextLen={len(m['text'])} | ID={m['utt_id']}")
            print(f"   -> File: {m['audio_path']}")
        
        print(f"Found {len(metadata)} valid audio-text pairs. Total duration: {total_hours:.2f} hours.")
        return metadata
    
    
    transcripts = {}
    metadata = []
    
    # koreaspeech特别处理，直接获得metadata
    if dataset=="koreaspeech":
        print(f"Processing koreaspeech parquet files...")
        metadata = get_koreaspeech_metadata(dataset_root, in_language, num_workers=workers)
    elif dataset=="lemas":
        print(f"Searching for LEMAS jsonl files in {dataset_root}...")
        # 找到该语言目录下所有的jsonl文件
        jsonl_files = list(dataset_root.glob("*.jsonl"))
        for jsonl_path in tqdm(jsonl_files, desc=f"Parsing LEMAS_{in_language} jsonls"):
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        rel_audio_path = record.get('audio')
                        audio_path = dataset_root / rel_audio_path
                        if True: #audio_path.exists():
                            if random.random() <0.000001:
                                print(f"\n[example]: audio: {rel_audio_path}, text: {record['txt']}")
                            speaker_id = Path(rel_audio_path).parent.name
                            metadata.append({
                                "utt_id": Path(rel_audio_path).stem,
                                "speaker_id": speaker_id,
                                "audio_path": audio_path,
                                "text": record['txt'],
                                "duration": float(record['dur']),
                            })
                        
                    except (json.JSONDecodeError, KeyError, Exception) as e:
                        print(e)
                        continue
    else: 
        # 第一步：获取transcripts的路径
        if dataset=="voxpopuli":
            transcripts_path = Path(transcripts_root) / Path(transcripts_name)
        else:
            transcripts_path = dataset_root / Path(transcripts_name)
        if not transcripts_path.exists():
            print(f"Error: TSV file not found at {transcripts_path}", file=sys.stderr)
            return []
        
        # 第二部：把text记录到record，获取duration   
        # 如果是emiilia，已经有duration了
        if dataset=="emilia":
            with open(transcripts_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"Parsing emilia_{in_language}.jsonl"):
                    try:
                        record = json.loads(line)
                        if not all(k in record for k in ['wav_path', 'text', 'duration', 'language']):
                            continue
                        # 过滤出中文/英文和训练集数据
                        if record['language'] != in_language or record.get('split') != 'train':
                            continue
                        audio_path = dataset_root / "audios" / record['wav_path']
                        if audio_path.exists():
                            metadata.append({
                                'audio_path': audio_path,
                                'text': record['text'],
                                'duration': float(record['duration']),
                            })
                        print(audio_path)
                    except (json.JSONDecodeError, KeyError):
                        continue # 跳过格式错误或缺少关键字段的行
        # 对于其他数据集，先读取一遍数据获得时长
        else:
            print(f"Parsing transcripts from {transcripts_path}...")
            # 如果是voxpopuli，存的是jsonl的形式
            if dataset == "voxpopuli":
                with open(transcripts_path, 'r', encoding='utf-8') as f:
                    for line in tqdm(f, desc=f"Parsing voxpopuli_{in_language}.jsonl"):
                        try:
                            record = json.loads(line)
                            if not all(k in record for k in ['audio_filepath', 'text', 'utt_id']):
                                print("record error, skip it")
                                continue
                            utt_id = record['utt_id']
                            text = record['text']
                            transcripts[utt_id] = text
                            if random.random() <0.000001:
                                print(f"\n[example]: audio: {utt_id}, text: {text}")
                        except (json.JSONDecodeError, KeyError):
                            continue 
            else:
                with open(transcripts_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            # 按第一个空白块（空格或制表符）分割字符串，只分割一次
                            # \s+ 匹配一个或多个任意空白字符
                            parts = re.split(r'\s+', line, 1)
                            # 确保分割后至少有两部分：ID 和 文本
                            if len(parts) == 2:
                                utt_id, text = parts
                                transcripts[utt_id] = text
                                if random.random() <0.000001:
                                    print(utt_id, text)
                            else:
                                # 如果一行只有一个ID没有文本，可以选择忽略或记录
                                print(f"Warning: Skipping malformed line (no text found): {line}")
                        except Exception as e:
                            print(f"Warning: Could not parse line: '{line}'. Error: {e}")
            print(f"Parsed {len(transcripts)} unique utterances.")
            print("\nscanning audio files for duration")
            with ProcessPoolExecutor(max_workers=80) as executor:
                futures = (executor.submit(get_duration_worker, utt_id, text, dataset, dataset_root) 
                        for utt_id, text in transcripts.items())
                for future in tqdm(as_completed(futures), total=len(transcripts), desc="Scanning audio files"):
                    try:
                        result = future.result()
                        if result:
                            metadata.append(result)
                    except Exception as e:
                        print(f"A worker process failed with an unexpected error: {e}")
                
    total_hours = sum([m['duration'] for m in metadata]) / 3600
    print(f"\nFound {len(metadata)} valid audio-text pairs. Total duration: {total_hours:.2f} hours.")
    # 后面统一保存为简称
    if dataset == "voxpopuli" and in_language in interset:
        new_cache_file = output_dir /f"metadata_{in_language}_vox_cache.pkl"
    elif dataset == "mls":
        new_cache_file = output_dir /f"metadata_{in_language}_mls_cache.pkl"
    elif dataset == "lemas":
        new_cache_file = output_dir /f"metadata_{in_language}_lemas_cache.pkl"
    else:
        new_cache_file = output_dir /f"metadata_{in_language}_cache.pkl"
    print(f"\nSaving scanned metadata to cache file: '{new_cache_file}'")
    with open(new_cache_file, 'wb') as f:
        pickle.dump(metadata, f)
    print("Cache saved.")
    return metadata

def process_and_link_item(item,dataset,in_language,output_dir):
    """
    为单个样本创建符号链接，并返回用于写入CSV的数据
    """
    try:
        
        if dataset=="emilia":
            target_speaker_dir = output_dir / "wavs" / in_language
        else:
            target_speaker_dir = output_dir / "wavs" / in_language / item["speaker_id"]
        target_speaker_dir.mkdir(parents=True, exist_ok=True)
        
        source_audio_path = item["audio_path"]
        target_audio_path = target_speaker_dir / source_audio_path.name
        
        # 创建软链接
        if not target_audio_path.exists():
            os.symlink(source_audio_path.resolve(), target_audio_path)
        
        # 返回软连接的相对路径
        if dataset=="emilia":
            csv_path = Path(in_language) /  source_audio_path.name
        else:
            csv_path = Path(in_language) / item["speaker_id"] / source_audio_path.name
        
        return (csv_path.as_posix(), item["text"], item['duration'])
        
    except Exception as e:
        print(f"Error processing {item['audio_path']}: {e}", file=sys.stderr)
        return None
def process_single_item(m, target_lang):
    correct_lang = False
    if target_lang not in ['mt','nl']:
        try:
            langs = detect_langs(m['text'])
            for result in langs[:3]:
                if result.lang == target_lang:
                    correct_lang = True
                    break
        except LangDetectException:
            return f"ERROR|0|{m['text']}\n", None
    else:
        correct_lang = True

    # 计算时长比例
    duration = m.get('duration', 0)
    if duration > 0:
        each_len = len(m['text'].split()) / duration
    else:
        each_len = 0
    csv_row = f"{correct_lang}|{each_len}|{m['text']}\n"
    if random.random()<0.00001:
        print(each_len, correct_lang)
    if 0.85 <= each_len <= 2.25 and correct_lang:
        return csv_row, m  
    else:
        return csv_row, None  

def main(args):
    in2lang = { 
        "th":"thai", "id":"indonesian", "vi":"vietnamese", 
        "zh":"chinese", "en":"english",
        "de":"german", "fr":"french", "es":"spanish", "pl":"polish", "it":"italian", "nl":"dutch", "pt":"portuguese",
        "ko":"korean", "ja":"japanese", "ru":"russian",
        "ro":"romanian","hu":"hungarian","cs":"czech","fi":"finnish","hr":"croatian","sk":"slovak","sl":"slovenian","et":"estonian",
        "lt":"lithuanian","bg":"bulgarian","el":"greek","lv":"latvian","mt":"maltese","sv":"swedish","da":"danish",
    }
    lang2in = {value: key for key, value in in2lang.items()}
    interset = ["de", "fr", "es", "pl", "it", "nl", "pt"] # mls和voxpopuli的交集
    if args.language in in2lang:
        in_language=args.language
        language=in2lang[in_language]
    elif args.language in lang2in:
        language=args.language
        in_language=lang2in[language]
    else:
        print("invaild language")
        return 
    root_map = {
        "mls": Path(f"/inspire/dataset/multilingual-librispeech/v1/mls_{language}/train"),
        "gigaspeech2": Path(f"/inspire/dataset/gigaspeech2/v1/data/{in_language}"),
        "emilia": Path("/inspire/dataset/voxbox/v1"),
        "voxpopuli": Path(f"/inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/zhikang/voxpopuli-data/unlabelled_data/{in_language}"),
        "koreaspeech":Path(f"/inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/zhikang/KoreaSpeech/data"),
        "lemas": Path(f"/inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/zhikang/LEMAS-Dataset-train/train/{in_language}")
    }
    transcripts_map = {
        "mls": "transcripts.txt",
        "gigaspeech2": "train_refined.tsv",
        "emilia": f"metadata/emilia_{in_language}.jsonl",
        "voxpopuli":f"{in_language}_asr.jsonl",
        "koreaspeech": None,
        "lemas": None,  
    }
    # 对于voxpopuli，增加一个标注文本的目录
    transcripts_root= f"/inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/zhikang/nv-granary/{in_language}/voxpopuli" if args.dataset=="voxpopuli" else None
    dataset_root = root_map[args.dataset]
    transcripts_name = transcripts_map[args.dataset]
    if not dataset_root.exists():
        print(f"Error: MLS source directory not found at {dataset_root}")
        sys.exit(1)
    output_dir=Path(args.output_dir)    
    
    metadata = collect_metadata(
        dataset=args.dataset, 
        dataset_root=dataset_root, 
        language=language, 
        in_language=in_language, 
        workers=args.num_workers,
        output_dir=output_dir,
        transcripts_name=transcripts_name,
        transcripts_root=transcripts_root,
        force_rescan=args.force_rescan,
        interset=interset
        )
    print(f"\nFiltering utterances...")
    if args.dataset== "voxpopuli":
        csv_rows=[]
        filtered_metadata=[]
    #     for m in tqdm(metadata, desc="Filtering Data"):
    #         correct_lang = False
    #         try:
    #             langs = detect_langs(m['text'])
    #             for result in langs[:3]:
    #                 if result.lang == in_language:
    #                     correct_lang = True
    #                     break
    #         except LangDetectException:
    #             tqdm.write(f"Unable to detect language: {m['text']}, skip it")
    #             continue
            
    #         each_len=len(m['text'].split())/m['duration']
    #         csv_rows.append(f"{correct_lang}|{each_len}|{m['text']}\n")
    #         # 0.85-2.25
    #         if each_len>=0.85 and each_len<=2.25 and correct_lang:
    #             filtered_metadata.append(m)
    #     metadata_path = output_dir / f"{in_language }per.csv"
    #     print(f"\nWriting language-specific metadata to {metadata_path}")
    #     with open(metadata_path, "w", encoding="utf-8") as f:
    #         f.write("len|detect_len|text\n")
    #         csv_rows.sort()
    #         f.writelines(csv_rows)
    
        num_workers = 64
        print(f"Starting multiprocessing with {num_workers} workers...")
        worker_func = partial(process_single_item, target_lang=in_language)
        results = process_map(
            worker_func, 
            metadata, 
            max_workers=num_workers, 
            chunksize=10, 
            desc="Filtering Data (Multi-process)"
        )
        for row_str, m_data in results:
            csv_rows.append(row_str)
            if m_data is not None:
                filtered_metadata.append(m_data)
    else:
        filtered_metadata = [m for m in metadata if min_duration <= m['duration'] <= max_duration]
    filtered_hours = sum([m['duration'] for m in filtered_metadata]) / 3600
    print(f"After filtering, {len(filtered_metadata)} utterances remain. Total duration: {filtered_hours:.2f} hours.")

    # 随机选择目标时长的样本
    random.shuffle(filtered_metadata)
    
    target_seconds = args.duration * 3600
    if target_seconds == -1:
        target_seconds = filtered_hours * 3600
    tasks_to_process, current_duration = [], 0
    for item in filtered_metadata:
        if current_duration >= target_seconds: break
        tasks_to_process.append(item)
        current_duration += item['duration']

    print(f"\nSelected {len(tasks_to_process)} utterances for processing.")
    print(f"Estimated duration: {current_duration / 3600:.2f} hours.")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "wavs" / in_language).mkdir(parents=True, exist_ok=True)

    # 并行处理并收集CSV数据
    print("\nCreating symbolic links and collecting metadata...")
    csv_rows = []
    final_duration = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_and_link_item, item, args.dataset,in_language, output_dir): item for item in tasks_to_process}
        pbar = tqdm(total=len(tasks_to_process), desc="Processing")
        for future in as_completed(futures):
            result = future.result()
            if result:
                csv_path, csv_text, duration = result
                if args.write_duration:
                    csv_rows.append(f"{csv_path}|{duration}|{csv_text}\n")
                else:
                    csv_rows.append(f"{csv_path}|{csv_text}\n")
                final_duration += duration
            pbar.update(1)
            pbar.set_postfix_str(f"Hours: {final_duration/3600:.2f}")
        pbar.close()

    # 写入特定语言的 metadata.csv
    if args.dataset=="voxpopuli" and in_language in interset:
        csv_filename = f"metadata_{in_language}_vox.csv"
    elif args.dataset=="mls":
        csv_filename = f"metadata_{in_language}_mls.csv"
    elif args.dataset=="lemas":
        csv_filename = f"metadata_{in_language}_lemas.csv"
    else:
        csv_filename = f"metadata_{in_language}.csv"
    if args.write_duration:
        csv_filename = csv_filename.replace(".csv", "_with_duration.csv")
    metadata_path = output_dir / csv_filename
    print(f"\nWriting language-specific metadata to {metadata_path}")
    with open(metadata_path, "w", encoding="utf-8") as f:
        if args.write_duration:
            f.write("file_path|duration|text\n")
        else:
            f.write("file_path|text\n")
        csv_rows.sort()
        f.writelines(csv_rows)
        
    print(f"\n{in_language} dataset complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Options for data processing")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True,
        help="dataset name, supported: mls, emilia, gigaspeech2, voxpopuli, koreaspeech, lemas "
    )
    parser.add_argument(
        "--language", 
        type=str, 
        required=True, 
        help="language, either full name or short name"
    )
    parser.add_argument(
        "--duration", 
        type=int, 
        default=5000,
        help="time duration of the subset, -1 for the whole set"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=64,
        help="num_workers for ProcessPoolExecutor"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/inspire/hdd/project/multilingualspeechrecognition/chenxie-25019/rixixu/datasets",
        help="where to store the wavs and metadata"
    )
    parser.add_argument(
        "--force_rescan",
        action='store_true',
        help="whether to force rescan metadata.pkl"
    )
    parser.add_argument(
        "--write_duration",
        action='store_true',
        help="whether to write duration column into csv"
    )
     
    args = parser.parse_args()
    main(args)