import os
import sys
from pathlib import Path

sys.path.insert(0, str((Path(__file__) / "../../..").resolve()))

import json
import shutil
import pyphen
import re
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from datasets.arrow_writer import ArrowWriter

from model.utils import repetition_found

SEC_PER_HOUR = 3600

out_zh = {
    "ZH_B00041_S06226",
    "ZH_B00042_S09204",
    "ZH_B00065_S09430",
    "ZH_B00065_S09431",
    "ZH_B00066_S09327",
    "ZH_B00066_S09328",
}
zh_filters = ["い", "て"]
# seems synthesized audios, or heavily code-switched
out_en = {
    "EN_B00013_S00913",
    "EN_B00042_S00120",
    "EN_B00055_S04111",
    "EN_B00061_S00693",
    "EN_B00061_S01494",
    "EN_B00061_S03375",
    "EN_B00059_S00092",
    "EN_B00111_S04300",
    "EN_B00100_S03759",
    "EN_B00087_S03811",
    "EN_B00059_S00950",
    "EN_B00089_S00946",
    "EN_B00078_S05127",
    "EN_B00070_S04089",
    "EN_B00074_S09659",
    "EN_B00061_S06983",
    "EN_B00061_S07060",
    "EN_B00059_S08397",
    "EN_B00082_S06192",
    "EN_B00091_S01238",
    "EN_B00089_S07349",
    "EN_B00070_S04343",
    "EN_B00061_S02400",
    "EN_B00076_S01262",
    "EN_B00068_S06467",
    "EN_B00076_S02943",
    "EN_B00064_S05954",
    "EN_B00061_S05386",
    "EN_B00066_S06544",
    "EN_B00076_S06944",
    "EN_B00072_S08620",
    "EN_B00076_S07135",
    "EN_B00076_S09127",
    "EN_B00065_S00497",
    "EN_B00059_S06227",
    "EN_B00063_S02859",
    "EN_B00075_S01547",
    "EN_B00061_S08286",
    "EN_B00079_S02901",
    "EN_B00092_S03643",
    "EN_B00096_S08653",
    "EN_B00063_S04297",
    "EN_B00063_S04614",
    "EN_B00079_S04698",
    "EN_B00104_S01666",
    "EN_B00061_S09504",
    "EN_B00061_S09694",
    "EN_B00065_S05444",
    "EN_B00063_S06860",
    "EN_B00065_S05725",
    "EN_B00069_S07628",
    "EN_B00083_S03875",
    "EN_B00071_S07665",
    "EN_B00071_S07665",
    "EN_B00062_S04187",
    "EN_B00065_S09873",
    "EN_B00065_S09922",
    "EN_B00084_S02463",
    "EN_B00067_S05066",
    "EN_B00106_S08060",
    "EN_B00073_S06399",
    "EN_B00073_S09236",
    "EN_B00087_S00432",
    "EN_B00085_S05618",
    "EN_B00064_S01262",
    "EN_B00072_S01739",
    "EN_B00059_S03913",
    "EN_B00069_S04036",
    "EN_B00067_S05623",
    "EN_B00060_S05389",
    "EN_B00060_S07290",
    "EN_B00062_S08995",
}
en_filters = ["ا", "い", "て"]

def count_syllables(text):
    dic = pyphen.Pyphen(lang='en_US')
    total_syllables = 0
    
    # Define the regular expression
    #   [a-zA-Z']+  Matches one or more English letters or apostrophes (to handle words like "don't", "haven't")
    #   [\u4e00-\u9fff] Matches a single Chinese character
    pattern = re.compile(r"[a-zA-Z']+|[\u4e00-\u9fff]")
    tokens = pattern.findall(text)
    
    for token in tokens:
        if '\u4e00' <= token <= '\u9fff':
            total_syllables += 1
        else:
            try:
                syllables = dic.inserted(token.lower()).split("-")
                total_syllables += len(syllables)
            except Exception:
                total_syllables += 1
                
    return total_syllables

def check_valid_chars(input_str):
    valid_punctuation = '\'",.?!;。，'
    for c in input_str:
        # Check if it's a Latin letter (including uppercase and lowercase)
        if c.isalpha() and ord(c) <= 127:
            continue
        # Check if it's a Chinese character
        if '\u3100' <= c <= '\u9fff':
            continue
        # Check if it's a specified punctuation mark
        if c in valid_punctuation:
            continue
        if c == ' ':
            continue
        return False
        
    return True

def process_speed(audio_jsonl, output_path, total_duration, max_duration):
    judge = False
    results = []
    
    with open(audio_jsonl, 'r') as f:
        lines = f.readlines()
        for line in lines:
            obj = json.loads(line)
            duration = obj["duration"]
            text = obj["text"]
            obj["speed_syllables"] = map_to_class(count_syllables(text) / duration)
            if not check_valid_chars(text) or duration < 2.4:
                continue
            total_duration += duration
            results.append(obj)
            if total_duration > max_duration:
                judge = True
                break
    
    output_jsonl = output_path / audio_jsonl.name
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, 'w', encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')                        
    
    return judge, total_duration

def process_speed_val(audio_jsonl, output_path, val_sample_num):
    results = []
    total_num = 0
    
    with open(audio_jsonl, 'r') as f:
        lines = f.readlines()
        for line in lines:
            obj = json.loads(line)
            duration = obj["duration"]
            text = obj["text"]
            obj["speed_syllables"] = map_to_class(count_syllables(text) / duration)
            if not check_valid_chars(text) or (duration < 2.4 or duration > 10):
                continue
            total_num += 1
            results.append(obj)
            if total_num >= val_sample_num:
                print(f"total_num of audio_jsonl: {total_num}")
                break
    
    output_jsonl = output_path / audio_jsonl.name
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, 'w', encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')                        

def map_to_class(speed, delta=0.25):
    return round(speed / delta) * delta

def deal_with_dir(json_dir, dataset_path):
    audio_jsonl = json_dir
    sub_result, durations = [], []
    bad_case_zh = 0
    bad_case_en = 0
    with open(audio_jsonl, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc=f"{audio_jsonl.stem}"):
            obj = json.loads(line)
            text = obj["text"]
            if obj["language"] == "zh":
                if obj["wav"].split("/")[1] in out_zh or any(f in text for f in zh_filters) or repetition_found(text):
                    bad_case_zh += 1
                    continue
                else:
                    text = text.translate(
                        str.maketrans({",": "，", "!": "！", "?": "？"})
                    )  # not "。" cuz much code-switched
            if obj["language"] == "en":
                if (
                    obj["wav"].split("/")[1] in out_en
                    or any(f in text for f in en_filters)
                    or repetition_found(text, length=4)
                ):
                    bad_case_en += 1
                    continue
            duration = obj["duration"]
            speed_syllables = obj["speed_syllables"]
            text = obj["text"]
            sub_result.append({
                "audio_path": str(dataset_path / obj["wav"]), 
                "text": text,
                "duration": duration,
                "speed_syllables": speed_syllables
            })
            durations.append(duration)
    return sub_result, durations, bad_case_zh, bad_case_en

def main():
    # Step 1: Calculate Speed & Generate Intermediate JSONLs
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR)

    duration_stats = dict()
    # train set
    for lang in LANGS:
        total_duration = 0
        raw_lang_path = Path(DATASET_DIR) / lang
        tmp_lang_path = Path(TMP_DIR) / lang
        for audio_jsonl in sorted(raw_lang_path.glob('*.jsonl')):
            if "EN_B00113" in str(audio_jsonl) or "ZH_B00091" in str(audio_jsonl):
                continue
            judge, total_duration = process_speed(audio_jsonl, tmp_lang_path, total_duration, MAX_DURATION_HOURS * SEC_PER_HOUR)
            print(f"{audio_jsonl.name}: Total duration of {lang} now: {total_duration / SEC_PER_HOUR}h")
            if judge:
                duration_stats[lang] = total_duration
                break
    for lang, duration in duration_stats.items():
        print(f"{lang}\'s duration: {duration / SEC_PER_HOUR}")
    
    # validation set
    for lang in LANGS:
        dataset_path = Path(DATASET_DIR) / lang
        output_path = Path(TMP_DIR) / lang
        for audio_jsonl in sorted(dataset_path.glob('*.jsonl')):
            if "EN_B00113" in str(audio_jsonl) or "ZH_B00091" in str(audio_jsonl):
                process_speed_val(audio_jsonl, output_path, VAL_SAMPLE_NUM)
    print(f"Finish preprocessing.")

    # Step 2: Final Processing & Arrow Packing
    if not os.path.exists(FINAL_SAVE_DIR):
        os.makedirs(FINAL_SAVE_DIR)
    for split_name in ["train", "val"]:
        print(f"\nProcessing {split_name} split...")
        
        final_result = []
        final_durations = []
        total_bad_zh = 0
        total_bad_en = 0

        executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
        futures = []
        
        for lang in LANGS:
            tmp_lang_path = Path(TMP_DIR) / lang
            raw_lang_path = Path(DATASET_DIR) / lang
            
            for speed_jsonl in tmp_lang_path.iterdir():
                is_val_file = "EN_B00113" in str(speed_jsonl) or "ZH_B00091" in str(speed_jsonl)
                
                if split_name == "train" and is_val_file:
                    continue
                if split_name == "val" and not is_val_file:
                    continue
                
                futures.append(executor.submit(deal_with_dir, speed_jsonl, raw_lang_path))
        
        for f in tqdm(futures, total=len(futures), desc=f"Gathering {split_name}"):
            sub_res, durs, b_zh, b_en = f.result()
            final_result.extend(sub_res)
            final_durations.extend(durs)
            total_bad_zh += b_zh
            total_bad_en += b_en
        
        executor.shutdown()

        arrow_filename = "raw.arrow" if split_name == "train" else "raw_val.arrow"
        duration_filename = "duration.json" if split_name == "train" else "duration_val.json"
        
        print(f"Saving {arrow_filename} ...")
        with ArrowWriter(path=f"{FINAL_SAVE_DIR}/{arrow_filename}") as writer:
            for line in tqdm(final_result, desc="Writing Arrow"):
                writer.write(line)
            writer.finalize()

        with open(f"{FINAL_SAVE_DIR}/{duration_filename}", "w", encoding="utf-8") as f:
            json.dump({"duration": final_durations}, f, ensure_ascii=False)
        
        print(f"Stats for {split_name}:")
        print(f"  Samples: {len(final_result)}")
        print(f"  Total Duration: {sum(final_durations)/3600:.2f} hours")
        print(f"  Bad ZH: {total_bad_zh}")
        print(f"  Bad EN: {total_bad_en}")

    # Step 3: remove TMP_DIR
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
        print(f"Removed {TMP_DIR}")
    
    print("\nAll Done!")

if __name__ == "__main__":
    # 500 * 2 = 1000h
    MAX_DURATION_HOURS = 5 # 500 The total duration of the dataset for each language in hours.
    VAL_SAMPLE_NUM = 250
    MAX_WORKERS = 32
    LANGS = ["ZH", "EN"]
    DATASET_DIR = "<SOME_PATH>/Emilia_Dataset/raw"
    TMP_DIR = str(Path(__file__).parent.parent.parent.parent / "Speed_Emilia_use")
    DATASET_NAME = f"Emilia_{'_'.join(LANGS)}_speed"
    FINAL_SAVE_DIR = str(Path(__file__).parent.parent.parent.parent / "data" / DATASET_NAME)
    print(f"\nPrepare for {DATASET_NAME}, will save to {FINAL_SAVE_DIR}\n")
    
    main()