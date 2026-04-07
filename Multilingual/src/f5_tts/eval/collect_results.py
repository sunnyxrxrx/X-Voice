import os
import re
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Collect evaluation results for multilingual TTS")
    parser.add_argument("--decode_dir", type=str, required=True, help="Results root directory")
    parser.add_argument("--test_set", type=str, required=True, help="Space-separated list of languages")
    parser.add_argument("--ref_set", type=str, default=None, help="Space-separated list of languages")
    return parser.parse_args()

def main():
    args = parse_args()
    decode_dir = args.decode_dir
    test_set = args.test_set.split()
    ref_set = None
    if args.ref_set:
        ref_set = args.ref_set.split()
        summary_file = os.path.join(decode_dir, "summary_results_cross.csv")
    else:
        summary_file = os.path.join(decode_dir, "summary_results.csv")

    results = []
    print("collecting")
    for i, lang in enumerate(test_set):
        if ref_set:
            ref_lang = ref_set[i]
            lang_path = os.path.join(decode_dir, f"{ref_lang}-{lang}")
            if not os.path.exists(lang_path):
                lang_path = os.path.join(decode_dir, f"{ref_lang}_{lang}")
        else:
            lang_path = os.path.join(decode_dir, lang)
        row = {"Language": lang, "WER": "N/A", "SIM": "N/A", "UTMOS": "N/A", "DNSMOS": "N/A"}
        
        # 解析 WER
        wer_file = os.path.join(lang_path, "wav_res_ref_text.wer")
        if os.path.exists(wer_file):
            with open(wer_file, 'r') as f:
                content = f.read()
                match = re.search(r"WER:\s*([\d.]+)", content)
                if match:
                    row["WER"] = match.group(1)

        # 解析 SIM
        sim_file = os.path.join(lang_path, "spk_simi_scores.txt")
        if os.path.exists(sim_file):
            with open(sim_file, 'r') as f:
                lines = f.readlines()
                for line in reversed(lines):
                    if line.startswith("avg"):
                        row["SIM"] = line.split()[-1]
                        break

        # 解析 UTMOS
        utmos_file = os.path.join(lang_path, "_utmos_results.jsonl")
        if os.path.exists(utmos_file):
            with open(utmos_file, 'r') as f:
                content = f.read()
                match = re.search(r"UTMOS:\s*([\d.]+)", content)
                if match:
                    row["UTMOS"] = match.group(1)
                else:
                    # 备选方案：如果没找到统计行，手动计算平均值
                    f.seek(0)
                    scores = []
                    for line in f:
                        try:
                            data = json.loads(line)
                            if "utmos" in data:
                                scores.append(float(data["utmos"]))
                        except:
                            continue
                    if scores:
                        row["UTMOS"] = f"{sum(scores) / len(scores):.4f}"

        # 解析 DNSMOS
        dns_file = os.path.join(lang_path, "dnsmos_mean.txt")
        if os.path.exists(dns_file):
            with open(dns_file, 'r') as f:
                row["DNSMOS"] = f.read().strip()

        results.append(row)

    # 创建 DataFrame
    df = pd.DataFrame(results)

    print("\n" + "="*50)
    print("汇总评估结果")
    print("="*50)
    print(df.to_string(index=False))
    print("="*50)

    # 保存到本地 CSV
    df.to_csv(summary_file, index=False)
    print(f"结果已保存至: {summary_file}")
    
if __name__ == "__main__":
    main()
