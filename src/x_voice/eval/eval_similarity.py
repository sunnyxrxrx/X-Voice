import os
import sys
import numpy as np
import argparse
import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import math
import torch.multiprocessing as mp
from x_voice.eval.ecapa_tdnn import ECAPA_TDNN_SMALL
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Extract Speaker Embeddings using WavLM-ECAPA.')
parser.add_argument('--wavlm_ckpt_dir', default='', type=str, help='Path to WavLM-ECAPA checkpoint')
parser.add_argument('--prompt_wavs', default='', type=str, help='prompt_wav.scp')
parser.add_argument('--hyp_wavs', default='', type=str, help='wav.scp')
parser.add_argument('--log_file', default='spk_simi_scores.txt', type=str, help='File to save log')
parser.add_argument('--num_gpus', default=1, type=int, help='Number of GPUs to use')
parser.add_argument('--dump_dir', default='', type=str, help='Root dir for GT/Prompt wavs')
parser.add_argument('--decode_dir', default='', type=str, help='Root dir for Gen wavs')

def load_and_preprocess(wav_path, device, target_sr=16000):
    wav, sr = torchaudio.load(wav_path)
    if wav.shape[0] > 1:
        wav = wav[0, :].unsqueeze(0)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)
    return wav.to(device)

@torch.no_grad()
def compute_embedding(model, wav_path, device):
    wav = load_and_preprocess(wav_path, device)
    emb = model(wav)
    return emb

def worker(rank, args, sub_hyp_list, prompt_wavs_map, return_dict):
    device = f"cuda:{rank}"
    
    try:
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
        state_dict = torch.load(args.wavlm_ckpt_dir, weights_only=True, map_location='cpu')
        model.load_state_dict(state_dict["model"], strict=False)
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"GPU {rank} initialization failed: {e}")
        return

    results = []
    pbar = tqdm(sub_hyp_list, desc=f"GPU {rank}", unit="it", disable=(rank != 0))

    for uttid, hyp_rel_path in sub_hyp_list:
        lookup_id = uttid
        if lookup_id not in prompt_wavs_map:
            try:
                now_id = uttid.split("_")[1]
                lookup_id = "uttid_" + now_id
            except:
                pass
        
        if lookup_id not in prompt_wavs_map:
            continue

        try:
            prompt_full_path = os.path.join(args.dump_dir, prompt_wavs_map[lookup_id])
            hyp_full_path = os.path.join(args.decode_dir, hyp_rel_path) 

            prompt_emb = compute_embedding(model, prompt_full_path, device)
            hyp_emb = compute_embedding(model, hyp_full_path, device)
            
            # compute cosine similarity
            hyp_score = F.cosine_similarity(prompt_emb, hyp_emb).item()

            if not math.isnan(hyp_score):
                results.append((uttid, hyp_score))
        except Exception as e:
            print(f"Error processing {uttid} on GPU {rank}: {e}")
            continue
        
        if rank == 0:
            pbar.update(1)

    return_dict[rank] = results

def main():
    args = parser.parse_args()

    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        print("[ERROR] No CUDA devices found. This script requires GPUs.")
        return
    
    num_gpus = min(args.num_gpus, available_gpus)
    print(f"[INFO] Requested {args.num_gpus} GPUs, using {num_gpus} available GPUs.")

    def load_scp(path):
        data = {}
        if path and os.path.exists(path):
            with open(path, "rt") as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        data[parts[0]] = parts[1]
        return data

    prompt_wavs_map = load_scp(args.prompt_wavs)
    hyp_wavs_list = []
    if os.path.exists(args.hyp_wavs):
        with open(args.hyp_wavs, "rt") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    hyp_wavs_list.append((parts[0], parts[1]))

    total_files = len(hyp_wavs_list)
    if total_files == 0:
        print("[ERROR] No files found in hyp_wavs.")
        return

    print(f'Total files to process: {total_files}')

    # Split hyp_wavs_list into num_gpus chunks
    chunks = [hyp_wavs_list[i::num_gpus] for i in range(num_gpus)]

    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for rank in range(num_gpus):
        p = mp.Process(target=worker, args=(rank, args, chunks[rank], prompt_wavs_map, return_dict))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Aggregate results from all GPUs
    all_results = []
    for rank in range(num_gpus):
        if rank in return_dict:
            all_results.extend(return_dict[rank])

    print(f"\n[INFO] Saving results to {args.log_file}...")
    os.makedirs(os.path.dirname(args.log_file) if os.path.dirname(args.log_file) else ".", exist_ok=True)
    
    hyp_scores = [res[1] for res in all_results]
    
    with open(args.log_file, "wt") as out_f:
        for uttid, score in all_results:
            out_f.write(f"{uttid} {score:.4f}\n")
        
        if len(hyp_scores) > 0:
            avg_hyp = np.mean(hyp_scores)
            out_f.write(f"avg  {avg_hyp:.4f}\n")
            print(f"Final Avg Similarity Score: {avg_hyp:.4f}")
        else:
            print("[WARNING] No valid scores were calculated.")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()