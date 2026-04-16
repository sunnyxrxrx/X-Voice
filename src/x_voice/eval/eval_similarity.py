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
from x_voice.eval.ecapa_tdnn import ECAPA_TDNN_SMALL

parser = argparse.ArgumentParser(description='Extract Speaker Embeddings using WavLM-ECAPA.')
parser.add_argument('--ckpt_path', default='', type=str, help='Path to WavLM-ECAPA checkpoint')

parser.add_argument('--prompt_wavs', default='', type=str, help='prompt_wav.scp')
parser.add_argument('--hyp_wavs', default='', type=str, help='wav.scp')
parser.add_argument('--log_file', default='spk_simi_scores.txt', type=str, help='File to save log')
parser.add_argument('--devices', default="0", type=str, help='GPU device id')
parser.add_argument('--dump_dir', default='', type=str, help='Root dir for GT/Prompt wavs')
parser.add_argument('--decode_dir', default='', type=str, help='Root dir for Gen wavs')

def main():
    args = parser.parse_args()
    device = f"cuda:{args.devices}" if torch.cuda.is_available() else "cpu"
    
    print(f"[INFO] Loading WavLM-ECAPA model from {args.ckpt_path}...")
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state_dict = torch.load(args.ckpt_path, weights_only=True, map_location='cpu')
    model.load_state_dict(state_dict["model"], strict=False)
    model = model.to(device)
    model.eval()

    def load_and_preprocess(wav_path, target_sr=16000):
        wav, sr = torchaudio.load(wav_path)
        # Convert multi-channel audio to mono.
        if wav.shape[0] > 1:
            wav = wav[0, :].unsqueeze(0)
        # Resample to the target sample rate.
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            wav = resampler(wav)
        return wav.to(device)

    @torch.no_grad()
    def compute_embedding(wav_path):
        wav = load_and_preprocess(wav_path)
        emb = model(wav)
        return emb # [1, embedding_size]

    # Parse the SCP-style file lists.
    def load_scp(path):
        data = {}
        if path and os.path.exists(path):
            for line in open(path, "rt").readlines():
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    data[parts[0]] = parts[1]
        return data

    prompt_wavs_map = load_scp(args.prompt_wavs)
    hyp_wavs_list = []
    if os.path.exists(args.hyp_wavs):
        for line in open(args.hyp_wavs, "rt").readlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                hyp_wavs_list.append((parts[0], parts[1]))

    print(f'Calculating similarities for {len(hyp_wavs_list)} files...')
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    out_f = open(args.log_file, "wt")
    
    hyp_scores = []
    pbar = tqdm(hyp_wavs_list, desc="Scoring SIM (WavLM)", unit="it", ncols=100)

    for uttid, hyp_rel_path in pbar:
        if uttid not in prompt_wavs_map:
            # print(f"{uttid} not in prompt_wavs_map")
            now_id = uttid.split("_")[1]
            uttid = "uttid_"+now_id
        
        try:
            # Resolve the prompt audio path and extract its embedding.
            prompt_full_path = os.path.join(args.dump_dir, prompt_wavs_map[uttid])
            # print(prompt_full_path)
            prompt_emb = compute_embedding(prompt_full_path)

            hyp_full_path = os.path.join(args.decode_dir, hyp_rel_path) 
            hyp_emb = compute_embedding(hyp_full_path)
            hyp_score = F.cosine_similarity(prompt_emb, hyp_emb).item()

            if not math.isnan(hyp_score):
                hyp_scores.append(hyp_score)
                out_f.write(f"{uttid} {hyp_score:.4f}\n")
                out_f.flush()
            
        except Exception as e:
            pbar.write(f"Error processing {uttid}: {e}")
            continue

    if len(hyp_scores) > 0:
        avg_hyp = np.mean(hyp_scores)
        out_f.write(f"avg  {avg_hyp:.4f}\n")
        print(f"\nAvg Similarity: {avg_hyp:.4f}")
    
    out_f.close()

if __name__ == '__main__':
    main()
