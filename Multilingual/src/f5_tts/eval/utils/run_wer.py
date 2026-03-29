import sys, os
import torch
from tqdm import tqdm
import multiprocessing
from jiwer import compute_measures
from zhon.hanzi import punctuation
import string
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration 
import soundfile as sf
import scipy
import zhconv
from funasr import AutoModel
import whisper
from whisper.normalizers import EnglishTextNormalizer, BasicTextNormalizer
from nemo_text_processing.text_normalization.normalize import Normalizer
import unicodedata

import re
from f5_tts.eval.text_normalizer import TextNormalizer
import editdistance

# import debugpy
# debugpy.listen(('localhost', 5678))
# print("Waiting for debugger attach")
# debugpy.wait_for_client()

wav_res_text_path = sys.argv[1]
res_path = sys.argv[2]
lang = sys.argv[3] # zh or en

device = "cuda" if torch.cuda.is_available() else 'cpu'

def load_en_model(faster=False):
   
    if faster:
        from faster_whisper import WhisperModel

        model_id = "large-v3" 
        print(f"[INFO] Using asr model: faster whisper {model_id}")
        model = WhisperModel(model_id, device="cuda", compute_type="float16")
    else:
        model_id = "large-v3"
        print(f"[INFO] Using asr model: whisper {model_id}")
        model = whisper.load_model(model_id).to(device)
        model.eval()
    return model

def load_zh_model():
    local_model_path = "/inspire/hdd/project/embodied-multimodality/chenxie-25019/rixixu/paraformer"
    print(f"[INFO] 从本地加载ASR模型: {local_model_path}")
    
    model = AutoModel(
        model=local_model_path,  # 直接填本地文件夹路径
        disable_update=True      # 禁止联网检查更新，纯离线运行
    )
    return model

def normalize_one(hypo, truth, normalizer):
    truth = unicodedata.normalize('NFKC', truth)
    hypo = unicodedata.normalize('NFKC', hypo)
    raw_truth = truth
    raw_hypo = hypo
    if normalizer:
        truth = normalizer.normalize(truth, post=True)
        hypo = normalizer.normalize(hypo, post=True)
    if lang[-2:] in ["zh", "ja", "ko","th"]:
        truth = " ".join([x for x in truth])
        hypo = " ".join([x for x in hypo]) # 中文hypo自带空格
    truth = truth.lower()
    hypo = hypo.lower()
        
    def clean_special_chars(text):
        # 移除所有标点来进行比较
        cleaned = "".join(
            ch for ch in text 
            if unicodedata.category(ch)[0] not in ('P', 'S') 
        )
        # 将多个空格合并，并去掉首尾空格
        return " ".join(cleaned.split())
    
    truth = clean_special_chars(truth)
    hypo = clean_special_chars(hypo)
    return truth, hypo

def process_one(hypo, truth):
    measures = compute_measures(truth, hypo)
    ref_list = truth.split(" ")
    wer = measures["wer"]
    subs = measures["substitutions"] / len(ref_list)
    dele = measures["deletions"] / len(ref_list)
    inse = measures["insertions"] / len(ref_list)
    print(wer)
    return (truth, hypo, wer, subs, dele, inse)


def run_asr(wav_res_text_path, res_path, normalize_text=False, faster=False, whole=False):
    if lang[-2:] in ["zh", "hard_zh"]:
        model = load_zh_model()
    else:
        model = load_en_model(faster)

    params = []
    for line in open(wav_res_text_path).readlines():
        line = line.strip()
        if len(line.split('|')) == 2:
            wav_res_path, text_ref = line.split('|')
        elif len(line.split('|')) == 3:
            wav_res_path, wav_ref_path, text_ref = line.split('|')
        elif len(line.split('|')) == 4: # for edit
            wav_res_path, _, text_ref, wav_ref_path = line.split('|')
        else:
            raise NotImplementedError

        if not os.path.exists(wav_res_path):
            continue
        params.append((wav_res_path, text_ref))
    fout = open(res_path, "w")
    
    if normalize_text:
        normalizer = TextNormalizer(language=lang[-2:])
    else: 
        normalizer = None
    for wav_res_path, text_ref in tqdm(params):
        try:
            if lang[-2:] in ["zh", "hard_zh"]:
                res = model.generate(input=wav_res_path,
                        batch_size_s=300)
                transcription = res[0]["text"]
            else:
                if faster:
                    segments, _ = model.transcribe(
                        wav_res_path, 
                        beam_size=5, 
                        language=lang[-2:]
                        )
                    transcription = ""
                    for segment in segments:
                        transcription = transcription + " " + segment.text
                else:
                    result = model.transcribe(
                        wav_res_path, 
                        language=lang[-2:])
                    transcription = result["text"].strip()
                
        except Exception as e:
            print(e)
            continue
        if 'zh' in lang:
            transcription = zhconv.convert(transcription, 'zh-cn')

        truth_normed, hypo_normed = normalize_one(transcription, text_ref, normalizer)  

        if not whole:
            truth, hypo, wer, subs, dele, inse = process_one(hypo_normed, truth_normed)
            fout.write(f"{wav_res_path}\t{wer}\t{truth}\t{hypo}\t{inse}\t{dele}\t{subs}\n")
            fout.flush()
        else:
            h_list = hypo_normed.split()
            r_list = truth_normed.split()
            word = len(r_list)
            score = editdistance.eval(h_list, r_list)
            wer_per_sen = score / word if word > 0 else float("inf")
            fout.write(f"{wav_res_path}\t{score}\t{word}\t{wer_per_sen}\t{truth_normed}\t{hypo_normed}\n")
            fout.flush()
            


run_asr(wav_res_text_path, res_path, normalize_text=True, faster=False, whole=True)

