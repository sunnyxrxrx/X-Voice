import math
import os
import random
import string
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
import pyphen

import unicodedata
import re

from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL
from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import convert_char_to_pinyin

# from nemo_text_processing.text_normalization.normalize import Normalizer
from f5_tts.eval.text_normalizer import TextNormalizer
import pickle

def get_testset_metainfo(data_dir,in_language):
    """
    data_dir goes to: cv3_eval/zero_shot/[in_language] 
    metainfo: List[Tuple(utt_id, prompt_wav_path, prompt_text, target_text)]
    """
    path_parts = data_dir.split("/")
    data_idx = path_parts.index("zero_shot")
    root_dir = "/".join(path_parts[:data_idx])
    
    prompt_scp = os.path.join(data_dir, "prompt_wav.scp")
    prompt_text_file = os.path.join(data_dir, "prompt_text") 
    target_text_file = os.path.join(data_dir, "text")      

    utt2wav = {}
    with open(prompt_scp, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                utt2wav[parts[0]] = parts[1]
                #print(parts[1])
    
    # load target text
    utt2text = {}
    with open(target_text_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) >= 2:
                utt2text[parts[0]] = parts[1]
                
    # load prompt text
    utt2prompt = {}
    with open(prompt_text_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) >= 2:
                utt2prompt[parts[0]] = parts[1]
    
    metainfo = []
    for utt_id, wav_path in utt2wav.items():
        if utt_id in utt2text:
            target_text = utt2text[utt_id]
            prompt_text = utt2prompt[utt_id] 
            wav_path_clean = wav_path.replace("data/", "", 1)  
            full_wav_path = os.path.join(root_dir,wav_path_clean)
            metainfo.append((utt_id, prompt_text, full_wav_path, target_text))
            
    return metainfo

# seedtts testset metainfo: utt, prompt_text, prompt_wav, gt_text, gt_wav
def get_seedtts_testset_metainfo(metalst):
    f = open(metalst)
    lines = f.readlines()
    f.close()
    metainfo = []
    for line in lines:
        if len(line.strip().split("|")) == 5:
            utt, prompt_text, prompt_wav, gt_text, gt_wav = line.strip().split("|")
        elif len(line.strip().split("|")) == 4:
            utt, prompt_text, prompt_wav, gt_text = line.strip().split("|")
            gt_wav = os.path.join(os.path.dirname(metalst), "wavs", utt + ".wav")
        if not os.path.isabs(prompt_wav):
            prompt_wav = os.path.join(os.path.dirname(metalst), prompt_wav)
        metainfo.append((utt, prompt_text, prompt_wav, gt_text, gt_wav))
    return metainfo


# librispeech test-clean metainfo: gen_utt, ref_txt, ref_wav, gen_txt, gen_wav
def get_librispeech_test_clean_metainfo(metalst, librispeech_test_clean_path):
    f = open(metalst)
    lines = f.readlines()
    f.close()
    metainfo = []
    for line in lines:
        ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt = line.strip().split("\t")

        # ref_txt = ref_txt[0] + ref_txt[1:].lower() + '.'  # if use librispeech test-clean (no-pc)
        ref_spk_id, ref_chaptr_id, _ = ref_utt.split("-")
        ref_wav = os.path.join(librispeech_test_clean_path, ref_spk_id, ref_chaptr_id, ref_utt + ".flac")

        # gen_txt = gen_txt[0] + gen_txt[1:].lower() + '.'  # if use librispeech test-clean (no-pc)
        gen_spk_id, gen_chaptr_id, _ = gen_utt.split("-")
        gen_wav = os.path.join(librispeech_test_clean_path, gen_spk_id, gen_chaptr_id, gen_utt + ".flac")

        metainfo.append((gen_utt, ref_txt, ref_wav, " " + gen_txt, gen_wav))

    return metainfo


# padded to max length mel batch
def padded_mel_batch(ref_mels):
    max_mel_length = torch.LongTensor([mel.shape[-1] for mel in ref_mels]).amax()
    padded_ref_mels = []
    for mel in ref_mels:
        padded_ref_mel = F.pad(mel, (0, max_mel_length - mel.shape[-1]), value=0)
        padded_ref_mels.append(padded_ref_mel)
    padded_ref_mels = torch.stack(padded_ref_mels)
    padded_ref_mels = padded_ref_mels.permute(0, 2, 1)
    return padded_ref_mels

def count(text):
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

    def count_punctuations(text):
        punct_map = {
            '.': 1.5, '!': 1.5, '?': 1.5,
            ',': 0.7, ';': 1.0, ':': 1.0,
            '。': 1.3, '！': 1.3, '？': 1.3,
            '，': 0.6, '；': 0.8, '：': 0.8,
        }
        punct_syllables = 0
        for char in text:
            if char in punct_map:
                punct_syllables += punct_map[char]
        return punct_syllables
    
    return round(count_syllables(text) + count_punctuations(text))
# get prompts from metainfo containing: utt, prompt_text, prompt_wav, gt_text, gt_wav


def get_inference_prompt(
    metainfo,
    speed=1.0,
    tokenizer="pinyin",
    polyphone=True,
    target_sample_rate=24000,
    n_fft=1024,
    win_length=1024,
    n_mel_channels=100,
    hop_length=256,
    mel_spec_type="vocos",
    target_rms=0.1,
    use_truth_duration=False,
    infer_batch_size=1,
    num_buckets=200,
    min_secs=3,
    max_secs=31,
    language=None,
    ipa_tokenizer=None,
    force_rescan=False,
    cache_dir=None,
    normalize_text=False,
    drop_text=False,
    reverse=False,
    model_sp=None,
    device=None,
):
    prompts_all = []

    min_tokens = min_secs * target_sample_rate // hop_length
    max_tokens = max_secs * target_sample_rate // hop_length

    batch_accum = [0] * num_buckets
    utts, ref_rms_list, ref_mels, ref_mel_lens, total_mel_lens, final_text_list = (
        [[] for _ in range(num_buckets)] for _ in range(6)
    )

    mel_spectrogram = MelSpec(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )
    if normalize_text:
        print("Using text normalizer to pre-process the texts.")
        normalizer = TextNormalizer(language=language)
    for item in tqdm(metainfo, desc="Processing prompts..."):
        if len(item)==5:
            utt, prompt_text, prompt_wav, gt_text, gt_wav = item
        elif len(item)==4:
            utt, prompt_text, prompt_wav, gt_text = item
            gt_wav = None
            use_truth_duration=False
        # Audio
        try:
            ref_audio, ref_sr = torchaudio.load(prompt_wav)
            # cut slience part
            # SILENCE_THRESHOLD = 0.01
            # MAX_SILENCE_SEC = 2.0
            # LOW_THRESHOLD = 0.05

            # 
            # if ref_audio.shape[0] > 1:
            #     amplitude = torch.mean(ref_audio.abs(), dim=0)
            # else:
            #     amplitude = ref_audio.abs().squeeze(0)

            # # find all non silent indices
            # non_silent_indices = torch.where(amplitude > SILENCE_THRESHOLD)[0]
            # is_low = torch.where(amplitude > LOW_THRESHOLD)[0]
            # if len(is_low) == 0:
            #     print(f"Skipping silent audio:{prompt_wav}")
            #     continue
            # else:
            #     first_non_silent = non_silent_indices[0].item()
            #     last_non_silent = non_silent_indices[-1].item()
                
            #     # check slience at the beginning
            #     start_silence_duration = first_non_silent / ref_sr
            #     new_start = 0
            #     if start_silence_duration > MAX_SILENCE_SEC:
            #         # 裁剪到声音开始的地方减去0.5秒
            #         new_start = max(0,first_non_silent - int(0.5* ref_sr))
            #         # print(f"Trimmed start silence for {utt}: {start_silence_duration:.2f}s")

            #     # check slience at the end
            #     end_silence_duration = (ref_audio.shape[-1] - last_non_silent) / ref_sr
            #     new_end = ref_audio.shape[-1]
            #     if end_silence_duration > MAX_SILENCE_SEC:
            #         # 0.5s more
            #         new_end = min(last_non_silent + int(0.5* ref_sr), ref_audio.shape[-1])
                    
            #     # apply cutting
            #     if new_start > 0 or new_end < ref_audio.shape[-1]:
            #         ref_audio = ref_audio[:, new_start:new_end]
            #         debug_wav_path = prompt_wav[:-4]+"cut.wav"
            #         # save new audio to test
            #         torchaudio.save(debug_wav_path, ref_audio, ref_sr)
            #         print(f"Cut_radio:{prompt_wav}")
            # ============================================================
            ref_rms = torch.sqrt(torch.mean(torch.square(ref_audio)))
            if ref_rms < target_rms:
                ref_audio = ref_audio * target_rms / ref_rms
            assert ref_audio.shape[-1] > 5000, f"Empty prompt wav: {prompt_wav}, or torchaudio backend issue."
            if ref_sr != target_sample_rate:
                resampler = torchaudio.transforms.Resample(ref_sr, target_sample_rate)
                ref_audio = resampler(ref_audio)

            # Text
            if len(prompt_text[-1].encode("utf-8")) == 1:
                prompt_text = prompt_text + " "
            if normalize_text:
                # print(gt_text)
                prompt_text = normalizer.normalize(prompt_text)
                gt_text = normalizer.normalize(gt_text)
                # print(f"{gt_text}\n")
            if drop_text:
                text = [gt_text]
            elif reverse:
                gt_text = gt_text + " "
                text = [gt_text + prompt_text]
            else:
                text = [prompt_text + gt_text]
            if tokenizer == "pinyin":
                text_list = convert_char_to_pinyin(text, polyphone=polyphone)
            elif tokenizer.startswith("ipa") and language and ipa_tokenizer:
                text_list = [ipa_tokenizer(text)] 
                import random
                if random.random()<0.001:
                    print(f"==========\n{text}\n{text_list}\n============")
            else:
                text_list = text

            # to mel spectrogram
            ref_mel = mel_spectrogram(ref_audio)
            ref_mel = ref_mel.squeeze(0)

            # Duration, mel frame length
            ref_mel_len = ref_mel.shape[-1]

            if use_truth_duration:
                gt_audio, gt_sr = torchaudio.load(gt_wav)
                if gt_sr != target_sample_rate:
                    resampler = torchaudio.transforms.Resample(gt_sr, target_sample_rate)
                    gt_audio = resampler(gt_audio)
                total_mel_len = ref_mel_len + int(gt_audio.shape[-1] / hop_length / speed)

                # # test vocoder resynthesis
                # ref_audio = gt_audio
            else:
                if model_sp is not None:
                    gt_num_unit = count(gt_text)
                    ref_mel_t = ref_mel.unsqueeze(0).permute(0, 2, 1)
                    ref_mel_tensor = ref_mel_t.to(device)
                    ref_mel_len_tensor = torch.tensor([ref_mel_len], dtype=torch.long).to(device)
                    speed = model_sp.predict_speed(
                        audio=ref_mel_tensor,
                        lens=ref_mel_len_tensor,
                    )
                    # pred_duration = gt_num_unit / speed.item()
                    pred_duration = min(max(gt_num_unit / speed.item(), 2), 30)
                    gen_mel_len = int((pred_duration * target_sample_rate) / hop_length)
                    total_mel_len = ref_mel_len + gen_mel_len
                else:
                    ref_text_len = len(prompt_text.encode("utf-8"))
                    gen_text_len = len(gt_text.encode("utf-8"))
                    total_mel_len = ref_mel_len + int(ref_mel_len / ref_text_len * gen_text_len / speed)

            # deal with batch
            assert infer_batch_size > 0, "infer_batch_size should be greater than 0."
            assert min_tokens <= total_mel_len <= max_tokens, (
                f"Audio {utt} has duration {total_mel_len * hop_length // target_sample_rate}s out of range [{min_secs}, {max_secs}]."
            )
            bucket_i = math.floor((total_mel_len - min_tokens) / (max_tokens - min_tokens + 1) * num_buckets)

            utts[bucket_i].append(utt)
            ref_rms_list[bucket_i].append(ref_rms)
            ref_mels[bucket_i].append(ref_mel)
            ref_mel_lens[bucket_i].append(ref_mel_len)
            total_mel_lens[bucket_i].append(total_mel_len)
            final_text_list[bucket_i].extend(text_list)

            batch_accum[bucket_i] += total_mel_len

            if batch_accum[bucket_i] >= infer_batch_size:
                # print(f"\n{len(ref_mels[bucket_i][0][0])}\n{ref_mel_lens[bucket_i]}\n{total_mel_lens[bucket_i]}")
                prompts_all.append(
                    (
                        utts[bucket_i],
                        ref_rms_list[bucket_i],
                        padded_mel_batch(ref_mels[bucket_i]),
                        ref_mel_lens[bucket_i],
                        total_mel_lens[bucket_i],
                        final_text_list[bucket_i],
                    )
                )
                batch_accum[bucket_i] = 0
                (
                    utts[bucket_i],
                    ref_rms_list[bucket_i],
                    ref_mels[bucket_i],
                    ref_mel_lens[bucket_i],
                    total_mel_lens[bucket_i],
                    final_text_list[bucket_i],
                ) = [], [], [], [], [], []
        except Exception as e:
            print(f"[WARN] Failed to load {utt}: {e}")
            continue  

    # add residual
    for bucket_i, bucket_frames in enumerate(batch_accum):
        if bucket_frames > 0:
            prompts_all.append(
                (
                    utts[bucket_i],
                    ref_rms_list[bucket_i],
                    padded_mel_batch(ref_mels[bucket_i]),
                    ref_mel_lens[bucket_i],
                    total_mel_lens[bucket_i],
                    final_text_list[bucket_i],
                )
            )
    # not only leave easy work for last workers
    random.seed(666)
    random.shuffle(prompts_all)
    # with open(cache_file, 'wb') as f:
    #     pickle.dump(prompts_all, f)
    # print(f"Prompts cache saved to {cache_file}")

    return prompts_all


# get wav_res_ref_text of seed-tts test metalst
# https://github.com/BytedanceSpeech/seed-tts-eval


def get_seed_tts_test(metalst, gen_wav_dir, gpus):
    f = open(metalst)
    lines = f.readlines()
    f.close()

    test_set_ = []
    for line in tqdm(lines):
        if len(line.strip().split("|")) == 5:
            utt, prompt_text, prompt_wav, gt_text, gt_wav = line.strip().split("|")
        elif len(line.strip().split("|")) == 4:
            utt, prompt_text, prompt_wav, gt_text = line.strip().split("|")

        if not os.path.exists(os.path.join(gen_wav_dir, utt + ".wav")):
            continue
        gen_wav = os.path.join(gen_wav_dir, utt + ".wav")
        if not os.path.isabs(prompt_wav):
            prompt_wav = os.path.join(os.path.dirname(metalst), prompt_wav)

        test_set_.append((gen_wav, prompt_wav, gt_text))

    num_jobs = len(gpus)
    if num_jobs == 1:
        return [(gpus[0], test_set_)]

    wav_per_job = len(test_set_) // num_jobs + 1
    test_set = []
    for i in range(num_jobs):
        test_set.append((gpus[i], test_set_[i * wav_per_job : (i + 1) * wav_per_job]))

    return test_set


# get librispeech test-clean cross sentence test


def get_librispeech_test(metalst, gen_wav_dir, gpus, librispeech_test_clean_path, eval_ground_truth=False):
    f = open(metalst)
    lines = f.readlines()
    f.close()

    test_set_ = []
    for line in tqdm(lines):
        ref_utt, ref_dur, ref_txt, gen_utt, gen_dur, gen_txt = line.strip().split("\t")

        if eval_ground_truth:
            gen_spk_id, gen_chaptr_id, _ = gen_utt.split("-")
            gen_wav = os.path.join(librispeech_test_clean_path, gen_spk_id, gen_chaptr_id, gen_utt + ".flac")
        else:
            if not os.path.exists(os.path.join(gen_wav_dir, gen_utt + ".wav")):
                raise FileNotFoundError(f"Generated wav not found: {gen_utt}")
            gen_wav = os.path.join(gen_wav_dir, gen_utt + ".wav")

        ref_spk_id, ref_chaptr_id, _ = ref_utt.split("-")
        ref_wav = os.path.join(librispeech_test_clean_path, ref_spk_id, ref_chaptr_id, ref_utt + ".flac")

        test_set_.append((gen_wav, ref_wav, gen_txt))

    num_jobs = len(gpus)
    if num_jobs == 1:
        return [(gpus[0], test_set_)]

    wav_per_job = len(test_set_) // num_jobs + 1
    test_set = []
    for i in range(num_jobs):
        test_set.append((gpus[i], test_set_[i * wav_per_job : (i + 1) * wav_per_job]))

    return test_set


# load asr model


def load_asr_model(lang, ckpt_dir=""):
    if lang == "zh":
        from funasr import AutoModel

        model = AutoModel(
            model=os.path.join(ckpt_dir, "paraformer-zh"),
            # vad_model = os.path.join(ckpt_dir, "fsmn-vad"),
            # punc_model = os.path.join(ckpt_dir, "ct-punc"),
            # spk_model = os.path.join(ckpt_dir, "cam++"),
            disable_update=True,
        )  # following seed-tts setting
    elif lang == "en":
        from faster_whisper import WhisperModel

        model_size = "large-v3" if ckpt_dir == "" else ckpt_dir
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
    return model


# WER Evaluation, the way Seed-TTS does


def run_asr_wer(args):
    rank, lang, test_set, ckpt_dir = args

    if lang == "zh":
        import zhconv

        torch.cuda.set_device(rank)
    elif lang == "en":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    else:
        raise NotImplementedError(
            "lang support only 'zh' (funasr paraformer-zh), 'en' (faster-whisper-large-v3), for now."
        )

    asr_model = load_asr_model(lang, ckpt_dir=ckpt_dir)

    from zhon.hanzi import punctuation

    punctuation_all = punctuation + string.punctuation
    wer_results = []

    from jiwer import compute_measures

    for gen_wav, prompt_wav, truth in tqdm(test_set):
        if lang == "zh":
            res = asr_model.generate(input=gen_wav, batch_size_s=300, disable_pbar=True)
            hypo = res[0]["text"]
            hypo = zhconv.convert(hypo, "zh-cn")
        elif lang == "en":
            segments, _ = asr_model.transcribe(gen_wav, beam_size=5, language="en")
            hypo = ""
            for segment in segments:
                hypo = hypo + " " + segment.text

        raw_truth = truth
        raw_hypo = hypo

        for x in punctuation_all:
            truth = truth.replace(x, "")
            hypo = hypo.replace(x, "")

        truth = truth.replace("  ", " ")
        hypo = hypo.replace("  ", " ")

        if lang == "zh":
            truth = " ".join([x for x in truth])
            hypo = " ".join([x for x in hypo])
        elif lang == "en":
            truth = truth.lower()
            hypo = hypo.lower()

        measures = compute_measures(truth, hypo)
        wer = measures["wer"]

        # ref_list = truth.split(" ")
        # subs = measures["substitutions"] / len(ref_list)
        # dele = measures["deletions"] / len(ref_list)
        # inse = measures["insertions"] / len(ref_list)

        wer_results.append(
            {
                "wav": Path(gen_wav).stem,
                "truth": raw_truth,
                "hypo": raw_hypo,
                "wer": wer,
            }
        )

    return wer_results


# SIM Evaluation


def run_sim(args):
    rank, test_set, ckpt_dir = args
    device = f"cuda:{rank}"

    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
    state_dict = torch.load(ckpt_dir, weights_only=True, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict["model"], strict=False)

    use_gpu = True if torch.cuda.is_available() else False
    if use_gpu:
        model = model.cuda(device)
    model.eval()

    sim_results = []
    for gen_wav, prompt_wav, truth in tqdm(test_set):
        wav1, sr1 = torchaudio.load(gen_wav)
        wav2, sr2 = torchaudio.load(prompt_wav)

        resample1 = torchaudio.transforms.Resample(orig_freq=sr1, new_freq=16000)
        resample2 = torchaudio.transforms.Resample(orig_freq=sr2, new_freq=16000)
        wav1 = resample1(wav1)
        wav2 = resample2(wav2)

        if use_gpu:
            wav1 = wav1.cuda(device)
            wav2 = wav2.cuda(device)
        with torch.no_grad():
            emb1 = model(wav1)
            emb2 = model(wav2)

        sim = F.cosine_similarity(emb1, emb2)[0].item()
        # print(f"VSim score between two audios: {sim:.4f} (-1.0, 1.0).")
        sim_results.append(
            {
                "wav": Path(gen_wav).stem,
                "sim": sim,
            }
        )

    return sim_results
