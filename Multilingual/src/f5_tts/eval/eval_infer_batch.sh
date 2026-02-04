#!/bin/bash
# bash src/f5_tts/eval/eval_infer_batch.sh
lang=en
asr_gpu=1
task=zero_shot
ckpt=500000
exp_name=F5TTS_v1_Base_multilingual_v4 # M3TTS_Small_multilingual_v1


pip install ctranslate2==4.5.0
export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH


# change update
accelerate launch  src/f5_tts/eval/eval_infer_batch.py  -s 0 -n "${exp_name}" -c ${ckpt} -t "seedtts_test_${lang}" -nfe 16 -l "${lang}" --force_rescan
#accelerate launch src/f5_tts/eval/eval_infer_batch.py -s 0 -n "F5TTS_Small" -c 500000 -t "ls_pc_test_clean" -nfe 16

#change 1.gpu 2.update  3. --language 4. check --local for zh do not use local (I have not downloaded it)
python src/f5_tts/eval/eval_seedtts_testset.py -e wer -l ${lang} --gen_wav_dir results/${exp_name}_${ckpt}/seedtts_test_${lang}/seed0_euler_nfe16_vocos_ss-1_cfg2.0_speed1.0 --gpu_nums ${asr_gpu}
python src/f5_tts/eval/eval_seedtts_testset.py -e sim -l ${lang} --gen_wav_dir results/${exp_name}_${ckpt}/seedtts_test_${lang}/seed0_euler_nfe16_vocos_ss-1_cfg2.0_speed1.0 --gpu_nums ${asr_gpu}
python src/f5_tts/eval/eval_utmos.py --audio_dir results/${exp_name}_${ckpt}/seedtts_test_${lang}/seed0_euler_nfe16_vocos_ss-1_cfg2.0_speed1.0