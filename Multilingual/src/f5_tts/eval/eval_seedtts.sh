# 修改这里的配置
# bash src/f5_tts/eval/eval_seedtts.sh
asr_gpu=8
task=zero_shot
dataset=seedtts_test_zh # lemas_eval, mixseeded_eval, lemas_eval_new
ckpt=100000
exp_name=F5TTS_v1_Base_multilingual_full_catada_sft # M3TTS_Small_multilingual_v1
test_set="zh"


seed=0
nfe=16
sp_type=syllable
cfg_schedule=linear
cfg_decay_time=0.6
cfg_strength=2.5
layered=True
cfg_strength2=4.0


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" # to eval/
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../../" && pwd )" # to F5-TTS/
cv3_dir="${PROJECT_ROOT}/data/${dataset}"

if [ "$layered" = "True" ]; then
    decode_dir="${PROJECT_ROOT}/results/${exp_name}_${ckpt}/${dataset}/${sp_type}_seed${seed}_concat${concat_method}_schedule${cfg_schedule}_euler_nfe${nfe}_vocos_ss-1_cfgI${cfg_strength}_cfgII${cfg_strength2}_speed1.0zero_shot"
    accelerate launch --main_process_port 29508 src/f5_tts/eval/eval_infer_batch.py \
        -s ${seed} -n "${exp_name}" -c ${ckpt} -t "${dataset}" -nfe ${nfe} -l "${test_set// /,}" \
        --cfg_strength ${cfg_strength} --layered --cfg_strength2 ${cfg_strength2} --cfg_schedule "${cfg_schedule}" --cfg_decay_time ${cfg_decay_time} \
        --normalize_text --sp_type ${sp_type} \
        --decode_dir "${decode_dir}" # -ns "SpeedPredict_Base" -cs 20000  #--reverse 
else
    decode_dir="${PROJECT_ROOT}/results/${exp_name}_${ckpt}/${dataset}/${sp_type}_seed${seed}_concat${concat_method}_schedule${cfg_schedule}_euler_nfe${nfe}_vocos_ss-1_cfg${cfg_strength}_speed1.0zero_shot"
    decode_dir="${PROJECT_ROOT}/results/${exp_name}_${ckpt}/${dataset}/seed${seed}_concat${concat_method}_schedule${cfg_schedule}_euler_nfe${nfe}_vocos_ss-1_cfg${cfg_strength}_speed1.0zero_shot"
    # Inference
    accelerate launch --main_process_port 29508 src/f5_tts/eval/eval_infer_batch.py \
        -s ${seed} -n "${exp_name}" -c ${ckpt} -t "${dataset}" -nfe ${nfe} -l "${test_set// /,}" \
        --cfg_strength ${cfg_strength} --cfg_schedule "${cfg_schedule}" --cfg_decay_time ${cfg_decay_time} \
        --normalize_text --sp_type ${sp_type} \
        --decode_dir "${decode_dir}" #--reverse # -ns "SpeedPredict_Base" -cs 20000 
fi


export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH
python src/f5_tts/eval/eval_seedtts_testset.py -e wer -l zh --gen_wav_dir ${decode_dir} --gpu_nums ${asr_gpu} --local
python src/f5_tts/eval/eval_seedtts_testset.py -e sim -l zh --gen_wav_dir ${decode_dir} --gpu_nums ${asr_gpu} --local
python src/f5_tts/eval/eval_utmos.py --audio_dir ${decode_dir}

