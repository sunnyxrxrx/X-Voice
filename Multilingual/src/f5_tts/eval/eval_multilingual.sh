# 修改这里的配置
# bash src/f5_tts/eval/eval_multilingual.sh
asr_gpu=2
task=zero_shot
dataset=lemas_eval # lemas_eval, mixed_eval, lemas_eval_new
ckpt=600000
exp_name=F5TTS_v1_Base_multilingual_fullv4 # M3TTS_Small_multilingual_v1
test_set="en" # da el es et fi fr hr hu id it lt mt nl pl pt sk sl sv th de" #cs da el es et fi fr ko" 


seed=0
nfe=16
cfg_schedule=linear
cfg_decay_time=0.6
cfg_strength=2.0
layered=False
cfg_strength2=2.0


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" # to eval/
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../../" && pwd )" # to F5-TTS/
cv3_dir="${PROJECT_ROOT}/data/${dataset}"

if [ "$layered" = "True" ]; then
    decode_dir="${PROJECT_ROOT}/results/${exp_name}_${ckpt}/${dataset}/seed${seed}_concat${concat_method}_schedule${cfg_schedule}_euler_nfe${nfe}_vocos_ss-1_cfgI${cfg_strength}_cfgII${cfg_strength2}_speed1.0zero_shot"
    accelerate launch --main_process_port 29507 src/f5_tts/eval/eval_infer_batch.py \
        -s ${seed} -n "${exp_name}" -c ${ckpt} -t "${dataset}" -nfe ${nfe} -l "${test_set// /,}" \
        --cfg_strength ${cfg_strength} --layered --cfg_strength2 ${cfg_strength2} --cfg_schedule "${cfg_schedule}" --cfg_decay_time ${cfg_decay_time} \
        --normalize_text --sp_type "syllable" \
        --decode_dir "${decode_dir}" #-ns "SpeedPredict_Base" -cs 20000 --reverse 
else
    decode_dir="${PROJECT_ROOT}/results/${exp_name}_${ckpt}/${dataset}/seed${seed}_concat${concat_method}_schedule${cfg_schedule}_euler_nfe${nfe}_vocos_ss-1_cfg${cfg_strength}_speed1.0zero_shot"
    # Inference
    accelerate launch --main_process_port 29507 src/f5_tts/eval/eval_infer_batch.py \
        -s ${seed} -n "${exp_name}" -c ${ckpt} -t "${dataset}" -nfe ${nfe} -l "${test_set// /,}" \
        --cfg_strength ${cfg_strength} --cfg_schedule "${cfg_schedule}" --cfg_decay_time ${cfg_decay_time} \
        --normalize_text --sp_type "syllable" \
        --decode_dir "${decode_dir}" # -ns "SpeedPredict_Base" -cs 20000 --reverse  
fi


# Evaluation
apt-get install -y sox libsox-dev 
cd "$SCRIPT_DIR"
pip install -r requirements.txt
export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH


DNSMOS_LAB=utils/DNSMOS
. utils/parse_options.sh || exit 1;
dumpdir=${cv3_dir}/${task}
test_gt="false"
for lang in ${test_set}; do
    # WER
    echo "[INFO] Scoring WER for ${decode_dir}/${lang}"
    bash utils/cal_wer.sh ${dumpdir}/${lang}/text  ${decode_dir}/${lang} ${lang} ${asr_gpu} "${test_gt}"
    find ${decode_dir}/${lang}/wavs -name *.wav | awk -F '/' '{print $NF, $0}' | sed "s@\.wav @ @g" > ${decode_dir}/${lang}/wav.scp
    
    # SIM
    echo "[INFO] Scoring SIM for inferences in ${decode_dir}/${lang}" 
    python eval_similarity.py \
    --ckpt_path ${PROJECT_ROOT}/wavlm_large_finetune.pth \
    --prompt_wavs ${dumpdir}/${lang}/prompt_wav.scp \
    --hyp_wavs ${decode_dir}/${lang}/wav.scp \
    --log_file ${decode_dir}/${lang}/spk_simi_scores.txt \
    --devices "0" \
    --decode_dir ${decode_dir}\
    --dump_dir ${cv3_dir}

    # UTMOS
    echo "[INFO] Scoring UTSMOS  for ${decode_dir}/${lang}"  
    python eval_utmos.py --audio_dir ${decode_dir}/${lang} --ext "wav"

    # # DNSMOS
    # echo "[INFO] Scoring DNSMOS  for ${decode_dir}/${lang}"  
    # python ${DNSMOS_LAB}/dnsmos_local_wavscp.py -t ${decode_dir}/${lang}/wav.scp -e ${DNSMOS_LAB} -o ${decode_dir}/${lang}/mos.csv
    # cat ${decode_dir}/${lang}/mos.csv | sed '1d' |awk -F ',' '{ sum += $NF; count++ } END { if (count > 0) print sum / count }'  > ${decode_dir}/${lang}/dnsmos_mean.txt
done
