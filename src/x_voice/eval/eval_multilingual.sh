# bash src/x_voice/eval/eval_multilingual.sh
num_gpus=1
task=zero_shot
dataset=x_voice_eval # lemas_eval
exp_name=XVoice_Base_Stage2
ckpt=70000
drop_text=True # set True for stage 2 model, False for stage 1 model
test_set="it zh ru ko en ko ru zh it en"
ref_set="en it zh ru ko en ko ru zh it"

seed=0
nfe=16
speed=1.0
sp_type=pretrained # syllable, pretrained, if choose "pretrained", "srp_exp_name" and "srp_ckpt" must be provided 
srp_exp_name=SpeedPredict_Multilingual
srp_ckpt=28000

cfg_schedule=square
cfg_decay_time=0.6
cfg_strength=2.5
decoupled=True
cfg_strength2=4.0 # if decouple is set False, "cfg_strength2" will be ignored
reverse=False


if [ -z "${ref_set}" ]; then
    ref_set="${test_set}"
fi
test_langs=($test_set)
ref_langs=($ref_set)
if [ ${#test_langs[@]} -ne ${#ref_langs[@]} ]; then
    echo "[ERROR] test_set and ref_set must have the same number of languages for cross-lingual eval."
    exit 1
fi
drop_text_args=""
if [ "$drop_text" = "True" ]; then
    drop_text_args="--drop_text"
fi
layered_args=""
if [ "$decoupled" = "True" ]; then
    layered_args="--layered"
fi
reverse_args=""
if [ "$reverse" = "True" ]; then
    reverse_args="--reverse"
fi
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" # to eval/
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../../" && pwd )" # to X-Voice/
cv3_dir="${PROJECT_ROOT}/data/${dataset}"

if [ "${decoupled}" = "True" ]; then
    decode_dir="${PROJECT_ROOT}/results/${exp_name}_${ckpt}/${dataset}/${sp_type}_seed${seed}_euler_nfe${nfe}_vocos_ss-1_schedule${cfg_schedule}_cfgI${cfg_strength}_cfgII${cfg_strength2}_speed${speed}zero_shot"
else
    decode_dir="${PROJECT_ROOT}/results/${exp_name}_${ckpt}/${dataset}/${sp_type}_seed${seed}_euler_nfe${nfe}_vocos_ss-1_schedule${cfg_schedule}_cfg${cfg_strength}_speed${speed}zero_shot"
fi

accelerate launch --main_process_port 29507 --num_processes ${num_gpus} src/x_voice/eval/eval_infer_batch.py \
    -s ${seed} -n "${exp_name}" -c ${ckpt} -t "${dataset}" -nfe ${nfe} --speed ${speed} \
    -l "${test_set// /,}" -rl "${ref_set// /,}" \
    --cfg_strength ${cfg_strength} \
    ${layered_args} --cfg_strength2 ${cfg_strength2} \
    --cfg_schedule "${cfg_schedule}" --cfg_decay_time ${cfg_decay_time} \
    --normalize_text --post_processing \
    --decode_dir "${decode_dir}" \
    --sp_type ${sp_type} -ns "${srp_exp_name}" -cs ${srp_ckpt} \
    ${drop_text_args} ${reverse_args}




# Evaluation
apt-get install -y sox libsox-dev 
cd "$SCRIPT_DIR"
pip install -r requirements.txt
export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH


DNSMOS_LAB=utils/DNSMOS
. utils/parse_options.sh || exit 1;
dumpdir=${cv3_dir}/${task}
test_gt="false"

for ((i=0; i<${#test_langs[@]}; i++)); do
    lang=${test_langs[$i]}
    ref_lang=${ref_langs[$i]}
    # WER
    echo "[INFO] Scoring WER for ${decode_dir}/${lang}"
    bash utils/cal_wer.sh ${dumpdir}/${lang}/text  ${decode_dir}/${ref_lang}_${lang} ${lang} ${num_gpus} "${test_gt}"
    find ${decode_dir}/${ref_lang}_${lang}/wavs -name *.wav | awk -F '/' '{print $NF, $0}' | sed "s@\.wav @ @g" > ${decode_dir}/${ref_lang}_${lang}/wav.scp
    
    # SIM
    echo "[INFO] Scoring SIM: Comparing ${lang} output with ${ref_lang} prompt" 
    python eval_similarity.py \
    --wavlm_ckpt_dir ${PROJECT_ROOT}/wavlm_large_finetune.pth \
    --prompt_wavs ${dumpdir}/${ref_lang}/prompt_wav.scp \
    --hyp_wavs ${decode_dir}/${ref_lang}_${lang}/wav.scp \
    --log_file ${decode_dir}/${ref_lang}_${lang}/spk_simi_scores.txt \
    --num_gpus ${num_gpus} \
    --decode_dir ${decode_dir}\
    --dump_dir ${cv3_dir}

    # UTMOS
    echo "[INFO] Scoring UTSMOS  for ${decode_dir}/${ref_lang}_${lang}"  
    python eval_utmos.py --audio_dir ${decode_dir}/${ref_lang}_${lang} --ext "wav"

    # # DNSMOS
    # echo "[INFO] Scoring DNSMOS  for ${decode_dir}/${ref_lang}_${lang}"  
    # python ${DNSMOS_LAB}/dnsmos_local_wavscp.py -t ${decode_dir}/${ref_lang}_${lang}/wav.scp -e ${DNSMOS_LAB} -o ${decode_dir}/${ref_lang}_${lang}/dnsmos.csv
    # cat ${decode_dir}/${ref_lang}_${lang}/mos.csv | sed '1d' |awk -F ',' '{ sum += $NF; count++ } END { if (count > 0) print sum / count }'  > ${decode_dir}/${ref_lang}_${lang}/dnsmos_mean.txt
done
echo "[INFO] Collecting final result"
python collect_results.py --decode_dir "${decode_dir}" --test_set "${test_set}" --ref_set "${ref_set}"
