# bash src/x_voice/eval/eval_multilingual_seedtts.sh
num_gpus=1
task=zero_shot
dataset=seedtts_testset
ckpt=70000
exp_name=XVoice_Base_Stage2
test_set="zh en"
drop_text=True # set True for stage 2 model, False for stage 1 model


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



export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH
for lang in ${test_set}; do
    python src/x_voice/eval/eval_seedtts_testset.py -e wer -l ${lang} --gen_wav_dir ${decode_dir} --gpu_nums ${num_gpus} --local
    python src/x_voice/eval/eval_seedtts_testset.py -e sim -l ${lang} --gen_wav_dir ${decode_dir} --gpu_nums ${num_gpus} --local
    python src/x_voice/eval/eval_utmos.py --audio_dir ${decode_dir}/${lang}_${lang}
done
