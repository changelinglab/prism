#!/usr/bin/env bash
# Copyright 2025  Carnegie Mellon University (Author: Shikhar Bharadwaj)
set -euo pipefail
log() {
    echo "$(date '+%Y-%m-%dT%H:%M:%S') [${BASH_SOURCE[1]##*/}:${BASH_LINENO[0]}] $*"
}

model=""
recipe=""
data=""
cluster="babel"
fft=false
dry_run=false
run_name=""
sbatch_args=""
extra_args=""
wait_time=1s
nseeds=1
seedlist=""

help_message=$(cat << 'EOF'
Usage: $0 [OPTIONS]

Options:
  --model LIST        Models: powsm, powsm_ctc, ctag, lv60, xlsr53, zipactc, zipactc_ns, or "all"
  --recipe LIST       Recipes: geo_in, l1cls_cmu, l1cls_ed, l2as, lid_fl, atyp_ec, atyp_ua, atyp_us, inference, predict, cascade_rnn_cls
  --data LIST         Datasets: timit, geo_in, edacc, cmul2arctic, speechocean, fleurs, or "all"
  --cluster NAME      Cluster: babel (default: babel)
  --fft               Enable full fine-tuning
  --dry_run           Print commands only (explicitly set to --dry_run true)
  --run_name STR      Run name (default: timestamp)
  --sbatch_args STR   Extra sbatch arguments
  --extra_args STR    Extra training arguments passed at the end to override all config values
  --nseeds INT        Number of seeds for training (default: 1)
  --seedlist STR      Comma-separated list of seeds (overrides --nseeds)
  --wait_time STR    Wait time between job submissions (default: 2s)

Examples:
  $0 --model powsm,ctag --recipe lif --fft
  $0 --model all --recipe lid_fl --cluster babel
EOF
)
. scripts/parse_options.sh 2>/dev/null || true

setup="probing"
if [ "$recipe" = "inference" ]; then
    setup="inference"
elif [ "$recipe" = "predict" ]; then
    setup="predict"
elif [[ "$recipe" == *cascade* ]]; then
    setup="cascade"
fi
if [ "$setup" != "probing" ] && [ -z "$data" ]; then
    log "Error: --data must be provided for setup=$setup"
    echo "$help_message"
    exit 1
fi

[ -z "$run_name" ] && run_name=$(date "+%Y%m%d%H%M%S")
exp_dir="$(pwd)/exp/runs"
mkdir -p "$exp_dir"
summary_log="${exp_dir}/${run_name}.summary.log"

# Cluster configurations
declare -A cluster_configs=(
    ["babel"]="scripts/babel.batch"
    ["vllm"]="scripts/vllm_dai.batch"
)

# Model configurations: base_model|hf_repo
declare -A model_configs=(
    ["powsm"]="powsm|"
    ["powsm_ctc"]="powsm_ctc|"
    ["owsm_v3"]="powsm|espnet/owsm_v3"
    ["ctag"]="w2v2ph|ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns"
    ["lv60"]="w2v2ph|facebook/wav2vec2-lv-60-espeak-cv-ft"
    ["xlsr53"]="w2v2ph|facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    ["zipactc"]="zipactc|anyspeech/zipa-large-crctc-500k"
    ["zipactc_ns"]="zipactc|anyspeech/zipa-large-crctc-ns-800k"
    ["gemini"]="gemini|"
    ["wavlm"]="wavlm|microsoft/wavlm-base"
    ["whisper"]="whisper|openai/whisper-small"
    ["qweni"]="qweninstruct"
    ["qwent"]="qwenthinking"
)

# Recipe = task_dataset
declare -A recipe_configs=(
    ["geo_in"]="geolocation_vaani"
    ["l1cls_cmu"]="l1cls_cmul2arctic"
    ["l1cls_ed"]="l1cls_edacc"
    ["l2as"]="l2as_speechocean"
    ["lid_fl"]="lid_fleurs"
    ["atyp_ec"]="atypical_easycall"
    ["atyp_ua"]="atypical_uaspeech"
    ["atyp_us"]="atypical_ultrasuite"
    ["inference"]="transcribe"
    ["predict"]="predict"
    ["cascade_rnn_cls"]="rnn_classification"
    ["cascade_rnn_reg"]="rnn_regression"
    ["cascade_rnn_geo"]="rnn_geolocation"
)

# Dataset -> recipe code for predict (only prompt-capable datasets; reuse recipe_configs for prompt name)
declare -A data_to_recipe=(
    ["cmul2arctic"]="l1cls_cmu"
    ["edacc"]="l1cls_ed"
    ["speechocean"]="l2as"
    ["fleurs"]="lid_fl"
    ["geo_in"]="geo_in"
    ["easycall"]="atyp_ec"
    ["uaspeech"]="atyp_ua"
    ["ultrasuite"]="atyp_us"
)

# Dataset configurations: dataset|num_classes (num_classes is empty if not applicable)
declare -A dataset_configs=(
    ["timit"]="timit|"
    ["doreco"]="doreco|"
    ["geo_in"]="vaanigeo|1"
    ["cmul2arctic"]="cmul2arcticl1|7"
    ["edacc"]="edacc|13"
    ["speechocean"]="speechocean|11"
    ["fleurs"]="fleurs|24"
    ["easycall"]="easycall|4"
    ["uaspeech"]="uaspeech|5"
    ["ultrasuite"]="ultrasuite_child|2"
)

get_base_model() {
    IFS='|' read -r base _ <<< "${model_configs[$1]}"
    echo "$base"
}

get_hf_repo() {
    IFS='|' read -r _ repo <<< "${model_configs[$1]}"
    echo "$repo"
}

config_exists() {
    local config_file=$1
    [ -f "$config_file" ]
}

generate_list() {
    local input=$1
    local -n config_ref=$2
    local items=()
    if [ "$input" = "all" ]; then
        items=("${!config_ref[@]}")
    else
        IFS=',' read -ra requested <<< "$input"
        for item in "${requested[@]}"; do
            [[ -v config_ref[$item] ]] && items+=("$item")
        done
    fi
    echo "${items[@]}"
}

construct_config_name() {
    local model_var=$1 recipe_code=$2
    local base=$(get_base_model "$model_var")
    if [ "$setup" = "predict" ]; then
        echo "configs/experiment/inference/predict_${base}.yaml"
        return
    fi
    local recipe_full="${recipe_configs[$recipe_code]}"
    if [ "$setup" = "cascade" ]; then
        echo "configs/experiment/${setup}/${recipe_full}.yaml"
    else
        echo "configs/experiment/${setup}/${recipe_full}_${base}.yaml"
    fi
}
construct_cmd_for_probing() {
    local model_var=$1 config_file=$2
    local repo=$(get_hf_repo "$model_var")
    local script="${cluster_configs[$cluster]}"
    local cmd="sbatch${sbatch_args:+ $sbatch_args} $script experiment=${setup}/${config_file##*/}"
    if [ -n "$repo" ]; then
        cmd+=" model.net.hf_repo=$repo"
    fi
    if $fft; then
        cmd+=" model.freeze_encoder=false tags+=[\\\"fft\\\"]"
    fi
    [ -n "$extra_args" ] && cmd+=" $extra_args"
    local cmds=()
    # Generate commands with multiple seeds
    for seed in "${seed_array[@]}"; do
        cmds+=("${cmd} seed=$seed")
    done
    printf '%s\n' "${cmds[@]}"
}

construct_cmd_for_inference() {
    local model_var=$1 config_file=$2
    shift 2
    local datasets=("$@")
    local repo=$(get_hf_repo "$model_var")
    local script="${cluster_configs[$cluster]}"
    local cmds=()
    for dataset_code in "${datasets[@]}"; do
        [ -z "$dataset_code" ] && continue
        local dataset_name="${dataset_configs[$dataset_code]%%|*}"
        local task_name="inf_${dataset_name}_${model_var}"
        local cmd="sbatch${sbatch_args:+ $sbatch_args} $script experiment=${setup}/${config_file##*/} data=$dataset_name task_name=$task_name"
        if [ -n "$repo" ]; then
            cmd+=" inference.inference_runner.hf_repo=$repo"
        fi
        [ -n "$extra_args" ] && cmd+=" $extra_args"
        cmds+=("$cmd")
    done
    printf '%s\n' "${cmds[@]}"
}

construct_cmd_for_predict() {
    local model_var=$1 config_file=$2
    shift 2
    local datasets=("$@")
    local script="${cluster_configs[$cluster]}"
    local config_basename="${config_file##*/}"
    local model_short="${config_basename%.yaml}"
    model_short="${model_short#predict_}"
    local cmds=()
    for dataset_code in "${datasets[@]}"; do
        [ -z "$dataset_code" ] && continue
        [[ ! -v data_to_recipe[$dataset_code] ]] && continue
        local recipe_code="${data_to_recipe[$dataset_code]}"
        local prompt_name="${recipe_configs[$recipe_code]}"
        local dataset_name="${dataset_configs[$dataset_code]%%|*}"
        local task_name="dp_${model_short}_${prompt_name}"
        local cmd="sbatch${sbatch_args:+ $sbatch_args} $script experiment=inference/${config_basename} data=$dataset_name prompt=$prompt_name task_name=$task_name"
        [ -n "$extra_args" ] && cmd+=" $extra_args"
        cmds+=("$cmd")
    done
    printf '%s\n' "${cmds[@]}"
}

_construct_adhoc_args_for_cascade() {
    local dataset_name=$1 model_var=$2
    # Pick the latest non-empty transcription.json under the run tree
    prefix="./"
    local runs_dir="${prefix}exp/runs/inf_${dataset_name}_${model_var}"
    local transcription_json
    transcription_json="$(
        find "$runs_dir" -type f -name 'transcription.json' -size +0c \
          -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
    )"
    echo "Found $transcription_json" >&2
    [ -z "$transcription_json" ] && return 1
    # Construct vocabulary size = number of unique symbols in transcription.json
    local vocab_size
    vocab_size="$(
        python3 - <<PY
import re
s = open("$transcription_json", encoding="utf-8", errors="ignore").read()
print(len(set(s)))
PY
    )"
    printf 'data.json_path=%s model.net.vocab_size=%s' "$transcription_json" "$vocab_size"
}

construct_cmd_for_cascade() {
    local model_var=$1 config_file=$2
    shift 2
    local datasets=("$@")
    local repo=$(get_hf_repo "$model_var")
    local script="${cluster_configs[$cluster]}"
    local cmds=()
    for dataset_code in "${datasets[@]}"; do
        [ -z "$dataset_code" ] && continue
        local dataset_conf="${dataset_configs[$dataset_code]}"
        local dataset_name="${dataset_conf%%|*}"
        local adhoc_args="$(_construct_adhoc_args_for_cascade "$dataset_name" "$model_var")" || continue
        local task_name="cascade.${dataset_code}_${model_var}"
        local num_classes="${dataset_conf#*|}"
        local cmd="sbatch${sbatch_args:+ $sbatch_args} $script experiment=${setup}/${config_file##*/} $adhoc_args"
        cmd+=" task_name=$task_name"
        cmd+=" tags=[\\\"$setup\\\",\\\"$dataset_code\\\",\\\"$model_var\\\"]"
        [ -n "$num_classes" ] && cmd+=" data.num_classes=$num_classes"
        [ -n "$extra_args" ] && cmd+=" $extra_args"
        # Generate commands with multiple seeds
        for seed in "${seed_array[@]}"; do
            cmds+=("${cmd} seed=$seed")
        done
    done
    printf '%s\n' "${cmds[@]}"
}

run_experiment() {
    local model_var=$1 recipe_code=$2
    shift 2
    local datasets=("$@")
    local config_file=$(construct_config_name "$model_var" "$recipe_code")
    if ! config_exists "$config_file"; then
        log "Skip: $model_var on $recipe_code (config not found: $config_file)"
        echo "SKIP: $model_var on $recipe_code" >> "$summary_log"
        return 2
    fi
    local cmd_list
    case "$setup" in
        inference)
            cmd_list=$(construct_cmd_for_inference "$model_var" "$config_file" "${datasets[@]}") || {
                log "Error constructing commands for $model_var on $recipe_code (config: $config_file)"
                echo "ERROR: $model_var on $recipe_code" >> "$summary_log"
                return 1
            }
            ;;
        predict)
            cmd_list=$(construct_cmd_for_predict "$model_var" "$config_file" "${datasets[@]}") || {
                log "Error constructing commands for $model_var on $recipe_code (config: $config_file)"
                echo "ERROR: $model_var on $recipe_code" >> "$summary_log"
                return 1
            }
            ;;
        cascade)
            cmd_list=$(construct_cmd_for_cascade "$model_var" "$config_file" "${datasets[@]}") || {
                log "Error constructing commands for $model_var on $recipe_code (config: $config_file)"
                echo "ERROR: $model_var on $recipe_code" >> "$summary_log"
                return 1
            }
            ;;
        *)
            cmd_list=$(construct_cmd_for_probing "$model_var" "$config_file") || {
                log "Error constructing command for $model_var on $recipe_code (config: $config_file)"
                echo "ERROR: $model_var on $recipe_code" >> "$summary_log"
                return 1
            }
            ;;
    esac
    mapfile -t cmds <<< "$cmd_list"
    log "Run: $model_var on $recipe_code"
    echo "RUN: $model_var on $recipe_code" >> "$summary_log"
    for cmd in "${cmds[@]}"; do
        [ -z "$cmd" ] && continue
        log "CMD: $cmd"
        echo "CMD: $cmd" >> "$summary_log"
        [[ "$dry_run" = true ]] && continue
        if eval "$cmd"; then
            echo "SUCCESS: $model_var on $recipe_code" >> "$summary_log"
        else
            log "Failed: $model_var on $recipe_code"
            echo "FAILED: $model_var on $recipe_code" >> "$summary_log"
            return 1
        fi
        sleep "$wait_time"
    done
    return 0
}

{
    echo "=== Run: $run_name ==="
    echo "Started: $(date)"
    echo "Setup: $setup | Cluster: $cluster | FFT: $fft"
    echo ""
} > "$summary_log"

models=$(generate_list "$model" model_configs)
recipes=$(generate_list "$recipe" recipe_configs)
read -ra datasets <<< "$(generate_list "$data" dataset_configs)"
# generate seeds
# if seedlist is provided use it to generate seedlist 
# else generate using a sequence from 1 to nseeds
if [ -n "$seedlist" ]; then
    IFS=',' read -ra seed_array <<< "$seedlist"
    nseeds=${#seed_array[@]}
else
    seed_array=()
    for seed in $(seq 1 "$nseeds"); do
        seed_array+=("$seed")
    done
fi

log "Models: $models"
log "Recipes: $recipes"
log "Datasets: ${datasets[*]}"

total=0 successful=0 failed=0 skipped=0

for m in $models; do
    for r in $recipes; do
        total=$((total + 1))
        set +e
        run_experiment "$m" "$r" "${datasets[@]}"
        rc=$?
        set -e
        case $rc in
            0) successful=$((successful + 1)) ;;
            2) skipped=$((skipped + 1)) ;;
            *) failed=$((failed + 1)) ;;
        esac
    done
done
{
    echo ""
    echo "=== Summary ==="
    echo "Finished: $(date)"
    echo "Total: $total | Success: $successful | Failed: $failed | Skipped: $skipped"
} >> "$summary_log"
log "$successful/$total jobs successfully submitted (skipped: $skipped, failed: $failed)"
log "Log: $summary_log"