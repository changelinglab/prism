scripts/run.sh --recipe predict \
	--cluster vllm \
	--model qweni \
	--data ultrasuite \
	--extra_args "inference.num_workers=40 \
	run_folder=5shot \
	inference.inference_runner.prompting.mode=few_shot \
	+prompt.prompt_config.few_shot.policy=k_sample \
	+prompt.prompt_config.few_shot.k=5"

# cmul2arctic,edacc,easycall,uaspeech,ultrasuite,fleurs,speechocean,geo_in

scripts/run.sh --recipe predict \
	--cluster vllm \
	--model qweni \
	--data ultrasuite \
	--extra_args "inference.num_workers=40 run_folder=fewshot inference.inference_runner.prompting.mode=few_shot"

python -m src.metrics.zeroshot_eval \
    --dataset easycall \
    --predictions "exp/runs/dp_qweninstruct_atypical_easycall/5shot/prediction.0.jsonl"

python -m src.metrics.zeroshot_eval \
  --dataset uaspeech \
  --predictions "exp/runs/dp_qweninstruct_atypical_uaspeech/5shot/prediction.0.jsonl"

python -m src.metrics.zeroshot_eval \
  --dataset ultrasuite_child \
  --predictions "exp/runs/dp_qweninstruct_atypical_ultrasuite/fewshot/prediction.0.jsonl"

python -m src.metrics.zeroshot_eval \
  --dataset vaanigeo \
  --predictions "exp/runs/dp_qweninstruct_geolocation_vaani/5shot/prediction.0.jsonl"

python -m src.metrics.zeroshot_eval \
  --dataset cmul2arcticl1 \
  --predictions "exp/runs/dp_qweninstruct_l1cls_cmul2arctic/5shot/prediction.0.jsonl"

python -m src.metrics.zeroshot_eval \
  --dataset edacc \
  --predictions "exp/runs/dp_qweninstruct_l1cls_edacc/5shot/prediction.0.jsonl"

python -m src.metrics.zeroshot_eval \
  --dataset speechocean \
  --predictions "exp/runs/dp_qweninstruct_l2as_speechocean/5shot/prediction.0.jsonl"

python -m src.metrics.zeroshot_eval \
  --dataset fleurs \
  --predictions "exp/runs/dp_qweninstruct_lid_fleurs/5shot/prediction.0.jsonl"