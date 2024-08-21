
deepspeed run_evaluation.py --task xnli --llm_path LLaMAX/LLaMAX2-7B-XNLI --mt_path google/mt5-xl/ \
       --init_checkpoint outputs/MindMerger/xnli/augmentation/pytorch_model.bin --augmentation True
