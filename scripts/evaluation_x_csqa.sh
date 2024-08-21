
deepspeed run_evaluation.py --task x_csqa --llm_path LLaMAX/LLaMAX2-7B-X-CSQA --mt_path google/mt5-xl/ \
       --init_checkpoint outputs/MindMerger/x-csqa/augmentation/pytorch_model.bin --augmentation True
