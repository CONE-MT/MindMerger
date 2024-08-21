
deepspeed run_evaluation.py --task math_mgsm --llm_path meta-math/MetaMath-7B-V1.0 --mt_path google/mt5-xl/ \
       --init_checkpoint outputs/MindMerger/math/augmentation/pytorch_model.bin --augmentation True

