deepspeed --master_port 50002 run_training.py --deepspeed --llm_path  LLaMAX/LLaMAX2-7B-X-CSQA \
          --mt_path google/mt5-xl/ --stage_name mapping --task x-csqa --augmentation False \
          --train_num 100000 --train_batch_size 128 --train_micro_batch_size_per_gpu 8 --epoch_num 3 \
          --max_seq_len 200 --max_gen_len 200 --train_batch_size 128 --eval_batch_size 2


deepspeed --master_port 50002 run_training.py --deepspeed --llm_path LLaMAX/LLaMAX2-7B-X-CSQA \
          --mt_path google/mt5-xl/ --stage_name augmentation --task x-csqa --augmentation True \
          --train_num 30000 --train_batch_size 128 --train_micro_batch_size_per_gpu 8 --epoch_num 3 \
          --max_seq_len 200 --max_gen_len 200 --train_batch_size 128 --eval_batch_size 2

