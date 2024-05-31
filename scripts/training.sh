
deepspeed --master_port 50002 run_training.py --deepspeed --llm_path meta-math/MetaMath-7B-V1.0 \
          --mt_path google/mt5-xl --stage_name mapping --train_num 100000 --train_batch_size 128 \
          --train_micro_batch_size_per_gpu 8 --gradient_accumulation 4 --augmentation False --epoch_num 3 \
          --gpu 0,1,2,3 --max_seq_len 200 --max_gen_len 200 --train_batch_size 128 --eval_batch_size 2

deepspeed --master_port 50002 run_training.py --deepspeed --llm_path meta-math/MetaMath-7B-V1.0 \
          --mt_path google/mt5-xl --stage_name augmentation --train_num 30000 --train_batch_size 128 \
          --train_micro_batch_size_per_gpu 2 --gradient_accumulation 16 --augmentation True --epoch_num 3 \
          --gpu 0,1,2,3 --max_seq_len 512 --max_gen_len 512 --train_batch_size 128 --eval_batch_size 2
