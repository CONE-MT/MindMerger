# MindMerger
Code for [MindMerger: Efficient Boosting LLM Reasoning in non-English Languages](https://arxiv.org/pdf/2405.17386)

MindMerger is a new method for multilingual reasoning, which merges LLMs with the external language understanding capabilities from multilingual models to boost the multilingual reasoning performance. 
A two-step training scheme is introduced to first train to embeded the external capabilities into LLMs and then train the collaborative utilization of the external capabilities and the built-in capabilities in LLMs.


![model](model.png)

## Pip Installation
```angular2html
pip install -r requirements.txt
```

## Data Preparation
Download the datasets and checkpoint in [here](https://drive.google.com/drive/folders/1DzlAZfvJAHBUyKWi4Uwi0ayZDdnfZw6O?usp=sharing) and put them under current folder.


## Evaluation
The checkpoint is the parameters of mapping layer for specfic LLM and multilingual model. To evaluate the performance of MindMerger, you can run as follow:
```angular2html
python run_evaluation.py \
    --llm_path meta-math/MetaMath-7B-V1.0 \
    --mt_path google/mt5-xl \
    --init_checkpoint outputs/MindMerger/pytorch_model.bin \
    --augmentation True
```

## Training
We use a two-stage scheme to train MergeMinds.

**Mapping stage** helps LLM learn to use the capabilities of multilingual model.
```angular2html
deepspeed run_training.py --deepspeed \
    --llm_path meta-math/MetaMath-7B-V1.0 \
    --mt_path google/mt5-xl \
    --stage_name mapping --train_num 100000 \
    --train_batch_size 128 \
    --train_micro_batch_size_per_gpu 8 \
    --gradient_accumulation 16 --augmentation False \
    --epoch_num 3 \
    --max_seq_len 200 \
    --max_gen_len 200
```

**Augmentation stage** helps LLM collaboratively utilize its own and the capabilities from multilingual model.
```angular2html
deepspeed run_training.py --deepspeed \
    --llm_path meta-math/MetaMath-7B-V1.0 \
    --mt_path google/mt5-xl \
    --stage_name augmentation --train_num 30000 \
    --train_batch_size 128 \
    --train_micro_batch_size_per_gpu 2 \
    --gradient_accumulation 64 --augmentation False \
    --epoch_num 3 \
    --max_seq_len 512 \
    --max_gen_len 512
```


### Reference

Please cite this paper in your publications if it helps your research:

```
@inproceedings{Huang2024MindMergerEB,
  title={MindMerger: Efficient Boosting LLM Reasoning in non-English Languages},
  author={Zixian Huang and Wenhao Zhu and Gong Cheng and Lei Li and Fei Yuan},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:270063337}
}
```
