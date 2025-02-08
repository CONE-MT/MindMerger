# MindMerger
Code for [MindMerger: Efficient Boosting LLM Reasoning in non-English Languages](https://arxiv.org/pdf/2405.17386) (NeurIPS 2024)

MindMerger is a new method for multilingual reasoning, which merges LLMs with the external language understanding capabilities from multilingual models to boost the multilingual reasoning performance. 
A two-step training scheme is introduced to first train to embeded the external capabilities into LLMs and then train the collaborative utilization of the external capabilities and the built-in capabilities in LLMs.

![model](model.png)

## Pip Installation
```angular2html
pip install -r requirements.txt
```

## Data Preparation
Download the datasets and checkpoint in [here](https://drive.google.com/drive/folders/1Rm5ppr1fCd4KbiDR2LSFKNChq_uSfiSE?usp=drive_link) and put them under current folder.

In the folder, we provide two stage training data and evaluation data for math, x-csqa, and xnli tasks.
We provide the checkpoint of MindMerger for math based on [MetaMath-Llama-7B](https://huggingface.co/meta-math/MetaMath-7B-V1.0), for x-csqa based on [LLaMAX-7B-X-CSQA](https://huggingface.co/LLaMAX/LLaMAX2-7B-X-CSQA), and for xnli based on [LLaMAX-7B-X-XNLI](https://huggingface.co/LLaMAX/LLaMAX2-7B-XNLI). mT5-xl is used as multilingual encoder.

## Evaluation
The checkpoint is the parameters of mapping layer for specfic LLM and multilingual model. To evaluate the performance of MindMerger, you can run as follows:
```angular2html
deepspeed run_evaluation.py --deepspeed \
    --llm_path meta-math/MetaMath-7B-V1.0 \
    --mt_path google/mt5-xl \
    --init_checkpoint outputs/MergeMinds/math/augmentation/pytorch_model.bin \
    --augmentation True
```

Evaluation results on MGSM dataset:

| MGSM              | Avg.  | Te   |Bn | Th   | Sw   | Ja   | Zh   | De   | Fr | Ru   | Es   | En   |
|-------------------|-------|------|------|------|------|------|------|----|------|------|------|------|
| MindMerger (MetaMath-Llama-7B)  | 57.6  | 52.8 | 52.0 | 59.2 | 56.8 | 51.2 | 55.2 |61.2| 55.2 | 61.6 | 62.4 | 66.0 |


Evaluation results on X-CSQA dataset:

| X-CSQA                        | Avg. | Sw   | Ur   | Hi   | Ar   | Vi    | Ja    | Pl    | Zh     | Nl   | Ru   | It    | De    | Pt     | Fr     | Es    | En     |
|-------------------------------|------|------|------|------|------|-------|-------|-------|--------|------|------|-------|-------|--------|--------|-------|--------|
| Llama2-7B-X-CSQA              | 50.9 | 23.2 | 24.7 | 32.9 | 32.4 | 51.0  | 50.0  | 51.5  | 55.6   | 56.9 | 55.8 | 58.8  | 59.9  | 60.4   | 61.8   | 61.9  | 78.1   | 
| MindMerger (Llama2-7B-X-CSQA) | 61.0 | 45.5 | 46.2 | 48.4 | 51.4 | 60.6 | 53.9 | 63.3 | 62.9 | 63.8 | 63.7 | 66.8 | 67.0 | 67.1 | 68.1 | 69.1 | 78.1 |
| LLaMAX-7B-X-CSQA              | 55.1 | 43.5 | 39.0 | 44.1 | 45.1 | 54.0  | 49.9  | 54.6  | 58.2   | 58.9 | 57.1 | 59.1  | 59.0  | 60.9   | 61.6   | 62.7  | 74.0   | 
| MindMerger (LLaMAX-7B-X-CSQA) | 61.2 | 51.2 | 50.7 | 50.8 | 54.4 | 60.4  | 55.9  | 63.8  | 64.4   | 64.3 | 61.5 | 64.2  | 64.1  | 65.3   | 64.6   | 67.7  | 75.4   |


Evaluation results on XNLI dataset:

| XNLI                          | Avg.  | Sw   | Ur   | Hi    | Th    | Ar   | Tr    | El   | Vi    | Zh    | Ru   | Bg    | De    | Fr    | Es    | En    |
|-------------------------------|-------|------|------|-------|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|------|-------|-------|------|-------|-------|-------|-------|-------|
| Llama2-7B-X-XNLI              | 70.6  | 44.6 | 55.1 | 62.2  | 58.4  | 64.7 | 64.9  | 65.6 | 75.4  | 75.9  | 78.9 | 78.6  | 80.7  | 81.7  | 83.1  | 89.5  |
| MindMerer (Llama2-7B-X-XNLI)  | 78.4 | 66.6 | 69.4 | 74.7 | 71.8 | 76.2 | 75.7 | 78.5 | 80.3 | 80.0 | 80.7 | 82.4 | 83.5 | 83.9 | 84.4 | 88.7 |
| LLaMAX-7B-X-XNLI              | 76.2  | 66.7 | 65.3 | 69.1  | 66.2  | 73.6 | 71.8  | 74.3 | 77.4  | 78.3  | 80.3 | 81.6  | 82.2  | 83.0  | 84.1  | 89.7  | 
| MindMerer (LLaMAX-7B-X-XNLI)  | 79.2  | 72.6 | 71.5 | 74.9  | 73.4  | 77.1 | 76.4  | 78.7 | 80.4  | 80.5  | 80.8 | 82.4  | 83.1  | 84.1  | 84.5  | 88.5  |


## Training
We use a two-stage scheme to train MergeMinds.

**Mapping stage** helps LLM learn to use the capabilities of multilingual model.
```angular2html
deepspeed run_training.py --deepspeed \
    --llm_path meta-math/MetaMath-7B-V1.0 \
    --mt_path google/mt5-xl \
    --task math \
    --stage_name mapping --train_num 100000 \
    --train_batch_size 128 \
    --train_micro_batch_size_per_gpu 8 \
    --augmentation False \
    --epoch_num 3 \
    --max_seq_len 200 \
    --max_gen_len 200 
```

**Augmentation stage** helps LLM collaboratively utilize its own and the capabilities from multilingual model.
```angular2html
deepspeed run_training.py --deepspeed \
    --llm_path meta-math/MetaMath-7B-V1.0 \
    --mt_path google/mt5-xl \
    --task math \
    --stage_name augmentation --train_num 30000 \
    --train_batch_size 128 \
    --train_micro_batch_size_per_gpu 2 \
    --augmentation False \
    --epoch_num 3 \
    --max_seq_len 512 \
    --max_gen_len 512
```
You can also use the script to run our codes:
```
bash scripts/training_math.sh
```

### Reference

Please cite this paper in your publications if it helps your research:

```
@inproceedings{DBLP:conf/nips/HuangZ0LY24,
  author       = {Zixian Huang and
                  Wenhao Zhu and
                  Gong Cheng and
                  Lei Li and
                  Fei Yuan},
  editor       = {Amir Globersons and
                  Lester Mackey and
                  Danielle Belgrave and
                  Angela Fan and
                  Ulrich Paquet and
                  Jakub M. Tomczak and
                  Cheng Zhang},
  title        = {MindMerger: Efficiently Boosting {LLM} Reasoning in non-English Languages},
  booktitle    = {Advances in Neural Information Processing Systems 38: Annual Conference
                  on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver,
                  BC, Canada, December 10 - 15, 2024},
  year         = {2024},
  url          = {http://papers.nips.cc/paper\_files/paper/2024/hash/3bf80b34f731313b8292f4578e820c90-Abstract-Conference.html}
}
```
