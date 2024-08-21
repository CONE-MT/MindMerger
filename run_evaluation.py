import torch.fx
from transformers import AutoTokenizer
import torch
from mindmerger_tools.read_datasets import *
from mindmerger_tools.utils import save_model, set_seed, extract_last_num
import argparse
import ast
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import json
from modeling_mindmerger import MindMerger
import os
from evaluation import *
import deepspeed
from mindmerger_tools.deepspeed_config import get_train_ds_config

def main(args):
    llm_path = args.llm_path
    mt_path = args.mt_path

    max_seq_len = args.max_seq_len
    max_gen_len = args.max_gen_len

    eval_batch_size = args.eval_batch_size

    augmentation = args.augmentation
    save_name = args.save_name
    task = args.task

    result_path_base = f'./results/{save_name}/{task}/'

    if 'mgsm' in task:
        test_sets = read_mgsms()
        task = 'math'
    elif 'msvamp' in task:
        test_sets = read_msvamp()
        task = 'math'
    elif 'csqa' in task:
        test_sets = read_x_csqa()
    else:
        test_sets = read_xnli()

    os.makedirs(result_path_base, exist_ok=True)
    tokenizer_m2m = AutoTokenizer.from_pretrained(mt_path)
    tokenizer_llm = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
    tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.padding_side = "left"
    # tokenizer_llm.pad_token = "[PAD]"
    print(json.dumps({
        'llm_path': llm_path,
        'mt_path': mt_path,
        'max_seq_len': max_seq_len,
        'max_gen_len': max_gen_len,
        'save_name': save_name,
        'result_path_base': result_path_base
    }, indent=2))

    train_micro_batch_size_per_gpu = args.train_micro_batch_size_per_gpu
    train_batch_size = args.train_batch_size
    gpu_num = torch.cuda.device_count()
    gradient_accumulation = train_batch_size // (train_micro_batch_size_per_gpu * gpu_num)
    assert train_micro_batch_size_per_gpu * gpu_num * gradient_accumulation == train_batch_size
    ds_config = get_train_ds_config(train_batch_size=train_batch_size,
                                    train_micro_batch_size_per_gpu=train_micro_batch_size_per_gpu,
                                    gradient_accumulation_steps=gradient_accumulation,
                                    )


    model = MindMerger(mt_path, llm_path, max_gen_len,
                       tokenizer_llm.bos_token_id,
                       tokenizer_llm.pad_token_id)


    if args.init_checkpoint is not None:
        init_checkpoint = args.init_checkpoint
        checkpoint = torch.load(init_checkpoint, map_location='cpu')
        model_dict = checkpoint['model_state_dict']
        model.mapping.load_state_dict(model_dict, True)
        print('mapping init from:', init_checkpoint)
    # model = model.cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model, optimizer, _, __ = deepspeed.initialize(
        config=ds_config,
        model=model,
        model_parameters=parameters,
        training_data=None)
    scores_map = {}
    avg = 0
    for test_lang in test_sets:
        test_set = test_sets[test_lang]
        test_sampler = SequentialSampler(test_set)
        test_set = MathDataset(test_set, task)
        test_set = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=eval_batch_size,
            sampler=test_sampler,
            shuffle=False,
            num_workers=1,
            drop_last=False)
        if 'math' in task:
            acc, results_list = evaluate_math(model, test_set, tokenizer_llm, tokenizer_m2m,
                                                     max_seq_len, max_gen_len, augmentation, langs_map)
        else:
            acc, results_list = evaluate_classification(model, test_set, tokenizer_llm, tokenizer_m2m,
                                              max_seq_len, max_gen_len, augmentation, langs_map)
        print('test_lang:', test_lang, 'acc:', acc)
        scores_map[test_lang] = acc
        result_path = f'{result_path_base}/{test_lang}.json'
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, ensure_ascii=False, indent=2)
        avg += acc
    print(scores_map)
    print('Average accuracy :', round(avg / len(test_sets), 1))
    score_path = f'{result_path_base}/scores.tsv'
    with open(score_path, 'w', encoding='utf-8') as f:
        for lang in scores_map:
            score = scores_map[lang]
            f.write(f'{lang}\t{score}\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm_path",
        type=str,
        default='../LLMs/MetaMath-7B-V1.0/'
    )
    parser.add_argument(
        "--mt_path",
        type=str,
        default='../LLMs/mt5-xl/'
    )
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default='MindMerger',
    )
    parser.add_argument(
        "--task",
        type=str,
        default='math',
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512
    )
    parser.add_argument(
        "--max_gen_len",
        type=int,
        default=512
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default='0'
    )
    parser.add_argument(
        "--augmentation",
        type=ast.literal_eval,
        default=True
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=128
    )
    parser.add_argument(
        "--train_micro_batch_size_per_gpu",
        type=int,
        default=1
    )
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(0)

    langs = ['Thai', 'Swahili', 'Bengali', 'Chinese', 'German', 'Spanish', 'French', 'Japanese', 'Russian', 'English']
    langs_map_flores = {'Swahili': 'swh', 'Bengali': 'ben', 'English': 'eng', 'Thai': 'tha', 'Chinese': 'zho_simpl',
                        'German': 'deu', 'Spanish': 'spa', 'French': 'fra', 'Japanese': 'jpn', 'Russian': 'rus', }
    langs_map_m2m = {'English': 'en', 'Swahili': 'sw', 'Chinese': 'zh', 'Bengali': 'bn',
                     'German': 'de', 'Spanish': 'es', 'French': 'fr', 'Japanese': 'ja',
                     'Russian': 'ru', 'Thai': 'th', 'Greek': 'el', 'Telugu': 'te',
                     'Arabic': 'ar', 'Bulgarian': 'bg', 'Croatian': 'hr', 'Hungarian': 'hu',
                     'Italian': 'it', 'Lithuanian': 'lt', 'Macedonian': 'mk', 'Polish': 'pl',
                     'Portuguese': 'pt', 'Albanian': 'sq', 'Serbian': 'sr', 'Turkish': 'tr',
                     'Vietnamese': 'vi', 'Hindi': 'hi', 'Flemish': 'nl', 'Urdu': 'ur'}
    langs_map_nllb = {
        'English': 'eng_Latn', 'Swahili': 'swh_Latn', 'Chinese': 'zho_Hans', 'Bengali': 'ben_Beng',
        'German': 'deu_Latn', 'Spanish': 'spa_Latn', 'French': 'fra_Latn', 'Japanese': 'jpn_Jpan',
        'Russian': 'rus_Cyrl', 'Thai': 'tha_Thai'
    }
    if 'nllb' in args.mt_path:
        langs_map = langs_map_nllb
    else:
        langs_map = langs_map_m2m
    main(args)
