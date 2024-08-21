#coding=utf-8
import torch.fx
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
from mindmerger_tools.utils import save_model, set_seed
from mindmerger_tools.read_datasets import *
import argparse
import ast
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import json
import deepspeed
from mindmerger_tools.input_features import *
from modeling_mindmerger import MindMerger
import os
from mindmerger_tools.deepspeed_config import get_train_ds_config
from evaluation import evaluate_ppl


def main(args):
    llm_path = args.llm_path
    mt_path = args.mt_path

    train_num = args.train_num
    stage_name = args.stage_name
    task = args.task
    augmentation = args.augmentation
    save_name = args.save_name
    result_path_base = f'./results/{save_name}/{task}/{stage_name}/'
    output_model_path_base = f'./outputs/{save_name}/{task}/{stage_name}/'

    if stage_name == 'mapping':
        if 'math' in task:
            languages = ['Bengali', 'Thai', 'Swahili', 'Japanese', 'Chinese', 'German', 'French', 'Russian',
                         'Spanish']
            train_set = read_lego(train_num, languages)

        elif 'csqa' in task:
            languages = ['Urdu', 'Hindi', 'Swahili', 'Japanese', 'Vietnamese', 'Polish', 'Chinese',
                         'Flemish', 'Russian', 'Italian', 'German', 'Portuguese', 'French', 'Spanish', 'Arabic']
            train_set = read_lego(train_num, languages)

        else:
            languages = ['Swahili', 'Urdu', 'Hindi', 'Thai', 'Arabic', 'Turkish', 'Greek',
                          'Vietnamese', 'Chinese', 'Russian', 'Bulgarian', 'German', 'French', 'Spanish']
            train_set = read_lego(train_num, languages)
        task = 'translation'
    else:
        if 'math' in task:
            train_set = read_math_train(train_num)
        elif 'csqa' in task:
            train_set = read_x_csqa_train()
        else:
            train_set = read_xnli_train()

    dev_set = train_set[:args.dev_size]
    train_set = train_set[args.dev_size:]

    train_set = MathDataset(train_set, task)
    dev_set = MathDataset(dev_set, task)
    lr = args.lr
    epoch_num = args.epoch_num

    max_seq_len = args.max_seq_len
    max_gen_len = args.max_gen_len

    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    train_micro_batch_size_per_gpu = args.train_micro_batch_size_per_gpu
    gpu_num = torch.cuda.device_count()
    gradient_accumulation = train_batch_size // (train_micro_batch_size_per_gpu * gpu_num)
    assert train_micro_batch_size_per_gpu * gpu_num * gradient_accumulation == train_batch_size
    ds_config = get_train_ds_config(train_batch_size, train_micro_batch_size_per_gpu, lr, gradient_accumulation)

    os.makedirs(output_model_path_base, exist_ok=True)
    os.makedirs(result_path_base, exist_ok=True)
    tokenizer_m2m = AutoTokenizer.from_pretrained(mt_path)
    tokenizer_llm = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
    tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.padding_side = "left"
    # tokenizer_llm.pad_token = "[PAD]"

    print(json.dumps({
        'llm_path': llm_path,
        'mt_path': mt_path,
        'lr': lr,
        'epoch_num': epoch_num,
        'gradient_accumulation': gradient_accumulation,
        'train_set:': len(train_set),
        'dev_set:': len(dev_set),
        'max_seq_len': max_seq_len,
        'max_gen_len': max_gen_len,
        'train_batch_size': train_batch_size,
        'result_path': result_path_base,
        'output_model_path': output_model_path_base,
    }, indent=2))

    if stage_name != 'mapping' and args.init_checkpoint is None:
        args.init_checkpoint = f'./outputs/{save_name}/{task}/mapping/pytorch_model.bin'
    model = MindMerger(mt_path, llm_path, max_gen_len,
                       tokenizer_llm.bos_token_id,
                       tokenizer_llm.pad_token_id)
    if args.init_checkpoint is not None:
        init_checkpoint = args.init_checkpoint
        checkpoint = torch.load(init_checkpoint, map_location='cpu')
        model_dict = checkpoint['model_state_dict']
        model.mapping.load_state_dict(model_dict, False)
        print('mapping layer init from:', init_checkpoint)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model, optimizer, _, __ = deepspeed.initialize(
        config=ds_config,
        model=model,
        model_parameters=parameters,
        training_data=None)

    train_sampler = DistributedSampler(train_set)
    dev_sampler = SequentialSampler(dev_set)

    train_set = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=train_micro_batch_size_per_gpu,
        sampler=train_sampler,
    )
    dev_set = torch.utils.data.DataLoader(
        dataset=dev_set,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=dev_sampler,
        num_workers=1,
        drop_last=False)

    global_rank = torch.distributed.get_rank()
    # best_perplexity = 1000000000
    best_perplexity = evaluate_ppl(model, dev_set, tokenizer_llm, tokenizer_m2m,
                                   max_seq_len, max_gen_len, langs_map, augmentation)
    eval_step = 2000
    for epoch in range(epoch_num):
        model.train()
        tr_loss, nb_tr_steps = 0, 0
        step_count = 0
        step_trange = tqdm(train_set)
        for train_step in step_trange:
            sources = train_step['source']
            prompts = train_step['prompt']
            targets = train_step['target']
            source_languages = train_step['source_language']

            input_ids_m2m, attention_mask_m2m = mt_input_features(sources, tokenizer_m2m,
                                                                  max_seq_len, source_languages,
                                                                  langs_map)
            add_bos_token = False
            add_eos_token = True
            labels, mask_label = llm_input_features(targets, tokenizer_llm,
                                                    max_gen_len, add_bos_token, add_eos_token)

            input_ids_prompt, mask_prompt = None, None
            if augmentation:
                add_bos_token = False
                add_eos_token = False
                input_ids_prompt, mask_prompt = llm_input_features(prompts, tokenizer_llm,
                                                                   max_gen_len, add_bos_token,
                                                                   add_eos_token)

            loss = model(input_ids_m2m, attention_mask_m2m,
                         input_ids_prompt=input_ids_prompt, mask_prompt=mask_prompt,
                         labels=labels, mask_label=mask_label)
            loss = loss.mean()
            tr_loss += loss.item()
            nb_tr_steps += 1
            model.backward(loss)
            model.step()

            loss_show = ' Epoch:' + str(epoch) + " loss:" + str(round(tr_loss / nb_tr_steps, 4)) #+ f" lr:{'%.2E' % scheduler.get_last_lr()[0]}"
            step_trange.set_postfix_str(loss_show)

            if step_count % eval_step == 0 and step_count > 0:
                perplexity = evaluate_ppl(model, dev_set, tokenizer_llm, tokenizer_m2m,
                                          max_seq_len, max_gen_len, langs_map, augmentation)
                print('ppl:', perplexity)
                if global_rank == 0 and perplexity < best_perplexity:
                    best_perplexity = perplexity
                    save_model(output_model_path_base, model.mapping)
                    print('save new best')
            step_count += 1



        perplexity = evaluate_ppl(model, dev_set, tokenizer_llm, tokenizer_m2m,
                                  max_seq_len, max_gen_len, langs_map, augmentation)
        print('ppl:', perplexity)
        if global_rank == 0 and perplexity < best_perplexity:
            best_perplexity = perplexity
            save_model(output_model_path_base, model.mapping)
            print('save new best')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm_path",
        type=str,
        default='../LLMs/Llama-2-7b-hf/'
    )
    parser.add_argument(
        "--mt_path",
        type=str,
        default='../LLMs/mt5-xl/'
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default='MindMerger'
    )
    parser.add_argument(
        "--task",
        type=str,
        default='translation'
    )
    parser.add_argument(
        "--stage_name",
        type=str,
        default='mapping'
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5
    )
    parser.add_argument(
        "--epoch_num",
        type=int,
        default=3
    )
    parser.add_argument(
        "--train_num",
        type=int,
        default=100000
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=24
    )
    parser.add_argument(
        "--train_micro_batch_size_per_gpu",
        type=int,
        default=1
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=2
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
        "--dev_size",
        type=int,
        default=3000
    )
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default='0'
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0
    )
    parser.add_argument(
        "--augmentation",
        type=ast.literal_eval,
        default=False
    )
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(0)

    langs = ['Thai', 'Swahili', 'Bengali', 'Chinese', 'German', 'Spanish', 'French', 'Japanese', 'Russian', 'English']
    langs_map_flores = {'Swahili': 'swh', 'Benli': 'ben', 'English': 'eng', 'Thai': 'tha', 'Chinese': 'zho_simpl',
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
