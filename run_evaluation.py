import torch.fx
from transformers import AutoTokenizer
import torch
from tools.read_datasets import *
from tools.utils import save_model, set_seed, extract_last_num
import argparse
import ast
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import json
from modeling_mindmerger import MindMerger
import os
from evaluation import *

def main(args):
    llm_path = args.llm_path
    mt_path = args.mt_path

    max_seq_len = args.max_seq_len
    max_gen_len = args.max_gen_len

    eval_batch_size = args.eval_batch_size

    augmentation = args.augmentation
    save_name = args.save_name

    result_path_base = f'./results/{save_name}/'
    test_sets = read_mgsms()
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
    model = MindMerger(mt_path, llm_path, max_gen_len,
                       tokenizer_llm.bos_token_id,
                       tokenizer_llm.pad_token_id)
    if args.init_checkpoint is not None:
        init_checkpoint = args.init_checkpoint
        checkpoint = torch.load(init_checkpoint, map_location='cpu')
        model_dict = checkpoint['model_state_dict']
        model.mapping.load_state_dict(model_dict, True)
        print('mapping init from:', init_checkpoint)
    model = model.cuda()
    scores_map = {}
    avg = 0
    url_acc, hrl_acc = 0, 0
    for test_lang in test_sets:

        test_set = test_sets[test_lang]
        test_sampler = SequentialSampler(test_set)
        test_set = MathDataset(test_set, 'math')
        test_set = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=eval_batch_size,
            sampler=test_sampler,
            shuffle=False,
            num_workers=1,
            drop_last=False)
        acc, results_list = evaluate_math(model, test_set, tokenizer_llm, tokenizer_m2m,
                                                 max_seq_len, max_gen_len, augmentation, langs_map)
        print('test_lang:', test_lang, 'acc:', acc)
        scores_map[test_lang] = acc
        result_path = f'{result_path_base}/{test_lang}.json'
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, ensure_ascii=False, indent=2)
        avg += acc

        if test_lang in ['Thai', 'Swahili', 'Bengali']:
            url_acc += acc
        else:
            hrl_acc += acc
    print(scores_map)
    print('Average accuracy :', round(avg / len(test_sets), 1),
          'Low-resource accuracy:', round(url_acc / 3, 1),
          'High-resource accuracy:', round(hrl_acc / 7, 1))

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
        "--eval_batch_size",
        type=int,
        default=8
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
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    set_seed(0)

    langs = ['Thai', 'Swahili', 'Bengali', 'Chinese', 'German', 'Spanish', 'French', 'Japanese', 'Russian', 'English']
    langs_map_flores = {'Swahili': 'swh', 'Bengali': 'ben', 'English': 'eng', 'Thai': 'tha', 'Chinese': 'zho_simpl',
                        'German': 'deu', 'Spanish': 'spa', 'French': 'fra', 'Japanese': 'jpn', 'Russian': 'rus', }
    langs_map_m2m = {'English': 'en', 'Swahili': 'sw', 'Chinese': 'zh', 'Bengali': 'bn',
                     'German': 'de', 'Spanish': 'es', 'French': 'fr', 'Japanese': 'ja',
                     'Russian': 'ru', 'Thai': 'th',
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