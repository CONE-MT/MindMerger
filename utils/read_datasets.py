import json
from torch.utils.data import Dataset
from prompts import *
import random

langs_map = {'English': 'en', 'Swahili': 'sw', 'Chinese': 'zh', 'Bengali': 'bn',
                     'German': 'de', 'Spanish': 'es', 'French': 'fr', 'Japanese': 'ja',
                     'Russian': 'ru', 'Thai': 'th',
                     'Arabic': 'ar', 'Bulgarian': 'bg', 'Croatian': 'hr', 'Hungarian': 'hu',
                     'Italian': 'it', 'Lithuanian': 'lt', 'Macedonian': 'mk', 'Polish': 'pl',
                     'Portuguese': 'pt', 'Albanian': 'sq', 'Serbian': 'sr', 'Turkish': 'tr',
                     'Vietnamese': 'vi', 'Hindi': 'hi', 'Flemish': 'nl', 'Urdu': 'ur'}

class MathDataset(Dataset):
    def __init__(self, dataset, task) -> None:
        super().__init__()
        self.dataset = dataset
        self.task = task
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.task == 'translation':
            sample['prompt'] = construct_prompt_mt(sample['source'],
                                                   sample['source_language'],
                                                   sample['target_language'])
        else:
            sample['prompt'] = construct_prompt_math(sample['source'])
        return sample

def read_lego(train_num):
    train_list = ['Thai', 'Swahili', 'Bengali', 'Chinese', 'German',
                  'French', 'Japanese', 'Russian', 'Spanish']
    dataset_train = []
    for train_name in train_list:
        train_name_map = langs_map[train_name]
        path_base = f'./datas/lego_mt/en-{train_name_map}'
        path_src = f'{path_base}/train_100k.{train_name_map}'
        path_trg = f'{path_base}/train_100k.en'
        sources = read_dataset(path_src)[:train_num]
        targets = read_dataset(path_trg)[:train_num]
        train_set = [(source, target) for source, target in zip(sources, targets)]
        for source, target in train_set:
            dataset_train.append({
                'source': source,
                'target': target,
                'source_language': train_name,
                'target_language': 'English'
            })
    random.shuffle(dataset_train)
    return dataset_train

def read_flores():
    flores_langs_map = {'Swahili': 'swh', 'Bengali': 'ben', 'English': 'eng', 'Thai': 'tha', 'Chinese': 'zho_simpl',
                        'German': 'deu', 'Spanish': 'spa', 'French': 'fra', 'Japanese': 'jpn', 'Russian': 'rus', }
    test_list = ['Bengali', 'Thai', 'Swahili', 'Japanese', 'Chinese',
                 'German', 'Spanish', 'French', 'Russian']
    datasets_test = {}
    for test_name in test_list:
        dataset = []
        flores_name = flores_langs_map[test_name]
        path_base = f'./datas/flores101/devtest/'
        path_src = f'{path_base}/{flores_name}.devtest'
        flores_name = flores_langs_map['English']
        path_trg = f'{path_base}/{flores_name}.devtest'
        sources = read_dataset(path_src)
        targets = read_dataset(path_trg)
        for source, target in zip(sources, targets):
            dataset.append({
                'source': source,
                'target': target,
                'source_language': test_name,
                'target_language': 'English'
            })
        datasets_test[test_name] = dataset
    return datasets_test


def read_math_train(train_num):
    train_set = read_dataset('./datas/query_translation/metamath_615k.json')
    train_sets = {}
    for sample in train_set:
        lang = sample['lang']
        sample = {
            'source': sample['query'],
            'target': sample['response'],
            'source_language': lang,
            'target_language': 'English'
        }
        if lang not in train_sets:
            train_sets[lang] = [sample]
        else:
            if len(train_sets[lang]) < train_num:
                train_sets[lang].append(sample)
    dataset_train = []
    for lang in train_sets:
        dataset = train_sets[lang]
        for sample in dataset:
            dataset_train.append(sample)
    random.shuffle(dataset_train)
    return dataset_train


def read_msvamp():
    datasets_test = {}
    test_list = ['Bengali', 'Thai', 'Swahili', 'Japanese', 'Chinese', 'German', 'French', 'Russian',
                 'Spanish', 'English']
    for test_name in test_list:
        test_path = f'./datas/msvamp/test_{test_name}.json'
        dataset = read_msvamp(test_path, test_name)
        dataset_test = []
        for sample in dataset:
            dataset_test.append({
                'source': sample['question'],
                'target': sample['answer'],
                'source_language': test_name,
                'target_language': test_name
            })
        datasets_test[test_name] = dataset_test
    return datasets_test


def read_docs(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def read_dataset(path):
    if 'jsonl' in path:
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                dataset.append(json.loads(line))
    elif 'json' in path:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        if isinstance(dataset, dict):
            if 'data' in dataset:
                dataset = dataset['data']
    else:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = f.readlines()
    return dataset


def read_mgsms():
    datasets_test = {}
    test_list = ['Bengali', 'Thai', 'Swahili', 'Japanese', 'Chinese', 'German', 'French', 'Russian',
                 'Spanish', 'English']
    for test_name in test_list:
        test_name_abb = langs_map[test_name]
        test_path = f'./datas/mgsm/mgsm_{test_name_abb}.tsv'
        dataset = read_mgsm(test_path, test_name)
        dataset_test = []
        for sample in dataset:
            dataset_test.append({
                'source': sample['question'],
                'target': sample['answer'],
                'source_language': test_name,
                'target_language': test_name
            })
        datasets_test[test_name] = dataset_test
    return datasets_test

def read_mgsm(path, language):
    dataset = read_docs(path)
    dataset_new = []
    for i, sample in enumerate(dataset):
        sp = sample.split("\t")
        question = sp[0]
        answer = sp[1].replace(",", '').strip()
        explanation = ""
        dataset_new.append({
            'id': i,
            'question': question,
            'explanation': explanation,
            'answer': answer,
            'language': language,
            'source_language': language,
            'target_language': language
        })
    return dataset_new
