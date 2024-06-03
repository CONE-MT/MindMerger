import torch
import os
import re
import random
import numpy as np
import json

def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)
    res = re.findall(r"(\d+(\.\d+)?)", text)
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 0.0

def save_model(output_model_file, model, model_name='pytorch_model.bin'):
    os.makedirs(output_model_file, exist_ok=True)
    output_model_file += model_name
    torch.save({
        'model_state_dict': model.state_dict(),
    }, output_model_file, _use_new_zipfile_serialization=False)


def save_dataset(path, name, dataset):
    os.makedirs(path, exist_ok=True)
    path = path + '/' + name
    if path.endswith('txt'):
        with open(path, 'w', encoding='utf-8') as f:
            for line in dataset:
                line = line.strip()
                f.write(line + '\n')
    else:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # gpu
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu

def save_hf_format(model, tokenizer, output_dir):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    # model_to_save = model.module if hasattr(model, 'module') else model
    # CONFIG_NAME = "config.json"
    # WEIGHTS_NAME = "pytorch_model.bin"
    # os.makedirs(output_dir, exist_ok=True)
    # output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    # output_config_file = os.path.join(output_dir, CONFIG_NAME)
    # save_dict = model_to_save.state_dict()
    # for key in list(save_dict.keys()):
    #     if "lora" in key:
    #         del save_dict[key]
    # torch.save(save_dict, output_model_file)
    # model_to_save.config.to_json_file(output_config_file)
    # tokenizer.save_vocabulary(output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
