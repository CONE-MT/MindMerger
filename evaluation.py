
from mindmerger_tools.input_features import *
from tqdm import tqdm
from mindmerger_tools.utils import extract_last_num
import math
import torch

def evaluate_math(model, test_set, tokenizer_llm, tokenizer_mt, max_seq_len,
                  max_gen_len, use_prompt, langs_map):
    model.eval()
    results_list = []
    hit = 0
    step_trange = tqdm(test_set)
    preds, golds = [], []
    for test_step in step_trange:
        sources = test_step['source']
        prompts = test_step['prompt']
        targets = test_step['target']
        source_languages = test_step['source_language']
        input_ids_m2m, attention_mask_m2m = mt_input_features(sources, tokenizer_mt,
                                                              max_seq_len, source_languages, langs_map)
        input_ids_prompt, mask_prompt = None, None
        if use_prompt:
            add_bos_token = False
            add_eos_token = False
            input_ids_prompt, mask_prompt = llm_input_features(prompts, tokenizer_llm, max_gen_len, add_bos_token,
                                                           add_eos_token)
        generate_ids = model(input_ids_m2m, attention_mask_m2m,
                             input_ids_prompt=input_ids_prompt,
                             mask_prompt=mask_prompt)

        results = tokenizer_llm.batch_decode(generate_ids,
                                               skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)

        preds += results
        golds += targets
        results_p = [extract_last_num(text) for text in results]
        for result_p, result, prompt, question, target in zip(results_p, results, prompts, sources, targets):
            answer = extract_last_num(target)
            results_list.append({
                'question': question,
                'answer': answer,
                'prompt': prompt,
                'prediction': str(result_p),
                'output': result
            })
            if float(answer) == float(result_p):
                hit += 1
        acc = round(hit / len(results_list) * 100, 2)
        loss_show = 'Acc:' + str(acc)
        step_trange.set_postfix_str(loss_show)

    acc = round(hit / len(results_list) * 100, 2)
    return acc, results_list

def evaluate_classification(model, test_set, tokenizer_llm, tokenizer_mt, max_seq_len,
                  max_gen_len, use_prompt, langs_map):
    model.eval()
    results_list = []
    hit = 0
    step_trange = tqdm(test_set)
    preds, golds = [], []
    for test_step in step_trange:
        prompts = test_step['prompt']
        targets = test_step['target']
        source_languages = test_step['source_language']
        input_ids_m2m, attention_mask_m2m = mt_input_features(prompts, tokenizer_mt,
                                                              max_seq_len, source_languages, langs_map)
        input_ids_prompt, mask_prompt = None, None
        if use_prompt:
            add_bos_token = False
            add_eos_token = False
            input_ids_prompt, mask_prompt = llm_input_features(prompts, tokenizer_llm, max_gen_len, add_bos_token,
                                                           add_eos_token)
        generate_ids = model(input_ids_m2m, attention_mask_m2m,
                             input_ids_prompt=input_ids_prompt,
                             mask_prompt=mask_prompt)

        results = tokenizer_llm.batch_decode(generate_ids,
                                               skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)

        preds += results
        golds += targets

        for result, prompt, target in zip(results, prompts, targets):
            result = result.strip()
            results_list.append({
                'prompt': prompt,
                'prediction': result,
                'answer': target
            })
            if target == result:
                hit += 1

        acc = round(hit / len(results_list) * 100, 2)
        loss_show = 'Acc:' + str(acc)
        step_trange.set_postfix_str(loss_show)

    acc = round(hit / len(results_list) * 100, 2)
    return acc, results_list


def evaluate_ppl(model, test_set, tokenizer_llm, tokenizer_mt, max_seq_len, max_gen_len, langs_map, use_prompt):
    model.eval()
    step_trange = tqdm(test_set)
    loss_all = 0
    step_i = 0
    for test_step in step_trange:
        step_i += 1
        sources = test_step['source']
        prompts = test_step['prompt']
        targets = test_step['target']
        source_languages = test_step['source_language']
        input_ids_m2m, attention_mask_m2m = mt_input_features(sources, tokenizer_mt,
                                                              max_seq_len, source_languages, langs_map)
        add_bos_token = False
        add_eos_token = True
        labels, mask_label = llm_input_features(targets, tokenizer_llm,
                                                  max_gen_len, add_bos_token, add_eos_token)

        input_ids_prompt, mask_prompt = None, None
        if use_prompt:
            add_bos_token = False
            add_eos_token = False
            input_ids_prompt, mask_prompt = llm_input_features(prompts, tokenizer_llm,
                                                                 max_gen_len, add_bos_token,
                                                                 add_eos_token)
        loss = model(input_ids_m2m, attention_mask_m2m,
                     input_ids_prompt=input_ids_prompt, mask_prompt=mask_prompt,
                     labels=labels, mask_label=mask_label)
        loss_all += loss.mean().item()
        loss_show = 'loss:' + str(round(loss_all / (step_i), 4))
        step_trange.set_postfix_str(loss_show)

    loss = loss_all / step_i
    perplexity = math.exp(loss)
    model.train()
    torch.cuda.empty_cache()
    return perplexity
