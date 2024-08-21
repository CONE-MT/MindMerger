
def construct_prompt_math(query):
    # return query
    prompt_no_input = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{query}\n\n### Response: Let's think step by step."
    )
    return prompt_no_input


def construct_prompt_xnli(sample):
    sentence1 = sample['sentence1']
    sentence2 = sample['sentence2']
    source = f'Premise: {sentence1}\nHypothesis: {sentence2}\nLabel:'
    return source

def construct_prompt_x_csqa(sample):
    question = sample['question']['stem']
    choices = sample['question']['choices']
    choice_text = [chr(ord('A') + i) + '. ' + choice['text'] for i, choice in enumerate(choices)]
    source = question + '\nOptions:' + '\t'.join(choice_text) + '\nAnswer:'
    return source

def construct_prompt_mt(query, src_lang_name, trg_lang_name):
    instruction = f'Translate the following sentences from {src_lang_name} to {trg_lang_name}.'
    prompt_no_input = (
        'Below is an instruction that describes a task, paired with an input that provides further context. '
        'Write a response that appropriately completes the request.\n'
        f'### Instruction:\n{instruction}\n'
        f'### Input:\n{query}\n### Response:'
    )
    return prompt_no_input