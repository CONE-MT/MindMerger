
def construct_prompt_math(query):
    # return query
    prompt_no_input = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{query}\n\n### Response: Let's think step by step."
    )
    return prompt_no_input

def construct_prompt_mt(query, src_lang_name, trg_lang_name):
    instruction = f'Translate the following sentences from {src_lang_name} to {trg_lang_name}.'
    prompt_no_input = (
        'Below is an instruction that describes a task, paired with an input that provides further context. '
        'Write a response that appropriately completes the request.\n'
        f'### Instruction:\n{instruction}\n'
        f'### Input:\n{query}\n### Response:'
    )
    return prompt_no_input