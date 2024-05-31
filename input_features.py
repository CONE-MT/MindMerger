
import torch

def mt_input_features(input_texts_m2m, tokenizer_m2m, max_seq_len, source_languages, langs_map):
    input_ids_m2m, attention_mask_m2m = [], []
    for input_text_m2m, source_language in zip(input_texts_m2m, source_languages):
        tokenizer_m2m.src_lang = langs_map[source_language]
        encoding_m2m = tokenizer_m2m(input_text_m2m,
                                     padding='longest',
                                     max_length=max_seq_len,
                                     truncation=True)
        input_ids_m2m_item = encoding_m2m.input_ids
        attention_mask_m2m_item = encoding_m2m.attention_mask
        input_ids_m2m.append(input_ids_m2m_item)
        attention_mask_m2m.append(attention_mask_m2m_item)
    max_len = max([len(item) for item in input_ids_m2m])
    m2m_pad_id = tokenizer_m2m.pad_token_id
    for input_ids_m2m_item, attention_mask_m2m_item in zip(input_ids_m2m, attention_mask_m2m):
        while len(input_ids_m2m_item) < max_len:
            input_ids_m2m_item.append(m2m_pad_id)
            attention_mask_m2m_item.append(0)
    input_ids_m2m = torch.tensor(input_ids_m2m, dtype=torch.long).cuda()
    attention_mask_m2m = torch.tensor(attention_mask_m2m, dtype=torch.long).cuda()
    return input_ids_m2m, attention_mask_m2m

def bert_t5_input_features(input_texts_m2m, tokenizer_m2m, max_seq_len):
    encoding_m2m = tokenizer_m2m(input_texts_m2m,
                                 padding='longest',
                                 max_length=max_seq_len,
                                 truncation=True,
                                 add_special_tokens=True,
                                 return_tensors="pt")
    input_ids_m2m = encoding_m2m.input_ids.cuda()
    attention_mask_m2m = encoding_m2m.attention_mask.cuda()
    return input_ids_m2m, attention_mask_m2m

def llm_input_features(input_texts_llm, tokenizer_llm,
                         max_seq_len, add_bos_token, add_eos_token):
    tokenizer_llm.add_bos_token = add_bos_token
    tokenizer_llm.add_eos_token = add_eos_token
    encoding_llm = tokenizer_llm(input_texts_llm,
                         padding='longest',
                         max_length=max_seq_len,
                         truncation=True,
                         return_tensors="pt")
    input_ids_llm = encoding_llm.input_ids.cuda()
    attention_mask_llm = encoding_llm.attention_mask.cuda()
    return input_ids_llm, attention_mask_llm
