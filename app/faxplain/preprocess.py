import torch


def expred_inputs_preprocess(queries, masked_docs, tokenizer, max_length=512):
    cls_token = torch.LongTensor([tokenizer.cls_token_id])
    sep_token = torch.LongTensor([tokenizer.sep_token_id])
    # print(queries, masked_docs)
    inputs = []
    attention_masks = []
    for query, mdoc in zip(queries, masked_docs):
        d = torch.cat((cls_token, query, sep_token, mdoc), dim=-1)
        if len(d) > max_length:
            inputs.append(d[:max_length])
            attention_masks.append(torch.ones(max_length).type(torch.float))
        else:
            pad = torch.zeros(max_length - len(d))
            inputs.append(torch.cat((d, pad)))
            attention_masks.append(torch.cat((torch.ones_like(d), pad)).type(torch.float))
    if isinstance(inputs, list):
        return torch.vstack(inputs).type(torch.long), torch.vstack(attention_masks)
    return torch.LongTensor(inputs), torch.FloatTensor(attention_masks)


def clean(query):
    query = query.strip().lower()
    return query