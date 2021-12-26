import csv
from typing import Dict, Union, List


def restore_from_temp(temp_fname, idxs=None):
    query = None
    urls, docs, exps, labels = [], [], [], []
    with open(temp_fname, 'r', newline='') as fin:
        reader = csv.reader(fin)
        for idx, (query, url, evidence, label) in enumerate(reader):
            if idxs is None or idx in idxs:
                evidence = eval(evidence)
                doc, exp = zip(*evidence)
                urls.append(url)
                docs.append([doc])
                exps.append(exp)
                labels.append(label)
    return query, urls, docs, exps, labels


def dump_quel(fname, query, urls, docs, exps, labels, mode='a+'):
    with open(fname, mode, newline='') as fout:
        for url, doc, exp, label in zip(urls, docs, exps, labels):
            #             print(doc, exp)
            assert len(doc[0]) == len(exp)
            writer = csv.writer(fout)
            writer.writerow([query, url, list(zip(doc[0], exp)), label])


def highlight_exp_pred(exp, doc, highlight='yellow', shorten=True):
    ret = ''
    abrcount = 0
    abrflag = False  # for abbreviation
    for e, w in zip(exp, doc[0]):
        if e == 1:
            if highlight == 'bold':
                ret += f'<b class="token">{w}&nbsp;</b>'
            else:
                ret += f'<span style="background-color:#FFFF00; float: left">{w}&nbsp;</span>'
            abrcount = 0
            abrflag = False
        else:
            if abrflag:
                continue
            abrcount += 1
            if abrcount > 4 and shorten:
                abrflag = True
                ret += f'<span class=“token”>...&nbsp;</span>'
            else:
                ret += f'<span class="token">{w}&nbsp;</span>'
    return ret


def color_cls_pred(c,
                   pos_label='SUPPORTS', pos_color='green',
                   neg_label='REFUTES', neg_color='red',
                   default_color='gray'):
    color = default_color
    if c == pos_label:
        color = pos_color
    elif c == neg_label:
        color = neg_color
    return f'<p style="color:{color};">{c}</p>'


def machine_rationale_mask_to_html(cls_preds, exp_preds, docs_clean, urls):
    cls_strs = [color_cls_pred(c) for c in cls_preds]
    evi_strs = [highlight_exp_pred(exp, doc) for exp, doc in zip(exp_preds, docs_clean)]
    urls = [url.split('/')[-1] for url in urls]
    pred = {
        'clses': cls_strs,
        'evis': evi_strs,
        'links': urls
    }
    return pred


def parse_sentence(raw_orig_doc_html, ret_custom_mask=False)->Dict[str, Union[str, List[int]]]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(raw_orig_doc_html, 'html.parser')
    ret = {'doc': [],
           'mask': []}
    for span in soup.find_all('span'):
        ret['doc'].append(span.string)
        if ret_custom_mask:
            ret['mask'].append(1 if 'ann-pos' in span['class'] else 0)
    ret['doc'] = ' '.join(ret['doc'])
    return ret


def random_select_data(data, docs):
    # k = random.choice(list(data.keys()))
    k = 'negR_260.txt'
    selected_ann = data[k]
    # print("key: ", k)
    ann_id = selected_ann.ann_id
    docid = selected_ann.docid
    doc = docs[docid]
    query = selected_ann.query
    return ann_id, query, doc