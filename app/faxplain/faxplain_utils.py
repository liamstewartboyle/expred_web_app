import csv


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