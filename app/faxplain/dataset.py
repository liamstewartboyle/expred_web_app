import os
from itertools import chain
from typing import List, Tuple

from transformers import BasicTokenizer
import random
from counter_assist import CounterfactualInput
from expred.expred.eraser_utils import (load_documents, load_eraser_data)


def maybe_download_dataset(dataset_name, dataset_base_dir):
    if not os.path.isfile(f"{dataset_base_dir}/{dataset_name}/test.jsonl"):
        dataset_url = f'http://www.eraserbenchmark.com/zipped/{dataset_name}.tar.gz'
        import shutil
        import urllib
        dataset_tarfilename = '{dataset_base_dir}/{dataset_name}.tar.gz'
        os.makedirs(dataset_base_dir, exist_ok=True)
        urllib.request.urlretrieve(dataset_url, dataset_tarfilename)
        shutil.unpack_archive(dataset_tarfilename, dataset_base_dir, 'gztar')


class Dataset():
    def __init__(self, dataset_name, dataset_base_dir) -> None:
        maybe_download_dataset(dataset_name, dataset_base_dir)
        dataset_dir = f'{dataset_base_dir}/{dataset_name}'
        _raw_data = load_eraser_data(dataset_dir, merge_evidences=True)
        self.raw_data = {x.ann_id: x for x in chain.from_iterable(_raw_data)}
        self.docs = load_documents(dataset_dir)

    def random_select_data(self, basic_tokenizer: BasicTokenizer) -> Tuple[str, List[str], List[List[str]], str]:
        # ann_id = random.choice(list(self.raw_data.keys()))
        ann_id = 'posR_018.txt'
        selected_ann = self.raw_data[ann_id]
        docid = selected_ann.docid

        query = basic_tokenizer.tokenize(selected_ann.query)
        doc = self.docs[docid]
        label = self.raw_data[ann_id].label

        return ann_id, query, doc, label
