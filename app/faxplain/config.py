import os
import random
from itertools import chain

import torch

from expred.expred.eraser_utils import load_eraser_data, load_documents

application = 'counterfactual'


if application == 'faxplain':
    evi_finder_loc = './trained_models/fever/evidence_token_identifier.pt'
    evi_finder_url = 'https://www.dropbox.com/s/qwinyap4kbxzdvn/evidence_token_identifier.pt?dl=1'
    cls_loc = 'trained_models/fever/evidence_classifier.pt'
    cls_url = 'https://www.dropbox.com/s/oc3qrgl0tqn9mqd/evidence_classifier.pt?dl=1'
    class_names = ["SUPPORTS", "REFUTES"]
elif application == 'counterfactual':
    evi_finder_loc = './trained_models/movies/evidence_token_identifier.pt'
    evi_finder_url = 'https://www.dropbox.com/s/qen0vx2uz6ksn3m/evidence_token_identifier.pt?dl=1'
    cls_loc = 'trained_models/movies/evidence_classifier.pt'
    cls_url = 'https://www.dropbox.com/s/0sfrdykcg6cf6kh/evidence_classifier.pt?dl=1'
    class_names = ["NEG", "POS"]
    dataset_dir = os.environ.get('HOME') + '/.keras/datasets/movies'
    dataset_url = 'http://www.eraserbenchmark.com/zipped/movies.tar.gz'
    if not os.path.isfile(dataset_dir + '/test.jsonl'):
        dataset_base_dir = os.environ.get('HOME') + '/.keras/datasets/'
        os.makedirs(dataset_base_dir, exist_ok=True)
        dataset_tarfilename = dataset_base_dir + 'movies.tar.gz'
        import urllib
        urllib.request.urlretrieve(dataset_url, dataset_tarfilename)
        import shutil
        shutil.unpack_archive(dataset_tarfilename, dataset_base_dir, 'gztar')
    data = {x.ann_id: x for x in chain.from_iterable(load_eraser_data(dataset_dir, merge_evidences=True))}
    docs = load_documents(dataset_dir)

    session_id = hex(int(random.random() * 1e13))[2:]

    if os.path.isdir('data'):
        print('data folder exists')
    else:
        print('creating data folder')
        os.mkdir('data')

    ugc_data_fname = f'data/ugc_{session_id}.csv'  # user generated content
    mgc_data_fname = f'data/mgc_{session_id}.csv'  # machine genarated content
    temp_data_fname = f'data/temp_{session_id}.pkl'
    temp_fname = f'data/temp_{session_id}.txt'
    bert_dir = 'bert-base-uncased'

    device = torch.device('cpu')
    default_ndocs = 3
    max_sentence = 30
    n_word_replace = 5
    debug = True
    debug = False
    top_pos = 10