from __future__ import absolute_import, unicode_literals  # noqa

import os

import guidedlda.utils


_test_dir = os.path.join(os.path.dirname(__file__), 'tests')

REUTERS = 'reuters'

# NYT data source https://github.com/moorissa/nmf_nyt
NYT = 'nyt'


def load_data(dataset_name):
    dataset_ldac_fn = os.path.join(_test_dir, dataset_name + '.ldac')
    return guidedlda.utils.ldac2dtm(open(dataset_ldac_fn), offset=0)


def load_vocab(dataset_name):
    vocab_fn = os.path.join(_test_dir, dataset_name + '.tokens')
    with open(vocab_fn) as f:
        vocab = tuple(f.read().split())
    return vocab


def load_titles(dataset_name):
    titles_fn = os.path.join(_test_dir, dataset_name + '.titles')
    with open(titles_fn) as f:
        titles = tuple(line.strip() for line in f.readlines())
    return titles
