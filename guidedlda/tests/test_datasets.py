# coding=utf-8
from __future__ import absolute_import, unicode_literals  # noqa

import unittest

import guidedlda.datasets


class TestDatasets(unittest.TestCase):

    def test_datasets(self):
        X = guidedlda.datasets.load_data(guidedlda.datasets.REUTERS)
        self.assertEqual(X.shape, (395, 4258))
        titles = guidedlda.datasets.load_titles(guidedlda.datasets.REUTERS)
        self.assertEqual(len(titles), X.shape[0])
        vocab = guidedlda.datasets.load_vocab(guidedlda.datasets.REUTERS)
        self.assertEqual(len(vocab), X.shape[1])

        X = guidedlda.datasets.load_data(guidedlda.datasets.NYT)
        self.assertEqual(X.shape, (8447, 3012))
        titles = guidedlda.datasets.load_titles(guidedlda.datasets.NYT)
        self.assertEqual(len(titles), X.shape[0])
        vocab = guidedlda.datasets.load_vocab(guidedlda.datasets.NYT)
        self.assertEqual(len(vocab), X.shape[1])
