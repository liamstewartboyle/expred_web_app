from unittest import TestCase

import torch

from sparcassist import HotflipCounterAssist


class TestHotFlip(TestCase):
    word_scores = torch.tensor([[[1, 2, 3, 4],
                                 [2, 3, 4, 5],
                                 [9, 8, 7, 6]]], dtype=torch.float)
    candidate_pos = torch.tensor([[0, 2]], dtype=torch.int64)

    def test_select_top_candidate_words_legacy(self):
        ret_should = torch.tensor([[3, 0]], dtype=torch.long)
        ret = HotflipCounterAssist._select_top_candidate_words_legacy(self.word_scores, self.candidate_pos)
        self.assertTrue(torch.equal(ret, ret_should))

    def test_select_candidate_words(self):
        ret_should = torch.tensor([[[3, 2], [0, 1]]], dtype=torch.long)
        top_k = 2
        ret = HotflipCounterAssist._select_candidate_words(self.word_scores, self.candidate_pos, top_k)
        self.assertTrue(torch.equal(ret, ret_should))
