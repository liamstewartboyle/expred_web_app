from unittest import TestCase

from expred.expred.tokenizer import BertTokenizerWithSpans


class TestBertTokenizerWithSpans(TestCase):
    tokenizer = BertTokenizerWithSpans.from_pretrained('bert-base-uncased')
    input_doc = 'i would say that i feel this is essentially a joyful movie .'.split()

    def test_encode_doc_with_spans(self):
        encoded_should = [1045, 2052, 2360, 2008, 1045, 2514, 2023, 2003, 7687, 1037, 6569, 3993, 3185, 1012]
        span_should = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
                       (10, 12), (12, 13), (13, 14)]
        encoded, span = self.tokenizer.encode_doc_with_spans(self.input_doc)
        self.assertSequenceEqual(encoded, encoded_should)
        self.assertSequenceEqual(span, span_should)
