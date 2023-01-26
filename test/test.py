import unittest

from tfidf_transformer import Tfidf


class TestTfidfTransformation(unittest.TestCase):
    def setUp(self):
        # GIVEN a list of sentences
        self.data = [
            'I enjoy reading about Machine Learning and Machine Learning is my PhD subject.',
            'I would enjoy a walk in the park.',
            'I was reading in the library.'
        ]

    def test_tfidf_transformation_result(self):
        # WHEN TF-IDF transformation is applied
        df = Tfidf().apply_transformation(data=self.data)

        # THEN expected word and TF-IDF score are returned
        self.assertEqual('machine', df.index[0])
        self.assertEqual(0.5137197367065607, df.iloc[0][0])
