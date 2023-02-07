# -*- coding: utf-8 -*-
# Author: M Yasin Yıldırım <myasiny@gmail.com>

"""
TF-IDF Transformer
==================

Provides a framework that does TF-IDF transformation.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import List, Tuple
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class Tfidf(TfidfTransformer):
    """
    Converts a collection of text documents into TF-IDF representation.

    TF-IDF means term-frequency times inverse document-frequency.

    Example
    -------
    >>> from Term_Frequency_Inverse_Document_Frequency_Transformer import tfidf_transformer
    >>> tfidf_transformer.Tfidf().apply_transformation(data=['A sentence.', 'Another sentence.'])
              TF-IDF
    sentence     1.0
    another      0.0
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def __vectorize(data: List[str]) -> Tuple[np.ndarray, sp.csr_matrix]:
        """
        Learns the vocabulary dictionary and builds a matrix of token counts.
        :param: data: List of strings.
        :return: Transformed feature names and document-term matrix.
        """

        count_vectorizer = CountVectorizer()
        word_counts = count_vectorizer.fit_transform(data)
        return count_vectorizer.get_feature_names_out(), word_counts

    def __fit_transform(self, word_counts: sp.csr_matrix) -> sp.csr_matrix:
        """
        Calls `TransformerMixin.fit_transform` method by inheritance.
        :param: word_counts: Document-term matrix.
        :return: Transformed array.
        """

        return self.fit_transform(word_counts)

    def apply_transformation(self, data: List[str], column_name: str = 'TF-IDF') -> pd.DataFrame:
        """
        Applies TF-IDF transformation to given set of text data.
        :param: data: List of strings.
        :param: column_name: Column name for TF-IDF scores.
        :return: Dataframe containing words and TF-IDF scores.
        """

        words, word_counts = self.__vectorize(data)
        transformed_data = self.__fit_transform(word_counts)
        return pd.DataFrame(
            transformed_data[0].T.todense(),
            index=words,
            columns=[column_name]
        ).sort_values(column_name, ascending=False)
