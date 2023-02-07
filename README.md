# TF-IDF Transformer

It is a Python package providing a framework that does TF-IDF transformation. 

### Dependencies
- `numpy`
- `pandas`
- `scipy`
- `scikit_learn`

### Install
```
pip install -i https://test.pypi.org/simple/ Term-Frequency-Inverse-Document-Frequency-Transformer==1.0.0
```

### Usage
```
from Term_Frequency_Inverse_Document_Frequency_Transformer import tfidf_transformer

sample_data = [
    'I enjoy reading about Machine Learning and Machine Learning is my PhD subject.',
    'I would enjoy a walk in the park.',
    'I was reading in the library.'
]
result_df = tfidf_transformer.Tfidf().apply_transformation(data=sample_data)
```

### Contact
Reach me out by `myasiny@gmail.com` for any question or suggestion.