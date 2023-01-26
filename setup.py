import setuptools

setuptools.setup(
    name='TF-IDF Transformer',
    version='1.0.0',
    author='M Yasin Yıldırım',
    author_email='myasiny@gmail.com',
    description='Framework that does TF-IDF transformation.',
    packages=['tfidf_transformer'],
    requires=['numpy', 'pandas', 'scipy', 'scikit_learn']
)
