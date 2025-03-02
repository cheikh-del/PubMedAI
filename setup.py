from setuptools import setup, find_packages

setup(
    name="pubmed_pipeline",
    version="0.1",
    author="Cheikh Omar Ba",
    author_email="cheikh.omar.ba@aims-senegal.org",
    description="Pipeline for PubMed article collection, entity extraction, embeddings, cooccurrences, and N-grams.",
    packages=find_packages(),
    install_requires=[
        "biopython",
        "pandas",
        "numpy",
        "tqdm",
        "torch",
        "spacy",
        "scispacy",
        "transformers",
        "sklearn",
        "joblib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
