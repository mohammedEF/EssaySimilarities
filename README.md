# Quantifying changes in writing style: Pre-GPT vs Post-GPT Essays

This project analyzes the changes in writing style of student essays before and after the introduction of GPT-based tools. It uses **TF-IDF vectorization** and **transformer-based embeddings** to quantify and compare the similarities between essays. Statistical tests (t-test and Mann-Whitney U test) are performed to validate the significance of the observed changes.

# Key Features

**Preprocessing:**

Essays are preprocessed by lowercasing, removing punctuation, tokenizing, removing stopwords, and lemmatizing.

Length normalization is applied to ensure consistent essay lengths.

**Similarity Analysis:**

TF-IDF Vectorization: Measures surface-level similarities (word choice, structure).

Transformer Embeddings: Measures semantic similarities (meaning, themes).

Cosine similarity is used to compute pairwise similarities between essays.

**Statistical Validation:**

Independent t-test: Compares the means of similarity scores.

Mann-Whitney U test: A non-parametric test to compare distributions of similarity scores.

**Visualization:**

Heatmaps for similarity matrices.

Bar plots for average similarity scores.
