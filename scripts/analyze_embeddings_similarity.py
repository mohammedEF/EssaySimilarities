import torch
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import pandas as pd
import pickle

def generate_mpnet_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generate MPNet embeddings for a list of texts using all-mpnet-base-v2.

    Args:
        texts: List of text strings

    Returns:
        Array of document embeddings
    """
    print("Loading MPNet model (all-mpnet-base-v2)...")
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def analyze_embedding_similarity(df):
    """
    Analyze embedding-based similarity within and across time periods.

    Returns:
        Dictionary of embedding-based metrics
    """
    if df is None:
        raise ValueError("No data loaded. Call load_data() first.")

    # Split essays by time period
    pre_gpt_essays = df[df['time_period'] == 'pre_gpt']['essay_text'].tolist()
    post_gpt_essays = df[df['time_period'] == 'post_gpt']['essay_text'].tolist()

    # Generate embeddings
    print("Generating MPNet embeddings for pre-GPT essays...")
    pre_gpt_embeddings = generate_mpnet_embeddings(pre_gpt_essays)

    print("Generating MPNet embeddings for post-GPT essays...")
    post_gpt_embeddings = generate_mpnet_embeddings(post_gpt_essays)

    # Calculate similarity within groups
    pre_sim_matrix = cosine_similarity(pre_gpt_embeddings)
    post_sim_matrix = cosine_similarity(post_gpt_embeddings)

    # Calculate mean similarity within each group (excluding self-similarity)
    pre_mean_sim = (np.sum(pre_sim_matrix) - pre_sim_matrix.shape[0]) / (pre_sim_matrix.size - pre_sim_matrix.shape[0])
    post_mean_sim = (np.sum(post_sim_matrix) - post_sim_matrix.shape[0]) / (post_sim_matrix.size - post_sim_matrix.shape[0])

    # Calculate cross-group similarity
    cross_sim_matrix = cosine_similarity(pre_gpt_embeddings, post_gpt_embeddings)
    cross_mean_sim = np.mean(cross_sim_matrix)

    results = {
        'pre_gpt_internal_similarity': pre_mean_sim,
        'post_gpt_internal_similarity': post_mean_sim,
        'cross_period_similarity': cross_mean_sim,
        'pre_gpt_embeddings': pre_gpt_embeddings,
        'post_gpt_embeddings': post_gpt_embeddings
    }

    return results


df = pd.read_csv("C:/Users/PRECISION 5550/Desktop/Essays Project/dataset/combined_processed.csv")
embeddings_similarity = analyze_embedding_similarity(df)
print('done analysis .... ')




# statistical test ____________________________________________________________________________________________________________________________________________

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle 

# --- Helper Function to Extract Unique Similarities ---
def get_unique_pairwise_similarities(similarity_matrix):
    """
    Extracts the unique pairwise similarity scores from the upper triangle
    of a similarity matrix (excluding the diagonal).
    """
    # Ensure it's a square matrix
    assert similarity_matrix.shape[0] == similarity_matrix.shape[1], "Matrix must be square"
    # Get indices of the upper triangle, excluding the diagonal (k=1)
    upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
    # Extract the values at these indices
    unique_similarities = similarity_matrix[upper_triangle_indices]
    return unique_similarities

# --- Permutation Test Function ---
def permutation_test_similarity(embeddings1, embeddings2, n_permutations=10000):
    """
    Performs a permutation test to compare the mean internal cosine similarity
    between two groups of embeddings.

    Args:
        embeddings1 (np.ndarray): Embeddings for the first group.
        embeddings2 (np.ndarray): Embeddings for the second group.
        n_permutations (int): Number of permutations to perform.

    Returns:
        tuple: (observed_difference, p_value)
    """
    # 1. Calculate similarity matrices
    sim_matrix1 = cosine_similarity(embeddings1)
    sim_matrix2 = cosine_similarity(embeddings2)

    # 2. Extract unique pairwise similarities for each group
    similarities1 = get_unique_pairwise_similarities(sim_matrix1)
    similarities2 = get_unique_pairwise_similarities(sim_matrix2)

    # Handle cases with insufficient data for pairwise comparison
    if len(similarities1) < 1 or len(similarities2) < 1:
        print("Warning: Not enough data points in one or both groups for pairwise similarity comparison.")
        # Return NaN or raise an error depending on desired behavior
        # Returning NaN here to indicate the test couldn't be performed
        return np.nan, np.nan

    # 3. Calculate the observed difference in means
    mean_sim1 = np.mean(similarities1)
    mean_sim2 = np.mean(similarities2)
    observed_difference = mean_sim2 - mean_sim1 # post_gpt - pre_gpt

    # 4. Pool all similarity scores
    pooled_similarities = np.concatenate([similarities1, similarities2])
    n1 = len(similarities1) # Number of unique pairs in group 1
    # n2 = len(similarities2) # Not strictly needed, it's total - n1

    # 5. Perform permutations
    permuted_differences = np.zeros(n_permutations)
    for i in range(n_permutations):
        # Shuffle the pooled scores
        np.random.shuffle(pooled_similarities)
        # Split into two permuted groups
        permuted_group1 = pooled_similarities[:n1]
        permuted_group2 = pooled_similarities[n1:]
        # Calculate the difference in means for this permutation
        permuted_differences[i] = np.mean(permuted_group2) - np.mean(permuted_group1)

    # 6. Calculate the p-value
    # Count how many permuted differences are >= the observed difference
    # (for a one-tailed test checking if group 2 > group 1)
    # If testing for any difference (two-tailed), use abs()
    p_value_one_tailed = np.sum(permuted_differences >= observed_difference) / n_permutations
    p_value_two_tailed = np.sum(np.abs(permuted_differences) >= np.abs(observed_difference)) / n_permutations

    print(f"Observed Mean Similarity (Pre-GPT): {mean_sim1:.4f}")
    print(f"Observed Mean Similarity (Post-GPT): {mean_sim2:.4f}")
    print(f"Observed Difference (Post - Pre): {observed_difference:.4f}")
    
    # two tail test - ANY difference
    print(f"P-value (two-tailed): {p_value_two_tailed:.4f}")
    
    return observed_difference, p_value_two_tailed # Returning two-tailed p-value

# Check if the necessary keys exist before running the test
if 'embeddings_similarity' in locals() and \
   'pre_gpt_embeddings' in embeddings_similarity and \
   'post_gpt_embeddings' in embeddings_similarity:

    print("Running permutation test...")
    # Extract embeddings
    pre_embeddings = np.array(embeddings_similarity['pre_gpt_embeddings'])
    post_embeddings = np.array(embeddings_similarity['post_gpt_embeddings'])

    # Ensure there are enough embeddings in each group to calculate pairwise similarity
    if pre_embeddings.shape[0] >= 2 and post_embeddings.shape[0] >= 2:
        # Run the test
        observed_diff, p_val = permutation_test_similarity(pre_embeddings, post_embeddings)

        # Interpret the result
        alpha = 0.05 # Significance level
        if not np.isnan(p_val): # Check if test ran successfully
             if p_val < alpha:
                 print(f"\nThe difference in internal similarity is statistically significant (p={p_val:.4f}).")
             else:
                 print(f"\nThe difference in internal similarity is not statistically significant (p={p_val:.4f}).")
    else:
        print("\nError: Need at least two documents in each time period to perform the similarity comparison.")

else:
    print("Error: 'embeddings_similarity' variable not found or missing required embedding keys.")
    print("Please ensure the variable is loaded correctly and contains 'pre_gpt_embeddings' and 'post_gpt_embeddings'.")



# visualization ________________________________________________________________________________________________________________________________________________________

import matplotlib.pyplot as plt
import seaborn as sns

similarity = {
    "pre_gpt_internal": embeddings_similarity["pre_gpt_internal_similarity"],
    "post_gpt_internal": embeddings_similarity["post_gpt_internal_similarity"],
    "cross_period": embeddings_similarity["cross_period_similarity"]
}

# Plot
plt.figure(figsize=(8, 5))
sns.barplot(x=list(similarity.keys()), 
            y=list(similarity.values()),
            palette="viridis")
plt.title("Embeddings Similarity Comparison")
plt.ylabel("Cosine Similarity")
plt.ylim(0, 0.5)
plt.show()


#_________________________________________________________________________________________________________________________________________

import numpy as np

# Create a similarity matrix
labels = ['Pre-GPT', 'Post-GPT']
similarity_matrix = np.array([
    [embeddings_similarity["pre_gpt_internal_similarity"], embeddings_similarity["cross_period_similarity"]],
    [embeddings_similarity["cross_period_similarity"], embeddings_similarity["post_gpt_internal_similarity"]]
])

# Plot
plt.figure(figsize=(6, 6))
sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
            xticklabels=labels, yticklabels=labels)
plt.title("Similarity Matrix (Pre-GPT vs. Post-GPT)")
plt.show()


#_________________________________________________________________________________________________________________________________________


from sklearn.decomposition import PCA

# Assuming you have pre_gpt_embeddings and post_gpt_embeddings arrays
pre_gpt_embeddings = np.array(embeddings_similarity["pre_gpt_embeddings"])
post_gpt_embeddings = np.array(embeddings_similarity["post_gpt_embeddings"])  # Hypothetical

# Combine and project
all_embeddings = np.vstack([pre_gpt_embeddings, post_gpt_embeddings])
pca = PCA(n_components=2)
projected = pca.fit_transform(all_embeddings)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(projected[:len(pre_gpt_embeddings), 0], projected[:len(pre_gpt_embeddings), 1], 
            label="Pre-GPT", alpha=0.6)
plt.scatter(projected[len(pre_gpt_embeddings):, 0], projected[len(pre_gpt_embeddings):, 1], 
            label="Post-GPT", alpha=0.6)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA Projection of Embeddings")
plt.legend()
plt.show()


#_________________________________________________________________________________________________________________________________________



# Data
pre_post = ["Pre-GPT", "Post-GPT"]
internal_similarities = [similarity["pre_gpt_internal"], 
                         similarity["post_gpt_internal"]]

# Plot
plt.figure(figsize=(6, 4))
plt.plot(internal_similarities, pre_post, marker='o', linestyle='--', color='gray')
plt.title("Internal Similarity: Pre-GPT vs. Post-GPT")
plt.xlabel("Cosine Similarity")
plt.xlim(0, 0.5)
plt.show()