import numpy as np
import spacy
import pickle
try:
    nlp = spacy.load('en_core_web_lg')
except:
    print("Installing spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_lg"])
    nlp = spacy.load('en_core_web_lg')

def analyze_discourse_features(preprocessed):
    """
    Analyze discourse and cohesion features for each time period.
    
    Args:
        preprocessed: Dictionary of preprocessed essays by time period
        
    Returns:
        Dictionary of discourse metrics by time period
    """
    # Define cohesion markers
    transition_markers = [
        'however', 'therefore', 'thus', 'hence', 'consequently', 'furthermore',
        'moreover', 'nevertheless', 'in addition', 'on the other hand', 'in contrast',
        'for example', 'for instance', 'specifically', 'in particular', 'in conclusion'
    ]
    
    results = {period: {} for period in preprocessed.keys()}
    
    for period, essays in preprocessed.items():
        marker_counts = []
        sent_similarity_scores = []
        
        for essay in essays:
            text = essay['raw'].lower()
            sentences = essay['sentences']
            
            # Count transition markers
            marker_count = sum(1 for marker in transition_markers if marker in text)
            marker_density = marker_count / len(sentences) if sentences else 0
            marker_counts.append(marker_density)
            
            # Calculate adjacent sentence similarity (if enough sentences)
            if len(sentences) >= 3:
                doc_sentences = [nlp(sent) for sent in sentences]
                sim_scores = []
                for i in range(len(doc_sentences) - 1):
                    sim = doc_sentences[i].similarity(doc_sentences[i + 1])
                    sim_scores.append(sim)
                sent_similarity_scores.append(np.mean(sim_scores))
        
        # Aggregate results
        results[period]['marker_density_mean'] = np.mean(marker_counts) if marker_counts else 0
        results[period]['marker_density_std'] = np.std(marker_counts) if marker_counts else 0
        results[period]['marker_counts'] = marker_counts if marker_counts else 0
        results[period]['sent_similarity_mean'] = np.mean(sent_similarity_scores) if sent_similarity_scores else 0
        results[period]['sent_similarity_std'] = np.std(sent_similarity_scores) if sent_similarity_scores else 0
        results[period]['sent_similarity_scores'] = sent_similarity_scores if sent_similarity_scores else 0
        
    return results

with open("C:/Users/PRECISION 5550/Desktop/Essays Project/dataset/preprocessed.pkl", "rb") as f:
    preprocessed = pickle.load(f)

results = analyze_discourse_features(preprocessed)

# visualization ______________________________________________________________________________________________________________________________________

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create 2x2 matrices for each category
def create_matrix(category):
    return np.array([
        [results[category]['marker_density_mean'], results[category]['marker_density_std']],
        [results[category]['sent_similarity_mean'], results[category]['sent_similarity_std']]
    ])

pre_matrix = create_matrix('pre_gpt')
post_matrix = create_matrix('post_gpt')

# Plotting function
def plot_matrix(matrix, title, ax):
    sns.heatmap(matrix, 
                annot=True, 
                fmt=".3f",
                cmap='Blues',
                cbar=False,
                ax=ax,
                annot_kws={'size': 12})
    ax.set_xticklabels(['Mean', 'Std Dev'])
    ax.set_yticklabels(['Marker Density', 'Sentence Similarity'], rotation=0)
    ax.set_title(title, pad=20, fontsize=14)

# Create separate figures
fig1, ax1 = plt.subplots(figsize=(6, 4))
plot_matrix(pre_matrix, 'Pre-GPT Analysis Matrix', ax1)
plt.tight_layout()

fig2, ax2 = plt.subplots(figsize=(6, 4))
plot_matrix(post_matrix, 'Post-GPT Analysis Matrix', ax2)
plt.tight_layout()

plt.show()

print('done analysis ....')

# statistical validation ___________________________________________________________________________________________________________________________


import numpy as np
from scipy import stats

pre_marker = results['pre_gpt']['marker_counts']
post_marker = results['post_gpt']['marker_counts']
pre_sent = results['pre_gpt']['sent_similarity_scores']
post_sent = results['post_gpt']['sent_similarity_scores']


def check_normality(data, name):
    """Check normality using Shapiro-Wilk test."""
    stat, p = stats.shapiro(data)
    print(f"Shapiro-Wilk test for {name}: p = {p:.4f}")
    return p > 0.05  # Return True if normal

def perform_independent_test(data1, data2, name):
    """Perform appropriate test based on normality."""
    normal1 = check_normality(data1, f"{name} Group 1")
    normal2 = check_normality(data2, f"{name} Group 2")
    
    if normal1 and normal2:
        print(f"Both groups normal. Using independent t-test.")
        stat, p = stats.ttest_ind(data1, data2)
        test_name = "t-test"
    else:
        print(f"Non-normal data. Using Mann-Whitney U test.")
        stat, p = stats.mannwhitneyu(data1, data2)
        test_name = "Mann-Whitney U"
    
    print(f"{test_name} statistic: {stat:.4f}, p-value: {p:.4f}\n")
    return p

# Perform tests
print("=== Marker Density ===")
marker_p = perform_independent_test(pre_marker, post_marker, "Marker Density")

print("=== Sentence Similarity ===")
sent_p = perform_independent_test(pre_sent, post_sent, "Sentence Similarity")

# Interpret results
alpha = 0.05
print("\n--- Conclusion ---")
print(f"Marker Density: {'Significant' if marker_p < alpha else 'No difference'} (p={marker_p:.4f})")
print(f"Sentence Similarity: {'Significant' if sent_p < alpha else 'No difference'} (p={sent_p:.4f})")