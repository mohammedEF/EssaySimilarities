import numpy as np 
from collections import Counter
import pickle
import matplotlib.pyplot as plt

def analyze_syntactic_features(preprocessed):
    """
    Analyze syntactic features for each time period.
    
    Args:
        preprocessed: Dictionary of preprocessed essays by time period
        
    Returns:
        Dictionary of syntactic metrics by time period
    """
    results = {period: {} for period in preprocessed.keys()}
    
    for period, essays in preprocessed.items():
        # Essay-level metrics
        sent_lengths = []
        clauses_per_sent = []
        subordination_ratios = []
        pos_distributions = []
        
        for essay in essays:
            doc = essay['spacy_doc']
            sentences = list(doc.sents)
            
            # Calculate sentence length
            if sentences:
                sent_lens = [len(sent) for sent in sentences]
                sent_lengths.extend(sent_lens)
            
            # Estimate clausal density using verb counts as proxy
            for sent in sentences:
                verbs = [token for token in sent if token.pos_ in ("VERB", "AUX")]
                if len(sent) > 0:
                    clauses_per_sent.append(len(verbs) / 1.0)
            
            # Calculate subordination vs. coordination
            subordinating_conj = len([token for token in doc if token.dep_ == "mark"])
            coordinating_conj = len([token for token in doc if token.dep_ == "cc"])
            if coordinating_conj > 0:
                subordination_ratios.append(subordinating_conj / (subordinating_conj + coordinating_conj))
            
            # POS tag distribution
            pos_counts = Counter([token.pos_ for token in doc])
            total_tokens = len(doc)
            pos_dist = {pos: count/total_tokens for pos, count in pos_counts.items()}
            pos_distributions.append(pos_dist)
        
        # Aggregate results
        results[period]['mean_sentence_length'] = np.mean(sent_lengths) if sent_lengths else 0
        results[period]['std_sentence_length'] = np.std(sent_lengths) if sent_lengths else 0
        results[period]['sent_lengths'] = sent_lengths if sent_lengths else 0
        results[period]['mean_clauses_per_sent'] = np.mean(clauses_per_sent) if clauses_per_sent else 0
        results[period]['clauses_per_sent'] = clauses_per_sent if clauses_per_sent else 0
        results[period]['mean_subordination_ratio'] = np.mean(subordination_ratios) if subordination_ratios else 0
        results[period]['subordination_ratios'] = subordination_ratios if subordination_ratios else 0

        
        # Aggregate POS distributions
        all_pos = {}
        for dist in pos_distributions:
            for pos, freq in dist.items():
                if pos not in all_pos:
                    all_pos[pos] = []
                all_pos[pos].append(freq)
        
        pos_means = {pos: np.mean(freqs) for pos, freqs in all_pos.items()}
        results[period]['pos_distribution'] = pos_means
        
    return results


with open("C:/Users/PRECISION 5550/Desktop/Essays Project/dataset/preprocessed.pkl", "rb") as f:
    preprocessed = pickle.load(f)

results = analyze_syntactic_features(preprocessed)

# print(results)


# visualization _____________________________________________________________________________________________________________________________________________________________________

import matplotlib.pyplot as plt
import numpy as np

# Extract data from results dictionary
def convert_np_float(value):
    """Convert numpy float to native float for matplotlib compatibility"""
    return float(value)

# Sentence Complexity Metrics
def plot_sentence_complexity():
    metrics = ['mean_sentence_length', 'mean_clauses_per_sent']
    labels = ['Mean Sentence Length', 'Mean Clauses per Sentence']
    
    for metric, label in zip(metrics, labels):
        plt.figure(figsize=(10, 6))
        pre = convert_np_float(results['pre_gpt'][metric])
        post = convert_np_float(results['post_gpt'][metric])
        
        plt.bar(['Pre-GPT', 'Post-GPT'], [pre, post], color=['#1f77b4', '#ff7f0e'])
        plt.ylabel('Count')
        plt.title(label)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

# Subordination Ratio
def plot_subordination():
    plt.figure(figsize=(10, 6))
    pre = convert_np_float(results['pre_gpt']['mean_subordination_ratio'])
    post = convert_np_float(results['post_gpt']['mean_subordination_ratio'])
    
    plt.bar(['Pre-GPT', 'Post-GPT'], [pre, post], color=['#1f77b4', '#ff7f0e'])
    plt.ylabel('Ratio')
    plt.title('Mean Subordination Ratio')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# POS Distribution
def plot_pos_distribution(pre_post):
    plt.figure(figsize=(12, 8))
    pos_data = results[f'{pre_post}_gpt']['pos_distribution']
    tags = sorted(pos_data.keys())
    values = [convert_np_float(pos_data[tag]) for tag in tags]
    
    plt.barh(tags, values, color='#2ca02c')
    plt.xlabel('Proportion')
    plt.title(f'POS Distribution ({pre_post.capitalize()}-GPT)')
    plt.gca().invert_yaxis()  # Highest values at top
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Generate all plots
plot_sentence_complexity()
plot_subordination()
plot_pos_distribution('pre')
plot_pos_distribution('post')



# statistical validation___________________________________________________________________________________________________________________________________


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def perform_statistical_analysis(results):
    metrics = ['sent_lengths', 'clauses_per_sent', 'subordination_ratios']
    
    for metric in metrics:
        print(f"Analyzing {metric.replace('_', ' ').title()}...")
        
        # Extract data
        pre_data = np.array(results['pre_gpt'][metric])
        post_data = np.array(results['post_gpt'][metric])
        
        # Check sample sizes
        n_pre = len(pre_data)
        n_post = len(post_data)
        print(f"  Pre-GPT N = {n_pre}, Post-GPT N = {n_post}")

        # Strategy for large datasets:
        if n_pre > 5000 or n_post > 5000:
            print("  Large sample detected - using non-parametric test directly")
            stat, p = stats.mannwhitneyu(pre_data, post_data)
            test_name = "Mann-Whitney U test"
        else:
            # Normality check only for smaller samples
            _, pre_p = stats.shapiro(pre_data)
            _, post_p = stats.shapiro(post_data)
            print(f"  Pre-GPT Normality p-value: {pre_p:.4f}")
            print(f"  Post-GPT Normality p-value: {post_p:.4f}")
            
            if pre_p > 0.05 and post_p > 0.05:
                stat, p = stats.ttest_ind(pre_data, post_data)
                test_name = "Independent t-test"
            else:
                stat, p = stats.mannwhitneyu(pre_data, post_data)
                test_name = "Mann-Whitney U test"

        print(f"  Test Used: {test_name}")
        print(f"  p-value: {p:.4f}")
        if p < 0.05:
            print("  **Significant difference detected**")
        else:
            print("  No significant difference")
        print("\n" + "-"*50 + "\n")

# plot normality 
def plot_qq(data, title):
    plt.figure()
    stats.probplot(data, plot=plt)
    plt.title(f"Q-Q Plot for {title}")
    plt.show()


perform_statistical_analysis(results)


plot_qq(results['post_gpt']['sent_lengths'], "Post-GPT Sentence Lengths")
