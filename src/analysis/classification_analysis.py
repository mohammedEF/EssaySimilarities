from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import json


def run_classification_analysis(df):
    """
    Run a classification model to predict essay time period.
    This helps identify the most discriminative features.
    
    Returns:
        Dictionary of classification results and feature importances
    """
            
    # Extract all essays and labels
    texts = df['essay_text'].tolist()
    labels = (df['time_period'] == 'post_gpt').astype(int).tolist()
    
    # Feature extraction - TF-IDF
    tfidf = TfidfVectorizer(max_features=500)
    X_tfidf = tfidf.fit_transform(texts).toarray()
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, labels, test_size=0.3, random_state=42)
    
    # Train Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Get feature importances
    importances = clf.feature_importances_
    feature_names = tfidf.get_feature_names_out()
    
    # Get top features
    indices = np.argsort(importances)[::-1]
    top_features = [(feature_names[i], importances[i]) for i in indices[:20]]
    
    results = {
        'classification_report': report,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'top_discriminative_features': top_features
    }
    
    return results


df = pd.read_csv("C:/Users/PRECISION 5550/Desktop/Essays Project/dataset/combined_processed.csv")
results = run_classification_analysis(df)

with open("C:/Users/PRECISION 5550/Desktop/Essays Project/new/classification modelling/classification_results.txt", 'w') as f:
    json.dump(results, f)

print('done saving ...')


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_results(results):
    # Set style
    sns.set(style="whitegrid")
    plt.figure(figsize=(15, 12))
    
    # Classification metrics plot
    class_report = results['classification_report']
    classes = ['0 (pre-GPT)', '1 (post-GPT)']
    
    metrics = ['precision', 'recall', 'f1-score']
    class_data = {
        'Class': [],
        'Metric': [],
        'Value': []
    }
    
    for cls in ['0', '1']:
        for metric in metrics:
            class_data['Class'].append(classes[int(cls)])
            class_data['Metric'].append(metric.capitalize())
            class_data['Value'].append(class_report[cls][metric])
    
    plt.subplot(2, 2, 1)
    sns.barplot(x='Metric', y='Value', hue='Class', data=class_data, palette=['#1f77b4', '#ff7f0e'])
    plt.title('Classification Metrics by Class')
    plt.ylim(0, 1)
    plt.legend(title='Class')
    
    # Confusion matrix plot
    plt.subplot(2, 2, 2)
    cm = np.array(results['confusion_matrix'])
    labels = ['pre-GPT', 'post-GPT']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Feature importance plot
    plt.subplot(2, 1, 2)
    features = results['top_discriminative_features']
    feature_names = [f[0] for f in features]
    importance_scores = [float(f[1]) for f in features]  # Convert numpy float to Python float
    
    sns.barplot(x=importance_scores, y=feature_names, palette='viridis')
    plt.title('Top Discriminative Features')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()


visualize_results(results)