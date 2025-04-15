import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, classification_report,confusion_matrix

df = pd.read_csv("Classification/NER_with_ensemble_sentiment.csv")

# Function to extract sentiment labels from the JSON formatted columns
def calculate_kappa(df):
    human1_labels = []
    human2_labels = []
    
    # Iterate through each row in the dataframe
    for _, row in df.iterrows():
        h1_sentiment = row['human1_sentiment']
        h2_sentiment = row['human2_sentiment']
        
        # Skip if either annotation is missing
        if pd.isna(h1_sentiment) or pd.isna(h2_sentiment):
            continue
            
        # Convert string representations to dictionaries if needed
        if isinstance(h1_sentiment, str):
            try:
                h1_sentiment = json.loads(h1_sentiment.replace("'", '"'))
            except:
                continue
                
        if isinstance(h2_sentiment, str):
            try:
                h2_sentiment = json.loads(h2_sentiment.replace("'", '"'))
            except:
                continue
        
        # Extract company tickers and their corresponding sentiment labels
        for company in h1_sentiment:
            if company in h2_sentiment:
                # Check if both humans provided labels for this company
                if "human1" in h1_sentiment[company] and "human2" in h2_sentiment[company]:
                    if "label" in h1_sentiment[company]["human1"] and "label" in h2_sentiment[company]["human2"]:
                        h1_label = h1_sentiment[company]["human1"]["label"]
                        h2_label = h2_sentiment[company]["human2"]["label"]
                        
                        # Add the pair of labels to our lists
                        human1_labels.append(h1_label)
                        human2_labels.append(h2_label)
    
    return human1_labels, human2_labels

# Extract sentiment labels
human1_labels, human2_labels = calculate_kappa(df)

# Print the number of valid pairs found
print(f"Found {len(human1_labels)} valid annotation pairs")

# Calculate Cohen's Kappa
if len(human1_labels) > 0 and len(human2_labels) > 0:
    kappa = cohen_kappa_score(human1_labels, human2_labels)
    print(f"Cohen's Kappa Score: {kappa:.4f}")
    
    # Create a confusion matrix
    confusion_matrix = pd.crosstab(
        pd.Series(human1_labels, name='Human 1'),
        pd.Series(human2_labels, name='Human 2')
    )
    
    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Inter-annotator Agreement: Human1 vs Human2')
    plt.tight_layout()
    plt.show()
    
    # Calculate percentage agreement
    matches = sum(1 for h1, h2 in zip(human1_labels, human2_labels) if h1 == h2)
    percent_agreement = (matches / len(human1_labels)) * 100
    print(f"Percentage Agreement: {percent_agreement:.2f}%")

else:
    print("No valid pairs of annotations found")
