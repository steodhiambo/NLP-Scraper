import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import pickle
import os

def train_topic_classifier():
    """
    Train a topic classification model to categorize articles into:
    Tech, Sport, Business, Entertainment, or Politics
    """
    print("Training topic classification model...")
    
    # Sample training data - in a real scenario, this would come from a labeled dataset
    # Creating synthetic data for demonstration
    tech_texts = [
        "Apple announces new iPhone with revolutionary features",
        "Google's new AI breakthrough changes everything",
        "Microsoft releases Windows 11 update",
        "Tesla unveils new electric vehicle model",
        "Amazon Web Services launches new cloud computing platform",
        "Facebook introduces new virtual reality headset",
        "Samsung develops foldable smartphone technology",
        "Intel releases new processor with improved performance",
        "Netflix streams record number of original series",
        "YouTube introduces new video editing features"
    ] * 20  # Multiply to increase dataset size
    
    sport_texts = [
        "Champions League final scheduled for May",
        "Olympics postponed due to pandemic",
        "World Cup preparations underway",
        "NBA finals conclude with exciting finish",
        "Tennis tournament sees unexpected upset",
        "Football team wins championship after years of effort",
        "Golf major concludes with record-breaking score",
        "Soccer match draws largest crowd in history",
        "Basketball player breaks scoring record",
        "Swimming competition sets new world records"
    ] * 20
    
    business_texts = [
        "Tesla stock price reaches new high",
        "Amazon reports record profits",
        "New banking regulations announced",
        "Stock market hits all-time high",
        "Company announces merger with competitor",
        "Economic forecast predicts growth",
        "Startup receives major investment funding",
        "Retail giant opens stores worldwide",
        "Oil prices fluctuate dramatically",
        "Real estate market shows strong growth"
    ] * 20
    
    entertainment_texts = [
        "Hollywood movie wins multiple awards",
        "New streaming service launches",
        "Celebrity couple announces divorce",
        "Music festival lineup announced",
        "Broadway show extends run",
        "Television series renewed for another season",
        "Actor wins prestigious award",
        "Movie studio announces sequel plans",
        "Concert tour dates revealed",
        "Book adaptation coming to screen"
    ] * 20
    
    politics_texts = [
        "Government passes new legislation",
        "Election results announced",
        "International summit addresses climate change",
        "Political candidate announces campaign",
        "Parliament debates important bill",
        "Diplomatic relations strengthened",
        "Policy changes affect citizens",
        "Political party holds convention",
        "Vote counting underway",
        "Leader addresses nation"
    ] * 20
    
    # Combine all data
    all_texts = tech_texts + sport_texts + business_texts + entertainment_texts + politics_texts
    all_labels = (
        ['Technology'] * len(tech_texts) +
        ['Sports'] * len(sport_texts) +
        ['Business'] * len(business_texts) +
        ['Entertainment'] * len(entertainment_texts) +
        ['Politics'] * len(politics_texts)
    )
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Classes: {set(all_labels)}")
    
    # Create and train the model pipeline
    # Using TF-IDF vectorizer with Multinomial Naive Bayes
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),  # Use both unigrams and bigrams
            lowercase=True,
            strip_accents='ascii'
        )),
        ('classifier', MultinomialNB(alpha=0.1))  # Smoothing parameter
    ])
    
    # Train the model
    print("Training the model...")
    model_pipeline.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = model_pipeline.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Check if model meets the 95% accuracy requirement
    if accuracy > 0.95:
        print(f"\n✓ Model meets accuracy requirement (>95%): {accuracy:.4f}")
    else:
        print(f"\n⚠ Model does not meet accuracy requirement (>95%): {accuracy:.4f}")
        print("Consider improving the model with more data or different algorithms.")
    
    # Generate learning curves to check for overfitting
    print("\nGenerating learning curves...")
    plot_learning_curves(model_pipeline, X_train, y_train)
    
    # Save the trained model
    os.makedirs('results', exist_ok=True)
    model_path = 'results/topic_classifier.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_pipeline, f)
    print(f"\nModel saved to {model_path}")
    
    return model_pipeline, accuracy

def plot_learning_curves(model, X_train, y_train):
    """
    Plot learning curves to visualize model performance vs training set size
    This helps identify if the model is overfitting or underfitting
    """
    # Generate learning curves
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, 
        cv=5,  # 5-fold cross-validation
        n_jobs=-1,  # Use all available processors
        train_sizes=np.linspace(0.1, 1.0, 10),  # 10 different sizes
        scoring='accuracy'
    )
    
    # Calculate mean and standard deviation for plotting
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot the learning curves
    plt.figure(figsize=(10, 6))
    
    # Training curve
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    # Validation curve
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Cross-Validation Accuracy')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    # Labels and title
    plt.title('Learning Curves for Topic Classification Model')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Save the plot
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Learning curves saved to results/learning_curves.png")

def evaluate_different_models():
    """
    Compare different algorithms to find the best performing one
    """
    print("\nEvaluating different models...")
    
    # Sample training data (same as above)
    tech_texts = [
        "Apple announces new iPhone with revolutionary features",
        "Google's new AI breakthrough changes everything",
        "Microsoft releases Windows 11 update",
        "Tesla unveils new electric vehicle model",
        "Amazon Web Services launches new cloud computing platform",
        "Facebook introduces new virtual reality headset",
        "Samsung develops foldable smartphone technology",
        "Intel releases new processor with improved performance",
        "Netflix streams record number of original series",
        "YouTube introduces new video editing features"
    ] * 10
    
    sport_texts = [
        "Champions League final scheduled for May",
        "Olympics postponed due to pandemic",
        "World Cup preparations underway",
        "NBA finals conclude with exciting finish",
        "Tennis tournament sees unexpected upset",
        "Football team wins championship after years of effort",
        "Golf major concludes with record-breaking score",
        "Soccer match draws largest crowd in history",
        "Basketball player breaks scoring record",
        "Swimming competition sets new world records"
    ] * 10
    
    business_texts = [
        "Tesla stock price reaches new high",
        "Amazon reports record profits",
        "New banking regulations announced",
        "Stock market hits all-time high",
        "Company announces merger with competitor",
        "Economic forecast predicts growth",
        "Startup receives major investment funding",
        "Retail giant opens stores worldwide",
        "Oil prices fluctuate dramatically",
        "Real estate market shows strong growth"
    ] * 10
    
    entertainment_texts = [
        "Hollywood movie wins multiple awards",
        "New streaming service launches",
        "Celebrity couple announces divorce",
        "Music festival lineup announced",
        "Broadway show extends run",
        "Television series renewed for another season",
        "Actor wins prestigious award",
        "Movie studio announces sequel plans",
        "Concert tour dates revealed",
        "Book adaptation coming to screen"
    ] * 10
    
    politics_texts = [
        "Government passes new legislation",
        "Election results announced",
        "International summit addresses climate change",
        "Political candidate announces campaign",
        "Parliament debates important bill",
        "Diplomatic relations strengthened",
        "Policy changes affect citizens",
        "Political party holds convention",
        "Vote counting underway",
        "Leader addresses nation"
    ] * 10
    
    # Combine all data
    all_texts = tech_texts + sport_texts + business_texts + entertainment_texts + politics_texts
    all_labels = (
        ['Technology'] * len(tech_texts) +
        ['Sports'] * len(sport_texts) +
        ['Business'] * len(business_texts) +
        ['Entertainment'] * len(entertainment_texts) +
        ['Politics'] * len(politics_texts)
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    # Define models to compare
    models = {
        'MultinomialNB': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
            ('classifier', MultinomialNB(alpha=0.1))
        ]),
        'LogisticRegression': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'SVM': Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
            ('classifier', SVC(kernel='linear', random_state=42))
        ])
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"{name} Accuracy: {accuracy:.4f}")
    
    # Print comparison
    print("\nModel Comparison:")
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {acc:.4f}")
    
    return results

if __name__ == "__main__":
    # Train the main topic classifier
    model, accuracy = train_topic_classifier()
    
    # Optionally, evaluate different models
    # model_comparison = evaluate_different_models()