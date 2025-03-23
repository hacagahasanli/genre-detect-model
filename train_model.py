import json
import nltk
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Download NLTK data (run once)
nltk.download('punkt')

# Enhanced Azerbaijani stemmer with stop word removal
def azerbaijani_stemmer(text):
    # Common Azerbaijani suffixes
    suffixes = [
        "dir", "dır", "dur", "dür", 
        "miş", "mış", "muş", "müş", 
        "lar", "lər", "dan", "dən", 
        "a", "ə", "ın", "in", "un", "ün",
        "da", "də", "ki"
    ]
    
    # Common Azerbaijani stop words
    stop_words = [
        "bir", "və", "amma", "ki", "ilə", "bu", "o", "üçün", 
        "belə", "də", "artıq", "daha", "lakin", "çünki", "əgər"
    ]
    
    # Tokenize and lowercase
    words = word_tokenize(text.lower())
    
    # Remove stop words and apply stemming
    stemmed_words = []
    for word in words:
        # Skip stop words
        if word in stop_words:
            continue
            
        # Apply stemming
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:  # Ensure word is long enough
                word = word[:-len(suffix)]
                break
        
        # Only add non-empty words
        if word and len(word) > 1:
            stemmed_words.append(word)
            
    return " ".join(stemmed_words)

def load_and_prepare_data(file_path="data.json"):
    """Load dataset from JSON file and prepare for training"""
    print(f"Loading data from {file_path}...")
    
    # Load dataset from JSON
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} samples")
    
    # Extract text and genre
    texts = [entry["text"] for entry in dataset]
    labels = [entry["genre"] for entry in dataset]
    
    # Check class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Genre '{label}': {count} samples ({count/len(labels)*100:.1f}%)")
    
    # Preprocess texts
    print("Preprocessing texts...")
    preprocessed_texts = [azerbaijani_stemmer(text) for text in texts]
    
    return preprocessed_texts, labels

def train_and_evaluate_model(X, y, output_dir="model_output"):
    """Train, evaluate and save the genre classification model"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create a pipeline with TF-IDF and SVM
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            max_features=10000,  # More features for better performance
            ngram_range=(1, 2),  # Include bigrams
            min_df=3,            # Ignore terms that appear in fewer than 3 documents
            max_df=0.9           # Ignore terms that appear in more than 90% of documents
        )),
        ('classifier', SVC(probability=True))
    ])
    
    # Define parameter grid for grid search
    param_grid = {
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__kernel': ['linear', 'rbf'],
        'vectorizer__max_features': [5000, 10000]
    }
    
    # Perform grid search
    print("Performing grid search for hyperparameter tuning...")
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Generate classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)
    
    # Save classification report to file
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Model Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=grid_search.classes_, yticklabels=grid_search.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Save model and vectorizer
    if accuracy > 0.7:  # Lower threshold for Azerbaijani genre classification which can be challenging
        # Save the complete pipeline
        with open(os.path.join(output_dir, 'model_pipeline.pkl'), 'wb') as f:
            pickle.dump(best_model, f)
        print("Model pipeline saved successfully!")
        
        # Save class names separately for reference
        with open(os.path.join(output_dir, 'class_names.pkl'), 'wb') as f:
            pickle.dump(grid_search.classes_, f)
        
        return best_model, grid_search.classes_
    else:
        print("Accuracy too low. Consider more data or parameter tuning.")
        return None, None

if __name__ == "__main__":
    # Load and prepare data
    X, y = load_and_prepare_data("data.json")
    
    # Train and evaluate model
    model, class_names = train_and_evaluate_model(X, y)
    
    print("Model training complete!")
    
    
    