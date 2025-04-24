import torch
import torch.nn as nn
import torch.optim as optim
from model.model import DetectSpamV0
from Preprocess.preprocess import pre_process
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import random
import re

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_model(model, X, y, criterion, device, threshold=0.5):
    model.eval()
    with torch.no_grad():
        outputs = model(X.to(device)).squeeze()
        loss = criterion(outputs, y.to(device))
        predictions = (torch.sigmoid(outputs) > threshold).float()
        accuracy = (predictions == y.to(device)).float().mean()
        
        # Calculate precision and recall for both classes
        y_cpu = y.cpu().numpy()
        pred_cpu = predictions.cpu().numpy()
        
        # Handle edge case - ensure there's at least one sample of each class
        if len(np.unique(pred_cpu)) == 1 or len(np.unique(y_cpu)) == 1:
            cm = np.zeros((2, 2))
            # If all predictions are the same class
            if len(np.unique(pred_cpu)) == 1:
                if pred_cpu[0] == 1:  # All predicted as ham
                    cm[1, 1] = np.sum(y_cpu == 1)  # True positives
                    cm[0, 1] = np.sum(y_cpu == 0)  # False positives
                else:  # All predicted as spam
                    cm[0, 0] = np.sum(y_cpu == 0)  # True negatives
                    cm[1, 0] = np.sum(y_cpu == 1)  # False negatives
        else:
            # Normal case - compute confusion matrix
            cm = confusion_matrix(y_cpu, pred_cpu)
        
        # Make sure confusion matrix has correct shape (2x2)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            print("Warning: Confusion matrix has incorrect shape:", cm.shape)
            # Default values in case of issues
            tn, fp, fn, tp = 0, 0, 0, 0
        
        # Calculate metrics (handle division by zero)
        precision_ham = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_ham = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_spam = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall_spam = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Calculate F1 scores
        f1_ham = 2 * (precision_ham * recall_ham) / (precision_ham + recall_ham) if (precision_ham + recall_ham) > 0 else 0
        f1_spam = 2 * (precision_spam * recall_spam) / (precision_spam + recall_spam) if (precision_spam + recall_spam) > 0 else 0
        
        # Calculate balanced accuracy
        balanced_acc = (recall_ham + recall_spam) / 2
        
    return {
        'loss': loss.item(),
        'accuracy': accuracy.item(),
        'balanced_accuracy': balanced_acc,
        'precision_ham': precision_ham,
        'recall_ham': recall_ham,
        'f1_ham': f1_ham,
        'precision_spam': precision_spam,
        'recall_spam': recall_spam, 
        'f1_spam': f1_spam,
        'confusion_matrix': (tn, fp, fn, tp)
    }

def add_common_phrases_data(X, y):
    """Add common non-spam phrases to balance the dataset"""
    common_phrases = [
        "hello", "hi", "hey", "how are you", "good morning", "good afternoon",
        "good evening", "what's up", "nice to meet you", "thanks", "thank you",
        "how are you doing", "hello there", "hi there", "hey there", "wassup",
        "how's it going", "what's new", "long time no see", "how have you been",
        "nice day", "good to see you", "good night", "see you soon", "talk to you later",
        "have a good day", "have a nice weekend", "hello world", "greetings",
        "welcome", "how's everything", "what's happening", "hey friend", "hi friend",
        "hello friend", "good day", "pleasure to meet you", "hey buddy", "hi team",
        "hello everyone", "greetings everyone", "hello team", "how are things"
    ]
    
    # Add these as ham (non-spam) examples
    for phrase in common_phrases:
        X = np.append(X, phrase)
        y = np.append(y, 1)  # 1 = ham (not spam)
    
    return X, y

def preprocess_text(text):
    # Convert text to feature vector
    words = text.lower().split()
    vec = [0] * 100  # Fixed size vector
    for i, word in enumerate(words[:100]):
        vec[i] = 1
    return torch.FloatTensor(vec)

def load_and_preprocess_data():
    # Read CSV file and print columns to debug
    data = pd.read_csv('data/email_spam_dataset_20000.csv')
    print("\nColumns in dataset:", data.columns.tolist())
    
    X = []
    y = []
    
    # Spam patterns with weights
    spam_patterns = [
        r'congratulations.*won',
        r'cash.*prize',
        r'click.*link',
        r'verify.*account',
        r'won.*lottery',
        r'claim.*prize',
        r'free.*gift',
        r'earn.*money'
    ]
    
    for _, row in data.iterrows():
        try:
            # Use v2 for text and v1 for label based on dataset structure
            text = str(row['v2']).lower()  # Changed from 'text' to 'v2'
            label = str(row['v1']).lower()  # Changed to use 'v1' column
            
            # Check for spam patterns
            spam_score = sum(1 for pattern in spam_patterns if re.search(pattern, text))
            is_spam = label == 'spam' or spam_score > 0
            
            # Convert text to features
            features = preprocess_text(text)
            X.append(features)
            y.append(1.0 if is_spam else 0.0)
            
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    
    print(f"\nProcessed {len(X)} samples")
    print(f"Spam messages: {sum(y)}")
    print(f"Not spam messages: {len(y) - sum(y)}")
    
    return torch.stack(X), torch.tensor(y)

def train():
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Convert numpy arrays to lists before processing
    texts = X.numpy().tolist()  # Convert tensor to list
    
    # Create vocabulary
    all_words = set()
    for text in texts:  # Now text is a regular string
        words = str(text).lower().split()
        all_words.update(words)
    
    vocab_size = min(len(all_words) + 1, 10000)
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DetectSpamV0(vocab_size=vocab_size).to(device)
    
    # Convert back to tensors for training
    X = torch.FloatTensor(X).to(device)
    y = torch.FloatTensor(y).to(device)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # ... rest of training code ...

if __name__ == "__main__":
    train()