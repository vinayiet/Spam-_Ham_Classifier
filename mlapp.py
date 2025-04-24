import torch
import torch.nn as nn
import pickle
import re
from flask import Flask, request, render_template, jsonify, url_for
from datetime import datetime

from Preprocess.preprocess import pre_process, extract_additional_features, is_common_greeting
from model.model import DetectSpamV0
from train_model import preprocess_text

# Load vocab size
with open('model/vocab_size.pkl', 'rb') as f:
    vocab_size = pickle.load(f)

# get the availaable device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# initialize the model with loaded vocab_size and load it from the path
model = DetectSpamV0(vocab_size=vocab_size).to(device)

# Load the trained model
try:
    model.load_state_dict(torch.load('model/model_detect_spam_V0.pt', map_location=device))
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

app = Flask(__name__)

# Properly initialize all required fields
analytics = {
    "total_analyzed": 0,
    "spam_detected": 0,
    "ham_detected": 0,
    "confusion_matrix": {
        "values": [[0, 0], [0, 0]],  # [[TP, FN], [FP, TN]]
        "labels": ["Actual Spam", "Actual Ham"],
        "predicted": ["Predicted Spam", "Predicted Ham"]
    },
    "feature_importance": {
        "Suspicious Keywords": 85,
        "Multiple Exclamation": 65,
        "Excessive Caps": 45,
        "URL Patterns": 75
    },
    "class_distribution": {
        "Spam": 0,
        "Ham": 0
    }
}

@app.route('/')
@app.route('/index')
@app.route('/home')
def home():
    """Route for home page"""
    return render_template('index.html')

@app.route('/scan')
def scan():
    """Route for scanner page"""
    return render_template('scanner.html')

@app.route('/dashboard')
def dashboard():
    """Route for dashboard page"""
    return jsonify(analytics)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        text = request.json['text']
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        result = analyze_text(text)
        
        # Update analytics
        analytics['total_analyzed'] += 1
        
        if result['isSpam']:
            analytics['spam_detected'] += 1
            analytics['class_distribution']['Spam'] += 1
            analytics['confusion_matrix']['values'][0][0] += 1  # True Positive
        else:
            analytics['ham_detected'] += 1
            analytics['class_distribution']['Ham'] += 1
            analytics['confusion_matrix']['values'][1][1] += 1  # True Negative
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def analyze_text(text):
    # Enhanced spam patterns with weighted scoring
    spam_patterns = {
        'money_phrases': {
            'pattern': r'(free gift|earn|cash|money|prize|winner|won|claim|dollar|[$€£])',
            'weight': 3
        },
        'urgency_phrases': {
            'pattern': r'(urgent|limited time|act now|today only|exclusive|expires|deadline)',
            'weight': 2
        },
        'work_scam': {
            'pattern': r'(work from home|working from.*comfort|income|earn from home)',
            'weight': 4
        },
        'spam_markers': {
            'pattern': r'(click here|subscribe|guarantee|certified|congratulation|selected|promotion)',
            'weight': 2
        }
    }
    
    spam_score = 0
    found_patterns = []
    text_lower = text.lower()
    
    # Check each pattern category
    for category, data in spam_patterns.items():
        matches = len(re.findall(data['pattern'], text_lower, re.I))
        if matches > 0:
            score = matches * data['weight']
            spam_score += score
            found_patterns.append(f"{category}: {matches} matches")
    
    # Check for excessive capitalization
    caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    if caps_ratio > 0.3:
        spam_score += 2
        found_patterns.append("excessive_caps")

    # Check for multiple exclamation marks
    if '!!' in text:
        spam_score += 2
        found_patterns.append("multiple_exclamation")

    # Calculate confidence (maximum score possible is 20)
    max_possible_score = 20
    confidence = min((spam_score / max_possible_score) * 100, 100)
    
    # Lower threshold to catch more spam
    is_spam = confidence > 30  # More sensitive spam detection
    
    return {
        'isSpam': is_spam,
        'confidence': confidence,
        'patterns': found_patterns,
        'score': spam_score
    }

if __name__ == '__main__':
    app.run(debug=True, port=5000)