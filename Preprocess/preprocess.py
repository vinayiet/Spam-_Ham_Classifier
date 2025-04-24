import re
import nltk
import string
import pickle
import torch
import numpy as np
from nltk.tokenize import word_tokenize

try:
    # Try importing PorterStemmer and stopwords
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords
except:
    # If import fails, download necessary NLTK resources
    nltk.download('punkt')
    nltk.download('stopwords')
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords

# path to the saved vectorizer and load it
VEC_PATH = "Preprocess/tfidf_vectorizer.pkl"
with open(VEC_PATH, 'rb') as file:
    tfidf = pickle.load(file)

def is_common_greeting(text):
    """Check if the text is a common greeting or short benign message"""
    common_greetings = [
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "how are you", "what's up", "wassup", "yo", "hola", "howdy", "greetings"
    ]
    
    # Clean text for comparison
    cleaned_text = text.lower().strip().replace("!", "").replace(".", "")
    
    # Check for exact matches
    if cleaned_text in common_greetings:
        return True
    
    # Check for short messages (3 words or less) that don't contain suspicious terms
    if len(cleaned_text.split()) <= 3:
        suspicious_terms = ["urgent", "money", "offer", "win", "click", "account", "bank"]
        if not any(term in cleaned_text for term in suspicious_terms):
            return True
    
    return False

def extract_additional_features(text):
    """Extract additional spam detection features"""
    features = {}
    
    # 1. Number of URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    features['url_count'] = len(re.findall(url_pattern, text))
    
    # 2. Email Length Features
    features['char_length'] = len(text)
    features['word_count'] = len(text.split())
    
    # 3. Spam Keyword Count - EXPANDED LIST
    spam_keywords = [
        'prize', 'cash', 'winner', 'lottery', 'offer expires', 
        'click here', 'money back', 'order now', 'buy now',
        'limited time', 'credit card', 'act now', 'free gift',
        'investment', 'double your', 'account', 'bank service', 
        'earn', 'claim', 'link', 'billing', 'limited time offer',
        'cashback', 'urgent', 'verify', 'password', 'suspension',
        'authenticate', 'update required', 'security alert',
        'bank', 'access', 'restricted', 'wire transfer', 'deposit',
        'funds', 'payment', 'blocked', 'confirm', 'verification',
        'expire', 'reward', 'bonus', 'discount', 'exclusive'
    ]
    
    # Check for whole words and partial matches in common spam phrases
    text_lower = text.lower()
    spam_word_count = 0
    
    # Count individual keywords
    for word in spam_keywords:
        # Use word boundary for single words, flexible matching for phrases
        if ' ' in word:
            if word in text_lower:
                spam_word_count += 1.5  # Give phrases higher weight but reduced from 2
        else:
            pattern = r'\b' + re.escape(word) + r'\b'
            matches = len(re.findall(pattern, text_lower))
            spam_word_count += matches
    
    # Apply lower multiplier for short texts that seem like greetings
    if is_common_greeting(text):
        features['spam_keyword_count'] = spam_word_count * 0.5  # Much less weight for greetings
    else:
        # Apply standard multiplier
        features['spam_keyword_count'] = spam_word_count * 2  # Reduced from 3
    
    # 4. Email Domain Count (if any)
    email_pattern = r'[\w\.-]+@[\w\.-]+'
    features['email_domain_count'] = len(re.findall(email_pattern, text))
    
    # 5. Special characters density (often used in spam)
    special_chars = sum(1 for char in text if char in '!$%*+?#')
    
    # For very short texts, discount the special character density
    if len(text) < 10:
        features['special_chars_density'] = special_chars / max(1, len(text)) * 0.5
    else:
        features['special_chars_density'] = special_chars / max(1, len(text))
    
    # 6. Add greeting signal - strong bias toward ham for greetings
    features['greeting_signal'] = 1.0 if is_common_greeting(text) else 0.0
    
    return np.array([
        features['url_count'], 
        features['char_length'],
        features['word_count'],
        features['spam_keyword_count'],
        features['email_domain_count'],
        features['special_chars_density'] * 5,  # Reduced from 10
        features['greeting_signal'] * 2  # Strong signal for greetings
    ], dtype='float32')

def pre_process(text):
    # Get additional features first
    additional_features = extract_additional_features(text)
    
    # Lowercasing
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Stopword Removal
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Punctuation Removal
    tokens = [word for word in tokens if word not in string.punctuation]

    # Stemming
    ps = PorterStemmer()
    stemmed_tokens = [ps.stem(word) for word in tokens]

    # Join tokens back into a string
    processed_text = ' '.join(stemmed_tokens)

    # Get TF-IDF features
    tfidf_features = tfidf.transform([processed_text]).toarray().astype('float32')
    
    # Combine TF-IDF and additional features
    combined_features = np.concatenate([tfidf_features.flatten(), additional_features])
    
    # Convert to tensor
    features_tensor = torch.from_numpy(combined_features)
    
    return features_tensor