import nltk                                                                                # Import NLTK
from nltk.corpus import stopwords                                                          # Import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize                                     # Import tokenizers

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')                                                     # Check for punkt
except LookupError:
    nltk.download('punkt')                                                                 # Download if missing
    nltk.download('stopwords')                                                             # Download stopwords if missing

# OPTIMIZATION: Load stopwords once globally
STOP_WORDS = set(stopwords.words('english'))                                               # Initialize global stopword set

def clean_for_extractive(text):
    """Cleans text for TextRank/RF (removes stopwords, lowercases)."""
    sentences = sent_tokenize(text)                                                        # Split text into sentences
    
    # Generate list of cleaned sentences using global STOP_WORDS
    clean_sentences = [" ".join([w for w in word_tokenize(s.lower()) if w.isalnum() and w not in STOP_WORDS]) for s in sentences]
    
    return sentences, clean_sentences                                                      # Return original and clean sentences

def preprocess_for_transformer(examples, tokenizer, prefix=""):
    """Formats text for BART/T5 (adds prefix, tokenizes)."""
    inputs = [prefix + doc for doc in examples["document"]]                                # Add prefix (needed for T5)
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)                      # Tokenize input text
    
    with tokenizer.as_target_tokenizer():                                                  # Context for target tokenization
        labels = tokenizer(examples["summary"], max_length=128, truncation=True)           # Tokenize summaries
    
    model_inputs["labels"] = labels["input_ids"]                                           # Add labels to inputs
    return model_inputs                                                                    # Return processed batch