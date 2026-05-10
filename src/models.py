import numpy as np                                                                         # Import NumPy for array manipulation
from scipy.sparse import hstack                                                            # Import hstack to combine features
import networkx as nx                                                                      # Import NetworkX for graph algorithms
from sklearn.feature_extraction.text import TfidfVectorizer                                # Import TF-IDF vectorizer
from sklearn.ensemble import RandomForestClassifier                                        # Import Random Forest model
from sklearn.metrics.pairwise import cosine_similarity                                     # Import Cosine Similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM                              # Import Transformer classes
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq  # Import Training utilities
from datasets import Dataset                                                               # Import Hugging Face Dataset
from src.processing import clean_for_extractive, preprocess_for_transformer             # Import custom cleaning logic

# ==========================================
# 1. TEXTRANK (Unsupervised Baseline)
# ==========================================

def run_textrank(text, top_n=2):
    """Runs TextRank algorithm on a single text."""
    sentences, clean_sentences = clean_for_extractive(text)                                # Get original and cleaned sentences
    if len(sentences) < top_n: return text                                                 # Return full text if too short
    
    if not clean_sentences: return sentences[0]                                            # Safety check for empty text
    
    try:
        vectorizer = TfidfVectorizer()                                                     # Initialize TF-IDF
        vectors = vectorizer.fit_transform(clean_sentences)                                # Convert sentences to vectors
        sim_mat = cosine_similarity(vectors)                                               # Calculate similarity matrix
        nx_graph = nx.from_numpy_array(sim_mat)                                            # Build graph from similarity
        scores = nx.pagerank(nx_graph)                                                     # Calculate PageRank scores
        ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)   # Sort sentences by score
        return " ".join([s for score, s in ranked[:top_n]])                                # Return top N sentences
    except:
        return sentences[0]                                                                # Fallback if math fails

# ==========================================
# 2. RANDOM FOREST (Structure-Aware Innovation)
# ==========================================

def train_rf(df, n_trees=50):
    """Trains RF using TF-IDF content + Sentence Position metadata."""
    train_sentences = []                                                                   # List for text features
    train_positions = []                                                                   # List for position features
    train_labels = []                                                                      # List for labels
    
    for _, row in df.iterrows():                                                           # Loop through documents
        # Get cleaned sentences (Index 1 from preprocessing)
        doc_sents = clean_for_extractive(row['document'])[1]                               
        sum_sents = clean_for_extractive(row['summary'])[1]                                
        
        total_sents = len(doc_sents)                                                       # Get total sentence count
        if total_sents == 0: total_sents = 1                                               # Avoid division by zero

        for i, sent in enumerate(doc_sents):                                               # Loop through sentences
            train_sentences.append(sent)                                                   # Add text to list
            # INNOVATION: Calculate Relative Position (0.0 = Start, 1.0 = End)
            train_positions.append([i / total_sents])                                      
            
            # Label generation: 1 if sentence overlaps with summary, else 0
            label = 1 if any(sent in s or s in sent for s in sum_sents) else 0             
            train_labels.append(label)                                                     # Add label

    # 1. Generate TF-IDF Matrix (Content Features)
    vectorizer = TfidfVectorizer(stop_words='english')                                     # Initialize Vectorizer
    tfidf_matrix = vectorizer.fit_transform(train_sentences)                               # Fit and Transform text
    
    # 2. Combine TF-IDF with Position Feature (Structure Features)
    # This creates a "Structure-Aware" matrix for the innovation
    X = hstack([tfidf_matrix, np.array(train_positions)])                                  # Stack matrices horizontally
    
    clf = RandomForestClassifier(n_estimators=n_trees, random_state=42)                    # Initialize Random Forest
    clf.fit(X, train_labels)                                                               # Train the model
    return clf, vectorizer                                                                 # Return model and vectorizer

def run_rf(text, model, vectorizer, top_n=2):
    """Generates summary using Structure-Aware RF."""
    sentences, clean_sents = clean_for_extractive(text)                                    # Get sentences
    if not clean_sents: return ""                                                          # Handle empty
    
    try:
        # Prepare content features
        tfidf_mat = vectorizer.transform(clean_sents)                                      # Vectorize text
        # Prepare position features (same logic as training)
        positions = np.array([[i / len(clean_sents)] for i in range(len(clean_sents))])    # Calculate positions
        
        # Combine features exactly like in training
        X = hstack([tfidf_mat, positions])                                                 # Stack features
        
        probs = model.predict_proba(X)[:, 1]                                               # Predict "Importance" probability
        ranked_idx = np.argsort(probs)[::-1][:top_n]                                       # Get top N indices
        ranked_idx.sort()                                                                  # Sort indices for logical flow
        return " ".join([sentences[i] for i in ranked_idx])                                # Return summary using original sentences
    except:
        return sentences[0]                                                                # Fallback

# ==========================================
# 3. TRANSFORMERS (DistilBART / T5)
# ==========================================

def fine_tune_transformer(df, model_name, output_dir):
    """Fine-tunes a transformer model with CPU optimizations."""
    print(f"Preparing {model_name}...")                                                    # Print status
    tokenizer = AutoTokenizer.from_pretrained(model_name)                                  # Load tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)                              # Load model
    
    dataset = Dataset.from_pandas(df)                                                      # Create dataset
    dataset = dataset.train_test_split(test_size=0.2, seed=42)                             # Split train/test
    
    prefix = "summarize: " if "t5" in model_name else ""                                   # Handle T5 prefix requirement
    tokenized = dataset.map(lambda x: preprocess_for_transformer(x, tokenizer, prefix), batched=True) # Preprocess data
    
    args = Seq2SeqTrainingArguments(                                                       # Set training args
        output_dir,                                                                        # Output folder
        eval_strategy="no",                                                                # Skip eval during training
        learning_rate=2e-5,                                                                # Low LR to prevent forgetting
        per_device_train_batch_size=2,                                                     # Small batch for CPU
        num_train_epochs=1,                                                                # 1 Epoch
        max_steps=50,                                                                      # FORCE STOP at 50 steps (Speed optimization)
        weight_decay=0.01,                                                                 # Regularization
        predict_with_generate=True,                                                        # Enable generation
        logging_steps=10,                                                                  # Log frequency
        use_cpu=True                                                                       # Force CPU usage
    )
    
    trainer = Seq2SeqTrainer(model, args, train_dataset=tokenized["train"],                # Init Trainer
                             eval_dataset=tokenized["test"], tokenizer=tokenizer, 
                             data_collator=DataCollatorForSeq2Seq(tokenizer, model))
    
    trainer.train()                                                                        # Run training
    return trainer, tokenized["test"]                                                      # Return trainer and test set