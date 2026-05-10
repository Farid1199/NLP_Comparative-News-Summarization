from rouge_score import rouge_scorer                                                       # Import ROUGE
from nltk.translate.bleu_score import sentence_bleu                                        # Import BLEU
import numpy as np                                                                         # Import NumPy
from nltk.tokenize import word_tokenize                                                    # Import tokenizer

def evaluate_metrics(predictions, references):
    """Calculates ROUGE and BLEU scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)    # Init ROUGE
    r1, r2, rl, bleus = [], [], [], []                                                     # Init lists
    
    for pred, ref in zip(predictions, references):                                         # Loop pairs
        # ROUGE
        scores = scorer.score(ref, pred)                                                   # Calc ROUGE
        r1.append(scores['rouge1'].fmeasure)                                               # Store R1
        r2.append(scores['rouge2'].fmeasure)                                               # Store R2
        rl.append(scores['rougeL'].fmeasure)                                               # Store RL
        
        # BLEU (Requires tokenized lists)
        ref_tokens = [word_tokenize(ref.lower())]                                          # Tokenize ref
        pred_tokens = word_tokenize(pred.lower())                                          # Tokenize pred
        # Calculate BLEU with smoothing to prevent 0 scores for short texts
        bleu = sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5))                  # Calc BLEU-2
        bleus.append(bleu)                                                                 # Store BLEU
        
    return np.mean(r1), np.mean(r2), np.mean(rl), np.mean(bleus)                           # Return averages