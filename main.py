"""
NLP Assessment: Main Execution Script
Farid Mikidam

Description:
This script implements a comparative study of four text summarisation approaches:
1. TextRank (Unsupervised Extractive)
2. Random Forest (Supervised Extractive + Feature Engineering)
3. DistilBART (Abstractive Transformer)
4. T5-Small (Abstractive Transformer)
"""

import pandas as pd                                                                        # Import Pandas for data tables
import numpy as np                                                                         # Import NumPy for math operations
import matplotlib.pyplot as plt                                                            # Import Matplotlib for graphing
from src.data_loader import load_data                                                      # Import custom data loading function
from src.models import run_textrank, train_rf, run_rf, fine_tune_transformer               # Import our 4 model functions
from src.evaluation import evaluate_metrics                                                # Import evaluation metrics
import gc                                                                                  # Import Garbage Collector to free RAM
import torch                                                                               # Import PyTorch

def main():
    # 1. Load Data
    print("\n=== STEP 1: Data Loading ===")                                                # Print Step Header
    df = load_data("data/articles", "data/summaries")                                      # Load text files from folders
    df_run = df.iloc[:200].copy()                                                          # Select 200 docs for Extractive models
    df_dl = df.iloc[:50].copy()                                                            # Select 50 docs for Transformers (CPU limit)
    print(f"Total Docs: {len(df)}. Using {len(df_run)} for Extractive, {len(df_dl)} for Abstractive.")

    results = {}                                                                           # Dictionary to store all final scores

    # Helper function to print and store scores
    def print_scores(name, r1, bleu):
        # INNOVATION: Custom Combined Metric (F-Mean)
        # Balances Recall (ROUGE) and Precision (BLEU) equally
        combined = (r1 + bleu) / 2                                                         # Calculate the custom F-Mean
        print(f"{name} -> R1: {r1:.4f}, BLEU: {bleu:.4f} | F-Mean: {combined:.4f}")        # Print results to terminal
        return {'R1': r1, 'BLEU': bleu, 'F-Mean': combined}                                # Return results dictionary

    # 2. TextRank
    print("\n=== STEP 2: TextRank (Unsupervised) ===")                                     # Print Step Header
    try:
        preds = df_run['document'].apply(lambda x: run_textrank(x)).tolist()               # Apply TextRank to every article
        r1, r2, rl, bleu = evaluate_metrics(preds, df_run['summary'].tolist())             # Calculate ROUGE and BLEU scores
        results['TextRank'] = print_scores('TextRank', r1, bleu)                           # Store and print results
    except Exception as e: print(f"TextRank Failed: {e}")                                  # Catch errors so code continues

    # 3. Random Forest (Structure-Aware Innovation)
    print("\n=== STEP 3: Random Forest (Structure-Aware) ===")                             # Print Step Header
    best_rf_score = 0                                                                      # Variable to track the best tree setting
    
    # Sensitivity Analysis Loop: Test 10, 50, and 100 trees
    for trees in [10, 50, 100]:
        print(f"Training RF with {trees} trees...")                                        # Print current experiment
        rf_model, rf_vec = train_rf(df_run, n_trees=trees)                                 # Train Random Forest on the data
        preds = df_run['document'].apply(lambda x: run_rf(x, rf_model, rf_vec)).tolist()   # Generate summaries
        r1, r2, rl, bleu = evaluate_metrics(preds, df_run['summary'].tolist())             # Evaluate performance
        
        # Calculate combined score for decision making
        combined = (r1 + bleu) / 2                                                         # Calculate F-Mean
        print(f"Trees: {trees} -> R1: {r1:.4f}, BLEU: {bleu:.4f} | F-Mean: {combined:.4f}") # Print result
        
        # Keep the result if it is the best one so far
        if r1 > best_rf_score:
            best_rf_score = r1
            results['RF (Best)'] = {'R1': r1, 'BLEU': bleu, 'F-Mean': combined}            # Store best result for plotting

    # Helper function to decode Transformer outputs safely
    def decode_predictions(trainer, out):
        decoded_preds = trainer.tokenizer.batch_decode(out.predictions, skip_special_tokens=True) # Convert IDs to text
        # FIX: Replace -100 in labels with pad_token_id to prevent OverflowError
        label_ids = np.where(out.label_ids != -100, out.label_ids, trainer.tokenizer.pad_token_id)
        decoded_labels = trainer.tokenizer.batch_decode(label_ids, skip_special_tokens=True)      # Convert labels to text
        return decoded_preds, decoded_labels

    # 4. DistilBART
    print("\n=== STEP 4: DistilBART (Abstractive) ===")                                    # Print Step Header
    try:
        # Fine-tune the model (returns trainer and test set)
        trainer, test_set = fine_tune_transformer(df_dl, "sshleifer/distilbart-cnn-12-6", "bart_out")
        print("Generating with Beam Search (Beams=4)...")                                  # Print generation strategy
        out = trainer.predict(test_set, num_beams=4)                                       # Generate summaries using Beams
        
        preds, refs = decode_predictions(trainer, out)                                     # Decode numbers to text
        
        r1, r2, rl, bleu = evaluate_metrics(preds, refs)                                   # Evaluate metrics
        results['DistilBART'] = print_scores('DistilBART', r1, bleu)                       # Store results
        
        del trainer                                                                        # Delete model to free memory
        gc.collect()                                                                       # Force garbage collection
    except Exception as e: print(f"DistilBART Failed: {e}")                                # Handle errors

    # 5. T5-Small
    print("\n=== STEP 5: T5-Small ===")                                                    # Print Step Header
    try:
        # Fine-tune T5-Small
        trainer, test_set = fine_tune_transformer(df_dl, "t5-small", "t5_out")
        out = trainer.predict(test_set)                                                    # Generate summaries
        
        preds, refs = decode_predictions(trainer, out)                                     # Decode numbers to text
        
        r1, r2, rl, bleu = evaluate_metrics(preds, refs)                                   # Evaluate metrics
        results['T5-Small'] = print_scores('T5-Small', r1, bleu)                           # Store results
        
        del trainer                                                                        # Delete model
        gc.collect()                                                                       # Clear memory
    except Exception as e: print(f"T5 Failed: {e}")                                        # Handle errors

    # 6. Plotting
    print("\n=== STEP 6: Visualization ===")                                               # Print Step Header
    if results:
        names = list(results.keys())                                                       # Get model names
        # Extract scores from the results dictionary
        r1_scores = [results[n]['R1'] for n in names]
        bleu_scores = [results[n]['BLEU'] for n in names]
        fmean_scores = [results[n]['F-Mean'] for n in names]                               # Innovation Metric
        
        x = np.arange(len(names))                                                          # Create label locations
        width = 0.25                                                                       # Set bar width

        fig, ax = plt.subplots(figsize=(12, 6))                                            # Create figure
        
        # Plot 3 bars side-by-side
        rects1 = ax.bar(x - width, r1_scores, width, label='ROUGE-1 (Recall)', color='salmon')
        rects2 = ax.bar(x, bleu_scores, width, label='BLEU (Precision)', color='skyblue')
        rects3 = ax.bar(x + width, fmean_scores, width, label='F-Mean (Combined)', color='lightgreen')

        # Add labels, title, and custom x-axis tick labels
        ax.set_ylabel('Score (0.0 - 1.0)')
        ax.set_title('Model Comparison: ROUGE vs BLEU vs F-Mean')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend()
        ax.set_ylim(0, 1.0)

        # Function to add labels on top of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)

        plt.savefig("final_comparison.png")                                                # Save chart
        print("Graph saved as final_comparison.png")                                       # Confirm save
    else:
        print("No results to plot.")                                                       # Error message if empty

if __name__ == "__main__":
    main()