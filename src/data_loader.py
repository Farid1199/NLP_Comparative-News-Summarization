import os                                                                                  # Import OS for path handling
import glob                                                                                # Import glob to find files
import pandas as pd                                                                        # Import Pandas for dataframes
from tqdm import tqdm                                                                      # Import progress bar

def load_data(articles_path, summaries_path):
    """Loads articles and summaries into a Pandas DataFrame."""
    data = []                                                                              # Initialize empty list for data
    article_files = glob.glob(os.path.join(articles_path, "*.txt"))                        # Get list of all article text files
    print(f"Found {len(article_files)} files. Loading...")                                 # Print count of files

    for art_file in tqdm(article_files, desc="Loading Files"):                             # Loop through files with progress bar
        try:                                                                               # Start error handling
            basename = os.path.basename(art_file)                                          # Get filename (e.g., 001.txt)
            sum_file = os.path.join(summaries_path, basename)                              # Find corresponding summary file
            
            if os.path.exists(sum_file):                                                   # Check if summary exists
                with open(art_file, 'r', encoding='utf-8', errors='ignore') as f:          # Open article
                    article_text = f.read().strip()                                        # Read and trim text
                with open(sum_file, 'r', encoding='utf-8', errors='ignore') as f:          # Open summary
                    summary_text = f.read().strip()                                        # Read and trim text
                
                if article_text and summary_text:                                          # Ensure files aren't empty
                    data.append({'document': article_text, 'summary': summary_text})       # Add to list
        except Exception as e:                                                             # Catch errors
            continue                                                                       # Skip bad files
    
    return pd.DataFrame(data)                                                              # Return as DataFrame