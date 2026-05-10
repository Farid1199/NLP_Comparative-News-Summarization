# News Article Summarization: A Comparative Study

## Project Overview
This project implements a modular NLP pipeline to compare Extractive and Abstractive summarization techniques on sports news articles. It features a custom "Structure-Aware" Random Forest implementation and fine-tuned Transformer models.

## Repository Structure
- `data/`: Contains the article and summary text files.
- `src/`: Contains the modular Python code.
  - `data_loader.py`: Handles file ingestion and error checking.
  - `processing.py`: Contains cleaning logic (customized for each model type).
  - `models.py`: Encapsulates TextRank, Random Forest, BART, and T5 logic.
  - `evaluation.py`: Computes ROUGE and BLEU metrics.
- `main.py`: The execution script that orchestrates the pipeline and generates plots.
- `requirements.txt`: List of dependencies.

## Key Features (Innovation)
1. **Structure-Aware Feature Engineering:** The Random Forest model uses a hybrid feature set combining TF-IDF vectors with explicit **Sentence Position** metadata.
2. **Custom F-Mean Metric:** The evaluation pipeline calculates a harmonic mean of ROUGE-1 (Recall) and BLEU (Precision) for a balanced assessment.

## System Requirements
- Python 3.10 or higher
- Windows OS
- At least 8GB RAM (16GB recommended)

## How to Run (Windows)

### Step 1: Create Virtual Environment
Open PowerShell and navigate to the project directory:

```powershell
cd "path\to\DG3NLP_Assessment"
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**If you get an execution policy error, run this first:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again: `.\venv\Scripts\Activate.ps1`

You should see `(venv)` at the start of your prompt.

### Step 2: Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Installation takes approximately 5-10 minutes.

### Step 3: Run the Program
```powershell
python main.py

Run the following if having Panda error issues:
pip install pandas numpy matplotlib torch transformers datasets rouge-score nltk scikit-learn
pip install 'accelerate>=0.26.0'
```

**Expected Runtime:** 20-35 minutes total
- TextRank & Random Forest: ~2-5 minutes
- Transformer models: ~15-30 minutes

## Output
After completion, you will have:
- Console output showing ROUGE and BLEU scores for each model
- `final_comparison.png` - Bar chart comparing all models
- `bart_out/` - DistilBART model checkpoints
- `t5_out/` - T5 model checkpoints

## Troubleshooting

**Import Errors:** 
- Ensure virtual environment is activated (you should see `(venv)` in prompt)
- Reinstall dependencies: `pip install -r requirements.txt`

**Memory Errors:**
- Close other applications
- Reduce dataset size in main.py if needed

**Corrupted Model Cache:**
- Delete: `C:\Users\<YourUsername>\.cache\huggingface\hub\models--sshleifer--distilbart-cnn-12-6`
- Rerun the program


**If you get a panda erro**
- DRun the following: `C:\Users\<YourUsername>\.cache\huggingface\hub\models--sshleifer--distilbart-cnn-12-6`

