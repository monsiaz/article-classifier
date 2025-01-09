# Article Classification Project

This repository provides a Python script (`script.py`) designed to **classify news articles** based on their relevance to a specific company or subject context. The script uses [Ollama](https://github.com/jmorganca/ollama) for local LLM inference in a *few-shot* prompt approach.


---

## Objective

**Goal**: Determine whether an article is **relevant** to the company (e.g., discussing business strategy, finances, mergers/acquisitions, major events, etc.) or **not relevant** (e.g., promotional offers, sports events, unrelated facts, local news, etc.).

---

## Overview

1. **Training CSV**  
   - **Path**: `/Users/simonazoulay/Codes/sample_test_relevance.csv`  
   - **Columns**: `Title`, `Société`, `Pertinent`, `Commentaire`  
   - Contains manually labeled articles ("Oui" or "Non") along with a brief comment explaining the rationale.

2. **Target CSV**  
   - **Path**: `/Users/simonazoulay/Codes/articles_to_classify.csv`  
   - **Columns**: `Title`, `Société`, `Pertinent`  
   - This file is **updated in place** by the script: each row’s `Pertinent` column is replaced with "Oui" or "Non."

3. **How the Script Works**  
   - For each article in `articles_to_classify.csv`, the script randomly selects **8 “Oui”** samples and **8 “Non”** samples from `sample_test_relevance.csv`.  
   - Constructs a **few-shot** prompt in French (because the dataset is in French).  
   - Runs `ollama run llama3.1` (or `llama3.1:8b`) via **STDIN** with that prompt.  
   - Parses the model’s output: if it contains “oui” (in French), the article is labeled “Oui,” otherwise “Non.”  
   - Saves the updated CSV with the classification result.

---

## Setup and Requirements

- **Ollama**  
  - Install from [Ollama GitHub](https://github.com/jmorganca/ollama).  
  - Pull the chosen model (e.g., `llama3.1` or `llama3.1:8b`):
    ```bash
    ollama pull llama3.1
    ```

- **Python 3.9+**

- **CSV Files**  
  - `/Users/simonazoulay/Codes/sample_test_relevance.csv`  
  - `/Users/simonazoulay/Codes/articles_to_classify.csv`  

---

## Usage

1. **Place your CSV files**:
   - `sample_test_relevance.csv` at `/Users/simonazoulay/Codes/sample_test_relevance.csv`
   - `articles_to_classify.csv` at `/Users/simonazoulay/Codes/articles_to_classify.csv`

2. **Run the script**:
    ```bash
    python /Users/simonazoulay/Codes/script.py
    ```
    This will iterate through each article in `articles_to_classify.csv`, classify it, and overwrite the `Pertinent` column with "Oui" or "Non."

3. **Monitor the console output**:
   - You will see lines like `[DEBUG] idx=..., Titre=... => ...` indicating classification results.
   - At the end, the script shows a summary, e.g., `[INFO] Durée d'exécution : XX.XX secondes`

4. **Check the final result**:
   - Open `/Users/simonazoulay/Codes/articles_to_classify.csv` to see the updated classification.



---

## Design Choices

### Few-Shot Approach
We randomly pick 8 “Oui” and 8 “Non” samples from the training CSV for each new article. This variety helps avoid using the same examples repeatedly.

### Local LLM with Ollama
We selected Ollama to run a local `llama3.1` (or `llama3.1:8b`) model, thus avoiding external API calls.

### Prompt in French
Since the dataset is in French, the prompt remains in French. The docstrings are also in French for clarity.

### Naive Parsing
If the model output contains “oui,” we classify as “Oui.” If it contains “non,” we classify as “Non.” This simple approach is often sufficient for this task.

### Performance
For each article, we do repeated sampling (8 “Oui” + 8 “Non”) from the training set. While CPU/GPU intensive, it provides better coverage and randomization.

---
