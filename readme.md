# AmbedkarGPT - AI Intern Assessment

This repository contains the complete submission for the AI Intern hiring tasks (Phase 1 & Phase 2).

The project implements a **Retrieval-Augmented Generation (RAG)** system capable of answering questions based on a corpus of Dr. B.R. Ambedkar's speeches. It also includes a comprehensive **Evaluation Framework** to quantitatively assess the system's performance under different configurations.

---

## 📋 Project Overview

### Phase 1: Functional Prototype
A command-line Q&A system built using Python and LangChain. It ingests text documents from a single file ([`speech.txt`](speech.txt)), creates vector embeddings, and uses a local LLM (Mistral via Ollama) to answer user queries based on the retrieved context.

### Phase 2: Evaluation Framework
An automated testing suite that evaluates the RAG system's accuracy across a larger corpus ([`corpus/`](corpus/)). It measures performance using multiple metrics (Retrieval Accuracy, ROUGE, BLEU, Cosine Similarity) and compares three different chunking strategies to identify the optimal configuration for this dataset.

---

## 🛠️ Technical Stack

*   **Language:** Python 3.8+
*   **LLM:** Ollama (`mistral` model)
*   **Framework:** LangChain
*   **Vector Store:** ChromaDB
*   **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
*   **Evaluation Libraries:** `rouge-score`, `nltk`, `scikit-learn`

---

## 🚀 Setup & Installation

### 1. Prerequisites
Ensure you have [Ollama](https://ollama.ai/) installed and running on your local machine.

Pull the required LLM model by running:
```bash
ollama pull mistral
```

### 2. Clone & Install Dependencies
Clone this repository and install the necessary Python packages:

```bash
git clone <your-repo-url>
cd AmbedkarGPT-Intern-Task
pip install -r requirements.txt
```

### 3. Data Verification
Ensure the following files and directories are present in the project root:
*   `corpus/`: A folder containing the text files (`speech1.txt` to `speech6.txt`).
*   [`test_dataset.json`](test_dataset.json): The JSON file with questions and ground truths for evaluation.
*   [`speech.txt`](speech.txt): Required for running the Phase 1 prototype.

---

## 💻 Usage Instructions

### Task 1: Run the Chatbot (Prototype)
To interact with the basic Q&A system which uses [`speech.txt`](speech.txt) as its knowledge base:

```bash
python main.py
```
The system will initialize and prompt you for questions. Type `exit` to quit.

### Task 2: Run the Evaluation Framework
To run the automated performance benchmarks on the full corpus:

```bash
python evaluation.py
```
This script will:
1.  Load all documents from the `corpus/` directory.
2.  Test three chunking strategies: **Small** (250 chars), **Medium** (550 chars), and **Large** (900 chars).
3.  Run all questions from [`test_dataset.json`](test_dataset.json) against each strategy.
4.  Calculate metrics: Hit Rate, MRR, ROUGE-L, BLEU, and Cosine Similarity.
5.  Save the detailed raw results to [`test_results.json`](test_results.json).

---

## 📊 Evaluation Results

A detailed analysis of the evaluation run can be found in [**results_analysis.md**](results_analysis.md).

### Summary of Findings:

*   **Optimal Strategy:** **Medium Chunking (550 chars)** proved to be the most effective.
*   **Key Metrics (Medium Strategy):**
    *   **Hit Rate:** 1.00
    *   **ROUGE-L:** 0.2948
    *   **BLEU Score:** 0.2145
*   **Analysis:** The medium chunk size provided the best balance between capturing sufficient context for the LLM and avoiding irrelevant noise. Small chunks fragmented ideas, while large chunks often included too much non-relevant text, degrading answer quality.

---

## 📂 Repository Structure

```
AmbedkarGPT-Intern-Task/
├── corpus/                  # Folder containing Dr. Ambedkar's speeches
├── main.py                  # Phase 1: Interactive Chatbot script
├── evaluation.py            # Phase 2: Automated Evaluation script
├── test_dataset.json        # The 25 Q&A pairs for evaluation
├── test_results.json        # Raw JSON output of the evaluation metrics
├── results_analysis.md      # Detailed report on findings and recommendations
├── requirements.txt         # Project dependencies
├── speech.txt               # Single speech file for the Phase 1 prototype
└── README.md                # This file
```