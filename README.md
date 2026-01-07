🏥 Medical Report Summarization & Clinical Entity Extraction
📌 Overview

This project focuses on automatically summarizing medical reports, extracting clinically relevant entities, and linking medical terms to meaningful information to improve readability for non-expert users.
It combines state-of-the-art transformer models for abstractive summarization with biomedical Named Entity Recognition (NER) techniques.

The system is designed to handle long, unstructured clinical text while preserving medical accuracy and interpretability.

🎯 Key Objectives

Condense lengthy medical reports into concise, human-readable summaries

Extract medical entities such as diseases, symptoms, drugs, and procedures

Compare multiple summarization approaches using quantitative evaluation metrics

Improve accessibility of clinical documentation for patients and non-technical users

🧠 Models & Techniques Used
🔹 Summarization

PEGASUS – Transformer-based abstractive summarization model optimized for long documents

BERT-based summarization – Extractive/abstractive experimentation for comparison

🔹 Named Entity Recognition

BioBERT – Pretrained biomedical language model fine-tuned for clinical NER

Entity extraction focused on medical terminology present in reports

🗂️ Repository Structure
Medical_Report_Summarization/
│
├── BERT_summary/                  # BERT-based summarization outputs
├── Pegasus_summary/               # PEGASUS-generated summaries
├── NER_output/                    # Extracted medical entities
│
├── NER.ipynb                      # Named Entity Recognition notebook
├── Pegasus.py                     # PEGASUS summarization pipeline
├── bioBERT2.py                    # BioBERT-based NER implementation
├── extract_information2.py        # Medical term extraction & linking logic
│
├── summary_Scores.xlsx            # Evaluation metrics (BERT summaries)
├── summary_scores_pegasus.xlsx    # Evaluation metrics (PEGASUS summaries)
├── Medical Report Summarization.pptx  # Project presentation
└── README.md

📊 Evaluation

Summaries were evaluated using standard NLP summarization metrics, with results stored in Excel for comparison:

ROUGE-1

ROUGE-2

ROUGE-L

This enables direct benchmarking between BERT-based and PEGASUS-based approaches.

⚙️ How to Run
1️⃣ Install Dependencies
pip install transformers torch pandas nltk scikit-learn

2️⃣ Run PEGASUS Summarization
python Pegasus.py

3️⃣ Run Named Entity Recognition
python bioBERT2.py


or explore interactively via:

jupyter notebook NER.ipynb

🚀 Results & Insights

PEGASUS outperformed BERT in generating fluent and context-aware summaries for long medical documents

BioBERT effectively identified domain-specific medical entities

Combining summarization + NER improved interpretability and usability of clinical text

🧩 Future Improvements

Integrate RAG (Retrieval-Augmented Generation) for grounded summaries

Add clinical concept linking using UMLS or SNOMED

Build a web interface for real-time report summarization

Deploy as an API service using FastAPI

👤 Author

Prithiv Rajkumar
MS in Data Science
Focus: NLP, Healthcare AI, LLMs, Applied Machine Learning
