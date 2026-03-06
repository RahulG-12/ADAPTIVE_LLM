# Adaptive AI Intelligence Platform

An advanced AI system that combines **Retrieval-Augmented Generation (RAG)**, **Hybrid Retrieval**, and **Hallucination Detection** to improve the reliability and accuracy of Large Language Model (LLM) responses.

This platform demonstrates how modern AI systems integrate **information retrieval, generative models, and validation mechanisms** to build more trustworthy AI applications.

---

## Project Overview

Large Language Models often generate responses that are **incorrect or hallucinated** when they lack external knowledge.

This system solves that problem by:

1. Retrieving relevant documents from a knowledge base.
2. Injecting context into the LLM.
3. Generating grounded responses.
4. Detecting potential hallucinations.
5. Monitoring system behavior through a dashboard.

---

## Key Features

### Hybrid Retrieval System
Combines two retrieval approaches:

- **Dense Retrieval** using vector embeddings
- **Sparse Retrieval** using BM25 ranking

These results are combined to improve search accuracy.

---

### RAG (Retrieval-Augmented Generation)

Instead of answering from model memory, the system:

1. Searches knowledge base
2. Retrieves relevant documents
3. Injects context into prompt
4. Generates grounded response

This significantly improves reliability.

---

### Hallucination Detection

A classifier analyzes generated responses and predicts whether the response may be hallucinated or unsupported by retrieved context.

This helps identify unreliable answers.

---

### AI Monitoring Dashboard

A Streamlit dashboard provides insights into:

- Retrieved documents
- Retrieval scores
- Response output
- Hallucination detection status
- System latency

This simulates **AI observability tools used in production systems**.

---

## System Architecture

```
User Query
     │
     ▼
Hybrid Retrieval
(Dense + Sparse Search)
     │
     ▼
Top Relevant Documents
     │
     ▼
LLM Response Generation
     │
     ▼
Hallucination Detection
     │
     ▼
Monitoring Dashboard
```

---

## Tech Stack

### Programming
Python

### Machine Learning
Transformers  
Sentence Transformers  
Scikit-learn

### Retrieval
FAISS  
BM25

### Backend
FastAPI

### Dashboard
Streamlit  
Plotly

### Data Processing
Pandas  
NumPy

---

## Project Structure

```
adaptive_llm_platform
│
├── retrieval
│   ├── dense_retriever.py
│   ├── sparse_retriever.py
│   ├── hybrid.py
│   └── reranker.py
│
├── fine_tuning
│   ├── train.py
│   └── inference.py
│
├── hallucination
│   └── train_classifier.py
│
├── data
│   └── dataset.csv
│
├── integrated_system.py
├── app_dashboard.py
└── requirements.txt
```

---

## Installation

Clone the repository

```
git clone https://github.com/RahulG-12/ADAPTIVE_LLM.git
cd ADAPTIVE_LLM
```

Create virtual environment

```
python -m venv venv
```

Activate environment

Windows
```
venv\Scripts\activate
```

Install dependencies

```
pip install -r requirements.txt
```

---

## Run the System

Start the Streamlit dashboard

```
streamlit run app_dashboard.py
```

The dashboard will open at:

```
http://localhost:8501
```

---

## Example Query

```
What is Retrieval-Augmented Generation?
```

The system will:

1. Retrieve relevant documents
2. Generate LLM response
3. Detect hallucination probability
4. Display retrieval scores

---

## Future Improvements

- Vector database integration
- Multi-document reasoning
- Reinforcement learning feedback
- Production deployment pipeline
- Distributed retrieval system

---

🎥 Demo
 The project is demonstrated via recorded execution.

🔗 Demo Link: https://drive.google.com/file/d/1QTF95gs5q0tHJd4YTu10LPUkpINsVIoQ/view?usp=sharing



## Author

Rahul Giri  
AI / ML Engineer  
Mumbai, India

GitHub: https://github.com/RahulG-12

---

## License

This project is for educational and research purposes.
