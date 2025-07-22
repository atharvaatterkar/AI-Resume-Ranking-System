# AI Resume Ranking System 🔍📄

This repository contains an AI agent that ranks resumes based on relevance to a given job description using natural language processing, semantic similarity, and entity extraction.

---

## 🚀 Features

### Resume Ranking System
- 📄 PDF text extraction from resume files using `PyMuPDF`
- 🔍 NLP preprocessing using `spaCy` (tokenization, lemmatization, stopword removal)
- 🤖 Skill extraction via `BERT-based Named Entity Recognition (dslim/bert-base-NER)`
- 🧠 Semantic similarity scoring using `Sentence Transformers (MiniLM-L6-v2)`
- 📊 Multi-factor scoring algorithm combining:
  - Semantic similarity
  - Skill match (NER + keyword)
  - Experience (extracted via regex)
  - Education detection (degree keywords)
- 🔗 Section clustering using `Agglomerative Clustering` for better contextual analysis

---

## 🛠️ Technology Stack

- **PyMuPDF** – PDF parsing and text extraction  
- **spaCy** – Text preprocessing (NLP pipeline)  
- **Transformers** – BERT-based NER model (`dslim/bert-base-NER`)  
- **Sentence Transformers** – Semantic embeddings (`all-MiniLM-L6-v2`)  
- **scikit-learn** – Agglomerative clustering and cosine similarity  
- **Pandas** – Tabular results for resume scoring  
- **NumPy** – Vector operations  

---

## 📁 Project Structure

```
Main/
├── Ranking.ipynb               # Resume ranking notebook
└── data/                       # Sample resume PDFs (add your own here)
    ├── resume-01.pdf
    ├── resume-02.pdf
    ├── resume-03.pdf
    └── resume-04.pdf
requirements.txt                # Python dependencies
README.md                       # Project documentation
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8+
- Internet access (to download models)

---

### 🔧 Installation Steps

1. **Clone the repository**

```bash
git clone <repository-url>
cd resume-ranking-system
```

2. **Create virtual environment**

```bash
python -m venv venv
venv\Scripts\activate       # Windows
# or
source venv/bin/activate    # macOS/Linux
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download spaCy model**

```bash
python -m spacy download en_core_web_sm
```

---

## 🧪 Usage

### Running the Resume Ranking System

1. Open the notebook:

```bash
jupyter notebook Main/Ranking.ipynb
```

2. Define your job description in the notebook:

```python
JD_TEXT = """We are hiring a Data Science Intern with experience in 
NLP, Machine Learning and Deep Learning..."""
```

3. Place resume PDFs in the `Main/data/` directory.

4. Run the notebook cells. You will get a ranked table like:

```
filename        score
resume-03.pdf   0.283
resume-01.pdf   0.274
resume-02.pdf   0.261
```

---

## 🧠 Project's Working

### Resume Ranking Algorithm

- **Text Extraction**: Extracts resume content using `PyMuPDF`  
- **Preprocessing**: Cleans and lemmatizes text using `spaCy`  
- **Section Detection**: Groups similar content using `Agglomerative Clustering`  
- **Skill Extraction**: Combines BERT-based NER and keyword-matching  
- **Semantic Matching**: Measures similarity to JD using `Sentence Transformers`  
- **Scoring**: Computes final rank score based on:
  - Semantic match (**alpha**)
  - Skill overlap (**beta**)
  - Experience in years (**gamma**)
  - Education flag (**delta**)

---

## 🔧 Configuration

### Tuning Resume Ranking

You can adjust the weights in the scoring function:

```python
score = alpha * semantic_score + beta * skill_score + gamma * experience_score + delta * education_flag
```

| Weight | Meaning                     | Default |
|--------|-----------------------------|---------|
| alpha  | Semantic match              | 0.5     |
| beta   | Skill overlap               | 0.3     |
| gamma  | Experience in years         | 0.1     |
| delta  | Education presence (degree) | 0.1     |

---

