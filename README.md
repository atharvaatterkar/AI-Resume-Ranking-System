Factorial24
AI-Powered FAQ Chatbot & Resume Ranking System
This repository contains two main AI applications:

RAG-based FAQ Chatbot - An intelligent chatbot using Retrieval-Augmented Generation

Resume Ranking System - An AI agent for ranking resumes against job descriptions

🚀 Features
FAQ Chatbot
Conversational AI with memory using LangChain
Vector similarity search with FAISS and Cohere embeddings
Context-aware responses powered by Groq's LLaMA model
Interactive Streamlit interface
Source document retrieval for transparency
Resume Ranking System
PDF text extraction from resume files
NLP preprocessing with spaCy
Semantic similarity matching using Sentence Transformers
Named Entity Recognition for skill extraction
Multi-factor scoring algorithm combining semantic similarity, skills, and experience
Clustering-based section detection for better resume analysis
🛠️ Technology Stack
LangChain - LLM application framework
Streamlit - Web interface
FAISS - Vector database for similarity search
Cohere - Text embeddings
Groq - LLM inference (LLaMA 3 70B)
Sentence Transformers - Semantic embeddings
spaCy - Natural language processing
PyMuPDF - PDF text extraction
scikit-learn - Machine learning utilities
Pandas - Data manipulation
📁 Project Structure
.
├── rag/                                  # FAQ Chatbot Directory
│   ├── rag.py                        # Main Streamlit chatbot application
│   ├── test.ipynb                  # RAG chatbot development notebook
├── agent/
│   ├── test.ipynb                 # Resume ranking testing and evaluation
│   └── data/                        # Sample resume PDFs
│       ├── resume-01.pdf
│       ├── resume-02.pdf
│       └── resume-03.pdf
│── requirements.txt          # Python dependencies for chatbot
├── README.md             # Project documentation
└── .env                            # Environment variables (create this)
⚙️ Setup & Installation
Prerequisites
Python 3.8+
API Keys for:
Groq (for LLM)
Cohere (for embeddings)
Installation Steps
Clone the repository

git clone <repository-url>
cd factorial24
Create virtual environment

python -m venv myenv
myenv\Scripts\activate  
Install dependencies

pip install -r requirements.txt
Download spaCy model

python -m spacy download en_core_web_sm
Set up environment variables Create a .env file in the root directory:

GROQ_API_KEY=your_groq_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
Usage
Running the FAQ Chatbot
Start the Streamlit application in rag/ directory

streamlit run rag.py
Open your browser eg. http://localhost:8501

Using the Resume Ranking System
Open the Jupyter notebooks in the agent/ directory:

jupyter notebook agent/test.ipynb
Place resume PDFs in the agent/data/ directory

Define your job description in the notebook

Run the ranking algorithm to get scored and ranked resumes

Example Usage
FAQ Chatbot:

User: "What services do you provide?"
Bot: "We offer cloud platforms, data analytics, and cybersecurity solutions..."
Resume Ranking:

JD_TEXT = """We are hiring a Data Science Intern with experience in 
NLP, Machine Learning and Deep Learning..."""

# Results show ranked resumes with scores
filename        score
resume-03.pdf   0.283
resume-01.pdf   0.274
resume-02.pdf   0.261
Project's Working
RAG Chatbot Architecture
Document Processing: FAQ data is converted to LangChain documents
Chunking: Text is split into manageable chunks
Embedding: Cohere creates vector embeddings for semantic search
Vector Storage: FAISS stores embeddings for fast retrieval
Query Processing: User questions are embedded and matched with relevant chunks
Response Generation: Groq's LLaMA model generates contextual responses
Memory Management: Conversation history is maintained for context
Resume Ranking Algorithm
Text Extraction: PDF content extracted using PyMuPDF
Preprocessing: Text cleaning, tokenization, and lemmatization with spaCy
Section Detection: Agglomerative clustering groups similar content
Skill Extraction: Named Entity Recognition identifies relevant skills
Semantic Matching: Sentence Transformers compute job description similarity
Multi-factor Scoring: Combines semantic similarity, skill matches, and experience
Configuration
Customizing the Chatbot
Modify RAW_FAQS in rag.py to add your own FAQ data
Adjust LLM parameters (temperature, max_tokens) for different response styles
Change embedding models in the Cohere configuration
Tuning Resume Ranking
Adjust scoring weights in score_resume function:
alpha: Semantic similarity weight
beta: Skill matching weight
gamma: Experience factor weight
Modify clustering parameters for better section detection
Add custom skill extraction patterns.
🔗 Resources
LangChain Documentation
Streamlit Documentation
Cohere API Documentation
Groq API Documentation
Sentence Transformers
