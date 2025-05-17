# NLP_driven_resume_ranking_system
nlp driven resume ranking system using spacy transformer model and sbert model for automates the candidate screening process


Resume Ranking System
  
The Resume Ranking System is a web-based application designed to assist HR administrators in efficiently ranking candidate resumes based on their relevance to a job description. Leveraging natural language processing (NLP) and machine learning, this system automates the resume screening process, saving time and ensuring a more objective evaluation of candidates.
Table of Contents

What We Do
How We Do It
Features
Tech Stack
Installation
Usage
Project Structure
Training the Model
Evaluation Metrics
Future Improvements
Contributing
License

What We Do
The Resume Ranking System streamlines the hiring process by automatically ranking candidate resumes based on their alignment with a given job description. It extracts key information from resumes (e.g., experience, skills, certifications, projects) and computes a relevance score using a hybrid approach that combines rule-based scoring and semantic similarity. HR administrators can upload resumes, view ranked candidates, filter results, and visualize skill distributions through an intuitive web interface.
Our goal is to reduce manual effort in resume screening, minimize bias, and help recruiters quickly identify the best candidates for a role.
How We Do It
The system follows a structured pipeline to process and rank resumes:

Resume Upload and Storage:

Resumes (PDF or DOCX) are uploaded via a web interface with an associated job ID.
Files are stored in a MongoDB database with metadata like filename, content type, and upload timestamp.


Text Extraction:

Text is extracted from resumes using PyMuPDF for PDFs and python-docx for DOCX files.
For scanned PDFs, PyTesseract performs OCR to extract text from images.


Feature Extraction:

spaCy (with models en_core_web_trf and en_core_web_sm) is used for NLP tasks, extracting entities such as:
Candidate name
Experience (in years)
Certifications
Projects
Skills


A RoleAnalyzer class evaluates the relevance of roles and companies mentioned in the resume.


Scoring:

Rule-Based Scoring: Assigns scores based on weighted factors (e.g., experience: 5 points/year, certifications: 3 points each, projects: 2 points each, skills: category-specific weights). This contributes 70% to the final score.
Semantic Similarity: A fine-tuned SentenceTransformer model (all-MiniLM-L6-v2) computes the cosine similarity between resume and job description embeddings. This contributes 30% to the final score.
A combined score is calculated as:Combined_Score = (Rule_Based_Score * 0.7) + (SBERT_Score * 0.3)


Ranking and Filtering:

Resumes are ranked by their combined scores in descending order.
HR admins can filter results by rank range or upload date via the web interface.


Visualization:

The frontend displays a ranked table of candidates with details like scores, experience, and skills.
Charts (pie and bar) visualize skill distributions among top candidates using Chart.js.


Logging and Monitoring:

A logging system (resume_processing.log) tracks processing activities, errors, and resource usage (CPU, memory) using psutil.



Features

Automated Resume Ranking: Ranks candidates based on relevance to the job description using a hybrid scoring model.
Resume Upload/Download: Supports PDF and DOCX formats with secure storage in MongoDB.
NLP-Powered Extraction: Extracts key entities (experience, skills, certifications, projects) using spaCy.
Semantic Similarity: Uses a fine-tuned SentenceTransformer model to measure semantic alignment between resumes and job descriptions.
Interactive Web Interface: Built with Bootstrap and JavaScript, offering filtering, sorting, and visualizations.
Efficient Caching: Stores processed rankings in MongoDB for quick retrieval.
Comprehensive Logging: Logs processing details and resource usage for debugging and performance monitoring.

Tech Stack

Frontend:
HTML, CSS, JavaScript
Bootstrap (for responsive design)
Chart.js (for visualizations)


Backend:
Python 3.8+
Flask (web framework)
PyMuPDF (PDF text extraction)
python-docx (DOCX text extraction)
PyTesseract (OCR for scanned PDFs)


NLP and Machine Learning:
spaCy (entity extraction, models: en_core_web_trf, en_core_web_sm)
SentenceTransformer (all-MiniLM-L6-v2 for semantic similarity)


Database:
MongoDB (via PyMongo)


Other Libraries:
pandas (data manipulation)
psutil (resource monitoring)
numpy, torch, scipy (for computations)



Installation
Follow these steps to set up the project locally:
Prerequisites

Python 3.8 or higher
MongoDB (running locally or accessible remotely)
Tesseract-OCR (for OCR functionality, install via apt-get on Linux or download for Windows/Mac)

Steps

Clone the Repository:
git clone https://github.com/yourusername/resume-ranking-system.git
cd resume-ranking-system


Set Up a Virtual Environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt

Example requirements.txt:
flask
pymupdf
python-docx
pytesseract
spacy
sentence-transformers
pandas
psutil
pymongo
numpy
torch
scipy
matplotlib
seaborn


Download spaCy Models:
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_sm


Set Up MongoDB:

Ensure MongoDB is running (mongod).
Update the database connection settings in DATABASE/db.py with your MongoDB URI if needed.


Run the Application:
python app.py

The application will be available at http://localhost:5000.


Usage

Access the Web Interface:

Open http://localhost:5000 in your browser.
Navigate to the upload page to submit resumes (PDF or DOCX) with a job ID (e.g., python_backend_developer).


Upload Resumes:

Select resumes and provide a job description.
The system will process the resumes and store them in MongoDB.


View Rankings:

Go to the ranking page for a specific job ID (e.g., /ranking/python_backend_developer).
View the ranked list of candidates with their combined scores, experience, skills, and more.
Use filters to narrow down results by rank or upload date.


Visualize Skills:

Check the pie and bar charts to see skill distributions among the top candidates.


Download Resumes:

Download individual resumes by their ID via the provided endpoint.



Project Structure
resume-ranking-system/
│
├── app.py                    # Flask application entry point
├── resume_shortlister.py     # Core logic for resume processing and ranking
├── DATABASE/
│   └── db.py                 # MongoDB connection setup
├── KEYWORDS/
│   └── NLP4.csv             # Skill keywords for matching
├── TEMPORARY/               # Temporary directory for OCR processing
├── static/                  # Static files (CSS, JS, images)
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── ranking_script.js
├── templates/               # HTML templates
│   ├── ranking_page.html    # Ranking page template
│   └── upload.html          # Upload page template
├── resume_processing.log    # Log file for processing activities
├── ranking_cache.pkl        # Cache file for processed rankings
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation

Training the Model
The SentenceTransformer model (all-MiniLM-L6-v2) was fine-tuned using a dataset of 8000 resume-job description pairs, each with a target relevance score between 0 and 1. The training process involved:

Dataset Format:

Each data point: {"resume": "...", "job_desc": "...", "score": float}
Example: {"resume": "Content Writer specializing in SEO, blogs, and copywriting.", "job_desc": "Seeking a cloud engineer with AWS, Azure, and cloud architecture skills.", "score": 0.47}


Training Setup:

The model was fine-tuned to minimize the difference between the cosine similarity of resume-job description embeddings and the target score.
Training was performed on 80% of the data (6400 samples), with 20% (1600 samples) reserved for testing.


Training Code (example):
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer('all-MiniLM-L6-v2')
train_examples = [InputExample(texts=[data["resume"], data["job_desc"]], label=data["score"]) for data in train_data]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=4, warmup_steps=100)
model.save("path/to/fine-tuned-model")



Evaluation Metrics
The fine-tuned SentenceTransformer model was initially evaluated on a small test set of 2 samples, with plans to expand to the full test set (1600 samples). The evaluation metrics are:

Mean Absolute Error (MAE): 0.2432The predicted scores deviated from the actual scores by about 24% on average, indicating good but improvable performance.
Mean Squared Error (MSE): 0.0598The squared error was low, showing that larger errors were minimal.
Pearson and Spearman Correlations: Both 1.0These metrics suggest perfect linear and rank correlation, but they are not meaningful due to the small test set size (only 2 samples).

A larger evaluation on 1600 samples is planned to obtain reliable correlation metrics, which are critical for assessing ranking performance.
Future Improvements

Expand Test Set Evaluation: Evaluate the model on the full test set (1600 samples) to obtain meaningful correlation metrics for ranking performance.
Model Fine-Tuning: Reduce the MAE (currently 0.2432) by experimenting with different loss functions, learning rates, or additional training epochs.
Enhanced Filtering: Add more filtering options (e.g., by specific skills or experience levels) to the web interface.
Batch Download: Implement a feature to download all filtered resumes as a ZIP file.
Scalability: Optimize the system for larger datasets by implementing parallel processing for resume analysis.
User Authentication: Add authentication for HR admins to secure access to the system.

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Make your changes and commit (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

Please ensure your code follows PEP 8 guidelines and includes appropriate tests.
License
This project is licensed under the Apache License. See the LICENSE file for details.

