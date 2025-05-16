import fitz  # PyMuPDF
import docx
import io
import os
import base64
import pandas as pd
from spacy.matcher import PhraseMatcher
from datetime import datetime
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta
from collections import Counter
import re
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from pdf2image import convert_from_path
import pytesseract
import pickle
import logging
import time
import psutil  # For resource monitoring
import shutil
from DATABASE.db import get_database

# Set up logging
logging.basicConfig(
    filename='resume_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Configuration
KEYWORD_FILE = 'KEYWORDS/NLP4.csv'
CACHE_FILE = 'ranking_cache.pkl'
TEMP_DIR = 'TEMPORARY'
BYPASS_CACHE = True  # Set to True to bypass cache for debugging

# MongoDB Configuration
db, client = get_database()
DB_NAME = "Resume_ranking"
COLLECTION_NAME = "Resumes"
TEMP_RANKING_COLLECTION_NAME = "TempRankings"
TEMP_COLLECTION_NAME = "temp_collection"  # Add temp_collection to MongoDB configuration
temp_collection = db[TEMP_COLLECTION_NAME]

# Scoring weights
SCORE_WEIGHTS = {
    "experience": 5,  # per year
    "certification": 3,  # per cert
    "project": 2,  # per project
    "role_relevance": 3,  # multiplier for role score
    "skills": {  # skill category weights
        "Statistics": 8,
        "Machine Learning": 9,
        "Deep Learning": 10,
        "R Language": 5,
        "Python Language": 5,
        "NLP": 10,
        "Data Engineering": 4,
        "Web Development": 2
    }
}

# MongoDB Connection
try:
    db, client = get_database()
    collection = db[COLLECTION_NAME]
    temp_ranking_collection = db[TEMP_RANKING_COLLECTION_NAME]
    logging.info("Connected to MongoDB using db_connection")
except Exception as e:
    logging.error(f"Failed to connect to MongoDB: {str(e)}")
    raise

# Resource Monitoring
def log_resource_usage(step_name):
    process = psutil.Process()
    memory_info = process.memory_info()
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_used = memory_info.rss / (1024 * 1024)  # Convert to MB
    logging.info(f"Resource usage at {step_name}: CPU={cpu_percent}%, Memory={memory_used:.2f}MB")

class RoleAnalyzer:
    def __init__(self):
        self.company_weights = {
            "google": 1.5, "microsoft": 1.4, "amazon": 1.4, "facebook": 1.5,
            "apple": 1.4, "ibm": 1.3, "oracle": 1.2, "netflix": 1.4,
            "twitter": 1.3, "linkedin": 1.3, "nvidia": 1.4, "tesla": 1.4,
            "stanford": 1.3, "mit": 1.3, "berkeley": 1.2
        }
        self.role_weights = {
            "engineer": 1.3, "scientist": 1.5, "researcher": 1.4,
            "developer": 1.2, "manager": 1.3, "director": 1.6,
            "architect": 1.4, "specialist": 1.2, "intern": 0.7,
            "assistant": 0.8, "analyst": 1.1, "consultant": 1.2
        }
        self.role_keywords = {
            "machine learning": ["machine learning", "ml", "ai engineer", "deep learning"],
            "data science": ["data scientist", "data analyst", "data engineer"],
            "software": ["software engineer", "backend", "frontend", "full stack"],
            "research": ["research scientist", "research engineer", "research assistant"]
        }

    def extract_roles(self, text, nlp):
        doc = nlp(text)
        roles = []
        # Use ROLE entity if available
        for ent in doc.ents:
            if ent.label_ == "ROLE":
                # Look for nearby ORG entity to associate company
                company = None
                for ent2 in doc.ents:
                    if ent2.label_ == "ORG":
                        # Check proximity (within 50 characters)
                        if abs(ent.start_char - ent2.start_char) < 50:
                            company = ent2.text.lower()
                            break
                roles.append({
                    'title': ent.text.lower(),
                    'company': company if company else "unknown"
                })
        # Fallback to regex if no ROLE entities are found
        if not roles:
            title_patterns = [
                r"(?P<title>[A-Za-z ]+?)\s(?:at|in|@)\s(?P<company>[A-Za-z0-9 &]+)",
                r"(?P<title>[A-Za-z ]+?),\s*(?:at\s*)?(?P<company>[A-Za-z0-9 &]+)",
                r"(?P<company>[A-Za-z0-9 &]+)\s*[-–]\s*(?P<title>[A-Za-z ]+)"
            ]
            for pattern in title_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    roles.append({
                        'title': match.group('title').strip().lower(),
                        'company': match.group('company').strip().lower()
                    })
        unique_roles = []
        seen = set()
        for role in roles:
            key = (role['title'], role['company'])
            if key not in seen:
                seen.add(key)
                unique_roles.append(role)
        return unique_roles

    def calculate_role_score(self, roles):
        total_score = 0
        role_details = []
        for role in roles:
            company = role['company']
            company_weight = 1.0
            for known_company, weight in self.company_weights.items():
                if known_company in company:
                    company_weight = weight
                    break
            title = role['title']
            role_weight = 1.0
            for role_type, keywords in self.role_keywords.items():
                if any(keyword in title for keyword in keywords):
                    role_weight = 1.5
                    break
            if "senior" in title:
                role_weight *= 1.3
            elif "junior" in title:
                role_weight *= 0.8
            elif any(word in title for word in ["lead", "principal", "head"]):
                role_weight *= 1.5
            elif any(word in title for word in ["associate", "assistant"]):
                role_weight *= 0.9
            for role_key, weight in self.role_weights.items():
                if role_key in title:
                    role_weight *= weight
                    break
            role_score = company_weight * role_weight
            total_score += role_score
            role_details.append({
                'title': role['title'],
                'company': role['company'],
                'score': role_score
            })
        return {
            'total_score': total_score,
            'role_details': role_details,
            'normalized_score': min(10, total_score)
        }

def extract_candidate_name(text, nlp):
    import re
    import logging

    # Step 1: Remove extension
    text = re.sub(r'\.(pdf|docx|doc)', '', text, flags=re.IGNORECASE)

    # Step 2: Replace underscores and hyphens with spaces
    text = re.sub(r'[_\-]', ' ', text)

    # Step 3: Remove common resume-related words and digits
    text = re.sub(r'\b(resume|cv|cts|infosys|tcs|wipro|techmahindra|accenture)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+', '', text)

    # Step 4: Split camel case
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Step 5: Normalize whitespace
    cleaned_text = text.strip()
    logging.info(f"Cleaned filename text: '{cleaned_text}'")

    # Step 6: Use spaCy NER
    doc = nlp(cleaned_text[:500])
    for ent in doc.ents:
        if ent.label_ == "NAME":
            return ent.text.title()

    # Step 7: Fallback: first 1 or 2 words
    parts = cleaned_text.split()
    if len(parts) >= 2:
        fallback_name = f"{parts[0].title()} {parts[1].title()}"
    elif len(parts) == 1:
        fallback_name = parts[0].title()
    else:
        fallback_name = "Unknown Candidate"

    logging.warning(f"No NAME entity found, falling back to: {fallback_name}")
    return fallback_name

def cleanup_temp_files():
    try:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            logging.info(f"Cleaned up temporary directory: {TEMP_DIR}")
    except Exception as e:
        logging.error(f"Error cleaning up temporary directory: {str(e)}")

def extract_text(binary_data, content_type):
    if 'pdf' in content_type.lower():
        return extract_text_from_pdf_binary(binary_data)
    elif 'wordprocessingml.document' in content_type.lower():
        return extract_text_from_docx_binary(binary_data)
    else:
        logging.warning(f"Unsupported content type: {content_type}")
        return ""

def extract_text_from_pdf_binary(binary_data):
    try:
        file_stream = io.BytesIO(binary_data)
        text = ""
        with fitz.open(stream=file_stream, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text() or ""
        if text.strip():
            logging.info(f"Successfully extracted text from PDF binary, length: {len(text)}")
            print(text)
            return text.replace('\n', ' ').replace('\r', ' ')
        else:
            logging.info("No text extracted with PyMuPDF, attempting OCR")
            os.makedirs(TEMP_DIR, exist_ok=True)
            temp_pdf = os.path.join(TEMP_DIR, "temp.pdf")
            with open(temp_pdf, "wb") as f:
                f.write(binary_data)
            images = convert_from_path(temp_pdf)
            text = ""
            for image in images:
                text += pytesseract.image_to_string(image) or ""
            print(text)
            logging.info(f"Extracted text from PDF using OCR, length: {len(text)}")
            return text.replace('\n', ' ').replace('\r', ' ')
    except Exception as e:
        logging.error(f"Error reading PDF binary with PyMuPDF: {str(e)}", exc_info=True)
        return ""

def extract_text_from_docx_binary(binary_data):
    try:
        file_stream = io.BytesIO(binary_data)
        doc = docx.Document(file_stream)
        text = ' '.join([para.text for para in doc.paragraphs if para.text.strip()])
        logging.info(f"Successfully extracted text from DOCX binary, length: {len(text)}")
        print(text)
        return text.replace('\n', ' ').replace('\r', ' ')
    except Exception as e:
        logging.error(f"Error reading DOCX binary: {str(e)}", exc_info=True)
        return ""

def detect_experience(text, nlp):
    doc = nlp(text.lower())
    total_years = 0

    # Use EXPERIENCE entity if available
    for ent in doc.ents:
        if ent.label_ == "EXPERIENCE":
            # Look for numerical values or date ranges in the entity text
            years = re.search(r'(\d+)\s+(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)', ent.text)
            if years:
                total_years += int(years.group(1))
                logging.info(f"Found experience via EXPERIENCE entity: {ent.text}, {years.group(1)} years")
            # Look for date ranges within the entity
            range_pattern = r'''
                (?:                         
                    \b                      
                    (?:jan|feb|mar|apr|may|jun|  
                     jul|aug|sep|oct|nov|dec|
                     january|february|march|april|may|june|
                     july|august|september|october|november|december)
                    [a-z]*                  
                    \s+                     
                    (?:19|20)?\d{2}         
                    \b                      
                )
            '''
            date_range = re.search(
                rf'({range_pattern})\s*(?:-|to|–)\s*({range_pattern}|present|current|now)',
                ent.text, re.VERBOSE | re.IGNORECASE
            )
            if date_range:
                try:
                    start_date = parse(date_range.group(1), fuzzy=True)
                    end_str = date_range.group(2).lower()
                    if any(x in end_str for x in ['present', 'current', 'now']):
                        end_date = datetime.now()
                    else:
                        end_date = parse(end_str, fuzzy=True)
                    delta = relativedelta(end_date, start_date)
                    years = delta.years + delta.months / 12
                    total_years += years
                    logging.info(f"Found experience via EXPERIENCE entity date range: {ent.text}, {years:.1f} years")
                except Exception as e:
                    logging.warning(f"Failed to parse date range in EXPERIENCE entity: {ent.text}, error: {str(e)}")

    # Fallback to regex if no EXPERIENCE entities are found
    if total_years == 0:
        year_matches = re.findall(r'(\d+)\s+(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)', text)
        if year_matches:
            total_years = sum(int(y) for y in year_matches)
            logging.info(f"Found experience via regex: {total_years} years")

        duration_pattern = r'''
            (?:                         
                \b                      
                (?:jan|feb|mar|apr|may|jun|  
                 jul|aug|sep|oct|nov|dec|
                 january|february|march|april|may|june|
                 july|august|september|october|november|december)
                [a-z]*                  
                \s+                     
                (?:19|20)?\d{2}         
                \b                      
            )
        '''
        date_pattern = re.compile(duration_pattern, re.VERBOSE | re.IGNORECASE)
        dates = date_pattern.findall(text)
        range_pattern = re.compile(
            rf'({duration_pattern})\s*(?:-|to|–)\s*({duration_pattern}|present|current|now)',
            re.VERBOSE | re.IGNORECASE
        )
        for start, end in range_pattern.findall(text):
            try:
                start_date = parse(start, fuzzy=True)
                if any(x in end.lower() for x in ['present', 'current', 'now']):
                    end_date = datetime.now()
                else:
                    end_date = parse(end, fuzzy=True)
                delta = relativedelta(end_date, start_date)
                total_years += delta.years + delta.months / 12
                logging.info(f"Found experience via regex date range: {start} to {end}, {delta.years + delta.months / 12:.1f} years")
            except:
                continue

    return min(10, max(0, total_years))

def count_certifications(text, nlp):
    doc = nlp(text.lower())
    counts = set()  # Use a set to avoid duplicates

    # Use CERTIFICATION entity if available
    for ent in doc.ents:
        if ent.label_ == "CERTIFICATION":
            counts.add(ent.text)
            logging.info(f"Found certification via CERTIFICATION entity: {ent.text}")

    # Fallback to regex if no CERTIFICATION entities are found
    if not counts:
        cert_section = re.search(
            r'(certifications?|licenses?|qualifications?|credentials|courses|certificates)(.*?)(?=(education|experience|projects|skills|\n\s*\n|$))',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if cert_section:
            section_text = cert_section.group(2)
            items = re.findall(r'(•|\d+\.|-\s|\[.\]\s|✓\s)(.*?)(?=(•|\d+\.|-\s|\[.\]\s|✓\s|$))', section_text)
            for item in items:
                counts.add(item[1].strip())
                logging.info(f"Found certification via regex section: {item[1].strip()}")

        patterns = [
            r'\b(?:certified|licensed)\s+(?:in|as|for)\s+[a-z\s]+',
            r'\b(?:[a-z]+\s)?(?:certification|certificate|license|qualification)\b',
            r'\b(?:aws|google|microsoft|oracle|cisco)\s+certified\b',
            r'\b(?:passed|completed)\s+(?:the\s)?[a-z]+\s+(?:certification|exam)',
            r'\b(?:pmp|ccna|ccnp|awscsa|ocp|mcse|comptia)\b'
        ]
        verification_keywords = [
            'certification', 'certificate', 'license',
            'credential', 'accredited', 'validated'
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                context = text[max(0, match.start() - 20):min(len(text), match.end() + 20)]
                if any(kw in context for kw in verification_keywords):
                    counts.add(match.group().strip())
                    logging.info(f"Found certification via regex pattern: {match.group().strip()}")

        cert_abbreviations = {
            'pmp', 'ccna', 'ccnp', 'awscsa', 'ocp', 'mcse', 'comptia',
            'cissp', 'ceh', 'gcp', 'azure', 'ocpjp', 'ocpjd', 'rhce',
            'cfa', 'cfp', 'cia', 'cpa', 'cisco', 'aws', 'gcp', 'azure'
        }
        for abbrev in cert_abbreviations:
            if re.search(r'\b' + re.escape(abbrev) + r'\b', text):
                counts.add(abbrev)
                logging.info(f"Found certification via abbreviation: {abbrev}")

        issuing_bodies = {
            'aws', 'microsoft', 'oracle', 'cisco', 'google', 'comptia',
            'pmi', 'isc2', 'ecouncil', 'red hat', 'linux foundation',
            'sas', 'cloudera', 'databricks', 'snowflake', 'nptel'
        }
        for body in issuing_bodies:
            if re.search(r'\b' + re.escape(body) + r'\b', text):
                counts.add(body)
                logging.info(f"Found certification via issuing body: {body}")

        date_matches = re.findall(
            r'(?:certified|licensed|passed)\s+(?:in\s)?(?:19|20)\d{2}',
            text
        )
        for match in date_matches:
            counts.add(match)
            logging.info(f"Found certification via date match: {match}")

    return min(20, len(counts))

def count_projects(text, nlp):
    doc = nlp(text.lower())
    counts = set()  # Use a set to avoid duplicates

    # Use PROJECT entity if available
    for ent in doc.ents:
        if ent.label_ == "PROJECT":
            counts.add(ent.text)
            logging.info(f"Found project via PROJECT entity: {ent.text}")

    # Fallback to regex if no PROJECT entities are found
    if not counts:
        project_headers = [
            r'(projects?|portfolio|initiatives?|work\s+experience|selected\s+work|personal\s+work)',
            r'(?:^|\n)\s*#+\s*projects?\s*#*',
            r'(?:^|\n)\s*projects?\s*[:]'
        ]
        for header_pattern in project_headers:
            project_section = re.search(
                f'{header_pattern}(.*?)(?=(education|experience|certifications|skills|\\n\\s*\\n|$))',
                text,
                re.IGNORECASE | re.DOTALL
            )
            if project_section:
                section_text = project_section.group(2)
                items = re.findall(
                    r'(?:^|\n)\s*(?:[\•\-\*\+‣⁃]|\d+[\.\)]|\[[x\s]\]|✓)\s*(.+?)(?=\n\s*(?:[\•\-\*\+‣⁃]|\d+[\.\)]|\[[x\s]\]|✓|\w+\s+\d{4}|$))',
                    section_text
                )
                for item in items:
                    counts.add(item[0].strip())
                    logging.info(f"Found project via regex section: {item[0].strip()}")

        tech_keywords = [
            r'python', r'java', r'c\+\+', r'javascript', r'typescript',
            r'react', r'angular', r'vue', r'node\.?js', r'express',
            r'django', r'flask', r'spring', r'html', r'css',
            r'machine\s+learning', r'tensorflow', r'pytorch', r'keras',
            r'nlp', r'computer\s+vision', r'deep\s+learning',
            r'sql', r'mysql', r'postgres', r'mongodb', r'nosql',
            r'aws', r'azure', r'gcp', r'docker', r'kubernetes',
            r'mern', r'full\s+stack', r'web\s+app', r'web\s+application'
        ]
        project_indicators = [
            r'\b(?:developed|created|built|implemented|designed|constructed|programmed|made)\b',
            r'\b(?:project|initiative|application|system|website|portal|platform)\b',
            r'\bgithub\.com/[a-z0-9-]+/[a-z0-9-]+\b'
        ]
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            has_tech = any(re.search(tech, sentence) for tech in tech_keywords)
            has_indicator = any(re.search(ind, sentence) for ind in project_indicators)
            if has_tech and has_indicator:
                counts.add(sentence.strip())
                logging.info(f"Found project via regex sentence: {sentence.strip()}")

        title_matches = re.findall(
            r'(?:^|\n)\s*.+?\s*[-–]\s*(?:web\s+app|application|system|project)',
            text
        )
        for match in title_matches:
            counts.add(match.strip())
            logging.info(f"Found project via regex title: {match.strip()}")

        date_bound = re.findall(
            r'(?:^|\n)\s*.+?\s*\((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{4}\s*(?:to|-|–)\s*(?:present|now|current|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{4})\)',
            text, re.IGNORECASE
        )
        for match in date_bound:
            counts.add(match.strip())
            logging.info(f"Found project via regex date: {match.strip()}")

    return min(4, len(counts)) if counts else 0

def preprocess_text(text):
    text = re.sub(r'[\•\-\*\+‣⁃]', ' • ', text)
    text = re.sub(r'[,\;]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

def extract_sections_with_keywords(text, section_keywords, nlp):
    doc = nlp(text.lower())
    sections = []
    for sent in doc.sents:
        sent_text = sent.text.lower()
        if any(keyword in sent_text for keyword in section_keywords):
            next_sents = list(doc[sent.end:].sents)[:3]
            sections.append(" ".join([s.text for s in next_sents]))
    return sections

def is_valid_skill_context(span):
    if re.search(r'^[\•\-\*\+]\s*', span.sent.text):
        return True
    if len(span.sent) > 1 and any(t.text == ',' for t in span.sent):
        return True
    if any(keyword in span.sent.text.lower()
           for keyword in ['skills', 'technical', 'competencies']):
        return True
    return False

def is_skills_sentence(sent):
    skill_keywords = ['skills', 'technical', 'expertise', 'proficient', 'competencies']
    if any(keyword in sent.text.lower() for keyword in skill_keywords):
        return True
    if sum(1 for token in sent if token.text == ',') >= 2:
        return True
    return False

def extract_skills(text, keyword_df, nlp):
    if not hasattr(nlp, 'vocab') or nlp.vocab is None:
        logging.error("spaCy model has no valid vocab attribute in extract_skills")
        raise ValueError("spaCy model has no valid vocab attribute in extract_skills")

    skill_counts = Counter()
    text = preprocess_text(text)
    skills_sections = extract_sections_with_keywords(text, ['skills', 'technical skills', 'competencies'], nlp)
    processed_text = text + " " + " ".join(skills_sections) * 2
    doc = nlp(processed_text)

    # Prioritize SKILLS entity from trained model
    for ent in doc.ents:
        if ent.label_ == "SKILLS":
            skill_counts[ent.text] += 1
            logging.info(f"Found skill via SKILLS entity: {ent.text}")

    # Fallback to PhraseMatcher if SKILLS entity doesn't capture all skills
    if not skill_counts:
        matcher = PhraseMatcher(nlp.vocab)
        for column in keyword_df.columns:
            patterns = [
                nlp(skill.strip().lower())
                for skill in keyword_df[column].dropna()
                if skill.strip()
            ]
            if patterns:
                matcher.add(column, patterns)
        for sent in doc.sents:
            if is_skills_sentence(sent):
                for token in sent:
                    if token.text == ',':
                        continue
                    for column in keyword_df.columns:
                        if token.text.lower() in [s.lower() for s in keyword_df[column].dropna()]:
                            skill_counts[column] += 1
                            logging.info(f"Found skill via keyword match: {token.text}")
        matches = matcher(doc)
        seen_spans = set()
        for match_id, start, end in matches:
            span = doc[start:end]
            if span.text.lower() in seen_spans:
                continue
            seen_spans.add(span.text.lower())
            if is_valid_skill_context(span):
                skill_category = nlp.vocab.strings[match_id]
                skill_counts[skill_category] += 1
                logging.info(f"Found skill via PhraseMatcher: {span.text}")

    return skill_counts

def get_job_description_skills(job_description, nlp):
    doc = nlp(job_description.lower())
    skills = set()
    for ent in doc.ents:
        if ent.label_ in ["SKILL", "PRODUCT", "ORG", "SKILLS"]:
            skills.add(ent.text.lower())
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            if any(keyword in token.text.lower() for keyword in ["python", "java", "machine learning", "sql"]):
                skills.add(token.text.lower())
    return {skill: 10 for skill in skills}

def calculate_scores(data, base_weights, job_desc_weights=None):
    scores = {
        'experience': data['experience'] * base_weights['experience'],
        'certifications': data['certifications'] * base_weights['certification'],
        'projects': data['projects'] * base_weights['project'],
        'role_relevance': data['role_relevance'] * base_weights['role_relevance']
    }
    skill_score = 0
    for skill, count in data['skills'].items():
        base_weight = base_weights['skills'].get(skill, 1)
        jd_boost = 1
        if job_desc_weights:
            for jd_skill, jd_weight in job_desc_weights.items():
                if jd_skill.lower() in skill.lower():
                    jd_boost = jd_weight / 5
                    break
        skill_score += min(3, count) * base_weight * jd_boost
    scores['skills'] = skill_score
    scores['total'] = sum(scores.values())
    return scores

def get_matched_jd_skills(candidate_skills, job_desc_weights):
    if not job_desc_weights:
        return {}
    matched = {}
    for skill in candidate_skills:
        for jd_skill, weight in job_desc_weights.items():
            if jd_skill.lower() in skill.lower():
                matched[skill] = weight
                break
    return matched

def process_single_resume(args, nlp):
    resume_data, job_description, job_desc_weights, keyword_df = args
    try:
        filename = resume_data['filename']
        logging.info(f"Starting processing for {filename}")
        log_resource_usage(f"before processing {filename}")
        content_type = resume_data['content_type']
        file_data = resume_data['file_data']
        # Fetch uploaded_at from MongoDB document
        uploaded_at = resume_data.get('uploaded_at')
        # Handle different formats of uploaded_at
        if isinstance(uploaded_at, dict) and '$date' in uploaded_at:
            uploaded_at = uploaded_at['$date']  # Extract the date string
        elif isinstance(uploaded_at, datetime):
            uploaded_at = uploaded_at.isoformat() + "+00:00"  # Convert datetime to ISO string
        if uploaded_at is None:
            logging.warning(f"Resume {filename} in Resumes collection has no uploaded_at field")
            uploaded_at = datetime.utcnow().isoformat() + "+00:00"  # Fallback to current time

        # Handle different formats of file_data
        if isinstance(file_data, dict) and '$binary' in file_data:
            logging.info(f"file_data for {filename} is in $binary format")
            base64_str = file_data['$binary']['base64']
            binary_data = base64.b64decode(base64_str)
            logging.info(f"Successfully decoded base64 data for {filename}, length: {len(binary_data)}")
        elif isinstance(file_data, bytes):
            logging.info(f"file_data for {filename} is raw bytes, length: {len(file_data)}")
            binary_data = file_data  # Use the raw bytes directly
        else:
            logging.error(f"Unexpected file_data format for {filename}: {type(file_data)}")
            return None

        # Validate binary data
        if not binary_data:
            logging.error(f"Binary data for {filename} is empty")
            return None

        logging.info(f"Extracting text for {filename} with content_type: {content_type}")
        text = extract_text(binary_data, content_type)
        if not text.strip():
            logging.warning(f"Skipping empty resume: {filename}")
            return None

        logging.info(f"Extracted text sample: {text[:100]}...")  # Log a sample of the extracted text
        logging.info(f"Extracting candidate name for {filename}")
        candidate_name = extract_candidate_name(filename, nlp)
        role_analyzer = RoleAnalyzer()
        logging.info(f"Detecting experience for {filename}")
        experience = detect_experience(text, nlp)
        logging.info(f"Counting certifications for {filename}")
        certifications = count_certifications(text, nlp)
        logging.info(f"Counting projects for {filename}")
        projects = count_projects(text, nlp)
        logging.info(f"Extracting skills for {filename}")
        skills = extract_skills(text, keyword_df, nlp)
        logging.info(f"Analyzing roles for {filename}")
        role_analysis = role_analyzer.calculate_role_score(role_analyzer.extract_roles(text, nlp))
        matched_jd_skills = get_matched_jd_skills(skills.keys(), job_desc_weights)
        scores = calculate_scores(
            {
                'experience': experience,
                'certifications': certifications,
                'projects': projects,
                'skills': skills,
                'role_relevance': role_analysis['normalized_score']
            },
            SCORE_WEIGHTS,
            job_desc_weights
        )
        logging.info(f"Processed resume {filename}: Candidate={candidate_name}, Score={scores['total']}")
        log_resource_usage(f"after processing {filename}")
        return {
            'Candidate': candidate_name,
            'Experience': round(experience, 1),
            'Certifications': certifications,
            'Projects': projects,
            'Skills': dict(skills),
            'JD_Matched_Skills': matched_jd_skills,
            'Roles': role_analysis['role_details'],
            'Role Score': scores['role_relevance'],
            'Experience Score': scores['experience'],
            'Certification Score': scores['certifications'],
            'Project Score': scores['projects'],
            'Skill Score': scores['skills'],
            'Total Score': scores['total'],
            'File': filename,
            'uploaded_at': uploaded_at  # Use uploaded_at instead of submission_date
        }
    except Exception as e:
        logging.error(f"Error processing {filename}: {str(e)}", exc_info=True)
        return None

class HybridResumeAnalyzer:
    def __init__(self, sbert_model, nlp):
        self.sbert_model = sbert_model
        self.nlp = nlp
        self.role_analyzer = RoleAnalyzer()
        try:
            self.keyword_df = pd.read_csv(KEYWORD_FILE)
            logging.info(f"Loaded keyword file: {KEYWORD_FILE}")
        except Exception as e:
            logging.error(f"Error loading keyword file: {e}")
            self.keyword_df = pd.DataFrame()
        log_resource_usage("after analyzer initialization")

    def _split_resume_into_chunks(self, text, max_chunk_length=300):
        sections = re.split(r'\n\s*\n|\b(?:experience|education|projects|skills):?\b', text, flags=re.IGNORECASE)
        chunks = []
        for section in sections:
            if not section.strip():
                continue
            if len(section) > max_chunk_length:
                sentences = [sent.text for sent in self.nlp(section).sents]
                current_chunk = ""
                for sent in sentences:
                    if len(current_chunk) + len(sent) < max_chunk_length:
                        current_chunk += " " + sent
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = sent
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
            else:
                chunks.append(section.strip())
        return chunks

    def enhanced_process_resumes(self, job_id, job_description=None, job_desc_weights=None, nlp=None):
        start_time = time.time()
        log_resource_usage("before processing resumes")

        cache = {}
        if not BYPASS_CACHE and os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)

        cache_key = f"{job_id}:{job_description}"
        if not BYPASS_CACHE and cache_key in cache:
            logging.info(f"Using cached results for {cache_key}")
            return pd.DataFrame(cache[cache_key])

        logging.info(f"Processing job_id: {job_id}")
        query = {"job_id": job_id}
        resumes = list(collection.find(query))
        logging.info(f"Found {len(resumes)} resumes in MongoDB for job_id {job_id}: {[resume['filename'] for resume in resumes]}")
        if not resumes:
            logging.warning(f"No resumes found in MongoDB for job_id: {job_id}")
            result = pd.DataFrame()
            cache[cache_key] = result.to_dict('records')
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(cache, f)
            return result

        jd_embedding = None
        if job_description:
            jd_embedding = self.sbert_model.encode(job_description, convert_to_tensor=True)

        pool_args = [(resume, job_description, job_desc_weights, self.keyword_df) for resume in resumes]
        rule_based_results = []
        for args in pool_args:
            result = process_single_resume(args, self.nlp)
            if result is not None:
                rule_based_results.append(result)

        logging.info(f"After rule-based processing, {len(rule_based_results)} resumes remain")
        log_resource_usage("after rule-based processing")

        results = []
        for result in rule_based_results:
            filename = result['File']
            resume = next((r for r in resumes if r['filename'] == filename), None)
            if not resume:
                continue
            file_data = resume['file_data']
            if isinstance(file_data, dict) and '$binary' in file_data:
                base64_str = file_data['$binary']['base64']
                binary_data = base64.b64decode(base64_str)
            elif isinstance(file_data, bytes):
                binary_data = file_data
            else:
                logging.error(f"Unexpected file_data format for {filename}: {type(file_data)}")
                continue
            text = extract_text(binary_data, resume['content_type'])
            if not text.strip():
                logging.warning(f"Skipping empty resume: {filename}")
                continue

            sbert_score = 0
            similarity_details = {}
            if jd_embedding is not None:
                resume_chunks = self._split_resume_into_chunks(text)
                chunk_embeddings = self.sbert_model.encode(resume_chunks, convert_to_tensor=True, batch_size=16)
                similarities = util.pytorch_cos_sim(jd_embedding, chunk_embeddings)[0]
                sbert_score = torch.max(similarities).item() * 100
                top_indices = torch.topk(similarities, min(3, len(similarities))).indices
                similarity_details = {
                    'top_matches': [(resume_chunks[i], similarities[i].item())
                                    for i in top_indices],
                    'max_similarity': sbert_score,
                    'mean_similarity': torch.mean(similarities).item() * 100
                }

            combined_score = result['Total Score'] * 0.7 + sbert_score * 0.3
            results.append({
                'job_id': job_id,
                'Candidate': result['Candidate'],
                'File': filename,
                'Rule_Based_Score': result['Total Score'],
                'SBERT_Score': sbert_score,
                'Combined_Score': combined_score,
                'Experience': result['Experience'],
                'Certifications': result['Certifications'],
                'Projects': result['Projects'],
                'Skills': result['Skills'],
                'JD_Matched_Skills': result['JD_Matched_Skills'],
                'Role_Relevance': result['Role Score'],
                'Similarity_Details': similarity_details,
                'uploaded_at': result['uploaded_at'],  # Use uploaded_at instead of submission_date
                **result
            })

        logging.info(f"Final processed resumes: {len(results)}")
        df = pd.DataFrame(results)

        # Store results in TempRankings collection
        if not df.empty:
            # Clear existing rankings for this job_id to avoid duplicates
            temp_ranking_collection.delete_many({"job_id": job_id})
            # Insert new rankings
            temp_ranking_collection.insert_many(df.to_dict('records'))
            logging.info(f"Stored {len(results)} rankings in TempRankings for job_id: {job_id}")

            # Update cache
            cache[cache_key] = df.to_dict('records')
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(cache, f)
        else:
            logging.warning(f"Not caching result for {cache_key} as no resumes were processed successfully")

        cleanup_temp_files()
        log_resource_usage("after processing all resumes")
        logging.info(f"Total processing time: {time.time() - start_time:.2f} seconds")

        return df

def process_resumes_for_web(job_id, job_description=None, spacy_model=None, sbert_model=None):
    if spacy_model is None:
        logging.error("spaCy model is not loaded. Cannot process resumes.")
        raise ValueError("spaCy model is not loaded. Cannot process resumes.")
    if sbert_model is None:
        logging.error("Sentence-BERT model is not loaded. Cannot process resumes.")
        raise ValueError("Sentence-BERT model is not loaded. Cannot process resumes.")

    logging.info(f"Job description: {job_description}")
    analyzer = HybridResumeAnalyzer(sbert_model, spacy_model)
    job_desc_weights = get_job_description_skills(job_description, spacy_model) if job_description else None
    logging.info(f"Job description weights: {job_desc_weights}")
    df = analyzer.enhanced_process_resumes(job_id, job_description, job_desc_weights)
    if df.empty:
        logging.info(f"No valid resumes processed for job_id: {job_id}, returning error")
        return {"error": "No valid resumes processed"}

    df = df.sort_values('Combined_Score', ascending=False)
    ranked_results = df.to_dict('records')

    top_candidates = df.head(5)
    skills_data = pd.DataFrame(top_candidates['Skills'].tolist(), index=top_candidates['Candidate'])
    skills_data = skills_data.fillna(0).astype(int)

    return {
        "ranked_resumes": ranked_results,
        "skills_data": skills_data.to_dict(),
        "total_resumes": len(ranked_results)
    }

# Function to normalize dates to YYYY-MM-DD format
def normalize_date(date_str):
    if not date_str:
        return None
    try:
        # If the date string is in ISO format with time (e.g., "2025-05-01T20:39:44.091+00:00" or "2025-05-01T20:39:44.091Z")
        if "T" in date_str:
            # Handle both Z and +00:00 suffixes
            if date_str.endswith("Z"):
                date_str = date_str.replace("Z", "+00:00")
            # Parse the ISO format and extract the date part
            return datetime.fromisoformat(date_str).date().isoformat()
        # If the date string is already in YYYY-MM-DD format (e.g., "2025-04-01"), return as is
        return date_str
    except ValueError as e:
        logging.error(f"Failed to normalize date {date_str}: {str(e)}")
        return None

def filter_resumes_from_temp_collection(job_id, filter_criteria):
    try:
        # Fetch all rankings for the given job_id from TempRankings
        query = {"job_id": job_id}
        ranked_resumes = list(temp_ranking_collection.find(query))
        if not ranked_resumes:
            logging.warning(f"No rankings found in TempRankings for job_id: {job_id}")
            return {"error": "No rankings found for this job_id"}

        # Convert ObjectId to string to make the data JSON serializable
        ranked_resumes = convert_objectid_to_str(ranked_resumes)

        # Sort by Combined_Score in descending order
        ranked_resumes = sorted(ranked_resumes, key=lambda x: x['Combined_Score'], reverse=True)

        # Apply filters
        filtered_resumes = ranked_resumes

        # Filter by rank range
        if "minRank" in filter_criteria or "maxRank" in filter_criteria:
            min_rank = filter_criteria.get("minRank", 1)
            max_rank = filter_criteria.get("maxRank", len(ranked_resumes))
            filtered_resumes = [
                resume for idx, resume in enumerate(ranked_resumes, 1)
                if min_rank <= idx <= max_rank
            ]

        # Filter by date range using uploaded_at
        if "startDate" in filter_criteria or "endDate" in filter_criteria:
            start_date = normalize_date(filter_criteria.get("startDate"))
            end_date = normalize_date(filter_criteria.get("endDate"))
            logging.info(f"Applying date filter: startDate={start_date}, endDate={end_date}")

            filtered_resumes_temp = []
            for resume in filtered_resumes:
                uploaded_at = resume.get("uploaded_at")
                normalized_uploaded_at = normalize_date(uploaded_at)
                logging.info(
                    f"Resume {resume['Candidate']}: uploaded_at={uploaded_at}, normalized={normalized_uploaded_at}")

                if normalized_uploaded_at is None:
                    logging.warning(
                        f"Excluding resume {resume['Candidate']} due to unparseable uploaded_at: {uploaded_at}")
                    continue  # Exclude resumes with unparseable dates
                if (
                        (not start_date or normalized_uploaded_at >= start_date) and
                        (not end_date or normalized_uploaded_at <= end_date)
                ):
                    filtered_resumes_temp.append(resume)
                else:
                    logging.info(
                        f"Excluding resume {resume['Candidate']} as {normalized_uploaded_at} is outside range {start_date} to {end_date}")

            filtered_resumes = filtered_resumes_temp

        # Prepare skills_data for the top 5 candidates
        df = pd.DataFrame(filtered_resumes)
        if df.empty:
            return {
                "ranked_resumes": [],
                "skills_data": {}
            }

        top_candidates = df.head(5)
        skills_data = pd.DataFrame(top_candidates['Skills'].tolist(), index=top_candidates['Candidate'])
        skills_data = skills_data.fillna(0).astype(int)

        return {
            "ranked_resumes": filtered_resumes,
            "skills_data": skills_data.to_dict()
        }

    except Exception as e:
        logging.error(f"Error filtering resumes from TempRankings for job_id {job_id}: {str(e)}", exc_info=True)
        return {"error": "Failed to apply filters"}

# Helper function to convert ObjectId to string in a dictionary
def convert_objectid_to_str(data):
    if isinstance(data, list):
        return [convert_objectid_to_str(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_objectid_to_str(value) if key != '_id' else str(value)
                for key, value in data.items()}
    return data