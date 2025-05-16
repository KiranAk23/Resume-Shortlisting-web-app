from flask import Flask, render_template, request, jsonify, send_from_directory
from SCRIPTS.resume_shortlister import process_resumes_for_web, filter_resumes_from_temp_collection
from DATABASE import db_access
from werkzeug.exceptions import HTTPException
from sentence_transformers import SentenceTransformer
import spacy
import os
import datetime
import time
import logging
import logging.handlers
from DATABASE.db import get_database


app = Flask(__name__, template_folder='WEB_PAGE/templates', static_folder='WEB_PAGE/static')

# Optional: Enable CORS if frontend and backend are on different ports
try:
    from flask_cors import CORS
    CORS(app)
except ImportError:
    pass  # Install flask-cors if needed: pip install flask-cors

# Set maximum upload size (5MB)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB for bulk upload

# Set up logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler('app.log', maxBytes=10*1024*1024, backupCount=5)
    ]
)

# Define a function to load the spaCy model
def load_spacy_model():
    model_path = r"D:\Phase1\Final_project\Resume_ranking\MODELS\spacy_model\model2\model-last"
    try:
        logging.info(f"Loading spaCy model from {model_path}")
        model = spacy.load(model_path)  # Load the custom model, do NOT exclude transformer
        logging.info("spaCy model loaded with components: %s", model.pipe_names)
        if not hasattr(model, 'vocab') or model.vocab is None:
            logging.error("Loaded spaCy model has no valid vocab attribute")
            raise ValueError("Loaded spaCy model has no valid vocab attribute")

        # Add sentencizer if the pipeline lacks sentence segmentation
        if not model.has_pipe("sentencizer") and not model.has_pipe("parser") and not model.has_pipe("senter"):
            model.add_pipe("sentencizer")
            logging.info("Added sentencizer component to spaCy pipeline")
        logging.info(f"spaCy pipeline components after modification: {list(model.pipe_names)}")
        return model
    except Exception as e:
        logging.error(f"Failed to load custom spaCy model: {str(e)}")
        logging.info("Falling back to default spaCy model 'en_core_web_sm'...")
        try:
            model = spacy.load("en_core_web_sm")
            logging.info("Default spaCy model 'en_core_web_sm' loaded with components: %s", model.pipe_names)
            if not hasattr(model, 'vocab') or model.vocab is None:
                logging.error("Default spaCy model has no valid vocab attribute")
                raise ValueError("Default spaCy model has no valid vocab attribute")

            # Add sentencizer if the pipeline lacks sentence segmentation
            if not model.has_pipe("sentencizer") and not model.has_pipe("parser") and not model.has_pipe("senter"):
                model.add_pipe("sentencizer")
                logging.info("Added sentencizer component to default spaCy pipeline")
            logging.info(f"spaCy pipeline components after modification (default model): {list(model.pipe_names)}")
            return model
        except Exception as fallback_e:
            logging.error(f"Failed to load default spaCy model: {str(fallback_e)}")
            raise RuntimeError(f"Failed to load any spaCy model: {str(fallback_e)}")

# Load models globally at startup
spacy_model = None
try:
    logging.info("Attempting to load spaCy model at startup...")
    spacy_model = load_spacy_model()
except Exception as e:
    logging.error(f"Failed to load spaCy model at startup: {str(e)}")
    raise RuntimeError(f"Failed to load spaCy model at startup: {str(e)}")

try:
    logging.info("Loading Sentence-BERT model...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')  # Use CPU to avoid GPU driver issues
    logging.info("Sentence-BERT model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load Sentence-BERT model: {str(e)}")
    raise RuntimeError(f"Failed to load Sentence-BERT model: {str(e)}")

# Verify spacy_model is not None before proceeding
if spacy_model is None:
    logging.error("spaCy model is None after loading attempts at startup. Cannot start application.")
    raise RuntimeError("spaCy model is None after loading attempts at startup. Cannot start application.")

# Serve the index.html template
@app.route('/')
def serve_index():
    # Fetch job descriptions from the database
    job_descriptions = db_access.get_job_descriptions()
    return render_template('index.html', job_descriptions=job_descriptions)

@app.route('/candidate_page')
def serve_candidate():
    return render_template('candidate_page.html')

# Upload resume endpoint
@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    try:
        if 'resume' not in request.files or 'jobId' not in request.form:
            return jsonify({"error": "Resume file and jobId are required"}), 400

        file = request.files['resume']
        job_id = request.form['jobId']
        filename = file.filename
        content_type = file.content_type

        # Save resume to Resume_ranking.resumes with job_id
        resume_id = db_access.save_resume(file, filename, content_type, job_id)

        if resume_id:
            return jsonify({"message": "Resume uploaded successfully", "resume_id": resume_id}), 201
        else:
            return jsonify({"error": "Failed to upload resume"}), 500

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

@app.route('/bulk_upload', methods=['POST'])
def bulk_upload():
    try:
        if 'resumes' not in request.files or 'jobId' not in request.form:
            return jsonify({"error": "Resumes and jobId are required"}), 400

        files = request.files.getlist('resumes')
        job_id = request.form['jobId'].lower().replace(' ', '_')
        if not files or len(files) == 0:
            return jsonify({"error": "No files selected"}), 400

        allowed_types = [
            'application/pdf',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        ]
        max_size = 5 * 1024 * 1024  # 5MB per file
        invalid_files = []

        # Validate file types and sizes
        for file in files:
            if not file.filename:
                continue
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)  # Reset file pointer
            if file.content_type not in allowed_types or file_size > max_size:
                invalid_files.append(file.filename)

        if invalid_files:
            return jsonify({
                "error": "Invalid files detected. All uploads aborted.",
                "invalid_files": invalid_files,
                "message": "Please ensure all files are PDF, DOC, or DOCX and under 5MB."
            }), 400

        # Create a temporary collection name
        temp_collection = f"temp_{job_id}_{int(time.time())}"
        results = {"success": [], "failed": [], "total_processed": 0}

        try:
            # Process all files and store in temporary collection
            batch_results = db_access.process_batch(files, job_id, temp_collection)
            results["success"].extend(batch_results["success"])
            results["failed"].extend(batch_results["failed"])
            results["total_processed"] = len(files)

            if batch_results["failed"]:
                raise Exception("One or more files failed to process. Rolling back all uploads.")

            # Commit to Resumes collection if all succeeded
            if results["success"]:
                db_access.commit_batch(results["success"], job_id, temp_collection)

            return jsonify({
                "message": "Bulk upload completed successfully!",
                "successful_uploads": len(results["success"]),
                "failed_uploads": len(results["failed"]),
                "total_processed": results["total_processed"],
                "details": {"success": results["success"]}
            }), 200

        except Exception as e:
            # Rollback on any failure
            db_access.rollback_batch(temp_collection)
            error_message = str(e)
            if "failed to process" in error_message.lower():
                error_details = "Failed to process one or more files."
            else:
                error_details = "An error occurred while saving the files to the database."
            return jsonify({
                "error": "Bulk upload failed. No files were saved.",
                "details": error_details,
                "failed_uploads": len(results["failed"]),
                "total_processed": results["total_processed"],
                "failed_files": results["failed"]
            }), 500

    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

# Download resume endpoint
@app.route('/download_resume/<resume_id>', methods=['GET'])
def download_resume(resume_id):
    job_id = request.args.get('jobId')  # Optional jobId query parameter
    file_response = db_access.get_resume(resume_id, job_id)
    if file_response:
        return file_response
    else:
        return jsonify({"error": "Resume not found"}), 404

# List resumes for a specific job_id
@app.route('/list_resumes/<job_id>', methods=['GET'])
def list_resumes(job_id):
    try:
        resumes = db_access.list_resumes_by_job(job_id)
        if resumes:
            return jsonify({"resumes": resumes}), 200
        else:
            return jsonify({"error": "No resumes found for this job"}), 404
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

@app.route('/admin_page')
def serve_admin():
    return render_template('admin.html')

@app.route('/bulk_upload_page')
def serve_bulk_upload():
    job_id = request.args.get('jobId', '')
    return render_template('bulk_upload.html', job_id=job_id)

@app.route('/add_job_description', methods=['POST'])
def add_job_description():
    try:
        data = request.get_json()
        role = data.get('role')
        description = data.get('description')

        if not role or not description:
            return jsonify({"error": "Job Role and Description are required"}), 400

        job_id = role.lower().replace(' ', '_')
        job_description = {
            "job_id": job_id,
            "role": role,
            "description": description,
            "created_at": datetime.datetime.utcnow()
        }

        logging.info(f"Attempting to save job description: {job_description}")
        result = db_access.save_job_description(job_description)
        if result:
            logging.info(f"Job description saved with ID: {result}")
            return jsonify({"message": "Job Description added successfully", "job_id": job_id}), 201
        else:
            logging.error("Failed to save job description")
            return jsonify({"error": "Failed to add Job Description"}), 500

    except Exception as e:
        logging.error(f"Exception in add_job_description: {str(e)}")
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

@app.route('/edit_job_description', methods=['PUT'])
def edit_job_description():
    try:
        data = request.get_json()
        job_id = data.get('job_id')
        role = data.get('role')
        description = data.get('description')

        if not job_id or not role or not description:
            return jsonify({"error": "Job ID, Role, and Description are required"}), 400

        updated_job = {
            "job_id": job_id,
            "role": role,
            "description": description,
            "created_at": datetime.datetime.utcnow()  # Update timestamp
        }

        result = db_access.update_job_description(job_id, updated_job)
        if result:
            return jsonify({"message": "Job Description updated successfully", "job_id": job_id}), 200
        else:
            return jsonify({"error": "Failed to update Job Description"}), 500

    except Exception as e:
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

@app.route('/delete_job_description', methods=['DELETE'])
def delete_job_description():
    try:
        data = request.get_json()
        job_id = data.get('job_id')

        if not job_id:
            return jsonify({"error": "Job ID is required"}), 400

        result = db_access.delete_job_description(job_id)
        if result:
            return jsonify({"message": "Job Description deleted successfully"}), 200
        else:
            return jsonify({"error": "Failed to delete Job Description"}), 500

    except Exception as e:
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500



@app.route('/get_job_descriptions', methods=['GET'])
def get_job_descriptions_api():
    try:
        job_descriptions = db_access.get_job_descriptions()
        return jsonify({"job_descriptions": job_descriptions}), 200
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500



@app.route('/ranking_page/<job_id>')
def ranking_page(job_id):
    return render_template('ranking_page.html', job_id=job_id)



# Function to fetch job description by job_id
def get_job_description(job_id):
    logging.info(f"Fetching job description for job_id: {job_id}")
    job_descriptions = db_access.get_job_descriptions()
    for jd in job_descriptions:
        if jd['job_id'] == job_id:
            logging.info(f"Found job description for {job_id}: {jd['description']}")
            return jd['description']
    logging.warning(f"No job description found for job_id: {job_id}")
    return None



@app.route('/get_ranking/<job_id>', methods=['GET'])
def get_ranking(job_id):
    global spacy_model  # Access the global spacy_model variable
    try:
        logging.info(f"Processing resumes for job_id: {job_id}")
        # Check if spacy_model is None and reload if necessary
        if spacy_model is None:
            logging.warning("spaCy model is None. Attempting to reload...")
            spacy_model = load_spacy_model()
        logging.info(f"spaCy model state: {spacy_model}, components: {spacy_model.pipe_names if spacy_model else 'None'}")
        job_description = get_job_description(job_id)

        # Pass the preloaded models to the processing function
        result = process_resumes_for_web(job_id, job_description, spacy_model=spacy_model, sbert_model=sbert_model)
        logging.info(f"Processing result for job_id {job_id}: {result}")
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error processing resumes for job_id {job_id}: {str(e)}")
        return jsonify({"error": f"Error processing resumes: {str(e)}"}), 500





@app.route('/filter_ranking/<job_id>', methods=['POST'])
def filter_ranking(job_id):
    try:
        filter_criteria = request.get_json()
        logging.info(f"Received filter criteria for job_id {job_id}: {filter_criteria}")
        result = filter_resumes_from_temp_collection(job_id, filter_criteria)
        if "error" in result:
            return jsonify({"error": result["error"]}), 400
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error in filter_ranking endpoint for job_id {job_id}: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to apply filters"}), 500


# MongoDB Configuration
db, client = get_database()
DB_NAME = "Resume_ranking" #DATABASE NAME
TEMP_COLLECTION_NAME = "temp_collection"  # Add NEW COLLECTION CALLED temp_collection to MongoDB configuration
temp_collection = db[TEMP_COLLECTION_NAME] 

@app.route('/get_filter_skills/<job_id>', methods=['GET'])
def get_filter_skills(job_id):
    try:
        skills_doc = temp_collection.find_one({"job_id": job_id, "type": "filter_skills"})
        if not skills_doc:
            logging.warning(f"No filter skills found in temp_collection for job_id: {job_id}")
            return jsonify({"error": "No filter skills found"}), 404

        skills = skills_doc.get("skills", [])
        return jsonify({"skills": skills})
    except Exception as e:
        logging.error(f"Error fetching filter skills for job_id {job_id}: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to fetch filter skills"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)