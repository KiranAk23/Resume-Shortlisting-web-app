from flask import Flask, request, jsonify
from DATABASE import db_access
from werkzeug.exceptions import HTTPException

app = Flask(__name__)



@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    try:
        if 'resume' not in request.files or 'jobId' not in request.form:
            return jsonify({"error": "Resume file and jobId are required"}), 400

        file = request.files['resume']
        job_id = request.form['jobId']
        filename = file.filename
        content_type = file.content_type
        
        # save to resume collection with job_id
        resume_id = db_access.save_resume(file, filename, content_type, job_id)

        if resume_id:
            return jsonify({"message": "Resume uploaded successfully", "resume_id": resume_id}), 201
        else:
            return jsonify({"error": "Failed to upload resume"}), 500

    except HTTPException as http_err:
        # Rethrow HTTP errors for Flask to handle properly
        raise http_err
    except Exception as e:

        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500


@app.route('/download_resume/<resume_id>', methods=['GET'])
def download_resume(resume_id):
    job_id = request.args.get('jobId')
    file_response = db_access.get_resume(resume_id, job_id)
    if file_response:
        return file_response
    else:
        return jsonify({"error": "Resume not found"}), 404
    


@app.route('/list_resumes/<job_id>', methods=['GET'])
def list_resumes(job_id):
    try:
        resumes = db_access.list_resumes_by_job(job_id)
        if resumes is not None:
            return jsonify({"Resumes": resumes}), 200
        else:
            return jsonify({"error": "No resumes found for this job"}), 404
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
