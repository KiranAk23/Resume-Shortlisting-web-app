from  DATABASE.db import get_database
from bson import Binary, ObjectId
import datetime
import io
from flask import send_file
from pymongo import InsertOne

def save_resume(file_object, filename, content_type, job_id):
    """
    Save resume to the resumes collection in Resume_ranking database.

    :param file_object: File object from request
    :param filename: Name of the file
    :param content_type: MIME type of the file
    :param job_id: Identifier for the job (e.g., product_manager)
    :return: Resume ID as string or None if failed
    """
    try:
        db,client = get_database() # get db and client
        resumes_collection = db['Resumes'] # MongoDB Collection called Resumes to store resumes in binary format

        # Read the file content as binary
        file_data = file_object.read()

        resume_document = {
            "filename": filename,
            "content_type": content_type,
            "file_data": Binary(file_data),
            "job_id": job_id.lower().replace(' ', '_'),  # Normalize job_id
            "uploaded_at": datetime.datetime.utcnow()
        }

        result = resumes_collection.insert_one(resume_document)
        return str(result.inserted_id)

    except Exception as e:
        print("Error saving resume:", e)
        return None


def process_batch(files, job_id, temp_collection):
    """Process a batch of files and save to a temporary collection."""
    try:
        db, client = get_database()
        temp_coll = db[temp_collection] # MongoDB collection called a temp_collection to process ranking , bulk upload,...

        documents = []
        for file in files:
            if not file.filename:
                continue
            file_data = file.read()
            document = {
                "filename": file.filename,
                "content_type": file.content_type,
                "file_data": Binary(file_data),
                "job_id": job_id,
                "uploaded_at": datetime.datetime.utcnow()
            }
            documents.append(document)

        if not documents:
            return {"success": [], "failed": []}

        result = temp_coll.insert_many(documents, ordered=False)
        success = [{"filename": doc["filename"], "resume_id": str(id)} for doc, id in zip(documents, result.inserted_ids)]
        failed = []
        return {"success": success, "failed": failed}

    except Exception as e:
        print(f"Error processing batch: {e}")
        failed = [{"filename": file.filename, "error": str(e)} for file in files if file.filename]
        return {"success": [], "failed": failed}

def commit_batch(successful_docs, job_id, temp_collection):
    """Commit successful uploads from temp collection to Resumes collection."""
    try:
        db, client = get_database()
        resumes_collection = db['Resumes']
        temp_coll = db[temp_collection]

        documents = temp_coll.find({})
        operations = [
            InsertOne({
                "filename": doc["filename"],
                "content_type": doc["content_type"],
                "file_data": doc["file_data"],
                "job_id": job_id,
                "uploaded_at": doc["uploaded_at"]
            }) for doc in documents
        ]
        if operations:
            resumes_collection.bulk_write(operations, ordered=False)
        temp_coll.delete_many({})
    except Exception as e:
        print(f"Error committing batch: {e}")
        raise

def rollback_batch(temp_collection):
    """Rollback by clearing the temporary collection."""
    try:
        db, client = get_database()
        temp_coll = db[temp_collection]
        temp_coll.delete_many({})
        print(f"Rollback completed for temp collection: {temp_collection}")
    except Exception as e:
        print(f"Error during rollback: {e}")

def get_resume(resume_id,job_id=None):
    """
    Retrieve resume from the resumes collection.

    :param resume_id: MongoDB ObjectId of the resume
    :param job_id: Optional job identifier to filter
    :return: Flask send_file response or None if not found
    """
    try:
        db,client = get_database()
        resumes_collection = db['Resumes'] #MongoDB Collections
        
        query = {"_id": ObjectId(resume_id)}
        if job_id:
            query["job_id"] = job_id.lower().replace(' ', '_')

        # Find the document
        resume_doc = resumes_collection.find_one(query)
        if not resume_doc:
            return None

        # Create a file-like object
        return send_file(
            io.BytesIO(resume_doc['file_data']),
            mimetype=resume_doc['content_type'],
            as_attachment=True,
            download_name=resume_doc['filename']
        )

    except Exception as e:
        print("Error fetching resume:", e)
        return None
    

def list_resumes_by_job(job_id):
    """
    List all resumes for a specific job_id.

    :param job_id: Identifier for the job (e.g., product_manager)
    :return: List of resume metadata or None if failed
    """
    try:
        db, client = get_database()
        resumes_collection = db['Resumes']

        resumes = resumes_collection.find({"job_id": job_id.lower().replace(' ', '_')})
        resume_list = [
            {
                "resume_id": str(resume["_id"]),
                "filename": resume["filename"],
                "job_id": resume["job_id"],
                "uploaded_at": resume["uploaded_at"].isoformat()
            }
            for resume in resumes
        ]
        return resume_list if resume_list else []

    except Exception as e:
        print("Error listing resumes:", e)
        return None


def save_job_description(job_description):
    """Save a job description to the JobDescriptions collection."""
    try:
        db, client = get_database()
        job_descriptions_collection = db['JobDescriptions']
        result = job_descriptions_collection.insert_one(job_description)
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error saving job description: {e}")
        return 
    

def update_job_description(job_id, updated_job):
    """Update a job description in the JobDescriptions collection."""
    try:
        db, client = get_database()
        job_descriptions_collection = db['JobDescriptions']
        result = job_descriptions_collection.update_one(
            {"job_id": job_id},
            {"$set": updated_job}
        )
        return result.modified_count > 0
    except Exception as e:
        print(f"Error updating job description: {e}")
        return None


def delete_job_description(job_id):
    """Delete a job description from the JobDescriptions collection."""
    try:
        db, client = get_database()
        job_descriptions_collection = db['JobDescriptions']
        result = job_descriptions_collection.delete_one({"job_id": job_id})
        if result.deleted_count > 0:
            # Also delete associated resumes
            resumes_collection = db['Resumes']
            resumes_collection.delete_many({"job_id": job_id})
            return True
        return False
    except Exception as e:
        print(f"Error deleting job description: {e}")
        return None

def get_job_descriptions():
    """Retrieve all job descriptions from the JobDescriptions collection."""
    try:
        db, client = get_database()
        job_descriptions_collection = db['JobDescriptions']
        job_descriptions = job_descriptions_collection.find({})
        return [
            {
                "job_id": jd["job_id"],
                "role": jd["role"],
                "description": jd["description"],
                "created_at": jd["created_at"].isoformat()
            }
            for jd in job_descriptions
        ]
    except Exception as e:
        print(f"Error fetching job descriptions: {e}")
        return []
    


