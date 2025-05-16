document.addEventListener('DOMContentLoaded', () => {
    initializeUploadHandlers();
    loadJobDescriptions(); // Load JD Dynamically
});

// Function to initialize upload button and input handlers
function initializeUploadHandlers() {
    const uploadButtons = document.querySelectorAll('.upload-btn');
    const uploadInputs = document.querySelectorAll('.upload-input');

    uploadButtons.forEach((button, index) => {
        button.addEventListener('click', () => {
            uploadInputs[index].click(); // Trigger file input
        });
    });

    uploadInputs.forEach(input => {
        input.addEventListener('change', handleFileUpload);
    });
}

// Function to handle file upload
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Validate file type and size (max 5MB)
    if (!isValidFileType(file) || !isValidFileSize(file)) {
        return;
    }

    // Get job ID from the parent card
    const card = event.target.closest('.jd-card');
    const jobId = card.dataset.jobId;

    // Show loading animation
    const animation = card.querySelector('.loading-animation');
    animation.style.display = 'block';

    try {
        const response = await uploadResume(file, jobId);
        const result = await response.json();
        if (response.ok) {
            showSuccessMessage(`Resume uploaded successfully for ${jobId}! Resume ID: ${result.resume_id}`);
            event.target.value = ''; // Clear input
        } else {
            showErrorMessage(`Upload failed: ${result.error}`);
        }
    } catch (err) {
        showErrorMessage('Error uploading resume. Please try again.');
        console.error(err);
    } finally {
        // Hide loading animation
        animation.style.display= 'none';
    }
}

// Function to validate file type
function isValidFileType(file) {
    const allowedTypes = [
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ];
    if (!allowedTypes.includes(file.type)) {
        alert('Please upload a PDF, DOC, or DOCX file.');
        return false;
    }
    return true;
}

// Function to validate file size
function isValidFileSize(file) {
    if (file.size > 5 * 1024 * 1024) {
        alert('File size exceeds 5MB.');
        return false;
    }
    return true;
}

// Function to upload resume to the server
async function uploadResume(file, jobId) {
    const formData = new FormData();
    formData.append('resume', file);
    formData.append('jobId', jobId);

    return await fetch('/upload_resume', {
        method: 'POST',
        body: formData
    });
}

// Function to display success message
function showSuccessMessage(message) {
    alert(message);
}

// Function to display error message
function showErrorMessage(message) {
    alert(message);
}



// Load job descriptions dynamically
async function loadJobDescriptions() {
    const spinner = document.getElementById('loading-spinner');
    const container = document.getElementById('job-cards-container');
    spinner.style.display = 'block'; // Show spinner
    container.style.display = 'none'; // Hide container while loading


    try {
        const response = await fetch('/get_job_descriptions', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        });
        const jobDescriptions = await response.json();
        const container = document.getElementById('job-cards-container');
        container.innerHTML = ''; // Clear existing content

        if (jobDescriptions.job_descriptions && jobDescriptions.job_descriptions.length > 0) {
            jobDescriptions.job_descriptions.forEach(jd => {
                const col = document.createElement('div');
                col.className = 'col-md-4';
                col.innerHTML = `
                    <div class="card jd-card h-100" data-job-id="${jd.job_id}">
                        <h5 class="card-header">Job Description</h5>
                        <div class="card-body d-flex flex-column">
                            <h5 class="card-title">${jd.role}</h5>
                            <p class="card-text">${jd.description}</p>
                            <button type="button" class="btn btn-primary mt-auto upload-btn">Upload Resume</button>
                            <input type="file" class="upload-input" accept=".pdf,.doc,.docx" hidden>
                            <div class="loading-animation mt-2" style="display: none; position: relative; height: 40px;">
                                <div class="a"></div>
                                <div class="b"></div>
                            </div>
                        </div>
                    </div>
                `;
                container.appendChild(col);
            });
            initializeUploadHandlers(); // Reinitialize handlers for new cards
        } else {
            container.innerHTML = '<p class="text-center">No job descriptions available.</p>';
        }
    } catch (err) {
        console.error('Error loading job descriptions:', err);
        document.getElementById('job-cards-container').innerHTML = '<p class="text-center">Error loading job descriptions. Please try again.</p>';
    } finally{
        spinner.style.display = 'none'; // Hide spinner
        container.style.display = 'flex'; // Show container
    }
}

// Detect browser back/forward navigation and page visibility
window.addEventListener('popstate', () => {
    console.log('popstate event triggered'); // Debug log
    loadJobDescriptions();
});

document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
        console.log('Page became visible'); // Debug log
        loadJobDescriptions();
    }
});