document.addEventListener("DOMContentLoaded", () => {
  initializeAdminPage();
});

let allJobDescriptions = []; // Store all job descriptions globally for search and suggestions

function initializeAdminPage() {
  const addJobBtn = document.querySelector(".add-job-btn");
  const addJobSection = document.querySelector("#add-job-section");
  const saveJobBtn = document.querySelector(".save-job-btn");
  const cancelJobBtn = document.querySelector(".cancel-job-btn");
  const jobRoleInput = document.querySelector("#jobRole");
  const jobDescriptionInput = document.querySelector("#jobDescription");
  const searchInput = document.getElementById("searchInput");
  const searchSuggestions = document.getElementById("searchSuggestions");
  let editingJobId = null; // Track if editing a job

  // Toggle Add Job Description section
  addJobBtn.addEventListener("click", () => {
    addJobSection.style.display =
      addJobSection.style.display === "none" ? "block" : "none";
    editingJobId = null; // Reset editing state
    jobRoleInput.value = "";
    jobDescriptionInput.value = "";
    saveJobBtn.textContent = "Save Job Description"; // Reset button text
    document.querySelector(".card-title").textContent =
      "Add New Job Description"; // Reset form title
  });

  // Cancel button to hide section and clear form
  cancelJobBtn.addEventListener("click", () => {
    addJobSection.style.display = "none";
    jobRoleInput.value = "";
    jobDescriptionInput.value = "";
    editingJobId = null;
    saveJobBtn.textContent = "Save Job Description";
    document.querySelector(".card-title").textContent =
      "Add New Job Description";
  });

  // Save Job Description (Add or Edit)
  saveJobBtn.addEventListener("click", async () => {
    const jobRole = jobRoleInput.value.trim();
    const jobDescription = jobDescriptionInput.value.trim();

    if (!jobRole || !jobDescription) {
      alert("Please fill in both Job Role and Job Description.");
      return;
    }

    try {
      const endpoint = editingJobId
        ? "/edit_job_description"
        : "/add_job_description";
      const method = editingJobId ? "PUT" : "POST";
      const payload = editingJobId
        ? { job_id: editingJobId, role: jobRole, description: jobDescription }
        : { role: jobRole, description: jobDescription };

      const response = await fetch(endpoint, {
        method: method,
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const result = await response.json();
      if (response.ok) {
        alert(
          editingJobId
            ? "Job Description updated successfully!"
            : "Job Description added successfully! It is now visible to candidates on the jobs page."
        );
        addJobSection.style.display = "none";
        jobRoleInput.value = "";
        jobDescriptionInput.value = "";
        editingJobId = null;
        saveJobBtn.textContent = "Save Job Description";
        document.querySelector(".card-title").textContent =
          "Add New Job Description";
        loadJobDescriptions(); // Refresh the job list
      } else {
        alert(
          `Failed to ${editingJobId ? "update" : "add"} Job Description: ${
            result.error || "Unknown error"
          }`
        );
      }
    } catch (err) {
      console.error("Fetch error:", err);
      alert(
        `Error ${
          editingJobId ? "updating" : "adding"
        } Job Description. Please check the console for details.`
      );
    }
  });

  // Real-time search and suggestions as the user types
  searchInput.addEventListener("input", () => {
    const query = searchInput.value.trim();
    showSuggestions(query);
    filterJobDescriptions(query);
  });

  // Hide suggestions when clicking outside
  document.addEventListener("click", (e) => {
    if (
      !searchInput.contains(e.target) &&
      !searchSuggestions.contains(e.target)
    ) {
      searchSuggestions.style.display = "none";
    }
  });

  // Prevent form submission on search button click (since we're handling it dynamically)
  document.querySelector(".search-btn").addEventListener("click", (e) => {
    e.preventDefault();
    const query = searchInput.value.trim();
    filterJobDescriptions(query);
    searchSuggestions.style.display = "none"; // Hide suggestions after search
  });

  // Load job descriptions dynamically
  async function loadJobDescriptions(jobDescriptionsToRender = null) {
    try {
      // If no specific job descriptions are provided, fetch them
      if (!jobDescriptionsToRender) {
        const response = await fetch("/get_job_descriptions", {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Failed to fetch job descriptions");
        }
        allJobDescriptions = data.job_descriptions; // Store globally for search
        jobDescriptionsToRender = allJobDescriptions;
      }

      const container = document.getElementById("job-cards-container");
      container.innerHTML = ""; // Clear existing content

      if (jobDescriptionsToRender && jobDescriptionsToRender.length > 0) {
        jobDescriptionsToRender.forEach((jd) => {
          const card = document.createElement("div");
          card.className = "jd-card card mb-3";
          card.dataset.jobId = jd.job_id;
          card.dataset.role = jd.role.toLowerCase();
          card.dataset.description = jd.description.toLowerCase();
          card.innerHTML = `
                        <div class="card-body d-flex justify-content-between align-items-center">
                            <div>
                                <h5 class="card-title">${jd.role}</h5>
                                <p class="card-text">${jd.description}</p>
                            </div>
                            <div class="d-flex flex-column gap-2">
                                <button class="btn btn-outline-primary edit-btn">Edit</button>
                                <button class="btn btn-outline-danger delete-btn">Delete</button>
                                <button class="btn btn-outline-info bulk-upload-btn">Bulk Resume</button>
                                <button class="btn btn-outline-success ranking-btn">Ranking</button>
                            </div>
                        </div>
                    `;
          container.appendChild(card);
        });

        // Add event listeners to buttons
        document.querySelectorAll(".edit-btn").forEach((btn) => {
          btn.addEventListener("click", (e) => {
            const card = e.target.closest(".jd-card");
            const jobId = card.dataset.jobId;
            const job = allJobDescriptions.find((jd) => jd.job_id === jobId);
            jobRoleInput.value = job.role;
            jobDescriptionInput.value = job.description;
            editingJobId = jobId;
            saveJobBtn.textContent = "Update Job Description";
            document.querySelector(".card-title").textContent =
              "Edit Job Description";
            addJobSection.style.display = "block";
          });
        });

        document.querySelectorAll(".delete-btn").forEach((btn) => {
          btn.addEventListener("click", async (e) => {
            const card = e.target.closest(".jd-card");
            const jobId = card.dataset.jobId;
            if (
              confirm(
                `Are you sure you want to delete the job description for ${jobId}?`
              )
            ) {
              try {
                const response = await fetch("/delete_job_description", {
                  method: "DELETE",
                  headers: {
                    "Content-Type": "application/json",
                  },
                  body: JSON.stringify({ job_id: jobId }),
                });
                const result = await response.json();
                if (response.ok) {
                  alert("Job Description deleted successfully!");
                  loadJobDescriptions();
                } else {
                  alert(`Failed to delete Job Description: ${result.error}`);
                }
              } catch (err) {
                console.error("Delete error:", err);
                alert("Error deleting Job Description. Please try again.");
              }
            }
          });
        });

        document.querySelectorAll(".bulk-upload-btn").forEach((btn) => {
          btn.addEventListener("click", (e) => {
            const card = e.target.closest(".jd-card");
            const jobId = card.dataset.jobId;
            window.location.href = `/bulk_upload_page?jobId=${jobId}`; // Redirect with jobId
          });
        });

        document.querySelectorAll(".ranking-btn").forEach((btn) => {
          btn.addEventListener("click", (e) => {
            const card = e.target.closest(".jd-card");
            const jobId = card.dataset.jobId;
            window.location.href = `/ranking_page/${jobId}`; // Redirect to ranking page
          });
        });
      } else {
        container.innerHTML =
          '<p class="text-center">No job descriptions available.</p>';
      }
    } catch (err) {
      console.error("Error loading job descriptions:", err);
      document.getElementById("job-cards-container").innerHTML =
        '<p class="text-center">Error loading job descriptions. Please try again.</p>';
    }
  }

  // Filter job descriptions based on search query
  function filterJobDescriptions(query) {
    const filtered = allJobDescriptions.filter((jd) => {
      const role = jd.role.toLowerCase();
      const description = jd.description.toLowerCase();
      return (
        role.includes(query.toLowerCase()) ||
        description.includes(query.toLowerCase())
      );
    });
    loadJobDescriptions(filtered);
  }


  // Show search suggestions (only for role)
  function showSuggestions(query) {
    const suggestionsDiv = document.getElementById("searchSuggestions");
    suggestionsDiv.innerHTML = ""; // Clear previous suggestions

    if (!query) {
      suggestionsDiv.style.display = "none";
      return;
    }

    const matchedSuggestions = [];
    const seenSuggestions = new Set(); // Avoid duplicates

    allJobDescriptions.forEach((jd) => {
      const role = jd.role.toLowerCase();
      const queryLower = query.toLowerCase();

      // Check for matches in role only
      if (role.includes(queryLower) && !seenSuggestions.has(role)) {
        matchedSuggestions.push({
          type: "Role",
          value: jd.role,
          fullText: jd.role,
        });
        seenSuggestions.add(role);
      }
    });

    // Limit to top 5 suggestions
    const topSuggestions = matchedSuggestions.slice(0, 5);

    if (topSuggestions.length === 0) {
      suggestionsDiv.style.display = "none";
      return;
    }

    topSuggestions.forEach((suggestion) => {
      const suggestionItem = document.createElement("div");
      suggestionItem.classList.add("suggestion-item");
      suggestionItem.innerHTML = `${suggestion.fullText}`; // Removed "Role:" label since all suggestions are roles
      suggestionItem.addEventListener("click", () => {
        searchInput.value = suggestion.value;
        filterJobDescriptions(suggestion.value);
        suggestionsDiv.style.display = "none";
      });
      suggestionsDiv.appendChild(suggestionItem);
    });

    suggestionsDiv.style.display = "block";
  }

  // Initial load of job descriptions
  loadJobDescriptions();
}
