let allResumes = [];
let originalResumes = [];
let totalResumes = 0;
let currentPage = 1;
const resumesPerPage = 10;

async function loadRankingData(jobId) {
  showLoadingSpinner(true);
  try {
    // Fetch ranking data
    const response = await fetch(`/get_ranking/${jobId}`);
    const data = await response.json();
    if (data.error) {
      alert(`Error: ${data.error}`);
      return;
    }
    allResumes = data.ranked_resumes;
    originalResumes = [...allResumes];
    totalResumes = data.total_resumes || allResumes.length;
    document.getElementById("totalResumes").textContent = totalResumes;
    document.getElementById("minRank").setAttribute("max", totalResumes);
    document.getElementById("maxRank").setAttribute("max", totalResumes);
    document.getElementById("maxRank").setAttribute("placeholder", `Max Rank (e.g., ${totalResumes})`);

    renderTable();
    renderPagination();
    renderScorePieChart();
    renderRankingBarChart();
  } catch (err) {
    alert(`Error loading ranking data: ${err.message}`);
  } finally {
    showLoadingSpinner(false);
  }
}

async function applyFilters() {
  const modal = bootstrap.Modal.getInstance(document.getElementById('filterModal'));
  modal.hide();

  showLoadingSpinner(true);
  try {
    const minRank = document.getElementById('minRank').value;
    const maxRank = document.getElementById('maxRank').value;
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;

    const filterCriteria = {};
    if (minRank) filterCriteria.minRank = parseInt(minRank);
    if (maxRank) filterCriteria.maxRank = parseInt(maxRank);
    if (startDate) filterCriteria.startDate = startDate;
    if (endDate) filterCriteria.endDate = endDate;

    const response = await fetch(`/filter_ranking/${jobId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(filterCriteria)
    });

    const data = await response.json();
    if (data.error) {
      alert(`Error: ${data.error}`);
      return;
    }

    allResumes = data.ranked_resumes;
    currentPage = 1;
    renderTable();
    renderPagination();
    renderScorePieChart();
    renderRankingBarChart();
  } catch (err) {
    alert(`Error applying filters: ${err.message}`);
  } finally {
    showLoadingSpinner(false);
  }
}

function resetFilterForm() {
  document.getElementById('filterForm').reset(); // Reset the form inputs
  const modal = bootstrap.Modal.getInstance(document.getElementById('filterModal'));
  modal.hide(); // Close the modal
}

function renderTable() {
  const tbody = document.getElementById("rankingTableBody");
  tbody.innerHTML = "";
  const start = (currentPage - 1) * resumesPerPage;
  const end = start + resumesPerPage;
  const pageResumes = allResumes.slice(start, end);

  pageResumes.forEach((resume, index) => {
    const rank = start + index + 1;
    const row = `
            <tr>
                <td>${rank}</td>
                <td>${resume.Candidate}</td>
                <td>${resume.Combined_Score.toFixed(1)}</td>
                <td>${resume.Rule_Based_Score.toFixed(1)}</td>
                <td>${resume.SBERT_Score.toFixed(1)}</td>
                <td>${resume.Experience}</td>
                <td>${resume.Certifications}</td>
                <td>${resume.Projects}</td>
                <td>${Object.values(resume.Skills).reduce(
                  (a, b) => a + b,
                  0
                )}</td>
            </tr>`;
    tbody.innerHTML += row;
  });
}

function renderPagination() {
  const totalPages = Math.ceil(allResumes.length / resumesPerPage);
  const pagination = document.getElementById("pagination");
  pagination.innerHTML = "";

  pagination.innerHTML += `
        <li class="page-item ${currentPage === 1 ? "disabled" : ""}">
            <a class="page-link" href="#" onclick="changePage(${
              currentPage - 1
            })">Previous</a>
        </li>`;

  for (let i = 1; i <= totalPages; i++) {
    pagination.innerHTML += `
            <li class="page-item ${i === currentPage ? "active" : ""}">
                <a class="page-link" href="#" onclick="changePage(${i})">${i}</a>
            </li>`;
  }

  pagination.innerHTML += `
        <li class="page-item ${currentPage === totalPages ? "disabled" : ""}">
            <a class="page-link" href="#" onclick="changePage(${
              currentPage + 1
            })">Next</a>
        </li>`;

  if (JSON.stringify(allResumes) !== JSON.stringify(originalResumes)) {
    pagination.innerHTML += `
        <li class="page-item">
            <a class="page-link text-danger" href="#" onclick="resetFilters()">Reset Filters</a>
        </li>`;
  }
}

function changePage(page) {
  const totalPages = Math.ceil(allResumes.length / resumesPerPage);
  if (page < 1 || page > totalPages) return;
  currentPage = page;
  renderTable();
  renderPagination();
}

function resetFilters() {
  allResumes = [...originalResumes];
  currentPage = 1;
  renderTable();
  renderPagination();
  renderScorePieChart();
  renderRankingBarChart();

  document.getElementById('filterForm').reset();
}

function renderScorePieChart() {
  const topCandidates = allResumes.slice(0, Math.min(5, allResumes.length));
  const labels = topCandidates.map(resume => resume.Candidate);
  const scores = topCandidates.map(resume => resume.Combined_Score);
  const backgroundColors = topCandidates.map((_, index) =>
    `hsl(${(index * 60) % 360}, 70%, 50%)`
  );

  const ctx = document.getElementById("scorePieChart").getContext("2d");
  if (ctx.chart) {
    ctx.chart.destroy();
  }
  ctx.chart = new Chart(ctx, {
    type: "pie",
    data: {
      labels: labels,
      datasets: [{
        data: scores,
        backgroundColor: backgroundColors,
        borderWidth: 1,
        borderColor: "#ffffff"
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 1,
      plugins: {
        legend: {
          position: "right",
          labels: {
            font: {
              size: 14
            }
          }
        },
        title: {
          display: true,
          text: "Combined Score Distribution of Top Candidates",
          font: {
            size: 18
          },
          padding: {
            bottom: 20
          }
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              const label = context.label || '';
              const value = context.raw || 0;
              const total = context.dataset.data.reduce((a, b) => a + b, 0);
              const percentage = ((value / total) * 100).toFixed(1);
              return `${label}: ${value.toFixed(1)} (${percentage}%)`;
            }
          },
          titleFont: {
            size: 14
          },
          bodyFont: {
            size: 12
          }
        }
      }
    }
  });
}

function renderRankingBarChart() {
  const topCandidates = allResumes.slice(0, Math.min(10, allResumes.length));
  const labels = topCandidates.map(resume => resume.Candidate);
  const scores = topCandidates.map(resume => resume.Combined_Score);

  const ctx = document.getElementById("rankingBarChart").getContext("2d");
  if (ctx.chart) {
    ctx.chart.destroy();
  }
  ctx.chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Combined Score",
        data: scores,
        backgroundColor: "#4e73df",
        borderColor: "#2e59d9",
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 2,
      scales: {
        x: {
          title: {
            display: true,
            text: "Candidates",
            font: {
              size: 14
            }
          },
          ticks: {
            font: {
              size: 12
            },
            maxRotation: 45,
            minRotation: 45
          }
        },
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: "Combined Score",
            font: {
              size: 14
            }
          },
          ticks: {
            font: {
              size: 12
            }
          }
        }
      },
      plugins: {
        legend: {
          display: false
        },
        title: {
          display: true,
          text: "Ranking Levels of Candidates",
          font: {
            size: 18
          },
          padding: {
            bottom: 20
          }
        },
        tooltip: {
          titleFont: {
            size: 14
          },
          bodyFont: {
            size: 12
          },
          callbacks: {
            label: function(context) {
              return `Score: ${context.raw.toFixed(1)}`;
            }
          }
        }
      },
      barPercentage: 0.6,
      categoryPercentage: 0.8
    }
  });
}

function showLoadingSpinner(show) {
  document.getElementById("loadingSpinner").style.display = show
    ? "block"
    : "none";
}