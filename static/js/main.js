// LeafScan AI — Frontend JS

const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const previewBox = document.getElementById('previewBox');
const previewImg = document.getElementById('previewImg');
const analyzeBtn = document.getElementById('analyzeBtn');
const changeBtn = document.getElementById('changeBtn');
const loadingState = document.getElementById('loadingState');
const resultCard = document.getElementById('resultCard');
const newBtn = document.getElementById('newBtn');

let selectedFile = null;

// Click to upload
uploadBox.addEventListener('click', () => fileInput.click());

// Drag & drop
uploadBox.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadBox.classList.add('drag-over');
});
uploadBox.addEventListener('dragleave', () => uploadBox.classList.remove('drag-over'));
uploadBox.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadBox.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

// File input change
fileInput.addEventListener('change', (e) => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

// Change image
changeBtn.addEventListener('click', () => {
  fileInput.click();
});

function handleFile(file) {
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
    showToast('❌ Please upload a JPG, PNG, or WEBP image');
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    showToast('❌ File size exceeds 16MB');
    return;
  }

  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    uploadBox.style.display = 'none';
    previewBox.style.display = 'block';
    resultCard.style.display = 'none';
    analyzeBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

// Analyze button click
analyzeBtn.addEventListener('click', () => {
  if (!selectedFile) return;
  runPrediction();
});

// New image button
newBtn.addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = '';
  previewBox.style.display = 'none';
  resultCard.style.display = 'none';
  uploadBox.style.display = 'block';
  analyzeBtn.disabled = true;
});

async function runPrediction() {
  analyzeBtn.disabled = true;
  loadingState.style.display = 'block';
  resultCard.style.display = 'none';
  previewBox.style.display = 'none';

  // Animate progress bar
  const progressFill = document.getElementById('progressFill');
  let progress = 0;
  const progressInterval = setInterval(() => {
    progress = Math.min(progress + Math.random() * 15, 90);
    progressFill.style.width = progress + '%';
  }, 200);

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData
    });
    const data = await response.json();

    clearInterval(progressInterval);
    progressFill.style.width = '100%';

    await sleep(400);
    loadingState.style.display = 'none';

    if (data.success) {
      showResult(data);
    } else {
      showToast('❌ ' + (data.error || 'Prediction failed'));
      resetUploadState();
    }
  } catch (err) {
    clearInterval(progressInterval);
    loadingState.style.display = 'none';
    showToast('❌ Server error. Make sure Flask is running.');
    resetUploadState();
    console.error(err);
  }
}

function showResult(data) {
  // Set image
  document.getElementById('resultImg').src = data.image_url;

  // Set disease name & icon
  document.getElementById('resultTitle').textContent = data.disease_name;
  document.getElementById('resultIcon').textContent = data.disease_name.includes('healthy') ? '✅' : '🦠';

  // Severity badge
  const badge = document.getElementById('severityBadge');
  badge.textContent = 'Severity: ' + data.severity;
  const severityClass = {
    'None': 'severity-none',
    'Low': 'severity-low',
    'Moderate': 'severity-moderate',
    'High': 'severity-high'
  }[data.severity] || 'severity-moderate';
  badge.className = 'severity-badge ' + severityClass;

  // Confidence circle
  const circle = document.getElementById('confidenceCircle');
  const confidenceNum = document.getElementById('confidenceNum');
  const conf = data.confidence;
  setTimeout(() => {
    circle.style.strokeDasharray = conf + ', 100';
    circle.style.stroke = conf > 80 ? '#76ff7a' : conf > 60 ? '#f1c40f' : '#e74c3c';
  }, 100);
  animateNumber(confidenceNum, 0, conf, 1000, '%');

  // Description & treatment
  document.getElementById('resultDescription').textContent = data.description;
  document.getElementById('resultTreatment').textContent = data.treatment;

  // Top 3 bars
  const top3Container = document.getElementById('top3Bars');
  top3Container.innerHTML = '';
  data.top3.forEach((item, i) => {
    const div = document.createElement('div');
    div.className = 'top3-bar';
    div.innerHTML = `
      <span class="top3-label">${item.name}</span>
      <div class="top3-track">
        <div class="top3-fill ${i === 0 ? 'first' : ''}" style="width:0%" data-width="${item.confidence}"></div>
      </div>
      <span class="top3-pct">${item.confidence.toFixed(1)}%</span>
    `;
    top3Container.appendChild(div);
  });

  // Show result card
  resultCard.style.display = 'block';
  resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  // Animate bars
  setTimeout(() => {
    document.querySelectorAll('.top3-fill').forEach(bar => {
      bar.style.width = bar.dataset.width + '%';
    });
  }, 200);
}

function resetUploadState() {
  analyzeBtn.disabled = false;
  previewBox.style.display = 'block';
}

function animateNumber(el, start, end, duration, suffix = '') {
  const startTime = performance.now();
  const update = (time) => {
    const elapsed = time - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const val = Math.round(start + (end - start) * easeOut(progress));
    el.textContent = val + suffix;
    if (progress < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}
function easeOut(t) { return 1 - Math.pow(1 - t, 3); }
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// Toast notification
function showToast(msg) {
  const toast = document.createElement('div');
  toast.textContent = msg;
  toast.style.cssText = `
    position: fixed; bottom: 2rem; left: 50%; transform: translateX(-50%);
    background: #1a3d1e; border: 1px solid #2d6a32; color: #f5f0e8;
    padding: 0.9rem 1.8rem; border-radius: 8px; font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem; z-index: 9999; animation: slideUp 0.3s ease;
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
  `;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3500);
}
