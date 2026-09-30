/**
 * VoiceFusion AI – Frontend App Logic  v3
 * =========================================
 * Flow:
 *  ① User selects / drops a video  →  file is stored, "Detect Language" button enables
 *  ② "Detect Language & Continue"  →  POST /api/detect (uploads video)
 *                                      Spinner shows while backend runs Whisper
 *  ③ Language card shows:
 *       - Detected language + confidence
 *       - Top-5 alternatives as clickable chips  (clicking updates source dropdown)
 *       - Source language dropdown  (pre-selected from detection)
 *       - Target language dropdown  (user picks)
 *  ④ "Start Dubbing"               →  POST /api/process with language codes
 *  ⑤ Progress card polls /api/status every 3 s with animated step tracker
 *  ⑥ Result card: audio player + WAV download
 *  ⑦ Error card + retry
 */

// ── API base ──────────────────────────────────────────────────────────────
// When served via FastAPI at /app/index.html, window.origin == server origin.
// When opened as file:// fallback, point explicitly to localhost:8000.
const API = location.protocol === 'file:'
  ? 'http://localhost:8000'
  : location.origin;

// ── DOM refs ──────────────────────────────────────────────────────────────
const dropzone      = document.getElementById('dropzone');
const videoInput    = document.getElementById('video-input');
const fileInfo      = document.getElementById('file-info');
const btnDetect     = document.getElementById('btn-detect');
const detectingBar  = document.getElementById('detecting-bar');

const uploadCard    = document.getElementById('upload-card');
const langCard      = document.getElementById('lang-card');
const progressCard  = document.getElementById('progress-card');
const resultCard    = document.getElementById('result-card');
const errorCard     = document.getElementById('error-card');

// Language card
const detFlag       = document.getElementById('det-flag');
const detLabel      = document.getElementById('det-label');
const detConf       = document.getElementById('det-confidence');
const detBadge      = document.getElementById('det-badge');
const altList       = document.getElementById('alt-list');
const selSource     = document.getElementById('sel-source');
const selTarget     = document.getElementById('sel-target');
const btnBack       = document.getElementById('btn-back');
const btnStart      = document.getElementById('btn-start');

// Progress
const progressMsg   = document.getElementById('progress-msg');
const progressBar   = document.getElementById('progress-bar');
const langRoute     = document.getElementById('lang-route');
const stepEls       = document.querySelectorAll('.step-list .step');

// Result / Error
const audioPlayer   = document.getElementById('audio-player');
const btnDownload   = document.getElementById('btn-download');
const btnReset      = document.getElementById('btn-reset');
const btnRetry      = document.getElementById('btn-retry');
const errorMsg      = document.getElementById('error-msg');
const resultSub     = document.getElementById('result-sub');

// ── State ─────────────────────────────────────────────────────────────────
let selectedFile   = null;    // File object
let detectData     = null;    // Response from /api/detect
let allLangs       = [];      // Array from /api/languages
let pollTimer      = null;
let currentStep    = -1;

const POLL_MS      = 3000;
const STEP_KW      = {
  'extract': 0, 'denoise': 0,
  'transcrib': 1,
  'translat': 2,
  'voice': 3, 'clone': 3, 'synth': 3,
  'merge': 4, 'video': 4,
};

// ── Drag-and-drop / file picker ────────────────────────────────────────────
['dragenter', 'dragover'].forEach(ev =>
  dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.add('dragover'); })
);
['dragleave', 'drop'].forEach(ev =>
  dropzone.addEventListener(ev, () => dropzone.classList.remove('dragover'))
);
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  const f = e.dataTransfer.files[0];
  if (f) setFile(f);
});
dropzone.addEventListener('click', () => videoInput.click());
videoInput.addEventListener('change', () => {
  if (videoInput.files[0]) setFile(videoInput.files[0]);
});

function setFile(file) {
  selectedFile = file;
  const mb = (file.size / 1024 / 1024).toFixed(1);
  fileInfo.textContent = `📁 ${file.name}  (${mb} MB)`;
  btnDetect.disabled = false;
}

// ── Step 1 → 2 : Detect language ──────────────────────────────────────────
btnDetect.addEventListener('click', async () => {
  if (!selectedFile) return;

  btnDetect.disabled = true;
  detectingBar.classList.remove('hidden');

  const fd = new FormData();
  fd.append('video', selectedFile);

  try {
    const res = await fetch(`${API}/api/detect`, { method: 'POST', body: fd });
    if (!res.ok) {
      let detail = `Detection failed (${res.status})`;
      try {
        const errJson = await res.json();
        if (errJson && errJson.detail) detail = errJson.detail;
      } catch (_) {}
      throw new Error(detail);
    }
    detectData = await res.json();
    allLangs   = detectData.target_languages || [];
    showLangCard(detectData);
  } catch (err) {
    if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
      showError('Cannot reach the backend. Make sure "python server.py" is running on port 8000.');
    } else {
      showError(err.message);
    }
  } finally {
    detectingBar.classList.add('hidden');
    btnDetect.disabled = false;
  }
});

// ── Render Language card ───────────────────────────────────────────────────
function showLangCard(data) {
  // Banner
  detFlag.textContent  = data.top5?.[0]?.flag || '🎙️';
  detLabel.textContent = data.label || data.detected;
  detConf.textContent  = `Confidence: ${data.confidence_pct || (data.confidence * 100).toFixed(1) + '%'}`;

  const conf = data.confidence || 0;
  if (!data.supported) {
    detBadge.textContent = 'Not supported';
    detBadge.className   = 'det-badge unsupported';
  } else if (conf >= 0.8) {
    detBadge.textContent = 'High confidence';
    detBadge.className   = 'det-badge';
  } else {
    detBadge.textContent = 'Low confidence';
    detBadge.className   = 'det-badge warn';
  }

  // Top-5 chips
  altList.innerHTML = '';
  (data.top5 || []).forEach((lang, i) => {
    const chip = document.createElement('button');
    chip.className = 'alt-chip' + (i === 0 ? ' selected' : '');
    chip.innerHTML = `<span>${lang.flag} ${lang.label}</span><span class="chip-conf">${(lang.confidence * 100).toFixed(0)}%</span>`;
    chip.title = lang.supported ? '' : 'Not supported by VoiceFusion';
    chip.disabled = !lang.supported;
    chip.addEventListener('click', () => {
      document.querySelectorAll('.alt-chip').forEach(c => c.classList.remove('selected'));
      chip.classList.add('selected');
      selSource.value = lang.code;
    });
    altList.appendChild(chip);
  });

  // Populate source dropdown
  populateSelect(selSource, allLangs, data.detected);

  // Populate target dropdown (exclude source)
  populateSelect(selTarget, allLangs, guessTarget(data.detected));

  showCard(langCard);
}

function populateSelect(sel, langs, defaultCode) {
  sel.innerHTML = '';
  langs.forEach(lang => {
    const opt = document.createElement('option');
    opt.value       = lang.code;
    opt.textContent = `${lang.flag}  ${lang.label}`;
    if (lang.code === defaultCode) opt.selected = true;
    sel.appendChild(opt);
  });
}

/** Suggest a sensible default target given the detected source. */
function guessTarget(srcCode) {
  const defaults = { ta: 'hi', hi: 'en', te: 'hi', kn: 'hi', ml: 'hi', bn: 'hi', en: 'hi', fr: 'en', es: 'en', de: 'en' };
  return defaults[srcCode] || 'en';
}

// ── Back button ────────────────────────────────────────────────────────────
btnBack.addEventListener('click', () => showCard(uploadCard));

// ── Step 2 → 3 : Start dubbing ────────────────────────────────────────────
btnStart.addEventListener('click', async () => {
  const srcCode  = selSource.value;
  const tgtCode  = selTarget.value;

  if (srcCode === tgtCode) {
    alert('Source and target languages must be different.');
    return;
  }

  // Resolve NLLB + TTS codes from allLangs
  const srcLang = allLangs.find(l => l.code === srcCode);
  const tgtLang = allLangs.find(l => l.code === tgtCode);

  if (!srcLang || !tgtLang) {
    alert('Invalid language selection.');
    return;
  }

  const fd = new FormData();
  fd.append('video',          selectedFile);
  fd.append('source_whisper', srcCode);
  fd.append('source_nllb',    srcLang.nllb);
  fd.append('target_nllb',    tgtLang.nllb);
  fd.append('target_tts',     tgtLang.tts);
  fd.append('whisper_model',  document.getElementById('opt-whisper').value);
  fd.append('nllb_size',      document.getElementById('opt-nllb').value);
  fd.append('preserve_slang', document.getElementById('opt-slang').checked ? 'true' : 'false');
  fd.append('skip_lipsync',   document.getElementById('opt-lipsync').checked ? 'true' : 'false');

  const speakerFile = document.getElementById('opt-speaker').files[0];
  if (speakerFile) fd.append('speaker_wav', speakerFile);

  // Show lang route in progress card
  langRoute.innerHTML =
    `<span class="lang-pill">${srcLang.flag} ${srcLang.label}</span>
     <span class="arrow">→</span>
     <span class="lang-pill">${tgtLang.flag} ${tgtLang.label}</span>`;

  showCard(progressCard);
  resetSteps();
  setProgress(5, 'Uploading video…');

  try {
    const res = await fetch(`${API}/api/process`, { method: 'POST', body: fd });
    if (!res.ok) {
      let detail = `Server error (${res.status})`;
      try {
        const errJson = await res.json();
        if (errJson && errJson.detail) detail = errJson.detail;
      } catch (_) {}
      throw new Error(detail);
    }
    const { job_id } = await res.json();
    setProgress(10, 'Pipeline queued. Starting…');
    startPolling(job_id, srcLang, tgtLang);
  } catch (err) {
    showError(`Processing request failed: ${err.message}`);
  }
});

// ── Polling ────────────────────────────────────────────────────────────────
function startPolling(jobId, srcLang, tgtLang) {
  clearInterval(pollTimer);
  pollTimer = setInterval(() => pollStatus(jobId, srcLang, tgtLang), POLL_MS);
}

async function pollStatus(jobId, srcLang, tgtLang) {
  try {
    const res = await fetch(`${API}/api/status/${jobId}`);
    if (!res.ok) return;
    const { status, message } = await res.json();

    updateStep(message);

    if (status === 'done') {
      clearInterval(pollTimer);
      setProgress(100, 'Complete!');
      markAllDone();
      showResult(jobId, srcLang, tgtLang);
    } else if (status === 'error') {
      clearInterval(pollTimer);
      showError(message);
    } else {
      const pct = 10 + (currentStep + 1) * (85 / stepEls.length);
      setProgress(pct, message);
    }
  } catch {
    // network hiccup – keep polling
  }
}

// ── Result ─────────────────────────────────────────────────────────────────
function showResult(jobId, srcLang, tgtLang) {
  const url = `${API}/api/download/${jobId}`;
  audioPlayer.src   = url;
  btnDownload.href  = url;
  resultSub.textContent = `${srcLang?.flag || ''} ${srcLang?.label || ''} → ${tgtLang?.flag || ''} ${tgtLang?.label || ''} dubbed audio is ready.`;
  showCard(resultCard);
}

// ── Error ──────────────────────────────────────────────────────────────────
function showError(msg) {
  errorMsg.textContent = msg;
  showCard(errorCard);
}

// ── Reset ──────────────────────────────────────────────────────────────────
function reset() {
  clearInterval(pollTimer);
  selectedFile = null; detectData = null; allLangs = [];
  videoInput.value = ''; fileInfo.textContent = '';
  btnDetect.disabled = true;
  audioPlayer.src = ''; btnDownload.href = '#';
  showCard(uploadCard);
}
btnReset.addEventListener('click', reset);
btnRetry.addEventListener('click', reset);

// ── Helpers ────────────────────────────────────────────────────────────────
function showCard(card) {
  [uploadCard, langCard, progressCard, resultCard, errorCard]
    .forEach(c => c.classList.add('hidden'));
  card.classList.remove('hidden');
}

function setProgress(pct, msg) {
  progressBar.style.width = `${pct}%`;
  progressMsg.textContent = msg;
}

function resetSteps() {
  currentStep = -1;
  stepEls.forEach(el => el.classList.remove('active', 'done'));
}

function markAllDone() {
  stepEls.forEach(el => { el.classList.remove('active'); el.classList.add('done'); });
}

function updateStep(msg) {
  if (!msg) return;
  const lower = msg.toLowerCase();
  let idx = -1;
  for (const [kw, i] of Object.entries(STEP_KW)) {
    if (lower.includes(kw)) { idx = i; break; }
  }
  if (idx > currentStep) {
    for (let i = 0; i < idx; i++) {
      stepEls[i]?.classList.remove('active');
      stepEls[i]?.classList.add('done');
    }
    stepEls[idx]?.classList.add('active');
    stepEls[idx]?.classList.remove('done');
    currentStep = idx;
  }
}
