// DCP Web Studio - Frontend Application
// Drag-and-drop song mixer with Spotify-style animated transition visualizer

(function() {
    'use strict';

    // ============================================================
    // STATE MANAGEMENT
    // ============================================================
    const state = {
        tracks: {
            a: { file: null, analysis: null, waveform: [], element: null, objectUrl: null },
            b: { file: null, analysis: null, waveform: [], element: null, objectUrl: null }
        },
        currentMode: 'mix', // 'mix' or 'baseline'
        isPlaying: false,
        currentTime: 0,
        duration: 0,
        audioContext: null,
        audioSource: null,
        audioBuffer: null,
        gainNode: null,
        animationFrame: null,
        canvas: null,
        ctx: null,
        visualizerData: {
            mix: [],
            baseline: []
        },
        currentMixId: null
    };

    // DOM Elements
    const elements = {};

    // ============================================================
    // INITIALIZATION
    // ============================================================
    function init() {
        cacheElements();
        setupEventListeners();
        initCanvas();
        initAudioContext();
        console.log('DCP Web Studio initialized');
    }

    function cacheElements() {
        // Deck elements
        elements.dropZoneA = document.getElementById('drop-zone-a');
        elements.dropZoneB = document.getElementById('drop-zone-b');
        elements.trackInfoA = document.getElementById('track-info-a');
        elements.trackInfoB = document.getElementById('track-info-b');
        elements.trackTitleA = document.getElementById('track-title-a');
        elements.trackTitleB = document.getElementById('track-title-b');
        elements.trackBpmA = document.getElementById('track-bpm-a');
        elements.trackBpmB = document.getElementById('track-bpm-b');
        elements.trackKeyA = document.getElementById('track-key-a');
        elements.trackKeyB = document.getElementById('track-key-b');
        elements.trackDurA = document.getElementById('track-dur-a');
        elements.trackDurB = document.getElementById('track-dur-b');

        // Controls
        elements.strategySelect = document.getElementById('strategy-select');
        elements.transitionDuration = document.getElementById('transition-duration');
        elements.durationValue = document.getElementById('duration-value');
        elements.generateBtn = document.getElementById('generate-btn');

        // Player controls
        elements.playBtn = document.getElementById('play-btn');
        elements.stopBtn = document.getElementById('stop-btn');
        elements.timeDisplay = document.getElementById('time-display');
        elements.scrubber = document.getElementById('scrubber');
        elements.compareToggle = document.getElementById('compare-toggle');
        elements.downloadBtn = document.getElementById('download-btn');

        // Visualizer
        elements.canvas = document.getElementById('visualizer-canvas');
        elements.timeDisplay = document.getElementById('time-display');
        elements.scrubber = document.getElementById('scrubber');
        elements.playBtn = document.getElementById('play-btn');
        elements.stopBtn = document.getElementById('stop-btn');
        elements.compareToggle = document.getElementById('compare-toggle');
        elements.downloadBtn = document.getElementById('download-btn');

        // Visualizer canvas
        state.canvas = document.getElementById('visualizer-canvas');
        state.ctx = state.canvas.getContext('2d');
    }

    function setupEventListeners() {
        // Drag and drop
        setupDragAndDrop();

        // Strategy selection
        document.getElementById('strategy-select').addEventListener('change', (e) => {
            // Strategy changed - could update UI
        });

        // Transition duration
        document.getElementById('transition-duration').addEventListener('input', (e) => {
            document.getElementById('duration-value').textContent = `${parseFloat(e.target.value).toFixed(1)}s`;
        });

        // Generate button
        document.getElementById('generate-btn').addEventListener('click', handleGenerate);

        // Player controls
        document.getElementById('play-btn').addEventListener('click', togglePlay);
        document.getElementById('stop-btn').addEventListener('click', stopPlayback);
        document.getElementById('compare-toggle').addEventListener('change', (e) => {
            state.currentMode = e.target.checked ? 'mix' : 'baseline';
            if (state.isPlaying) {
                stopPlayback();
                playCurrentMode();
            }
        });

        // Scrubber
        const scrubber = document.getElementById('scrubber');
        scrubber.addEventListener('input', (e) => {
            const percent = parseFloat(e.target.value) / 100;
            seek(percent * state.duration);
        });

        // Download
        document.getElementById('download-btn').addEventListener('click', handleDownload);

        // Swap decks
        document.getElementById('swap-btn').addEventListener('click', swapDecks);

        // Canvas resize
        window.addEventListener('resize', handleResize);
    }

    function setupDragAndDrop() {
        ['a', 'b'].forEach(deck => {
            const zone = document.getElementById(`drop-zone-${deck}`);

            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(event => {
                document.getElementById(`drop-zone-${deck}`).addEventListener(event, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                }, false);
            });

            document.getElementById(`drop-zone-${deck}`).addEventListener('dragenter', (e) => {
                e.target.closest('.drop-zone').classList.add('drag-active');
            });

            document.getElementById(`drop-zone-${deck}`).addEventListener('dragleave', (e) => {
                if (!e.currentTarget.contains(e.relatedTarget)) {
                    e.target.closest('.drop-zone').classList.remove('drag-active');
                }
            });

            document.getElementById(`drop-zone-${deck}`).addEventListener('drop', (e) => {
                e.preventDefault();
                e.target.closest('.drop-zone').classList.remove('drag-active');

                const file = e.dataTransfer.files[0];
                if (file && file.type.startsWith('audio/')) {
                    handleFileUpload(file, deck);
                }
            });

            // Click to upload
            document.getElementById(`drop-zone-${deck}`).addEventListener('click', () => {
                createFileInput(deck).click();
            });
        });
    }

    function createFileInput(deck) {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'audio/*';
        input.style.display = 'none';
        input.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) handleFileUpload(file, deck);
        });
        document.body.appendChild(input);
        return input;
    }

    function handleFileUpload(file, deck) {
        const track = state.tracks[deck];
        track.file = file;

        // Show loading state
        const zone = document.getElementById(`drop-zone-${deck}`);
        zone.classList.add('loading');
        zone.innerHTML = '<div class="loading-spinner"></div><p>Analyzing...</p>';

        // Create object URL for preview
        track.objectUrl = URL.createObjectURL(file);

        // Analyze the file
        analyzeAudioFile(file, deck).then(analysis => {
            state.tracks[deck].analysis = analysis;
            track.file = file;
            track.waveform = generateWaveformPlaceholder(); // Placeholder until real analysis

            // Update UI
            updateTrackInfo(deck, file, analysis);
            document.getElementById(`track-info-${deck}`).style.display = 'block';
            document.getElementById(`drop-zone-${deck}`).classList.add('has-file');
            document.getElementById(`drop-zone-${deck}`).innerHTML = `
                <svg class="file-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                    <polyline points="14 2 14 8 20 8"/>
                </svg>
                <p class="file-name">${file.name}</p>
                <p class="file-size">${formatFileSize(file.size)}</p>
            `;
        }).catch(err => {
            console.error('Analysis failed:', err);
            alert('Failed to analyze audio file: ' + err.message);
            document.getElementById(`drop-zone-${deck}`).classList.remove('has-file');
        });
    }

    function analyzeAudioFile(file, deck) {
        return new Promise((resolve, reject) => {
            const formData = new FormData();
            formData.append('audio', file);

            fetch('/api/analyze', {
                method: 'POST',
                body: new FormData().append('audio', file)
            })
            .then(response => response.json())
            .then(data => resolve(data))
            .catch(reject);
        });
    }

    function updateTrackInfo(deck, file, analysis) {
        const info = document.getElementById(`track-info-${deck}`);
        const titleEl = document.getElementById(`track-title-${deck}`);
        const bpmEl = document.getElementById(`track-bpm-${deck}`);
        const keyEl = document.getElementById(`track-key-${deck}`);
        const durEl = document.getElementById(`track-dur-${deck}`);

        titleEl.textContent = file.name;
        bpmEl.textContent = analysis.tempo_bpm ? `${analysis.tempo_bpm.toFixed(1)} BPM` : 'Unknown';
        keyEl.textContent = analysis.key ? `${analysis.key} (${analysis.camelot_code || ''})` : 'Unknown';
        durEl.textContent = formatTime(analysis.duration_s);
    }

    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    function formatTime(seconds) {
        if (isNaN(seconds) || !isFinite(seconds)) return '0:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    function generateWaveformPlaceholder() {
        // Generate a placeholder waveform until real analysis is done
        const arr = [];
        for (let i = 0; i < 100; i++) {
            arr.push(Math.random() * 0.5);
        }
        return arr;
    }

    // ============================================================
    // AUDIO PLAYBACK
    // ============================================================
    function initAudioContext() {
        if (!state.audioContext) {
            state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (state.audioContext.state === 'suspended') {
            state.audioContext.resume();
        }
    }

    function playCurrentMode() {
        const mode = state.currentMode;
        const url = mode === 'mix'
            ? `/api/audio/mix_${state.currentMixId}.wav`
            : `/api/audio/baseline_${state.currentMixId}.wav`;

        fetch(url)
            .then(response => response.arrayBuffer())
            .then(arrayBuffer => state.audioContext.decodeAudioData(arrayBuffer))
            .then(audioBuffer => {
                playAudioBuffer(audioBuffer);
            })
            .catch(err => {
                console.error('Failed to load audio:', err);
                alert('Failed to load audio: ' + err.message);
            });
    }

    function playAudioBuffer(audioBuffer) {
        if (state.audioSource) {
            state.audioSource.stop();
        }

        state.audioBuffer = audioBuffer;
        state.audioSource = state.audioContext.createBufferSource();
        state.audioSource.buffer = audioBuffer;

        state.gainNode = state.audioContext.createGain();
        state.audioSource.connect(state.gainNode);
        state.gainNode.connect(state.audioContext.destination);

        state.audioSource.start(0, state.currentTime);
        state.isPlaying = true;
        state.duration = audioBuffer.duration;

        updatePlayButton(true);
        startAnimationLoop();
    }

    function stopPlayback() {
        if (state.audioSource) {
            state.audioSource.stop();
            state.audioSource.disconnect();
            state.audioSource = null;
        }
        if (state.gainNode) {
            state.gainNode.disconnect();
            state.gainNode = null;
        }
        state.isPlaying = false;
        state.currentTime = 0;
        cancelAnimationFrame(state.animationFrame);
        state.animationFrame = null;
        updatePlayButton(false);
        updateTimeDisplay(0, state.duration);
        state.scrubber.value = 0;
    }

    function togglePlay() {
        if (state.isPlaying) {
            stopPlayback();
        } else {
            playCurrentMode();
        }
    }

    function playAudioBuffer(audioBuffer) {
        if (state.audioSource) {
            state.audioSource.stop();
        }

        state.audioBuffer = audioBuffer;
        state.audioSource = state.audioContext.createBufferSource();
        state.audioSource.buffer = audioBuffer;

        state.gainNode = state.audioContext.createGain();
        state.audioSource.connect(state.gainNode);
        state.gainNode.connect(state.audioContext.destination);

        state.audioSource.start(0, state.currentTime);
        state.isPlaying = true;
        state.duration = audioBuffer.duration;

        updatePlayButton(true);
        startAnimationLoop();
    }

    function updatePlayButton(playing) {
        const btn = document.getElementById('play-btn');
        if (playing) {
            btn.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>';
        } else {
            btn.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21"/></svg>';
        }
    }

    function seek(percent) {
        if (!state.audioBuffer) return;
        stopPlayback();
        state.currentTime = percent * state.duration;
        playCurrentMode(); // This would need to seek in the audio buffer
    }

    function updateTimeDisplay(current, total) {
        const currentStr = formatTime(current);
        const totalStr = formatTime(total);
        document.getElementById('time-display').textContent = `${currentStr} / ${totalStr}`;
        document.getElementById('scrubber').value = total > 0 ? (current / total) * 100 : 0;
    }

    function formatTime(seconds) {
        if (isNaN(seconds) || !isFinite(seconds)) return '0:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    function startAnimationLoop() {
        function animate() {
            if (!state.isPlaying) return;

            state.currentTime = state.audioContext.currentTime - state.startTime;
            if (state.currentTime >= state.duration) {
                stopPlayback();
                return;
            }

            updateTimeDisplay(state.currentTime, state.duration);
            renderVisualizer();
            state.animationFrame = requestAnimationFrame(animate);
        }
        state.startTime = state.audioContext.currentTime - state.currentTime;
        state.animationFrame = requestAnimationFrame(animate);
    }

    // ============================================================
    // VISUALIZER
    // ============================================================
    function initCanvas() {
        const canvas = document.getElementById('visualizer-canvas');
        state.canvas = canvas;
        state.ctx = canvas.getContext('2d');

        handleResize();
        window.addEventListener('resize', handleResize);
    }

    function handleResize() {
        const canvas = state.canvas;
        const container = canvas.parentElement;
        const rect = container.getBoundingClientRect();

        canvas.width = rect.width * window.devicePixelRatio;
        canvas.height = rect.height * window.devicePixelRatio;
        canvas.style.width = rect.width + 'px';
        canvas.style.height = rect.height + 'px';
        canvas.getContext('2d').scale(window.devicePixelRatio, window.devicePixelRatio);
    }

    function renderVisualizer() {
        const canvas = state.canvas;
        const ctx = state.ctx;
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.fillStyle = '#0a0b0e';
        ctx.fillRect(0, 0, width, height);

        // Draw waveforms for both tracks
        drawWaveform(ctx, state.visualizerData.mix || [], 0, height / 2, '#00f5d4');
        drawWaveform(ctx, state.visualizerData.baseline || [], height / 2, height, '#ff007f');

        // Draw transition zone
        drawTransitionZone(ctx);

        // Draw playhead
        drawPlayhead(ctx);
    }

    function drawWaveform(ctx, data, yOffset, height, color) {
        if (!data || data.length === 0) return;

        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.shadowColor = color;
        ctx.shadowBlur = 8;

        const width = state.canvas.width;
        const centerY = yOffset + height / 2;
        const scale = height / 2 * 0.8;

        ctx.beginPath();
        ctx.moveTo(0, centerY);

        const step = width / (data.length - 1);
        for (let i = 0; i < data.length; i++) {
            const x = i * step;
            const y = centerY - data[i] * scale;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.shadowBlur = 0;
    }

    function drawTransitionZone(ctx) {
        const width = state.canvas.width;
        const height = state.canvas.height;
        const centerX = width / 2;
        const zoneWidth = width * 0.15; // 15% of width for transition zone

        ctx.save();
        ctx.strokeStyle = '#a78bfa';
        ctx.lineWidth = 2;
        ctx.setLineDash([10, 10]);
        ctx.beginPath();
        ctx.moveTo(centerX - zoneWidth/2, 0);
        ctx.lineTo(centerX - zoneWidth/2, canvas.height);
        ctx.moveTo(centerX + zoneWidth/2, 0);
        ctx.lineTo(centerX + zoneWidth/2, canvas.height);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
    }

    function drawPlayhead(ctx, x, progress) {
        const height = state.canvas.height;

        // Playhead line
        ctx.save();
        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();

        // Time marker
        ctx.save();
        ctx.fillStyle = '#fbbf24';
        ctx.font = '11px JetBrains Mono';
        ctx.textAlign = 'center';
        ctx.fillText(formatTime(state.currentTime), x, 20);
        ctx.restore();
    }

    // ============================================================
    // DOWNLOAD
    // ============================================================
    function handleDownload() {
        const mode = state.currentMode;
        const filename = mode === 'mix' ? `mix_${Date.now()}.wav` : `baseline_${Date.now()}.wav`;
        const url = mode === 'mix'
            ? `/api/audio/mix_${state.currentMixId}.wav`
            : `/api/audio/baseline_${state.currentMixId}.wav`;

        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
    }

    // ============================================================
    // DECK MANAGEMENT
    // ============================================================
    function swapDecks() {
        // Swap track A and B
        const tempFile = state.tracks.a.file;
        const tempAnalysis = state.tracks.a.analysis;
        const tempWaveform = state.tracks.a.waveform;

        state.tracks.a.file = state.tracks.b.file;
        state.tracks.a.analysis = state.tracks.b.analysis;
        state.tracks.a.waveform = state.tracks.b.waveform;

        state.tracks.b.file = tempFile;
        state.tracks.b.analysis = tempAnalysis;
        state.tracks.b.waveform = tempWaveform;

        // Update UI
        const tempInfo = document.getElementById('track-info-a').innerHTML;
        document.getElementById('track-info-a').innerHTML = document.getElementById('track-info-b').innerHTML;
        document.getElementById('track-info-b').innerHTML = tempInfo;

        // Swap file references
        const tempFileA = state.tracks.a.file;
        state.tracks.a.file = state.tracks.b.file;
        state.tracks.b.file = tempFileA;
    }

    function swapDecks() {
        // Swap track A and B
        const tempFile = state.tracks.a.file;
        const tempAnalysis = state.tracks.a.analysis;
        const tempWaveform = state.tracks.a.waveform;

        state.tracks.a.file = state.tracks.b.file;
        state.tracks.a.analysis = state.tracks.b.analysis;
        state.tracks.a.waveform = state.tracks.b.waveform;

        state.tracks.b.file = tempFile;
        state.tracks.b.analysis = tempAnalysis;
        state.tracks.b.waveform = tempWaveform;

        // Update UI
        const tempInfo = document.getElementById('track-info-a').innerHTML;
        document.getElementById('track-info-a').innerHTML = document.getElementById('track-info-b').innerHTML;
        document.getElementById('track-info-b').innerHTML = tempInfo;

        // Swap file references
        const tempFileA = state.tracks.a.file;
        state.tracks.a.file = state.tracks.b.file;
        state.tracks.b.file = tempFileA;
    }

    function handleDownload() {
        // Implementation for downloading the current mix
        alert('Download functionality will be implemented with the actual audio file URLs');
    }

    // ============================================================
    // INITIALIZATION
    // ============================================================
    function init() {
        initAudioContext();
        cacheElements();
        setupEventListeners();
        initCanvas();
        console.log('DCP Web Studio initialized');
    }

    // ============================================================
    // INITIALIZATION
    // ============================================================
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();