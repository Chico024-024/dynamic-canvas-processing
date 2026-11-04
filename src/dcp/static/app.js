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
        audioBuffers: { mix: null, baseline: null },
        gainNode: null,
        animationFrame: null,
        canvas: null,
        ctx: null,
        visualizerData: {
            mix: [],
            baseline: []
        },
        transitionStartS: 0,
        transitionDurationS: 8,
        songADurationS: 0,
        songBDurationS: 0,
        currentMixId: null,
        // Studio Remix (Song C) visual markers
        studioRemix: {
            launchpadS: null,      // T_launch in Song A
            impactS: null,         // T_impact in Song B
            breathStartS: null,    // Breath window start
            breathEndS: null,      // Breath window end
            bridgeStartS: null,    // Bridge start in Song A
            bridgeBars: 12,        // Total bridge bars (8 + 4 ghost)
            phaseLocked: false,
            lowEndMonogamy: false,
            zdfSweeps: false,
            ghostResolution: false,
            sidechainDucking: false,
        }
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
            const isRemix = e.target.value === 'studio_remix';
            const durationInput = document.getElementById('transition-duration');
            const durationVal = document.getElementById('duration-value');
            if (isRemix) {
                durationInput.disabled = true;
                durationInput.dataset.previousValue = durationInput.value;
                durationVal.textContent = "Autonomous (AI Decides)";
                durationVal.style.color = "var(--color-accent-amber)";
            } else {
                durationInput.disabled = false;
                durationVal.style.color = "";
                durationVal.textContent = `${parseFloat(durationInput.value).toFixed(1)}s`;
            }
        });

        // Transition duration
        document.getElementById('transition-duration').addEventListener('input', (e) => {
            document.getElementById('duration-value').textContent = `${parseFloat(e.target.value).toFixed(1)}s`;
        });

        // Generate button
        document.getElementById('generate-btn').addEventListener('click', handleGenerate);

        // Generate transition handler
        async function handleGenerate() {
            // Validate both tracks are loaded
            if (!state.tracks.a.file || !state.tracks.b.file) {
                alert('Please load both Song A and Song B before generating a transition.');
                return;
            }

            // Get form values
            const strategy = document.getElementById('strategy-select').value;
            const transitionDuration = parseFloat(document.getElementById('transition-duration').value) || 8.0;

            // Show loading state
            const generateBtn = document.getElementById('generate-btn');
            const originalBtnText = generateBtn.innerHTML;
            generateBtn.disabled = true;
            generateBtn.innerHTML = '<div class="loading-spinner"></div><span>Generating...</span>';

            try {
                // Prepare form data
                const formData = new FormData();
                formData.append('song_a', state.tracks.a.file);
                formData.append('song_b', state.tracks.b.file);
                formData.append('strategy', strategy);
                formData.append('transition_duration', transitionDuration);

                // Call the mix API
                const response = await fetch('/api/mix', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.error || 'Failed to generate transition');
                }

                const result = await response.json();

                // Store the mix URLs and ID for playback/download
                state.currentMixId = result.mix_id;
                state.mixUrl = result.mix_url;
                state.baselineUrl = result.baseline_url;
                state.duration = result.duration_s || 0;
                state.transitionStartS = result.transition_start_s || 0;
                state.transitionDurationS = result.transition_duration_s || 8;
                state.songADurationS = result.song_a_duration_s || 0;
                state.songBDurationS = result.song_b_duration_s || 0;
                state.currentTime = 0;
                state.audioBuffers = { mix: null, baseline: null };

                // Update visualizer data
                if (result.waveform_a) state.visualizerData.mix = result.waveform_a;
                if (result.waveform_b) state.visualizerData.baseline = result.waveform_b;

                // Extract Studio Remix metadata if available
                if (result.metadata && result.metadata.studio_remix) {
                    const meta = result.metadata;
                    state.studioRemix = {
                        launchpadS: meta.launchpad_a_s || null,
                        impactS: meta.impact_b_s || null,
                        breathStartS: meta.breath_start_s || null,
                        breathEndS: meta.breath_end_s || null,
                        bridgeStartS: meta.bridge_start_a_s || null,
                        bridgeBars: meta.total_bridge_bars || 12,
                        phaseLocked: meta.phase_locked || false,
                        lowEndMonogamy: meta.low_end_monogamy || false,
                        zdfSweeps: meta.zdf_filter_sweeps || false,
                        ghostResolution: meta.ghost_resolution || false,
                        sidechainDucking: meta.sidechain_ducking || false,
                    };
                    console.log('Studio Remix anchors loaded:', state.studioRemix);
                }

                // Update visualizer
                handleResize(); // Re-render with new data

                // Enable player and download
                document.getElementById('play-btn').disabled = false;
                document.getElementById('download-btn').disabled = false;

                // Show success
                alert('Transition generated successfully!');
            } catch (error) {
                console.error('Generation failed:', error);
                alert('Failed to generate transition: ' + error.message);
            } finally {
                // Restore button
                generateBtn.disabled = false;
                generateBtn.innerHTML = originalBtnText;
            }
        }

        // Swap decks

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
            const fraction = parseFloat(e.target.value) / 100;
            seek(fraction * state.duration);
        });

        // Jump to transition button
        const jumpBtn = document.getElementById('jump-btn');
        if (jumpBtn) {
            jumpBtn.addEventListener('click', () => {
                if (!state.duration) return;
                const target = Math.max(0, (state.transitionStartS || 0) - 3.5);
                seek(target);
                if (!state.isPlaying) {
                    playCurrentMode();
                }
            });
        }

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
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => Promise.reject(new Error(err.error || 'Analysis failed')));
                }
                return response.json();
            })
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

    function startSourceAt(audioBuffer, offset) {
        if (state.audioSource) {
            try { state.audioSource.stop(); } catch (e) {}
            try { state.audioSource.disconnect(); } catch (e) {}
            state.audioSource = null;
        }

        offset = Math.max(0, Math.min(audioBuffer.duration, offset));
        state.audioSource = state.audioContext.createBufferSource();
        state.audioSource.buffer = audioBuffer;

        state.gainNode = state.audioContext.createGain();
        state.audioSource.connect(state.gainNode);
        state.gainNode.connect(state.audioContext.destination);

        state.audioSource.start(0, offset);
        state.isPlaying = true;
        state.startTime = state.audioContext.currentTime - offset;
        state.duration = audioBuffer.duration;

        updatePlayButton(true);
        startAnimationLoop();
    }

    function playCurrentMode() {
        initAudioContext();
        const mode = state.currentMode;
        const url = mode === 'mix'
            ? (state.mixUrl || `/api/audio/mix_${state.currentMixId}.wav`)
            : (state.baselineUrl || `/api/audio/baseline_${state.currentMixId}.wav`);

        if (!url) return;

        if (state.audioBuffers[mode]) {
            startSourceAt(state.audioBuffers[mode], state.currentTime);
            return;
        }

        const playBtn = document.getElementById('play-btn');
        playBtn.classList.add('loading');
        fetch(url)
            .then(response => {
                if (!response.ok) throw new Error('Audio file not found');
                return response.arrayBuffer();
            })
            .then(arrayBuffer => state.audioContext.decodeAudioData(arrayBuffer))
            .then(audioBuffer => {
                playBtn.classList.remove('loading');
                state.audioBuffers[mode] = audioBuffer;
                state.duration = audioBuffer.duration;
                startSourceAt(audioBuffer, state.currentTime);
            })
            .catch(err => {
                playBtn.classList.remove('loading');
                console.error('Failed to load audio:', err);
                alert('Failed to load audio: ' + err.message);
            });
    }

    function stopPlayback() {
        if (state.audioSource) {
            try { state.audioSource.stop(); } catch (e) {}
            try { state.audioSource.disconnect(); } catch (e) {}
            state.audioSource = null;
        }
        if (state.gainNode) {
            try { state.gainNode.disconnect(); } catch (e) {}
            state.gainNode = null;
        }
        state.isPlaying = false;
        state.currentTime = 0;
        cancelAnimationFrame(state.animationFrame);
        state.animationFrame = null;
        updatePlayButton(false);
        updateTimeDisplay(0, state.duration);
        renderVisualizer();
    }

    function togglePlay() {
        if (state.isPlaying) {
            if (state.audioSource) {
                try { state.audioSource.stop(); } catch (e) {}
                try { state.audioSource.disconnect(); } catch (e) {}
                state.audioSource = null;
            }
            state.isPlaying = false;
            cancelAnimationFrame(state.animationFrame);
            state.animationFrame = null;
            updatePlayButton(false);
        } else {
            playCurrentMode();
        }
    }

    function updatePlayButton(playing) {
        const btn = document.getElementById('play-btn');
        if (playing) {
            btn.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>';
        } else {
            btn.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21"/></svg>';
        }
    }

    function seek(targetTime) {
        if (!state.duration) return;
        targetTime = Math.max(0, Math.min(state.duration, targetTime));
        state.currentTime = targetTime;
        updateTimeDisplay(state.currentTime, state.duration);
        renderVisualizer();

        if (state.isPlaying) {
            const buffer = state.audioBuffers[state.currentMode];
            if (buffer) {
                startSourceAt(buffer, state.currentTime);
            }
        }
    }

    function updateTimeDisplay(current, total) {
        const currentStr = formatTime(current);
        const totalStr = formatTime(total);
        document.getElementById('time-display').textContent = `${currentStr} / ${totalStr}`;
        const scrubber = document.getElementById('scrubber');
        if (scrubber) scrubber.value = total > 0 ? (current / total) * 100 : 0;
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

        let isDragging = false;
        function seekFromMouseEvent(e) {
            if (!state.duration) return;
            const rect = canvas.getBoundingClientRect();
            const clientX = e.clientX || (e.touches && e.touches[0] ? e.touches[0].clientX : 0);
            const fraction = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
            seek(fraction * state.duration);
        }

        canvas.addEventListener('click', seekFromMouseEvent);
        canvas.addEventListener('mousedown', (e) => {
            isDragging = true;
            seekFromMouseEvent(e);
        });
        window.addEventListener('mousemove', (e) => {
            if (isDragging) {
                seekFromMouseEvent(e);
            }
        });
        window.addEventListener('mouseup', () => {
            isDragging = false;
        });

        canvas.style.cursor = 'pointer';
    }

    function handleResize() {
        const canvas = state.canvas;
        if (!canvas) return;
        const container = canvas.parentElement;
        if (!container) return;
        const rect = container.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return;

        canvas.width = rect.width;
        canvas.height = rect.height;
        renderVisualizer();
    }

    function renderVisualizer() {
        const canvas = state.canvas;
        if (!canvas) return;
        const ctx = state.ctx;
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.fillStyle = '#0a0b0e';
        ctx.fillRect(0, 0, width, height);

        const duration = state.duration || 1;
        const transStart = state.transitionStartS || (duration * 0.45);
        const transDur = state.transitionDurationS || 8;
        const transEnd = transStart + transDur;

        const transStartX = (transStart / duration) * width;
        const transEndX = Math.min(width, (transEnd / duration) * width);
        const transWidth = Math.max(10, transEndX - transStartX);

        // Draw Transition Zone Background Glow
        ctx.save();
        const grad = ctx.createLinearGradient(transStartX, 0, transEndX, 0);
        grad.addColorStop(0, 'rgba(0, 245, 212, 0.15)');
        grad.addColorStop(0.5, 'rgba(167, 139, 250, 0.3)');
        grad.addColorStop(1, 'rgba(255, 0, 127, 0.15)');
        ctx.fillStyle = grad;
        ctx.fillRect(transStartX, 0, transWidth, height);

        // Transition zone border lines
        ctx.strokeStyle = '#a78bfa';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(transStartX, 0);
        ctx.lineTo(transStartX, height);
        ctx.moveTo(transEndX, 0);
        ctx.lineTo(transEndX, height);
        ctx.stroke();
        ctx.setLineDash([]);

        // Label transition zone
        ctx.fillStyle = '#c4b5fd';
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`⚡ TRANSITION (${transDur.toFixed(1)}s)`, (transStartX + transEndX) / 2, 14);
        ctx.restore();

        // Draw Deck A waveform (plays from 0 to transEnd)
        const waveformA = state.visualizerData.mix || [];
        if (waveformA.length > 0) {
            drawDeckWaveform(ctx, waveformA, 0, transEndX, 0, height / 2, '#00f5d4', 'DECK A');
        }

        // Draw Deck B waveform (enters at transStartX and plays to the end)
        const waveformB = state.visualizerData.baseline || [];
        if (waveformB.length > 0) {
            drawDeckWaveform(ctx, waveformB, transStartX, width, height / 2, height / 2, '#ff007f', 'DECK B');
        }

        // Draw playhead
        drawPlayhead(ctx);

        // Draw Studio Remix visual markers if available
        drawStudioRemixMarkers(ctx, width, height, duration);
    }

    function drawStudioRemixMarkers(ctx, width, height, duration) {
        const sr = state.studioRemix;
        if (!sr || !sr.phaseLocked) return;

        // Helper to convert time to x position
        const timeToX = (timeS) => (timeS / duration) * width;

        // Draw Launchpad marker (Song A)
        if (sr.launchpadS !== null) {
            const x = timeToX(sr.launchpadS);
            drawVerticalMarker(ctx, x, height, '#00f5d4', '🚀 LAUNCHPAD', 0);
        }

        // Draw Impact marker (Song B) - aligned with launchpad
        if (sr.impactS !== null) {
            const x = timeToX(sr.impactS);
            drawVerticalMarker(ctx, x, height, '#ff007f', '💥 IMPACT', height / 2);
        }

        // Draw Breath Window
        if (sr.breathStartS !== null && sr.breathEndS !== null) {
            const startX = timeToX(sr.breathStartS);
            const endX = timeToX(sr.breathEndS);
            drawBreathWindow(ctx, startX, endX, height);
        }

        // Draw Bridge Bars (12-bar grid)
        if (sr.bridgeStartS !== null && sr.bridgeBars > 0) {
            drawBridgeBars(ctx, sr.bridgeStartS, sr.bridgeBars, duration, width, height);
        }

        // Draw Low-End Monogamy indicator
        if (sr.lowEndMonogamy && sr.launchpadS !== null) {
            drawLowEndMonogamyIndicator(ctx, sr.launchpadS, duration, width, height);
        }

        // Draw ZDF Sweep indicator
        if (sr.zdfSweeps && sr.launchpadS !== null) {
            drawZDFSweepIndicator(ctx, sr.launchpadS, sr.bridgeBars, duration, width, height);
        }

        // Draw Ghost Resolution indicator
        if (sr.ghostResolution && sr.launchpadS !== null) {
            drawGhostResolutionIndicator(ctx, sr.launchpadS, sr.bridgeBars, duration, width, height);
        }

        // Draw Sidechain Ducking indicator
        if (sr.sidechainDucking && sr.impactS !== null) {
            drawSidechainIndicator(ctx, sr.impactS, duration, width, height);
        }
    }

    function drawVerticalMarker(ctx, x, height, color, label, labelYOffset) {
        ctx.save();
        // Glowing vertical line
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.shadowColor = color;
        ctx.shadowBlur = 8;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Label
        ctx.fillStyle = color;
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        const labelY = labelYOffset === 0 ? 20 : height - 20;
        ctx.fillText(label, x, labelY);
        ctx.restore();
    }

    function drawBreathWindow(ctx, startX, endX, height) {
        ctx.save();
        // Semi-transparent breath window
        const grad = ctx.createLinearGradient(startX, 0, endX, 0);
        grad.addColorStop(0, 'rgba(167, 139, 250, 0.05)');
        grad.addColorStop(0.5, 'rgba(0, 245, 212, 0.15)');
        grad.addColorStop(1, 'rgba(255, 0, 127, 0.05)');
        ctx.fillStyle = grad;
        ctx.fillRect(startX, 0, endX - startX, height);

        // Breath window borders
        ctx.strokeStyle = '#a78bfa';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(startX, 0);
        ctx.lineTo(startX, height);
        ctx.moveTo(endX, 0);
        ctx.lineTo(endX, height);
        ctx.stroke();
        ctx.setLineDash([]);

        // Label
        ctx.fillStyle = '#a78bfa';
        ctx.font = '9px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.fillText('🫧 BREATH WINDOW', (startX + endX) / 2, height / 2);
        ctx.restore();
    }

    function drawBridgeBars(ctx, bridgeStartS, bridgeBars, duration, width, height) {
        const startX = (bridgeStartS / duration) * width;
        const barWidth = (bridgeBars * (duration / bridgeBars) / duration) * width / bridgeBars; // Simplified
        const totalBridgeWidth = (bridgeBars * (duration / bridgeBars)) / duration * width; // Actually just proportional

        ctx.save();
        ctx.strokeStyle = 'rgba(167, 139, 250, 0.3)';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 6]);

        // Calculate bar duration in seconds
        const barDurationS = duration / bridgeBars;

        for (let i = 1; i < bridgeBars; i++) {
            const barTime = bridgeStartS + i * barDurationS;
            const x = (barTime / duration) * width;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }

        // Highlight bar 4 (end of low-end monogamy) and bar 8 (breath/impact)
        ctx.strokeStyle = '#a78bfa';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([]);

        // Bar 4 boundary
        if (bridgeBars >= 4) {
            const bar4Time = bridgeStartS + 4 * barDurationS;
            const x = (bar4Time / duration) * width;
            ctx.beginPath();
            ctx.moveTo(x, height / 2);
            ctx.lineTo(x, height);
            ctx.stroke();
            ctx.fillStyle = '#a78bfa';
            ctx.font = '8px JetBrains Mono, monospace';
            ctx.textAlign = 'center';
            ctx.fillText('BAR 4: Bass Fade End', x, height - 8);
        }

        // Bar 8 boundary (breath/impact)
        if (bridgeBars >= 8) {
            const bar8Time = bridgeStartS + 8 * barDurationS;
            const x = (bar8Time / duration) * width;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height / 2);
            ctx.stroke();
            ctx.fillStyle = '#a78bfa';
            ctx.font = '8px JetBrains Mono, monospace';
            ctx.textAlign = 'center';
            ctx.fillText('BAR 8: BREATH / IMPACT', x, height / 2 - 8);
        }

        ctx.restore();
    }

    function drawLowEndMonogamyIndicator(ctx, launchpadS, duration, width, height) {
        ctx.save();
        const x = (launchpadS / duration) * width;
        const barDurationS = (duration - launchpadS) / 12; // Approximate
        const bar4Time = launchpadS + 4 * barDurationS;
        const bar4X = (bar4Time / duration) * width;

        // Draw "Low-End Monogamy" zone on Song A side (bottom half)
        ctx.fillStyle = 'rgba(167, 139, 250, 0.1)';
        ctx.fillRect(x, height / 2, bar4X - x, height / 2);

        ctx.fillStyle = '#a78bfa';
        ctx.font = '9px JetBrains Mono, monospace';
        ctx.textAlign = 'left';
        ctx.fillText('🔒 LOW-END MONOGAMY: Song A bass ends, Song B bass begins', x + 4, height * 0.75);
        ctx.restore();
    }

    function drawZDFSweepIndicator(ctx, launchpadS, bridgeBars, duration, width, height) {
        ctx.save();
        const barDurationS = (duration - launchpadS) / bridgeBars;
        const sweepStartTime = launchpadS + 4 * barDurationS; // Bar 5
        const sweepEndTime = launchpadS + 8 * barDurationS;   // Bar 8
        const startX = (sweepStartTime / duration) * width;
        const endX = (sweepEndTime / duration) * width;

        // Draw sweep arrow on Song A side
        ctx.strokeStyle = '#00f5d4';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 3]);
        ctx.beginPath();
        ctx.moveTo(startX, height * 0.3);
        ctx.lineTo(endX, height * 0.2);
        ctx.stroke();
        ctx.setLineDash([]);

        // Arrow head
        ctx.fillStyle = '#00f5d4';
        ctx.beginPath();
        ctx.moveTo(endX, height * 0.2);
        ctx.lineTo(endX - 8, height * 0.25);
        ctx.lineTo(endX - 8, height * 0.15);
        ctx.closePath();
        ctx.fill();

        ctx.fillStyle = '#00f5d4';
        ctx.font = '9px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.fillText('📈 ZDF HPF SWEEP 30→950 Hz', (startX + endX) / 2, height * 0.15);
        ctx.restore();
    }

    function drawGhostResolutionIndicator(ctx, launchpadS, bridgeBars, duration, width, height) {
        ctx.save();
        const barDurationS = (duration - launchpadS) / bridgeBars;
        const ghostStartTime = launchpadS + 8 * barDurationS; // Bar 9
        const ghostEndTime = duration;
        const startX = (ghostStartTime / duration) * width;
        const endX = (ghostEndTime / duration) * width;

        // Ghost resolution zone (frozen vocal tail)
        const grad = ctx.createLinearGradient(startX, 0, endX, 0);
        grad.addColorStop(0, 'rgba(167, 139, 250, 0.3)');
        grad.addColorStop(1, 'rgba(167, 139, 250, 0.0)');
        ctx.fillStyle = grad;
        ctx.fillRect(startX, 0, endX - startX, height);

        // Sparkle particles
        ctx.fillStyle = '#a78bfa';
        for (let i = 0; i < 12; i++) {
            const px = startX + Math.random() * (endX - startX);
            const py = Math.random() * height;
            const size = 1 + Math.random() * 2;
            ctx.beginPath();
            ctx.arc(px, py, size, 0, Math.PI * 2);
            ctx.fill();
        }

        ctx.fillStyle = '#c4b5fd';
        ctx.font = '9px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.fillText('✨ GHOST RESOLUTION: Frozen vocal tail + sidechain duck', (startX + endX) / 2, height * 0.95);
        ctx.restore();
    }

    function drawSidechainIndicator(ctx, impactS, duration, width, height) {
        ctx.save();
        const x = (impactS / duration) * width;

        // Draw sidechain pumping indicator on Song B side (bottom half)
        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);

        // Pump wave
        ctx.beginPath();
        const waveWidth = 60;
        const waveHeight = 15;
        const centerY = height * 0.75;
        for (let i = 0; i < 5; i++) {
            const px = x + i * (waveWidth / 4);
            const py = centerY + (i % 2 === 0 ? -waveHeight : waveHeight);
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        }
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = '#fbbf24';
        ctx.font = '9px JetBrains Mono, monospace';
        ctx.textAlign = 'left';
        ctx.fillText('🎚 SIDECHAIN: Song B kick ducks Song A vocal tail', x + 4, height * 0.85);
        ctx.restore();
    }

    function drawDeckWaveform(ctx, data, startX, endX, yOffset, laneHeight, color, label) {
        if (!data || data.length === 0) return;
        const trackWidth = endX - startX;
        if (trackWidth <= 1) return;

        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.shadowColor = color;
        ctx.shadowBlur = 6;

        const centerY = yOffset + laneHeight / 2;
        const scale = laneHeight / 2 * 0.75;

        ctx.beginPath();
        const step = trackWidth / (data.length - 1);
        for (let i = 0; i < data.length; i++) {
            const x = startX + i * step;
            const y = centerY - data[i] * scale;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Lane label
        ctx.fillStyle = color;
        ctx.font = '9px JetBrains Mono, monospace';
        ctx.textAlign = 'left';
        ctx.fillText(label, startX + 8, yOffset + 14);
        ctx.restore();
    }

    function drawPlayhead(ctx) {
        if (!state.canvas) return;
        const width = state.canvas.width;
        const height = state.canvas.height;
        const progress = state.duration > 0 ? (state.currentTime / state.duration) : 0;
        const x = progress * width;

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
        ctx.font = '11px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.fillText(formatTime(state.currentTime), Math.max(30, Math.min(width - 30, x)), 20);
        ctx.restore();
    }

    // ============================================================
    // DOWNLOAD
    // ============================================================
    function handleDownload() {
        const mode = state.currentMode;
        const filename = mode === 'mix' ? `mix_${state.currentMixId || Date.now()}.wav` : `baseline_${state.currentMixId || Date.now()}.wav`;
        const url = mode === 'mix'
            ? (state.mixUrl || `/api/audio/mix_${state.currentMixId}.wav`)
            : (state.baselineUrl || `/api/audio/baseline_${state.currentMixId}.wav`);

        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    // ============================================================
    // DECK MANAGEMENT
    // ============================================================
    function swapDecks() {
        const temp = { ...state.tracks.a };
        state.tracks.a = { ...state.tracks.b };
        state.tracks.b = temp;

        const infoA = document.getElementById('track-info-a');
        const infoB = document.getElementById('track-info-b');
        const tempDisplay = infoA.style.display;
        infoA.style.display = infoB.style.display;
        infoB.style.display = tempDisplay;

        const tempHTML = infoA.innerHTML;
        infoA.innerHTML = infoB.innerHTML;
        infoB.innerHTML = tempHTML;

        const dropA = document.getElementById('drop-zone-a');
        const dropB = document.getElementById('drop-zone-b');
        const tempDropHTML = dropA.innerHTML;
        dropA.innerHTML = dropB.innerHTML;
        dropB.innerHTML = tempDropHTML;

        const hasA = dropA.classList.contains('has-file');
        const hasB = dropB.classList.contains('has-file');
        dropA.classList.toggle('has-file', hasB);
        dropB.classList.toggle('has-file', hasA);
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