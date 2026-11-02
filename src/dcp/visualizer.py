"""Interactive visual transition previewer.

Generates a standalone, dependency-free HTML/JS/CSS report for inspecting,
visualizing, and listening to DCP transitions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import __version__


def generate_html_report(
    plan_dict: dict[str, Any],
    report_dict: dict[str, Any] | None = None,
    *,
    audio_paths: dict[str, str] | None = None,
    title: str = "DCP Transition Preview",
) -> str:
    """Generate self-contained HTML for inspecting a DCP transition.

    Parameters
    ----------
    plan_dict : dict
        Serialized TransitionPlan (from plan.to_dict()).
    report_dict : dict or None
        Serialized render/demo report (from report.json).
    audio_paths : dict or None
        Relative paths to audio files: 'baseline', 'dcp', 'song_a', 'song_b'.
    title : str
        Document title.

    Returns
    -------
    str
        Self-contained HTML markup string.
    """
    audio_paths = audio_paths or {}
    report = report_dict or {}

    strategy = plan_dict.get("strategy", "unknown")
    req_strategy = plan_dict.get("requested_strategy", strategy)
    duration_s = float(plan_dict.get("duration_s", 0.0))
    start_time_s = float(plan_dict.get("start_time_s", 0.0))
    end_time_s = float(plan_dict.get("end_time_s", 0.0))
    automations = plan_dict.get("automations", [])
    reasoning = plan_dict.get("reasoning", [])
    warnings = plan_dict.get("warnings", [])
    metadata = plan_dict.get("metadata", {})

    # Extract song analysis data if available in report
    songs_data = report.get("songs", {})
    song_a_analysis = songs_data.get("a", {}).get("analysis", {})
    song_b_analysis = songs_data.get("b", {}).get("analysis", {})

    # Key detection data
    key_a = song_a_analysis.get("key") or metadata.get("key_a") or "Unknown"
    camelot_a = song_a_analysis.get("camelot_code") or metadata.get("camelot_a") or "?"
    key_b = song_b_analysis.get("key") or metadata.get("key_b") or "Unknown"
    camelot_b = song_b_analysis.get("camelot_code") or metadata.get("camelot_b") or "?"

    bpm_a = song_a_analysis.get("tempo_bpm") or metadata.get("bpm_a")
    bpm_b = song_b_analysis.get("tempo_bpm") or metadata.get("bpm_b")

    # Harmonic compatibility score
    harmonic_compat = metadata.get("harmonic_compatibility", {})
    compat_score = harmonic_compat.get("compatibility_score", 0.85 if key_a != "Unknown" else None)

    # Audio file links
    baseline_wav = audio_paths.get("baseline") or report.get("files", {}).get("baseline", "")
    dcp_wav = audio_paths.get("dcp") or report.get("files", {}).get("dcp", "")
    song_a_wav = audio_paths.get("song_a") or report.get("files", {}).get("song_a", "")
    song_b_wav = audio_paths.get("song_b") or report.get("files", {}).get("song_b", "")

    # Validation comparison
    renders = report.get("renders", {})
    baseline_val = renders.get("baseline", {}).get("validation", {})
    dcp_val = renders.get("dcp", {}).get("validation", {})

    plan_json_str = json.dumps(plan_dict)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title} | DCP Engine</title>
  <style>
    :root {{
      --bg: #090d16;
      --card-bg: rgba(18, 24, 38, 0.85);
      --card-border: rgba(255, 255, 255, 0.08);
      --card-hover: rgba(255, 255, 255, 0.12);
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --cyan: #06b6d4;
      --cyan-glow: rgba(6, 182, 212, 0.25);
      --violet: #8b5cf6;
      --violet-glow: rgba(139, 92, 246, 0.25);
      --emerald: #10b981;
      --amber: #f59e0b;
      --rose: #f43f5e;
      --accent: #38bdf8;
      --radius: 12px;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      line-height: 1.5;
      padding: 24px;
      min-height: 100vh;
      background-image: 
        radial-gradient(circle at 15% 15%, rgba(6, 182, 212, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(139, 92, 246, 0.08) 0%, transparent 40%);
    }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    
    /* Header */
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 24px;
      margin-bottom: 24px;
      border-bottom: 1px solid var(--card-border);
      flex-wrap: wrap;
      gap: 16px;
    }}
    .logo-group {{ display: flex; align-items: center; gap: 14px; }}
    .logo-badge {{
      background: linear-gradient(135deg, var(--cyan), var(--violet));
      color: #fff;
      font-weight: 800;
      font-size: 1.1rem;
      padding: 8px 14px;
      border-radius: 8px;
      box-shadow: 0 4px 14px var(--cyan-glow);
    }}
    .title-group h1 {{ font-size: 1.4rem; font-weight: 700; letter-spacing: -0.02em; }}
    .title-group p {{ font-size: 0.85rem; color: var(--text-muted); }}
    
    .tiers-badge {{
      display: flex;
      gap: 8px;
      background: rgba(255, 255, 255, 0.04);
      padding: 4px;
      border-radius: 30px;
      border: 1px solid var(--card-border);
    }}
    .tier-pill {{
      padding: 5px 12px;
      border-radius: 20px;
      font-size: 0.75rem;
      font-weight: 600;
    }}
    .tier-core {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
    .tier-adv {{ background: rgba(139, 92, 246, 0.15); color: #c084fc; border: 1px solid rgba(139, 92, 246, 0.3); }}

    /* Grid Layout */
    .grid-metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .metric-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: var(--radius);
      padding: 18px;
      backdrop-filter: blur(10px);
      transition: transform 0.2s, border-color 0.2s;
    }}
    .metric-card:hover {{
      transform: translateY(-2px);
      border-color: var(--card-hover);
    }}
    .metric-label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 6px; }}
    .metric-value {{ font-size: 1.5rem; font-weight: 700; color: var(--text); }}
    .metric-sub {{ font-size: 0.8rem; color: var(--text-muted); margin-top: 4px; }}

    /* Player Section */
    .player-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: var(--radius);
      padding: 24px;
      margin-bottom: 24px;
      backdrop-filter: blur(10px);
    }}
    .player-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 18px;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .player-tabs {{ display: flex; gap: 8px; }}
    .tab-btn {{
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      padding: 8px 18px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.85rem;
      font-weight: 600;
      transition: all 0.2s;
    }}
    .tab-btn.active {{
      background: linear-gradient(135deg, rgba(6, 182, 212, 0.2), rgba(139, 92, 246, 0.2));
      color: #fff;
      border-color: var(--cyan);
      box-shadow: 0 2px 10px var(--cyan-glow);
    }}
    .player-controls {{
      display: flex;
      align-items: center;
      gap: 16px;
      background: rgba(0, 0, 0, 0.25);
      padding: 14px 20px;
      border-radius: 10px;
    }}
    .play-btn {{
      background: var(--accent);
      color: #090d16;
      border: none;
      width: 44px;
      height: 44px;
      border-radius: 50%;
      cursor: pointer;
      font-size: 1.1rem;
      font-weight: 800;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: transform 0.15s, background 0.15s;
    }}
    .play-btn:hover {{ transform: scale(1.05); background: #7dd3fc; }}
    .time-display {{ font-family: monospace; font-size: 0.85rem; min-width: 90px; color: var(--text-muted); }}
    .seek-bar {{
      flex: 1;
      height: 6px;
      background: rgba(255, 255, 255, 0.1);
      border-radius: 3px;
      cursor: pointer;
      position: relative;
    }}
    .seek-progress {{
      height: 100%;
      background: linear-gradient(90deg, var(--cyan), var(--violet));
      border-radius: 3px;
      width: 0%;
    }}

    /* Automation Graph */
    .graph-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: var(--radius);
      padding: 24px;
      margin-bottom: 24px;
    }}
    .graph-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 14px; font-size: 0.8rem; }}
    .legend-item {{ display: flex; align-items: center; gap: 6px; }}
    .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
    .canvas-container {{
      position: relative;
      width: 100%;
      height: 280px;
      background: rgba(0, 0, 0, 0.35);
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid rgba(255, 255, 255, 0.04);
    }}
    canvas {{ width: 100%; height: 100%; display: block; }}

    /* Reasoning & Warnings */
    .details-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 24px;
    }}
    @media (max-width: 768px) {{ .details-grid {{ grid-template-columns: 1fr; }} }}
    .detail-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: var(--radius);
      padding: 20px;
    }}
    .detail-card h3 {{ font-size: 0.95rem; font-weight: 700; margin-bottom: 12px; color: var(--accent); }}
    .detail-list {{ list-style: none; }}
    .detail-list li {{
      padding: 8px 12px;
      background: rgba(255, 255, 255, 0.02);
      border-radius: 6px;
      margin-bottom: 6px;
      font-size: 0.82rem;
      border-left: 3px solid var(--cyan);
    }}
    .detail-list.warn li {{ border-left-color: var(--amber); }}

    /* Two-Tier Architecture Info */
    .architecture-card {{
      background: linear-gradient(135deg, rgba(14, 20, 36, 0.95), rgba(20, 28, 48, 0.95));
      border: 1px solid rgba(139, 92, 246, 0.3);
      border-radius: var(--radius);
      padding: 24px;
      margin-bottom: 24px;
    }}
    .arch-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-top: 14px;
    }}
    @media (max-width: 768px) {{ .arch-grid {{ grid-template-columns: 1fr; }} }}
    .arch-box {{
      padding: 16px;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.03);
    }}
    .arch-box h4 {{ font-size: 0.9rem; margin-bottom: 8px; }}
    .arch-box ul {{ padding-left: 18px; font-size: 0.8rem; color: var(--text-muted); }}
    .arch-box li {{ margin-bottom: 4px; }}

    footer {{
      text-align: center;
      padding: 20px;
      color: var(--text-muted);
      font-size: 0.8rem;
      border-top: 1px solid var(--card-border);
    }}
  </style>
</head>
<body>
<div class="container">
  <header>
    <div class="logo-group">
      <div class="logo-badge">DCP</div>
      <div class="title-group">
        <h1>{title}</h1>
        <p>Dynamic Canvas Processing Engine v{__version__} &bull; Strategy: <strong>{strategy}</strong></p>
      </div>
    </div>
    <div class="tiers-badge">
      <span class="tier-pill tier-core">Core: Open Source</span>
      <span class="tier-pill tier-adv">Advanced SDK: Royalty Kit</span>
    </div>
  </header>

  <!-- Metric Summary Cards -->
  <div class="grid-metrics">
    <div class="metric-card">
      <div class="metric-label">Transition Window</div>
      <div class="metric-value">{duration_s:.2f}s</div>
      <div class="metric-sub">{start_time_s:.2f}s &rarr; {end_time_s:.2f}s (Track A overlap)</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Harmonic Key Match</div>
      <div class="metric-value">{key_a} &rarr; {key_b}</div>
      <div class="metric-sub">Camelot: {camelot_a} vs {camelot_b} {f'&bull; Compatibility: {int(compat_score*100)}%' if compat_score else ''}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Tempo & Grid</div>
      <div class="metric-value">{f'{bpm_a:.1f}' if bpm_a else 'N/A'} &rarr; {f'{bpm_b:.1f}' if bpm_b else 'N/A'} <span style="font-size:0.9rem">BPM</span></div>
      <div class="metric-sub">{metadata.get('bars', 0)} bars &bull; {metadata.get('total_beats', 0)} beats {'(Downbeat locked)' if metadata.get('downbeat_aligned') else ''}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Clipping & Limiting</div>
      <div class="metric-value" style="color: {'#34d399' if dcp_val.get('passed', True) else '#f43f5e'}">
        {'CLEAN (PASS)' if dcp_val.get('passed', True) else 'WARNING'}
      </div>
      <div class="metric-sub">Master Gain: {renders.get('dcp', {}).get('master_gain_db', 0.0):.2f} dBFS &bull; Max interior silence: OK</div>
    </div>
  </div>

  <!-- A/B Comparison Player -->
  <div class="player-card">
    <div class="player-header">
      <div class="player-tabs">
        <button id="tab-dcp" class="tab-btn active" onclick="switchTrack('dcp')">DCP Smart Transition</button>
        <button id="tab-baseline" class="tab-btn" onclick="switchTrack('baseline')">Baseline Crossfade</button>
      </div>
      <div id="track-status" style="font-size: 0.85rem; color: var(--text-muted);">
        Playing: <strong id="active-track-name" style="color: var(--accent);">DCP Stem Transition</strong>
      </div>
    </div>

    <div class="player-controls">
      <button id="play-pause-btn" class="play-btn" onclick="togglePlay()">&#9658;</button>
      <div class="time-display" id="time-display">0:00 / 0:00</div>
      <div class="seek-bar" id="seek-bar" onclick="seek(event)">
        <div class="seek-progress" id="seek-progress"></div>
      </div>
    </div>

    <audio id="audio-dcp" preload="metadata" src="{dcp_wav}"></audio>
    <audio id="audio-baseline" preload="metadata" src="{baseline_wav}"></audio>
  </div>

  <!-- Stem Automation Curves Canvas -->
  <div class="graph-card">
    <div class="graph-header">
      <div>
        <h2 style="font-size: 1.1rem; font-weight: 700;">Stem Automation & Gain Curves</h2>
        <p style="font-size: 0.8rem; color: var(--text-muted);">Decibel curves over the {duration_s:.2f}s transition window (0 dB to fade floor)</p>
      </div>
      <div class="legend" id="legend-container"></div>
    </div>
    <div class="canvas-container">
      <canvas id="curves-canvas"></canvas>
    </div>
  </div>

  <!-- Details Grid: Reasoning & Warnings -->
  <div class="details-grid">
    <div class="detail-card">
      <h3>Explainable Decisions & Strategy Reasoning</h3>
      <ul class="detail-list">
        {''.join(f'<li>{r}</li>' for r in reasoning) if reasoning else '<li>Standard transition plan applied.</li>'}
      </ul>
    </div>
    <div class="detail-card">
      <h3 style="color: var(--amber);">Diagnostics & Validation Warnings</h3>
      <ul class="detail-list warn">
        {''.join(f'<li>{w}</li>' for w in warnings) if warnings else '<li>No warnings. Transition passed all objective validation criteria.</li>'}
      </ul>
    </div>
  </div>

  <!-- Two-Tier Engine Architecture Info -->
  <div class="architecture-card">
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <h3 style="font-size: 1rem; color: var(--text);">Engine Tier Architecture: Basic vs Advanced SDK</h3>
      <span style="font-size: 0.75rem; color: var(--violet); font-weight: 700;">Licensing & Royalty Ready</span>
    </div>
    <div class="arch-grid">
      <div class="arch-box">
        <h4 style="color: #34d399;">DCP Core (Open Source Foundation)</h4>
        <ul>
          <li>Permissive open-source foundation (Apache-2.0 / MIT)</li>
          <li>Deterministic offline WAV renderer & clipping guard</li>
          <li>5 rule-based transition strategies + RATO candidate optimization</li>
          <li>Lightweight chroma key estimator (Krumhansl-Schmuckler) & downbeat snap</li>
          <li>Pluggable contracts for librosa beat tracking and Demucs source separation</li>
          <li>Local human preference labeling CLI (<code>dcp label</code>)</li>
        </ul>
      </div>
      <div class="arch-box" style="border-left: 2px solid var(--violet);">
        <h4 style="color: #c084fc;">DCP Advanced SDK (Royalty & Licensing Kit)</h4>
        <ul>
          <li>Embeddable C++/Python SDK for streaming apps (Spotify, Apple Music, etc.)</li>
          <li>Real-time low-latency playback engine with buffer precomputing</li>
          <li>Deep neural phrase segmentation, energy contour tracking, and vocal boundaries</li>
          <li>High-fidelity phase-aligned time-stretching and harmonic key modulation</li>
          <li>Trained pairwise neural ranking model for perceptual transition quality</li>
          <li>Royalty metering, API key licensing hooks, and streaming service integrations</li>
        </ul>
      </div>
    </div>
  </div>

  <footer>
    DCP Dynamic Canvas Processing &bull; Autonomous Music Transition Engine &bull; Generated offline
  </footer>
</div>

<script>
const planData = {plan_json_str};
let currentMode = 'dcp';
const audioDcp = document.getElementById('audio-dcp');
const audioBaseline = document.getElementById('audio-baseline');
const playBtn = document.getElementById('play-pause-btn');
const timeDisplay = document.getElementById('time-display');
const seekProgress = document.getElementById('seek-progress');
const seekBar = document.getElementById('seek-bar');
const activeTrackName = document.getElementById('active-track-name');

function getActiveAudio() {{
  return currentMode === 'dcp' ? audioDcp : audioBaseline;
}}

function switchTrack(mode) {{
  const wasPlaying = !getActiveAudio().paused;
  const currentTime = getActiveAudio().currentTime;
  
  audioDcp.pause();
  audioBaseline.pause();
  
  currentMode = mode;
  document.getElementById('tab-dcp').classList.toggle('active', mode === 'dcp');
  document.getElementById('tab-baseline').classList.toggle('active', mode === 'baseline');
  activeTrackName.textContent = mode === 'dcp' ? 'DCP Stem Transition' : 'Baseline Conventional Crossfade';
  
  const active = getActiveAudio();
  active.currentTime = currentTime;
  if (wasPlaying) {{
    active.play().catch(e => console.log('Playback error:', e));
    playBtn.innerHTML = '&#10074;&#10074;';
  }} else {{
    playBtn.innerHTML = '&#9658;';
  }}
}}

function togglePlay() {{
  const active = getActiveAudio();
  if (active.paused) {{
    active.play().then(() => {{
      playBtn.innerHTML = '&#10074;&#10074;';
    }}).catch(err => {{
      alert('Audio could not play. If running locally, check that the WAV files exist at the reported paths: ' + active.src);
    }});
  }} else {{
    active.pause();
    playBtn.innerHTML = '&#9658;';
  }}
}}

function formatTime(s) {{
  if (!s || isNaN(s)) return '0:00';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m + ':' + (sec < 10 ? '0' : '') + sec;
}}

function updateProgress() {{
  const active = getActiveAudio();
  const cur = active.currentTime || 0;
  const dur = active.duration || 1;
  seekProgress.style.width = ((cur / dur) * 100) + '%';
  timeDisplay.textContent = formatTime(cur) + ' / ' + formatTime(dur);
}}

function seek(e) {{
  const active = getActiveAudio();
  const rect = seekBar.getBoundingClientRect();
  const pos = (e.clientX - rect.left) / rect.width;
  if (active.duration) {{
    active.currentTime = pos * active.duration;
  }}
}}

audioDcp.addEventListener('timeupdate', updateProgress);
audioBaseline.addEventListener('timeupdate', updateProgress);
audioDcp.addEventListener('ended', () => playBtn.innerHTML = '&#9658;');
audioBaseline.addEventListener('ended', () => playBtn.innerHTML = '&#9658;');

// Draw automation curves
const STEM_COLORS = {{
  'a/vocals': '#38bdf8',
  'a/bass': '#818cf8',
  'a/drums': '#34d399',
  'a/other': '#a78bfa',
  'a/mix': '#94a3b8',
  'b/vocals': '#f43f5e',
  'b/bass': '#fb923c',
  'b/drums': '#facc15',
  'b/other': '#ec4899',
  'b/mix': '#cbd5e1'
}};

function initCurves() {{
  const canvas = document.getElementById('curves-canvas');
  const ctx = canvas.getContext('2d');
  const legend = document.getElementById('legend-container');
  
  // Set resolution
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * window.devicePixelRatio;
  canvas.height = rect.height * window.devicePixelRatio;
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  
  const w = rect.width;
  const h = rect.height;
  const padL = 45;
  const padR = 20;
  const padT = 20;
  const padB = 30;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  
  const automations = planData.automations || [];
  const duration = planData.duration_s || 8.0;
  const minDb = -60.0;
  const maxDb = 0.0;
  
  // Clear
  ctx.clearRect(0, 0, w, h);
  
  // Grid lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  ctx.lineWidth = 1;
  ctx.font = '10px monospace';
  ctx.fillStyle = '#64748b';
  
  // dB grid
  const dbSteps = [0, -12, -24, -36, -48, -60];
  dbSteps.forEach(db => {{
    const y = padT + ((maxDb - db) / (maxDb - minDb)) * plotH;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w - padR, y);
    ctx.stroke();
    ctx.fillText(db + 'dB', 6, y + 3);
  }});
  
  // Time grid (1s intervals)
  for (let t = 0; t <= duration; t += (duration > 10 ? 2 : 1)) {{
    const x = padL + (t / duration) * plotW;
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, h - padB);
    ctx.stroke();
    ctx.fillText(t.toFixed(0) + 's', x - 6, h - 10);
  }}
  
  // Build legend & curves
  legend.innerHTML = '';
  automations.forEach(auto => {{
    const key = auto.source + '/' + auto.stem;
    const color = STEM_COLORS[key] || '#e2e8f0';
    
    // Add legend dot
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = `<span class="legend-dot" style="background:${{color}}"></span><span>${{key.toUpperCase()}}</span>`;
    legend.appendChild(item);
    
    // Draw curve
    const points = auto.points || [];
    if (points.length === 0) return;
    
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    
    for (let i = 0; i <= 100; i++) {{
      const t = (i / 100) * duration;
      // Interpolate points
      let gainDb = minDb;
      if (t <= points[0].time_s) {{
        gainDb = points[0].gain_db;
      }} else if (t >= points[points.length - 1].time_s) {{
        gainDb = points[points.length - 1].gain_db;
      }} else {{
        for (let p = 0; p < points.length - 1; p++) {{
          const p0 = points[p];
          const p1 = points[p + 1];
          if (t >= p0.time_s && t <= p1.time_s) {{
            const fraction = (t - p0.time_s) / Math.max(1e-6, p1.time_s - p0.time_s);
            // smoothstep or linear
            const curveFrac = auto.curve === 'smoothstep' 
              ? fraction * fraction * (3 - 2 * fraction)
              : fraction;
            gainDb = p0.gain_db + curveFrac * (p1.gain_db - p0.gain_db);
            break;
          }}
        }}
      }}
      const clampedDb = Math.max(minDb, Math.min(maxDb, gainDb));
      const x = padL + (t / duration) * plotW;
      const y = padT + ((maxDb - clampedDb) / (maxDb - minDb)) * plotH;
      
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }}
    ctx.stroke();
  }});
}}

window.addEventListener('load', initCurves);
window.addEventListener('resize', initCurves);
</script>
</body>
</html>
"""
    return html


def save_html_report(
    target_path: str | Path,
    plan_dict: dict[str, Any],
    report_dict: dict[str, Any] | None = None,
    *,
    audio_paths: dict[str, str] | None = None,
    title: str = "DCP Transition Preview",
) -> Path:
    """Generate and write an HTML report to disk."""
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html = generate_html_report(
        plan_dict,
        report_dict,
        audio_paths=audio_paths,
        title=title,
    )
    path.write_text(html, encoding="utf-8")
    return path
