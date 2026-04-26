// ─── Mumbai Local OpenEnv — Enhanced Dashboard v2 ────────────────────────────

// ── Theme Toggle ──────────────────────────────────────────────────────────────
function getThemeColors() {
  const isLight = document.documentElement.classList.contains('light');
  return {
    txt: isLight ? '#4a5568' : '#3d4559',
    grid: isLight ? '#d0d8e0' : '#161b25'
  };
}

function toggleTheme() {
  const root = document.documentElement;
  const btn = document.getElementById('themeToggle');
  const isLight = root.classList.toggle('light');
  btn.textContent = isLight ? '☀️' : '🌙';
  localStorage.setItem('mlTheme', isLight ? 'light' : 'dark');
  if (typeof drawNetwork === 'function') drawNetwork();
  updateChartsTheme();
}

function updateChartsTheme() {
  const colors = getThemeColors();
  Object.values(charts).forEach(chart => {
    chart.options.scales.x.ticks.color = colors.txt;
    chart.options.scales.x.grid.color = colors.grid;
    chart.options.scales.y.ticks.color = colors.txt;
    chart.options.scales.y.grid.color = colors.grid;
    chart.options.scales.y.title.color = colors.txt;
    chart.update('none');
  });
}

(function () {
  if (localStorage.getItem('mlTheme') === 'light') {
    document.documentElement.classList.add('light');
    document.addEventListener('DOMContentLoaded', () => {
      const btn = document.getElementById('themeToggle');
      if (btn) btn.textContent = '☀️';
    });
  }
})();


const LINE_META = {
  Western: {
    color: '#FF6B35', stations: [
      "Churchgate", "Marine Lines", "Charni Road", "Grant Road", "Mumbai Central",
      "Mahalaxmi", "Lower Parel", "Elphinstone Road", "Dadar", "Matunga Road",
      "Mahim", "Bandra", "Khar Road", "Santacruz", "Vile Parle", "Andheri",
      "Jogeshwari", "Ram Mandir", "Goregaon", "Malad", "Kandivali", "Borivali",
      "Dahisar", "Mira Road", "Bhayandar", "Naigaon", "Vasai Road", "Nalasopara", "Virar"
    ]
  },
  Central: {
    color: '#4ECDC4', stations: [
      "CSMT", "Masjid", "Sandhurst Road", "Byculla", "Chinchpokli", "Currey Road",
      "Parel", "Dadar", "Matunga", "Sion", "Kurla", "Vidyavihar", "Ghatkopar",
      "Vikhroli", "Kanjurmarg", "Bhandup", "Nahur", "Mulund", "Thane", "Kalwa",
      "Mumbra", "Diva", "Kopar", "Dombivli", "Thakurli", "Kalyan"
    ]
  },
  Harbour: {
    color: '#A855F7', stations: [
      "CSMT", "Masjid", "Sandhurst Road", "Dockyard Road", "Reay Road", "Cotton Green",
      "Sewri", "Wadala Road", "King's Circle", "Mahim Junction", "Bandra", "Khar Road",
      "Santacruz", "Vile Parle", "Andheri", "Chembur", "Govandi", "Mankhurd",
      "Vashi", "Sanpada", "Juinagar", "Nerul", "Seawoods", "Belapur", "Kharghar", "Panvel"
    ]
  }
};

// ─── State ─────────────────────────────────────────────────────────────────
let state = null, isAuto = false, charts = {};
let trainSmooth = {}, animFrame = null, lastFetch = 0;
let compareMode = false, cmpChart = null;
let selectedSeverity = 'Medium';
let timeSliderDebounce = null;

// ─── Clock ─────────────────────────────────────────────────────────────────
function tickClock() {
  document.getElementById('clock').textContent =
    new Date().toLocaleTimeString('en-IN', { hour12: false });
}
setInterval(tickClock, 1000); tickClock();

// ─── Chart factory ──────────────────────────────────────────────────────────
function makeChart(id, color, yLabel) {
  const colors = getThemeColors();
  const ctx = document.getElementById(id).getContext('2d');
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels: [], datasets: [{
        data: [], borderColor: color,
        backgroundColor: color + '15', borderWidth: 2, fill: true,
        tension: 0.4, pointRadius: 0
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: { color: colors.txt, font: { family: 'Space Mono', size: 8 }, maxTicksLimit: 8 },
          grid: { color: colors.grid }
        },
        y: {
          ticks: { color: colors.txt, font: { family: 'Space Mono', size: 8 }, maxTicksLimit: 5 },
          grid: { color: colors.grid },
          title: { display: true, text: yLabel, color: colors.txt, font: { family: 'Space Mono', size: 8 } }
        }
      }
    }
  });
}

function initCharts() {
  charts.reward = makeChart('rewardChart', '#FF6B35', 'Cum. Reward');
  charts.loss   = makeChart('lossChart',   '#eab308', 'Loss');
  charts.step   = makeChart('stepChart',   '#4ECDC4', 'Step Reward');
  charts.arr    = makeChart('arrChart',    '#A855F7', '# Arrived');
}

function syncChartsFromHistory(h, steps) {
  if (!h || !steps) return;
  const N = h.rewards.length; if (N === 0) return;
  const labels = Array.from({ length: N }, (_, i) => String(steps - N + i));
  charts.reward.data.labels = labels; charts.reward.data.datasets[0].data = h.rewards;
  charts.loss.data.labels   = labels; charts.loss.data.datasets[0].data   = h.losses;
  charts.step.data.labels   = labels; charts.step.data.datasets[0].data   = h.step_r;
  charts.arr.data.labels    = labels; charts.arr.data.datasets[0].data    = h.arrivals;
  charts.reward.update('none'); charts.loss.update('none');
  charts.step.update('none');  charts.arr.update('none');
}

// ─── Network Canvas ───────────────────────────────────────────────────────────
const canvas = document.getElementById('netCanvas');
const ctx2d   = canvas.getContext('2d');
const LINE_LAYOUT = {};
let stationHitTargets = [];
let hoveredStation = null;

const TRANSFER_NODES = new Set(['Dadar','CSMT','Andheri','Bandra','Kurla','Mahim','Mahim Junction']);

function labelEvery(N) { return Math.max(1, Math.ceil(N / 7)); }

function computeLayout() {
  const W = canvas.offsetWidth, H = canvas.offsetHeight;
  canvas.width  = W * devicePixelRatio; canvas.height = H * devicePixelRatio;
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  ctx2d.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);

  const padL = 14, padR = 14;
  const yPos = { Western: H * 0.18, Central: H * 0.48, Harbour: H * 0.76 };
  stationHitTargets = [];
  for (const [lname, lmeta] of Object.entries(LINE_META)) {
    const N    = lmeta.stations.length;
    const step = (W - padL - padR) / (N - 1);
    LINE_LAYOUT[lname] = {
      color: lmeta.color, y: yPos[lname],
      xs: lmeta.stations.map((_, i) => padL + i * step),
      stations: lmeta.stations, step
    };
    lmeta.stations.forEach((st, i) => {
      stationHitTargets.push({ lname, st, x: padL + i * step, y: yPos[lname] });
    });
  }
}

function hitStation(mx, my, threshold = 16) {
  let best = null, bestD = threshold;
  for (const t of stationHitTargets) {
    const d = Math.hypot(mx - t.x, my - t.y);
    if (d < bestD) { bestD = d; best = t; }
  }
  return best;
}

canvas.addEventListener('click', e => {
  const rect = canvas.getBoundingClientRect();
  const t = hitStation(e.clientX - rect.left, e.clientY - rect.top);
  if (t) openStationPopover(t.st, t.x, t.y);
});

canvas.addEventListener('mousemove', e => {
  const rect = canvas.getBoundingClientRect();
  const t = hitStation(e.clientX - rect.left, e.clientY - rect.top, 18);
  const prev = hoveredStation;
  hoveredStation = t ? t.st : null;
  canvas.style.cursor = t ? 'pointer' : 'default';
});

canvas.addEventListener('mouseleave', () => {
  hoveredStation = null;
  canvas.style.cursor = 'default';
});

function stationXY(lineName, fracIdx) {
  const ll = LINE_LAYOUT[lineName]; if (!ll) return { x: 0, y: 0 };
  const i0 = Math.floor(fracIdx), i1 = Math.min(i0 + 1, ll.xs.length - 1);
  const t = fracIdx - i0;
  return { x: ll.xs[i0] * (1 - t) + ll.xs[i1] * t, y: ll.y };
}

function crowdColor(cv) {
  if (cv > 80) return '#ef4444';
  if (cv > 60) return '#f97316';
  if (cv > 40) return '#eab308';
  return '#22c55e';
}

function buildAgentIndex() {
  const idx = {};
  if (!state || !state.agents) return idx;
  for (const ag of state.agents) {
    if (ag.arrived) continue;
    const st = ag.current_station;
    if (!idx[st]) idx[st] = [];
    idx[st].push(ag);
  }
  return idx;
}

function buildDisruptSet() {
  const s = new Set();
  if (state && state.disruptions) state.disruptions.forEach(d => s.add(d.station));
  return s;
}

function drawFrame() {
  const W = canvas.offsetWidth, H = canvas.offsetHeight;
  ctx2d.clearRect(0, 0, W, H);
  if (!W || !H) { animFrame = requestAnimationFrame(drawFrame); return; }

  const crowd     = state ? state.crowd : {};
  const agentIdx  = buildAgentIndex();
  const disruptSt = buildDisruptSet();
  const isDark    = !document.documentElement.classList.contains('light');
  const labelCol  = isDark ? '#8b90a0' : '#4a5568';
  const termCol   = isDark ? '#e8eaf0' : '#1a202c';
  const bgCol     = isDark ? '#0d1117'  : '#f7f8fa';

  // ── Draw each line ──────────────────────────────────────────────────────────
  for (const [lname, ll] of Object.entries(LINE_LAYOUT)) {
    const { color, y, xs, stations } = ll;
    const N     = stations.length;
    const every = labelEvery(N);

    // Shadow track
    ctx2d.beginPath();
    ctx2d.strokeStyle = color + '18';
    ctx2d.lineWidth = 10;
    xs.forEach((x, i) => i === 0 ? ctx2d.moveTo(x, y) : ctx2d.lineTo(x, y));
    ctx2d.stroke();

    // Main track
    ctx2d.beginPath();
    ctx2d.strokeStyle = color + '70';
    ctx2d.lineWidth = 4;
    xs.forEach((x, i) => i === 0 ? ctx2d.moveTo(x, y) : ctx2d.lineTo(x, y));
    ctx2d.stroke();

    // Line label
    ctx2d.save();
    ctx2d.font = 'bold 9px "Space Mono", monospace';
    ctx2d.fillStyle = color;
    ctx2d.textAlign = 'left';
    ctx2d.fillText(lname.toUpperCase(), xs[0] + 2, y + 22);
    ctx2d.restore();

    // Per-station markers
    stations.forEach((st, i) => {
      const x          = xs[i];
      const cv         = crowd[st] || 30;
      const isHovered  = hoveredStation === st;
      const isTransfer = TRANSFER_NODES.has(st);
      const isTerminus = (i === 0 || i === N - 1);
      const hasAgents  = !!agentIdx[st];
      const hasDisrupt = disruptSt.has(st);
      const showLabel  = isTerminus || isTransfer || isHovered || (i % every === 0);

      // Tick line
      ctx2d.beginPath();
      ctx2d.strokeStyle = color + '55';
      ctx2d.lineWidth = isTransfer || isTerminus ? 2 : 1;
      ctx2d.moveTo(x, y - 5); ctx2d.lineTo(x, y + 5);
      ctx2d.stroke();

      // Crowd heat bar
      const barH = 4, barW = Math.max(3, ll.step * 0.55);
      const barX = x - barW / 2, barY = y + 6;
      ctx2d.fillStyle = '#1a1d24';
      ctx2d.fillRect(barX, barY, barW, barH);
      ctx2d.fillStyle = crowdColor(cv);
      ctx2d.fillRect(barX, barY, barW * (cv / 100), barH);

      // Station marker
      if (isTransfer) {
        const d = isHovered ? 9 : 7;
        ctx2d.save();
        ctx2d.translate(x, y);
        ctx2d.rotate(Math.PI / 4);
        ctx2d.beginPath();
        ctx2d.rect(-d / 2, -d / 2, d, d);
        ctx2d.fillStyle = bgCol;
        ctx2d.fill();
        ctx2d.strokeStyle = color;
        ctx2d.lineWidth = isHovered ? 2.5 : 2;
        ctx2d.stroke();
        ctx2d.restore();
        ctx2d.beginPath();
        ctx2d.arc(x, y, isHovered ? 3 : 2, 0, Math.PI * 2);
        ctx2d.fillStyle = color;
        ctx2d.fill();
      } else if (isTerminus) {
        const r = isHovered ? 7 : 6;
        ctx2d.beginPath(); ctx2d.arc(x, y, r, 0, Math.PI * 2);
        ctx2d.fillStyle = color; ctx2d.fill();
        ctx2d.beginPath(); ctx2d.arc(x, y, r - 2, 0, Math.PI * 2);
        ctx2d.fillStyle = bgCol; ctx2d.fill();
      } else {
        const r = isHovered ? 5 : 4;
        ctx2d.beginPath(); ctx2d.arc(x, y, r, 0, Math.PI * 2);
        ctx2d.fillStyle = bgCol; ctx2d.fill();
        ctx2d.strokeStyle = crowdColor(cv);
        ctx2d.lineWidth = isHovered ? 2 : 1.5;
        ctx2d.stroke();
      }

      // Disruption flash ring
      if (hasDisrupt) {
        const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 250);
        ctx2d.beginPath();
        ctx2d.arc(x, y, 9 + pulse * 3, 0, Math.PI * 2);
        ctx2d.strokeStyle = `rgba(239,68,68,${0.55 + pulse * 0.35})`;
        ctx2d.lineWidth = 1.5;
        ctx2d.stroke();
      }

      // Agent avatars
      if (hasAgents) {
        const ags = agentIdx[st];
        const MAX_SHOW = 3;
        const avatarR  = 5;
        const spacing  = avatarR * 2 + 2;
        const totalW   = Math.min(ags.length, MAX_SHOW) * spacing;
        const startX   = x - totalW / 2 + avatarR;
        const avatarY  = y - 20;

        ags.slice(0, MAX_SHOW).forEach((ag, ai) => {
          const ax    = startX + ai * spacing;
          const agCol = ag.line === 'Western' ? '#FF6B35' : ag.line === 'Central' ? '#4ECDC4' : '#A855F7';
          ctx2d.beginPath();
          ctx2d.arc(ax, avatarY, avatarR, 0, Math.PI * 2);
          ctx2d.fillStyle = agCol + 'dd'; ctx2d.fill();
          ctx2d.strokeStyle = agCol; ctx2d.lineWidth = 1; ctx2d.stroke();
          ctx2d.fillStyle = '#fff';
          ctx2d.font = 'bold 5px sans-serif';
          ctx2d.textAlign = 'center';
          ctx2d.fillText(ag.name[0], ax, avatarY + 2);
        });
        if (ags.length > MAX_SHOW) {
          const ox = startX + MAX_SHOW * spacing;
          ctx2d.fillStyle = labelCol;
          ctx2d.font = '7px "Space Mono"';
          ctx2d.textAlign = 'left';
          ctx2d.fillText(`+${ags.length - MAX_SHOW}`, ox, avatarY + 3);
        }
      }

      // Station label
      if (showLabel) {
        const labelY = y - (hasAgents ? 34 : 13);
        ctx2d.save();
        ctx2d.translate(x, labelY);
        ctx2d.rotate(-Math.PI / 5);
        ctx2d.font = isHovered
          ? 'bold 8.5px "Space Mono", monospace'
          : isTransfer || isTerminus
            ? 'bold 8px "Space Mono", monospace'
            : '7.5px "Space Mono", monospace';
        ctx2d.fillStyle = isHovered ? '#fff' : isTerminus ? termCol : isTransfer ? color : labelCol;
        ctx2d.textAlign = 'left';
        ctx2d.fillText(st.length > 11 ? st.substring(0, 10) + '…' : st, 0, 0);
        ctx2d.restore();

        if (isHovered) {
          const cv2 = crowd[st] || 0;
          ctx2d.font = 'bold 8px "Space Mono"';
          ctx2d.fillStyle = crowdColor(cv2);
          ctx2d.textAlign = 'center';
          ctx2d.fillText(`${cv2}%`, x, y + 18);
        }
      }

      // Transfer interchange symbol
      if (isTransfer && !isHovered) {
        ctx2d.font = '7px sans-serif';
        ctx2d.fillStyle = '#fff';
        ctx2d.textAlign = 'center';
        ctx2d.fillText('⇄', x, y + 18);
      }
    });
  }

  // ── Train sprites ──────────────────────────────────────────────────────────
  for (const [, ts] of Object.entries(trainSmooth)) {
    ts.x += (ts.tx - ts.x) * 0.12;
    ts.y += (ts.ty - ts.y) * 0.12;
    const col = ts.delayed ? '#ef4444' : ts.color;

    ctx2d.beginPath(); ctx2d.arc(ts.x, ts.y, 11, 0, Math.PI * 2);
    ctx2d.fillStyle = col + '22'; ctx2d.fill();
    ctx2d.beginPath(); ctx2d.arc(ts.x, ts.y, 6, 0, Math.PI * 2);
    ctx2d.fillStyle = col; ctx2d.fill();
    ctx2d.strokeStyle = '#fff3'; ctx2d.lineWidth = 1; ctx2d.stroke();
    ctx2d.fillStyle = '#fff';
    ctx2d.font = 'bold 5.5px "Space Mono"';
    ctx2d.textAlign = 'center';
    ctx2d.fillText(ts.id, ts.x, ts.y + 2);
    if (ts.delayed) {
      ctx2d.fillStyle = '#ef4444';
      ctx2d.font = '8px sans-serif';
      ctx2d.fillText('⚠', ts.x + 7, ts.y - 5);
    }
  }

  // ── Node-type legend: single row below Harbour line ───────────────────────
  const harbourY  = LINE_LAYOUT['Harbour'] ? LINE_LAYOUT['Harbour'].y : H * 0.76;
  const legY      = harbourY + 32;
  const legItems  = [
    { label: 'Transfer node', shape: 'diamond', color: '#a855f7' },
    { label: 'Terminus',      shape: 'square',  color: '#4ECDC4' },
    { label: 'Station',       shape: 'circle',  color: '#7c859a' },
    { label: 'Agent',         shape: 'circle',  color: '#FF6B35' },
  ];
  const itemW  = 105, totalLegW = legItems.length * itemW;
  const legStartX = (W - totalLegW) / 2;

  ctx2d.fillStyle = 'rgba(13,17,23,0.78)';
  ctx2d.beginPath();
  ctx2d.roundRect(legStartX - 8, legY - 10, totalLegW + 16, 22, 5);
  ctx2d.fill();

  legItems.forEach(({ label, shape, color }, i) => {
    const ix = legStartX + i * itemW + 10;
    ctx2d.fillStyle = color;
    if (shape === 'diamond') {
      const s = 4;
      ctx2d.beginPath();
      ctx2d.moveTo(ix, legY - s); ctx2d.lineTo(ix + s, legY);
      ctx2d.lineTo(ix, legY + s); ctx2d.lineTo(ix - s, legY);
      ctx2d.closePath(); ctx2d.fill();
    } else if (shape === 'square') {
      ctx2d.fillRect(ix - 4, legY - 4, 8, 8);
    } else {
      ctx2d.beginPath(); ctx2d.arc(ix, legY, 4, 0, Math.PI * 2); ctx2d.fill();
    }
    ctx2d.fillStyle = '#c4cad8';
    ctx2d.font = '8px "Space Mono"';
    ctx2d.textAlign = 'left';
    ctx2d.fillText(label, ix + 9, legY + 3);
  });

  animFrame = requestAnimationFrame(drawFrame);
}

// ─── Train smooth updater ─────────────────────────────────────────────────────
function updateTrainSmooth(trains) {
  if (!trains) return;
  trains.forEach(t => {
    const ll = LINE_LAYOUT[t.line]; if (!ll) return;
    const fracIdx = parseFloat(t.frac ?? t.pos ?? 0);
    const { x, y } = stationXY(t.line, fracIdx);
    if (!trainSmooth[t.id]) {
      trainSmooth[t.id] = { x, y, tx: x, ty: y, color: ll.color, id: t.id, delayed: t.delayed, occ: t.occupancy };
    } else {
      trainSmooth[t.id].tx = x; trainSmooth[t.id].ty = y;
      trainSmooth[t.id].delayed = t.delayed; trainSmooth[t.id].occ = t.occupancy;
    }
  });
}

// ─── Station Popover ─────────────────────────────────────────────────────────
async function openStationPopover(stName, cx, cy) {
  const pop = document.getElementById('stationPopover');
  document.getElementById('popTitle').textContent = stName;
  document.getElementById('popContent').innerHTML = '<div class="pop-loading">Loading…</div>';
  pop.classList.add('show');
  try {
    const r = await fetch(`/api/station/${encodeURIComponent(stName)}`);
    const d = await r.json();
    const crowdCol = d.crowd > 75 ? '#ef4444' : d.crowd > 50 ? '#f97316' : d.crowd > 25 ? '#eab308' : '#22c55e';
    let html = `
      <div class="pop-crowd"><span style="color:${crowdCol};font-size:22px;font-family:'Bebas Neue'">${d.crowd}%</span><span class="pop-lbl"> crowd</span></div>
      <div class="pop-section"><div class="pop-sec-hd">👤 Agents Here (${d.agents_here.length})</div>`;
    if (d.agents_here.length) {
      html += d.agents_here.map(a =>
        `<div class="pop-row"><span>${a.name}</span><span style="color:${a.line === 'Western' ? '#FF6B35' : a.line === 'Central' ? '#4ECDC4' : '#A855F7'}">${a.line}</span><span>→${a.dest}</span></div>`
      ).join('');
    } else html += '<div class="pop-empty">No agents</div>';
    html += `</div><div class="pop-section"><div class="pop-sec-hd">🚂 Trains At Station</div>`;
    if (d.trains_at_station.length) {
      html += d.trains_at_station.map(t =>
        `<div class="pop-row"><span>${t.id}</span><span>${t.occ}% full</span>${t.delayed ? '<span style="color:#ef4444">⚠ delayed</span>' : ''}</div>`
      ).join('');
    } else html += '<div class="pop-empty">None</div>';
    html += `</div><div class="pop-section"><div class="pop-sec-hd">🔜 Arriving Soon</div>`;
    if (d.arriving_soon.length) {
      html += d.arriving_soon.map(t =>
        `<div class="pop-row"><span>${t.id}</span><span style="color:#4ECDC4">~${t.eta_min}min</span>${t.delayed ? '<span style="color:#ef4444">delayed</span>' : ''}</div>`
      ).join('');
    } else html += '<div class="pop-empty">No incoming</div>';
    html += '</div>';
    if (d.disruptions.length) {
      html += `<div class="pop-section"><div class="pop-sec-hd" style="color:#ef4444">⚡ Active Disruptions</div>`;
      html += d.disruptions.map(x => `<div class="pop-row" style="color:#ef4444"><span>${x.type}</span><span>${x.severity}</span></div>`).join('');
      html += '</div>';
    }
    document.getElementById('popContent').innerHTML = html;
  } catch (e) {
    document.getElementById('popContent').innerHTML = '<div class="pop-empty">Error loading</div>';
  }
}

function closePopover() {
  document.getElementById('stationPopover').classList.remove('show');
}

// ─── Agent Journey Timeline ───────────────────────────────────────────────────
function openJourney(agentIdx) {
  if (!state || !state.agents) return;
  const ag = state.agents[agentIdx];
  if (!ag) return;
  document.getElementById('journeyTitle').textContent = `${ag.name} — ${ag.origin} → ${ag.destination}`;
  const lineCol = ag.line === 'Western' ? '#FF6B35' : ag.line === 'Central' ? '#4ECDC4' : '#A855F7';
  const journey = ag.journey || [];
  let html = '<div class="timeline">';
  journey.forEach((step, i) => {
    const isLast    = i === journey.length - 1;
    const isArrived = step.action === 'arrived';
    const icon      = isArrived ? '✅' : step.action === 'departed' || step.action === 'new_episode' ? '🚉' : '📍';
    const rewColor  = step.reward >= 0 ? '#4ECDC4' : '#ef4444';
    html += `
      <div class="tl-item ${isArrived ? 'arrived' : ''}">
        <div class="tl-dot" style="background:${isArrived ? '#22c55e' : lineCol}">${icon}</div>
        <div class="tl-line" ${isLast ? 'style="display:none"' : ''}></div>
        <div class="tl-content">
          <div class="tl-station">${step.station}</div>
          <div class="tl-meta">
            <span class="tl-step">Step ${step.step}</span>
            <span class="tl-action">${step.action.replace('_', ' ')}</span>
            <span class="tl-reward" style="color:${rewColor}">${step.reward >= 0 ? '+' : ''}${step.reward.toFixed ? step.reward.toFixed(3) : step.reward}</span>
          </div>
        </div>
      </div>`;
  });
  html += '</div>';
  document.getElementById('journeyTimeline').innerHTML = html;
  document.getElementById('journeyModal').classList.add('show');
}

function closeJourneyModal() {
  document.getElementById('journeyModal').classList.remove('show');
}

// ─── Disruption Scenario Builder ──────────────────────────────────────────────
function openDisruptBuilder() { document.getElementById('disruptModal').classList.add('show'); }
function closeDisruptModal()  { document.getElementById('disruptModal').classList.remove('show'); }
function setSev(btn) {
  document.querySelectorAll('.sev-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  selectedSeverity = btn.dataset.sev;
}
function updateDisruptStations() {
  const line = document.getElementById('disLine').value;
  const sel  = document.getElementById('disStn');
  sel.innerHTML = '<option value="">Random</option>';
  if (line && LINE_META[line]) {
    LINE_META[line].stations.forEach(s => {
      const o = document.createElement('option'); o.value = s; o.textContent = s; sel.appendChild(o);
    });
  }
}
async function triggerCustomDisrupt() {
  const line = document.getElementById('disLine').value || null;
  const stn  = document.getElementById('disStn').value  || null;
  const type = document.getElementById('disType').value || null;
  try {
    const r = await fetch('/api/disrupt', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ line, station: stn, type, severity: selectedSeverity })
    });
    const s = await r.json();
    applyState(s);
    toast(`⚡ ${s.triggered?.type || 'Disruption'} on ${s.triggered?.line} — ${s.triggered?.station}`, false);
    closeDisruptModal();
  } catch (e) { }
}

// ─── NL Command Bar ───────────────────────────────────────────────────────────
async function sendNLCommand() {
  const inp  = document.getElementById('nlInput');
  const text = inp.value.trim();
  if (!text) return;
  const resultEl = document.getElementById('nlResult');
  resultEl.innerHTML = '<span class="nl-thinking">⏳ Interpreting…</span>';
  try {
    const r = await fetch('/api/command', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const s = await r.json();
    applyState(s);
    const rw = s.last_reward;
    resultEl.innerHTML = `<span class="nl-interp">→ <b>${s.interpreted_action.replace('_', ' ')}</b> ${s.matched_keyword ? `(matched: "${s.matched_keyword}")` : ''}</span><span class="nl-rew" style="color:${rw >= 0 ? '#4ECDC4' : '#ef4444'}">${rw >= 0 ? '+' : ''}${rw.toFixed(3)}</span>`;
    toast(`AI: "${text}" → ${s.interpreted_action.replace('_', ' ')}`, rw >= 0);
    inp.value = '';
  } catch (e) {
    resultEl.innerHTML = '<span style="color:#ef4444">Error</span>';
  }
}
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('nlInput').addEventListener('keydown', e => {
    if (e.key === 'Enter') sendNLCommand();
  });
});

// ─── Rush Hour Time Slider ────────────────────────────────────────────────────
function onTimeSlider(val) {
  const h    = parseInt(val);
  const disp = `${String(h).padStart(2, '0')}:00`;
  document.getElementById('timeDisplay').textContent = disp;
  const badge = document.getElementById('timeBadge');
  if      (h >= 8  && h <= 10) { badge.textContent = '🔴 MORNING RUSH'; badge.className = 'time-badge rush'; }
  else if (h >= 17 && h <= 20) { badge.textContent = '🔴 EVENING RUSH'; badge.className = 'time-badge rush'; }
  else if (h >= 22 || h <= 5)  { badge.textContent = '🌙 NIGHT HOURS';  badge.className = 'time-badge night'; }
  else                          { badge.textContent = '🟡 OFF-PEAK';     badge.className = 'time-badge offpeak'; }

  clearTimeout(timeSliderDebounce);
  timeSliderDebounce = setTimeout(async () => {
    try {
      const r = await fetch('/api/time', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ hour: h })
      });
      const s = await r.json();
      applyState(s);
      toast(`🕐 Time set to ${disp} — crowds updated`, true);
    } catch (e) { }
  }, 400);
}

// ─── Leaderboard + Compare ────────────────────────────────────────────────────
function updateLeaderboard(lb) {
  const el = document.getElementById('leaderboard');
  if (!el) return;
  if (!lb || !lb.length) { el.innerHTML = '<div class="lb-empty">No episodes yet</div>'; return; }
  el.innerHTML = lb.map((e, i) => `
    <div class="lb-row ${i === 0 ? 'lb-gold' : i === 1 ? 'lb-silver' : i === 2 ? 'lb-bronze' : ''}">
      <span class="lb-rank">${['🥇','🥈','🥉','4','5','6','7','8','9','10'][i]}</span>
      <span class="lb-ep">Ep ${e.episode}</span>
      <span class="lb-score">${e.score.toFixed(1)}</span>
      <span class="lb-arr">${e.arrival_rate}% arr</span>
      <span class="lb-time">${e.timestamp}</span>
    </div>`).join('');
}

function toggleCompareMode() {
  compareMode = !compareMode;
  const cw  = document.getElementById('compareWrap');
  const lb  = document.getElementById('leaderboard');
  const btn = document.getElementById('compareBtn');
  cw.style.display  = compareMode ? 'block' : 'none';
  lb.style.display  = compareMode ? 'none'  : 'block';
  btn.textContent   = compareMode ? 'LEADERBOARD' : 'COMPARE';
  if (compareMode) loadEpisodeSelects();
}

async function loadEpisodeSelects() {
  try {
    const r = await fetch('/api/leaderboard');
    const d = await r.json();
    const keys = d.episode_keys || [];
    ['cmpA', 'cmpB'].forEach(id => {
      const sel = document.getElementById(id);
      const cur = sel.value;
      sel.innerHTML = '<option value="">Select Episode</option>';
      keys.forEach(k => { const o = document.createElement('option'); o.value = k; o.textContent = `Episode ${k}`; sel.appendChild(o); });
      if (cur) sel.value = cur;
    });
  } catch (e) { }
}

async function loadCompare() {
  const epA = document.getElementById('cmpA').value;
  const epB = document.getElementById('cmpB').value;
  if (!epA || !epB) return;
  try {
    const [rA, rB] = await Promise.all([fetch(`/api/episode/${epA}`), fetch(`/api/episode/${epB}`)]);
    const [dA, dB] = await Promise.all([rA.json(), rB.json()]);
    if (dA.error || dB.error) return;

    const ctx = document.getElementById('cmpChart').getContext('2d');
    if (cmpChart) cmpChart.destroy();
    const maxLen = Math.max(dA.history.rewards.length, dB.history.rewards.length);
    const labels = Array.from({ length: maxLen }, (_, i) => String(i));
    cmpChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          { label: `Ep ${epA}`, data: dA.history.rewards, borderColor: '#FF6B35', backgroundColor: '#FF6B3515', borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0 },
          { label: `Ep ${epB}`, data: dB.history.rewards, borderColor: '#4ECDC4', backgroundColor: '#4ECDC415', borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0 },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: { legend: { display: true, labels: { color: '#7c859a', font: { family: 'Space Mono', size: 9 } } } },
        scales: {
          x: { ticks: { color: '#3d4559', font: { family: 'Space Mono', size: 8 }, maxTicksLimit: 6 }, grid: { color: '#161b25' } },
          y: { ticks: { color: '#3d4559', font: { family: 'Space Mono', size: 8 } }, grid: { color: '#161b25' } }
        }
      }
    });

    const mA = dA.metrics, mB = dB.metrics;
    const better = (a, b, higher = true) => higher ? (a > b ? '#22c55e' : '#ef4444') : (a < b ? '#22c55e' : '#ef4444');
    document.getElementById('compareStats').innerHTML = `
      <div class="cmp-stats-grid">
        <div class="cmp-stat-hd"></div><div class="cmp-stat-hd">Ep ${epA}</div><div class="cmp-stat-hd">Ep ${epB}</div>
        <div class="cmp-stat-lbl">Score</div>
          <div style="color:${better(mA.score,mB.score)}">${mA.score.toFixed(1)}</div>
          <div style="color:${better(mB.score,mA.score)}">${mB.score.toFixed(1)}</div>
        <div class="cmp-stat-lbl">Arr Rate</div>
          <div style="color:${better(mA.arrival_rate,mB.arrival_rate)}">${mA.arrival_rate}%</div>
          <div style="color:${better(mB.arrival_rate,mA.arrival_rate)}">${mB.arrival_rate}%</div>
        <div class="cmp-stat-lbl">Avg Rew</div>
          <div style="color:${better(mA.avg_reward,mB.avg_reward)}">${mA.avg_reward.toFixed(4)}</div>
          <div style="color:${better(mB.avg_reward,mA.avg_reward)}">${mB.avg_reward.toFixed(4)}</div>
        <div class="cmp-stat-lbl">Disruptions</div>
          <div style="color:${better(mA.disruptions,mB.disruptions,false)}">${mA.disruptions}</div>
          <div style="color:${better(mB.disruptions,mA.disruptions,false)}">${mB.disruptions}</div>
        <div class="cmp-stat-lbl">Steps</div>
          <div>${mA.steps}</div><div>${mB.steps}</div>
      </div>`;
  } catch (e) { }
}

// ─── DOM updaters ─────────────────────────────────────────────────────────────
function setTxt(id, val) { const el = document.getElementById(id); if (el) el.textContent = val; }

function updateHeader(s) {
  setTxt('hdr-reward',  s.total_reward.toFixed(2));
  setTxt('hdr-arrived', `${s.agents_arrived}/10`);
  setTxt('hdr-episode', s.episode);
  setTxt('hdr-step',    s.step);
  const h = s.sim_hour !== undefined ? s.sim_hour : new Date().getHours();
  setTxt('hdr-simtime', `${String(h).padStart(2, '0')}:00`);
}

function updateSidebar(s) {
  setTxt('sb-eps', s.epsilon.toFixed(4));
  setTxt('sb-act', s.last_action);
  const sr    = s.last_reward;
  const rewEl = document.getElementById('sb-rew');
  if (rewEl) { rewEl.textContent = (sr >= 0 ? '+' : '') + sr.toFixed(4); rewEl.style.color = sr >= 0 ? '#4ECDC4' : '#ef4444'; }
  setTxt('sb-loss', s.last_loss.toFixed(4));
  setTxt('sb-dis',  s.disruptions_count);
  setTxt('sb-crowd', s.avg_crowd.toFixed(1) + '%');
}

function updateAgents(agents) {
  const el = document.getElementById('agentsList');
  if (!el || !agents) return;
  el.innerHTML = agents.map((a, idx) => `
    <div class="ag ${a.status}" onclick="openJourney(${idx})" style="cursor:pointer" title="View journey timeline">
      <div class="ag-dot"></div>
      <div class="ag-info">
        <div class="ag-name">${a.name} · <span style="color:${a.line === 'Western' ? '#FF6B35' : a.line === 'Central' ? '#4ECDC4' : '#A855F7'}">${a.line}</span></div>
        <div class="ag-route">${a.origin} → ${a.destination}</div>
        <div class="ag-at">${a.arrived ? '✓ Arrived' : '📍 ' + a.current_station}</div>
      </div>
      <div class="ag-rew">${a.reward.toFixed(1)}</div>
    </div>`).join('');
}

function updateDisruptions(dis) {
  const el = document.getElementById('disList');
  if (!el) return;
  if (!dis || !dis.length) { el.innerHTML = '<div class="no-dis">All lines clear</div>'; return; }
  el.innerHTML = [...dis].reverse().slice(0, 6).map(d => `
    <div class="dis-card">
      <div class="dis-hd">
        <span class="dis-type">⚡ ${d.type}</span>
        <span class="dis-sev ${d.severity}">${d.severity}</span>
      </div>
      <div class="dis-info">${d.line} · ${d.station} · +${d.delay_minutes}min · ${d.time}</div>
    </div>`).join('');
}

function updateHeatmap(crowd) {
  const el = document.getElementById('heatmap');
  if (!el || !crowd) return;
  const sorted = Object.entries(crowd).sort((a, b) => b[1] - a[1]).slice(0, 22);
  el.innerHTML = sorted.map(([st, v]) => {
    const col = v > 75 ? '#ef4444' : v > 50 ? '#f97316' : v > 25 ? '#eab308' : '#22c55e';
    return `<div class="hm-row">
      <span class="hm-name">${st}</span>
      <div class="hm-bar-bg"><div class="hm-bar" style="width:${v}%;background:${col}30;border-right:2px solid ${col}"></div></div>
      <span class="hm-pct">${v}%</span>
    </div>`;
  }).join('');
}

function updateTrainsList(trains) {
  const el = document.getElementById('trainsList');
  if (!el || !trains) return;
  el.innerHTML = trains.slice(0, 20).map(t => {
    const oc = t.occupancy > 80 ? 'full' : t.occupancy > 60 ? 'busy' : 'ok';
    return `<div class="tr-row ${t.line}${t.delayed ? ' delayed' : ''}">
      <span class="tr-id">${t.id}</span>
      <span class="tr-st">${t.station}${t.delayed ? ' ⚠' : ''}</span>
      <span class="tr-occ ${oc}">${t.occupancy}%</span>
    </div>`;
  }).join('');
}

// ─── Toast ────────────────────────────────────────────────────────────────────
let toastTimer;
function toast(msg, positive = true) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.style.color       = positive ? '#FF6B35' : '#ef4444';
  el.style.borderColor = positive ? '#FF6B35' : '#ef4444';
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), 2500);
}

// ─── Apply full state ─────────────────────────────────────────────────────────
function applyState(s) {
  state = s;
  updateHeader(s);
  updateSidebar(s);
  updateAgents(s.agents);
  updateDisruptions(s.disruptions);
  updateHeatmap(s.crowd);
  updateTrainsList(s.trains);
  updateTrainSmooth(s.trains);
  syncChartsFromHistory(s.history, s.step);
  updateLeaderboard(s.leaderboard);

  const btn = document.getElementById('autoBtn');
  if (s.is_auto) { btn.textContent = '⏸ PAUSE SIM'; btn.classList.add('running'); }
  else           { btn.textContent = '▶ AUTO SIM';  btn.classList.remove('running'); }
  isAuto = s.is_auto;

  if (s.sim_hour !== undefined) {
    const slider = document.getElementById('timeSlider');
    if (slider && parseInt(slider.value) !== s.sim_hour) {
      slider.value = s.sim_hour;
      onTimeSlider(s.sim_hour);
    }
  }
}

// ─── Fetch ────────────────────────────────────────────────────────────────────
async function fetchState() {
  try {
    const r = await fetch('/api/state');
    if (!r.ok) return;
    applyState(await r.json());
    lastFetch = Date.now();
  } catch (e) { console.warn('fetch failed', e); }
}

// ─── User actions ─────────────────────────────────────────────────────────────
async function manualStep(btn) {
  const action = btn.dataset.action;
  btn.classList.add('flash');
  setTimeout(() => btn.classList.remove('flash'), 400);
  try {
    const r = await fetch('/api/step', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action }) });
    const s = await r.json();
    applyState(s);
    const rw = s.last_reward;
    toast(`${action.replace('_', ' ')} → ${rw >= 0 ? '+' : ''}${rw.toFixed(3)}`, rw >= 0);
  } catch (e) { }
}

async function toggleAuto() {
  try {
    const r = await fetch('/api/auto', { method: 'POST' });
    const d = await r.json();
    isAuto = d.is_auto;
    await fetchState();
  } catch (e) { }
}

async function doReset() {
  try {
    await fetch('/api/reset', { method: 'POST' });
    trainSmooth = {};
    for (const ch of Object.values(charts)) { ch.data.labels = []; ch.data.datasets[0].data = []; ch.update('none'); }
    await fetchState();
    toast('↺ Environment reset', true);
    if (compareMode) loadEpisodeSelects();
  } catch (e) { }
}

// ─── Poll loop ────────────────────────────────────────────────────────────────
function startPolling() {
  setInterval(async () => {
    if (isAuto || Date.now() - lastFetch > 2000) await fetchState();
  }, 400);
}

// ─── Resize ───────────────────────────────────────────────────────────────────
window.addEventListener('resize', () => {
  computeLayout();
  if (state) updateTrainSmooth(state.trains);
});

// ─── Boot ─────────────────────────────────────────────────────────────────────
window.addEventListener('load', async () => {
  initCharts();
  computeLayout();
  drawFrame();
  const h = new Date().getHours();
  document.getElementById('timeSlider').value = h;
  onTimeSlider(h);
  await fetchState();
  startPolling();
});