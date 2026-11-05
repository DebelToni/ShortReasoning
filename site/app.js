'use strict';
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const signed = (value) => `${value < 0 ? '−' : '+'}${Math.abs(value).toFixed(1)}%`;
const fmt = (value, unit) => unit === 'usd' ? `$${value.toFixed(2)}` : `${value.toFixed(2)}h`;
const svg = (content, label) => `<svg viewBox="0 0 640 240" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${label}">${content}</svg>`;
function box(x, y, w, h, title, subtitle = '', type = '', faded = false) {
  return `<g${faded ? ' class="faded"' : ''}><rect class="node ${type}" x="${x}" y="${y}" width="${w}" height="${h}" rx="9"/><text x="${x + w / 2}" y="${y + h / 2 + (subtitle ? -3 : 5)}" text-anchor="middle">${title}</text>${subtitle ? `<text class="small-text" x="${x + w / 2}" y="${y + h / 2 + 16}" text-anchor="middle">${subtitle}</text>` : ''}</g>`;
}
function path(d, active = true) { return `<path class="flow-line" d="${d}"/>${active ? `<path class="flow-beam" d="${d}"/>` : ''}`; }
function options(id, attribute, callback) {
  const root = $(id);
  root.addEventListener('click', (event) => {
    const button = event.target.closest(`button[${attribute}]`);
    if (!button) return;
    root.querySelectorAll('button').forEach((b) => b.setAttribute('aria-pressed', String(b === button)));
    callback(button.getAttribute(attribute));
  });
}

function drawModel(mode = 'self') {
  const modes = [
    {key:'self', title:'Same model', sub:'self-rewrite', x:250, y:18, w:150, h:56},
    {key:'specialist', title:'Specialist', sub:'fine-tuned', x:269, y:104, w:112, h:44},
    {key:'small', title:'Small model', sub:'fast / selector', x:250, y:178, w:150, h:54}
  ];
  let content = '';
  modes.forEach(({key, title, sub, x, y, w, h}) => {
    const cy = y + h / 2;
    content += path(`M170 120 C205 120 205 ${cy} ${x} ${cy}`, mode === key);
    content += path(`M${x + w} ${cy} C445 ${cy} 445 120 488 120`, mode === key);
    const shape = key === 'small'
      ? `<g><polygon class="node compactor-node" points="${x+16},${y} ${x+w-16},${y} ${x+w},${cy} ${x+w-16},${y+h} ${x+16},${y+h} ${x},${cy}"/><text x="${x+w/2}" y="${cy-3}" text-anchor="middle">${title}</text><text class="small-text" x="${x+w/2}" y="${cy+16}" text-anchor="middle">${sub}</text></g>`
      : box(x, y, w, h, title, sub, 'compactor-node');
    content += `<g class="model-choice${mode === key ? ' selected' : ''}" data-choice="${key}">${shape}</g>`;
  });
  content += box(20, 88, 150, 64, 'Reasoning block', 'facts · decisions · doubts');
  content += box(488, 88, 132, 64, 'Compact state', 'for the same target', 'tool-node');
  $('#model-animation').innerHTML = svg(content, `Illustrative compactor model choice: ${mode}`);
}
function drawContext(mode = 'local') {
  let content = '';
  [['Task', 'Goal + constraints'], ['Reasoning block', '+ action name'], ['Prior history', 'Earlier observations']].forEach(([title, sub], i) => {
    const y = 18 + i * 76;
    content += path(`M200 ${y + 27} C260 ${y + 27} 255 121 325 121`, mode === 'history' || i === 1);
    content += box(25, y, 175, 54, title, sub, i === 1 ? '' : 'muted-node', mode === 'local' && i !== 1);
  });
  content += box(325, 89, 135, 64, 'Compactor', 'read-only snapshot', 'compactor-node');
  content += path('M460 121 H495');
  content += box(495, 89, 125, 64, 'State', 'pending result hidden', 'tool-node');
  $('#context-animation').innerHTML = svg(content, `Illustrative compactor context: ${mode === 'local' ? 'one reasoning block and action' : 'task and previous history'}`);
}
function drawBatch(mode = 'one') {
  let content = '';
  [27, 97, 167].forEach((y, i) => {
    const active = mode !== 'one' || i === 1;
    content += path(`M180 ${y + 22} C225 ${y + 22} 230 120 265 120`, active);
    content += box(30, y, 150, 44, `Reasoning ${i + 1}`, '', '', !active);
    if (mode !== 'merge') {
      content += path(`M395 120 C430 120 435 ${y + 22} 480 ${y + 22}`, active);
      content += box(480, y, 130, 44, `State ${i + 1}`, '', 'tool-node', !active);
    }
  });
  content += box(265, 88, 130, 64, 'Compactor', mode === 'merge' ? 'merge the phase' : 'rewrite the blocks', 'compactor-node');
  if (mode === 'merge') content += path('M395 120 H480') + box(480, 82, 130, 76, 'One state', 'shared facts + decisions', 'tool-node');
  $('#batch-animation').innerHTML = svg(content, `Illustrative rewrite granularity: ${mode}`);
}
function drawFidelity(mode = 'preserve') {
  const facts = ['Budget: 10', 'A costs 12 · rejected', 'B costs 8', 'Weather uncertain', 'A audited yesterday'];
  let content = '<text class="small-text" x="30" y="16">Illustrative task state</text>';
  facts.forEach((fact, i) => {
    const y = 27 + i * 39;
    const keep = mode === 'preserve' || [0, 2, 3].includes(i);
    content += box(30, y, 205, 30, fact, '', '');
    content += path(`M235 ${y + 15} H380`, keep);
    content += box(380, y, 230, 30, fact, '', keep ? 'tool-node' : 'muted-node', !keep);
    if (!keep) content += `<path d="M395 ${y + 15} H595" stroke="#87928a" stroke-width="1.5"/>`;
  });
  $('#fidelity-animation').innerHTML = svg(content, `Illustrative ${mode === 'preserve' ? 'information-preserving rewrite' : 'selective state retention with two details removed'}`);
}
function drawSchedule(mode = 'parallel') {
  const parallel = mode === 'parallel';
  const toolX = parallel ? 150 : 290;
  const end = parallel ? 360 : 500;
  let content = '<text x="25" y="75">Rewrite</text><text x="25" y="130">Tool</text>';
  for (let i = 0; i <= 5; i++) content += `<line x1="${150 + i * 70}" y1="35" x2="${150 + i * 70}" y2="153" stroke="#e6e8e2"/><text class="small-text" x="${150 + i * 70}" y="177" text-anchor="middle">${i}s</text>`;
  content += '<rect class="timeline-block" x="150" y="48" width="140" height="37" rx="6"/><text x="220" y="72" text-anchor="middle">Compact snapshot</text>';
  content += `<rect class="timeline-block tool" x="${toolX}" y="103" width="210" height="37" rx="6"/><text x="${toolX + 105}" y="127" text-anchor="middle">Execute fixed tool call</text>`;
  content += `<line x1="${end}" y1="42" x2="${end}" y2="153" stroke="#007561" stroke-width="2" stroke-dasharray="4 4"/><text x="${end + 9}" y="196" fill="#007561">Resume at ${parallel ? 3 : 5}s</text>`;
  content += '<line class="time-marker" x1="150" y1="38" x2="150" y2="151"/>';
  content += '<text class="small-text" x="150" y="224">The next target request waits for both results.</text>';
  $('#schedule-animation').innerHTML = svg(content, `Illustrative schedule: a two-second rewrite and three-second tool take ${parallel ? 'three seconds in parallel' : 'five seconds sequentially'}`);
}
drawModel(); drawContext(); drawBatch(); drawFidelity(); drawSchedule();
options('#model-options', 'data-model', drawModel);
options('#context-options', 'data-context', drawContext);
options('#batch-options', 'data-batch', drawBatch);
options('#fidelity-options', 'data-fidelity', drawFidelity);
options('#schedule-options', 'data-schedule', drawSchedule);

const track = $('#variant-track');
const cards = $$('.variant');
const tabs = $$('.variant-tabs [role=tab]');
let activeCard = 0;
function selectCard(index, move = true) {
  activeCard = Math.max(0, Math.min(cards.length - 1, index));
  tabs.forEach((tab, i) => { tab.setAttribute('aria-selected', String(i === activeCard)); tab.tabIndex = i === activeCard ? 0 : -1; });
  cards.forEach((card, i) => { card.classList.toggle('active', i === activeCard); card.inert = i !== activeCard; });
  $('#previous-card').disabled = activeCard === 0;
  $('#next-card').disabled = activeCard === cards.length - 1;
  if (move) track.scrollTo({left: cards[activeCard].offsetLeft - cards[0].offsetLeft, behavior: reducedMotion.matches ? 'auto' : 'smooth'});
}
tabs.forEach((tab, i) => tab.addEventListener('click', () => selectCard(i)));
$('#previous-card').addEventListener('click', () => selectCard(activeCard - 1));
$('#next-card').addEventListener('click', () => selectCard(activeCard + 1));
$('.variant-tabs').addEventListener('keydown', (event) => {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
  event.preventDefault();
  selectCard(event.key === 'Home' ? 0 : event.key === 'End' ? 4 : activeCard + (event.key === 'ArrowRight' ? 1 : -1));
  tabs[activeCard].focus();
});
let scrollFrame = 0;
track.addEventListener('scroll', () => {
  if (scrollFrame) return;
  scrollFrame = requestAnimationFrame(() => {
    scrollFrame = 0;
    const index = cards.reduce((best, card, i) => Math.abs(card.offsetLeft - cards[0].offsetLeft - track.scrollLeft) < Math.abs(cards[best].offsetLeft - cards[0].offsetLeft - track.scrollLeft) ? i : best, 0);
    selectCard(index, false);
  });
}, {passive: true});
new IntersectionObserver((entries) => track.classList.toggle('in-view', entries[0].isIntersecting), {threshold: .1}).observe(track);
selectCard(0, false);

async function loadResults() {
  const response = await fetch(new URL('data/results.json', document.baseURI));
  if (!response.ok) throw new Error(`Results failed to load: HTTP ${response.status}`);
  const data = await response.json();
  if (data.controlled.routes.length !== 6 || data.deployment.models.length !== 4) throw new Error('Unexpected results schema');
  const chart = $('#route-chart');
  const domainMin = Math.floor(Math.min(...data.controlled.routes.flatMap((r) => r.ci)) / 20) * 20;
  const domainMax = Math.max(0, Math.ceil(Math.max(...data.controlled.routes.flatMap((r) => r.ci)) / 20) * 20);
  const pos = (value) => (value - domainMin) / (domainMax - domainMin) * 100;
  $('#route-axis').innerHTML = Array.from({length:(domainMax-domainMin)/20+1}, (_,i) => {
    const value = domainMin + i * 20;
    return `<span class="axis-tick" data-value="${value}" style="left:${pos(value)}%"><span>${value === 0 ? '0' : `${value < 0 ? '−' : '+'}${Math.abs(value)}%`}</span></span>`;
  }).join('');
  data.controlled.routes.forEach((route, i) => {
    const row = document.createElement('button'); row.className = 'route-row'; row.setAttribute('aria-pressed', String(i === 0));
    row.setAttribute('aria-label', `${route.label}: ${signed(route.medianChangePercent)} median future reasoning change; 95% interval ${signed(route.ci[0])} to ${signed(route.ci[1])}`);
    row.innerHTML = `<span class="route-label">${route.label}</span><span class="route-plot" aria-hidden="true"><i class="route-zero" style="left:${pos(0)}%"></i><i class="route-interval" style="left:${pos(route.ci[0])}%;width:${pos(route.ci[1])-pos(route.ci[0])}%"></i><i class="route-median" style="left:${pos(route.medianChangePercent)}%"></i></span><span class="route-value">${signed(route.medianChangePercent)}</span>`;
    row.addEventListener('click', () => {
      $$('.route-row').forEach((r) => r.setAttribute('aria-pressed', String(r === row)));
      $('#route-detail').innerHTML = `<h3>${route.label}</h3><div class="detail-change">${signed(route.medianChangePercent)}</div><p class="detail-sub">median change in future reasoning</p><div class="detail-grid"><div><strong>${route.accepted}/${route.attempted}</strong><span>histories accepted</span></div><div><strong>${route.shorter}/${route.comparable}</strong><span>pairs shorter</span></div><div><strong>${route.cleanSuccess} → ${route.compactSuccess}</strong><span>success / ${route.qualityPairs}</span></div></div>`;
    });
    chart.append(row);
    if (i === 0) row.click();
  });
  const metrics = {
    spendChangePercent: {heading: 'Spend change', value: 'spendUsd', unit: 'usd', scope: 'Spend includes captured rewriting charges. Nemotron target costs use a common-tariff estimate. All 50 assigned tasks remain, including unsolved and model-limit outcomes.'},
    targetTimeChangePercent: {heading: 'Target-only change', value: 'targetHours', unit: 'hours', scope: 'Summed target-model request durations only. Rewriting, tools, orchestration and evaluation are excluded; this is not end-to-end task latency.'}
  };
  function deployment(metric) {
    const descriptor = metrics[metric];
    $('#metric-heading').textContent = descriptor.heading;
    $('#metric-scope').textContent = descriptor.scope;
    $('#deployment-rows').innerHTML = data.deployment.models.map((model) => ['external', 'self'].map((armName) => {
      const arm = model[armName]; const value = arm[metric]; const negative = value < 0; const width = Math.abs(value) / 2;
      return `<div class="deployment-row" data-model="${model.id}" data-arm="${armName}"><div class="deployment-name"><b>${model.label}${model.estimatedTariff && metric === 'spendChangePercent' ? ' †' : ''}</b><span>${armName === 'self' ? 'Self compaction' : 'External compaction'}</span></div><div class="deployment-score">${arm.score}/50<span>Normal: ${model.normal.score}/50</span></div><div class="delta-cell"><div class="delta-values"><strong class="${negative ? 'negative' : 'positive'}">${signed(value)}</strong><span class="absolute">${fmt(model.normal[descriptor.value], descriptor.unit)} → ${fmt(arm[descriptor.value], descriptor.unit)}</span></div><span class="delta-meter" aria-hidden="true"><i class="${negative ? '' : 'worse'}" style="left:${negative ? 50 - width : 50}%;width:${width}%"></i></span></div></div>`;
    }).join('')).join('');
  }
  $('#metric-tabs').addEventListener('click', (event) => {
    const button = event.target.closest('[data-metric]'); if (!button) return;
    $$('#metric-tabs button').forEach((b) => b.setAttribute('aria-pressed', String(b === button)));
    deployment(button.dataset.metric);
  });
  deployment('spendChangePercent');
  document.documentElement.dataset.ready = 'true';
}
loadResults().catch((error) => { console.error(error); const p = document.createElement('p'); p.setAttribute('role', 'alert'); p.className = 'wrap'; p.textContent = 'The result data could not load. Please reload or read the paper for the original tables.'; $('#results').append(p); });

$('#cite-button').addEventListener('click', () => $('#citation-dialog').showModal());
$('#copy-citation').addEventListener('click', async () => {
  const button = $('#copy-citation');
  try { await navigator.clipboard.writeText($('#citation-text').textContent); button.textContent = 'Copied'; }
  catch (error) { button.textContent = 'Copy failed'; console.error(error); }
});
