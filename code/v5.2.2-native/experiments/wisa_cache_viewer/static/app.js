'use strict';

const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
}[c]));
const state = {
  catalog: null, filtered: [], page: 0, perPage: 80,
  selected: null, detail: null, arm: 'text',
};

function showEmpty(text) {
  $('empty').textContent = text;
  $('empty').classList.remove('hidden');
  $('sample').classList.add('hidden');
}

function applyFilters(resetPage = true) {
  if (!state.catalog) return;
  const query = $('search').value.trim().toLowerCase();
  const scope = $('scope').value;
  const split = $('split').value;
  const oldMode = $('oldFilter').value;
  const newMode = $('newFilter').value;
  state.filtered = state.catalog.samples.filter((sample) => {
    if (scope === 'qwen38' && !sample.qwen38) return false;
    if (split !== 'all' && sample.split !== split) return false;
    const oldAny = sample.old_text || sample.old_null;
    const newAny = sample.new_text || sample.new_null;
    if (oldMode === 'available' && !oldAny) return false;
    if (oldMode === 'missing' && oldAny) return false;
    if (newMode === 'available' && !newAny) return false;
    if (newMode === 'missing' && newAny) return false;
    if (!query) return true;
    return `${sample.id} ${sample.caption || ''} ${sample.original_caption || ''}`
      .toLowerCase().includes(query);
  });
  if (resetPage) state.page = 0;
  const pages = Math.max(1, Math.ceil(state.filtered.length / state.perPage));
  state.page = Math.min(state.page, pages - 1);
  renderList();
  renderPageInfo(pages);
}

function renderStats() {
  const c = state.catalog.counts;
  $('stats').innerHTML =
    `${state.catalog.total} 条 WISA · Qwen3.8 ${state.catalog.selected_qwen38} 条` +
    `<br>旧 Qwen text ${c.old_text} / null ${c.old_null}` +
    ` · 新 Qwen text ${c.new_text} / null ${c.new_null}`;
}

function renderPageInfo(pages) {
  $('pageInfo').textContent = `第 ${state.page + 1} / ${pages} 页`;
  $('prevPage').disabled = state.page <= 0;
  $('nextPage').disabled = state.page >= pages - 1;
  $('resultCount').textContent = `筛选结果：${state.filtered.length} 条`;
}

function renderList() {
  const list = $('sampleList');
  list.replaceChildren();
  const start = state.page * state.perPage;
  const page = state.filtered.slice(start, start + state.perPage);
  if (!page.length) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = '没有匹配样本';
    list.append(empty);
    return;
  }
  for (const sample of page) {
    const item = document.createElement('div');
    item.className = 'sample-item' + (sample.id === state.selected ? ' active' : '');
    item.tabIndex = 0;
    item.innerHTML =
      `<div class="id">${esc(sample.id)}</div>` +
      `<div class="split">${esc(sample.split)}</div>` +
      `<div class="mini-badges">` +
      `${sample.qwen38 ? '<span class="mini-badge on">3.8</span>' : ''}` +
      `${sample.old_text || sample.old_null ? '<span class="mini-badge on">old</span>' : '<span class="mini-badge">old</span>'}` +
      `${sample.new_text || sample.new_null ? '<span class="mini-badge on">new</span>' : '<span class="mini-badge">new</span>'}` +
      `</div>` +
      `<div class="cap">${esc(sample.caption || sample.original_caption || '')}</div>`;
    item.addEventListener('click', () => selectSample(sample.id));
    item.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') selectSample(sample.id);
    });
    list.append(item);
  }
}

async function selectSample(id, updateHash = true) {
  state.selected = id;
  if (updateHash) history.replaceState(null, '', `#${encodeURIComponent(id)}`);
  renderList();
  showEmpty('加载样本…');
  try {
    const response = await fetch(`/api/sample/${encodeURIComponent(id)}`);
    if (!response.ok) throw new Error(await response.text());
    state.detail = await response.json();
    renderDetail();
  } catch (error) {
    showEmpty(`加载失败：${error.message}`);
  }
}

function badge(text, on = true) {
  return `<span class="badge${on ? ' on' : ''}">${esc(text)}</span>`;
}

function renderDetail() {
  const d = state.detail;
  $('empty').classList.add('hidden');
  $('sample').classList.remove('hidden');
  $('sampleId').textContent = d.id;
  $('sampleSplit').textContent = `${d.split} · ${d.clip.frames} frames`;
  $('sampleBadges').innerHTML = [
    badge(d.qwen38 ? 'Qwen3.8 400-step' : 'Qwen3.8 未处理', d.qwen38),
    badge('old text', d.old.text.available),
    badge('old null', d.old.null.available),
    badge('new text', d.new.text.available),
    badge('new null', d.new.null.available),
  ].join('');

  const clip = d.clip;
  $('clipMeta').textContent =
    `${clip.frames} frames · ${clip.fps} fps · ${clip.duration.toFixed(3)} s`;
  $('clipDetails').textContent =
    `frame ${clip.start_frame}–${clip.end_frame}\n` +
    `PTS ${clip.pts_start.toFixed(3)}–${clip.pts_end.toFixed(3)} s\n` +
    `cache ${d.geometry.w}×${d.geometry.h}`;
  $('firstMeta').textContent = `原始首帧 ${d.geometry.w}×${d.geometry.h}`;

  const video = $('clip');
  const message = $('videoMessage');
  video.pause();
  video.removeAttribute('src');
  video.removeAttribute('poster');
  video.load();
  if (d.first_image) {
    $('firstFrame').src = d.first_image;
    $('firstFrame').classList.remove('hidden');
    video.poster = d.first_image;
  } else {
    $('firstFrame').removeAttribute('src');
    $('firstFrame').classList.add('hidden');
  }
  message.textContent = '首次打开时按缓存规则裁剪视频，通常需要 0.5–2 秒…';
  message.classList.remove('hidden');
  video.src = d.video;
  video.load();
  video.oncanplay = () => message.classList.add('hidden');
  video.onwaiting = () => {
    message.textContent = '正在裁剪首帧视频…';
    message.classList.remove('hidden');
  };
  video.onerror = () => {
    message.textContent = '该片段生成失败。请检查源视频路径和服务器日志。';
    message.classList.remove('hidden');
  };

  $('originalCaption').textContent = d.original_caption || '（无原始文本）';
  $('manifestCaption').textContent = d.manifest_caption || '（无）';
  $('armText').classList.toggle('active', state.arm === 'text');
  $('armNull').classList.toggle('active', state.arm === 'null');
  renderQwen();
  renderSource();
}

function evidenceBlock(evidence, kind) {
  if (!evidence || !evidence.available) {
    return `<div class="missing">${kind === 'old'
      ? '该条件的旧 Qwen 缓存不存在。'
      : '该条件的 Qwen3.8 reasoning 不存在。'}</div>`;
  }
  const relations = (evidence.relations || []).length
    ? `<ul class="relation-list">${evidence.relations.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>`
    : '<div class="meta-line">无 relations</div>';
  const model = evidence.model || evidence.producer || '';
  const meta = [
    model ? `model: ${model}` : '',
    evidence.scope ? `scope: ${evidence.scope}` : '',
    evidence.analysis_complete === undefined ? '' : `analysis_complete: ${evidence.analysis_complete}`,
    evidence.thinking === undefined ? '' : `thinking: ${evidence.thinking}`,
    evidence.seconds === undefined ? '' : `seconds: ${evidence.seconds}`,
    evidence.request_id ? `request: ${evidence.request_id}` : '',
    evidence.finish_reason ? `finish: ${evidence.finish_reason}` : '',
  ].filter(Boolean).join(' · ');
  const raw = evidence.raw
    ? `<details><summary>trace.raw</summary><pre>${esc(evidence.raw)}</pre></details>`
    : '';
  const usage = evidence.usage
    ? `<details><summary>usage</summary><pre>${esc(JSON.stringify(evidence.usage, null, 2))}</pre></details>`
    : '';
  const prompt = evidence.prompt
    ? `<details><summary>prompt</summary><pre>${esc(evidence.prompt)}</pre></details>`
    : '';
  return `
    <h3>caption</h3>
    <pre>${esc(evidence.caption || '（空 caption / null 条件）')}</pre>
    <h3>relations</h3>
    ${relations}
    <h3>analysis</h3>
    <pre>${esc(evidence.analysis || '（无 analysis）')}</pre>
    <h3>metadata</h3>
    <div class="meta-line">${esc(meta || '无')}</div>
    ${raw}${usage}${prompt}
    <h3>source</h3>
    <div class="meta-line">${esc(evidence.source || '')}</div>`;
}

function renderQwen() {
  const d = state.detail;
  if (!d) return;
  $('oldText').innerHTML = evidenceBlock(d.old[state.arm], 'old');
  $('newText').innerHTML = evidenceBlock(d.new[state.arm], 'new');
}

function renderSource() {
  const d = state.detail;
  $('sourceInfo').textContent =
    `source video: ${d.source_video}\n` +
    `frame range: ${d.clip.start_frame}–${d.clip.end_frame} (${d.clip.frames} frames)\n` +
    `fps: ${d.clip.fps} · duration: ${d.clip.duration}s · PTS ${d.clip.pts_start}–${d.clip.pts_end}\n` +
    `geometry: ${JSON.stringify(d.geometry)}\n` +
    `manifest: ${d.source.manifest}\n` +
    `qwen38 reasoning: ${d.source.qwen38_reasoning}`;
}

function page(delta) {
  const pages = Math.max(1, Math.ceil(state.filtered.length / state.perPage));
  state.page = Math.max(0, Math.min(pages - 1, state.page + delta));
  renderList();
  renderPageInfo(pages);
}

async function init() {
  try {
    const response = await fetch('/api/catalog');
    if (!response.ok) throw new Error(await response.text());
    state.catalog = await response.json();
    renderStats();
    applyFilters();
    const initial = decodeURIComponent((location.hash || '').replace(/^#/, ''));
    if (initial) await selectSample(initial, false);
    else if (state.filtered.length) await selectSample(state.filtered[0].id, false);
  } catch (error) {
    showEmpty(`无法加载目录：${error.message}`);
  }
}

$('search').addEventListener('input', () => applyFilters());
for (const id of ['scope', 'split', 'oldFilter', 'newFilter']) {
  $(id).addEventListener('change', () => applyFilters());
}
$('prevPage').addEventListener('click', () => page(-1));
$('nextPage').addEventListener('click', () => page(1));
$('armText').addEventListener('click', () => {
  state.arm = 'text';
  $('armText').classList.add('active');
  $('armNull').classList.remove('active');
  renderQwen();
});
$('armNull').addEventListener('click', () => {
  state.arm = 'null';
  $('armNull').classList.add('active');
  $('armText').classList.remove('active');
  renderQwen();
});
document.addEventListener('keydown', (event) => {
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
  if (event.key === 'ArrowLeft') page(-1);
  if (event.key === 'ArrowRight') page(1);
});
init();
