import {normalizeText, assertInput, occlusionPlan, joinHeatSegments, signed, validateManifest, MAX_INPUT} from './core.mjs';
const $ = id => document.getElementById(id);
const examples = {
  ja: '測定値は変わった。しかし、センサーの位置も変わったため、この結果だけでは故障と断定できない。',
  en: 'The reading changed, but the sensor also moved. This result alone cannot establish that the sensor is broken.',
  short: '今日は晴れている。',
};
let worker = null, request = 0, ready = false, busy = false, hasModel = false;
let manifest = null, current = null, runtime = null;
function error(message) { $('error').textContent = message; $('error').hidden = !message; }
function progress(message, fraction = null) {
  $('progress-text').textContent = message;
  $('progress').hidden = !busy;
  if (Number.isFinite(fraction)) $('progress').value = Math.max(0, Math.min(1, fraction));
  else $('progress').removeAttribute('value');
}
function sync() {
  const input = normalizeText($('text').value);
  $('char-count').textContent = `${[...input].length}文字 · 空白正規化後`;
  $('text').disabled = busy;
  $('load').disabled = busy || ready || !hasModel;
  $('score').disabled = busy || !ready || !input;
  $('explain').disabled = busy || !ready || !current || current.text !== input;
  $('granularity').disabled = busy;
  $('cancel').hidden = !busy;
  $('export').disabled = busy || !current;
  $('clear').disabled = busy;
  document.querySelectorAll('[data-example]').forEach(b => b.disabled = busy);
}
function resetResult() {
  current = null;
  $('iq').textContent = '—'; $('complexity').textContent = '— / 100'; $('tokens').textContent = '— tokens';
  $('scale-fill').style.width = '0%'; $('scale-marker').hidden = true;
  $('score-note').textContent = ready ? '文章を入力して計算してください。' : 'モデルを読み込み、文章を入力して計算してください。';
  $('heatmap').replaceChildren(); $('heatmap').textContent = 'スコア計算後に、ヒートマップを実行できます。'; $('heatmap').classList.add('placeholder');
  $('influences').hidden = true;
  $('heat-detail').textContent = '語・文字を選ぶと、除去前後の点数を確認できます。';
  sync();
}
function showScore(result) {
  const {score} = result;
  $('iq').textContent = score.iq.toFixed(1);
  $('complexity').textContent = score.complexity.toFixed(1) + ' / 100';
  $('tokens').textContent = result.tokens + ' tokens';
  $('scale-fill').style.width = `${score.complexity}%`;
  $('scale-marker').style.left = `${score.complexity}%`; $('scale-marker').hidden = false;
  $('score-note').textContent = `${(result.elapsed_ms / 1000).toFixed(2)}秒${score.clipped ? ' · 表示範囲外のためクリップしています。raw IQ代理値: ' + score.raw_iq.toFixed(2) : ''}`;
}
function detail(span) {
  const message = `「${span.text}」を空白に置換: raw代理値 ${current.score.raw_iq.toFixed(2)} → ${span.without_raw_iq.toFixed(2)}。元の値への差分 ${signed(span.delta)}。${span.empty_variant ? '除去後は空文です（学習分布外の可能性）。' : ''}`;
  $('heat-detail').textContent = message;
  return message;
}
function showHeatmap(result) {
  const root = $('heatmap'); root.replaceChildren(); root.classList.remove('placeholder');
  const max = Math.max(0.001, ...result.spans.map(s => Math.abs(s.delta)));
  for (const part of joinHeatSegments(result.text, result.spans)) {
    if (part.delta === null) { root.append(document.createTextNode(part.text)); continue; }
    const button = document.createElement('button'); button.type = 'button'; button.className = 'heat-token';
    button.textContent = part.text;
    const opacity = Math.abs(part.delta) < 0.005 ? 0.04 : 0.12 + 0.48 * Math.abs(part.delta) / max;
    button.style.backgroundColor = part.delta >= 0 ? `rgba(226,122,77,${opacity})` : `rgba(81,138,224,${opacity})`;
    button.title = `${part.text} / 除去差分 ${signed(part.delta)} / 除去後の表示値 ${part.without_iq.toFixed(2)}`;
    button.setAttribute('aria-label', button.title);
    for (const event of ['mouseenter', 'focus', 'click']) button.addEventListener(event, () => detail(part));
    root.append(button);
  }
  $('heat-detail').textContent = `${result.spans.length}区間 · ${(result.elapsed_ms / 1000).toFixed(2)}秒 · 色の濃さはこの文章内で相対化。各区間を選ぶと数値を確認できます。`;
  const list = $('influence-list'); list.replaceChildren();
  for (const part of [...result.spans].sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta)).slice(0, 12)) {
    const row = document.createElement('div'); row.className = 'influence-row';
    const text = document.createElement('code'); text.textContent = part.text;
    const value = document.createElement('span'); value.textContent = signed(part.delta) + ' raw代理点';
    row.append(text, value); list.append(row);
  }
  $('influences').hidden = false;
}
function onMessage({data}) {
  if (data.id !== request) return;
  if (data.type === 'progress') { progress(data.message, data.fraction); return; }
  busy = false;
  if (data.type === 'error') {
    error(data.message); progress('処理を停止しました。');
    if (!ready) { worker?.terminate(); worker = null; $('model-status').textContent = '読み込みに失敗'; }
  } else if (data.type === 'ready') {
    ready = true; runtime = data;
    $('model-status').textContent = 'CPU / WebAssembly · 数値照合済み';
    $('model-dot').classList.add('ready'); $('load').textContent = '読み込み済み';
    $('model-detail').textContent = `${data.verification.cases}例でPython出力と照合しました。採点精度の保証ではありません。`;
    $('model-info').textContent = JSON.stringify({model_id: data.model_id, backend: data.backend,
      verification: data.verification, training: manifest.training, features: manifest.features}, null, 2);
    progress('モデルの準備ができました。');
  } else if (data.type === 'score') {
    current = {...data}; showScore(data);
    $('heatmap').textContent = '「ヒートマップを計算」で影響を可視化できます。';
    $('heatmap').classList.add('placeholder'); $('influences').hidden = true;
    progress('スコアの計算が完了しました。');
  } else if (data.type === 'explanation') {
    current = {...data}; showScore(data); showHeatmap(data); progress('ヒートマップの計算が完了しました。');
  }
  sync();
}
function send(type, payload = {}) {
  busy = true; error(''); request++;
  sync(); progress(type === 'init' ? 'モデルを読み込んでいます。' : '計算しています。');
  worker.postMessage({id: request, type, ...payload});
}
$('load').addEventListener('click', () => {
  if (busy || ready || !hasModel) return;
  try {
    worker = new Worker(new URL('./worker.mjs', import.meta.url), {type: 'module'});
    worker.onmessage = onMessage;
    worker.onerror = () => { busy = false; ready = false; worker?.terminate(); worker = null; $('model-status').textContent = 'Workerが停止しました'; $('model-dot').classList.remove('ready'); $('load').textContent = 'モデルを再読み込み'; error('推論Workerを開始できません。ブラウザの開発者コンソールとネットワークを確認してください。'); progress('処理を停止しました。'); sync(); };
    send('init', {siteURL: new URL('../', import.meta.url).href});
  } catch (e) { error(e.message); busy = false; sync(); }
});
$('score').addEventListener('click', () => {
  if (busy || !ready) return;
  try { const text = assertInput($('text').value); resetResult(); send('score', {text}); }
  catch (e) { error(e.message); }
});
$('explain').addEventListener('click', () => {
  if (busy || !ready || !current) return;
  try {
    const mode = $('granularity').value;
    occlusionPlan(current.text, mode); // Reject oversized explanations before any expensive work.
    send('explain', {text: current.text, mode});
  } catch (e) { error(e.message); }
});
$('cancel').addEventListener('click', () => {
  request++; worker?.terminate(); worker = null; busy = false; ready = false;
  $('model-status').textContent = '中止しました'; $('model-dot').classList.remove('ready'); $('load').textContent = 'モデルを再読み込み';
  $('model-detail').textContent = 'Workerを終了しました。続行時はモデルを再読み込みしてください。';
  progress('中止しました。完了していないヒートマップは表示しません。'); sync();
});
$('text').addEventListener('input', () => { error(''); resetResult(); });
$('clear').addEventListener('click', () => { $('text').value = ''; error(''); resetResult(); $('text').focus(); });
$('text').addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !e.isComposing) { e.preventDefault(); $('score').click(); }
});
document.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click', () => {
  $('text').value = examples[button.dataset.example]; error(''); resetResult();
}));
$('export').addEventListener('click', () => {
  if (!current) return;
  const report = {format: 'text-iq-result-v1', created_at: new Date().toISOString(), model_id: manifest.id,
    runtime: {backend: runtime?.backend, verification: runtime?.verification}, ...current,
    warning: '文章の代理指標。人間のIQではない。除去差分は教師の思考過程/因果的根拠/加法的寄与ではない。'};
  delete report.id; delete report.type;
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], {type: 'application/json'}));
  const a = document.createElement('a'); a.href = url; a.download = 'text-iq-result.json'; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
async function inspect() {
  try {
    const response = await fetch(new URL('../model/manifest.json', import.meta.url), {cache: 'no-store'});
    if (!response.ok) throw new Error(`モデル設定がありません (HTTP ${response.status})。`);
    const value = await response.json();
    if (value.status === 'missing') {
      $('setup').hidden = false; $('model-status').textContent = '学習済みモデル未配置';
      $('model-detail').textContent = 'サイトの画面は公開できますが、採点には学習済みモデルが必要です。';
      progress('配置ガイドに沿ってモデルを変換してください。');
    } else {
      manifest = validateManifest(value);
      const vendorResponse = await fetch(new URL('../vendor-lock.json', import.meta.url), {cache: 'no-store'});
      const vendor = vendorResponse.ok ? await vendorResponse.json() : null;
      if (vendor?.status !== 'ready') throw new Error('推論ライブラリ未配置です。公開者が prepare.sh を実行し、docs/ を丸ごと配置してください。');
      hasModel = true;
      const total = manifest.encoder.bytes + manifest.tokenizer.files.reduce((s, f) => s + f.bytes, 0);
      $('model-status').textContent = 'モデルを読み込みできます';
      $('model-detail').textContent = `モデル約${(total / 1024 ** 2).toFixed(0)} MiB + 推論ライブラリ。初回は通信とメモリが必要です。`;
      progress('「モデルを読み込む」から開始してください。');
    }
  } catch (e) { error(e.message); $('model-status').textContent = '設定を確認できません'; progress('配置を確認してください。'); }
  sync();
}
// Reject overlong input at inference; do not silently truncate pasted text.
sync(); inspect();
