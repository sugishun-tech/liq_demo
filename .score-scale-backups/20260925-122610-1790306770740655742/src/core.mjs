/** Pure inference/attribution utilities. No network, DOM, or model substitutes. */
export const FORMAT = 'text-iq-pages-v1';
export const MAX_INPUT = 4000;
export const MAX_SEGMENTS = 128;
// Match Python str.split() whitespace exactly; JavaScript \s also removes U+FEFF.
const PY_SPACE = /[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+/gu;
export function normalizeText(text) {
  if (typeof text !== 'string') throw new Error('テキストを入力してください。');
  return text.normalize('NFC').split(PY_SPACE).filter(Boolean).join(' ');
}
export function assertInput(text, allowEmpty = false) {
  if (typeof text !== 'string' || text.length > MAX_INPUT) throw new Error(`入力は${MAX_INPUT} UTF-16コード単位以下にしてください。`);
  const clean = normalizeText(text);
  if (!allowEmpty && !clean) throw new Error('空の文章は採点しません。');
  return clean;
}
export function finite(x, name) {
  if (typeof x !== 'number' || !Number.isFinite(x)) throw new Error(`${name} が有限数ではありません。`);
  return x;
}
export function safePath(path) {
  if (typeof path !== 'string' || !path || !/^[A-Za-z0-9_.\/-]+$/.test(path) || path.startsWith('/') || path.split('/').some(x => !x || x === '.' || x === '..')) {
    throw new Error('モデル内の相対パスが不正です。');
  }
  return path;
}
function checkFile(f) {
  if (!f || typeof f !== 'object') throw new Error('モデルファイルの記述が不正です。');
  safePath(f.path);
  if (!/^[0-9a-f]{64}$/.test(f.sha256) || !Number.isSafeInteger(f.bytes) || f.bytes < 1) throw new Error('モデルファイルのハッシュ/サイズが不正です。');
}
export function validateManifest(m) {
  if (!m || m.format !== FORMAT) throw new Error('対応していないモデル形式です。');
  if (m.status !== 'ready') throw new Error('学習済みモデル未配置です。tools/export_web.py でv0.3の student/ を変換してください。');
  if (m.schema_version !== 1 || !/^[0-9a-f]{64}$/.test(m.id)) throw new Error('モデルのID/版が不正です。');
  const f = m.features;
  if (!f || f.adapter !== 'e5-masked-mean-l2-v1' || f.prefix !== 'query: ' || f.normalization !== 'nfc-python-whitespace' || f.dimension !== 384 || !Number.isInteger(f.max_length) || f.max_length < 8 || f.max_length > 512) throw new Error('学習時と対応しない特徴量仕様です。');
  const r = m.ridge;
  if (!r || !Array.isArray(r.coef) || r.coef.length !== f.dimension) throw new Error('回帰係数の次元が不正です。');
  r.coef.forEach(x => finite(x, '回帰係数')); finite(r.intercept, '切片');
  if (JSON.stringify(r.clip) !== '[0,100]' || r.iq_offset !== 70 || r.iq_scale !== 0.6) throw new Error('スコア変換仕様が不正です。');
  const e = m.encoder;
  if (!e || e.precision !== 'fp32' || e.output !== 'embedding' || JSON.stringify(e.inputs) !== '["input_ids","attention_mask","token_type_ids"]') throw new Error('対応していないONNX入出力です。');
  if (!Array.isArray(e.chunks) || !e.chunks.length || e.chunks.length > 32) throw new Error('ONNX分割ファイルがありません。');
  e.chunks.forEach(checkFile);
  if (new Set(e.chunks.map(x => x.path)).size !== e.chunks.length || e.chunks.some(x => x.bytes > 48 * 1024 ** 2) || e.chunks.reduce((s, x) => s + x.bytes, 0) !== e.bytes || e.bytes > 850 * 1024 ** 2) throw new Error('ONNXのファイルサイズ/構成が不正です。');
  if (!m.tokenizer || !Array.isArray(m.tokenizer.files)) throw new Error('Tokenizer情報がありません。');
  m.tokenizer.files.forEach(checkFile);
  for (const name of ['tokenizer/tokenizer.json', 'tokenizer/tokenizer_config.json']) {
    if (!m.tokenizer.files.some(x => x.path === name)) throw new Error(`${name} がありません。`);
  }
  checkFile(m.parity);
  if (m.parity.python_onnx_passed !== true || !Number.isFinite(m.parity.embedding_atol) || m.parity.embedding_atol <= 0 || m.parity.embedding_atol > 0.001 || !Number.isFinite(m.parity.score_atol) || m.parity.score_atol <= 0 || m.parity.score_atol > 0.1) throw new Error('変換時の数値検証記録が不正です。');
  if (r.retrained_head) validateRetrainedHead(r.retrained_head, f.dimension);
  return m;
}
// LIQ_RETRAIN_HEAD_V1_BEGIN
function validateRetrainedHead(h, dim, depth = 0) {
  if (!h || depth > 2) throw new Error('再学習headが不正です。');
  const array = (v, n) => Array.isArray(v) && v.length === n && v.every(Number.isFinite);
  let ok = false;
  if (h.kind === 'linear') ok = array(h.coef, dim) && Number.isFinite(h.intercept);
  else if (h.kind === 'rbf') ok = Array.isArray(h.centers) && h.centers.length > 0 && h.centers.length <= 4096 && h.centers.every(c => array(c, dim)) && array(h.coef, h.centers.length) && Number.isFinite(h.gamma) && h.gamma > 0 && Number.isFinite(h.intercept);
  else if (h.kind === 'mlp') {
    ok = array(h.mean, dim) && array(h.scale, dim) && h.scale.every(x => x > 0) && Number.isFinite(h.y_mean) && Number.isFinite(h.y_scale) && h.y_scale > 0 && Array.isArray(h.coefs) && Array.isArray(h.biases) && h.coefs.length > 0 && h.coefs.length <= 4 && h.coefs.length === h.biases.length;
    let width = dim;
    if (ok) for (let i = 0; i < h.coefs.length; i++) {
      const w = h.coefs[i], b = h.biases[i];
      if (!Array.isArray(b) || !b.length || b.length > 2048 || !b.every(Number.isFinite) || !Array.isArray(w) || w.length !== width || !w.every(row => array(row, b.length))) { ok = false; break; }
      width = b.length;
    }
    ok = ok && width === 1;
  } else if (h.kind === 'blend') {
    ok = Array.isArray(h.heads) && h.heads.length > 0 && h.heads.length <= 3 && array(h.weights, h.heads.length) && h.weights.every(x => x >= 0) && Math.abs(h.weights.reduce((a,b) => a+b,0)-1) < 1e-8;
    if (ok) h.heads.forEach(x => validateRetrainedHead(x, dim, depth+1));
  }
  if (!ok) throw new Error('再学習headの係数/形状が不正です。');
}
function rawRetrainedHead(v, h) {
  if (h.kind === 'linear') return h.intercept + v.reduce((s,x,i) => s+x*h.coef[i],0);
  if (h.kind === 'rbf') {
    let s = h.intercept;
    for (let j=0;j<h.centers.length;j++) {
      let d2=0;
      for (let i=0;i<v.length;i++) { const d=v[i]-h.centers[j][i]; d2+=d*d; }
      s+=Math.exp(-h.gamma*d2)*h.coef[j];
    }
    return s;
  }
  if (h.kind === 'mlp') {
    let x=v.map((a,i)=>(a-h.mean[i])/h.scale[i]);
    for (let k=0;k<h.coefs.length;k++) {
      const w=h.coefs[k], next=h.biases[k].slice();
      for (let i=0;i<x.length;i++) for (let j=0;j<next.length;j++) next[j]+=x[i]*w[i][j];
      x=k<h.coefs.length-1 ? next.map(a=>Math.max(0,a)) : next;
    }
    return x[0]*h.y_scale+h.y_mean;
  }
  if (h.kind === 'blend') return h.heads.reduce((s,x,i)=>s+h.weights[i]*rawRetrainedHead(v,x),0);
  throw new Error('未知の再学習headです。');
}
export function scoreVector(vector, ridge) {
  if (vector.length !== ridge.coef.length) throw new Error('埋め込み次元が不一致です。');
  const v=Array.from(vector, x=>finite(Number(x),'埋め込み'));
  const norm=Math.sqrt(v.reduce((s,x)=>s+x*x,0));
  if (Math.abs(norm-1)>.01) throw new Error('E5の正規化に失敗しました。');
  const sum=ridge.retrained_head ? rawRetrainedHead(v,ridge.retrained_head) : ridge.intercept+v.reduce((s,x,i)=>s+x*ridge.coef[i],0);
  finite(sum,'予測値');
  const complexity=Math.max(ridge.clip[0],Math.min(ridge.clip[1],sum));
  return {raw:sum,complexity,iq:ridge.iq_offset+ridge.iq_scale*complexity,
    raw_iq:ridge.iq_offset+ridge.iq_scale*sum,clipped:sum!==complexity};
}
// LIQ_RETRAIN_HEAD_V1_END

export function segmentsFor(text, mode = 'word') {
  if (!['word', 'grapheme'].includes(mode)) throw new Error('ヒートマップの単位が不正です。');
  if (!globalThis.Intl?.Segmenter) throw new Error('このブラウザは文字分割に対応していません。更新したブラウザをご利用ください。');
  // Include punctuation, not just isWordLike: punctuation can affect the model too.
  return [...new Intl.Segmenter('ja', {granularity: mode}).segment(text)]
    .filter(p => normalizeText(p.segment) !== '')
    .map(p => ({start: p.index, end: p.index + p.segment.length, text: p.segment}));
}
export function occlusionPlan(text, mode = 'word', limit = MAX_SEGMENTS) {
  const clean = assertInput(text);
  const spans = segmentsFor(clean, mode);
  if (spans.length > limit) throw new Error(`説明対象が${spans.length}区間あります。上限は${limit}です。文章を短くするか「語・記号」単位を選んでください。`);
  return {text: clean, spans, variants: spans.map(p => normalizeText(clean.slice(0, p.start) + ' ' + clean.slice(p.end)))};
}
export function attribution(base, altered, scale = 0.6) {
  return finite((finite(base.raw, '元スコア') - finite(altered.raw, '除去後スコア')) * scale, '除去差分');
}
export function joinHeatSegments(text, spans) {
  const parts = [];
  let end = 0;
  for (const s of spans) {
    if (!Number.isInteger(s.start) || !Number.isInteger(s.end) || s.start < end || s.end <= s.start || s.end > text.length || text.slice(s.start, s.end) !== s.text) throw new Error('ヒートマップと文章の位置が一致しません。');
    if (s.start > end) parts.push({text: text.slice(end, s.start), delta: null});
    parts.push(s); end = s.end;
  }
  if (end < text.length) parts.push({text: text.slice(end), delta: null});
  return parts;
}
export function signed(value, digits = 2) {
  finite(value, '差分');
  return (value > 0 ? '+' : '') + value.toFixed(digits);
}
