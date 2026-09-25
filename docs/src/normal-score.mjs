/** Frozen reference-rank scoring. No random numbers, training, or network. */
function finite(x, name) {
  if (typeof x !== 'number' || !Number.isFinite(x)) throw new Error(name + 'が有限数ではありません。');
  return x;
}
// LIQ_RETRAIN_HEAD_V1_BEGIN
export function validateRetrainedHead(h, dim, depth = 0) {
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
export function rawRetrainedHead(v, h) {
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

export function validateCalibration(c) {
  if (!c || c.format !== 'rank-truncated-normal-v1' || c.lower !== 70 || c.mean !== 100 || c.upper !== 130) throw new Error('再尺度化の形式/範囲が不正です。');
  const {x, y} = c;
  if (!Array.isArray(x) || !Array.isArray(y) || x.length !== y.length || x.length < 2 || x.length > 2000000 || c.knots !== x.length || !Number.isSafeInteger(c.reference_rows) || c.reference_rows < x.length) throw new Error('再尺度化の変換表が不正です。');
  if (!Number.isFinite(c.parent_normal_scale) || c.parent_normal_scale <= 0) throw new Error('分布の幅が不正です。');
  for (let i=0; i<x.length; i++) {
    finite(x[i], '変換元'); finite(y[i], '変換先');
    if (i && (x[i] <= x[i-1] || y[i] <= y[i-1])) throw new Error('変換表は厳密単調増加である必要があります。');
  }
  if (y[0]<70-1e-8 || y[y.length-1]>130+1e-8) throw new Error('変換値が範囲外です。');
  return c;
}
export function validateNormalModel(m, dimension=384) {
  if (!m || m.format !== 'liq-rank-normal-head-v1' || !/^[0-9a-f]{64}$/.test(m.id) || m.dimension !== dimension) throw new Error('再尺度化モデルが不正です。');
  validateRetrainedHead(m.head, dimension); validateCalibration(m.calibration);
  return m;
}
// Validate the table once during manifest loading; inference is O(log n).
export function mapRankScore(raw, c) {
  finite(raw, '元スコア');
  const {x, y} = c;
  if (raw <= x[0]) return y[0];
  let hi = x.length-1;
  if (raw >= x[hi]) return y[hi];
  let lo = 0;
  while (hi-lo>1) { const mid = (lo+hi) >>> 1; if (x[mid]<=raw) lo=mid; else hi=mid; }
  return y[lo] + (raw-x[lo])/(x[hi]-x[lo])*(y[hi]-y[lo]);
}
export function scoreNormalModel(vector, m) {
  if (vector.length !== m.dimension) throw new Error('埋め込み次元が不一致です。');
  const v = Array.from(vector, x=>finite(Number(x), '埋め込み'));
  const norm = Math.sqrt(v.reduce((s,x)=>s+x*x,0));
  if (Math.abs(norm-1)>.01) throw new Error('E5の正規化に失敗しました。');
  const raw = finite(rawRetrainedHead(v, m.head), '回帰値');
  const complexity = Math.max(0,Math.min(100,raw));
  const c = m.calibration, iq = mapRankScore(raw, c);
  return {raw, complexity, iq, raw_iq:70+.6*raw, original_iq:70+.6*complexity,
    calibrated:true, calibration_id:m.id, clipped:raw!==complexity,
    outside_reference:raw<c.x[0] || raw>c.x[c.x.length-1]};
}
