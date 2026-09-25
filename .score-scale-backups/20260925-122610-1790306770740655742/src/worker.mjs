import {createEngine} from './runtime.mjs';
import {occlusionPlan, attribution} from './core.mjs';
let engine = null;
let busy = false;
self.onmessage = async ({data}) => {
  const {id, type} = data;
  const send = (kind, payload = {}) => self.postMessage({id, type: kind, ...payload});
  if (busy) { send('error', {message: '前の処理が実行中です。'}); return; }
  busy = true;
  try {
    if (type === 'init') {
      if (engine) await engine.dispose();
      engine = null;
      engine = await createEngine({siteURL: data.siteURL,
        progress: (message, fraction) => send('progress', {message, fraction})});
      send('ready', {backend: engine.backend, 
        verification: engine.verification, model_id: engine.manifest.id});
    } else if (type === 'score') {
      if (!engine) throw new Error('先にモデルを読み込んでください。');
      const started = performance.now();
      const result = await engine.run(data.text);
      send('score', {text: result.text, score: result.score, tokens: result.ids.length,
        elapsed_ms: performance.now() - started});
    } else if (type === 'explain') {
      if (!engine) throw new Error('先にモデルを読み込んでください。');
      const plan = occlusionPlan(data.text, data.mode);
      const started = performance.now();
      const base = await engine.run(plan.text);
      const spans = [];
      // Cache only within this request; no user text is persisted.
      const memo = new Map();
      for (let i = 0; i < plan.spans.length; i++) {
        const variant = plan.variants[i];
        if (!memo.has(variant)) memo.set(variant, (await engine.run(variant, true)).score);
        const altered = memo.get(variant);
        spans.push({...plan.spans[i], delta: attribution(base.score, altered, engine.manifest.ridge.iq_scale),
          without_iq: altered.iq, without_raw_iq: altered.raw_iq, variant, empty_variant: !variant});
        send('progress', {message: `差分を計算中 ${i + 1}/${plan.spans.length}`, fraction: (i + 1) / plan.spans.length});
      }
      send('explanation', {text: plan.text, score: base.score, tokens: base.ids.length, spans,
        mode: data.mode, elapsed_ms: performance.now() - started,
        method: 'one-span-space-occlusion; delta=0.6*(raw_original-raw_occluded); NOT additive/causal'});
    } else { throw new Error('不明な操作です。'); }
  } catch (error) {
    send('error', {message: error?.message || String(error)});
  } finally { busy = false; }
};
