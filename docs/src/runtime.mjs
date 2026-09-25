import {validateManifest, assertInput, normalizeText, scoreVector} from './core.mjs';
import {checkedFetch, modelCache} from './assets.mjs';

const JSON_HEADERS = {cache: 'no-store', credentials: 'omit', referrerPolicy: 'no-referrer'};
export async function readJSON(url) {
  const r = await fetch(url, JSON_HEADERS);
  if (!r.ok) throw new Error(`設定を取得できません (HTTP ${r.status})。file:// ではなくHTTPで開いてください。`);
  return r.json();
}
export function runtimeURL(value, baseURL, directory = false) {
  if (typeof value !== 'string' || !value.startsWith('./vendor/') || value.includes('\\') || value.includes('%') || value.split('/').includes('..')) throw new Error('ランタイムは同梱vendor/の相対パスだけを使用できます。');
  const base = new URL(baseURL), u = new URL(value, base);
  if (u.origin !== base.origin || !u.pathname.startsWith(new URL('./vendor/', base).pathname) || u.username || u.password || u.search || u.hash) throw new Error('外部ランタイムは使用しません。');
  if (directory && !u.pathname.endsWith('/')) throw new Error('WASMディレクトリは / で終わる必要があります。');
  return u.href;
}
export function tokensFor(tokenizer, text, manifest) {
  const result = tokenizer.encode(manifest.features.prefix + normalizeText(text), {add_special_tokens: true});
  const ids = result.ids;
  if (!Array.isArray(ids) || !ids.length || ids.some(x => !Number.isSafeInteger(x) || x < 0)) throw new Error('TokenizerのトークンIDが不正です。');
  if (ids.length > manifest.features.max_length) throw new Error(`入力は${ids.length}トークンです。上限${manifest.features.max_length}を超えるため停止しました。本文は切り捨てません。`);
  return ids;
}
export function verifyTokenIDs(tokenizer, manifest, parity) {
  if (!Array.isArray(parity.cases) || parity.cases.length < 4 || parity.cases.length > 50) throw new Error('変換確認用のテスト文が不足しています。');
  for (const [i, sample] of parity.cases.entries()) {
    if (JSON.stringify(tokensFor(tokenizer, sample.text, manifest)) !== JSON.stringify(sample.input_ids)) throw new Error(`PythonとブラウザのトークンIDが不一致です (確認文${i + 1})。モデルの取得と採点を停止します。`);
  }
}
export async function verifyParity(engine, parity, progress) {
  let maxEmbeddingError = 0, maxScoreError = 0, maxCalibratedError = 0;
  for (let i = 0; i < parity.cases.length; i++) {
    const sample = parity.cases[i];
    if (!Array.isArray(sample.embedding) || sample.embedding.length !== engine.manifest.features.dimension || sample.embedding.some(x => !Number.isFinite(x)) || !Number.isFinite(sample.raw_score)) throw new Error('変換確認データが不正です。');
    const actual = await engine.run(sample.text, true);
    const error = Math.max(...actual.embedding.map((x, j) => Math.abs(x - sample.embedding[j])));
    maxEmbeddingError = Math.max(maxEmbeddingError, error);
    maxScoreError = Math.max(maxScoreError, Math.abs(actual.score.raw - sample.raw_score));
    if (engine.manifest.ridge.normal_model) {
      if (!Number.isFinite(sample.calibrated_iq) || !Number.isFinite(actual.score.iq)) throw new Error('再尺度化スコアの照合データが不正です。');
      maxCalibratedError = Math.max(maxCalibratedError, Math.abs(actual.score.iq-sample.calibrated_iq));
    }
    progress(`数値照合 ${i + 1}/${parity.cases.length}`, (i + 1) / parity.cases.length);
  }
  const limits = engine.manifest.parity;
  if (maxEmbeddingError > limits.embedding_atol || maxScoreError > limits.score_atol) throw new Error(`数値照合に失敗しました (埋め込み差=${maxEmbeddingError.toPrecision(3)}, スコア差=${maxScoreError.toPrecision(3)})。`);
  if (engine.manifest.ridge.normal_model && maxCalibratedError > limits.calibrated_score_atol) throw new Error(`再尺度化スコアの数値照合に失敗しました (差=${maxCalibratedError})。`);
  return {passed: true, cases: parity.cases.length, max_embedding_error: maxEmbeddingError, max_raw_score_error: maxScoreError, max_calibrated_iq_error: maxCalibratedError};
}
export async function createEngine({siteURL, progress = () => {}}) {
  if (!globalThis.isSecureContext || !globalThis.crypto?.subtle) throw new Error('HTTPSまたはlocalhostで開いてください。');
  const base = new URL('model/', siteURL).href;
  const manifest = validateManifest(await readJSON(new URL('manifest.json', base)));
  const settings = await readJSON(new URL('runtime-config.json', siteURL));
  if (settings.tokenizers !== './vendor/tokenizers/tokenizers.mjs' || settings.onnxruntime !== './vendor/onnxruntime/ort.wasm.min.mjs' || settings.wasm_path !== './vendor/onnxruntime/') throw new Error('CPU専用の同梱ランタイム設定と一致しません。');
  progress('同梱のCPU推論ライブラリを読み込み中', null);
  const [tk, ort] = await Promise.all([
    import(runtimeURL(settings.tokenizers, siteURL)),
    import(runtimeURL(settings.onnxruntime, siteURL)),
  ]);
  ort.env.wasm.numThreads = 1; // Requires no custom COOP/COEP headers on GitHub Pages.
  ort.env.wasm.proxy = false; // This code already runs in a dedicated worker.
  ort.env.wasm.wasmPaths = runtimeURL(settings.wasm_path, siteURL, true);
  const cache = await modelCache(manifest.id, base);
  const decoder = new TextDecoder(), tokenFiles = {};
  for (const file of manifest.tokenizer.files) tokenFiles[file.path] = JSON.parse(decoder.decode(await checkedFetch(base, file, {cache})));
  const config = tokenFiles['tokenizer/tokenizer_config.json'];
  if (config.tokenizer_class?.replace(/Fast$/, '') !== 'XLMRobertaTokenizer') throw new Error('この版はE5-smallのXLMRobertaTokenizer専用です。');
  const tokenizer = new tk.Tokenizer(tokenFiles['tokenizer/tokenizer.json'], config);
  const parity = JSON.parse(decoder.decode(await checkedFetch(base, manifest.parity, {cache})));
  verifyTokenIDs(tokenizer, manifest, parity); // Fail before downloading the much larger encoder.
  const bytes = new Uint8Array(manifest.encoder.bytes);
  let offset = 0;
  for (const [i, file] of manifest.encoder.chunks.entries()) {
    const chunk = await checkedFetch(base, file, {cache, onProgress: n => progress(`E5モデル ${i + 1}/${manifest.encoder.chunks.length}`, (offset + n) / bytes.length)});
    bytes.set(chunk, offset); offset += chunk.length;
  }
  let session;
  try {
    progress('CPU / WebAssemblyを初期化中', null);
    session = await ort.InferenceSession.create(bytes, {executionProviders: ['wasm'], graphOptimizationLevel: 'all'});
    if (session.inputNames.join(',') !== manifest.encoder.inputs.join(',') || !session.outputNames.includes('embedding')) throw new Error('ONNXの入出力がmanifestと不一致です。');
    const engine = {
      manifest, backend: 'wasm',
      async run(text, allowEmpty = false) {
        const clean = assertInput(text, allowEmpty), ids = tokensFor(tokenizer, clean, manifest), dims = [1, ids.length];
        const feeds = {
          input_ids: new ort.Tensor('int64', BigInt64Array.from(ids, x => BigInt(x)), dims),
          attention_mask: new ort.Tensor('int64', new BigInt64Array(ids.length).fill(1n), dims),
          token_type_ids: new ort.Tensor('int64', new BigInt64Array(ids.length), dims),
        };
        let outputs;
        try {
          outputs = await session.run(feeds);
          const out = outputs.embedding;
          if (!out || out.dims.length !== 2 || out.dims[0] !== 1 || out.dims[1] !== manifest.features.dimension) throw new Error('ONNXの埋め込み形状が不正です。');
          const embedding = Array.from(await out.getData());
          return {text: clean, ids, embedding, score: scoreVector(embedding, manifest.ridge)};
        } finally {
          for (const tensor of Object.values(feeds)) tensor.dispose();
          if (outputs) for (const tensor of Object.values(outputs)) tensor.dispose();
        }
      },
      async dispose() { await session.release(); },
    };
    engine.verification = await verifyParity(engine, parity, progress);
    return engine;
  } catch (error) {
    try { if (session) await session.release(); } catch { /* Preserve the original error. */ }
    throw error;
  }
}
