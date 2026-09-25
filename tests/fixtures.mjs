// Unit-test metadata only. Never imported by the deployed app.
export function manifestFixture() {
  const file = path => ({path, bytes: 4, sha256: 'a'.repeat(64)});
  return {format: 'text-iq-pages-v1', schema_version: 1, status: 'ready', id: 'b'.repeat(64),
    features: {adapter: 'e5-masked-mean-l2-v1', prefix: 'query: ', normalization: 'nfc-python-whitespace', dimension: 384, max_length: 512},
    ridge: {coef: [5, ...Array(383).fill(0)], intercept: 50, clip: [0,100], iq_offset: 70, iq_scale: 0.6},
    encoder: {precision: 'fp32', inputs: ['input_ids','attention_mask','token_type_ids'], output:'embedding', bytes:4, chunks:[file('encoder/part-000.bin')]},
    tokenizer: {files:[file('tokenizer/tokenizer.json'),file('tokenizer/tokenizer_config.json')]},
    parity: {...file('parity.json'), python_onnx_passed:true,embedding_atol:0.0002,score_atol:0.02},
    training:{warning:'UNIT TEST ONLY, NOT A TRAINED MODEL'}};
}
