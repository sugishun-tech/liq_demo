import test from 'node:test';
import assert from 'node:assert/strict';
import {normalizeText, assertInput, safePath, validateManifest, scoreVector,
  segmentsFor, occlusionPlan, attribution, joinHeatSegments, signed} from '../docs/src/core.mjs';
import {tokensFor} from '../docs/src/runtime.mjs';
import {manifestFixture} from './fixtures.mjs';

test('NFC and Python whitespace normalization', () => {
  assert.equal(normalizeText('  Cafe\u0301\t\n 日本語\u0085\u001cです。 '), 'Café 日本語 です。');
  assert.equal(normalizeText('\ufeffA\ufeff'), '\ufeffA\ufeff');
  assert.equal(normalizeText('  \u3000'), '');
});
test('Empty and oversized input fail explicitly', () => {
  assert.throws(() => assertInput('   ')); assert.throws(() => assertInput('a'.repeat(4001)));
  assert.equal(assertInput('', true), ''); assert.throws(() => normalizeText(null));
});
for (const path of ['../secrets','/model.bin','https://bad/file','a\\b','a//b','a/./b','a/%2e%2e/b','']) {
  test('Reject path '+JSON.stringify(path), () => assert.throws(() => safePath(path)));
}
test('Valid artifact paths', () => assert.equal(safePath('encoder/part-000.bin'), 'encoder/part-000.bin'));
test('Accept matching ready manifest', () => assert.equal(validateManifest(manifestFixture()).status,'ready'));
for (const [name, change] of [
  ['missing',m=>m.status='missing'], ['wrong format',m=>m.format='old'], ['nonfinite',m=>m.ridge.coef[0]=NaN],
  ['dimension',m=>m.features.dimension=768],['normalization',m=>m.features.normalization='NFKC'],
  ['prefix',m=>m.features.prefix='passage: '],['moving id',m=>m.id='main'],['weights',m=>m.ridge.coef.pop()],
  ['clip',m=>m.ridge.clip=[0,200]],['token limit',m=>m.features.max_length=1024],
  ['chunk sum',m=>m.encoder.bytes=5],['checksum',m=>m.encoder.chunks[0].sha256='bad'],
  ['unverified',m=>m.parity.python_onnx_passed=false],['loose tolerance',m=>m.parity.score_atol=50],
  ['quantization mismatch',m=>m.encoder.precision='int8'],['tokenizer missing',m=>m.tokenizer.files.pop()],
  ['invalid graph',m=>m.encoder.inputs=['input_ids']],
]) {
  test('Manifest rejects '+name,()=>{const m=manifestFixture();change(m);assert.throws(()=>validateManifest(m));});
}
test('Linear projection and IQ transform',()=>{
  const m=manifestFixture();const v=[1,...Array(383).fill(0)];const s=scoreVector(v,m.ridge);
  assert.equal(s.raw,55);assert.equal(s.iq,103);assert.equal(s.clipped,false);
});
test('Clipping preserves the unbounded score',()=>{
  const m=manifestFixture();m.ridge.intercept=120;const s=scoreVector([1,...Array(383).fill(0)],m.ridge);
  assert.equal(s.complexity,100);assert.equal(s.raw,125);assert.equal(s.iq,130);assert.equal(s.raw_iq,145);assert.ok(s.clipped);
});
test('Vector shape/nonfinite/norm checks',()=>{
  const m=manifestFixture();assert.throws(()=>scoreVector([1],m.ridge));
  assert.throws(()=>scoreVector(Array(384).fill(0),m.ridge));
  assert.throws(()=>scoreVector([NaN,...Array(383).fill(0)],m.ridge));
});
test('Japanese word segmentation preserves all nonspace spans',()=>{
  const text='結果は変わった。しかし 原因は不明。'; const spans=segmentsFor(text);
  const parts=joinHeatSegments(text,spans);
  assert.equal(parts.map(p=>p.text).join(''),text);assert.ok(spans.length>3);
});
test('Grapheme mode keeps composed letters and emoji together',()=>{
  const p=occlusionPlan('e\u0301 🧑🏽‍💻家族👨‍👩‍👧‍👦','grapheme');
  assert.ok(p.spans.some(x=>x.text==='🧑🏽‍💻')); assert.ok(p.spans.some(x=>x.text==='👨‍👩‍👧‍👦'));
  assert.equal(p.spans[0].text,'é');
});
test('Space occlusion preserves normalized text alignment',()=>{
  const p=occlusionPlan('abc def','word'); assert.deepEqual(p.variants,['def','abc']);
});
test('Single fragment may lead to empty variant',()=>assert.deepEqual(occlusionPlan('abc').variants,['']));
test('Explain budget fails rather than silently skipping characters',()=>assert.throws(()=>occlusionPlan('a'.repeat(129),'grapheme')));
test('Sign of occlusion effect and clipping independence',()=>{
  assert.equal(attribution({raw:125},{raw:115}),6);
  assert.equal(attribution({raw:10},{raw:20}),-6);
});
test('Heatmap rejects stale/overlapping offsets',()=>{
  assert.throws(()=>joinHeatSegments('abc',[{start:0,end:2,text:'xx'}]));
  assert.throws(()=>joinHeatSegments('abc',[{start:0,end:2,text:'ab'},{start:1,end:3,text:'bc'}]));
});
test('XSS payload remains plain text in segmentation',()=>{
  const x='<img src=x onerror=alert(1)>';assert.equal(joinHeatSegments(x,segmentsFor(x)).map(p=>p.text).join(''),x);
});
test('Signed output',()=>{assert.equal(signed(3),'+3.00');assert.equal(signed(-2),'-2.00');});
test('Tokenizer.encode returns IDs; max length is checked without truncation',()=>{
  const m=manifestFixture();let options;
  const tok={encode(t,o){assert.ok(t.startsWith('query: '));options=o;return {ids:[0,5,2]};}};
  assert.deepEqual(tokensFor(tok,'日 本',m),[0,5,2]);assert.equal(options.add_special_tokens,true);
  assert.throws(()=>tokensFor({encode:()=>({ids:Array(513).fill(2)})},'x',m));
  assert.throws(()=>tokensFor({encode:()=>({ids:[NaN]})},'x',m));
});
