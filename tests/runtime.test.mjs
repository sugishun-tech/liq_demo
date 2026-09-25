import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {runtimeURL, verifyTokenIDs} from '../docs/src/runtime.mjs';
import {manifestFixture} from './fixtures.mjs';

for(const base of ['https://account.github.io/repo/','https://account.github.io/']) {
  test(`Local URLs work under ${base}`,()=>{
    assert.equal(runtimeURL('./vendor/onnxruntime/ort.wasm.min.mjs',base),base+'vendor/onnxruntime/ort.wasm.min.mjs');
    assert.equal(runtimeURL('./vendor/onnxruntime/',base,true),base+'vendor/onnxruntime/');
  });
}
for(const value of ['https://cdn.jsdelivr.net/file.mjs','//evil.test/file','../vendor/a','/vendor/a','./vendor/../x','./vendor/%2e%2e/x','./vendor/a?x=1','./vendor/a#x','./vendor/a\\x']) {
  test(`Reject nonlocal runtime: ${value}`,()=>assert.throws(()=>runtimeURL(value,'https://example.test/repo/')));
}
test('Token mismatches fail before weights load',()=>{
  const tokenizer={encode:()=>({ids:[0,4,2]})},m=manifestFixture();
  const parity={cases:Array.from({length:4},()=>({text:'test',input_ids:[0,4,2]}))};
  verifyTokenIDs(tokenizer,m,parity);
  parity.cases[3].input_ids=[1];
  assert.throws(()=>verifyTokenIDs(tokenizer,m,parity),/不一致/);
});
test('The app has no GPU selection and loads only a CPU provider',()=>{
  const runtime=fs.readFileSync(new URL('../docs/src/runtime.mjs',import.meta.url),'utf8');
  const index=fs.readFileSync(new URL('../docs/index.html',import.meta.url),'utf8');
  assert.ok(runtime.includes("executionProviders: ['wasm']"));
  assert.ok(runtime.includes('ort.env.wasm.numThreads = 1'));
  assert.ok(!/webgpu|navigator\??\.gpu|\.jsep/i.test(runtime));
  assert.ok(!index.includes('id="backend"'));
  assert.ok(!index.includes('https://cdn.'));
});
