import test from 'node:test';import assert from 'node:assert/strict';
import {sha256,checkedFetch} from '../docs/src/assets.mjs';
const base='https://example.test/repo/model/';
test('SHA256 implementation',async()=>assert.equal(await sha256(new TextEncoder().encode('abc')),'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'));
test('Fetch validates bytes and checksum',async()=>{
 const original=globalThis.fetch;const b=new Uint8Array([1,2,3]);
 globalThis.fetch=async url=>{assert.equal(url,base+'a.bin');return new Response(b)};
 try{const out=await checkedFetch(base,{path:'a.bin',bytes:3,sha256:await sha256(b)});assert.deepEqual(out,b)}finally{globalThis.fetch=original}
});
test('Corrupt downloaded bytes are rejected',async()=>{
 const original=globalThis.fetch;globalThis.fetch=async()=>new Response('bad');
 try{await assert.rejects(checkedFetch(base,{path:'a.bin',bytes:3,sha256:'0'.repeat(64)}));}finally{globalThis.fetch=original}
});
test('Corrupt cache is evicted and refetched once',async()=>{
 const original=globalThis.fetch;let deleted=0,called=0;const good=new TextEncoder().encode('new');
 const cache={match:async()=>new Response('old'),delete:async()=>deleted++,put:async()=>{}};
 globalThis.fetch=async()=>{called++;return new Response(good)};
 try{assert.deepEqual(await checkedFetch(base,{path:'a.bin',bytes:3,sha256:await sha256(good)},{cache}),good);assert.equal(deleted,1);assert.equal(called,1)}finally{globalThis.fetch=original}
});
test('Truncated or oversized responses do not produce a model',async()=>{
 const original=globalThis.fetch;
 try{for(const data of ['a','abcde']){globalThis.fetch=async()=>new Response(data);await assert.rejects(checkedFetch(base,{path:'a.bin',bytes:3,sha256:'0'.repeat(64)}))}}finally{globalThis.fetch=original}
});
test('Unavailable browser cache is not required for inference',async()=>{
 const original=globalThis.fetch;const b=new TextEncoder().encode('abc');globalThis.fetch=async()=>new Response(b);
 const cache={match:async()=>null,put:async()=>{throw new Error('quota')}};
 try{assert.deepEqual(await checkedFetch(base,{path:'a.bin',bytes:3,sha256:await sha256(b)},{cache}),b)}finally{globalThis.fetch=original}
});

for(const bad of ['x','abcde']) test('Wrong-size cached data is refetched: '+bad.length,async()=>{
 const original=globalThis.fetch;const good=new TextEncoder().encode('abc');let calls=0;
 const cache={match:async()=>new Response(bad),delete:async()=>{},put:async()=>{}};
 globalThis.fetch=async()=>{calls++;return new Response(good)};
 try{assert.deepEqual(await checkedFetch(base,{path:'a.bin',bytes:3,sha256:await sha256(good)},{cache}),good);assert.equal(calls,1)}finally{globalThis.fetch=original}
});
test('Cache lookup denied by browser does not prevent download',async()=>{
 const original=globalThis.fetch;const good=new TextEncoder().encode('abc');globalThis.fetch=async()=>new Response(good);
 const cache={match:async()=>{throw new Error('denied')},put:async()=>{throw new Error('denied')}};
 try{assert.deepEqual(await checkedFetch(base,{path:'a.bin',bytes:3,sha256:await sha256(good)},{cache}),good)}finally{globalThis.fetch=original}
});
test('Encoded Content-Length is not mistaken for decompressed size',async()=>{
 const original=globalThis.fetch;const good=new TextEncoder().encode('abc');
 globalThis.fetch=async()=>new Response(good,{headers:{'content-encoding':'gzip','content-length':'23'}});
 try{assert.deepEqual(await checkedFetch(base,{path:'a.bin',bytes:3,sha256:await sha256(good)}),good)}finally{globalThis.fetch=original}
});
