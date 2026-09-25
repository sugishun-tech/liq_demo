import {safePath} from './core.mjs';
export async function sha256(bytes) {
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(digest)].map(x => x.toString(16).padStart(2, '0')).join('');
}
export async function checkedFetch(baseURL, item, {cache = null, onProgress = () => {}} = {}) {
  const url = new URL(safePath(item.path), baseURL).href;
  let response = null;
  try { if (cache) response = await cache.match(url); } catch { /* Cache is optional. */ }
  let fromCache = !!response;
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      response ||= await fetch(url, {cache: 'no-store', credentials: 'omit', referrerPolicy: 'no-referrer'});
      if (!response.ok) throw new Error(`モデルファイルを取得できません: ${item.path} (HTTP ${response.status})`);
      const length = Number(response.headers.get('content-length'));
      if (length && !response.headers.get('content-encoding') && length !== item.bytes) throw new Error(`ファイルサイズが不一致です: ${item.path}`);
      let bytes;
      if (response.body?.getReader) {
        bytes = new Uint8Array(item.bytes);
        const reader = response.body.getReader(); let offset = 0;
        try {
          while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            if (offset + value.length > bytes.length) throw new Error(`ファイルが宣言サイズを超えました: ${item.path}`);
            bytes.set(value, offset); offset += value.length; onProgress(offset, item.bytes);
          }
          if (offset !== bytes.length) throw new Error(`モデルファイルが途中で途切れています: ${item.path}`);
        } catch (error) {
          try { await reader.cancel(); } catch { /* Preserve the original read failure. */ }
          throw error;
        } finally { reader.releaseLock(); }
      } else { bytes = new Uint8Array(await response.arrayBuffer()); }
      if (bytes.length !== item.bytes || await sha256(bytes) !== item.sha256) throw new Error(`チェックサムが不一致です: ${item.path}。モデルを再配置してください。`);
      if (cache && !fromCache) {
        try { await cache.put(url, new Response(bytes, {headers: {'content-type': 'application/octet-stream'}})); }
        catch { /* Quota/private mode must not prevent inference. */ }
      }
      return bytes;
    } catch (error) {
      if (!fromCache) throw error;
      try { await cache.delete(url); } catch { /* Retry with network, even if eviction fails. */ }
      response = null; fromCache = false;
    }
  }
  throw new Error('モデルファイルの検証に失敗しました。');
}
export async function modelCache(id, baseURL) {
  try {
    // Separate project sites on the same github.io origin, without evicting other projects.
    const scope = (await sha256(new TextEncoder().encode(baseURL))).slice(0, 12);
    const prefix = `text-iq-${scope}-`;
    const name = prefix + id;
    for (const old of await caches.keys()) if (old.startsWith(prefix) && old !== name) await caches.delete(old);
    return await caches.open(name);
  } catch { return null; }
}
