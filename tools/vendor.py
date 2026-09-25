#!/usr/bin/env python3
"""Fetch only pinned CPU browser runtime files, once, before publication.

No npm install, build, CUDA package, JSEP binary, or GPU runtime is used.
The deployed site loads these files locally. It never falls back to a CDN.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request
from bundle import BundleError, describe_file, digest, read_json, write_json
ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'text-iq-runtime-v2'
SETTINGS = {
    'tokenizers': './vendor/tokenizers/tokenizers.mjs',
    'onnxruntime': './vendor/onnxruntime/ort.wasm.min.mjs',
    'wasm_path': './vendor/onnxruntime/',
}
SPECS = [
    ('tokenizers/tokenizers.mjs', 'https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.mjs'),
    ('tokenizers/LICENSE', 'https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/LICENSE'),
    ('onnxruntime/ort.wasm.min.mjs', 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.wasm.min.mjs'),
    ('onnxruntime/ort-wasm-simd-threaded.mjs', 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort-wasm-simd-threaded.mjs'),
    ('onnxruntime/ort-wasm-simd-threaded.wasm', 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort-wasm-simd-threaded.wasm'),
    # onnxruntime-web's npm package declares MIT in package.json, but its published
    # package does not expose LICENSE at the npm root. Fetch the license from the
    # exact upstream v1.22.0 tag instead of constructing a non-existent CDN URL.
    ('onnxruntime/LICENSE', 'https://raw.githubusercontent.com/microsoft/onnxruntime/v1.22.0/LICENSE'),
]
EXPECTED = {'vendor/' + dest for dest, _ in SPECS}
MAX_BYTES = 64 * 1024**2

def check_payload(path):
    if not path.is_file() or path.stat().st_size < 32 or path.stat().st_size > MAX_BYTES:
        raise BundleError(f'ランタイムのサイズが不正です: {path.name}')
    with path.open('rb') as f: head = f.read(4096)
    if b'<html' in head.lower() or b'<!doctype html' in head.lower():
        raise BundleError(f'ライブラリの代わりにHTMLが返されました: {path.name}')
    if path.suffix == '.wasm' and head[:8] != b'\x00asm\x01\x00\x00\x00':
        raise BundleError('WASMのヘッダーが不正です。')

def validate_vendor(site: Path, allow_missing=False):
    lock = read_json(site / 'vendor-lock.json')
    if lock.get('format') != FORMAT: raise BundleError('ランタイムlock形式が不正です。')
    if lock.get('status') == 'missing':
        if allow_missing and not (site / 'vendor').exists(): return False
        raise BundleError('同梱ランタイムが未準備です。prepare.sh または tools/vendor.py を実行してください。')
    if lock.get('status') != 'ready' or {x.get('path') for x in lock.get('files', [])} != EXPECTED or len(lock['files']) != len(EXPECTED):
        raise BundleError('ランタイムの必要ファイル一覧が不一致です。')
    if read_json(site / 'runtime-config.json') != SETTINGS: raise BundleError('CPU専用ランタイム設定が不一致です。')
    actual = set()
    for file in (site / 'vendor').rglob('*'):
        if file.is_symlink(): raise BundleError('vendor/ のsymlinkは使用しません。')
        if file.is_file(): actual.add(file.relative_to(site).as_posix())
    if actual != EXPECTED: raise BundleError('vendor/ に不足または管理外のファイルがあります。GPU用ファイル等を混ぜないでください。')
    for item in lock['files']:
        file = site / item['path']
        check_payload(file)
        if file.stat().st_size != item['bytes'] or digest(file) != item['sha256']:
            raise BundleError(f'ランタイムのサイズ/ハッシュが不一致です: {item["path"]}')
    return True

def download(url: str, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'text-iq-pages-static/2.0', 'Accept-Encoding': 'identity'})
    try:
        with urllib.request.urlopen(req, timeout=60) as response, target.open('wb') as out:
            count = 0
            while chunk := response.read(1024 * 1024):
                count += len(chunk)
                if count > MAX_BYTES: raise BundleError('ランタイム配布ファイルがサイズ上限を超えました。')
                out.write(chunk)
        check_payload(target)
    except BaseException:
        target.unlink(missing_ok=True)
        raise

def prepare_vendor(site: Path, offline=False):
    site = site.resolve()
    if (site / 'vendor-lock.json').is_file() and read_json(site / 'vendor-lock.json').get('status') == 'ready':
        validate_vendor(site)
        print('CPUランタイム: 準備済みファイルを再利用。', flush=True)
        return
    if (site / 'vendor').exists(): raise BundleError('管理外のvendor/があります。別の新規ディレクトリで準備してください。')
    if offline: raise BundleError('ランタイムが未取得です。初回だけ--offlineを外して実行してください。')
    cache = ROOT / '.cache/vendor'; cache.mkdir(parents=True, exist_ok=True)
    site.mkdir(parents=True, exist_ok=True)
    # Stage on the same filesystem as the public directory for atomic directory replacement.
    with tempfile.TemporaryDirectory(prefix='.vendor-stage-', dir=site.parent) as temp:
        stage = Path(temp) / 'vendor'; stage.mkdir()
        records = []
        for dest, source_url in SPECS:
            local = stage / dest
            cached = cache / dest
            if not cached.exists():
                print(f'取得: {dest}', flush=True)
                temporary = cached.with_name(cached.name + '.part')
                download(source_url, temporary)
                cached.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temporary, cached)
            check_payload(cached)
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, local)
            item = describe_file(local, Path(temp))
            item['source'] = source_url
            records.append(item)
        os.replace(stage, site / 'vendor')
        try:
            write_json(site / 'runtime-config.json', SETTINGS)
            write_json(site / 'vendor-lock.json', {'format': FORMAT, 'status': 'ready', 'files': records,
                'versions': {'@huggingface/tokenizers': '0.2.0', 'onnxruntime-web': '1.22.0'},
                'integrity_note': 'SHA-256 recorded from version-pinned HTTPS downloads; not an upstream signature.'})
            validate_vendor(site)
        except BaseException:
            shutil.rmtree(site / 'vendor')
            write_json(site / 'vendor-lock.json', {'format': FORMAT, 'status': 'missing', 'files': []})
            raise

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--site', type=Path, default=ROOT / 'docs')
    p.add_argument('--offline', action='store_true')
    args = p.parse_args(argv)
    try:
        prepare_vendor(args.site, args.offline)
        print('同梱完了。フロントエンドのビルドは不要です。')
        return 0
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr); return 2
if __name__ == '__main__': raise SystemExit(main())
