#!/usr/bin/env python3
"""Validate a static directory in place. No build, deploy, or files rewritten."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from bundle import BundleError, digest, read_json
from vendor import SETTINGS, validate_vendor
ROOT = Path(__file__).resolve().parents[1]

def safe_file(root: Path, value: str):
    if not isinstance(value, str) or not value or '\\' in value or any(p in ('', '.', '..') for p in value.split('/')):
        raise BundleError('不正なバンドル相対パスです。')
    p = root / value
    if not p.is_file() or not p.resolve().is_relative_to(root.resolve()) or p.is_symlink():
        raise BundleError(f'ファイルがないかパスが不正です: {value}')
    return p

def validate_model(directory: Path, allow_missing=False):
    manifest = read_json(directory / 'manifest.json')
    if manifest.get('format') != 'text-iq-pages-v1': raise BundleError('モデル形式が不正です。')
    if manifest.get('status') == 'missing':
        if not allow_missing: raise BundleError('学習済みモデル未配置です。prepare.sh にstudent/を渡してください。')
        return False
    if manifest.get('status') != 'ready': raise BundleError('モデル状態が不正です。')
    if manifest.get('parity', {}).get('python_onnx_passed') is not True: raise BundleError('ONNXの数値照合を通っていません。')
    files = manifest['encoder']['chunks'] + manifest['tokenizer']['files'] + [manifest['parity']]
    if len({x['path'] for x in files}) != len(files): raise BundleError('モデルに重複ファイル指定があります。')
    for item in files:
        p = safe_file(directory, item['path'])
        if p.stat().st_size != item['bytes'] or digest(p) != item['sha256']:
            raise BundleError(f'モデルファイルのサイズ/ハッシュが不一致です: {item["path"]}')
    expected = {x['path'] for x in files} | {'manifest.json'}
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    if actual != expected: raise BundleError('model/ に管理外のファイルがあります。公開用のモデルだけを配置してください。')
    if sum(f['bytes'] for f in manifest['encoder']['chunks']) != manifest['encoder']['bytes']:
        raise BundleError('ONNX分割ファイルの合計サイズが不一致です。')
    return True

def validate_site(site: Path, allow_missing=False):
    if site.is_symlink(): raise BundleError('公開ディレクトリにsymlinkは使用しません。')
    validate_model(site / 'model', allow_missing)
    validate_vendor(site, allow_missing)
    if read_json(site / 'runtime-config.json') != SETTINGS: raise BundleError('同梱CPUランタイム設定が不一致です。')
    total = 0
    for p in site.rglob('*'):
        if p.is_symlink(): raise BundleError(f'公開物にsymlinkは含めません: {p}')
        if p.is_file():
            n = p.stat().st_size; total += n
            if n >= 100 * 1024**2: raise BundleError(f'100MiB以上の単一ファイルです: {p}')
            if p.suffix in {'.sqlite3', '.sqlite', '.npz', '.pkl', '.joblib', '.py'}: raise BundleError(f'公開不要の学習/開発ファイルです: {p}')
        if p.name.startswith(('.model-build-', 'model.previous-', '.vendor-stage-')): raise BundleError(f'未完了の準備ファイルが残っています: {p}')
        if p.name in {'.github', '.cache', '.venv', 'node_modules'}: raise BundleError(f'公開物に不要なディレクトリです: {p}')
    if total >= 950_000_000: raise BundleError('サイトの合計が950MB以上です。Pagesの容量制限に余裕がありません。')
    for name in ('index.html', '.nojekyll', 'styles.css', 'src/app.mjs', 'src/worker.mjs', 'src/runtime.mjs', 'runtime-config.json'):
        safe_file(site, name)
    return total

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--site', type=Path, default=ROOT / 'docs')
    p.add_argument('--allow-missing-model', action='store_true', help='未配置の案内画面のみを検査。完成したデモとは扱いません。')
    args = p.parse_args(argv)
    try:
        total = validate_site(args.site, args.allow_missing_model)
        state = read_json(args.site / 'model/manifest.json')['status']
        print(f'静的ファイル検査OK: {total:,} bytes; model={state}。ファイルの変更/ビルドは行っていません。')
        return 0
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr); return 2
if __name__ == '__main__': raise SystemExit(main())
