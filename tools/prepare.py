#!/usr/bin/env python3
"""Prepare all public files once on CPU; publication then only copies docs/."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from bundle import BundleError, fingerprint, load_student, read_json
from check import validate_site
from cpu_only import require_cpu
from vendor import prepare_vendor
import export_web
ROOT = Path(__file__).resolve().parents[1]

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--student', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--offline', action='store_true')
    p.add_argument('--encoder-dir', type=Path)
    p.add_argument('--replace-model', action='store_true')
    args = p.parse_args(argv)
    try:
        if args.threads < 1: raise BundleError('--threadsは1以上です。')
        source, _, _ = load_student(args.student)
        require_cpu()
        export_web.check_versions(source['embedding_spec'])
        prepare_vendor(ROOT / 'docs', args.offline)
        target = ROOT / 'docs/model'
        existing = read_json(target / 'manifest.json') if (target / 'manifest.json').is_file() else {}
        same = existing.get('status') == 'ready' and existing.get('training', {}).get('source_manifest_sha256') == fingerprint(source)
        if not same:
            flags = ['--student', str(args.student), '--out', str(target), '--threads', str(args.threads)]
            if args.offline: flags.append('--offline')
            if args.encoder_dir: flags += ['--encoder-dir', str(args.encoder_dir)]
            if args.replace_model: flags.append('--replace')
            result = export_web.main(flags)
            if result: return result
        total = validate_site(ROOT / 'docs')
        print(f'準備完了: docs/ ({total:,} bytes)。')
        print('公開: Settings → Pages → Deploy from a branch → main /docs → Save')
        print('または docs/ の中身を公開リポジトリのルートへコピーし、main /(root) を選択。')
        print('公開時のビルド、.github、Python、Node.js、推論APIは不要です。')
        print('公開前に python tools/serve.py で実ブラウザの数値照合を確認してください。')
        return 0
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr); return 2
if __name__ == '__main__': raise SystemExit(main())
