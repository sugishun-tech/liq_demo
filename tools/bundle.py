"""Validated v0.3 artifact reader and static bundle helpers; no model downloads."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import unicodedata
import uuid
import zipfile

FORMAT = 'text-iq-pages-v1'
PATTERNS = ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
            'special_tokens_map.json', 'sentencepiece.bpe.model', 'vocab.txt']

class BundleError(ValueError):
    pass

def read_json(path: Path):
    with path.open(encoding='utf-8') as f:
        return json.load(f, parse_constant=lambda x: (_ for _ in ()).throw(BundleError(f'JSONの非有限数: {x}')))

def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def digest(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def normalize(text: str):
    if not isinstance(text, str): raise BundleError('textは文字列である必要があります。')
    return ' '.join(unicodedata.normalize('NFC', text).split())

def safe_name(name: str):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in ('.', '..'):
        raise BundleError('重みファイル名が不正です。')
    return name

def load_student(path: Path):
    import numpy as np
    path = path / 'model.json' if path.is_dir() else path
    if not path.is_file(): raise BundleError(f'学習済みmodel.jsonがありません: {path}。v0.3のliq trainを先に完了してください。')
    m = read_json(path)
    if m.get('format') != 'liq-linear-v3' or m.get('dimension') != 384:
        raise BundleError('この変換器はv0.3の384次元E5 + Ridge成果物専用です。')
    weight = path.parent / safe_name(m.get('weights'))
    if not weight.is_file() or digest(weight) != m.get('weights_sha256'):
        raise BundleError('回帰重みが欠損または改変されています。')
    with zipfile.ZipFile(weight) as z:
        if sum(f.file_size for f in z.infolist()) > 1024 * 1024:
            raise BundleError('384次元回帰器として重みが大きすぎます。')
    with np.load(weight, allow_pickle=False) as z:
        coef = np.asarray(z['coef'], dtype=np.float64)
        intercept_arr = np.asarray(z['intercept'], dtype=np.float64)
    if coef.shape != (384,) or intercept_arr.shape != () or not np.isfinite(coef).all() or not np.isfinite(intercept_arr):
        raise BundleError('係数・切片に不正な形状または非有限数があります。')
    spec = m.get('embedding_spec', {})
    if (spec.get('adapter') != 'e5-masked-mean-l2-v1' or spec.get('prefix') != 'query: '
        or spec.get('dimension') != 384 or spec.get('normalize') is not True
        or spec.get('precision') != 'float32-cpu'
        or not isinstance(spec.get('max_length'), int) or not 8 <= spec['max_length'] <= 512):
        raise BundleError('対応していない埋め込み仕様です。学習時と別の特徴量へ黙って変更はしません。')
    ref = spec.get('model', {})
    if ref.get('kind') not in ('hub', 'local') or not re.fullmatch(r'[0-9a-f]{40,64}', ref.get('revision', '')):
        raise BundleError('学習時の固定モデルリビジョンがありません。')
    if ref.get('kind') == 'hub' and ref.get('repo') != 'intfloat/multilingual-e5-small':
        raise BundleError('この版が対応するHubモデルはintfloat/multilingual-e5-smallだけです。')
    if m.get('clip') != [0, 100] or m.get('iq_proxy_transform') != {'offset': 70, 'scale': .6}:
        raise BundleError('未知のスコア変換仕様です。')
    if 'TEST-DOUBLE' in json.dumps(spec) or 'TEST-DOUBLE' in json.dumps(m.get('teacher_spec')):
        raise BundleError('テスト用教師/埋め込みを公開モデルへ変換しません。')
    return m, coef, float(intercept_arr)

def describe_file(path: Path, root: Path):
    return {'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size, 'sha256': digest(path)}

def split_file(source: Path, target: Path, chunk_bytes=48 * 1024 ** 2):
    if not isinstance(chunk_bytes, int) or not 1 <= chunk_bytes <= 48 * 1024 ** 2:
        raise BundleError('分割サイズは1バイト〜48MiBです。')
    if source.stat().st_size <= 0 or source.stat().st_size > 850 * 1024 ** 2:
        raise BundleError('ONNXモデルのサイズが空または850MiB超です。')
    folder = target / 'encoder'; folder.mkdir(parents=True, exist_ok=True)
    files = []
    with source.open('rb') as f:
        i = 0
        while block := f.read(chunk_bytes):
            p = folder / f'part-{i:03d}.bin'; p.write_bytes(block)
            files.append(describe_file(p, target)); i += 1
    return files

def target_is_replaceable(target: Path, replace: bool):
    if target.is_symlink(): raise BundleError('出力先のsymlinkは使用しません。')
    if not target.exists(): return
    if not target.is_dir(): raise BundleError('出力先はディレクトリである必要があります。')
    entries = list(target.iterdir())
    placeholder = len(entries) == 1 and entries[0].name == 'manifest.json' and read_json(entries[0]).get('status') == 'missing'
    if entries and not placeholder and not replace:
        raise BundleError('出力先に既存モデル/ファイルがあります。別の--outを選ぶか、意図した置換なら--replaceを明示してください。')

def publish_directory(staging: Path, target: Path, replace=False):
    target_is_replaceable(target, replace)
    backup = target.with_name(target.name + '.previous-' + uuid.uuid4().hex)
    existed = target.exists()
    if existed: os.replace(target, backup)
    try: os.replace(staging, target)
    except BaseException:
        if existed: os.replace(backup, target)
        raise
    if existed: shutil.rmtree(backup)

def public_metadata(source):
    spec = source['embedding_spec']
    reference = dict(spec['model'])
    if reference.get('kind') == 'local': reference['repo'] = 'local-checkpoint-sha256'
    m = source.get('metrics') or {}
    return {
        'source_manifest_sha256': fingerprint(source),
        'embedding_model': reference,
        'counts': source.get('counts'),
        'best_alpha': m.get('best_alpha'),
        'test_agreement_with_teacher': m.get('test'),
        'baseline_agreement': m.get('baselines'),
        'warnings': m.get('warnings', []),
        'japanese_accuracy': 'not established by export parity tests',
        'teacher_spec_sha256': fingerprint(source.get('teacher_spec')),
        'training_versions': spec.get('versions', {}),
    }
