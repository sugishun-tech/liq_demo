#!/usr/bin/env python3
"""Export an existing liq v0.3 student and its pinned E5 checkpoint for the browser.

No synthetic labels, fake weights, remote-code loading, or implicit retraining.
The output is committed only after PyTorch/ONNX numeric parity succeeds.
"""
from __future__ import annotations
import argparse
import fnmatch
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

from bundle import (BundleError, FORMAT, PATTERNS, digest, describe_file, fingerprint,
                    load_student, normalize, public_metadata, publish_directory,
                    read_json, split_file, target_is_replaceable, write_json)

# Original, non-training sentences. Scores are computed by the user's actual model.
PARITY_TEXTS = [
    'The sensor moved, so the changed reading alone does not establish a fault.',
    'Nice weather.',
    '測定値は変わった。しかし、センサーも移動したため、故障とは断定できない。',
    '今日は晴れ。',
    'A\tsecond  measurement\nwould help.',
    'Cafe\u0301 と カ\u3099ラス 🧑🏽‍💻。',
    '相関と原因は同じではない。 <mask> <s> </s>',
    '',
]

ROOT = Path(__file__).resolve().parents[1]

def log(message): print(message, file=sys.stderr, flush=True)

def resolve_encoder(spec, offline, encoder_dir):
    ref = spec['model']
    if ref['kind'] == 'local':
        directory = Path(encoder_dir or ref['repo']).expanduser()
        if ref.get('subfolder'): directory /= ref['subfolder']
        if not directory.is_dir(): raise BundleError('学習時のローカルE5が見つかりません。--encoder-dirで同一内容のコピーを指定してください。')
        files = [p for p in sorted(directory.rglob('*')) if p.is_file() and any(fnmatch.fnmatch(p.relative_to(directory).as_posix(), pat) for pat in PATTERNS)]
        hashes = {p.relative_to(directory).as_posix(): digest(p) for p in files}
        if fingerprint(hashes) != ref['revision']: raise BundleError('ローカルE5のハッシュが学習時と一致しません。')
        return directory
    if encoder_dir: raise BundleError('Hub由来のモデルは固定リビジョンから解決します。--encoder-dirはlocal由来専用です。キャッシュ済みの場合は--offlineを指定してください。')
    from huggingface_hub import snapshot_download
    directory = Path(snapshot_download(repo_id=ref['repo'], revision=ref['revision'],
        allow_patterns=PATTERNS, local_files_only=offline, max_workers=2))
    if directory.name != ref['revision']: raise BundleError('取得モデルの固定リビジョンを確認できません。')
    return directory

def check_versions(spec):
    actual = {}
    for name in ('torch', 'transformers', 'numpy'):
        installed = importlib.metadata.version(name)
        expected = spec.get('versions', {}).get(name)
        if expected != installed:
            raise BundleError(f'{name}が学習時と異なります: 学習={expected}, 現在={installed}。v0.3で学習したvenvを有効化してください。')
        actual[name] = installed
    for name in ('onnx', 'onnxruntime'): actual[name] = importlib.metadata.version(name)
    return actual

def make_wrapper(model):
    import torch
    class Encoder(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__(); self.backbone = backbone
        def forward(self, input_ids, attention_mask, token_type_ids):
            h = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids).last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp_min(1)
            return torch.nn.functional.normalize(pooled, p=2, dim=1)
    return Encoder(model).eval()

def make_inputs(tokenizer, texts, max_length):
    import torch
    batch = tokenizer(['query: ' + normalize(t) for t in texts], padding=True, truncation=False, return_tensors='pt')
    if batch['input_ids'].shape[1] > max_length: raise BundleError('変換確認文が学習モデルの上限を超えます。')
    batch['token_type_ids'] = batch.get('token_type_ids', torch.zeros_like(batch['input_ids']))
    return {k: batch[k] for k in ('input_ids', 'attention_mask', 'token_type_ids')}

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--student', type=Path, required=True, help='v0.3 student/ またはmodel.json')
    p.add_argument('--out', type=Path, default=ROOT / 'docs/model')
    p.add_argument('--offline', action='store_true')
    p.add_argument('--encoder-dir', type=Path)
    p.add_argument('--replace', action='store_true')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--temp-dir', type=Path, default=ROOT / '.cache/export')
    p.add_argument('--debug', action='store_true')
    args = p.parse_args(argv)
    staging = None
    try:
        if args.threads < 1: raise BundleError('--threadsは1以上です。')
        source, coef, bias = load_student(args.student)
        if args.out.is_symlink(): raise BundleError('出力先のsymlinkは使用しません。')
        target = args.out.resolve()
        target_is_replaceable(target, args.replace)
        args.temp_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault('TMPDIR', str(args.temp_dir.resolve()))
        os.environ.setdefault('USE_TF', '0'); os.environ.setdefault('USE_FLAX', '0')
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
        from cpu_only import require_cpu
        require_cpu()
        versions = check_versions(source['embedding_spec'])
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(args.threads)
        model_dir = resolve_encoder(source['embedding_spec'], args.offline, args.encoder_dir)
        cfg = read_json(model_dir / 'config.json')
        if cfg.get('model_type') != 'bert' or cfg.get('hidden_size') != 384:
            raise BundleError('対応外のE5アーキテクチャです。')
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, use_fast=True, trust_remote_code=False)
        if type(tokenizer).__name__.replace('Fast', '') != 'XLMRobertaTokenizer':
            raise BundleError('XLMRobertaTokenizer以外は対応しません。')
        model = AutoModel.from_pretrained(str(model_dir), local_files_only=True,
            use_safetensors=True, trust_remote_code=False).to('cpu').eval()
        wrapper = make_wrapper(model)
        maximum = source['embedding_spec']['max_length']
        started = time.perf_counter()
        cases = []
        log('学習時のE5から数値照合データを計算しています。')
        with torch.inference_mode():
            for text in PARITY_TEXTS:
                inputs = make_inputs(tokenizer, [text], maximum)
                vector = wrapper(**inputs).cpu().numpy()[0].astype(np.float64)
                cases.append({'text': text, 'input_ids': inputs['input_ids'][0].tolist(),
                              'embedding': vector.tolist(), 'raw_score': float(vector @ coef + bias)})
        # Use eager attention for a portable ONNX graph; compare against the original path above.
        if hasattr(model, 'set_attn_implementation'): model.set_attn_implementation('eager')
        else: model.config._attn_implementation = 'eager'
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.model-build-', dir=target.parent))
        with tempfile.TemporaryDirectory(prefix='onnx-', dir=args.temp_dir) as temp:
            graph = Path(temp) / 'encoder.onnx'
            example = make_inputs(tokenizer, ['An example.', '別の長さの例文です。'], maximum)
            log('E5 + masked mean + L2正規化をONNX FP32へ変換しています。')
            with torch.inference_mode():
                torch.onnx.export(wrapper, tuple(example.values()), str(graph),
                    input_names=list(example), output_names=['embedding'],
                    dynamic_axes={**{k: {0: 'batch', 1: 'sequence'} for k in example}, 'embedding': {0: 'batch'}},
                    opset_version=17, do_constant_folding=True, dynamo=False, external_data=False)
            onnx.checker.check_model(str(graph))
            options = ort.SessionOptions(); options.intra_op_num_threads = args.threads
            session = ort.InferenceSession(str(graph), sess_options=options, providers=['CPUExecutionProvider'])
            expected_names = ['input_ids', 'attention_mask', 'token_type_ids']
            if [x.name for x in session.get_inputs()] != expected_names:
                raise BundleError('ONNX入力が期待した3入力と一致しません。')
            max_emb = max_raw = 0.0
            log('PyTorchとONNXの埋め込み・回帰値を照合しています。')
            for case in cases:
                inputs = make_inputs(tokenizer, [case['text']], maximum)
                out = session.run(['embedding'], {k: v.numpy() for k, v in inputs.items()})[0][0].astype(np.float64)
                if out.shape != (384,) or not np.isfinite(out).all(): raise BundleError('ONNXの出力が不正です。')
                max_emb = max(max_emb, float(np.max(np.abs(out - case['embedding']))))
                max_raw = max(max_raw, abs(float(out @ coef + bias) - case['raw_score']))
            # Test dynamic batch length and padding, not just single unpadded sequences.
            batch = make_inputs(tokenizer, [c['text'] for c in cases[:4]], maximum)
            out = session.run(['embedding'], {k: v.numpy() for k, v in batch.items()})[0]
            for i in range(4):
                max_emb = max(max_emb, float(np.max(np.abs(out[i] - cases[i]['embedding']))))
                max_raw = max(max_raw, abs(float(out[i].astype(np.float64) @ coef + bias) - cases[i]['raw_score']))
            del session
            if max_emb > 0.0002 or max_raw > 0.02:
                raise BundleError(f'変換誤差が許容範囲を超えました。埋め込み={max_emb}, rawスコア={max_raw}。成果物は公開しません。')
            log('ONNXを48MiB以下のファイルへ分割しています。')
            chunks = split_file(graph, staging)
            graph_bytes = graph.stat().st_size
        token_dir = staging / 'tokenizer'; token_dir.mkdir()
        for name in ('tokenizer.json', 'tokenizer_config.json'):
            if not (model_dir / name).is_file(): raise BundleError(f'Tokenizerファイルがありません: {name}')
            shutil.copy2(model_dir / name, token_dir / name)
        # Do not re-save/rewrite tokenizer: preserve the exact training-time files.
        tconfig = read_json(token_dir / 'tokenizer_config.json')
        if tconfig.get('tokenizer_class', '').replace('Fast', '') != 'XLMRobertaTokenizer':
            raise BundleError('ブラウザ用Tokenizerクラスを特定できません。')
        parity_path = staging / 'parity.json'
        write_json(parity_path, {'format': 'text-iq-parity-v1', 'cases': cases,
            'note': 'Numerical implementation equivalence only; NOT a model quality benchmark.'})
        parity = {**describe_file(parity_path, staging), 'python_onnx_passed': True,
                  'embedding_atol': 0.0002, 'score_atol': 0.02,
                  'measured_max_embedding_error': max_emb, 'measured_max_raw_score_error': max_raw}
        manifest = {'format': FORMAT, 'schema_version': 1, 'status': 'ready',
            'features': {'adapter': 'e5-masked-mean-l2-v1', 'prefix': 'query: ',
                         'normalization': 'nfc-python-whitespace', 'dimension': 384,
                         'max_length': maximum},
            'ridge': {'coef': coef.tolist(), 'intercept': bias, 'clip': [0, 100], 'iq_offset': 70, 'iq_scale': .6},
            'encoder': {'precision': 'fp32', 'inputs': expected_names, 'output': 'embedding',
                        'bytes': graph_bytes, 'chunks': chunks},
            'tokenizer': {'files': [describe_file(f, staging) for f in sorted(token_dir.iterdir())]},
            'parity': parity, 'training': public_metadata(source),
            'export': {'versions': versions, 'seconds': time.perf_counter() - started},
            'warning': 'Arbitrary text complexity proxy, not human IQ. Occlusion is sensitivity, not causal reasoning.'}
        manifest['id'] = fingerprint(manifest)
        write_json(staging / 'manifest.json', manifest)
        publish_directory(staging, target, args.replace); staging = None
        log(f'変換完了: {target} / ONNX {graph_bytes / 1024 ** 2:.1f} MiB / {len(chunks)}分割')
        log('次に python tools/check.py と python tools/serve.py でブラウザの数値照合を実行してください。')
        return 0
    except (Exception, KeyboardInterrupt) as e:
        if args.debug and not isinstance(e, KeyboardInterrupt): raise
        print(f'ERROR: {e or "中断しました"}', file=sys.stderr)
        return 2
    finally:
        if staging is not None: shutil.rmtree(staging, ignore_errors=True)

if __name__ == '__main__': raise SystemExit(main())
