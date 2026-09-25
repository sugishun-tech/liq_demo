#!/usr/bin/env bash
# Run in the existing CPU training venv. This does not train or install Torch.
set -euo pipefail
if [[ $# -lt 1 ]]; then
    printf 'Usage: bash prepare.sh /absolute/path/to/student [--threads 4] [--offline]\n' >&2
    exit 2
fi
PYTHON="${PYTHON:-python}"
STUDENT="$1"; shift
if [[ "$STUDENT" != /* ]]; then STUDENT="$PWD/$STUDENT"; fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p .cache/tmp
export TMPDIR="${TMPDIR:-$PWD/.cache/tmp}"
mkdir -p "$TMPDIR"
export PIP_NO_CACHE_DIR=1
export USE_TF=0 USE_FLAX=0 HF_HUB_DISABLE_TELEMETRY=1
# Fail before dependency changes when the caller has not selected a CPU training environment.
"$PYTHON" - "$STUDENT" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, 'tools')
try:
    from cpu_only import require_cpu
    from bundle import load_student
    from export_web import check_versions
    source, _, _ = load_student(Path(sys.argv[1]))
    version = require_cpu()
    # Check only training packages before ONNX is installed.
    import importlib.metadata as md
    for name in ('torch', 'transformers', 'numpy'):
        expected = source['embedding_spec']['versions'].get(name)
        if md.version(name) != expected:
            raise RuntimeError(f'{name} の版が学習時と違います。学習に使ったvenvを有効化してください。')
    Path('.cache/constraints-existing-cpu.txt').write_text(f'torch=={version}\n')
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    print('laya-iq-distill v0.3 のCPU学習venvを有効にして実行してください。', file=sys.stderr)
    raise SystemExit(2)
PY
# --offline does not trigger pip or any network operation.
OFFLINE=0
for arg in "$@"; do if [[ "$arg" == '--offline' ]]; then OFFLINE=1; fi; done
if [[ "$OFFLINE" -eq 0 ]]; then
    "$PYTHON" -m pip install --no-cache-dir -c .cache/constraints-existing-cpu.txt -r requirements-export.txt
fi
"$PYTHON" tools/prepare.py --student "$STUDENT" "$@"
