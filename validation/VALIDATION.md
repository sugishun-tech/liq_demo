# 検証記録

対象: text-iq-pages-static 2.0.1。作成日: 2026-09-25。

## 配布物と結論

GitHub Pagesのブランチ公開用に、HTML・CSS・JavaScriptを `docs/` へ直接配置した。`.github/`、公開時のビルド、dist生成を廃止し、変換とブラウザ推論をCPU専用に変更した。

**このZIPには実学習済みモデルと第三者ランタイムのバイナリは含めていない。** モデルmanifestとvendor-lockは `status: missing` であり、採点しない。実Layaによる学習、実E5のONNX変換、実Tokenizerによる照合、実WASM推論、本番GitHub Pages公開の成功は報告しない。外部取得を伴う初回prepare.shの完走も未検証。

## 成功した検査

- JavaScript: **65件成功**。前処理、Ridgeの数値処理、文字・語分割、置換感度、Tokenizer呼び出し・照合、同一サイトのランタイムURL制約、取得ファイルの検証、CPU専用経路など。
- Python: **57件成功**。元成果物の形式・ハッシュ、分割モデルの保存、実sklearnとJavaScriptの回帰値比較、CPU環境検査、公開ファイル検査、ランタイム取得と再利用・失敗時の扱い、直接HTTP配信など。
- ChromiumのDOM操作: **6グループ成功**。未準備状態の表示、ロード・採点・ヒートマップのイベント、書記素、XSSを避ける文字列描画、中止、エラー時の古い点数の消去など。
- Bash構文検査、Python compileall、Node構文検査。

配布ZIPを別ディレクトリへ展開した後も、JavaScript65件・Python57件・未準備画面の静的検査が成功した。

数値比較は人工的な384次元ベクトルと実sklearnのRidgeを使用した。小さなテスト用Torchモジュールでマスク付き平均プーリングを検査した。実E5や実文章の採点品質を評価したものではない。

`tests/test_static.py` ではPythonのHTTPサーバーを実際に起動し、ルートと `/repository/` 相当の配置で、公開ファイルをそのまま取得できることとMIMEを確認した。ビルドは行わない。このHTTP配信検査はブラウザの実モデル推論とは別。

ネットワーク取得の単体テストは明示的なテスト用の応答を使い、ダウンロード対象がCPUランタイムの6ファイルだけであること、ハッシュ不一致・余分なファイル・シンボリックリンクの拒否、offline時に取得しないことを確認した。これらのfixtureは公開ディレクトリへ同梱していない。

## 実際に失敗した試行

`python tools/vendor.py` による実配布先からの取得は、最初のTokenizerファイルでDNS解決エラーとなった。`validation/runtime-download.txt` に記録する。未取得のランタイムをreadyにする処理はなく、元のmissing状態を保った。

Chromiumの通常のHTTP/Worker結合試験は、環境の管理ポリシーによる `net::ERR_BLOCKED_BY_ADMINISTRATOR` でページを開けなかった。ポリシーを変更していない。ログは `browser-http.txt`。

DOM検査ではabout:blankにソースを投入し、fetch・Worker・ダウンロードをテスト用実装へ置き換えた。これはHTTP、CSP、WASM、実Worker、実Tokenizer/E5を通した試験の代わりではない。`browser-dom.txt` に区別を記録した。プレビュー画像はこの未準備状態であり、実モデルの採点例ではない。

通常の `python tools/check.py` は学習済みモデル未配置を理由に終了コード2で停止した。`--allow-missing-model` でのみ未配置案内画面を確認できる。存在しないstudentを指定したprepare.shもpipを実行する前に終了コード2で停止した。

## 実行環境とログ

Python 3.13.5、Node.js 22.16.0、Torch 2.10.0+cpu、NumPy 2.3.5、scikit-learn 1.8.0。新規のクリーンvenvではなく、既存の検証環境を利用した。モデル変換用のtransformers/onnx/onnxruntimeと実モデル重みは未導入。

- `js-tests.txt`: JavaScript検査結果
- `python-tests.txt`: Python検査結果
- `browser-dom.txt`: DOM検査の成功範囲
- `browser-http.txt`: 通常ブラウザ結合検査の失敗
- `runtime-download.txt`: 実ランタイム取得の失敗
- `missing-student.txt`, `unprepared-check.txt`: 不足ファイル時の停止
- `archive-tests.txt`: 配布ZIPの別ディレクトリ展開後の再検査

再実行:

```bash
node --test tests/*.test.mjs
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m pytest -q tests/test_tools.py tests/test_static.py
```

## 実環境で残る確認

学習済みstudentを指定してCPU版venvで `bash prepare.sh /path/to/student --threads 4` を完了させ、`python tools/serve.py` で表示する。実ブラウザのモデル読込・数値照合・短文採点・ヒートマップを確認した後で `docs/` をPagesへ公開する。

変換器はPyTorchとPython ONNX Runtimeを照合し、ブラウザはToken IDs・埋め込み・回帰値を照合するコードを備える。失敗時に許容誤差を無根拠に緩めたり、ダミーの採点へ切り替えたりはしない。この照合が成功しても、IQという尺度の妥当性や日本語の採点精度が検証されたことにはならない。
