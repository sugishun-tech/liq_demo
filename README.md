# text-iq-pages-static

**学習済みモデルを一度CPUで準備し、`docs/index.html` と周囲のファイルをそのままGitHub Pagesへ公開するプロジェクトです。** `.github`、独自CI、フロントエンドのビルド、npm install、サーバー側の推論は不要です。

**この配布ZIPには、学習済みモデルと第三者ランタイムの実ファイルは含まれていません。** この作成環境では外部配布先へのDNS接続が失敗し、取得できませんでした。HTML/CSS/JavaScript、CPU変換器、必要ランタイムを揃えるコマンド、テストを同梱しています。未配置では採点しません。完成済みの採点デモではなく、既存の `student/` から公開用ファイルを準備する版です。

## 公開方法

準備後に公開するのは、次の静的ファイルだけです。

```text
docs/
  index.html
  .nojekyll
  styles.css
  guide.html
  src/                   画面・Worker・回帰・ヒートマップ
  model/                 ONNX、Tokenizerのデータ、回帰係数
  vendor/                CPU用JavaScriptとWASM
  runtime-config.json
  vendor-lock.json
  licenses/
  THIRD_PARTY_NOTICES.txt
```

リポジトリの **Settings → Pages → Build and deployment → Source → Deploy from a branch** を選び、**Branch: `main` / Folder: `/docs` → Save** にします。`docs/` がそのままサイトのルートとして公開されます。`/repository/docs/` ではなく `/repository/` が入口になります。

リポジトリのルートへ `index.html` を置きたい場合は、**`docs/` の中身を丸ごと**専用の公開リポジトリのルートへコピーし、`main` / `/(root)` を選んでください。隠しファイル `.nojekyll` も必要です。

```bash
# 新しい、公開用のディレクトリへコピーする例。
# 実行前に、そのディレクトリに不要な古いモデルが残っていないことを確認します。
mkdir -p ../text-iq-public
cp -a docs/. ../text-iq-public/
```

GitHubが公開サーバーへ配置する内部処理は動きますが、利用者が `.github/workflows/` を作成・保守する必要はありません。GitHub全体のActions機能を無効にする指示ではありません。[公式の公開元設定](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)

`index.html` **単体**ではE5の重みなどがありません。`docs/` 一式なら、準備後はコピーとPages設定だけです。

## 初回だけ：CPUで必要ファイルを揃える

別プロジェクト `laya-iq-distill v0.3` で実際に学習した次のファイルが必要です。

```text
student/
  model.json
  ridge-<hash>.npz
```

**学習に使ったCPU版venvを有効にした状態**で、この新規プロジェクトのルートから実行します。旧プロジェクトに上書きしないでください。

```bash
source ../laya_iq_distill_v3/.venv/bin/activate
bash prepare.sh ../laya_iq_distill_v3/runs/en-10k/student --threads 4
```

このコマンドは次を行います。

1. CPU版Torch、元モデル、学習時のライブラリ版を確認します。
2. CPU用の `onnx` と `onnxruntime` だけを追加します。TorchやLayaはインストールしません。
3. Tokenizers.jsとONNX Runtime Webの必要ファイル6個を `docs/vendor/` に保存します。GPU版・JSEPのランタイムやnpmパッケージ全体は取得しません。
4. 学習時と同じ固定リビジョンのE5をCPUでONNXへ変換し、日英の固定例でPyTorchとの数値照合を行います。
5. ONNX・Tokenizerデータ・回帰係数を `docs/model/` に保存し、必要ファイル・ハッシュ・容量を検査します。

これは**公開前のモデル変換・ファイル配置**であり、GitHub上のビルドではありません。一度成功すれば、公開時にこのコマンドを実行する必要はありません。同じモデルでの再実行は、検証済みの既存ファイルを再利用します。別モデルへの更新は `--replace-model` を明示します。

既存venvにGPU用パッケージが混ざっている場合は停止します。他の用途の環境を破壊しないよう、自動アンインストールはしません。既存のCPU専用venvを選んでください。

### モデルキャッシュと一時容量

`TMPDIR` の既定値はこのプロジェクトの `.cache/tmp` です。CPU用でも元重み、ONNX変換中のファイル、完成したモデルを保存する容量は必要です。`/tmp` を変更してもディスク容量そのものは増えません。

学習時のキャッシュを使う場合は、その実際の場所を指定できます。

```bash
export HF_HOME="/absolute/path/to/laya_iq_distill_v3/.cache/huggingface"
bash prepare.sh /absolute/path/to/runs/en-10k/student --threads 4
```

すべて取得済みでネットを使わず再確認する場合は `--offline` を付けます。pipも実行しません。未取得のランタイム・モデル・依存がある場合は失敗します。

### 旧デモでONNX変換まで済んでいる場合

再学習・ONNX再変換は不要です。旧 `site/model/` に `status: ready` のmanifestと実ファイルがあることを確認し、新規プロジェクトへコピーできます。

```bash
cp -a ../text-iq-pages/site/model/. docs/model/
python3 tools/vendor.py
python3 tools/check.py
```

この経路の準備はPython標準ライブラリだけで動きます。ブラウザ初期化時には新しいTokenizerも含めて数値照合します。不一致なら採点は有効になりません。旧 `.github`、旧 `site/vendor`、旧 `runtime-config.json` はコピーしないでください。

## ローカル確認

```bash
python3 tools/serve.py
```

表示されたlocalhostのURLを開きます。画面だけなら未準備でも表示できます。モデルの準備後は「モデルを読み込む」を押し、数値照合が通ったことを確認してから採点してください。点数だけでなく、短い文のヒートマップも確認します。

`tools/serve.py` は `docs/` を直接配信します。`dist/` やビルドは存在しません。`file://` でダブルクリックする使い方ではなく、HTTP/HTTPSで配信します。

```bash
# 公開前の任意の再検査。ファイルは書き換えません。
python3 tools/check.py
```

## CPU専用化した範囲

変換はCPU版PyTorchと `CPUExecutionProvider`。ブラウザは **ONNX Runtime WebのWASM専用版**と `executionProviders: ['wasm']` に固定しています。WebGPUの自動選択、GPUの初期化、GPU用WASMを除去しました。

ブラウザのTokenizerは、推論フレームワーク一式ではなく、依存ゼロの `@huggingface/tokenizers` を使います。PythonのTransformersは元のE5を読むために学習側venvのものを利用します。

GitHub Pagesで追加のCOOP/COEPヘッダーを必要としないよう、WASMは1スレッドです。画面操作と推論はWeb Workerで分離します。ヒートマップは区間数だけ推論を繰り返すので、文字単位では待ち時間が増えます。CPUの実測速度は提示していません。

## 通信と公開範囲

準備時にはモデルとライブラリの配布元へ接続します。**公開後の実行コードには外部CDN・推論APIへの参照はありません。** ランタイム・モデル・Tokenizerデータを同じサイトから取得します。入力文章は推論APIへ送らず、localStorageにも保存しません。通常のサイト配信リクエストは発生するので、完全オフラインPWAという意味ではありません。

`vendor-lock.json` の配布元URLは来歴の記録です。実行時の取得先ではありません。版固定のHTTPS取得後にSHA-256を記録し、以降はサイズとハッシュを確認します。これは上流の署名検証ではありません。

Gitへ登録するのは公開用コード・モデル・ランタイムです。学習DB、元のtweet、個人の評価文、venv、`.cache`、認証情報を登録しないでください。回帰係数とE5重みは閲覧者に配布するため、非公開にはできません。

## 容量

ONNXは48MiB以下に分割します。ファイル単位の制限に対応するもので、圧縮ではありません。GitHub Web画面のアップロードではなくGitを使ってください。モデルをGit LFSへ移す構成ではありません。

サイト合計950MB未満、単一ファイル100MiB未満を検査します。[GitHub Pagesの公式制限](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)に加え、モデル更新時のGit履歴増加や初回ダウンロード量にも注意してください。回帰器だけでなく、E5本体の容量とブラウザRAMが必要です。

## スコアとヒートマップ

`text_iq_proxy = 70 + 0.6 × clip(Ridge(E5(text)), 0, 100)`。文章に付ける任意の代理値で、人間のIQではありません。日本語を入力できますが、日本語での採点の妥当性を別に評価する必要があります。

ヒートマップは、語または書記素を1区間ずつ空白へ置き換え、クリップ前のスコア差を表示します。教師の思考過程、因果的な根拠、加法的な厳密寄与ではありません。差分の合計が元の点数になるわけでもありません。最大128区間、中止、JSON保存に対応しています。

## 検証

[検証記録](validation/VALIDATION.md)を参照してください。実モデルのONNX変換・WASM推論・本番Pages公開は未検証です。テスト用の人工モデルやランタイムを本物として同梱・採点する仕組みはありません。

```bash
node --test tests/*.test.mjs
python -m pytest -q tests/test_tools.py tests/test_static.py
```

Nodeとpytestは開発時のテスト専用です。公開時には不要です。

## ライセンス

新規コードはMITです。第三者ランタイムのライセンスは取得時に同梱し、E5と学習データの権利は別に扱います。[一次資料](validation/SOURCES.md)と公開ディレクトリ内の `THIRD_PARTY_NOTICES.txt` を参照してください。
