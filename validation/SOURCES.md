# 一次資料

確認日：2026-09-25。以下は公開仕様を確認した資料であり、この環境での実モデル実行の証拠ではありません。

- GitHub Pagesの公開元設定。任意ブランチのルートまたはdocsを公開元にできる。`.nojekyll`による処理省略と、GitHub内部のデプロイ処理の区別。
  https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site
- GitHub Pagesの容量等の制限。
  https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits
- ONNX Runtime Webの配布ファイル。WASM専用import、必要バイナリ、同一originでの配信。
  https://onnxruntime.ai/docs/tutorials/web/deploy.html
- ONNX Runtime 1.22.0のパッケージexport。`ort.wasm.min.mjs` の指定を確認。
  https://raw.githubusercontent.com/microsoft/onnxruntime/v1.22.0/js/web/package.json
- Tokenizers.jsの公式README。依存ゼロのブラウザTokenizer、Tokenizer JSONとconfigからの構築、encodeのids。
  https://github.com/huggingface/tokenizers.js
- Tokenizers.js 0.2.0の公式タグにあるpackage.json。配布先とライセンス。
  https://raw.githubusercontent.com/huggingface/tokenizers.js/v0.2.0/package.json
- Tokenizers.jsのTokenizer API実装。
  https://raw.githubusercontent.com/huggingface/tokenizers.js/main/src/core/Tokenizer.ts
- E5の元モデル。
  https://huggingface.co/intfloat/multilingual-e5-small

モデルIDが同じであるだけの別ONNXへ置き換えず、元成果物が記録した固定リビジョンを変換する方針です。
