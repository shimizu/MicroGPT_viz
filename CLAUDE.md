# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

Andrej Karpathyの`microgpt.py`をJavaScriptに移植した、依存ライブラリなしのGPT実装。名前データセット（`public/data/names.txt`、約32,000件）から学習し、新しい名前を生成するデモ。ブラウザとNode.jsの両方で動作する。

## 開発コマンド

- `npm run dev` — 開発サーバー起動（自動でブラウザが開く）
- `npm run build` — 本番ビルド（`dist/` に出力）
- `npm run preview` — ビルド成果物のプレビュー
- `npm run deploy` — GitHub Pagesへデプロイ（`gh-pages -d dist`）

## アーキテクチャ

### Vite設定の注意点
- **rootが`src/`**に設定されている（`vite.config.js`）。エントリーポイントは`src/index.html`
- `publicDir`は`../public`（プロジェクトルートの`public/`）
- ビルド出力先は`../dist`
- `base: "./"` — 相対パスでアセットを参照

### コアアルゴリズム（`src/index.js`）
単一ファイルに全アルゴリズムが含まれる：

- **`SeededRandom`** — シード付き乱数生成（再現性のため）。Box-Muller変換によるガウス分布サポート
- **`Value`** — 自動微分（Autograd）のためのスカラー値クラス。前方計算と逆伝播（`backward()`）を実装
- **`gpt()`** — GPTのフォワードパス。トークン埋め込み→位置埋め込み→RMSNorm→Multi-Head Attention→MLP
- **`main()`** — データ読み込み→文字レベルトークナイザ構築→パラメータ初期化→Adamオプティマイザによる学習→推論（名前生成）

### ハイパーパラメータ（定数）
`N_EMBD=16`, `N_HEAD=4`, `N_LAYER=1`, `BLOCK_SIZE=16` — 小規模なためブラウザで実行可能

### ブラウザ/Node.js両対応
- ブラウザ: `window.runMicroGPT`にmain関数を公開、`src/index.html`のボタンから実行
- Node.js: 直接`main()`を実行
- データ読み込み（`loadDataset()`）で環境判定して`fetch`/`fs`を切り替え

## 言語・地域設定
- コミットメッセージ、コメント、ドキュメントは日本語で記述
