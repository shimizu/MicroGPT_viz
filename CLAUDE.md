# CLAUDE.md - プロジェクト情報

## プロジェクト概要
- **名前**: prejs-minimum-template
- **説明**: preJSプロジェクトのミニマムテンプレート
- **バンドラー**: Vite
- **言語**: JavaScript (ES6+), SCSS

## 開発環境
### 必要なコマンド
- **開発サーバー起動**: `npm run dev`
- **本番ビルド**: `npm run build`
- **プレビュー**: `npm run preview`
- **デプロイ**: `npm run deploy`

### ディレクトリ構造
```
/
├── src/
│   ├── index.js        # メインJavaScriptファイル
│   └── index.scss      # メインスタイルファイル
├── index.html          # エントリーポイント
├── vite.config.js      # Vite設定
└── package.json        # 依存関係・スクリプト
```

## 技術スタック
- **Vite**: 高速ビルドツール・開発サーバー
- **Babel**: ES6+トランスパイル（React preset含む）
- **Sass**: CSS拡張
- **Video.js**: 動画プレイヤーライブラリ
- **Legacy Plugin**: 古いブラウザサポート

## 開発時の注意事項
- 開発サーバーは自動でブラウザを開く設定
- ビルド出力は `dist/` ディレクトリ
- IE11以外の古いブラウザをサポート
- GitHub Pagesへのデプロイが可能

## 言語・地域設定
- **主要言語**: 日本語
- **コミットメッセージ**: 日本語で記述
- **コメント・ドキュメント**: 日本語で記述

## よく使用するワークフロー
1. **新機能開発**: `npm run dev` で開発サーバー起動
2. **ビルド確認**: `npm run build` でビルド、`npm run preview` で確認
3. **デプロイ**: `npm run deploy` でGitHub Pagesに公開