# `src/microgpt.js` リーディングマップ

`src/microgpt.js` を追うときの目印です。まず `docs/microgpt.md` を読んでから、この対応表でコード本体に戻る使い方を想定しています。

## セクション対応

- 疑似乱数生成: `class SeededRandom`（`src/microgpt.js:14`）
- データ読み込み: `loadDataset`（`src/microgpt.js:62`）
- 自動微分エンジン: `class Value`（`src/microgpt.js:75`）
- ハイパーパラメータ定義: `N_EMBD` など（`src/microgpt.js:173`）
- 重み初期化: `matrix`（`src/microgpt.js:185`）
- 線形層: `linear`（`src/microgpt.js:199`）
- 確率化: `softmax`（`src/microgpt.js:207`）
- 正規化: `rmsnorm`（`src/microgpt.js:217`）
- モデル本体forward: `gpt`（`src/microgpt.js:234`）
- 学習と生成: `trainAndGenerate`（`src/microgpt.js:313`）
- エクスポート: `export { ... }`（`src/microgpt.js:477`）

## 先に読むべきポイント

- 勾配の流れを確認したい: `Value.backward()` 周辺
- Transformerの中身を確認したい: `gpt()` のAttention/MLPブロック
- 学習アルゴリズムを確認したい: `trainAndGenerate()` の `loss.backward()` と Adam 更新ループ
- 推論時のランダム性を確認したい: `temperature` と `rng.choices(...)`
