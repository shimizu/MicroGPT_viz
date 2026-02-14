# `src/microgpt.js` 読解ガイド

このドキュメントは、JavaScriptは使いこなせるが、AI/機械学習の仕組みには詳しくない技術者向けです。対象コードは `src/microgpt.js` です。

## 1. まず結論: このファイルは何をしているか

`src/microgpt.js` は、依存ライブラリなしで次の3つを1ファイルに実装しています。

1. 最小の自動微分エンジン（`Value` クラス）
2. GPT風の1層Transformer（Attention + MLP）
3. 学習ループ（損失計算、逆伝播、Adam更新）と文字列生成

つまり「AIモデルの中身」をブラックボックスにせず、数式をそのままJavaScriptオブジェクトで追える教材実装です。

## 2. AI側の最小用語

- トークン: モデルが扱う最小単位。この実装では「1文字」。
- 語彙（vocab）: 取りうるトークンの一覧。`names.txt` のユニーク文字 + `BOS`。
- 埋め込み（embedding）: トークンIDを実数ベクトルに変換した表現。
- logits: 各次トークン候補に対する生スコア（確率ではない）。
- softmax: logitsを確率分布に変換する関数。
- 損失（loss）: 予測がどれだけ外れているか。ここでは交差エントロピー。
- 逆伝播（backprop）: lossから各重みへの影響（勾配）を逆向きに計算。
- Adam: 勾配を使って重みを更新する最適化アルゴリズム。
- Attention: 現在位置が過去のどの位置をどれだけ参照するかを学習する仕組み。

## 3. 実行時の全体フロー

`trainAndGenerate()` が入口です。

1. データ読み込み (`loadDataset`)
2. 文字語彙化（文字→ID）
3. 重み初期化（`stateDict`）
4. 学習ループ
   1. 1つの名前を取り出す
   2. 1文字ずつ `gpt(...)` で次文字予測
   3. loss平均
   4. `loss.backward()`
   5. Adamで全重み更新
5. 学習後、`BOS` から文字を順次サンプリングして新しい名前を生成

## 4. 重要関数と役割

- `SeededRandom`
  - 再現性のある乱数、shuffle、重み付きサンプリング。
- `Value`
  - スカラー計算 + 計算グラフ + 勾配保持。
  - `add/mul/pow/log/exp/relu` が全て微分情報を持つ。
  - `backward()` でトポロジカル順に勾配伝播。
- `matrix(nout, nin)`
  - `Value` を要素に持つ重み行列をガウス初期化。
- `linear(x, w)`
  - 全結合層（行列×ベクトル）。
- `softmax(logits)`
  - 確率化（数値安定化あり）。
- `rmsnorm(x)`
  - 活性のスケールを正規化して学習を安定化。
- `gpt(tokenId, posId, keys, values, stateDict)`
  - 1トークン分のTransformer forward。
  - 埋め込み→Attention→MLP→logits。
- `trainAndGenerate(onStep, numSteps)`
  - 学習全体 + 生成。

## 5. `gpt()` の中身（最重要）

`gpt()` は「現在トークン1個」を入力し、次トークン候補の `logits` を返します。

1. 入力表現
   - `wte[tokenId]`（文字の意味） + `wpe[posId]`（位置情報）
2. Attentionブロック
   - `q`, `k`, `v` を線形変換で作る
   - 過去 `k` との内積でスコア
   - `softmax` で参照重み
   - 重み付き和で文脈ベクトル
   - 残差接続で元入力を加算
3. MLPブロック
   - 次元拡張 → ReLU → 次元圧縮
   - 残差接続
4. `lm_head` で語彙次元へ射影して `logits`

### KVキャッシュについて

`keys`/`values` は過去トークンの `k`/`v` を保持します。
これで、各時刻に過去トークンを再計算せずにAttentionできます。

## 6. 学習ループの要点

各ステップで1単語（例: `anna`）を次のように学習します。

- 入力列: `[BOS, a, n, n, a]`
- 教師列: `[a, n, n, a, BOS]`

各位置で `-log(p(正解文字))` を取り、平均したものがloss。
`loss.backward()` 後に、全パラメータをAdamで更新します。

この実装はミニバッチではなく「1サンプルずつ更新」です（SGDに近い運用）。

## 7. コードを読む順番（推奨）

1. `Value` クラス
2. `linear`, `softmax`, `rmsnorm`
3. `gpt`
4. `trainAndGenerate`

この順番だと「微分の土台」→「層の計算」→「モデル本体」→「学習」の依存関係通りに理解できます。

## 8. 実装上の割り切り（本番LLMとの違い）

- 文字レベル（サブワードではない）
- 1層・小次元（`N_EMBD=16`, `N_LAYER=1`）
- `Value`ベースのスカラー演算で遅い（教育目的）
- バッチ処理なし、GPU最適化なし

一方で、概念的には本格的なTransformerの主要要素（Embedding, Attention, MLP, Residual, Normalization, Cross-Entropy, Adam）を押さえています。

## 9. 変更時に壊れやすいポイント

- `N_EMBD % N_HEAD === 0` が必須（`HEAD_DIM`計算のため）
- `BLOCK_SIZE` を超える系列は切り詰められる
- `softmax` と `log()` は数値安定性に敏感
- `Value` 演算を通常の `number` に混在させると勾配が切れる

## 10. 最短で理解確認するチェックリスト

- `loss.backward()` がどの `Value` に勾配を配るか説明できる
- `gpt()` 内で `q/k/v` の役割を説明できる
- なぜ残差接続を2回使うか説明できる
- 生成時の `temperature` を上げると何が起きるか説明できる

以上を説明できれば、`src/microgpt.js` の中核は理解できています。
