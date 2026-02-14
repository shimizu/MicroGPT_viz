# Transformerの説明（`microgpt`対応）

この文書は、`src/microgpt.js` を読むためのTransformer解説です。

## 1. Transformerは何を解決する仕組みか

テキストは「前後関係（文脈）」で意味が決まります。
Transformerは、各位置のトークンが他の位置を参照して文脈を取り込めるようにしたモデルです。

従来RNNのような逐次依存を弱め、並列計算しやすい点が強みです。

## 2. 中核はSelf-Attention

Self-Attentionは「この位置を理解するために、過去のどこを見るべきか」を重みで決めます。

各トークンから3種類のベクトルを作ります。

- Query (`Q`): 何を探しているか
- Key (`K`): 自分が何を持っているか
- Value (`V`): 実際の情報

処理は次です。

1. `Q` と全 `K` の類似度を計算（内積）
2. `softmax` で重みに変換
3. その重みで `V` を加重平均

式としては `softmax(QK^T / sqrt(d)) V` です。

## 3. Multi-Head Attention

Attentionを1回だけでなく複数ヘッドで並行実行します。
ヘッドごとに異なる観点の関係（語尾、子音パターン、位置依存など）を捉えやすくなります。

`microgpt` では:

- `N_EMBD = 16`
- `N_HEAD = 4`
- `HEAD_DIM = 4`

で、16次元を4ヘッドに分割しています。

## 4. Transformerブロックの標準構成

1層のTransformerブロックは概ね次の流れです。

1. 正規化
2. Multi-Head Attention
3. 残差接続
4. 正規化
5. MLP（2層全結合 + 活性化）
6. 残差接続

残差接続は「元の情報を直通させる」ため、学習安定化に重要です。

## 5. GPTでの制約（Causal）

GPTは次トークン予測モデルなので、未来トークンを見てはいけません。
このためAttentionは「過去と現在のみ」を参照する因果制約つきです。

`microgpt` では1トークンずつ順に処理し、`keys/values` を蓄積する形で因果性を実現しています。

## 6. `microgpt.js` でどこがTransformerか

- 埋め込み: `wte`, `wpe`（`gpt()` 冒頭）
- Attention用射影: `attn_wq`, `attn_wk`, `attn_wv`
- ヘッド結合後の射影: `attn_wo`
- MLP: `mlp_fc1`, `mlp_fc2`
- 正規化: `rmsnorm()`
- 残差接続: `x = x + residual`

中心は `src/microgpt.js` の `gpt(...)`（`src/microgpt.js:234`）です。

## 7. 最低限の読み方

`gpt()` を次の順で追うと理解しやすいです。

1. `tokEmb + posEmb`
2. `q/k/v` 作成
3. ヘッドごとの `attnLogits -> softmax -> weighted sum`
4. `attn_wo` と残差
5. `mlp_fc1 -> relu -> mlp_fc2` と残差
6. `lm_head` で logits

## 8. よくある誤解

- 「Attentionだけでモデル全部」ではない
  - 実際はMLPと残差・正規化が重要。
- 「softmaxの最大値が答え」だけではない
  - 生成では温度やサンプリング戦略で出力多様性が変わる。
- 「小さい実装は本質が違う」わけではない
  - 規模は違っても骨格は同じ。

## 9. 1分まとめ

Transformerは、各トークンが他トークンを動的に参照するSelf-Attentionを核に、
MLP・正規化・残差接続を組み合わせた層を重ねる設計です。
GPTはそれを「次トークン予測」に特化して使うモデルです。
