# MicroGPT - Pure JavaScript Implementation

Andrej Karpathyの`microgpt.py`をJavaScriptに移植したバージョンです。
外部ライブラリなしの純粋なJavaScriptで、GPTの学習と推論を実装しています。

### Demo
https://shimizu.github.io/MicroGPT_viz/

## 特徴

- **最小限の依存**: GPTアルゴリズム本体は依存ライブラリなしの純粋なJavaScriptで実装（可視化にD3.jsを使用）
- **ブラウザ実行に特化**: 開発サーバー上で学習と可視化を実行
- **教育目的**: GPTの仕組みを理解するための最小限の実装
- **自動微分**: 計算グラフベースのバックプロパゲーション
- **Adam最適化**: モーメントベースの適応的学習率
- **リアルタイム可視化**: D3.jsによる損失曲線・Attention重み・予測確率・埋め込み空間の表示

## アーキテクチャ

- **モデル**: GPT-2スタイル（簡略版）
- **パラメータ数**: 約10,000
- **レイヤー数**: 1
- **アテンションヘッド**: 4
- **埋め込み次元**: 16
- **最大シーケンス長**: 16

## 実行方法

### ブラウザで実行（推奨）

```bash
npm install
npm run dev
```

開発サーバーが起動し、ブラウザで自動的に開きます。
「学習を開始」ボタンをクリックすると、学習の進捗とリアルタイム可視化が表示されます。
学習ステップ数はUIから変更可能です（デフォルト: 1,000、10〜5,000）。

### その他のコマンド

```bash
npm run build    # 本番ビルド（dist/ に出力）
npm run preview  # ビルド成果物のプレビュー
npm run deploy   # GitHub Pagesへデプロイ
```

## 学習の流れ

1. **データ準備**: 市区町村名データセット（1,756件）を読み込み
2. **トークン化**: 文字レベルでトークン化
3. **パラメータ初期化**: 重み行列をランダム初期化
4. **学習ループ** (デフォルト1,000ステップ、UIから変更可能):
   - 1文書をサンプリング
   - 順伝播で各位置の次トークンを予測
   - 交差エントロピー損失を計算
   - 逆伝播で勾配を計算
   - Adam最適化でパラメータ更新
5. **推論**: 学習済みモデルで新しい名前を生成（20サンプル）

## 出力例

```
step 1000 / 1000 | loss 1.8234

--- inference (new, hallucinated names) ---
sample  1: kayla
sample  2: jaxon
sample  3: emmeline
sample  4: karter
sample  5: brixton
...
```

## コードの主要部分

### 自動微分 (Value クラス)

```javascript
class Value {
    constructor(data, children = [], localGrads = []) {
        this.data = data;
        this.grad = 0;
        this._children = children;
        this._localGrads = localGrads;
    }
    
    backward() {
        // トポロジカルソートで計算順序を決定
        // 連鎖律を適用して勾配を伝播
    }
}
```

### GPTモデル

```javascript
function gpt(tokenId, posId, keys, values, stateDict) {
    // 1. トークン埋め込み + 位置埋め込み
    // 2. Multi-head Attention
    // 3. MLP (Feed Forward)
    // 4. 残差接続
    // 5. 次トークンのlogitsを返す
}
```

### Adam最適化

```javascript
// 1次モーメント: m
// 2次モーメント: v
// バイアス補正付きの適応的学習率更新
p.data -= lrT * mHat / (Math.sqrt(vHat) + epsAdam);
```

## Python版との違い

### 実装上の主な変更点

1. **乱数生成**: `random.seed()`の代わりに独自の`SeededRandom`クラスを実装
2. **配列操作**: Pythonのリスト内包表記をJavaScriptの`map()`/`reduce()`に変換
3. **演算子オーバーロード**: JavaScriptは演算子オーバーロード非対応のため、メソッド呼び出しに変更
   - `a + b` → `a.add(b)`
   - `a * b` → `a.mul(b)`
4. **非同期処理**: データ読み込みを`async/await`で実装

### 互換性

アルゴリズムとハイパーパラメータは元のPython版と完全に同一です。

## 学習のカスタマイズ

`src/index.js`の以下の定数を変更することで、モデルの挙動を調整できます:

```javascript
const N_EMBD = 16;      // 埋め込み次元を増やすとモデルの表現力が向上
const N_HEAD = 4;       // アテンションヘッド数
const N_LAYER = 1;      // レイヤー数を増やすと深いモデルに
const BLOCK_SIZE = 16;  // シーケンス長
const temperature = 0.5; // サンプリング温度（0-1: 低いほど確定的）
```

学習ステップ数はブラウザUIの入力欄から変更できます。

## 制限事項

- **速度**: 教育目的のため、効率より可読性を優先しています
- **メモリ**: ブラウザでは大規模なモデルは実行困難
- **データセット**: 名前データセット（英語）に特化

## ライセンス

元のPython実装: @karpathy (Andrej Karpathy)
JavaScript移植: 教育目的での利用を想定

## 参考リンク

- [元のPython実装](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
- [GPT-2論文](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
