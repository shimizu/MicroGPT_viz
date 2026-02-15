# 開発者ガイド — MicroGPT で学ぶ・拡張する

このドキュメントは、MicroGPT のコードベースを土台にして **GPT の仕組みを実験的に学びたい、あるいは機能を拡張したい** 開発者向けのガイドです。

「読んで終わり」ではなく「動かして壊して理解する」ことを目指しています。

---

## 目次

1. [開発環境のセットアップ](#1-開発環境のセットアップ)
2. [コードの全体構造](#2-コードの全体構造)
3. [最初に試す実験 5 選](#3-最初に試す実験-5-選)
4. [ハイパーパラメータの変更と影響](#4-ハイパーパラメータの変更と影響)
5. [アーキテクチャの拡張](#5-アーキテクチャの拡張)
6. [データセットの差し替え](#6-データセットの差し替え)
7. [可視化パネルの追加](#7-可視化パネルの追加)
8. [デバッグと問題解決](#8-デバッグと問題解決)
9. [パフォーマンス改善のヒント](#9-パフォーマンス改善のヒント)
10. [発展課題](#10-発展課題)

---

## 1. 開発環境のセットアップ

```bash
# クローンと依存インストール
git clone <repository-url>
cd MicroGPT_viz
npm install

# 開発サーバー起動（ブラウザが自動で開く）
npm run dev
```

### Vite設定の注意点

このプロジェクトでは `vite.config.js` の `root` が `src/` に設定されています。

```javascript
// vite.config.js
export default defineConfig({
  root: 'src',           // ← エントリーポイントは src/index.html
  publicDir: '../public', // ← データセットは public/data/ 以下
  build: { outDir: '../dist' }
})
```

新しいファイルを追加する場合、この構成を意識してください。

- HTML/JS のソース → `src/` 以下
- 静的アセット（データファイル等）→ `public/` 以下

---

## 2. コードの全体構造

### 2.1 ファイル構成

```
src/
├── index.html         UI とスタイル、学習パラメータのコントロール
├── microgpt.js        GPT コアアルゴリズム（自動微分 + モデル + 学習ループ）
└── viz/
    ├── vizManager.js   全チャートの統合マネージャ
    ├── lossCurve.js    損失曲線
    ├── embeddingScatter.js  埋め込み空間の PCA 散布図
    ├── attentionHeatmap.js  Attention 重みヒートマップ
    ├── tokenProbs.js   次トークン予測確率
    ├── residualStream.js    残差ストリーム変遷
    ├── headOutput.js   ヘッド出力ヒートマップ
    ├── mlpActivation.js     MLP ニューロン活性化
    ├── gradientFlow.js 勾配フロー
    └── pca.js          べき乗法による簡易 PCA
```

### 2.2 データの流れ

```
index.html（UIボタン押下）
  → trainAndGenerate(callback, numSteps, numLayers, seed)
    → 各ステップで callback(data) を呼ぶ
      → VizManager が data を受け取り、各チャートに分配
```

拡張するときは、この `callback` に渡す `data` オブジェクトにフィールドを追加し、対応する可視化コンポーネントを `viz/` に追加するのが基本パターンです。

### 2.3 microgpt.js の内部構造

```
SeededRandom        シード付き乱数（再現性のため）
Value               自動微分エンジン（計算グラフ + 逆伝播）
matrix()            重み行列の初期化
linear()            線形変換（行列×ベクトル）
softmax()           確率分布への変換
rmsnorm()           RMS 正規化
gpt()               1 トークンの Transformer フォワードパス
trainAndGenerate()   学習ループ + 推論（名前生成）
```

何かを変更する場合、まず **どのレイヤーに手を入れるか** を上の一覧で確認してください。

---

## 3. 最初に試す実験 5 選

コードを読むだけでは理解は深まりません。以下の実験を順に試すことで、各パーツの役割を体感できます。

### 実験 1: 学習率を極端に変えてみる

`src/microgpt.js` の `learningRate` を変更します。

```javascript
// src/microgpt.js:376
const learningRate = 0.01;  // デフォルト
```

| 変更値 | 予想される挙動 |
|--------|-------------|
| `0.1`  | 損失が乱高下し、収束しない（勾配爆発気味） |
| `0.001` | 損失がゆっくり下がる。1000ステップでは収束しきらない |
| `0.0001` | ほとんど学習が進まない |

**確認ポイント:** 損失曲線と勾配フローの変化を観察してください。学習率が高いと勾配フローのバーが激しく動き、低いと安定するが損失が下がらない。

### 実験 2: Attention を無効化してみる

`gpt()` 関数内で Attention の出力をゼロにし、残差接続だけにしてみます。

```javascript
// src/microgpt.js:297 付近（attn_wo の後）を変更
// 変更前
x = linear(xAttn, stateDict[`layer${li}.attn_wo`]);
// 変更後（Attention出力を無視）
x = xAttn.map(() => new Value(0));
```

損失が下がりにくくなるはずです。Attention がなければモデルは「過去の文脈」を参照できず、各位置を独立に処理するだけになります。

**確認ポイント:** Attention ありと無しで、生成される名前の品質がどう変わるか比較してみてください。

### 実験 3: ReLU を別の活性化関数に変えてみる

MLP の活性化関数を変更します。

```javascript
// src/microgpt.js:121-123 の relu() メソッドに加えて、新しい活性化を追加
// Value クラスに追加
tanh() {
    const t = Math.tanh(this.data);
    return new Value(t, [this], [1 - t * t]);  // ∂tanh/∂x = 1 - tanh²(x)
}
```

```javascript
// src/microgpt.js:308 を変更
// 変更前
x = x.map(xi => xi.relu());
// 変更後
x = x.map(xi => xi.tanh());
```

**確認ポイント:** MLP 活性化パネルで、ReLU（0 か正の値）と tanh（-1〜1 の連続値）の違いを目で見てください。活性化パターンが変わり、学習の収束速度にも影響が出ます。

### 実験 4: 埋め込み次元を変えてみる

```javascript
// src/microgpt.js:173
const N_EMBD = 16;  // → 8 に減らす、または 32 に増やす
```

**注意:** `N_EMBD % N_HEAD === 0` でなければエラーになります。

| 値 | ヘッド数 | HEAD_DIM | パラメータ数 | 傾向 |
|----|---------|----------|------------|------|
| `8` | 4 | 2 | 約 2,500 | 表現力が低く、損失が十分に下がらない |
| `16` | 4 | 4 | 約 10,000 | デフォルト。バランスが良い |
| `32` | 4 | 8 | 約 37,000 | 表現力は高いが、学習が遅くなる |

**確認ポイント:** 埋め込み散布図で、低次元だと文字が密集して区別しにくく、高次元だと分離しやすくなる様子を確認してください。

### 実験 5: temperature を変えてみる

```javascript
// src/microgpt.js:491
const temperature = 0.5;  // → 0.1, 1.0, 2.0 に変えてみる
```

| 値 | 生成の傾向 |
|----|-----------|
| `0.1` | ほぼ同じ名前を繰り返す（最も確率の高い文字を選び続ける） |
| `0.5` | 適度な多様性（デフォルト） |
| `1.0` | 多様だが、時々不自然な名前が出る |
| `2.0` | ランダムに近い。ほぼ意味不明な文字列になる |

**確認ポイント:** これはフォワードパスのみの変更で、学習には影響しません。学習済みの同じモデルから異なる多様性の出力が得られることを確認してください。

---

## 4. ハイパーパラメータの変更と影響

### 4.1 一覧と安全な範囲

| パラメータ | デフォルト | 安全な実験範囲 | 変更時の注意 |
|-----------|----------|-------------|------------|
| `N_EMBD` | 16 | 8〜64 | `N_EMBD % N_HEAD === 0` 必須 |
| `N_HEAD` | 4 | 1〜8 | `N_EMBD % N_HEAD === 0` 必須 |
| `N_LAYER` | 2 (UI可変) | 1〜4 | 3以上はブラウザが重くなる |
| `BLOCK_SIZE` | 16 | 8〜32 | データ内の最長名前より短いと切り詰められる |
| `learningRate` | 0.01 | 0.001〜0.05 | 高すぎると発散、低すぎると未学習 |
| `beta1` | 0.85 | 0.8〜0.95 | Adam のモーメント減衰率 |
| `beta2` | 0.99 | 0.95〜0.999 | Adam の分散減衰率 |
| `std` (初期化) | 0.08 | 0.01〜0.2 | 大きすぎると勾配爆発 |
| `temperature` | 0.5 | 0.1〜2.0 | 推論のみに影響 |

### 4.2 パラメータ数の計算方法

パラメータ数を事前に計算することで、変更の影響を見積もれます。

```
wte:      vocabSize × N_EMBD
wpe:      BLOCK_SIZE × N_EMBD
lm_head:  vocabSize × N_EMBD

各レイヤー:
  attn_wq: N_EMBD × N_EMBD
  attn_wk: N_EMBD × N_EMBD
  attn_wv: N_EMBD × N_EMBD
  attn_wo: N_EMBD × N_EMBD
  mlp_fc1: (4 × N_EMBD) × N_EMBD
  mlp_fc2: N_EMBD × (4 × N_EMBD)

合計 ≈ vocabSize×N_EMBD×2 + BLOCK_SIZE×N_EMBD + N_LAYER × (4×N_EMBD² + 8×N_EMBD²)
     = vocabSize×N_EMBD×2 + BLOCK_SIZE×N_EMBD + N_LAYER × 12 × N_EMBD²
```

デフォルト（vocabSize≈28, N_EMBD=16, N_LAYER=2）では約 7,000〜10,000 パラメータです。

---

## 5. アーキテクチャの拡張

### 5.1 バイアス項の追加

現在の `linear()` は `y = Wx` ですが、バイアス `b` を追加して `y = Wx + b` にできます。

```javascript
// 新しい linear 関数
function linearWithBias(x, w, b) {
    return w.map((wo, i) => {
        const dot = wo.reduce((sum, wi, j) => sum.add(wi.mul(x[j])), new Value(0));
        return b ? dot.add(b[i]) : dot;
    });
}
```

バイアスベクトルの初期化:

```javascript
function bias(n) {
    return Array.from({ length: n }, () => new Value(0));
}
```

`stateDict` にバイアスを追加:

```javascript
stateDict[`layer${i}.attn_bq`] = bias(N_EMBD);
// ... 各層に同様に追加
```

**効果:** バイアスはオフセットを与えるため、入力がゼロの場合でも出力がゼロにならなくなります。小規模モデルでは学習の柔軟性が上がることがあります。

### 5.2 Dropout の実装

過学習を抑えるための正則化手法です。学習時にランダムにニューロンを無効化します。

```javascript
// Value クラスにメソッドを追加
function dropout(values, rate, training = true) {
    if (!training) return values;
    return values.map(v => {
        if (rng.random() < rate) {
            return new Value(0);  // このニューロンを無効化
        }
        return v.div(1 - rate);   // 残りをスケールアップ（期待値を保つ）
    });
}
```

使用箇所:

```javascript
// Attention 出力後と MLP 出力後に追加
x = linear(xAttn, stateDict[`layer${li}.attn_wo`]);
x = dropout(x, 0.1, isTraining);  // ← 追加
x = x.map((a, i) => a.add(xResidual[i]));
```

**注意:** 推論時には Dropout を無効化する必要があります（`training = false`）。

### 5.3 LayerNorm への切り替え

現在は RMSNorm を使っていますが、標準的な LayerNorm に変更できます。

```javascript
function layernorm(x, gamma, beta) {
    const n = x.length;
    // 平均
    const mean = x.reduce((s, xi) => s.add(xi), new Value(0)).div(n);
    // 分散
    const variance = x.reduce((s, xi) => {
        const diff = xi.sub(mean);
        return s.add(diff.mul(diff));
    }, new Value(0)).div(n);
    // 正規化
    const scale = variance.add(1e-5).pow(-0.5);
    return x.map((xi, i) => {
        const norm = xi.sub(mean).mul(scale);
        return gamma ? norm.mul(gamma[i]).add(beta[i]) : norm;
    });
}
```

RMSNorm との違いは「平均の引き算（中心化）」の有無です。LayerNorm のほうが計算は重いですが、平均シフトも正規化できます。

### 5.4 GeLU 活性化関数

GPT-2 以降で使われている GeLU を実装してみます。

```javascript
// Value クラスにメソッドを追加
gelu() {
    // 近似版: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    const x = this.data;
    const cdf = 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
    // 勾配の近似
    const pdf = Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
    const grad = cdf + x * pdf;
    return new Value(x * cdf, [this], [grad]);
}
```

ReLU が「0 以下を完全に遮断」するのに対し、GeLU は負の値も少し通す滑らかな関数です。

### 5.5 マルチレイヤーへの拡張時の注意

`N_LAYER` を増やすと以下に注意が必要です。

1. **計算量**: レイヤー数に比例してフォワード・バックワードの計算量が増える
2. **勾配消失**: 残差接続があるため緩和されるが、4層以上では勾配フローパネルを注視する
3. **初期化**: 深くなるほど初期化の `std` を小さくすると安定する場合がある（例: `std = 0.08 / Math.sqrt(N_LAYER)`）

---

## 6. データセットの差し替え

### 6.1 基本手順

1. `public/data/` に新しいテキストファイルを配置
2. `loadDataset()` のパスを変更

```javascript
// src/microgpt.js:64
const response = await fetch('./data/names.txt');
// → 例: './data/my_dataset.txt'
```

### 6.2 データセット要件

- **1行1サンプル** のテキストファイル
- 各行が `BLOCK_SIZE`（16文字）以下に収まること
- 文字種が増えすぎると語彙サイズが大きくなり、学習が遅くなる

### 6.3 具体例

#### ポケモン名

```text
pikachu
charizard
bulbasaur
squirtle
...
```

#### 英単語

```text
apple
banana
cherry
...
```

#### 日本語の苗字（ひらがな）

```text
さとう
すずき
たかはし
たなか
...
```

**注意:** 日本語を使う場合、文字種（語彙サイズ）が大幅に増えます。ひらがなだけなら約70種ですが、漢字を含めると数百〜数千になり、`N_EMBD=16` では表現力が不足します。`N_EMBD=32` 以上への変更を検討してください。

### 6.4 データセットのサイズと学習ステップ数の目安

| サンプル数 | 推奨ステップ数 | 理由 |
|-----------|-------------|------|
| 100以下 | 500〜1000 | すぐに過学習するため多くは不要 |
| 100〜1000 | 1000〜3000 | バランスが良い |
| 1000〜10000 | 2000〜5000 | 十分なパターンがあるため長く学習できる |
| 10000以上 | 3000〜 | ブラウザの速度が律速になる |

---

## 7. 可視化パネルの追加

### 7.1 追加の基本手順

1. `src/viz/` に新しいチャートクラスを作成
2. `src/viz/vizManager.js` にインポートして登録
3. `src/index.html` に表示先の DOM 要素を追加
4. （必要なら）`gpt()` や `trainAndGenerate()` から新しいデータを `callback` に追加

### 7.2 チャートクラスのテンプレート

```javascript
// src/viz/myNewChart.js
import * as d3 from 'd3';

export class MyNewChart {
    constructor(container) {
        const div = d3.select(container);
        this.svg = div.append('svg')
            .attr('viewBox', '0 0 400 250')
            .attr('preserveAspectRatio', 'xMidYMid meet')
            .style('width', '100%');

        // タイトル
        this.svg.append('text')
            .attr('x', 200).attr('y', 14)
            .attr('text-anchor', 'middle')
            .attr('fill', '#d4d4d4')
            .attr('font-size', '12px')
            .text('チャートタイトル');

        this.g = this.svg.append('g')
            .attr('transform', 'translate(50, 30)');
    }

    render(data) {
        if (!data) return;
        // D3.js で描画
    }
}
```

### 7.3 VizManager への登録

```javascript
// src/viz/vizManager.js
import { MyNewChart } from './myNewChart.js';

export class VizManager {
    constructor() {
        // ... 既存チャート ...
        this.myNewChart = new MyNewChart('#chart-my-new');
    }

    createCallback() {
        return (data) => {
            // ... 既存の描画 ...
            if (step % 10 === 0) {
                this.myNewChart.render(data.myNewData);
            }
        };
    }
}
```

### 7.4 HTML への追加

```html
<!-- src/index.html の chart-grid 内に追加 -->
<div class="chart-panel">
  <h3>チャートタイトル</h3>
  <div id="chart-my-new"></div>
  <details>
    <summary>解説</summary>
    <p>このチャートの説明をここに書く</p>
  </details>
</div>
```

### 7.5 可視化のアイデア

| アイデア | 必要なデータ | 難易度 |
|---------|------------|--------|
| 学習率スケジュール | `step`, `lrT` | 低 |
| パラメータの値分布（ヒストグラム） | `params` の `.data` | 低 |
| Attention パターンの系列全体表示 | `attnWeights` の全位置分 | 中 |
| 生成テキストのリアルタイム表示 | 推論ループのサンプル | 中 |
| 損失のトークン位置別内訳 | `losses` 配列 | 中 |
| 埋め込みの時系列アニメーション | 各ステップの `embeddings` | 高 |
| 重み行列のヒートマップ | `stateDict` の各行列 | 中 |

---

## 8. デバッグと問題解決

### 8.1 よくあるエラーと対処

#### `NaN` が出る

**原因:** 勾配爆発により値が `Infinity` → `NaN` に伝播。

**対処:**
- `learningRate` を下げる
- 初期化の `std` を小さくする
- 中間値を確認:
```javascript
// gpt() 内で値を確認
console.log('x norm:', Math.sqrt(x.reduce((s, v) => s + v.data ** 2, 0)));
```

#### 損失が下がらない

**原因候補:**
1. 学習率が低すぎる → `0.01` 〜 `0.05` を試す
2. ステップ数が少なすぎる → 2000 以上に増やす
3. モデルの表現力不足 → `N_EMBD` や `N_LAYER` を増やす
4. データの問題 → 空行やエンコーディング不正がないか確認

#### ブラウザが固まる

**原因:** 計算量がブラウザのメインスレッドを占有している。

**対処:**
- `N_LAYER` を減らす
- `numSteps` を減らす
- yield の頻度を上げる:
```javascript
// src/microgpt.js:481
if (typeof window !== 'undefined' && step % 10 === 0) {
// → step % 5 や step % 1 に変更
```

### 8.2 勾配のデバッグ（数値勾配チェック）

自動微分が正しく動いているか確認する方法です。新しい演算を `Value` に追加したときに使ってください。

```javascript
function numericalGradientCheck(f, param, eps = 1e-5) {
    // 中心差分による数値勾配
    const originalData = param.data;

    param.data = originalData + eps;
    const fPlus = f().data;

    param.data = originalData - eps;
    const fMinus = f().data;

    param.data = originalData;  // 元に戻す

    const numericalGrad = (fPlus - fMinus) / (2 * eps);

    // 自動微分の勾配と比較
    const result = f();
    result.backward();
    const autoGrad = param.grad;

    const diff = Math.abs(numericalGrad - autoGrad);
    console.log(`数値勾配: ${numericalGrad.toFixed(6)}`);
    console.log(`自動微分: ${autoGrad.toFixed(6)}`);
    console.log(`差分: ${diff.toFixed(8)} ${diff < 1e-4 ? '✓ OK' : '✗ 要確認'}`);
}

// 使用例
const a = new Value(2.0);
numericalGradientCheck(() => a.mul(a).add(a), a);
```

### 8.3 学習の可視化チェックポイント

学習が正常に進んでいるかの判断基準:

| ステップ | 損失 | 状態 |
|---------|------|------|
| 0 | 3.0〜4.0 | 正常（ランダム予測） |
| 100 | 2.5〜3.0 | 学習開始 |
| 500 | 1.5〜2.5 | パターンを学び始めている |
| 1000 | 1.0〜2.0 | 基本的な文字パターンを習得 |
| 2000+ | 0.5〜1.5 | 名前らしい構造を学習 |

損失が 3.0 以上から下がらない場合は設定を見直してください。

---

## 9. パフォーマンス改善のヒント

### 9.1 描画頻度の調整

学習速度のボトルネックは計算よりも **描画** のことがあります。

```javascript
// src/viz/vizManager.js:42
if (step % 10 === 0) {  // → step % 50 にすると描画負荷が 1/5 に
```

### 9.2 Web Worker への分離

学習ループを Web Worker に移すことで、UIスレッドの固まりを解消できます。

```javascript
// worker.js（新規作成）
importScripts('./microgpt-worker.js');

self.onmessage = (e) => {
    const { numSteps, numLayers, seed } = e.data;
    trainAndGenerate(
        (data) => self.postMessage({ type: 'step', data }),
        numSteps, numLayers, seed
    );
};
```

**注意:** Worker 内では DOM にアクセスできないため、`fetch` の代わりにメインスレッドからデータを渡す必要があります。

### 9.3 計算の最適化

教育目的のためスカラー `Value` 演算ですが、ホットスポットを最適化するなら:

- `linear()` の内部ループを展開する
- `softmax()` で `exp()` のキャッシュを再利用する
- `backward()` のトポロジカルソートを事前計算してキャッシュする

ただし、最適化するとコードの可読性が下がるため、教育目的との兼ね合いを考慮してください。

---

## 10. 発展課題

理解度に応じた発展課題です。各課題にはヒントを付けています。

### レベル 1: パラメータ調整

1. **最適な学習率を探す**
   - 0.001〜0.1 の範囲で5パターン試し、最も損失が低くなる学習率を見つけてください
   - ヒント: 損失曲線を比較する

2. **ステップ数と過学習の関係**
   - 学習データ100件に絞り、5000ステップ学習。生成された名前がデータセットのコピーになっていたら過学習です
   - ヒント: `docs = docs.slice(0, 100)` で制限できる

### レベル 2: アーキテクチャ変更

3. **ヘッド数1のAttentionと4の違いを比較する**
   - `N_HEAD=1` にして（`N_EMBD=16` のまま）学習し、ヘッド数4との差を観察
   - ヒント: Attention ヒートマップが1列だけになる

4. **残差接続を外して勾配消失を観察する**
   - `x = x.map((a, i) => a.add(xResidual[i]))` をコメントアウト
   - ヒント: 勾配フローパネルのバーが極端に小さくなるはず（特に `wte`, `wpe`）

5. **Positional Encoding を外してみる**
   - `wpe` の加算をスキップして、位置情報なしで学習
   - ヒント: モデルは文字の順序を区別できなくなり、アナグラム的な生成になる

### レベル 3: 機能拡張

6. **学習率のウォームアップを実装する**
   - 最初の100ステップは学習率を0から徐々に上げ、その後線形減衰
   - ヒント:
   ```javascript
   const warmupSteps = 100;
   const lrT = step < warmupSteps
       ? learningRate * (step / warmupSteps)
       : learningRate * (1 - (step - warmupSteps) / (numSteps - warmupSteps));
   ```

7. **Top-k サンプリングを実装する**
   - 推論時に確率上位k個のトークンからのみサンプリング
   - ヒント: `probs` をソートし、上位k個以外を0にしてから `rng.choices()` を呼ぶ

8. **勾配クリッピングを実装する**
   - 勾配の L2 ノルムが閾値を超えたらスケーリングで抑える
   - ヒント:
   ```javascript
   const maxNorm = 1.0;
   const totalNorm = Math.sqrt(params.reduce((s, p) => s + p.grad ** 2, 0));
   if (totalNorm > maxNorm) {
       const scale = maxNorm / totalNorm;
       for (const p of params) p.grad *= scale;
   }
   ```

### レベル 4: 研究的課題

9. **Attention パターンの解釈**
   - 学習済みモデルの各ヘッドが「何に注目しているか」を分析する
   - ヒント: 母音→子音パターン、位置依存パターンなどを探す

10. **異なるデータセット間のTransfer Learning**
    - 英語名で学習した重みを日本語名の学習の初期値に使うとどうなるか
    - ヒント: `stateDict` を保存・復元する仕組みを作る

---

## 付録: microgpt.js の変更を安全に行うためのチェックリスト

コードを変更する前に、以下を確認してください。

- [ ] **変更箇所の特定**: どの関数・クラスに手を入れるか明確か
- [ ] **`N_EMBD % N_HEAD === 0`** の制約を満たしているか
- [ ] **`Value` の演算のみを使っているか**: 通常の `number` 演算を混ぜると勾配が切れる
- [ ] **新しい `Value` 演算のローカル勾配が正しいか**: 数値勾配チェック（8.2節）で検証
- [ ] **`stateDict` に追加したパラメータが `params` 配列に含まれるか**: 含まれないと学習されない
- [ ] **可視化データの形式が変わっていないか**: `callback` に渡すオブジェクトの形式を確認
- [ ] **git で変更前のコミットがあるか**: 壊れたら戻せるように
