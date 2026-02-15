/**
 * 依存ライブラリなしの純粋なJavaScriptによるGPT実装。
 * このファイル1つがアルゴリズムの全体であり、それ以外はすべて効率化のための工夫に過ぎない。
 *
 * @karpathy の microgpt.py からの移植版
 * @see https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
 */

// ========================================
// シード付き疑似乱数生成器
// ========================================
// シードを固定することで、実行のたびに同じ結果を再現できる。
// 学習・推論のデバッグや比較実験に不可欠。
class SeededRandom {
    constructor(seed = 42) {
        this.seed = seed;
    }

    // sin関数の小数部分を利用した簡易的な疑似乱数（0〜1）
    random() {
        const x = Math.sin(this.seed++) * 10000;
        return x - Math.floor(x);
    }

    // Box-Muller変換によるガウス分布（正規分布）の乱数生成
    // ニューラルネットの重み初期化で使用（平均0・小さな標準偏差のランダム値）
    gauss(mean = 0, std = 1) {
        const u1 = this.random();
        const u2 = this.random();
        const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        return mean + z0 * std;
    }

    // Fisher-Yatesアルゴリズムによる配列シャッフル
    // データセットの順序をランダム化し、学習の偏りを防ぐ
    shuffle(array) {
        const arr = [...array];
        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.floor(this.random() * (i + 1));
            [arr[i], arr[j]] = [arr[j], arr[i]];
        }
        return arr;
    }

    // 重み付きランダム選択（推論時に確率分布からトークンをサンプリング）
    choices(population, weights) {
        const totalWeight = weights.reduce((a, b) => a + b, 0);
        let rand = this.random() * totalWeight;
        for (let i = 0; i < population.length; i++) {
            rand -= weights[i];
            if (rand <= 0) return population[i];
        }
        return population[population.length - 1];
    }
}

let rng = new SeededRandom(42);

// ========================================
// データセット読み込み
// ========================================
async function loadDataset() {
    // ブラウザ環境: fetch APIで取得
    const response = await fetch('./data/names.txt');
    const text = await response.text();
    return text.trim().split('\n').filter(l => l.trim());
}

// ========================================
// 自動微分（Autograd）エンジン — Valueクラス
// ========================================
// スカラー値をラップし、演算の計算グラフを自動構築する。
// forward（前方計算）で結果を求めつつ、各演算のローカル勾配を記録。
// backward（逆伝播）で連鎖律を適用し、すべてのパラメータの勾配を一度に計算する。
class Value {
    /**
     * @param {number} data - スカラー値
     * @param {Value[]} children - この値を生成した入力ノード
     * @param {number[]} localGrads - 各入力に対するローカル勾配 ∂(this)/∂(child)
     */
    constructor(data, children = [], localGrads = []) {
        this.data = data;       // 前方計算の結果値
        this.grad = 0;          // 逆伝播で蓄積される勾配 ∂Loss/∂(this)
        this._children = children;
        this._localGrads = localGrads;
    }

    // 加算: ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
    add(other) {
        other = other instanceof Value ? other : new Value(other);
        return new Value(this.data + other.data, [this, other], [1, 1]);
    }

    // 乗算: ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
    mul(other) {
        other = other instanceof Value ? other : new Value(other);
        return new Value(this.data * other.data, [this, other], [other.data, this.data]);
    }

    // 累乗: ∂(a^n)/∂a = n * a^(n-1)
    pow(exponent) {
        return new Value(
            Math.pow(this.data, exponent),
            [this],
            [exponent * Math.pow(this.data, exponent - 1)]
        );
    }

    // 自然対数: ∂ln(a)/∂a = 1/a（損失関数の計算で使用）
    log() {
        return new Value(Math.log(this.data), [this], [1 / this.data]);
    }

    // 指数関数: ∂exp(a)/∂a = exp(a)（softmaxの計算で使用）
    exp() {
        return new Value(Math.exp(this.data), [this], [Math.exp(this.data)]);
    }

    // ReLU活性化関数: max(0, x)（MLPの非線形変換で使用）
    // 勾配: x > 0 なら 1、x ≤ 0 なら 0
    relu() {
        return new Value(Math.max(0, this.data), [this], [this.data > 0 ? 1 : 0]);
    }

    neg() {
        return this.mul(-1);
    }

    sub(other) {
        return this.add(other instanceof Value ? other.neg() : -other);
    }

    div(other) {
        other = other instanceof Value ? other : new Value(other);
        return this.mul(other.pow(-1));
    }

    // 逆伝播: 計算グラフを末端から遡り、連鎖律で各パラメータの勾配を計算
    backward() {
        // トポロジカルソートで計算順序を決定
        const topo = [];
        const visited = new Set();

        const buildTopo = (v) => {
            if (!visited.has(v)) {
                visited.add(v);
                for (const child of v._children) {
                    buildTopo(child);
                }
                topo.push(v);
            }
        };

        buildTopo(this);
        this.grad = 1; // 損失自身の勾配 ∂L/∂L = 1

        // 逆順に辿り、連鎖律 ∂L/∂child += ∂L/∂v * ∂v/∂child を適用
        for (let i = topo.length - 1; i >= 0; i--) {
            const v = topo[i];
            for (let j = 0; j < v._children.length; j++) {
                const child = v._children[j];
                const localGrad = v._localGrads[j];
                child.grad += localGrad * v.grad;
            }
        }
    }
}

// ========================================
// ハイパーパラメータ
// ========================================
// ブラウザで実行可能な小規模設定（パラメータ数 ≈ 10,000）
const N_EMBD = 16;      // 埋め込み次元数（各トークンをN_EMBD次元のベクトルで表現）
const N_HEAD = 4;       // Attentionヘッド数（異なる視点で文脈を捉える）
let N_LAYER = 2;        // Transformerレイヤー数（UIから変更可能）
const BLOCK_SIZE = 16;  // 最大系列長（一度に処理できるトークン数の上限）
const HEAD_DIM = N_EMBD / N_HEAD; // 各ヘッドの次元数（= 4）

// ========================================
// ユーティリティ関数
// ========================================

// 重み行列を小さなランダム値で初期化（std=0.08のガウス分布）
// 初期値が大きすぎると勾配爆発、小さすぎると学習が進まないため適切な標準偏差が重要
function matrix(nout, nin, std = 0.08) {
    const mat = [];
    for (let i = 0; i < nout; i++) {
        const row = [];
        for (let j = 0; j < nin; j++) {
            row.push(new Value(rng.gauss(0, std)));
        }
        mat.push(row);
    }
    return mat;
}

// 線形変換 (y = Wx): 入力ベクトル x に重み行列 w を掛ける
// ニューラルネットの基本演算で、Attention・MLPの各所で使われる
function linear(x, w) {
    return w.map(wo => {
        return wo.reduce((sum, wi, i) => sum.add(wi.mul(x[i])), new Value(0));
    });
}

// Softmax関数: logits（生の出力スコア）を確率分布に変換
// 数値安定性のため最大値を引いてからexp計算（オーバーフロー防止）
function softmax(logits) {
    const maxVal = Math.max(...logits.map(v => v.data));
    const exps = logits.map(val => val.sub(maxVal).exp());
    const total = exps.reduce((sum, e) => sum.add(e), new Value(0));
    return exps.map(e => e.div(total));
}

// RMSNorm（Root Mean Square正規化）
// LayerNormの簡略版。ベクトルの各要素を二乗平均平方根で割り、スケールを揃える。
// 学習の安定化に寄与し、LLaMA等の最新モデルでも採用されている。
function rmsnorm(x) {
    const ms = x.reduce((sum, xi) => sum.add(xi.mul(xi)), new Value(0)).div(x.length);
    const scale = ms.add(1e-5).pow(-0.5); // ε=1e-5 でゼロ除算を防止
    return x.map(xi => xi.mul(scale));
}

// ========================================
// GPTフォワードパス（1トークンずつ処理）
// ========================================
// Transformerの推論を1トークン単位で実行する。
// 処理フロー:
//   トークン埋め込み + 位置埋め込み → RMSNorm
//   → Multi-Head Attention（+ 残差接続）
//   → MLP（+ 残差接続）
//   → 線形射影で次トークンの確率分布（logits）を出力
//
// KVキャッシュ: 過去のkey/valueを蓄積し、同じ計算の繰り返しを避ける
function gpt(tokenId, posId, keys, values, stateDict) {
    // トークン埋め込み（wte）と位置埋め込み（wpe）を加算
    // → 「どの文字か」+「何番目の位置か」の情報を統合
    const tokEmb = stateDict.wte[tokenId];
    const posEmb = stateDict.wpe[posId];
    let x = tokEmb.map((t, i) => t.add(posEmb[i]));
    x = rmsnorm(x);

    const attnWeightsAll = []; // 可視化用にAttention重みを保存
    const headOutputsAll = []; // ヘッド出力可視化用
    const mlpActivationsAll = []; // MLP活性化可視化用
    const residualStages = []; // 残差ストリーム可視化用
    residualStages.push({ label: 'embed', values: x.map(v => v.data) });

    for (let li = 0; li < N_LAYER; li++) {
        // --- Multi-Head Attention ---
        // 「文脈のどの部分に注目するか」を学習するメカニズム
        const xResidual = x; // 残差接続のために元の値を保持
        x = rmsnorm(x);

        // Q(Query), K(Key), V(Value) を線形変換で計算
        // Q: 「何を探しているか」、K: 「何を持っているか」、V: 「実際の情報」
        const q = linear(x, stateDict[`layer${li}.attn_wq`]);
        const k = linear(x, stateDict[`layer${li}.attn_wk`]);
        const v = linear(x, stateDict[`layer${li}.attn_wv`]);
        keys[li].push(k);     // KVキャッシュに追加
        values[li].push(v);

        const xAttn = [];
        const layerAttnWeights = [];
        const layerHeadOutputs = [];
        for (let h = 0; h < N_HEAD; h++) {
            // 各ヘッドはN_EMBD次元をN_HEAD分割したHEAD_DIM次元で動作
            const hs = h * HEAD_DIM;
            const qH = q.slice(hs, hs + HEAD_DIM);
            const kH = keys[li].map(ki => ki.slice(hs, hs + HEAD_DIM));
            const vH = values[li].map(vi => vi.slice(hs, hs + HEAD_DIM));

            // Scaled Dot-Product Attention: score = Q·K / √d
            // √d で割ることで、次元数が大きい場合のsoftmax飽和を防ぐ
            const attnLogits = kH.map(kHt => {
                return qH.reduce((sum, qHj, j) => sum.add(qHj.mul(kHt[j])), new Value(0))
                    .div(Math.sqrt(HEAD_DIM));
            });

            // Attention重み（どの位置にどれだけ注目するかの確率分布）
            const attnWeights = softmax(attnLogits);
            layerAttnWeights.push(attnWeights.map(w => w.data));

            // Value の重み付き和 → このヘッドの出力
            const headDimOutputs = [];
            for (let j = 0; j < HEAD_DIM; j++) {
                const headOut = vH.reduce((sum, vHt, t) =>
                    sum.add(attnWeights[t].mul(vHt[j])), new Value(0));
                xAttn.push(headOut);
                headDimOutputs.push(headOut.data);
            }
            layerHeadOutputs.push(headDimOutputs);
        }
        attnWeightsAll.push(layerAttnWeights);
        headOutputsAll.push(layerHeadOutputs);

        // 全ヘッドの出力を結合し、出力射影（Wo）で元の次元に変換
        x = linear(xAttn, stateDict[`layer${li}.attn_wo`]);
        // 残差接続: 元の入力を加算（勾配消失を防ぎ、学習を安定化）
        x = x.map((a, i) => a.add(xResidual[i]));
        residualStages.push({ label: `L${li}_attn`, values: x.map(v => v.data) });

        // --- MLP（フィードフォワードネットワーク）---
        // Attentionで集めた情報を非線形変換で処理する
        // 構造: 線形変換(拡大) → ReLU → 線形変換(縮小) + 残差接続
        const xResidual2 = x;
        x = rmsnorm(x);
        x = linear(x, stateDict[`layer${li}.mlp_fc1`]);  // N_EMBD → 4*N_EMBD に拡大
        x = x.map(xi => xi.relu());                       // 非線形活性化
        mlpActivationsAll.push(x.map(v => v.data));       // ReLU後の活性化を記録
        x = linear(x, stateDict[`layer${li}.mlp_fc2`]);  // 4*N_EMBD → N_EMBD に縮小
        x = x.map((a, i) => a.add(xResidual2[i]));       // 残差接続
        residualStages.push({ label: `L${li}_mlp`, values: x.map(v => v.data) });
    }

    // 言語モデルヘッド: 隠れ状態を語彙サイズの logits に射影
    const logits = linear(x, stateDict.lm_head);
    return { logits, attnWeightsAll, headOutputsAll, mlpActivationsAll, residualStages };
}

// ========================================
// メイン関数: 学習ループと推論
// ========================================
// @param {Function} onStep - 各ステップ後に呼ばれるコールバック（可視化用）
// @param {number} numSteps - 学習ステップ数（デフォルト: 1000）
async function trainAndGenerate(onStep, numSteps = 1000, numLayers = 2, seed = 42) {
    rng = new SeededRandom(seed);
    N_LAYER = numLayers;
    // --- データセット読み込みとトークナイザ構築 ---
    console.log('Loading dataset...');
    let docs = await loadDataset();
    docs = rng.shuffle(docs); // シャッフルして学習順序をランダム化
    console.log(`num docs: ${docs.length}`);

    // 文字レベルトークナイザ: データ中の全ユニーク文字を語彙とする
    // BOS（Beginning of Sequence）トークンを語彙末尾に追加し、名前の区切りに使用
    const uchars = [...new Set(docs.join(''))].sort();
    const BOS = uchars.length;
    const vocabSize = uchars.length + 1; // 文字数 + BOS
    console.log(`vocab size: ${vocabSize}`);

    // --- パラメータ初期化 ---
    console.log('Initializing parameters...');
    const stateDict = {
        wte: matrix(vocabSize, N_EMBD),       // トークン埋め込み行列
        wpe: matrix(BLOCK_SIZE, N_EMBD),      // 位置埋め込み行列
        lm_head: matrix(vocabSize, N_EMBD)    // 言語モデルヘッド（隠れ状態→語彙の確率）
    };

    // 各Transformerレイヤーの重み行列を初期化
    for (let i = 0; i < N_LAYER; i++) {
        stateDict[`layer${i}.attn_wq`] = matrix(N_EMBD, N_EMBD);  // Query射影
        stateDict[`layer${i}.attn_wk`] = matrix(N_EMBD, N_EMBD);  // Key射影
        stateDict[`layer${i}.attn_wv`] = matrix(N_EMBD, N_EMBD);  // Value射影
        stateDict[`layer${i}.attn_wo`] = matrix(N_EMBD, N_EMBD);  // 出力射影
        stateDict[`layer${i}.mlp_fc1`] = matrix(4 * N_EMBD, N_EMBD);  // MLP第1層（拡大）
        stateDict[`layer${i}.mlp_fc2`] = matrix(N_EMBD, 4 * N_EMBD);  // MLP第2層（縮小）
    }

    // 全パラメータをフラットな配列に集約（オプティマイザ用）
    const params = [];
    for (const mat of Object.values(stateDict)) {
        for (const row of mat) {
            for (const p of row) {
                params.push(p);
            }
        }
    }
    console.log(`num params: ${params.length}`);

    // --- Adamオプティマイザの初期化 ---
    // Adam: 勾配の1次モーメント（平均）と2次モーメント（分散）を追跡し、
    // パラメータごとに適応的な学習率で更新する最適化アルゴリズム
    const learningRate = 0.01;
    const beta1 = 0.85;   // 1次モーメントの指数移動平均の減衰率
    const beta2 = 0.99;   // 2次モーメントの指数移動平均の減衰率
    const epsAdam = 1e-8;  // ゼロ除算防止の微小値
    const m = new Array(params.length).fill(0); // 1次モーメント（勾配の移動平均）
    const v = new Array(params.length).fill(0); // 2次モーメント（勾配二乗の移動平均）

    // ========================================
    // 学習ループ
    // ========================================
    // 各ステップで1つの名前を取り出し、文字ごとに次の文字を予測。
    // 予測と正解の差（損失）から逆伝播で勾配を計算し、パラメータを更新する。
    console.log('\nTraining...');

    for (let step = 0; step < numSteps; step++) {
        // 学習データから1つの名前を選択（巡回）
        const doc = docs[step % docs.length];
        // BOSで名前を囲む: [BOS, 文字1, 文字2, ..., BOS]
        const tokens = [BOS, ...doc.split('').map(ch => uchars.indexOf(ch)), BOS];
        const n = Math.min(BLOCK_SIZE, tokens.length - 1);

        // KVキャッシュを各ステップで初期化（名前ごとに独立）
        const keys = Array(N_LAYER).fill(null).map(() => []);
        const values = Array(N_LAYER).fill(null).map(() => []);
        const losses = [];
        let lastAttnWeights = null;
        let lastProbs = null;
        let lastResidualStages = null;
        let lastHeadOutputs = null;
        let lastMlpActivations = null;

        // 各位置で次のトークンを予測し、損失を計算
        for (let posId = 0; posId < n; posId++) {
            const tokenId = tokens[posId];       // 入力トークン
            const targetId = tokens[posId + 1];   // 正解の次トークン
            const { logits, attnWeightsAll, headOutputsAll, mlpActivationsAll, residualStages } = gpt(tokenId, posId, keys, values, stateDict);
            const probs = softmax(logits);
            // 交差エントロピー損失: -log(正解トークンの予測確率)
            const lossT = probs[targetId].log().neg();
            losses.push(lossT);
            lastAttnWeights = attnWeightsAll;
            lastProbs = probs.map(p => p.data);
            lastResidualStages = residualStages;
            lastHeadOutputs = headOutputsAll;
            lastMlpActivations = mlpActivationsAll;
        }

        // 名前全体の平均損失
        const loss = losses.reduce((sum, l) => sum.add(l), new Value(0)).div(n);

        // 逆伝播: 損失から全パラメータの勾配を計算
        loss.backward();

        // 各パラメータグループの勾配L2ノルムを収集（Adam更新前）
        const gradNorms = {};
        for (const [name, mat] of Object.entries(stateDict)) {
            let sumSq = 0;
            for (const row of mat) {
                for (const p of row) {
                    sumSq += p.grad * p.grad;
                }
            }
            gradNorms[name] = Math.sqrt(sumSq);
        }

        // Adamによるパラメータ更新
        // 線形学習率減衰: 学習が進むにつれ更新幅を小さくし、収束を安定化
        const lrT = learningRate * (1 - step / numSteps);
        for (let i = 0; i < params.length; i++) {
            const p = params[i];
            // モーメントの更新
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad;
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2;
            // バイアス補正（学習初期のモーメント過小評価を修正）
            const mHat = m[i] / (1 - Math.pow(beta1, step + 1));
            const vHat = v[i] / (1 - Math.pow(beta2, step + 1));
            // パラメータ更新: p -= lr * m̂ / (√v̂ + ε)
            p.data -= lrT * mHat / (Math.sqrt(vHat) + epsAdam);
            p.grad = 0; // 勾配をリセット（次ステップに備える）
        }

        if ((step + 1) % 100 === 0 || step === 0) {
            console.log(`step ${step + 1} / ${numSteps} | loss ${loss.data.toFixed(4)}`);
        }

        // 可視化コールバック: 各ステップのメトリクスを外部に通知
        if (onStep) {
            onStep({
                step,
                loss: loss.data,
                attnWeights: lastAttnWeights,
                probs: lastProbs,
                tokens,
                embeddings: stateDict.wte.map(row => row.map(v => v.data)),
                uchars,
                vocabSize,
                BOS,
                residualStages: lastResidualStages,
                headOutputs: lastHeadOutputs,
                mlpActivations: lastMlpActivations,
                gradNorms
            });
        }

        // UIスレッドに描画機会を与える（ブラウザがフリーズしないよう定期的にyield）
        if (typeof window !== 'undefined' && step % 10 === 0) {
            await new Promise(resolve => setTimeout(resolve, 0));
        }
    }

    // ========================================
    // 推論: 学習済みモデルで新しい名前を生成
    // ========================================
    // BOSトークンから開始し、1文字ずつ確率分布からサンプリングして名前を生成。
    // temperature（温度）で生成の多様性を制御: 低いほど確信度の高い文字を選びやすい。
    const temperature = 0.5;
    console.log('\n--- inference (new, hallucinated names) ---');

    for (let sampleIdx = 0; sampleIdx < 20; sampleIdx++) {
        const keys = Array(N_LAYER).fill(null).map(() => []);
        const values = Array(N_LAYER).fill(null).map(() => []);
        let tokenId = BOS; // BOSから生成を開始
        const sample = [];

        for (let posId = 0; posId < BLOCK_SIZE; posId++) {
            const { logits } = gpt(tokenId, posId, keys, values, stateDict);
            // logitsをtemperatureで割ることで確率分布の尖り具合を調整
            const probs = softmax(logits.map(l => l.div(temperature)));
            tokenId = rng.choices(
                Array.from({length: vocabSize}, (_, i) => i),
                probs.map(p => p.data)
            );
            if (tokenId === BOS) break; // BOSが出たら名前の終わり
            sample.push(uchars[tokenId]);
        }

        console.log(`sample ${(sampleIdx + 1).toString().padStart(2)}: ${sample.join('')}`);
    }
}

export { trainAndGenerate, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE };
