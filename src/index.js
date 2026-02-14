/**
 * The most atomic way to train and inference a GPT in pure, dependency-free JavaScript.
 * This file is the complete algorithm.
 * Everything else is just efficiency.
 * 
 * Ported from @karpathy's microgpt.py
 */

// Random number generator with seed support
class SeededRandom {
    constructor(seed = 42) {
        this.seed = seed;
    }
    
    random() {
        const x = Math.sin(this.seed++) * 10000;
        return x - Math.floor(x);
    }
    
    gauss(mean = 0, std = 1) {
        // Box-Muller transform
        const u1 = this.random();
        const u2 = this.random();
        const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        return mean + z0 * std;
    }
    
    shuffle(array) {
        const arr = [...array];
        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.floor(this.random() * (i + 1));
            [arr[i], arr[j]] = [arr[j], arr[i]];
        }
        return arr;
    }
    
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

const rng = new SeededRandom(42);

// Load dataset (for Node.js environment)
async function loadDataset() {
    if (typeof window !== 'undefined') {
        // Browser environment - fetch from URL
        const response = await fetch('./data/names.txt');
        const text = await response.text();
        return text.trim().split('\n').filter(l => l.trim());
    } else {
        // Node.js environment
        const fs = require('fs');
        const https = require('https');
        
        if (!fs.existsSync('input.txt')) {
            console.log('Downloading input.txt...');
            const file = fs.createWriteStream('input.txt');
            await new Promise((resolve, reject) => {
                https.get('./data/names.txt', 
                    response => {
                        response.pipe(file);
                        file.on('finish', () => {
                            file.close();
                            resolve();
                        });
                    }).on('error', reject);
            });
        }
        
        const text = fs.readFileSync('input.txt', 'utf-8');
        return text.trim().split('\n').filter(l => l.trim());
    }
}

// Autograd - Value class for automatic differentiation
class Value {
    constructor(data, children = [], localGrads = []) {
        this.data = data;
        this.grad = 0;
        this._children = children;
        this._localGrads = localGrads;
    }
    
    add(other) {
        other = other instanceof Value ? other : new Value(other);
        return new Value(this.data + other.data, [this, other], [1, 1]);
    }
    
    mul(other) {
        other = other instanceof Value ? other : new Value(other);
        return new Value(this.data * other.data, [this, other], [other.data, this.data]);
    }
    
    pow(exponent) {
        return new Value(
            Math.pow(this.data, exponent),
            [this],
            [exponent * Math.pow(this.data, exponent - 1)]
        );
    }
    
    log() {
        return new Value(Math.log(this.data), [this], [1 / this.data]);
    }
    
    exp() {
        return new Value(Math.exp(this.data), [this], [Math.exp(this.data)]);
    }
    
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
    
    backward() {
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
        this.grad = 1;
        
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

// Hyperparameters
const N_EMBD = 16;      // embedding dimension
const N_HEAD = 4;       // number of attention heads
const N_LAYER = 1;      // number of layers
const BLOCK_SIZE = 16;  // maximum sequence length
const HEAD_DIM = N_EMBD / N_HEAD;

// Helper function to create a matrix of Values
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

// Linear transformation
function linear(x, w) {
    return w.map(wo => {
        return wo.reduce((sum, wi, i) => sum.add(wi.mul(x[i])), new Value(0));
    });
}

// Softmax
function softmax(logits) {
    const maxVal = Math.max(...logits.map(v => v.data));
    const exps = logits.map(val => val.sub(maxVal).exp());
    const total = exps.reduce((sum, e) => sum.add(e), new Value(0));
    return exps.map(e => e.div(total));
}

// RMSNorm
function rmsnorm(x) {
    const ms = x.reduce((sum, xi) => sum.add(xi.mul(xi)), new Value(0)).div(x.length);
    const scale = ms.add(1e-5).pow(-0.5);
    return x.map(xi => xi.mul(scale));
}

// GPT forward pass
function gpt(tokenId, posId, keys, values, stateDict) {
    const tokEmb = stateDict.wte[tokenId];
    const posEmb = stateDict.wpe[posId];
    let x = tokEmb.map((t, i) => t.add(posEmb[i]));
    x = rmsnorm(x);

    const attnWeightsAll = [];

    for (let li = 0; li < N_LAYER; li++) {
        // Multi-head attention
        const xResidual = x;
        x = rmsnorm(x);
        const q = linear(x, stateDict[`layer${li}.attn_wq`]);
        const k = linear(x, stateDict[`layer${li}.attn_wk`]);
        const v = linear(x, stateDict[`layer${li}.attn_wv`]);
        keys[li].push(k);
        values[li].push(v);

        const xAttn = [];
        const layerAttnWeights = [];
        for (let h = 0; h < N_HEAD; h++) {
            const hs = h * HEAD_DIM;
            const qH = q.slice(hs, hs + HEAD_DIM);
            const kH = keys[li].map(ki => ki.slice(hs, hs + HEAD_DIM));
            const vH = values[li].map(vi => vi.slice(hs, hs + HEAD_DIM));

            const attnLogits = kH.map(kHt => {
                return qH.reduce((sum, qHj, j) => sum.add(qHj.mul(kHt[j])), new Value(0))
                    .div(Math.sqrt(HEAD_DIM));
            });

            const attnWeights = softmax(attnLogits);
            layerAttnWeights.push(attnWeights.map(w => w.data));

            for (let j = 0; j < HEAD_DIM; j++) {
                const headOut = vH.reduce((sum, vHt, t) =>
                    sum.add(attnWeights[t].mul(vHt[j])), new Value(0));
                xAttn.push(headOut);
            }
        }
        attnWeightsAll.push(layerAttnWeights);

        x = linear(xAttn, stateDict[`layer${li}.attn_wo`]);
        x = x.map((a, i) => a.add(xResidual[i]));

        // MLP
        const xResidual2 = x;
        x = rmsnorm(x);
        x = linear(x, stateDict[`layer${li}.mlp_fc1`]);
        x = x.map(xi => xi.relu());
        x = linear(x, stateDict[`layer${li}.mlp_fc2`]);
        x = x.map((a, i) => a.add(xResidual2[i]));
    }

    const logits = linear(x, stateDict.lm_head);
    return { logits, attnWeightsAll };
}

// Main training and inference
async function main(onStep, numSteps = 1000) {
    console.log('Loading dataset...');
    let docs = await loadDataset();
    docs = rng.shuffle(docs);
    console.log(`num docs: ${docs.length}`);

    // Tokenizer
    const uchars = [...new Set(docs.join(''))].sort();
    const BOS = uchars.length;
    const vocabSize = uchars.length + 1;
    console.log(`vocab size: ${vocabSize}`);

    // Initialize parameters
    console.log('Initializing parameters...');
    const stateDict = {
        wte: matrix(vocabSize, N_EMBD),
        wpe: matrix(BLOCK_SIZE, N_EMBD),
        lm_head: matrix(vocabSize, N_EMBD)
    };

    for (let i = 0; i < N_LAYER; i++) {
        stateDict[`layer${i}.attn_wq`] = matrix(N_EMBD, N_EMBD);
        stateDict[`layer${i}.attn_wk`] = matrix(N_EMBD, N_EMBD);
        stateDict[`layer${i}.attn_wv`] = matrix(N_EMBD, N_EMBD);
        stateDict[`layer${i}.attn_wo`] = matrix(N_EMBD, N_EMBD);
        stateDict[`layer${i}.mlp_fc1`] = matrix(4 * N_EMBD, N_EMBD);
        stateDict[`layer${i}.mlp_fc2`] = matrix(N_EMBD, 4 * N_EMBD);
    }

    const params = [];
    for (const mat of Object.values(stateDict)) {
        for (const row of mat) {
            for (const p of row) {
                params.push(p);
            }
        }
    }
    console.log(`num params: ${params.length}`);

    // Adam optimizer buffers
    const learningRate = 0.01;
    const beta1 = 0.85;
    const beta2 = 0.99;
    const epsAdam = 1e-8;
    const m = new Array(params.length).fill(0);
    const v = new Array(params.length).fill(0);

    // Training
    console.log('\nTraining...');

    for (let step = 0; step < numSteps; step++) {
        const doc = docs[step % docs.length];
        const tokens = [BOS, ...doc.split('').map(ch => uchars.indexOf(ch)), BOS];
        const n = Math.min(BLOCK_SIZE, tokens.length - 1);

        const keys = Array(N_LAYER).fill(null).map(() => []);
        const values = Array(N_LAYER).fill(null).map(() => []);
        const losses = [];
        let lastAttnWeights = null;
        let lastProbs = null;

        for (let posId = 0; posId < n; posId++) {
            const tokenId = tokens[posId];
            const targetId = tokens[posId + 1];
            const { logits, attnWeightsAll } = gpt(tokenId, posId, keys, values, stateDict);
            const probs = softmax(logits);
            const lossT = probs[targetId].log().neg();
            losses.push(lossT);
            lastAttnWeights = attnWeightsAll;
            lastProbs = probs.map(p => p.data);
        }

        const loss = losses.reduce((sum, l) => sum.add(l), new Value(0)).div(n);

        loss.backward();

        // Adam update
        const lrT = learningRate * (1 - step / numSteps);
        for (let i = 0; i < params.length; i++) {
            const p = params[i];
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad;
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2;
            const mHat = m[i] / (1 - Math.pow(beta1, step + 1));
            const vHat = v[i] / (1 - Math.pow(beta2, step + 1));
            p.data -= lrT * mHat / (Math.sqrt(vHat) + epsAdam);
            p.grad = 0;
        }

        if ((step + 1) % 100 === 0 || step === 0) {
            console.log(`step ${step + 1} / ${numSteps} | loss ${loss.data.toFixed(4)}`);
        }

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
                BOS
            });
        }

        // UIスレッドに描画機会を与える
        if (typeof window !== 'undefined' && step % 10 === 0) {
            await new Promise(resolve => setTimeout(resolve, 0));
        }
    }

    // Inference
    const temperature = 0.5;
    console.log('\n--- inference (new, hallucinated names) ---');

    for (let sampleIdx = 0; sampleIdx < 20; sampleIdx++) {
        const keys = Array(N_LAYER).fill(null).map(() => []);
        const values = Array(N_LAYER).fill(null).map(() => []);
        let tokenId = BOS;
        const sample = [];

        for (let posId = 0; posId < BLOCK_SIZE; posId++) {
            const { logits } = gpt(tokenId, posId, keys, values, stateDict);
            const probs = softmax(logits.map(l => l.div(temperature)));
            tokenId = rng.choices(
                Array.from({length: vocabSize}, (_, i) => i),
                probs.map(p => p.data)
            );
            if (tokenId === BOS) break;
            sample.push(uchars[tokenId]);
        }

        console.log(`sample ${(sampleIdx + 1).toString().padStart(2)}: ${sample.join('')}`);
    }
}

export { main, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE };
