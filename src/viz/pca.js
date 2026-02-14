/**
 * 簡易PCA（べき乗法）
 * 16次元の埋め込みを2次元に射影する
 */

export function pca2d(data) {
    const n = data.length;
    const d = data[0].length;

    // 平均を計算
    const mean = new Array(d).fill(0);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < d; j++) {
            mean[j] += data[i][j];
        }
    }
    for (let j = 0; j < d; j++) mean[j] /= n;

    // 中心化
    const centered = data.map(row => row.map((v, j) => v - mean[j]));

    // 共分散行列 (d x d)
    const cov = Array.from({ length: d }, () => new Array(d).fill(0));
    for (let i = 0; i < n; i++) {
        for (let a = 0; a < d; a++) {
            for (let b = a; b < d; b++) {
                const val = centered[i][a] * centered[i][b];
                cov[a][b] += val;
                if (a !== b) cov[b][a] += val;
            }
        }
    }
    for (let a = 0; a < d; a++) {
        for (let b = 0; b < d; b++) {
            cov[a][b] /= (n - 1) || 1;
        }
    }

    // べき乗法で上位2固有ベクトルを求める
    const eigenvectors = [];
    const covWork = cov.map(row => [...row]);

    for (let ev = 0; ev < 2; ev++) {
        let vec = new Array(d).fill(0).map(() => Math.random() - 0.5);
        let norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
        vec = vec.map(v => v / norm);

        for (let iter = 0; iter < 100; iter++) {
            // 行列ベクトル積
            const newVec = new Array(d).fill(0);
            for (let i = 0; i < d; i++) {
                for (let j = 0; j < d; j++) {
                    newVec[i] += covWork[i][j] * vec[j];
                }
            }
            norm = Math.sqrt(newVec.reduce((s, v) => s + v * v, 0));
            if (norm < 1e-10) break;
            vec = newVec.map(v => v / norm);
        }

        eigenvectors.push(vec);

        // デフレーション: 見つかった固有ベクトル成分を除去
        const eigenvalue = vec.reduce((s, vi, i) => {
            let dot = 0;
            for (let j = 0; j < d; j++) dot += covWork[i][j] * vec[j];
            return s + vi * dot;
        }, 0);

        for (let i = 0; i < d; i++) {
            for (let j = 0; j < d; j++) {
                covWork[i][j] -= eigenvalue * vec[i] * vec[j];
            }
        }
    }

    // 射影
    return centered.map(row => {
        return eigenvectors.map(ev =>
            row.reduce((s, v, i) => s + v * ev[i], 0)
        );
    });
}
