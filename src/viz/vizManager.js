import { LossCurve } from './lossCurve.js';
import { AttentionHeatmap } from './attentionHeatmap.js';
import { TokenProbs } from './tokenProbs.js';
import { EmbeddingScatter } from './embeddingScatter.js';

export class VizManager {
    constructor() {
        this.lossCurve = new LossCurve('#chart-loss');
        this.attentionHeatmap = new AttentionHeatmap('#chart-attention', 4);
        this.tokenProbs = new TokenProbs('#chart-probs');
        this.embeddingScatter = new EmbeddingScatter('#chart-embedding');
    }

    createCallback() {
        return (data) => {
            const { step, loss, attnWeights, probs, tokens, embeddings, uchars, vocabSize, BOS } = data;

            // 損失データ蓄積: 毎ステップ
            this.lossCurve.addDataPoint(step, loss);

            // 描画: 10ステップごと
            if (step % 10 === 0) {
                this.lossCurve.render();
                this.attentionHeatmap.render(attnWeights, tokens, uchars, BOS);
                this.tokenProbs.render(probs, uchars, BOS);
            }

            // 埋め込み: 50ステップごと
            if (step % 50 === 0) {
                this.embeddingScatter.render(embeddings, uchars, BOS);
            }
        };
    }
}
