import { LossCurve } from './lossCurve.js';
import { AttentionHeatmap } from './attentionHeatmap.js';
import { TokenProbs } from './tokenProbs.js';
import { EmbeddingScatter } from './embeddingScatter.js';
import { ResidualStreamChart } from './residualStream.js';
import { HeadOutputChart } from './headOutput.js';
import { MlpActivationChart } from './mlpActivation.js';
import { GradientFlowChart } from './gradientFlow.js';

export class VizManager {
    constructor() {
        this.lossCurve = new LossCurve('#chart-loss');
        this.attentionHeatmap = new AttentionHeatmap('#chart-attention', 4);
        this.tokenProbs = new TokenProbs('#chart-probs');
        this.embeddingScatter = new EmbeddingScatter('#chart-embedding');
        this.residualStream = new ResidualStreamChart('#chart-residual');
        this.headOutput = new HeadOutputChart('#chart-head-output');
        this.mlpActivation = new MlpActivationChart('#chart-mlp-activation');
        this.gradientFlow = new GradientFlowChart('#chart-gradient-flow');
    }

    destroy() {
        const ids = [
            '#chart-loss', '#chart-embedding', '#chart-gradient-flow',
            '#chart-attention', '#chart-head-output', '#chart-mlp-activation',
            '#chart-residual', '#chart-probs'
        ];
        for (const id of ids) {
            const el = document.querySelector(id);
            if (el) el.innerHTML = '';
        }
    }

    createCallback() {
        return (data) => {
            const { step, loss, attnWeights, probs, tokens, embeddings, uchars, vocabSize, BOS, residualStages, headOutputs, mlpActivations, gradNorms } = data;

            // 損失データ蓄積: 毎ステップ
            this.lossCurve.addDataPoint(step, loss);

            // 描画: 10ステップごと
            if (step % 10 === 0) {
                this.lossCurve.render();
                this.attentionHeatmap.render(attnWeights, tokens, uchars, BOS);
                this.tokenProbs.render(probs, uchars, BOS);
                this.residualStream.render(residualStages);
                this.headOutput.render(headOutputs);
                this.mlpActivation.render(mlpActivations);
                this.gradientFlow.render(gradNorms);
            }

            // 埋め込み: 50ステップごと
            if (step % 50 === 0) {
                this.embeddingScatter.render(embeddings, uchars, BOS);
            }
        };
    }
}
