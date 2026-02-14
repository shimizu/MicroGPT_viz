import * as d3 from 'd3';

export class AttentionHeatmap {
    constructor(container, nHead) {
        this.nHead = nHead;
        this.margin = { top: 35, right: 10, bottom: 25, left: 10 };

        const div = d3.select(container);
        this.svg = div.append('svg')
            .attr('viewBox', '0 0 400 250')
            .attr('preserveAspectRatio', 'xMidYMid meet')
            .style('width', '100%');

        this.g = this.svg.append('g')
            .attr('transform', `translate(${this.margin.left},${this.margin.top})`);

        this.width = 400 - this.margin.left - this.margin.right;
        this.height = 250 - this.margin.top - this.margin.bottom;

        this.colorScale = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);

        // タイトル
        this.svg.append('text')
            .attr('x', 200).attr('y', 14)
            .attr('text-anchor', 'middle')
            .attr('fill', '#d4d4d4')
            .attr('font-size', '12px')
            .text('Attention重み（ヘッド別）');
    }

    render(attnWeights, tokens, uchars, BOS) {
        if (!attnWeights || !attnWeights[0]) return;

        // layer 0のattention weights（[head][seqLen]の配列）
        const layerWeights = attnWeights[0];
        const seqLen = layerWeights[0].length;

        // トークンラベルを作成
        const labels = tokens.slice(0, seqLen).map(t =>
            t === BOS ? '⟨B⟩' : (uchars[t] || '?')
        );

        const headWidth = this.width / this.nHead;
        const cellSize = Math.min(headWidth - 10, this.height / seqLen, 18);

        this.g.selectAll('*').remove();

        for (let h = 0; h < this.nHead; h++) {
            const weights = layerWeights[h];
            const hGroup = this.g.append('g')
                .attr('transform', `translate(${h * headWidth}, 0)`);

            // ヘッドラベル
            hGroup.append('text')
                .attr('x', headWidth / 2)
                .attr('y', -4)
                .attr('text-anchor', 'middle')
                .attr('fill', '#888')
                .attr('font-size', '9px')
                .text(`H${h}`);

            // セル描画
            for (let j = 0; j < weights.length; j++) {
                hGroup.append('rect')
                    .attr('x', (headWidth - cellSize * 1) / 2)
                    .attr('y', j * cellSize)
                    .attr('width', cellSize - 1)
                    .attr('height', cellSize - 1)
                    .attr('fill', this.colorScale(weights[j]))
                    .attr('rx', 1);

                if (cellSize >= 10) {
                    hGroup.append('text')
                        .attr('x', headWidth / 2 + cellSize)
                        .attr('y', j * cellSize + cellSize / 2 + 3)
                        .attr('text-anchor', 'start')
                        .attr('fill', '#888')
                        .attr('font-size', '8px')
                        .text(h === 0 ? labels[j] : '');
                }
            }
        }
    }
}
