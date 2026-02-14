import * as d3 from 'd3';

export class MlpActivationChart {
    constructor(container) {
        const div = d3.select(container);
        this.svg = div.append('svg')
            .attr('viewBox', '0 0 400 180')
            .attr('preserveAspectRatio', 'xMidYMid meet')
            .style('width', '100%');

        // タイトル
        this.svg.append('text')
            .attr('x', 200).attr('y', 14)
            .attr('text-anchor', 'middle')
            .attr('fill', '#d4d4d4')
            .attr('font-size', '12px')
            .text('MLPニューロン活性化');

        this.g = this.svg.append('g')
            .attr('transform', 'translate(0, 24)');

        // シーケンシャル色スケール（黒=0、明るい=高い活性化）
        this.colorScale = d3.scaleSequential(d3.interpolateInferno).domain([0, 1]);
    }

    render(mlpActivations) {
        // mlpActivations: [layer][neuron] の2次元配列（64ニューロン）
        if (!mlpActivations || mlpActivations.length === 0) return;

        const nLayers = mlpActivations.length;
        const nNeurons = mlpActivations[0].length; // 4*N_EMBD = 64
        const gridCols = 8;
        const gridRows = Math.ceil(nNeurons / gridCols); // 8

        // 全レイヤーの最大値でスケール正規化
        let maxVal = 0;
        for (const layer of mlpActivations) {
            for (const v of layer) {
                maxVal = Math.max(maxVal, v);
            }
        }
        if (maxVal === 0) maxVal = 1;
        this.colorScale.domain([0, maxVal]);

        const availWidth = 390;
        const availHeight = 140;
        const layerGap = 16;
        const labelOffsetY = 14; // レイヤーラベル分
        const statsHeight = 14;  // アクティブ率表示分

        const layerWidth = (availWidth - layerGap * (nLayers - 1)) / nLayers;
        const cellSize = Math.min(
            (layerWidth) / gridCols,
            (availHeight - labelOffsetY - statsHeight) / gridRows,
            20
        );

        const gridPixelWidth = gridCols * cellSize;
        const gridPixelHeight = gridRows * cellSize;
        const totalWidth = nLayers * gridPixelWidth + (nLayers - 1) * layerGap;
        const offsetX = (400 - totalWidth) / 2;
        const offsetY = (availHeight - labelOffsetY - gridPixelHeight - statsHeight) / 2;

        const data = [];
        for (let li = 0; li < nLayers; li++) {
            const lx = offsetX + li * (gridPixelWidth + layerGap);
            for (let n = 0; n < nNeurons; n++) {
                const col = n % gridCols;
                const row = Math.floor(n / gridCols);
                data.push({
                    key: `${li}-${n}`,
                    x: lx + col * cellSize,
                    y: offsetY + labelOffsetY + row * cellSize,
                    value: mlpActivations[li][n],
                    layer: li, neuron: n
                });
            }
        }

        // セル描画
        const cells = this.g.selectAll('rect.mlp-cell').data(data, d => d.key);
        cells.enter()
            .append('rect')
            .attr('class', 'mlp-cell')
            .attr('rx', 1)
            .merge(cells)
            .transition().duration(100)
            .attr('x', d => d.x)
            .attr('y', d => d.y)
            .attr('width', cellSize - 1)
            .attr('height', cellSize - 1)
            .attr('fill', d => this.colorScale(d.value));
        cells.exit().remove();

        // レイヤーラベル
        const layerLabels = Array.from({ length: nLayers }, (_, li) => ({
            key: `L${li}`,
            x: offsetX + li * (gridPixelWidth + layerGap) + gridPixelWidth / 2,
            y: offsetY + 4
        }));
        const ll = this.g.selectAll('text.layer-label').data(layerLabels, d => d.key);
        ll.enter()
            .append('text')
            .attr('class', 'layer-label')
            .attr('text-anchor', 'middle')
            .attr('fill', '#4ec9b0')
            .attr('font-size', '10px')
            .merge(ll)
            .attr('x', d => d.x)
            .attr('y', d => d.y)
            .text(d => d.key);
        ll.exit().remove();

        // アクティブ率（非ゼロ比率）
        const statsData = Array.from({ length: nLayers }, (_, li) => {
            const activeCount = mlpActivations[li].filter(v => v > 0).length;
            const rate = (activeCount / nNeurons * 100).toFixed(0);
            return {
                key: `s-${li}`,
                x: offsetX + li * (gridPixelWidth + layerGap) + gridPixelWidth / 2,
                y: offsetY + labelOffsetY + gridPixelHeight + 12,
                label: `Active: ${rate}%`
            };
        });
        const stats = this.g.selectAll('text.active-rate').data(statsData, d => d.key);
        stats.enter()
            .append('text')
            .attr('class', 'active-rate')
            .attr('text-anchor', 'middle')
            .attr('fill', '#dcdcaa')
            .attr('font-size', '9px')
            .merge(stats)
            .attr('x', d => d.x)
            .attr('y', d => d.y)
            .text(d => d.label);
        stats.exit().remove();
    }
}
