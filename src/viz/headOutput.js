import * as d3 from 'd3';

export class HeadOutputChart {
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
            .text('ヘッド出力ヒートマップ');

        this.g = this.svg.append('g')
            .attr('transform', 'translate(0, 24)');

        // 発散型色スケール（青=負、白=0、赤=正）
        this.colorScale = d3.scaleSequential(d3.interpolateRdBu).domain([1, -1]);
    }

    render(headOutputs) {
        // headOutputs: [layer][head][dim] の3次元配列
        if (!headOutputs || headOutputs.length === 0) return;

        const nLayers = headOutputs.length;
        const nHeads = headOutputs[0].length;   // 4
        const nDims = headOutputs[0][0].length; // HEAD_DIM=4

        // 全値のmax絶対値でスケールを正規化
        let maxAbs = 0;
        for (const layer of headOutputs) {
            for (const head of layer) {
                for (const v of head) {
                    maxAbs = Math.max(maxAbs, Math.abs(v));
                }
            }
        }
        if (maxAbs === 0) maxAbs = 1;
        this.colorScale.domain([maxAbs, -maxAbs]);

        const availWidth = 390;
        const availHeight = 210;
        const layerGap = 12;
        const cellMax = 28;

        // レイヤー間で均等に横幅を分割
        const layerWidth = (availWidth - layerGap * (nLayers - 1)) / nLayers;
        // セルサイズ: 行=nHeads, 列=nDims に基づいて決定
        const labelOffsetX = 22; // 行ラベル分のオフセット
        const labelOffsetY = 14; // 列ラベル + レイヤーラベル分
        const cellW = Math.min(cellMax, (layerWidth - labelOffsetX) / nDims);
        const cellH = Math.min(cellMax, (availHeight - labelOffsetY) / nHeads);

        // グリッド全体の中央配置
        const gridWidth = nLayers * (labelOffsetX + nDims * cellW) + (nLayers - 1) * layerGap;
        const gridHeight = labelOffsetY + nHeads * cellH;
        const offsetX = (400 - gridWidth) / 2;
        const offsetY = (availHeight - gridHeight) / 2;

        const data = [];
        for (let li = 0; li < nLayers; li++) {
            const lx = offsetX + li * (labelOffsetX + nDims * cellW + layerGap);
            for (let h = 0; h < nHeads; h++) {
                for (let d = 0; d < nDims; d++) {
                    data.push({
                        key: `${li}-${h}-${d}`,
                        x: lx + labelOffsetX + d * cellW,
                        y: offsetY + labelOffsetY + h * cellH,
                        value: headOutputs[li][h][d],
                        layer: li, head: h, dim: d
                    });
                }
            }
        }

        // セル描画
        const cells = this.g.selectAll('rect.head-cell').data(data, d => d.key);
        cells.enter()
            .append('rect')
            .attr('class', 'head-cell')
            .attr('rx', 2)
            .merge(cells)
            .transition().duration(100)
            .attr('x', d => d.x)
            .attr('y', d => d.y)
            .attr('width', cellW - 1)
            .attr('height', cellH - 1)
            .attr('fill', d => this.colorScale(d.value));
        cells.exit().remove();

        // レイヤーラベル
        const layerLabels = Array.from({ length: nLayers }, (_, li) => ({
            key: `L${li}`,
            x: offsetX + li * (labelOffsetX + nDims * cellW + layerGap) + labelOffsetX + (nDims * cellW) / 2,
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
            .attr('y', d => d.y);
        ll.text(d => d.key);
        ll.exit().remove();

        // 行ラベル（H0-H3）
        const rowLabels = [];
        for (let li = 0; li < nLayers; li++) {
            const lx = offsetX + li * (labelOffsetX + nDims * cellW + layerGap);
            for (let h = 0; h < nHeads; h++) {
                rowLabels.push({
                    key: `r-${li}-${h}`,
                    x: lx + labelOffsetX - 4,
                    y: offsetY + labelOffsetY + h * cellH + cellH / 2 + 3,
                    label: `H${h}`
                });
            }
        }
        const rl = this.g.selectAll('text.row-label').data(rowLabels, d => d.key);
        rl.enter()
            .append('text')
            .attr('class', 'row-label')
            .attr('text-anchor', 'end')
            .attr('fill', '#999')
            .attr('font-size', '8px')
            .merge(rl)
            .attr('x', d => d.x)
            .attr('y', d => d.y)
            .text(d => d.label);
        rl.exit().remove();

        // 列ラベル（d0-d3）- 最初のレイヤーのみ表示
        const colLabels = [];
        for (let d = 0; d < nDims; d++) {
            colLabels.push({
                key: `c-${d}`,
                x: offsetX + labelOffsetX + d * cellW + cellW / 2,
                y: offsetY + labelOffsetY - 3,
                label: `d${d}`
            });
        }
        const cl = this.g.selectAll('text.col-label').data(colLabels, d => d.key);
        cl.enter()
            .append('text')
            .attr('class', 'col-label')
            .attr('text-anchor', 'middle')
            .attr('fill', '#999')
            .attr('font-size', '7px')
            .merge(cl)
            .attr('x', d => d.x)
            .attr('y', d => d.y)
            .text(d => d.label);
        cl.exit().remove();
    }
}
