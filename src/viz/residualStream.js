import * as d3 from 'd3';

export class ResidualStreamChart {
    constructor(container) {
        this.margin = { top: 20, right: 50, bottom: 35, left: 50 };

        const div = d3.select(container);
        this.svg = div.append('svg')
            .attr('viewBox', '0 0 400 250')
            .attr('preserveAspectRatio', 'xMidYMid meet')
            .style('width', '100%');

        this.g = this.svg.append('g')
            .attr('transform', `translate(${this.margin.left},${this.margin.top})`);

        this.width = 400 - this.margin.left - this.margin.right;
        this.height = 250 - this.margin.top - this.margin.bottom;

        // X軸: ステージ名（scaleBand）
        this.xScale = d3.scaleBand().range([0, this.width]).padding(0.2);

        // 左Y軸: L2ノルム（折れ線）
        this.yScaleNorm = d3.scaleLinear().range([this.height, 0]);
        // 右Y軸: コサイン類似度（棒グラフ）
        this.yScaleCos = d3.scaleLinear().domain([0, 1]).range([this.height, 0]);

        this.xAxisG = this.g.append('g')
            .attr('transform', `translate(0,${this.height})`);
        this.yAxisLeftG = this.g.append('g');
        this.yAxisRightG = this.g.append('g')
            .attr('transform', `translate(${this.width},0)`);

        // 左Y軸ラベル
        this.g.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('x', -this.height / 2).attr('y', -38)
            .attr('text-anchor', 'middle')
            .attr('fill', '#4ec9b0')
            .attr('font-size', '9px')
            .text('L2 norm');

        // 右Y軸ラベル
        this.g.append('text')
            .attr('transform', 'rotate(90)')
            .attr('x', this.height / 2).attr('y', -38)
            .attr('text-anchor', 'middle')
            .attr('fill', '#569cd6')
            .attr('font-size', '9px')
            .text('cos sim');

        // タイトル
        this.svg.append('text')
            .attr('x', 200).attr('y', 14)
            .attr('text-anchor', 'middle')
            .attr('fill', '#d4d4d4')
            .attr('font-size', '12px')
            .text('残差ストリームの変遷');
    }

    render(residualStages) {
        if (!residualStages || residualStages.length === 0) return;

        // L2ノルムを計算
        const norms = residualStages.map(s => {
            const sumSq = s.values.reduce((acc, v) => acc + v * v, 0);
            return Math.sqrt(sumSq);
        });

        // コサイン類似度を計算（前ステージとの比較、最初はnull）
        const cosSims = residualStages.map((s, i) => {
            if (i === 0) return null;
            const prev = residualStages[i - 1].values;
            const curr = s.values;
            let dot = 0, normA = 0, normB = 0;
            for (let j = 0; j < curr.length; j++) {
                dot += prev[j] * curr[j];
                normA += prev[j] * prev[j];
                normB += curr[j] * curr[j];
            }
            const denom = Math.sqrt(normA) * Math.sqrt(normB);
            return denom === 0 ? 0 : dot / denom;
        });

        const labels = residualStages.map(s => s.label);

        this.xScale.domain(labels);
        this.yScaleNorm.domain([0, d3.max(norms) * 1.15 || 1]);

        // 棒グラフ（コサイン類似度）
        const barData = labels.map((label, i) => ({
            label,
            cos: cosSims[i]
        })).filter(d => d.cos !== null);

        const bars = this.g.selectAll('rect.cos-bar').data(barData, d => d.label);
        bars.enter()
            .append('rect')
            .attr('class', 'cos-bar')
            .attr('fill', '#569cd6')
            .attr('opacity', 0.35)
            .merge(bars)
            .transition().duration(100)
            .attr('x', d => this.xScale(d.label))
            .attr('width', this.xScale.bandwidth())
            .attr('y', d => this.yScaleCos(d.cos))
            .attr('height', d => this.height - this.yScaleCos(d.cos));
        bars.exit().remove();

        // 折れ線（L2ノルム）
        const lineData = labels.map((label, i) => ({
            x: this.xScale(label) + this.xScale.bandwidth() / 2,
            y: this.yScaleNorm(norms[i])
        }));

        const line = d3.line().x(d => d.x).y(d => d.y);

        let path = this.g.selectAll('path.norm-line').data([lineData]);
        path.enter()
            .append('path')
            .attr('class', 'norm-line')
            .attr('fill', 'none')
            .attr('stroke', '#4ec9b0')
            .attr('stroke-width', 2)
            .merge(path)
            .transition().duration(100)
            .attr('d', line);

        // ドット
        const dots = this.g.selectAll('circle.norm-dot').data(lineData);
        dots.enter()
            .append('circle')
            .attr('class', 'norm-dot')
            .attr('r', 3)
            .attr('fill', '#4ec9b0')
            .merge(dots)
            .transition().duration(100)
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);
        dots.exit().remove();

        // 軸の描画
        this.xAxisG.call(d3.axisBottom(this.xScale))
            .selectAll('text')
            .attr('font-size', '8px');
        this.yAxisLeftG.call(d3.axisLeft(this.yScaleNorm).ticks(4));
        this.yAxisRightG.call(d3.axisRight(this.yScaleCos).ticks(4).tickFormat(d3.format('.1f')));
    }
}
