import * as d3 from 'd3';

export class TokenProbs {
    constructor(container) {
        this.margin = { top: 20, right: 20, bottom: 35, left: 45 };

        const div = d3.select(container);
        this.svg = div.append('svg')
            .attr('viewBox', '0 0 400 250')
            .attr('preserveAspectRatio', 'xMidYMid meet')
            .style('width', '100%');

        this.g = this.svg.append('g')
            .attr('transform', `translate(${this.margin.left},${this.margin.top})`);

        this.width = 400 - this.margin.left - this.margin.right;
        this.height = 250 - this.margin.top - this.margin.bottom;

        this.xScale = d3.scaleBand().range([0, this.width]).padding(0.15);
        this.yScale = d3.scaleLinear().range([this.height, 0]);

        this.xAxisG = this.g.append('g')
            .attr('transform', `translate(0,${this.height})`);
        this.yAxisG = this.g.append('g');

        // タイトル
        this.svg.append('text')
            .attr('x', 200).attr('y', 14)
            .attr('text-anchor', 'middle')
            .attr('fill', '#d4d4d4')
            .attr('font-size', '12px')
            .text('Token Prediction Probabilities (Top 10)');
    }

    render(probs, uchars, BOS) {
        if (!probs) return;

        // 上位10件を取得
        const indexed = probs.map((p, i) => ({ prob: p, idx: i }));
        indexed.sort((a, b) => b.prob - a.prob);
        const top = indexed.slice(0, 10);

        const data = top.map(d => ({
            label: d.idx === BOS ? '⟨B⟩' : (uchars[d.idx] || '?'),
            prob: d.prob
        }));

        this.xScale.domain(data.map(d => d.label));
        this.yScale.domain([0, Math.max(d3.max(data, d => d.prob), 0.01)]);

        // バー更新
        const bars = this.g.selectAll('rect.bar').data(data, d => d.label);

        bars.enter()
            .append('rect')
            .attr('class', 'bar')
            .attr('fill', '#569cd6')
            .attr('x', d => this.xScale(d.label))
            .attr('width', this.xScale.bandwidth())
            .attr('y', this.height)
            .attr('height', 0)
            .merge(bars)
            .transition().duration(100)
            .attr('x', d => this.xScale(d.label))
            .attr('width', this.xScale.bandwidth())
            .attr('y', d => this.yScale(d.prob))
            .attr('height', d => this.height - this.yScale(d.prob));

        bars.exit().remove();

        this.xAxisG.call(d3.axisBottom(this.xScale));
        this.yAxisG.call(d3.axisLeft(this.yScale).ticks(5).tickFormat(d3.format('.0%')));
    }
}
