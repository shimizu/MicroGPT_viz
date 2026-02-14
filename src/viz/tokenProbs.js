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
        this.titleText = this.svg.append('text')
            .attr('x', 200).attr('y', 14)
            .attr('text-anchor', 'middle')
            .attr('fill', '#d4d4d4')
            .attr('font-size', '12px')
            .text('トークン予測確率（上位10件、⟨B⟩除く）');

        // BOS確率バッジ
        this.bosGroup = this.svg.append('g')
            .attr('transform', `translate(${400 - this.margin.right}, 14)`);
        this.bosGroup.append('rect')
            .attr('x', -70).attr('y', -11)
            .attr('width', 70).attr('height', 16)
            .attr('rx', 3)
            .attr('fill', '#3e3e3e');
        this.bosText = this.bosGroup.append('text')
            .attr('text-anchor', 'end')
            .attr('fill', '#9cdcfe')
            .attr('font-size', '10px')
            .text('⟨B⟩: —');
    }

    render(probs, uchars, BOS) {
        if (!probs) return;

        // BOS確率を取得してバッジに表示
        const bosProb = probs[BOS] || 0;
        this.bosText.text(`⟨B⟩: ${d3.format('.1%')(bosProb)}`);

        // BOS以外の上位10件を取得
        const indexed = probs.map((p, i) => ({ prob: p, idx: i }));
        const filtered = indexed.filter(d => d.idx !== BOS);
        filtered.sort((a, b) => b.prob - a.prob);
        const top = filtered.slice(0, 10);

        const data = top.map(d => ({
            label: uchars[d.idx] || '?',
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
