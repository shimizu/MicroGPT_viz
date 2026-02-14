import * as d3 from 'd3';

export class GradientFlowChart {
    constructor(container) {
        this.margin = { top: 20, right: 15, bottom: 55, left: 50 };

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

        // Y軸ラベル
        this.g.append('text')
            .attr('transform', 'rotate(-90)')
            .attr('x', -this.height / 2).attr('y', -38)
            .attr('text-anchor', 'middle')
            .attr('fill', '#d4d4d4')
            .attr('font-size', '9px')
            .text('L2 norm');

        // タイトル
        this.svg.append('text')
            .attr('x', 200).attr('y', 14)
            .attr('text-anchor', 'middle')
            .attr('fill', '#d4d4d4')
            .attr('font-size', '12px')
            .text('勾配フロー（L2ノルム）');
    }

    /** パラメータ名を短い表示名に変換 */
    _shortName(key) {
        if (key === 'wte') return 'wte';
        if (key === 'wpe') return 'wpe';
        if (key === 'lm_head') return 'head';
        // layer0.attn_wq → L0.wq
        const m = key.match(/^layer(\d+)\.(attn_|mlp_)(.+)$/);
        if (m) return `L${m[1]}.${m[3]}`;
        return key;
    }

    /** パラメータ名からカテゴリ色を決定 */
    _color(key) {
        if (key === 'wte' || key === 'wpe') return '#888';   // embed系=グレー
        if (key === 'lm_head') return '#e06c75';              // lm_head=赤
        const m = key.match(/^layer(\d+)\./);
        if (m) {
            const li = parseInt(m[1], 10);
            const colors = ['#569cd6', '#6a9955', '#ce9178']; // 青, 緑, オレンジ
            return colors[li] || '#569cd6';
        }
        return '#888';
    }

    render(gradNorms) {
        if (!gradNorms || Object.keys(gradNorms).length === 0) return;

        const data = Object.entries(gradNorms).map(([key, norm]) => ({
            key,
            label: this._shortName(key),
            norm,
            color: this._color(key)
        }));

        this.xScale.domain(data.map(d => d.label));
        this.yScale.domain([0, d3.max(data, d => d.norm) * 1.15 || 1]);

        // 棒グラフ
        const bars = this.g.selectAll('rect.grad-bar').data(data, d => d.label);
        bars.enter()
            .append('rect')
            .attr('class', 'grad-bar')
            .attr('opacity', 0.8)
            .merge(bars)
            .transition().duration(100)
            .attr('x', d => this.xScale(d.label))
            .attr('width', this.xScale.bandwidth())
            .attr('y', d => this.yScale(d.norm))
            .attr('height', d => this.height - this.yScale(d.norm))
            .attr('fill', d => d.color);
        bars.exit().remove();

        // X軸
        this.xAxisG.call(d3.axisBottom(this.xScale))
            .selectAll('text')
            .attr('font-size', '8px')
            .attr('text-anchor', 'end')
            .attr('transform', 'rotate(-45)')
            .attr('dx', '-0.5em')
            .attr('dy', '0.3em');

        // Y軸
        this.yAxisG.call(d3.axisLeft(this.yScale).ticks(4));
    }
}
