import * as d3 from 'd3';

export class LossCurve {
    constructor(container) {
        this.data = [];
        this.margin = { top: 20, right: 20, bottom: 35, left: 50 };

        const div = d3.select(container);
        this.svg = div.append('svg')
            .attr('viewBox', '0 0 400 250')
            .attr('preserveAspectRatio', 'xMidYMid meet')
            .style('width', '100%');

        this.g = this.svg.append('g')
            .attr('transform', `translate(${this.margin.left},${this.margin.top})`);

        this.width = 400 - this.margin.left - this.margin.right;
        this.height = 250 - this.margin.top - this.margin.bottom;

        this.xScale = d3.scaleLinear().range([0, this.width]);
        this.yScale = d3.scaleLinear().range([this.height, 0]);

        this.xAxis = this.g.append('g')
            .attr('transform', `translate(0,${this.height})`);
        this.yAxis = this.g.append('g');

        this.path = this.g.append('path')
            .attr('fill', 'none')
            .attr('stroke', '#4ec9b0')
            .attr('stroke-width', 1.5);

        // タイトル
        this.svg.append('text')
            .attr('x', 200).attr('y', 14)
            .attr('text-anchor', 'middle')
            .attr('fill', '#d4d4d4')
            .attr('font-size', '12px')
            .text('損失曲線');
    }

    addDataPoint(step, loss) {
        this.data.push({ step, loss });
    }

    render() {
        if (this.data.length < 2) return;

        this.xScale.domain([0, d3.max(this.data, d => d.step)]);
        this.yScale.domain([0, d3.max(this.data, d => d.loss) * 1.05]);

        const line = d3.line()
            .x(d => this.xScale(d.step))
            .y(d => this.yScale(d.loss));

        this.path.attr('d', line(this.data));

        this.xAxis.call(d3.axisBottom(this.xScale).ticks(5));
        this.yAxis.call(d3.axisLeft(this.yScale).ticks(5));
    }
}
