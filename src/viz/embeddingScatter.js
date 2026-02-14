import * as d3 from 'd3';
import { pca2d } from './pca.js';

export class EmbeddingScatter {
    constructor(container) {
        this.margin = { top: 20, right: 20, bottom: 25, left: 30 };

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

        // タイトル
        this.svg.append('text')
            .attr('x', 200).attr('y', 14)
            .attr('text-anchor', 'middle')
            .attr('fill', '#d4d4d4')
            .attr('font-size', '12px')
            .text('Embedding Space (PCA 2D)');
    }

    render(embeddings, uchars, BOS) {
        if (!embeddings || embeddings.length === 0) return;

        const projected = pca2d(embeddings);

        const labels = [];
        for (let i = 0; i < embeddings.length; i++) {
            labels.push(i === BOS ? '⟨B⟩' : (uchars[i] || '?'));
        }

        const data = projected.map((p, i) => ({
            x: p[0], y: p[1], label: labels[i]
        }));

        const xExtent = d3.extent(data, d => d.x);
        const yExtent = d3.extent(data, d => d.y);
        const xPad = (xExtent[1] - xExtent[0]) * 0.15 || 1;
        const yPad = (yExtent[1] - yExtent[0]) * 0.15 || 1;
        this.xScale.domain([xExtent[0] - xPad, xExtent[1] + xPad]);
        this.yScale.domain([yExtent[0] - yPad, yExtent[1] + yPad]);

        this.g.selectAll('*').remove();

        // ドット
        this.g.selectAll('circle')
            .data(data)
            .enter()
            .append('circle')
            .attr('cx', d => this.xScale(d.x))
            .attr('cy', d => this.yScale(d.y))
            .attr('r', 4)
            .attr('fill', '#dcdcaa')
            .attr('opacity', 0.8);

        // ラベル
        this.g.selectAll('text.label')
            .data(data)
            .enter()
            .append('text')
            .attr('class', 'label')
            .attr('x', d => this.xScale(d.x) + 6)
            .attr('y', d => this.yScale(d.y) + 3)
            .attr('fill', '#d4d4d4')
            .attr('font-size', '9px')
            .text(d => d.label);
    }
}
