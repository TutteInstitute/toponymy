import * as d3 from "d3";
import tippy from 'tippy.js';
import 'tippy.js/animations/shift-away.css';
import 'tippy.js/dist/tippy.css';
import { flavors } from "@catppuccin/palette";
import './widget.css'

// Adapted from https://observablehq.com/@d3/indented-tree

export default { render }

function shortenString(str, n) {
  if (str.length <= n) {
    return str; 
  } else {
    return str.slice(0, n) + '...'; 
  }
}

function render({ model, el }) {

    el.innerHTML = ""; // Clear prior renders
    
    const data = model.get('data');
    const color_scheme = model.get('palette');
    const width = model.get('width');
    let maxTextChars = Math.floor(width*0.21-53);
    maxTextChars = maxTextChars<5 ? 5 : maxTextChars;


    const background_color = flavors[color_scheme].colors.base.hex
    const edge_color = flavors[color_scheme].colors.subtext1.hex
    const node_color = flavors[color_scheme].colors.peach.hex
    const leaf_node_color = flavors[color_scheme].colors.yellow.hex
    const text_color = flavors[color_scheme].colors.text.hex
    const leaf_text_color = flavors[color_scheme].colors.subtext0.hex
    const hover_text_color = flavors[color_scheme].colors.blue.hex

    const format = d3.format(".2s");
    const nodeSize = 17;
    const root = d3.hierarchy(data)
        .eachBefore((i => d => d.index = i++)(0))
        .eachBefore((i => d => d.size = d.data.size)(0))
        .sum(d => d.children ? 0 : 1);
    const nodes = root.descendants();
    const height = 512;//(nodes.length + 1) * nodeSize;
    
    //el.style.backgroundColor = background_color

    const svg = d3.select(el).append("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [-nodeSize / 2, -nodeSize * 3 / 2, width, height])
        .attr("style", "max-width: 100%; height: auto; font: 10px sans-serif;cursor: pointer;")
        .style("background-color", background_color);

    const main_group = svg.append("g");

    const link = main_group.append("g")
        .attr("fill", "none")
        .attr("stroke", edge_color)
        .selectAll()
        .data(root.links())
        .join("path")
        .attr("d", d => `
            M${d.source.depth * nodeSize},${d.source.index * nodeSize}
            V${d.target.index * nodeSize}
            h${nodeSize}
        `);

    const node = main_group.append("g")
        .selectAll()
        .data(nodes)
        .join("g")
        .attr("transform", d => `translate(0,${d.index * nodeSize})`);

    node.append("circle")
        .attr("cx", d => d.depth * nodeSize)
        .attr("r", 2.5)
        .attr("fill", d => d.children ? node_color : leaf_node_color);

    node.append("text")
        .attr("dy", "0.32em")
        .attr("x", d => d.depth * nodeSize + 6)
        .attr("paint-order", "stroke")
        .attr("stroke", background_color)
        .attr("fill",  d => d.children ? text_color : leaf_text_color )
        .text(d => shortenString(d.data.name, maxTextChars))
        .style("--hover-fill", hover_text_color)
        .attr("class", "node-label")
        .each(function(d){
              tippy(d3.select(this).node(),
              {
                content: d.data.name,
                animation:'shift-away',
                delay:[100, 0],
              });
            });

    //tooltips
    node.append("title")
        .text(d => d.ancestors().reverse().map(d => d.data.name).join("/"));

    requestAnimationFrame(() => {
        // Initialize maxX
        let maxX = -Infinity;
        node.selectAll("text").each(function() {
            const bbox = this.getBBox();
            const rightEdge = bbox.x + bbox.width;
            if (rightEdge > maxX) {
                maxX = rightEdge;
            }
        });


        const columns = [
            {
                label: "Size", 
                value: 'size', 
                format, 
                x: maxX + 25
            },
            {
                label: "Sublabels", 
                value: 'value', 
                format: (value, d) => d.children ? format(value) : "-", 
                x: maxX + 115
            }
        ];


        for (const {label, value, format, x} of columns) {
            // column headers
            main_group.append("text")
                .attr("dy", "0.32em")
                .attr("y", -nodeSize)
                .attr("x", x)
                .attr("text-anchor", "end")
                .attr("font-weight", "bold")
                .attr("paint-order", "stroke")
                .attr("stroke", background_color)
                .attr("fill", text_color)
                .text(label);

            // column values
            node.append("text")
                .attr("dy", "0.32em")
                .attr("x", x)
                .attr("text-anchor", "end")
                .attr("paint-order", "stroke")
                .attr("stroke",background_color)
                .attr("fill",  d => d.children ? text_color : leaf_text_color )
            .data(root.descendants())
                .text(d => format(d[value], d));
        }
    })


    // Regular zoom/pan behaviour
    const zoom = d3.zoom()
        .filter(function(event) {
            return  !event.altKey; 
        })
        .on("zoom", (event) => {
            main_group.attr("transform", event.transform);
        });

   
    svg.call(zoom);

}