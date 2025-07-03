import * as d3 from "d3";
import tippy, {followCursor} from 'tippy.js';
import 'tippy.js/animations/shift-away.css';
import 'tippy.js/dist/tippy.css';
import { flavors } from "@catppuccin/palette";
import './widget.css'

// Adapted from https://observablehq.com/@d3/radial-tree/2

export default { render }

function update_tree_layout(radius, data){

    // Create a radial tree layout. The layout’s first dimension (x)
    // is the angle, while the second (y) is the radius.
    let tree = d3.tree()
        .size([2 * Math.PI, radius])
        .separation((a, b) => (a.parent == b.parent ? 1 : 2) / a.depth);

    // Sort the tree and apply the layout.
    let root = tree(d3.hierarchy(data)
        .sort((a, b) => d3.ascending(a.data.name, b.data.name)));

    return root;

}

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
    const color_scheme = model.get('palette')
    const width = model.get('width');
    const maxTextChars = model.get('maxTextChars');

    console.log(width)

    const background_color = flavors[color_scheme].colors.base.hex
    const edge_color = flavors[color_scheme].colors.subtext1.hex
    const node_color = flavors[color_scheme].colors.peach.hex
    const leaf_node_color = flavors[color_scheme].colors.yellow.hex
    const text_color = flavors[color_scheme].colors.text.hex
    const hover_text_color = flavors[color_scheme].colors.blue.hex

    // Specify the chart’s dimensions.
    const height = width;
    const cx = width * 0.5;
    const cy = height * 0.5;
    let radius = Math.min(width, height) / 2 - 10;

    //el.style.backgroundColor = background_color

    // Creates the SVG container.
    const svg = d3.select(el).append("svg")
        .attr("id","main_svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [-cx, -cy, width, height])
        .attr("style", "max-width:: 100%; height: auto; font: 10px sans-serif;cursor: pointer;")
        .style("background-color", background_color);

    console.log(svg.attr('width'))

    // Rect to handle Alt Zoom behaviour
    const altZoomRect = svg.append("rect")
        .attr('width',width)
        .attr('height',height)
        .attr('x',-cx).attr('y',-cy)
        .attr("fill", "none")
        .attr("stroke", "none")
        .style("pointer-events", "all");

    const main_group = svg.append("g").style('pointer-events', 'none');

    const links = main_group.append("g")
    const nodes = main_group.append("g")
    const labels = main_group.append("g")

    function update_positions( root ){

        links
            .attr("fill", "none")
            .attr("stroke", edge_color)
            .attr("stroke-opacity", 0.4)
            .attr("stroke-width", 1.5)
            .selectAll("path")
            .data(root.links(), function(d){return d.source.data.name+d.target.data.name})
            .join("path")
            .attr("d", d3.linkRadial()
                .angle(d => d.x)
                .radius(d => d.y));

        nodes
            .selectAll("circle")
            .data(root.descendants(), function(d){ return d.data.name})
            .join("circle")
            .attr("transform", d => `rotate(${d.x * 180 / Math.PI - 90}) translate(${d.y},0)`)
            .attr("fill", d => d.children ? node_color : leaf_node_color)
            .attr("r", 3);

        labels
            .attr("stroke-linejoin", "round")
            .attr("stroke-width", 3)
            .selectAll("text")
            .data(root.descendants(), function(d){ return d.data.name})
            .join("text")
            .attr("transform", d => `rotate(${d.x * 180 / Math.PI - 90}) translate(${d.y},0) rotate(${d.x >= Math.PI ? 180 : 0})`)
            .attr("dy", "0.31em")
            .attr("x", d => d.x < Math.PI === !d.children ? 6 : -6)
            .attr("text-anchor", d => d.x < Math.PI === !d.children ? "start" : "end")
            .attr("paint-order", "stroke")
            .attr("stroke", background_color)
            .attr("fill", text_color)
            .style("--hover-fill", hover_text_color)
            .text(d => shortenString(d.data.name, maxTextChars))
            .attr("class", "node-label")
            .style('pointer-events', 'all')
            .each(function(d){
              tippy(d3.select(this).node(),
              {
                content: d.data.name,
                animation:'shift-away',
                delay:[100, 0],
                followCursor: true,
                plugins: [followCursor],
              })
        });
        
    }

    const maxIters = 10;
    let i = 0;
    function plot(){
        const root = update_tree_layout(radius, data);
        update_positions(root);

        const collisions = flagLabelCollisions(labels,50)
        if (collisions && i<maxIters){
            i++;
            radius *= 1.25;
            requestAnimationFrame(plot)

        }

    }
    requestAnimationFrame(plot)
 

    // Regular zoom/pan behaviour
    const zoom = d3.zoom()
        .filter(function(event) {
            return  !event.altKey; 
        })
        .on("zoom", (event) => {
            main_group.attr("transform", event.transform);
        });

    // On Alt keypress, scroll redraws tree with different radius.
    // Use this to spread the labels if they overlap
    const zoomAlt = d3.zoom()
        .filter(function(event) {
            return event.altKey; 
        })
        .on("zoom", (event) => {

            const new_radius = event.transform.k*radius;
            const root = update_tree_layout(new_radius, data);
            update_positions(root);

        });

    svg.call(zoom);
    altZoomRect.call(zoomAlt);


}


// Geometry helper
/**
 * Returns true if the <text> element `el` touches or overlaps the circle
 * centred on (cx, cy) with radius r – all expressed in the *root* SVG
 * user-coordinate system.
 */
function labelCollidesCircle(el, cx = 0, cy = 0, r = 50) {
  if (!(el instanceof SVGGraphicsElement)) return false;

  const svg       = el.ownerSVGElement;
  if (!svg) return false;

  const elCTM     = el.getCTM();      // element → viewport
  const rootCTM   = svg.getCTM();     // root   → viewport
  if (!elCTM || !rootCTM) return false;

  // element → root-user matrix
  const m = rootCTM.inverse().multiply(elCTM);

  // local axis-aligned bbox
  const bb = el.getBBox();
  const makePt = (x, y) => {
    const p = svg.createSVGPoint();
    p.x = x; p.y = y;
    return p.matrixTransform(m);      // → root-user space
  };

  const pts = [
    makePt(bb.x, bb.y),
    makePt(bb.x + bb.width, bb.y),
    makePt(bb.x + bb.width, bb.y + bb.height),
    makePt(bb.x, bb.y + bb.height)
  ];

  const r2 = r * r;

  // 1. Any corner inside the circle?
  if (pts.some(p => (p.x - cx) ** 2 + (p.y - cy) ** 2 <= r2)) return true;

  // 2. Is the circle’s centre inside the label’s rectangle?
  let inside = true;
  for (let i = 0; i < 4; ++i) {
    const a = pts[i], b = pts[(i + 1) % 4];
    if ((b.x - a.x) * (cy - a.y) - (b.y - a.y) * (cx - a.x) < 0) {
      inside = false; break;
    }
  }
  if (inside) return true;

  // 3. Distance from circle centre to any edge ≤ r ?
  function dist2PointToSeg(px, py, ax, ay, bx, by) {
    const vx = bx - ax, vy = by - ay;
    const wx = px - ax, wy = py - ay;
    const t  = Math.max(0, Math.min(1, (vx * wx + vy * wy) / (vx * vx + vy * vy)));
    const dx = wx - t * vx, dy = wy - t * vy;
    return dx * dx + dy * dy;
  }
  for (let i = 0; i < 4; ++i) {
    const a = pts[i], b = pts[(i + 1) % 4];
    if (dist2PointToSeg(cx, cy, a.x, a.y, b.x, b.y) <= r2) return true;
  }

  return false;
}


function flagLabelCollisions(labels,radius = 50) {
  const cx = 0, cy = 0;
  const v = labels.selectAll("text")
        .nodes().some(function (d,i) {
            let collide;
            if (i===0){
                collide = false
            } else {
                collide = labelCollidesCircle(d, cx, cy, radius);
            };
            return collide

        });

    return v
}