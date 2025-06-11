import * as d3 from "d3";
import tippy, {followCursor} from 'tippy.js';
import 'tippy.js/animations/shift-away.css';
import 'tippy.js/dist/tippy.css';
import { flavors } from "@catppuccin/palette";

// Adapted from https://observablehq.com/@d3/zoomable-circle-packing

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
    const color_scheme = model.get('palette')
    const maxTextChars = model.get('maxTextChars');
    const width = model.get('width')
    
    const height = width;

    const background = flavors[color_scheme].colors.base.hex
    const color0 = flavors[color_scheme].colors.surface1.hex
    const color1 = flavors[color_scheme].colors.sky.hex
    const text_fill = flavors[color_scheme].colors.text.hex
    const text_stroke = flavors[color_scheme].colors.base.hex

    // Create the color scale.
    const color = d3.scaleLinear()
        .domain([1, 5])
        .range([color0,color1])
        .interpolate(d3.interpolateHcl);

    //el.style.backgroundColor = background
  
    // Compute the layout.
    const pack = data => d3.pack()
        .radius(d=>d.size)
        .size([width, height])
        .padding(3)
      (d3.hierarchy(data)
        .sum(d => d.children ? 0 : 1)
        .eachBefore((i => d => d.index = i++)(0))
        .eachBefore((i => d => d.size = d.data.size)(0))
        .sort((a, b) => b.size - a.size));
    
    const root = pack(data);
  
    // Create the SVG container.
    const svg = d3.select(el).append("svg")
        .attr("viewBox", `-${width / 2} -${height / 2} ${width} ${height}`)
        .attr("width", width)
        .attr("height", height)
        .attr("style", `
          max-width: 100%;
          height: auto;
          display: block;
          margin: 5px 5px;
          background: ${background};
          cursor: pointer;`);

    // Append the nodes.
    const node = svg.append("g")
      .selectAll("circle")
      .data(root.descendants().slice(1))
      .join("circle")
        .attr("fill", d => d.children ? color(d.depth) : color1)
        .attr('stroke-width',5)
        .on("mouseover", function(evt,d) { 
          d3.select(this)
          .transition()
          .duration(100)
            .attr("stroke",  text_fill); 
        })
        .on("mouseout", function(evt,d) {
           d3.select(this)
           .transition()
          .duration(100)
             .attr("stroke",  null); 
        })
        .on("click", function(event,d) {return focus !== d && (zoom(event, d), event.stopPropagation())}
      
        )
        .each(function(d){
          tippy(d3.select(this).node(),
          {
            content: d.data.name,
            animation:'shift-away',
            delay:[100, 0],
            followCursor: true,
            plugins: [followCursor],
          })
        })
        
    // Append the text labels.
    const label = svg.append("g")
        .style("font", "4px sans-serif")
        .attr("pointer-events", "none")
        .attr("text-anchor", "middle")
      .selectAll("text")
      .data(root.descendants())
      .join("text")
        .style("fill", text_fill)
        .style("font-weight", 'bold')
        .style("stroke", text_stroke)
        .style("stroke-linejoin", 'round')
        .style("fill-opacity", d => d.parent === root ? 1 : 0)
        .style("stroke-opacity", d => d.parent === root ? 1 : 0)
        .style("display", d => d.parent === root ? "inline" : "none")
        .each(function(d) {

            const text = d3.select(this);
            d.label =text;
            // Split the node's name into words
            const words = shortenString(d.data.name, maxTextChars).split(/\s+/);
            const middle = (words.length-1)/2;
            
            // Clear any existing tspans
            text.selectAll("tspan").remove();

            // Create a tspan for each word
            text
              .selectAll("tspan")
              .data(words)
              .enter()
              .append("tspan")
                .text(w => w)
                .attr("x", 0)
                .attr("dy", (d, i) => i === 0 ? `-${middle*1.2}em` : "1.2em");
        });

    // Create the zoom behavior and zoom immediately in to the initial focus node.
    svg.on("click", (event) => zoom(event, root));
    let focus = root;
    let view;

    requestAnimationFrame(()=>
      zoomTo([focus.x, focus.y, focus.r * 2], focus)
    );

    function zoomTo(v, focus) {
      const k = width / v[2];

      view = v;

      label.attr("transform", d => `translate(${(d.x - v[0]) * k},${(d.y - v[1]) * k})`);
      node.attr("transform", d => `translate(${(d.x - v[0]) * k},${(d.y - v[1]) * k})`);
      node.attr("r", function(d){ d.cr = d.r*k; return d.cr})
      label
        .filter(function(d) { return d.parent === focus })
        .each(function(d,i) {
        const textEl = this;
        fitLabelToCircle(i,textEl, d.cr);
      });

      
    }

    function zoom(event, d) {
      const focus0 = focus;

      focus = d;

      const transition = svg.transition()
          .duration(event.altKey ? 7500 : 750)
          .tween("zoom", d => {
            const i = d3.interpolateZoom(view, [focus.x, focus.y, focus.r * 2]);
            return t => zoomTo(i(t),focus);
          });

      label
        .filter(function(d) { return d.parent === focus || this.style.display === "inline"; })
        .transition(transition)
          .style("fill-opacity", d => d.parent === focus ? 1 : 0)
          .style("stroke-opacity", d => d.parent === focus ? 1 : 0)
          .on("start", function(d) { if (d.parent === focus) this.style.display = "inline"; })
          .on("end", function(d) { 
            if (d.parent !== focus) this.style.display = "none"; 
            
          })
      
      }


}

// Function to fit label within circle
function fitLabelToCircle(i, textElement, radius) {
  let fontSize = 4; 
  const maxFontSize = 100; 
  

  const d3textEl = d3.select(textElement)
  // Set initial font size
  d3textEl.style("font-size", fontSize + "px");
  
  // Loop to increase/decrease font size until it fits
  let bbox
  while (fontSize < maxFontSize) {
    bbox = textElement.getBBox();

    if (bbox.width <= 1.5*radius && bbox.height <= 1.5*radius) {
      // Can try increasing size
      fontSize += 2;
      d3textEl.style("font-size", fontSize + "px");
      d3textEl.style("stroke-width", fontSize/20);
    } else {
      // Too big, revert
      fontSize -= 1 ;
      d3textEl.style("font-size", fontSize + "px");
      d3textEl.style("stroke-width", fontSize/20);
      d3textEl.style('display', fontSize/20);

      d3textEl.style("display", fontSize>3 ? "inline" : "none")
      break;
    }
  }
  

}


