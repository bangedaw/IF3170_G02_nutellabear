// Render TeX to pure SVG paths with MathJax (no fonts, no <use>, no href), sized in px for a given font size.
import { mathjax } from "mathjax-full/js/mathjax.js";
import { TeX } from "mathjax-full/js/input/tex.js";
import { SVG } from "mathjax-full/js/output/svg.js";
import { liteAdaptor } from "mathjax-full/js/adaptors/liteAdaptor.js";
import { RegisterHTMLHandler } from "mathjax-full/js/handlers/html.js";
import { AllPackages } from "mathjax-full/js/input/tex/AllPackages.js";

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const document = mathjax.document("", { InputJax: new TeX({ packages: AllPackages }), OutputJax: new SVG({ fontCache: "none" }) });

export const INK = "#18212f";

/** @returns {{ svg: string, width: number, height: number }} inner <svg> sized in px at `fontSize` px per em. */
export function texToSvg(tex, fontSize = 16, display = false) {
  const node = document.convert(tex, { display });
  const markup = adaptor.innerHTML(node);
  const svg = /<svg\b[\s\S]*<\/svg>/.exec(markup)[0];
  const view = /viewBox="([-\d.]+) ([-\d.]+) ([-\d.]+) ([-\d.]+)"/.exec(svg);
  const vbW = Number(view[3]), vbH = Number(view[4]);
  const width = +(vbW / 1000 * fontSize).toFixed(2);
  const height = +(vbH / 1000 * fontSize).toFixed(2);
  const cleaned = svg
    .replace(/ xmlns:xlink="[^"]*"/g, "")
    .replace(/ style="[^"]*"/g, "")
    .replace(/ role="img"| focusable="false"| aria-hidden="true"| aria-labelledby="[^"]*"/g, "")
    .replace(/currentColor/g, INK)
    .replace(/ width="[^"]*"/, "").replace(/ height="[^"]*"/, "")
    .replace(/<title[\s\S]*?<\/title>/g, "");
  return { svg: cleaned, width, height };
}

/** Nested <svg> placed so the formula is centered on (cx, cy). */
export function placeMath(tex, cx, cy, fontSize = 16, display = false) {
  const { svg, width, height } = texToSvg(tex, fontSize, display);
  const x = +(cx - width / 2).toFixed(2), y = +(cy - height / 2).toFixed(2);
  return svg.replace("<svg", `<svg x="${x}" y="${y}" width="${width}" height="${height}"`);
}
