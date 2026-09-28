// Rock button: turns an existing <button> into a slab of stone with its label carved in.
//
//   <script src="fx/rock.js"></script>
//   const fx = ElementalFX.attachRock(document.querySelector('#delete'), { material: 'slate' });   // fx.detach() restores it
//
// A classic script (not a module) so pages work straight from file:// too.
//
// The <button> stays the real control (focus, keyboard, form submission, click handlers); this only
// draws it, in a transparent canvas laid over the page around it. Clicking the slab is the button's
// click and cracks it into chunks. The chunks can be dragged and thrown; one flung hard enough into
// another smashes it into smaller pieces (a wall never breaks anything). A few seconds after the
// last touch every fragment glides back into its place and the slab is whole again. Dragging never
// counts as a click. If WebGL2 with float render targets is unavailable, the button is left untouched.
//
// Options:
//   material: 'granite' (default) | 'slate' | 'basalt' | 'sandstone' | 'marble' | 'limestone' | 'concrete'
//             | 'auto' (a stone tinted from the button's own background colour)
//             | { family, base, exact, base2, grain, contrast, mottle, variation, vein, veinAmount, veinScale,
//                 speckle, speckleAmount, mineral, mineralAmount, bands, bandMode ('cleavage'), crossBedding,
//                 layerStep, pits, grainRelief, roughness, sheen, clear, fracture, wear, weather }
//               Colours are CSS colours. A bare { base } is a plain fine-grained stone in that colour;
//               with family: 'granite' | 'marble' | 'slate' | ... that stone is re-coloured around it. A base
//               is softened toward what stone can be unless exact: true.
//   labelColor: CSS colour filled into the carved letters, 'none' to leave them bare, or 'auto' (default).
//   margin: CSS px of stage around the button that pieces may travel in (default max(110, 1.8 x height)).
//   regroupAfter: seconds after the last touch before the pieces go home (default 2.5).
//   zIndex: of the stage canvas (default 10).   sound: false to stay silent.
//
// fx.setMaterial(m) changes the stone live; fx.reset() sends the pieces home now (fx.reset(true): at once).

(() => {
'use strict';

const TAU = Math.PI * 2;
const rand = (a, b) => a + Math.random() * (b - a);
const clamp = (v, a, b) => v < a ? a : v > b ? b : v;
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');

// ---------------------------------------------------------------- tuning (lengths in CSS px, time in s)
const LEAN = 0.2;                 // how much of a south wall the camera sees, per px of thickness
const LIGHT = (() => { const v = [-0.6, -0.5, 0.95], l = Math.hypot(...v); return v.map(x => x / l); })();  // toward the sun, y down
const SHADOW_DIR = [-LIGHT[0] / Math.hypot(LIGHT[0], LIGHT[1]), -LIGHT[1] / Math.hypot(LIGHT[0], LIGHT[1])];
const SHADOW_RUN = Math.hypot(LIGHT[0], LIGHT[1]) / LIGHT[2];   // shadow length per px of height
const H_STEP = 1 / 360;           // physics substep
const MAX_SUBSTEPS = 15;
const SOLVE_ITERS = 3, RELAX_ITERS = 1;
const GROUND_DECEL = 1050;        // Coulomb friction against the ground (mu * g)
const MAX_SPEED = 3800, MAX_SPIN = 45;
const MU = 0.55, RESTITUTION = 0.1, REST_THRESH = 70;
const SLOP = 0.35, PUSH_MAX = 180, DEEP_PUSH = 1;
const CONTACT_HZ = 30, CONTACT_ZETA = 10;
const DRAG_HZ = 7, DRAG_ZETA = 0.95, DRAG_ACCEL = 42000;
const BREAK_V = 600;              // normal impact speed that cracks a piece of BREAK_AREA
const BREAK_AREA = 3000;
const MIN_SPLIT = 3;              // pieces of fewer grains are gravel
const COOL = 0.2;                 // a fresh fragment can neither break nor break anything for this long
const CRACK_G = 2.2e5;            // energy a new crack absorbs per px² of fresh face (mass = area)
const BLOW_R = 150, BLOW_T = 0.15; // impacts this close in space and time belong to one blow, as do
const BLOW_CARRY = 0.7;           // later impacts, for this long, of any piece that took part in it...
const BLOW_E0 = 3.6e8;            // ...which may make 2 + energy / BLOW_E0 new pieces in all
const BLOW_MAX = 10;
const MAX_CHIPS = 60;
const DRAG_START = 6;             // CSS px of travel before a press on the slab becomes a drag
const MAXB = 512, SEGW = 512;     // piece outline texture: one row of segments per live piece

// ---------------------------------------------------------------- shaders
const NOISE = `
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*.1031); p3 += dot(p3, p3.yzx+33.33); return fract((p3.x+p3.y)*p3.z); }
vec2 hash22(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*vec3(.1031,.1030,.0973)); p3 += dot(p3, p3.yzx+33.33); return fract((p3.xx+p3.yz)*p3.zy); }
float vnoise(vec2 p){
  vec2 i = floor(p), f = fract(p);
  vec2 u = f*f*(3.-2.*f);
  return mix(mix(hash12(i), hash12(i+vec2(1,0)), u.x), mix(hash12(i+vec2(0,1)), hash12(i+vec2(1,1)), u.x), u.y);
}
float fbm(vec2 p){ float s = 0., a = .5; mat2 m = mat2(1.6, 1.2, -1.2, 1.6); for (int i = 0; i < 5; i++){ s += a*vnoise(p); p = m*p + 11.7; a *= .5; } return s / .96875; }
float fbm3(vec2 p){ float s = 0., a = .5; mat2 m = mat2(1.6, 1.2, -1.2, 1.6); for (int i = 0; i < 3; i++){ s += a*vnoise(p); p = m*p + 11.7; a *= .5; } return s / .875; }
// x = distance to the cell border, y..w = the cell's hashes
vec4 voro(vec2 x){
  vec2 n = floor(x), f = fract(x);
  vec2 mg, mr; float md = 8.;
  for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++){
    vec2 g = vec2(i, j), r = g + hash22(n + g) - f;
    float d = dot(r, r);
    if (d < md){ md = d; mr = r; mg = g; }
  }
  float mb = 8.;
  for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++){
    vec2 g = mg + vec2(i, j), r = g + hash22(n + g) - f;
    vec2 dr = r - mr;
    if (dot(dr, dr) > 1e-5) mb = min(mb, dot(0.5 * (mr + r), normalize(dr)));
  }
  vec2 cid = n + mg;
  return vec4(mb, hash12(cid), hash12(cid + 19.3), hash12(cid + 7.1));
}
`;
const HEAD = `#version 300 es
precision highp float;
precision highp sampler2D;
` + NOISE;

const VS_FULL = `#version 300 es
void main(){ vec2 p = vec2(float((gl_VertexID<<1)&2), float(gl_VertexID&2)); gl_Position = vec4(p*2.-1., 0., 1.); }`;

// Each piece is drawn as its own oriented box; the fragment shader finds the piece's outline from its
// boundary segments (one texture row per piece) and writes height and rock-frame coordinates, so the
// stone pattern and letters sampled later line up across every fracture.
// The camera sits a touch south of straight overhead, so each piece shows a sliver of its south-facing
// walls: a pixel is wall when it is outside the top outline but the outline, swept a little along the
// view's lean, covers it.
// Everything about the slab's ORIGINAL surface (its sawn outline, arris, walls, antialiasing) is taken
// from the analytic outline of the whole slab in the rock frame, never from the piece's own polygon, so
// fragments at home render exactly as the intact slab does; only the fresh breaks are per piece, and
// they fade out with the seam.
const VS_PIECE = `#version 300 es
layout(location=0) in vec2 aCorner;
layout(location=1) in vec4 aX;     // x, y, angle, lift
layout(location=2) in vec4 aBox;   // local bounds
layout(location=3) in vec4 aSeg;   // row, original segs, total segs, seam
layout(location=4) in vec4 aHome;  // centroid in rock frame, id, thickness
layout(location=5) in vec4 aMisc;  // at home (shares one height field with its neighbours)
uniform vec2 uCss;
uniform float uLean;
out vec2 vLocal;
flat out vec4 vSeg;
flat out vec4 vHome;
flat out vec4 vRot;
flat out float vGid;
void main(){
  float side = aHome.w * uLean;
  vec2 lp = mix(aBox.xy - 1.5 - side, aBox.zw + 1.5 + side, aCorner);
  float c = cos(aX.z), s = sin(aX.z);
  vec2 wp = aX.xy + vec2(c*lp.x - s*lp.y, s*lp.x + c*lp.y);
  gl_Position = vec4(wp.x / uCss.x * 2. - 1., 1. - wp.y / uCss.y * 2., 0., 1.);
  vLocal = lp; vSeg = aSeg; vHome = vec4(aHome.xyz, aX.w); vRot = vec4(c, s, aHome.w, side);
  vGid = aMisc.x > 0.5 ? 0. : aHome.z;
}`;
const FS_PIECE = HEAD + `
uniform sampler2D uSegs;
uniform sampler2D uText;
uniform vec4 uTextBox;
uniform vec3 uSlab;       // half width, half height, corner radius of the whole slab
uniform float uDpr, uWear;
uniform sampler2D uStone; // alpha: the material's surface relief
uniform vec4 uStoneBox;
in vec2 vLocal;
flat in vec4 vSeg;
flat in vec4 vHome;
flat in vec4 vRot;
flat in float vGid;
layout(location=0) out vec4 g0;   // height, rock-frame xy, group id*2 + wall
layout(location=1) out vec4 g1;   // fresh, carve, coverage, distance to the arris
layout(location=2) out vec4 g2;   // height, underside, arris / wall outward normal (screen)
bool cross1(vec2 a, vec2 b, vec2 p){ return ((a.y > p.y) != (b.y > p.y)) && (p.x < a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y)); }
// signed distance to the slab's original outline (negative inside) and its outward normal
float sdSlab(vec2 o, out vec2 n){
  vec2 s = vec2(o.x < 0. ? -1. : 1., o.y < 0. ? -1. : 1.);
  vec2 q = abs(o) - (uSlab.xy - uSlab.z);
  if (q.x > 0. && q.y > 0.){ float l = length(q); n = s * q / max(l, 1e-6); return l - uSlab.z; }
  if (q.x > q.y){ n = vec2(s.x, 0.); return q.x - uSlab.z; }
  n = vec2(0., s.y); return q.y - uSlab.z;
}
void main(){
  vec2 p = vLocal;
  float cr = vRot.x, sr = vRot.y, T = vRot.z, sl = vRot.w;
  vec2 V = vec2(sr, cr);                 // screen-down in the piece's frame
  vec2 pm = p - V * (sl * 0.5), pq = p - V * sl;
  int row = int(vSeg.x), nO = int(vSeg.y), nT = int(vSeg.z);
  float dF = 1e8, dN = 1e8, dQ = 1e8;
  vec2 qN = vec2(0.), qF = vec2(0.); bool fN = false;
  bool inP = false, inM = false, inQ = false;
  for (int i = 0; i < nT; i++){
    vec4 s = texelFetch(uSegs, ivec2(i, row), 0);
    vec2 a = s.xy, b = s.zw, e = b - a;
    float ee = max(dot(e, e), 1e-9);
    vec2 w = p - a;
    vec2 q = w - e * clamp(dot(w, e) / ee, 0., 1.);
    float dd = dot(q, q);
    if (i >= nO && dd < dF){ dF = dd; qF = q; }
    if (dd < dN){ dN = dd; qN = q; fN = i >= nO; }
    if (cross1(a, b, p)) inP = !inP;
    if (cross1(a, b, pm)) inM = !inM;
    if (cross1(a, b, pq)) inQ = !inQ;
    // the wall's foot is the south-facing part of the outline, shifted down by the visible wall height
    if (dot(vec2(e.y, -e.x), V) > 0.){
      vec2 wq = pq - a, qq = wq - e * clamp(dot(wq, e) / ee, 0., 1.);
      dQ = min(dQ, dot(qq, qq));
    }
  }
  dN = sqrt(dN); dQ = sqrt(dQ); dF = sqrt(dF);
  float px = 1. / uDpr;
  float seam = vSeg.w;
  float z = vHome.w;
  vec2 o = p + vHome.xy;
  vec2 gn; float sd = sdSlab(o, gn);
  // Away from its breaks, everything about a pixel (top or wall, drawn or not, its antialiasing) is
  // decided by the analytic slab, exactly as for the intact slab, never by this piece's polygonal copy
  // of that outline: pieces simplify the shared outline slightly differently, and at the swap any such
  // difference would show. "Away" has to include the wall a break shows below itself: those pixels are
  // inside the slab too, but they are this piece's break face, not its top. Once sealed, a neighbour's
  // identical top covers them either way. Near a break, the piece's own polygon decides.
  // Pixels just outside the silhouette are drawn too, with partial coverage, so the outline is
  // antialiased on both sides of the true edge.
  // (sealed, there are no breaks: every piece is then exactly its share of the intact slab)
  float breakReach = seam > 0.001 ? 1.75 * px + sl : -1.;
  bool fringe = false; float fx = 0.;
  bool inPoly = inP;
  vec2 dn0;
  float sw = min(sdSlab(o - V * (sl * 0.5), dn0), sdSlab(o - V * sl, dn0));
  if (dF > breakReach){
    // only stone that is this piece's: its polygon, its sweep, or the sliver its chords cut off the arc
    if (!inP && !inM && !inQ && min(dN, dQ) > 1.5 * px) discard;
    if (sd < 0.) inP = true;
    else if (sw < 0.) inP = false;
    else {
      fx = min(sd, sw);
      // zero coverage from here on; the cut sits off the half-pixel grid so float noise between a piece
      // and the intact slab can't put the same pixel on opposite sides of it
      if (fx > 0.49 * px) discard;
      fringe = true;
      inP = sd <= sw;
    }
  } else if (!inP && !inM && !inQ){
    fx = min(dN, dQ);
    if (fx > 0.75 * px) discard;
    fringe = true;
    if (dN <= dQ) inP = true;
  }
  float bias = fringe ? 0.001 : 0.;

  if (!inP){
    // wall: its height falls from the arris to the ground across the visible sliver. Inside the slab's
    // outline it can only be the side of a fresh break, which a neighbour at home covers.
    vec2 nl; float hw, fr, cov;
    if (sd < -0.3 * px){
      if (seam < 0.01) discard;
      nl = dN > 1e-4 ? qN / dN : V;
      float face = max(dot(nl, V), 0.08);
      hw = (T - 0.6) * (1. - clamp(dN / (sl * face), 0., 1.)) * seam;
      fr = seam;
      cov = fringe ? clamp(0.5 - fx * uDpr, 0., 1.) : clamp(0.5 + dQ * uDpr, 0., 1.);
    } else {
      nl = gn;
      float face = max(dot(nl, V), 0.08);
      hw = (T - 0.6) * (1. - clamp(max(sd, 0.) / (sl * face), 0., 1.));
      fr = 0.;
      cov = clamp(0.5 - sw * uDpr, 0., 1.);
    }
    vec2 nw = vec2(cr * nl.x - sr * nl.y, sr * nl.x + cr * nl.y);
    g0 = vec4(hw + z, o, vGid * 2. + 1.);
    g1 = vec4(fr, 0., cov, 0.);
    // the shadow and occlusion pass sees the edge's coverage as height, so a sloping, stair-stepped
    // edge casts smooth contact shade instead of a dark tick at every step
    g2 = vec4((hw + z) * cov, z, nw);
    gl_FragDepth = 0.5 - (hw + z) * 0.002 + bias;
    return;
  }

  float dO = max(-sd, 0.);
  // sawn arris, eased round so it carries a band of light on the sunny side and a line of shade on
  // the other, with the odd old knock taken out of it; never wider than a thin piece can carry
  float knock = clamp(uWear, 0., 1.) * smoothstep(0.66, 0.9, vnoise(o * 0.085 + 17.));
  float cw = min(2.0 + 0.5 * vnoise(o * 0.35) + 2.2 * knock * max(uWear, 1.), 0.1 * T + 0.3);
  float pr = clamp(1. - dO / cw, 0., 1.);
  float hO = T - cw * mix(1. - sqrt(1. - pr * pr), pr * pr * 1.2, knock) - knock * 0.6 * (vnoise(o * 1.5) - 0.35) * pr;

  // fresh break: a sharp arris about a device pixel wide; bigger pieces sometimes lose a shallow
  // flat flake off the top along the break, with a crisp step at its inner edge
  float spall = T > 12. ? smoothstep(0.68, 0.9, vnoise(o * 0.2 + mod(vHome.z, 997.) * 7.31)) : 0.;
  float wF = 2.6 * spall;
  float hF = T - seam * (0.8 * (1. - smoothstep(0., 1.2 * px, dF)) + 0.7 * spall * (1. - smoothstep(wF - px, wF, dF)));

  float h = min(hO, hF);
  // honed face: slight undulation and the odd plucked crystal
  h += 0.3 * (fbm3(o * 0.018) - 0.5) + 0.1 * (vnoise(o * 0.15) - 0.5);
  h -= 0.25 * smoothstep(0.86, 0.97, vnoise(o * 0.42 + 5.)) * clamp(dO / 3., 0., 1.);
  h += (textureLod(uStone, (o - uStoneBox.xy) / uStoneBox.zw, 0.).a - 0.5) * 4. * clamp(dO / 2., 0., 1.);

  // V-cut letters, only in stone this piece actually has: pixels outside its break polygon are its
  // antialiasing fringe, and must not carry letters that belong to the neighbour's side of the crack
  vec2 tuv = (o - uTextBox.xy) / uTextBox.zw;
  vec2 lt = (tuv.x > 0. && tuv.y > 0. && tuv.x < 1. && tuv.y < 1.) ? texture(uText, tuv).rg : vec2(0., 60.);
  if (seam > 0.001 && !inPoly && dF <= breakReach) lt = vec2(0., 60.);
  float ld = lt.x;
  float halo = ld > 0. ? 0. : 1. - smoothstep(0.6, 3.2, lt.y);
  float cut = min(ld * 1.8, min(4., T * 0.3));
  h -= cut;

  float freshZone = seam * (1. - smoothstep(0., wF + 1., dF));
  // coverage: the original outline from the analytic slab, fresh breaks from this piece's polygon;
  // south-facing edges run on into their wall, so they need no antialiasing against the ground
  float covO = (dot(gn, V) > 0.15 && sl > 0.3) ? 1. : clamp(0.5 - sd * uDpr, 0., 1.);
  float covF = 1.;
  vec2 fo = dF > 1e-4 ? -qF / dF : vec2(0.);
  if (seam > 0.001){
    if (fringe) covF = fN ? clamp(0.5 - fx * uDpr, 0., 1.) : 1.;
    else covF = (dot(fo, V) > 0.15 && sl > 0.3) ? 1. : clamp(0.5 + dF * uDpr, 0., 1.);
    covF = mix(1., covF, seam);
  }
  // the arris hairline follows whichever edge is nearer: the original one, or a break still open
  float dA = dO; vec2 an = gn; float aw = 0.5;
  if (seam > 0.001 && dF < dO){ dA = dF; an = fo; aw = 0.5 * seam; }
  vec2 os = vec2(cr * an.x - sr * an.y, sr * an.x + cr * an.y) * aw;
  g0 = vec4(h + z, o, vGid * 2.);
  // carve and the halo around the letters share a channel: a pixel is only ever one of them
  float cv = smoothstep(0., 1.1, cut);
  g1 = vec4(freshZone, cv > 0. ? cv : -halo, covO * covF, dA);
  g2 = vec4((h + z) * covO * covF, z, os);
  gl_FragDepth = 0.5 - (h + z) * 0.002 + bias;
}`;

// Tiny chips: a random shard (intersection of half-planes) standing a little above whatever is below.
const VS_CHIP = `#version 300 es
layout(location=0) in vec2 aCorner;
layout(location=1) in vec4 aX;     // x, y, z, angle
layout(location=2) in vec4 aP;     // size, seed, id, alpha
layout(location=3) in vec4 aO;     // rock-frame point it broke from
uniform vec2 uCss;
out vec2 vL;
flat out vec4 vC;
flat out vec2 vO;
void main(){
  vec2 lp = (aCorner * 2. - 1.) * 1.2;
  float c = cos(aX.w), s = sin(aX.w);
  vec2 wp = aX.xy + vec2(c*lp.x - s*lp.y, s*lp.x + c*lp.y) * aP.x;
  gl_Position = vec4(wp.x / uCss.x * 2. - 1., 1. - wp.y / uCss.y * 2., 0., 1.);
  vL = lp; vC = vec4(aX.z, aP.xyz); vO = aO.xy;
}`;
const FS_CHIP = HEAD + `
uniform float uDpr;
in vec2 vL;
flat in vec4 vC;
flat in vec2 vO;
layout(location=0) out vec4 g0;
layout(location=1) out vec4 g1;
layout(location=2) out vec4 g2;
void main(){
  float sd = -1e9;
  for (int i = 0; i < 4; i++){
    float a = vC.z * 40. + float(i) * 1.71 + hash12(vec2(vC.z * 91., float(i))) * 0.9;
    sd = max(sd, dot(vL, vec2(cos(a), sin(a))) - (0.5 + 0.45 * hash12(vec2(float(i), vC.z * 57.))));
  }
  if (sd > 0.) discard;
  float t = clamp(-sd * 2.5, 0., 1.);
  float h = vC.x + vC.y * (0.3 + 0.35 * t);
  // the stone it broke from, so each chip carries the minerals of that spot
  vec2 o = vO + vL * vC.y * 0.7;
  g0 = vec4(h, o, 2. * (4000000. + vC.w));
  float cov = clamp(0.5 - sd * vC.y * uDpr, 0., 1.);
  g1 = vec4(0.7, 0., cov, 0.);
  g2 = vec4(h * cov, vC.x, 0., 0.);
  gl_FragDepth = 0.5 - h * 0.002;
}`;

// The stone, baked once per size and material into a texture in the rock's own frame (one texel per
// device pixel, 16 samples each), so crystal edges are antialiased and the pattern costs nothing per
// frame. Scale: 1 CSS px is about a millimetre.
// Granite is a pink Rosa Porriño type: big blocky salmon K-feldspar phenocrysts in a finer groundmass
// of glassy smoky quartz, milky plagioclase, small feldspars and black biotite books, every boundary
// domain-warped so grains interlock. The other stones share one generator driven by uniforms: a
// mottled body of two tones, a crystal mosaic, bedding or cleavage bands, veins, specks and pits.
const FS_STONE = HEAD + `
uniform vec2 uOrg;
uniform float uTexel;
uniform int uKind;
uniform vec3 uBase, uBase2, uVein, uSpeck, uSpeck2;
uniform vec4 uGrain;   // grain size px, grain contrast, mottle amount, mottle scale px
uniform vec4 uVeinP;   // amount, scale px, width, warp
uniform vec4 uSpeckP;  // amount, size px, band amount, band spacing px
uniform vec4 uSurf;    // pits, roughness, sheen, sparkle
uniform vec4 uTint;    // linear colour the stone's mean is moved to, amount
uniform float uMeanLum, uBandAng, uSeed, uGScale;
uniform vec4 uVarP;    // value variation amount, scale px
uniform vec4 uMin;     // a second mineral: linear colour, fraction of grains
uniform vec4 uBandP;   // band mode (0 bedding, 1 cleavage layers), cross-bedding angle, layer step height px
uniform vec4 uRelP;    // relief px: grain, (unused), pits
uniform vec4 uSlabW;   // slab half size, corner radius, weathering
layout(location=0) out vec4 oA;
layout(location=1) out vec4 oM;
vec3 lin(vec3 s){ return pow(s, vec3(2.2)); }
vec3 hsv(float h, float s, float v){ vec3 p = abs(fract(h + vec3(0., 2. / 3., 1. / 3.)) * 6. - 3.); return v * mix(vec3(1.), clamp(p - 1., 0., 1.), s); }
float gcell(vec2 x, float S, out vec2 id1, out vec2 id2, out vec2 rel){
  vec2 p = x / S, n = floor(p), f = fract(p);
  float d1 = 8., d2 = 8.; vec2 r1 = vec2(0.), r2 = vec2(1.);
  id1 = n; id2 = n;
  for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++){
    vec2 g = vec2(i, j), cc = n + g;
    vec2 r = g + 0.5 + 0.92 * (hash22(cc) - 0.5) - f;
    float d = dot(r, r);
    if (d < d1){ d2 = d1; id2 = id1; r2 = r1; d1 = d; id1 = cc; r1 = r; }
    else if (d < d2){ d2 = d; id2 = cc; r2 = r; }
  }
  rel = -r1 * S;
  return dot(0.5 * (r1 + r2), normalize(r2 - r1)) * S;
}
float pheno(vec2 x, float S, out vec2 kid, out vec2 krel, out vec2 kext){
  vec2 n = floor(x / S);
  float best = 1e9;
  kid = n; krel = vec2(0.); kext = vec2(1.);
  for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++){
    vec2 cc = n + vec2(i, j);
    if (hash12(cc + 3.1) > 0.6) continue;
    vec2 ctr = (cc + 0.5 + 0.75 * (hash22(cc + 1.7) - 0.5)) * S;
    float an = hash12(cc + 5.3) * 6.2832;
    vec2 ax = vec2(cos(an), sin(an)), d = x - ctr;
    vec2 q = vec2(dot(d, ax), dot(d, vec2(-ax.y, ax.x)));
    float sz = mix(0.7, 1.3, hash12(cc + 8.));
    vec2 ext = vec2(0.44, 0.28 + 0.14 * hash12(cc + 2.9)) * S * sz;
    vec2 e = abs(q) - ext + 1.5;
    float sd = length(max(e, 0.)) + min(max(e.x, e.y), 0.) - 1.5;
    float ca = hash12(cc + 6.1) * 6.2832;
    sd = max(sd, dot(q, vec2(cos(ca), sin(ca))) - mix(0.45, 0.9, hash12(cc + 7.7)) * length(ext));
    ca += 2.2 + hash12(cc + 9.1);
    sd = max(sd, dot(q, vec2(cos(ca), sin(ca))) - mix(0.55, 1., hash12(cc + 4.9)) * length(ext));
    if (sd < best){ best = sd; kid = cc; krel = q; kext = ext; }
  }
  return best;
}
float gkind(vec2 id){
  float t = hash12(id + 0.37), cl = vnoise(id * 0.35 + 40.), bc = vnoise(id * 0.45 + 90.);
  if (hash12(id + 5.9) < 0.03 + 0.16 * smoothstep(0.5, 0.85, bc)) return 3.;
  if (t < 0.38 + 0.1 * cl) return 0.;
  if (t < 0.7) return 1.;
  return 2.;
}
vec3 kspar(vec2 id, float pale){
  float a = hash12(id + 19.3), b = hash12(id + 7.1), c = hash12(id + 2.2);
  return lin(hsv(0.02 + 0.036 * (a - 0.5), clamp(0.34 + 0.2 * (b - 0.5) - pale, 0.05, 0.8), 0.78 + 0.08 * (c - 0.5)));
}
vec3 plag(vec2 id){
  float a = hash12(id + 19.3), b = hash12(id + 7.1);
  return lin(hsv(0.1 + 0.03 * a, 0.05 + 0.06 * b, 0.76 + 0.07 * (a - 0.5)));
}
vec3 quartzBase(vec2 id){ return lin(vec3(0.52, 0.52, 0.54)) * (0.8 + 0.3 * hash12(id + 19.3)); }
vec3 kindColor(float k, vec2 id){ return k < 0.5 ? quartzBase(id) : k < 1.5 ? plag(id) : k < 2.5 ? kspar(id, 0.08) : plag(id); }

void granite(vec2 o, out vec3 c, out float rough, out float glint, out vec2 tilt){
  vec2 w = o + 2.6 * (vec2(fbm3(o * 0.075), fbm3(o * 0.075 + 5.2)) - 0.5) * 2.;
  w += 0.7 * (vec2(vnoise(o * 0.55 + 7.), vnoise(o * 0.55 + 13.)) - 0.5) * 2.;
  vec2 id1, id2, grel;
  float gb = gcell(w, 5.4, id1, id2, grel);
  float k1 = gkind(id1), k2 = gkind(id2);
  vec2 kid, krel, kext;
  float ksd = pheno(w, 25., kid, krel, kext) + 5. * (fbm3(o * 0.13 + 9.) - 0.5) + 1.5 * (vnoise(o * 0.5 + 3.) - 0.5);
  float bio = 0.; vec2 bid = id1;
  if (k1 > 2.5){
    float an = hash12(id1 + 4.4) * 6.2832;
    vec2 ax = vec2(cos(an), sin(an));
    vec2 q = abs(vec2(dot(grel, ax), dot(grel, vec2(-ax.y, ax.x)))) * vec2(1., mix(1., 1.7, hash12(id1 + 6.6)));
    float r = mix(1., 2.3, hash12(id1 + 9.9));
    float hx = max(dot(q, vec2(0.866, 0.5)), q.y) - r;
    bio = 1. - smoothstep(-0.12, 0.12, hx);
    k1 = k2 > 2.5 ? 1. : k2;
    id1 = id2;
  }
  bool inK = ksd < 0.;
  if (inK) bio = 0.;
  float bord;
  if (inK){
    vec2 u = krel / kext;
    c = kspar(kid, 0.);
    float tw = step(0., krel.y + 0.25 * kext.y * (hash12(kid + 4.) - 0.5));
    c *= mix(0.92, 1.06, hash12(kid + 3.) > 0.35 ? tw : 0.5);
    c *= 1. + 0.035 * sin(max(abs(u.x), abs(u.y)) * 20. + hash12(kid) * 6.);
    float per = smoothstep(0.55, 0.9, vnoise(vec2(krel.x * 0.3, krel.y * 3.4) + hash12(kid + 9.) * 50.));
    c = mix(c, lin(vec3(0.86, 0.76, 0.72)), 0.1 * per);
    c *= 0.88 + 0.2 * fbm3(o * 0.6 + hash12(kid + 5.) * 40.);
    c *= 1. + 0.12 * dot(u, vec2(hash12(kid + 11.) - 0.5, hash12(kid + 12.) - 0.5) * 2.);
    c = mix(c, lin(vec3(0.84, 0.72, 0.66)), 0.25 * (1. - smoothstep(0., 1.2, -ksd)));
    rough = 0.32; glint = 0.9;
    tilt = hash22(kid * 1.7 + tw * 3.1) - 0.5;
    bord = -ksd;
  } else if (k1 < 0.5){
    vec3 nb = kindColor(k2, id2);
    float depth = smoothstep(0.3, 2.5, gb);
    c = quartzBase(id1) * mix(0.98, 0.84, depth) * (0.9 + 0.2 * vnoise(o * 0.7 + hash12(id1) * 20.));
    c = mix(c, nb * 0.7, 0.28 * (1. - smoothstep(0., 0.9, gb)));
    float hl = abs(dot(grel, vec2(cos(hash12(id1) * 6.28), sin(hash12(id1) * 6.28))) - 0.6);
    c += lin(vec3(0.62)) * 0.1 * (1. - smoothstep(0., 0.12, hl)) * step(0.6, hash12(id1 + 2.));
    rough = 0.08; glint = 0.35;
    tilt = hash22(id1 * 3.3) - 0.5;
    bord = gb;
  } else if (k1 < 1.5){
    vec3 p0 = plag(id1);
    float an = hash12(id1 + 1.) * 6.2832;
    c = p0 * (1. + 0.035 * sin(dot(o, vec2(cos(an), sin(an))) * 3.8) * vnoise(o * 0.5 + an));
    c = mix(c, lin(vec3(0.7, 0.71, 0.64)), 0.35 * (1. - smoothstep(0.3, 1.4, length(grel) / 2.3)) * step(0.45, hash12(id1 + 5.)));
    c *= 0.92 + 0.12 * fbm3(o * 0.8 + hash12(id1) * 30.);
    rough = 0.36; glint = 0.5;
    tilt = hash22(id1 * 5.9) - 0.5;
    bord = gb;
  } else {
    c = kspar(id1, 0.1) * (0.9 + 0.14 * fbm3(o * 0.7 + hash12(id1) * 30.));
    rough = 0.32; glint = 0.7;
    tilt = hash22(id1 * 2.3) - 0.5;
    bord = gb;
  }
  c *= 0.965 + 0.035 * smoothstep(0., 0.5, bord);
  float fl = (1. - smoothstep(0.15, 0.45, bord)) * step(0.9, vnoise(o * 0.9 + 70.)) * (inK ? 0. : 1.);
  bio = max(bio, fl * 0.85);
  vec3 mica = mix(lin(vec3(0.045, 0.04, 0.037)), lin(vec3(0.2, 0.14, 0.08)), pow(hash12(bid + 2.), 4.));
  c = mix(c, mica, bio);
  rough = mix(rough, 0.15, bio);
  glint = mix(glint, 1., bio);
  if (bio > 0.5) tilt = hash22(bid * 7.7) - 0.5;
  glint *= smoothstep(0.25, 0.9, bord) * (1. - bio) + bio;
  c *= 1. + 0.06 * (fbm3(o * 0.012 + 3.) - 0.5);
}

void generic(vec2 o, out vec3 c, out float rough, out float glint, out vec2 tilt, out float rel){
  vec2 w = o + 5. * (vec2(fbm3(o * 0.035), fbm3(o * 0.035 + 5.2)) - 0.5);
  // body: broad clouds of the two base tones
  float m = fbm(w / uGrain.w + 3.7);
  c = mix(uBase, uBase2, clamp(0.5 + (m - 0.5) * 2.6 * uGrain.z, 0., 1.));
  rel = 0.;
  // value drifting over three scales at once: a single even tone is what makes a stone read as plastic
  if (uVarP.x > 0.){
    float v = (fbm3(w / uVarP.y + 1.3) - 0.5) + 0.7 * (fbm3(w / (uVarP.y * 0.28) + 7.1) - 0.5) + 0.5 * (vnoise(o / (uVarP.y * 0.07) + 3.3) - 0.5);
    c *= 1. + uVarP.x * 1.6 * v;
  }
  if (uSpeckP.z > 0.){
    vec2 dir = vec2(cos(uBandAng), sin(uBandAng));
    // cleavage planes run nearly straight; bedding wanders
    float s = uBandP.x < 0.5 ? dot(o, dir) + 16. * fbm3(o * 0.01) + 2.5 * vnoise(o * 0.07) : dot(o, dir) + 5. * fbm3(o * 0.006) + 0.6 * vnoise(o * 0.05);
    if (uBandP.x < 0.5){
      // bedding: sets of laminae, each set inclined its own way (cross-bedding) between flat bounding
      // surfaces, the tone drifting warmer and cooler over bands several laminae wide
      float setH = uSpeckP.w * 4.;
      float si = floor(s / setH);
      float an = uBandAng + (hash12(vec2(si, 3.7)) - 0.5) * 2. * uBandP.y;
      float s2 = dot(o, vec2(cos(an), sin(an))) + 5. * fbm3(o * 0.03 + si);
      float band = 0.55 * sin(s2 * 6.2832 / uSpeckP.w) + 0.45 * sin(s * 6.2832 / (uSpeckP.w * 3.3) + 1.3);
      c = mix(c, mix(uBase2, uBase, 0.5 + 0.5 * band), uSpeckP.z * 0.5);
      c *= 1. + 0.05 * uSpeckP.z * band;
    } else {
      // cleavage: layers of uneven thickness, each its own tone, each stepping a little in height
      float t = s / uSpeckP.w;
      t += 0.9 * (vnoise(vec2(t * 0.37, 1.7)) - 0.5) + 0.35 * (vnoise(vec2(t * 1.3, 8.1)) - 0.5);
      float li = floor(t), lf = fract(t);
      float tone = hash12(vec2(li, 9.1)) - 0.5;
      c *= 1. + uSpeckP.z * 0.35 * tone;
      c *= 1. - 0.18 * uSpeckP.z * (1. - smoothstep(0., 0.07, lf)) * step(0.45, hash12(vec2(li, 2.3)));
      // a layer is only partly flaked off, so its step comes and goes along it
      float flake = smoothstep(0.35, 0.6, vnoise(vec2(dot(o, vec2(-dir.y, dir.x)) * 0.05, li * 3.1)));
      rel += uBandP.z * (hash12(vec2(li, 4.2)) - 0.5) * flake;
    }
  }
  // the crystal or sand-grain mosaic
  vec4 v = voro(w / uGrain.x);
  c *= 1. + uGrain.y * (v.y - 0.5) * 2.;
  c *= 1. + uGrain.y * 0.6 * (vnoise(o * 1.7 / uGrain.x + 9.) - 0.5);
  rel += uRelP.x * (v.y - 0.5) * smoothstep(0.02, 0.15, v.x);
  // a second mineral: some grains are another colour altogether
  if (uMin.w > 0. && v.w < uMin.w) c = mix(c, uMin.rgb * (0.8 + 0.4 * v.z), 0.85 * smoothstep(0.02, 0.1, v.x));
  tilt = vec2(v.z, v.w) - 0.5;
  rough = uSurf.y;
  glint = uSurf.w * step(0.8, v.z) * smoothstep(0.04, 0.2, v.x);
  if (uVeinP.x > 0.){
    // veins are the isolines of a domain-warped field, so they wander, fork and pinch like real ones;
    // their width is taken in stone px through the field's gradient, and it and their strength drift
    vec2 q = o / uVeinP.y + 11.;
    vec2 wq = q + uVeinP.w * 0.9 * (vec2(fbm(q * 0.9 + 1.7), fbm(q * 0.9 + 8.3)) - 0.5);
    const float e = 0.02;
    float f0 = fbm(wq), fx1 = fbm(wq + vec2(e, 0.)), fy1 = fbm(wq + vec2(0., e));
    float gl0 = max(length(vec2(fx1 - f0, fy1 - f0)) / e / uVeinP.y, 1e-4);
    float d1 = abs(f0 - 0.5) / gl0;
    float wid = uVeinP.z * uVeinP.y * 0.22 * (0.3 + 1.4 * fbm3(q * 1.7 + 4.));
    float str = smoothstep(0.2, 0.6, fbm3(q * 0.8 + 30.));
    float v1 = (1. - smoothstep(wid * 0.3, wid + 0.5, d1)) * (0.55 + 0.45 * str);
    float halo1 = (1. - smoothstep(0., wid * 6. + 5., d1)) * (0.4 + 0.6 * str);
    vec2 q2 = wq * 2.3 + 17.;
    float f2 = fbm3(q2), g2x = fbm3(q2 + vec2(e, 0.)), g2y = fbm3(q2 + vec2(0., e));
    float d2 = abs(f2 - 0.5) / max(length(vec2(g2x - f2, g2y - f2)) / e * 2.3 / uVeinP.y, 1e-4);
    float v2 = (1. - smoothstep(0., wid * 0.4 + 0.3, d2)) * 0.55 * smoothstep(0.45, 0.65, fbm3(q * 0.6 + 21.));
    c = mix(c, mix(c, uVein, 0.45), halo1 * uVeinP.x * 0.8);
    float vein = clamp(v1 + v2, 0., 1.) * uVeinP.x;
    c = mix(c, uVein * (0.8 + 0.3 * vnoise(o * 0.3)), vein);
    glint *= 1. - vein;
  }
  if (uSpeckP.x > 0.){
    // rounded, irregular grains with many small and few large, each with a soft rim, as aggregate or
    // inclusions look once the face is honed through them
    float S = uSpeckP.y * 4.;
    vec2 n = floor(o / S);
    for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++){
      vec2 cc = n + vec2(i, j);
      if (hash12(cc + 3.3) > uSpeckP.x * 2.5) continue;
      vec2 ctr = (cc + 0.2 + 0.6 * hash22(cc)) * S;
      vec2 d = o - ctr;
      float r = uSpeckP.y * mix(0.3, 1.25, pow(hash12(cc + 7.), 2.2));
      float a = atan(d.y, d.x);
      float rr = r * (1. + 0.14 * sin(a * 2. + hash12(cc) * 6.) + 0.08 * sin(a * 3. + hash12(cc + 1.) * 6.) + 0.1 * (vnoise(d * 1.3 / max(r, 0.3) + cc) - 0.5));
      float soft = 0.25 + 0.12 * r;
      float k = 1. - smoothstep(rr - soft, rr + soft, length(d));
      vec3 sc = (hash12(cc + 1.1) < 0.5 ? uSpeck : uSpeck2) * (0.82 + 0.36 * hash12(cc + 2.));
      sc *= 0.9 + 0.2 * smoothstep(rr, 0., length(d));
      c = mix(c, sc, k);
      glint = mix(glint, 0.4 * uSurf.w, k);
    }
  }
  if (uSurf.x > 0.){
    // pits: pores, vesicles or air voids, dark in the middle
    vec2 n = floor(o / 5.);
    for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++){
      vec2 cc = n + vec2(i, j);
      if (hash12(cc + 5.1) > uSurf.x) continue;
      vec2 ctr = (cc + 0.2 + 0.6 * hash22(cc + 4.4)) * 5.;
      float r = 0.35 + 1.2 * pow(hash12(cc + 8.), 3.);
      float d = length(o - ctr) / r;
      float pit = 1. - smoothstep(0.6, 1., d);
      c *= 1. - 0.65 * pit;
      rough = mix(rough, 1., pit); glint *= 1. - pit;
      rel -= uRelP.z * pit;
    }
  }
}
void main(){
  vec3 ca = vec3(0.); vec4 ma = vec4(0.); float t2 = 0., rel = 0.;
  for (int j = 0; j < 4; j++) for (int i = 0; i < 4; i++){
    vec2 o = uOrg + (gl_FragCoord.xy - 0.5 + (vec2(i, j) + 0.5) / 4.) * uTexel;
    vec3 c; float r, g, rl = 0.; vec2 t;
    if (uKind == 0) granite((o + uSeed) * uGScale, c, r, g, t); else generic(o + uSeed, c, r, g, t, rl);
    ca += c; ma += vec4(r, g, t + 0.5); t2 += dot(t, t); rel += rl;
  }
  ca /= 16.; ma /= 16.; rel /= 16.;
  // weathering: the old outer skin of the slab is a shade darker and duller for a few px in from the
  // sawn edge, unevenly, the way handled stone goes
  {
    vec2 o = uOrg + (gl_FragCoord.xy - 0.5) * uTexel;
    vec2 q = abs(o) - (uSlabW.xy - uSlabW.z);
    float sd = length(max(q, 0.)) + min(max(q.x, q.y), 0.) - uSlabW.z;
    float edge = 1. - smoothstep(0., 3.5 + 3. * fbm3(o * 0.09 + 4.), -sd);
    ca *= 1. - uSlabW.w * edge * (0.1 + 0.22 * fbm3(o * 0.21 + 2.));
    ma.x = mix(ma.x, 1., uSlabW.w * edge * 0.4);
  }
  if (uTint.w > 0.){
    float lum = dot(ca, vec3(0.2126, 0.7152, 0.0722));
    ca = mix(ca, uTint.rgb * (lum / max(uMeanLum, 1e-4)), uTint.w);
  }
  // a texel straddling two crystals averages their facet tilts into one that neither has, which
  // would flash as a bright seam; such texels don't glint
  float tv = t2 / 16. - dot(ma.zw - 0.5, ma.zw - 0.5);
  ma.y *= 1. - smoothstep(0.003, 0.02, tv);
  // alpha carries the surface relief (px, around 0.5) for the piece pass's height field
  oA = vec4(pow(clamp(ca, 0., 1.), vec3(1. / 2.2)), clamp(0.5 + rel * 0.25, 0., 1.));
  oM = ma;
}`;

// Sun visibility and ambient occlusion over the height field, per device pixel. A mip of the height
// field lets pixels with no stone anywhere near skip the march.
const FS_SHADOW = HEAD + `
uniform sampler2D uH;
uniform vec2 uSize;
uniform float uDpr;
uniform vec3 uL;
uniform float uTop;
out vec4 o;
vec2 hs(vec2 dev){ return textureLod(uH, dev / uSize, 0.).xy; }
void main(){
  vec2 dev = gl_FragCoord.xy;
  vec3 L = normalize(uL);
  vec2 d = normalize(L.xy) * vec2(1., -1.) * uDpr;
  float rise = L.z / length(L.xy);
  vec2 uv = dev / uSize;
  float near = textureLod(uH, uv, 6.).x + textureLod(uH, (dev + d * 18.) / uSize, 6.).x;
  if (near <= 0.){ o = vec4(1., 1., 0., 1.); return; }
  float h0 = texelFetch(uH, ivec2(dev), 0).x;
  float j = fract(52.9829189 * fract(dot(dev, vec2(0.06711056, 0.00583715))));
  float vis = 1.;
  for (int i = 0; i < 20; i++){
    float fi = float(i) + j;
    float t = 0.2 + fi * 0.55 + fi * fi * 0.04;
    float ray = h0 + 0.35 + t * rise;
    if (ray > uTop) break;
    vec2 hb = hs(dev + d * t);
    if (ray < hb.y - 0.5) continue;
    vis = min(vis, clamp(0.5 + (ray - hb.x) / (0.3 + t * 0.08), 0., 1.));
  }
  float occ = 0.;
  for (int i = 0; i < 12; i++){
    float a = float(i) * 2.39996 + j * 6.283;
    float r = 0.6 + float(i) * 1.5 + j;
    vec2 hb = hs(dev + vec2(cos(a), sin(a)) * r * uDpr);
    occ += clamp((hb.x - h0) / (r * 0.8 + 1.5), 0., 1.) * (hb.y > h0 + 1. ? 0.5 : 1.);
  }
  o = vec4(vis, 1. - 0.8 * occ / 12., 0., 1.);
}`;

// Output is premultiplied: the stone where there is stone, and over the page around it only the
// darkening of its shadow and contact occlusion (the page is not ours to redraw).
const FS_COMP = HEAD + `
uniform sampler2D uG0;
uniform sampler2D uG1;
uniform sampler2D uG2;
uniform sampler2D uSh;
uniform sampler2D uStone;
uniform sampler2D uStoneM;
uniform vec4 uStoneBox;
uniform float uDpr, uShadowK, uSpec, uClear;
uniform vec4 uPillC;      // the slab's centre and size, CSS px of the stage (y down)
uniform vec3 uL, uPage;
uniform vec4 uInlay, uFrac;
out vec4 o;

vec3 stone(vec2 o, float fresh, float carve, float halo, out float rough, out float glint, out vec2 tilt, out float inl){
  vec2 uv = (o - uStoneBox.xy) / uStoneBox.zw;
  vec4 a = textureLod(uStone, uv, 0.), m = texture(uStoneM, uv);
  vec3 c = pow(a.rgb, vec3(2.2));
  // round the letters the stone's pattern is calmed toward its local mean, so a busy stone never
  // fights the label for legibility
  vec3 mean = pow(textureLod(uStone, uv, 3.).rgb, vec3(2.2));
  c = mix(c, mean, 0.6 * halo);
  rough = m.x; glint = m.y; tilt = m.zw - 0.5;
  vec3 old = mix(c, vec3(0.28, 0.24, 0.21), 0.07);
  // a fresh break shows the stone's true colour, paler and without the face's film; its grain stays
  float ratio = clamp(dot(c, vec3(0.3, 0.55, 0.15)) / max(dot(uFrac.rgb, vec3(0.3, 0.55, 0.15)), 1e-3), 0.5, 1.6);
  vec3 fr = mix(c * 1.04 + 0.006, uFrac.rgb * mix(1., ratio, 0.5), uFrac.w);
  c = mix(old, fr, fresh);
  rough = mix(rough, 0.85, fresh);
  float cm = smoothstep(0.05, 0.4, carve);
  c = mix(c, c * 0.75 + vec3(0.01), cm);
  rough = mix(rough, 0.85, cm); glint *= 1. - cm;
  // the paint sits in the surface skin, which a fresh break has just flaked off along its edge
  inl = smoothstep(0.08, 0.3, carve) * uInlay.w * (1. - 0.7 * fresh);
  c = mix(c, uInlay.rgb, inl);
  return c;
}

void main(){
  ivec2 ip = ivec2(gl_FragCoord.xy);
  ivec2 mx = textureSize(uG0, 0) - 1;
  vec4 A = texelFetch(uG0, ip, 0);
  bool rock = A.w >= 0.;
  float h0 = texelFetch(uG2, ip, 0).x;
  vec3 L = normalize(uL);

  vec2 sv = vec2(0.); float ws = 0.;
  for (int k = 0; k < 5; k++){
    ivec2 off = k == 0 ? ivec2(0) : k == 1 ? ivec2(1, 0) : k == 2 ? ivec2(-1, 0) : k == 3 ? ivec2(0, 1) : ivec2(0, -1);
    ivec2 q = clamp(ip + off, ivec2(0), mx);
    float hq = texelFetch(uG2, q, 0).x;
    float w = (k == 0 ? 2. : 1.) * exp(-abs(hq - h0) * 1.5);
    sv += texelFetch(uSh, q, 0).xy * w; ws += w;
  }
  sv /= ws;
  float vis = sv.x, ao = sv.y;
  float dith = (hash12(gl_FragCoord.xy + 0.37) - 0.5) / 255.;

  // what a flat patch of page would lose to the shadow, as the alpha of a black veil
  float sunI = 2.15 * L.z, skyI = 0.37;
  float shA = clamp((1. - (sunI * vis + skyI * ao) / (sunI + skyI)) * uShadowK + dith, 0., 1.);
  if (!rock){ o = vec4(0., 0., 0., shA); return; }

  vec3 sunC = vec3(1.0, 0.94, 0.85) * 2.15;
  vec3 skyC = vec3(0.52, 0.60, 0.74) * 0.62;
  vec3 V = vec3(0., 0., 1.);
  vec3 Hh = normalize(L + V);

  vec4 B = texelFetch(uG1, ip, 0);
  bool wall = mod(A.w, 2.) > 0.5;
  float rough, glint, inl; vec2 tilt;
  float fresh = B.x;
  float carve = max(B.y, 0.), halo = max(-B.y, 0.);
  vec3 alb = stone(A.yz, fresh, carve, halo, rough, glint, tilt, inl);
  vec3 n;
  if (wall){
    vec2 nw = texelFetch(uG2, ip, 0).zw;
    n = normalize(vec3(nw * 0.96, 0.28) + vec3(tilt * mix(0.25, 0.7, fresh), 0.));
    alb = mix(mix(alb, vec3(dot(alb, vec3(0.3, 0.55, 0.15))), 0.35) * 0.78, alb * 1.06, fresh);
  } else {
    vec4 Lx = texelFetch(uG0, clamp(ip - ivec2(1, 0), ivec2(0), mx), 0);
    vec4 Rx = texelFetch(uG0, clamp(ip + ivec2(1, 0), ivec2(0), mx), 0);
    vec4 Dn = texelFetch(uG0, clamp(ip - ivec2(0, 1), ivec2(0), mx), 0);
    vec4 Up = texelFetch(uG0, clamp(ip + ivec2(0, 1), ivec2(0), mx), 0);
    bool okR = Rx.w == A.w, okL = Lx.w == A.w, okD = Dn.w == A.w, okU = Up.w == A.w;
    float gx = (okR && okL) ? (Rx.x - Lx.x) * 0.5 : okR ? Rx.x - A.x : okL ? A.x - Lx.x : 0.;
    float gy = (okD && okU) ? (Dn.x - Up.x) * 0.5 : okD ? Dn.x - A.x : okU ? A.x - Up.x : 0.;
    gx *= uDpr; gy *= uDpr;
    n = normalize(vec3(-gx, -gy, 1.));
    n = normalize(n + vec3(tilt * mix(0.05, 0.5, fresh), 0.));
    if (carve > 0.02) n = normalize(n + vec3((vec2(vnoise(A.yz * 1.7), vnoise(A.yz * 1.7 + 5.)) - 0.5) * 0.12 * smoothstep(0.02, 0.3, carve), 0.));
    // paint in the letters lies flatter than the chisel marks under it
    n = normalize(mix(n, vec3(0., 0., 1.), inl * 0.5));
  }
  float carveAo = 1. - 0.35 * smoothstep(0.1, 0.9, carve) * (1. - inl * 0.6);

  float ndl = max(dot(n, L), 0.);
  float sky = 0.55 + 0.45 * n.z;
  vec3 c = alb * (sunC * ndl * vis + skyC * sky * ao * carveAo);
  c += alb * uPage * sunC * 0.16 * (1. - n.z) * vis;
  float nh = max(dot(n, Hh), 0.);
  float shin = 2. / (rough * rough);
  c += sunC * 0.035 * uSpec * (shin + 2.) / 8. * pow(nh, shin) * ndl * vis;
  vec3 nc = normalize(n + vec3(tilt * 0.9, 0.));
  c += sunC * 0.9 * glint * pow(max(dot(nc, Hh), 0.), 300.) * vis;
  if (!wall && uClear > 0.){
    // a polished face mirrors a soft window-like light across it; it bends with the surface relief and
    // is absent on fresh breaks and in the letters
    vec2 css = vec2(gl_FragCoord.x, float(mx.y + 1) - gl_FragCoord.y) / uDpr;
    vec2 pq = (css - uPillC.xy) / uPillC.zw;
    float x = pq.x * 0.35 + pq.y * 1.2 + (n.x + n.y) * 3.;
    float win = smoothstep(0.3, 0.05, abs(x + 0.1)) + 0.35 * smoothstep(0.12, 0.02, abs(x - 0.42));
    c += sunC * uClear * 0.045 * win * (1. - fresh) * (1. - smoothstep(0.05, 0.4, carve)) * vis;
  }
  if (!wall){
    vec2 en = texelFetch(uG2, ip, 0).zw;
    float ar = (1. - smoothstep(0.6, 1.8, B.w * uDpr)) * length(en);
    float lit = dot(normalize(en + 1e-6), normalize(L.xy));
    c *= 1. + ar * (0.7 * max(lit, 0.) - 0.45 * max(-lit, 0.));
    c += sunC * 0.05 * (1. - n.z) * max(dot(normalize(vec3(n.xy, 0.) + 1e-6), L), 0.) * vis;
  }
  vec3 col = c * 0.8;
  col = clamp((col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14), 0., 1.);
  col = pow(col, vec3(1. / 2.2)) + dith;
  float cov = B.z;
  o = vec4(col * cov, cov + (1. - cov) * shA);
}`;

// Stone dust drifting over the scene.
const VS_DUST = `#version 300 es
layout(location=0) in vec2 aCorner;
layout(location=1) in vec4 aD;     // x, y, radius, alpha
uniform vec2 uCss;
out vec2 vL;
out float vA;
void main(){
  vec2 wp = aD.xy + (aCorner * 2. - 1.) * aD.z;
  gl_Position = vec4(wp.x / uCss.x * 2. - 1., 1. - wp.y / uCss.y * 2., 0., 1.);
  vL = aCorner * 2. - 1.; vA = aD.w;
}`;
const FS_DUST = HEAD + `
uniform vec3 uCol;
in vec2 vL;
in float vA;
out vec4 o;
void main(){
  float r = dot(vL, vL);
  if (r > 1.) discard;
  float a = vA * exp(-r * 3.2) * (0.75 + 0.5 * vnoise(vL * 3. + vA * 20.));
  o = vec4(uCol * a, a);
}`;

// ---------------------------------------------------------------- materials
// Colours are sRGB. grain: crystal/sand grain size (px) and contrast; mottle: strength and size (px) of
// the cloudy variation between base and base2; bands: bedding/cleavage strength, spacing (px), angle;
// vein: colour, amount, scale (px), width, warp; speckle: two colours, amount, size (px); pits: pores;
// roughness, sheen (general lustre), sparkle (crystal faces catching the sun); fracture: the colour of a
// fresh break and how strongly it shows; wear: knocks along the old arris; dust: powder colour.
const PRESETS = {
  granite: { kind: 0, meanLum: 0.34, fracture: '#efe1d9', fractureAmt: 0.25, wear: 1, weather: 0.45, dust: '#d9cfc8', roughness: 0.3, sheen: 1, labelLum: 0.55 },
  slate: { base: '#48525c', base2: '#30373e', grain: 1.1, contrast: 0.04, mottle: 0.45, mottleScale: 90, bands: 0.6, bandSpacing: 2.6, bandAngle: 1.62,
    bandMode: 'cleavage', layerStep: 0.8, variation: 0.05, variationScale: 50, grainRelief: 0.12,
    speckle: ['#8a7b55', '#5d6870'], speckleAmount: 0.006, speckleSize: 0.5, roughness: 0.55, sheen: 0.55, sparkle: 0.1,
    fracture: '#5d676d', fractureAmt: 0.7, wear: 0.6, weather: 0.5, dust: '#9aa3a8' },
  basalt: { base: '#303032', base2: '#262628', grain: 1.0, contrast: 0.14, mottle: 0.35, mottleScale: 50, grainRelief: 0.25, pitDepth: 1.6,
    speckle: ['#76736c', '#1b1b1c'], speckleAmount: 0.09, speckleSize: 0.7, pits: 0.13, roughness: 0.6, sheen: 0.45, sparkle: 0.2,
    fracture: '#424245', fractureAmt: 0.6, wear: 0.5, weather: 0.35, dust: '#86858a' },
  sandstone: { base: '#d2ab80', base2: '#b27e56', grain: 0.75, contrast: 0.2, mottle: 0.45, mottleScale: 60, bands: 0.4, bandSpacing: 8, bandAngle: 1.52,
    crossBedding: 0.45, variation: 0.08, variationScale: 45, grainRelief: 0.7,
    speckle: ['#ecdcc0', '#e3cba5'], speckleAmount: 0.015, speckleSize: 0.55, roughness: 0.93, sheen: 0.15, sparkle: 0.18,
    fracture: '#dcbf98', fractureAmt: 0.55, wear: 1.3, weather: 0.7, dust: '#e3cfae' },
  marble: { base: '#f2efea', base2: '#d4d0c9', grain: 2.2, contrast: 0.025, mottle: 1.0, mottleScale: 38, variation: 0.035, variationScale: 60,
    vein: '#4d545b', veinAmount: 1, veinScale: 40, veinWidth: 0.17, veinWarp: 1.6, grainRelief: 0.04, roughness: 0.12, sheen: 1.4, sparkle: 0.25, clear: 1,
    fracture: '#fbfaf6', fractureAmt: 0.6, wear: 0.8, weather: 0.25, dust: '#f3f1ed' },
  limestone: { base: '#ddd3be', base2: '#c4b695', grain: 1.1, contrast: 0.08, mottle: 0.9, mottleScale: 45, grainRelief: 0.3,
    speckle: ['#ab9a7b', '#f1ebdc'], speckleAmount: 0.08, speckleSize: 1.5, pits: 0.06, roughness: 0.88, sheen: 0.3, sparkle: 0.05,
    fracture: '#ebe3d1', fractureAmt: 0.6, wear: 1.1, weather: 0.6, dust: '#ece5d5' },
  // not listed: the structure a bare colour gets. A dense fine-grained stone: its tone drifts over three
  // scales, a few grains are a dark mineral of the opposite hue, pale specks, the odd pore, a weathered
  // skin; the same in every direction, so any hue reads as stone rather than as plastic, fabric or wood
  _tint: { base: '#a08e86', base2: '#8f7d75', grain: 0.85, contrast: 0.1, mottle: 0.6, mottleScale: 50, variation: 0.21, variationScale: 42,
    grainRelief: 0.35, mineralAmount: 0.07,
    speckle: ['#6e5f58', '#c9bab2'], speckleAmount: 0.025, speckleSize: 0.55, pits: 0.03, roughness: 0.8, sheen: 0.35, sparkle: 0.12,
    fracture: '#b8a79f', fractureAmt: 0.5, wear: 0.9, weather: 0.9, dust: '#cdc0b9' },
  concrete: { base: '#a6a49f', base2: '#94928d', grain: 0.8, contrast: 0.12, mottle: 0.6, mottleScale: 70, grainRelief: 0.3, pitDepth: 1.8,
    speckle: ['#7b7873', '#c0bcb4'], speckleAmount: 0.2, speckleSize: 2.4, pits: 0.14, roughness: 0.93, sheen: 0.25, sparkle: 0.03,
    fracture: '#b9b6af', fractureAmt: 0.6, wear: 1.2, weather: 0.5, dust: '#d0cdc7' },
};

let probe = null;
function parseColor(css){
  if (!css) return null;
  if (Array.isArray(css)) return [css[0], css[1], css[2], css[3] ?? 1];
  let m = String(css).match(/^rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)(?:\s*[,/]\s*([\d.]+%?))?\s*\)$/);
  if (m){
    const a = m[4] === undefined ? 1 : m[4].endsWith('%') ? parseFloat(m[4]) / 100 : parseFloat(m[4]);
    return [m[1] / 255, m[2] / 255, m[3] / 255, a];
  }
  m = String(css).match(/^color\(srgb\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s*\/\s*([\d.]+%?))?\s*\)$/);
  if (m){
    const a = m[4] === undefined ? 1 : m[4].endsWith('%') ? parseFloat(m[4]) / 100 : parseFloat(m[4]);
    return [+m[1], +m[2], +m[3], a];
  }
  // any other colour syntax: let a 2D context normalise it
  if (!probe) probe = document.createElement('canvas').getContext('2d');
  probe.fillStyle = '#010203'; probe.fillStyle = css;
  const s = probe.fillStyle;
  if (s === '#010203' && !/^#010203$/i.test(css)) return null;
  if (s[0] === '#') return [parseInt(s.slice(1, 3), 16) / 255, parseInt(s.slice(3, 5), 16) / 255, parseInt(s.slice(5, 7), 16) / 255, 1];
  return s === css ? null : parseColor(s);
}
const toLin = c => c.slice(0, 3).map(v => Math.pow(clamp(v, 0, 1), 2.2));
const lumOf = c => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
function rgb2hsl([r, g, b]){
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), l = (mx + mn) / 2;
  if (mx === mn) return [0, 0, l];
  const d = mx - mn, s = l > 0.5 ? d / (2 - mx - mn) : d / (mx + mn);
  const h = mx === r ? (g - b) / d + (g < b ? 6 : 0) : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return [h / 6, s, l];
}
function hsl2rgb([h, s, l]){
  if (s === 0) return [l, l, l];
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s, p = 2 * l - q;
  const f = t => { t = (t + 1) % 1; return t < 1 / 6 ? p + (q - p) * 6 * t : t < 0.5 ? q : t < 2 / 3 ? p + (q - p) * (2 / 3 - t) * 6 : p; };
  return [f(h + 1 / 3), f(h), f(h - 1 / 3)];
}
const shade = (c, dl, ds = 0, dh = 0) => { const [h, s, l] = rgb2hsl(c); return hsl2rgb([(h + dh + 1) % 1, clamp(s + ds, 0, 1), clamp(l + dl, 0, 1)]); };
// a colour pushed toward what a real stone could be: less saturated, not quite black or white
function stoneify(c){ const [h, s, l] = rgb2hsl(c); return hsl2rgb([h, Math.min(s * 0.62, 0.42), clamp(l, 0.16, 0.84)]); }

// The page colour behind an element: its ancestors' background colours composited down to the canvas.
function pageColor(el){
  const layers = [];
  for (let e = el; e; e = e.parentElement){
    const c = parseColor(getComputedStyle(e).backgroundColor);
    if (c && c[3] > 0.004){ layers.push(c); if (c[3] >= 0.999) break; }
  }
  let rgb = [1, 1, 1];
  for (let i = layers.length - 1; i >= 0; i--){
    const [r, g, b, a] = layers[i];
    rgb = [r * a + rgb[0] * (1 - a), g * a + rgb[1] * (1 - a), b * a + rgb[2] * (1 - a)];
  }
  return rgb;
}

// Everything the shaders need for a material spec: a preset name, 'auto', or a custom object.
function resolveMaterial(spec, autoColor){
  let custom = null, name = spec;
  if (spec && typeof spec === 'object'){ custom = spec; name = spec.preset || spec.family; }
  const isAuto = name === 'auto' || (custom && custom.base === 'auto');
  if (isAuto){
    custom = Object.assign({}, custom || {}, { base: stoneify(autoColor || [0.6, 0.6, 0.6]) });
    if (name === 'auto') name = custom.family;
    if (!PRESETS[name]){ const l = rgb2hsl(custom.base)[2]; name = l < 0.2 ? 'basalt' : '_tint'; }
  }
  if (!isAuto && custom && custom.base){
    // a page colour taken as it is reads as painted plastic: soften it into something a stone could be
    // ({ exact: true } keeps it), and a bare colour gets an isotropic stone structure
    const c = parseColor(custom.base);
    if (c){
      const [h, sat, l] = rgb2hsl(c);
      if (!custom.exact) custom = Object.assign({}, custom, { base: hsl2rgb([h, Math.min(sat * 0.62, 0.38), clamp(l, 0.12, 0.84)]) });
      if (!PRESETS[name]) name = l < 0.2 ? 'basalt' : '_tint';
    }
  }
  if (!PRESETS[name]) name = 'granite';
  const P = Object.assign({}, PRESETS[name]);
  const col = (v, fb) => { const c = parseColor(v); return c ? c.slice(0, 3) : fb; };
  const base = custom && custom.base ? col(custom.base, null) : null;
  if (P.kind !== 0){
    P.base = col(P.base); P.base2 = col(P.base2);
    P.fracture = col(P.fracture); P.dust = col(P.dust);
    P.speckle = (P.speckle || ['#000', '#fff']).map(v => col(v));
    P.mineral = P.mineral ? col(P.mineral) : null;
    P.vein = col(P.vein || '#000');
    if (base){
      // re-colour the preset's structure around the new base, keeping its tonal relationships
      const ref = P.base, dl = rgb2hsl(base)[2] - rgb2hsl(ref)[2];
      const move = c => { const [h, s] = rgb2hsl(base), [, , l] = rgb2hsl(c); return hsl2rgb([h, s * clamp(rgb2hsl(c)[1] / Math.max(rgb2hsl(ref)[1], 0.05), 0.3, 1.6), clamp(l + dl, 0.03, 0.97)]); };
      P.base = base.slice(); P.base2 = custom.base2 ? col(custom.base2) : move(P.base2);
      P.fracture = move(P.fracture); P.dust = shade(move(P.dust), 0.05, -0.1);
      P.speckle = P.speckle.map(move);
      if (P.veinAmount) P.vein = shade(base, -0.38, -0.05);
      if (name === '_tint'){
        const [h, sat, l] = rgb2hsl(base);
        P.base2 = custom.base2 ? col(custom.base2) : hsl2rgb([(h + 0.02) % 1, sat * 0.8, clamp(l - 0.12, 0.05, 0.9)]);
        P.speckle = [hsl2rgb([h, sat * 0.6, l * 0.62]), hsl2rgb([h, sat * 0.35, Math.min(l + 0.22, 0.93)])];
        P.mineral = hsl2rgb([(h + 0.5) % 1, Math.min(sat * 0.4, 0.22), l * 0.5]);
      }
    }
  } else {
    P.fracture = col(P.fracture); P.dust = col(P.dust);
    if (base){ P.tint = base; P.fracture = shade(base, 0.25, -0.15); P.dust = shade(base, 0.3, -0.2); }
  }
  if (custom){
    const num = ['grain', 'contrast', 'mottle', 'mottleScale', 'bands', 'bandSpacing', 'bandAngle', 'veinAmount', 'veinScale', 'veinWidth', 'veinWarp',
      'speckleAmount', 'speckleSize', 'pits', 'roughness', 'sheen', 'sparkle', 'fractureAmt', 'wear', 'weather', 'variation', 'variationScale',
      'mineralAmount', 'crossBedding', 'layerStep', 'grainRelief', 'pitDepth', 'clear'];
    if (typeof custom.bandMode === 'string') P.bandMode = custom.bandMode;
    if (custom.mineral){ P.mineral = col(custom.mineral, P.mineral); if (typeof custom.mineralAmount !== 'number' && !P.mineralAmount) P.mineralAmount = 0.06; }
    for (const k of num) if (typeof custom[k] === 'number') P[k] = custom[k];
    if (custom.vein){ P.vein = col(custom.vein, P.vein); if (typeof custom.veinAmount !== 'number' && !P.veinAmount) P.veinAmount = 0.5; }
    if (custom.speckle){ const s = Array.isArray(custom.speckle) ? custom.speckle : [custom.speckle, custom.speckle]; P.speckle = s.map(v => col(v, [0, 0, 0])); if (!P.speckleAmount) P.speckleAmount = 0.05; }
    if (custom.fracture) P.fracture = col(custom.fracture, P.fracture);
    if (custom.dust) P.dust = col(custom.dust, P.dust);
    if (custom.base2 && P.kind !== 0) P.base2 = col(custom.base2, P.base2);
  }
  const lumBase = P.kind === 0 ? (P.tint ? lumOf(P.tint) : P.labelLum) : lumOf(P.base.map((v, i) => (v + P.base2[i]) / 2));
  P.name = name === '_tint' ? 'custom' : name;
  P.lum = lumBase;
  P.key = JSON.stringify([spec, spec === 'auto' || (custom && custom.base === 'auto') ? autoColor : 0]);
  return P;
}

// ---------------------------------------------------------------- audio (shared)
let actx = null, master = null, scrapeN = null;
const rockAudio = { muted: false };
function audio(){
  if (rockAudio.muted) return null;
  if (!actx){
    try {
      actx = new (window.AudioContext || window.webkitAudioContext)();
      const comp = actx.createDynamicsCompressor();
      comp.threshold.value = -14; comp.ratio.value = 4; comp.attack.value = 0.002; comp.release.value = 0.15;
      master = actx.createGain(); master.gain.value = 0.8;
      master.connect(comp).connect(actx.destination);
    } catch { return null; }
  }
  if (actx.state === 'suspended') actx.resume();
  return actx;
}
const noiseCache = new Map();
function noiseBuf(a, dur){
  const key = a.sampleRate + ':' + dur;
  if (noiseCache.has(key)) return noiseCache.get(key);
  const len = Math.floor(a.sampleRate * dur), b = a.createBuffer(1, len, a.sampleRate), d = b.getChannelData(0);
  for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
  noiseCache.set(key, b);
  return b;
}
function burst(type, freq, q, gain, dur, delay = 0, attack = 0.001){
  const a = audio(); if (!a || gain <= 0.0005) return;
  const t = a.currentTime + delay;
  const src = a.createBufferSource(); src.buffer = noiseBuf(a, 1);
  const f = a.createBiquadFilter(); f.type = type; f.frequency.value = freq; f.Q.value = q;
  const g = a.createGain();
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(gain, t + attack);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  src.connect(f).connect(g).connect(master);
  src.start(t, Math.random() * 0.8); src.stop(t + dur + 0.02);
}
function thump(f0, f1, gain, dur, delay = 0){
  const a = audio(); if (!a) return;
  const t = a.currentTime + delay;
  const o = a.createOscillator(), g = a.createGain();
  o.type = 'sine'; o.frequency.setValueAtTime(f0, t); o.frequency.exponentialRampToValueAtTime(f1, t + dur);
  g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(gain, t + 0.003); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  o.connect(g).connect(master); o.start(t); o.stop(t + dur + 0.02);
}
function grit(n, spread, gain, delay = 0){
  for (let i = 0; i < n; i++) burst('bandpass', rand(2200, 6500), rand(2, 6), gain * rand(0.3, 1), rand(0.006, 0.018), delay + Math.pow(Math.random(), 1.6) * spread);
}
const sounds = {
  // the first break: a sharp report, the slab's low knock, stone ringing briefly, grit trickling
  crack(power){
    burst('highpass', 1500, 0.7, 0.9 * power, 0.03);
    burst('bandpass', 3200, 1.2, 0.5 * power, 0.05, 0.002);
    thump(150, 48, 0.55, 0.16);
    for (const f of [1750, 2650, 3900]) burst('bandpass', f * rand(0.92, 1.08), 16, 0.16 * power, rand(0.06, 0.12), 0.003);
    grit(24, 0.45, 0.07, 0.02);
  },
  crunch(power, mass){
    const low = clamp(Math.sqrt(mass) / 120, 0.2, 1);
    burst('highpass', 1300, 0.7, 0.45 + 0.25 * power, 0.028);
    burst('bandpass', 2600 + 1400 * (1 - low), 2, 0.35 * power, 0.06, 0.002);
    thump(90 + 140 * (1 - low), 45, 0.4 * low + 0.1, 0.12);
    grit(Math.round(8 + power * 8), 0.3, 0.06, 0.01);
  },
  knock(v, m, wall){
    const a = audio(); if (!a) return;
    const s = clamp((v - 30) / 900, 0, 1);
    const f = clamp(4200 - Math.sqrt(m) * 26, 1100, 4200) * rand(0.9, 1.1);
    const g = 0.05 + 0.5 * Math.pow(s, 0.8);
    burst('bandpass', f, 9, g, wall ? 0.03 : 0.045);
    burst('bandpass', f * 1.58, 12, g * 0.5, 0.03);
    thump(f * 0.16, f * 0.12, g * 0.5 * clamp(Math.sqrt(m) / 60, 0.2, 1), 0.05);
  },
  tick(g){ burst('bandpass', rand(3500, 7000), 4, 0.06 * g + 0.01, 0.012); },
  settle(){ sounds.knock(260 + 80 * Math.random(), 800, false); },
  seal(){ thump(110, 55, 0.4, 0.28); burst('lowpass', 600, 0.7, 0.18, 0.2); burst('bandpass', 2400, 3, 0.06, 0.25, 0.01); },
  glide(){ burst('bandpass', 700, 0.8, 0.05, 1.1, 0, 0.35); burst('bandpass', 2200, 0.7, 0.03, 0.9, 0.1, 0.3); },
  scrape(level, bright){
    const a = actx; if (!a) return;
    if (!scrapeN){
      if (level <= 0.001 || rockAudio.muted) return;
      const src = a.createBufferSource(); src.buffer = noiseBuf(a, 2); src.loop = true;
      const f = a.createBiquadFilter(); f.type = 'bandpass'; f.frequency.value = 900; f.Q.value = 0.8;
      const f2 = a.createBiquadFilter(); f2.type = 'peaking'; f2.frequency.value = 3200; f2.Q.value = 1.5; f2.gain.value = 6;
      const chatter = a.createBufferSource(); chatter.buffer = noiseBuf(a, 2); chatter.loop = true; chatter.playbackRate.value = 0.012;
      const cg = a.createGain(); cg.gain.value = 0.6;
      const g = a.createGain(); g.gain.value = 0;
      const vca = a.createGain(); vca.gain.value = 0.7;
      chatter.connect(cg).connect(vca.gain);
      src.connect(f).connect(f2).connect(vca).connect(g).connect(master);
      src.start(); chatter.start();
      scrapeN = { f, g };
    }
    scrapeN.g.gain.setTargetAtTime(rockAudio.muted ? 0 : level, a.currentTime, 0.05);
    scrapeN.f.frequency.setTargetAtTime(600 + 1400 * bright, a.currentTime, 0.06);
  },
};
const silent = Object.fromEntries(Object.keys(sounds).map(k => [k, () => {}]));

// ---------------------------------------------------------------- pure helpers
function softness(hz, zeta, h){
  const w = TAU * hz, a1 = 2 * zeta + h * w, a2 = h * w * a1, a3 = 1 / (1 + a2);
  return { biasRate: w / a1, massScale: a2 * a3, impulseScale: a3 };
}
// Fast frames are stepped finer so a hurled piece moves at most a few px per substep; the soft
// constraint coefficients depend on the step, so there is one set per subdivision.
const SOFTS = [1, 2, 3].map(d => ({
  contact: softness(CONTACT_HZ, CONTACT_ZETA, H_STEP / d),
  wall: softness(CONTACT_HZ * 2, CONTACT_ZETA, H_STEP / d),
  drag: softness(DRAG_HZ, DRAG_ZETA, H_STEP / d),
}));
function clipPoly(xs, ys, ls, nx, ny, c, label){
  const ox = [], oy = [], ol = [];
  const n = xs.length;
  for (let i = 0; i < n; i++){
    const j = (i + 1) % n;
    const ax = xs[i], ay = ys[i], bx = xs[j], by = ys[j];
    const da = nx * ax + ny * ay - c, db = nx * bx + ny * by - c;
    if (da <= 0){
      ox.push(ax); oy.push(ay); ol.push(ls[i]);
      if (db > 0){ const t = da / (da - db); ox.push(ax + (bx - ax) * t); oy.push(ay + (by - ay) * t); ol.push(label); }
    } else if (db <= 0){
      const t = da / (da - db); ox.push(ax + (bx - ax) * t); oy.push(ay + (by - ay) * t); ol.push(ls[i]);
    }
  }
  return [ox, oy, ol];
}
function hullOf(xs, ys){
  const n = xs.length, idx = new Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  idx.sort((a, b) => xs[a] - xs[b] || ys[a] - ys[b]);
  const cr = (o, a, b) => (xs[a] - xs[o]) * (ys[b] - ys[o]) - (ys[a] - ys[o]) * (xs[b] - xs[o]);
  const lo = [], up = [];
  for (const i of idx){ while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], i) <= 0) lo.pop(); lo.push(i); }
  for (let q = n - 1; q >= 0; q--){ const i = idx[q]; while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], i) <= 0) up.pop(); up.push(i); }
  lo.pop(); up.pop();
  const h = lo.concat(up);
  return { x: h.map(i => xs[i]), y: h.map(i => ys[i]) };
}
function hullStats(hx, hy){
  let A = 0, D = 0;
  const n = hx.length;
  for (let i = 0; i < n; i++){ const j = (i + 1) % n; A += hx[i] * hy[j] - hx[j] * hy[i]; }
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) D = Math.max(D, (hx[i] - hx[j]) ** 2 + (hy[i] - hy[j]) ** 2);
  return { area: Math.abs(A) / 2, diam2: D };
}
function h1(n){ const s = Math.sin(n * 127.1 + 311.7) * 43758.5453; return s - Math.floor(s); }
function vn1(x){ const i = Math.floor(x), f = x - i, u = f * f * (3 - 2 * f); return h1(i) * (1 - u) + h1(i + 1) * u; }
function segHit(ax, ay, bx, by, cx, cy, dx, dy){
  const rx = bx - ax, ry = by - ay, sx = dx - cx, sy = dy - cy;
  const den = rx * sy - ry * sx;
  if (Math.abs(den) < 1e-12) return -1;
  const qx = cx - ax, qy = cy - ay;
  const t = (qx * sy - qy * sx) / den, u = (qx * ry - qy * rx) / den;
  return t >= 0 && t <= 1 && u >= 0 && u <= 1 ? t : -1;
}
// signed distance to a rounded rectangle centred on the origin (negative inside)
function sdRound(x, y, hw, hh, r){
  const qx = Math.abs(x) - (hw - r), qy = Math.abs(y) - (hh - r);
  return Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0) - r;
}
function inFixedContext(el){
  for (let e = el; e && e !== document.documentElement; e = e.parentElement){
    const p = getComputedStyle(e).position;
    if (p === 'fixed' || p === 'sticky') return true;
  }
  return false;
}

// ---------------------------------------------------------------- component
function attachRock(button, opts = {}){
  const noop = () => {};
  const unsupported = { supported: false, state: 'unsupported', detach: noop, reset: noop, setMaterial: noop, _manual: noop, _advance: noop, pieces: [] };
  const canvas = document.createElement('canvas');
  canvas.setAttribute('aria-hidden', 'true');
  Object.assign(canvas.style, { position: 'absolute', left: '0px', top: '0px', margin: '0', padding: '0', border: '0', display: 'block', pointerEvents: 'none', zIndex: String(opts.zIndex ?? 10) });
  const gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: true, antialias: false, depth: false, stencil: false });
  if (!gl || !gl.getExtension('EXT_color_buffer_float')) return unsupported;
  const snd = opts.sound === false ? silent : sounds;
  const REGROUP = opts.regroupAfter ?? 2.5;

  // ---- GL resources (rebuilt from scratch after a lost context)
  let P, emptyVao, quadBuf, pieceInst, chipInst, dustInst, segTex, textTex;
  let g0Tex, g1Tex, g2Tex, depthRb, gFb, shTex, shFb, stoneTex, stoneMTex, stoneFb;
  let stoneBox = [0, 0, 1, 1], stoneKey = '';
  const freeRows = [];
  function compileProgram(vs, fs){
    const sh = (type, src) => {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS) && !gl.isContextLost()) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS) && !gl.isContextLost()) throw new Error(gl.getProgramInfoLog(p));
    const u = {};
    for (let i = 0, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS) || 0; i < n; i++){
      const info = gl.getActiveUniform(p, i);
      u[info.name.replace(/\[0\]$/, '')] = gl.getUniformLocation(p, info.name);
    }
    return { p, u };
  }
  function makeTex(w, h, internal, filter, levels = 1){
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texStorage2D(gl.TEXTURE_2D, levels, internal, w, h);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, levels > 1 ? gl.LINEAR_MIPMAP_NEAREST : filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }
  function makeTarget(texs, depth){
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    texs.forEach((t, i) => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t, 0));
    if (depth) gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth);
    gl.drawBuffers(texs.map((_, i) => gl.COLOR_ATTACHMENT0 + i));
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return fb;
  }
  const bindTex = (unit, tex) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex); };
  function instVao(nAttr){
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    for (let i = 0; i < nAttr; i++){
      gl.enableVertexAttribArray(1 + i);
      gl.vertexAttribPointer(1 + i, 4, gl.FLOAT, false, nAttr * 16, i * 16);
      gl.vertexAttribDivisor(1 + i, 1);
    }
    gl.bindVertexArray(null);
    return { vao, buf, data: new Float32Array(0), n: nAttr * 4 };
  }
  function fillInst(inst, count){
    if (inst.data.length < count * inst.n) inst.data = new Float32Array(Math.max(count * inst.n * 2, 64));
    return inst.data;
  }
  function initGL(){
    gl.getExtension('EXT_color_buffer_float');
    P = {
      piece: compileProgram(VS_PIECE, FS_PIECE), chip: compileProgram(VS_CHIP, FS_CHIP), stone: compileProgram(VS_FULL, FS_STONE),
      comp: compileProgram(VS_FULL, FS_COMP), shadow: compileProgram(VS_FULL, FS_SHADOW), dust: compileProgram(VS_DUST, FS_DUST),
    };
    emptyVao = gl.createVertexArray();
    quadBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
    pieceInst = instVao(5); chipInst = instVao(3); dustInst = instVao(1);
    segTex = makeTex(SEGW, MAXB, gl.RGBA32F, gl.NEAREST);
    textTex = gl.createTexture();
    freeRows.length = 0;
    for (let i = MAXB - 1; i >= 0; i--) freeRows.push(i);
    // handles from before a context loss belong to the dead context: drop them, never delete them
    g0Tex = g1Tex = g2Tex = shTex = null; gFb = shFb = null; depthRb = null;
    stoneTex = stoneMTex = null; stoneFb = null; stoneKey = '';
  }
  try { initGL(); } catch (err) {
    console.warn('ElementalFX rock: shaders failed, leaving the button as it is.', err);
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return unsupported;
  }

  // ---- geometry. Physics runs in stage CSS px (origin at the stage's top-left, y down); the stage is
  // the canvas, placed so the button sits at (offX, offY) inside it. The stage's edges are the walls.
  const G = { fixed: false, bx: NaN, by: NaN, bw: 0, bh: 0, stageW: 0, stageH: 0, offX: 0, offY: 0 };
  let bounds = { w: 0, h: 0 }, cbOrigin = { x: 0, y: 0 };
  let dpr = 1, W = 0, H = 0;
  const pill = { cx: 0, cy: 0, w: 0, h: 0, r: 0 };
  const rock = { w: 0, h: 0, r: 0, T: 14 };
  let pageLin = [1, 1, 1], pageLum = 1;

  // ---- the button: invisible but still there
  const saved = button.getAttribute('style');
  const cs0 = getComputedStyle(button);
  const btnBg = parseColor(cs0.backgroundColor);
  const hideStyle = { background: 'transparent', borderColor: 'transparent', color: 'transparent', boxShadow: 'none', textShadow: 'none' };
  let hidden = false;
  function hideButton(){ hidden = true; Object.assign(button.style, hideStyle); }
  function showButton(){ hidden = false; if (saved === null) button.removeAttribute('style'); else button.setAttribute('style', saved); }
  const basePE = (() => { const t = document.createElement('div'); t.setAttribute('style', saved || ''); return t.style.pointerEvents; })();
  let peOff = false, peTimer = 0;
  function disablePointer(){
    clearTimeout(peTimer);
    if (phase === 'intact' || peOff || !hidden) return;
    peOff = true; button.style.pointerEvents = 'none';
  }
  function enablePointer(){ clearTimeout(peTimer); if (!peOff) return; peOff = false; button.style.pointerEvents = basePE; }

  // ---- material
  let mat = null, matSpec = opts.material ?? 'granite';
  function autoColor(){
    if (btnBg && btnBg[3] > 0.1) return btnBg.slice(0, 3);
    return pageColor(button.parentElement);
  }
  let inlay = [0, 0, 0, 0];
  function chooseInlay(lum){
    const lc = opts.labelColor ?? 'auto';
    if (lc === 'none' || lc === false) inlay = [0, 0, 0, 0];
    else {
      const c = lc === 'auto' ? (lum < 0.42 ? [0.95, 0.93, 0.89] : [0.15, 0.135, 0.125]) : (parseColor(lc) || [0.15, 0.135, 0.125]);
      inlay = [...toLin(c), 0.92];
    }
  }
  function applyMaterial(){
    mat = resolveMaterial(matSpec, autoColor());
    chooseInlay(mat.lum);
    stoneKey = '';
  }

  // ---- grains
  // The stone is pre-cut into small Voronoi grains (a few px across). Pieces are connected sets of
  // grains, so fracture lines run along grain boundaries: jagged, and shared exactly by the two sides.
  // Each grain edge remembers the grain on its other side (-1 = the original outside of the stone).
  let GR = null;
  let wX, wY, gMinX, gMinY, gMaxX, gMaxY, mark, own, markStamp = 0;
  let lookX0 = 0, lookY0 = 0, lookCell = 1, lookW = 0, lookH = 0, lookStart = null, lookList = null;
  let vGrain = null, vSeen = null, vStamp = 0;
  let totalArea0 = 0;

  function outlinePoly(hw, hh, r){
    const xs = [], ys = [];
    const corners = [[hw - r, -hh + r, -Math.PI / 2], [hw - r, hh - r, 0], [-hw + r, hh - r, Math.PI / 2], [-hw + r, -hh + r, Math.PI]];
    const n = r > 0.3 ? Math.max(2, Math.ceil(r * Math.PI / 2 / 2.2)) : 0;
    for (const [cx, cy, a0] of corners){
      for (let i = 0; i <= n; i++){
        const a = a0 + (Math.PI / 2) * (n ? i / n : 0);
        const x = cx + r * Math.cos(a), y = cy + r * Math.sin(a);
        if (xs.length && Math.hypot(x - xs[xs.length - 1], y - ys[ys.length - 1]) < 1e-6) continue;
        xs.push(x); ys.push(y);
      }
    }
    if (Math.hypot(xs[0] - xs[xs.length - 1], ys[0] - ys[ys.length - 1]) < 1e-6){ xs.pop(); ys.pop(); }
    return [xs, ys, xs.map(() => -1)];
  }

  function buildGrains(w, h, rad){
    const hw = w / 2, hh = h / 2;
    const s = clamp(Math.min(w / 40, h / 7), 3.6, 6.8);
    const sdf = (x, y) => sdRound(x, y, hw, hh, rad);
    const [stX, stY, stL] = outlinePoly(hw, hh, rad);
    const px = [], py = [];
    const dy = s * 0.866;
    // seeded, so every rebuild at a given size yields the identical grain set
    let seed = (Math.round(w * 8) * 73856093 ^ Math.round(h * 8) * 19349663) >>> 0;
    const rnd = () => { seed = (seed + 0x6D2B79F5) >>> 0; let q = seed; q = Math.imul(q ^ (q >>> 15), q | 1); q ^= q + Math.imul(q ^ (q >>> 7), q | 61); return ((q ^ (q >>> 14)) >>> 0) / 4294967296; };
    let row = 0;
    for (let y = -hh + dy * 0.4; y < hh; y += dy, row++){
      for (let x = -hw + (row & 1 ? s * 0.5 : 0) + s * 0.3; x < hw; x += s){
        const jx = x + (rnd() - 0.5) * 0.8 * s, jy = y + (rnd() - 0.5) * 0.8 * dy;
        if (sdf(jx, jy) < -0.22 * s){ px.push(jx); py.push(jy); }
      }
    }
    const N = px.length;
    const bs = s, bx0 = -hw - s, by0 = -hh - s;
    const bw = Math.ceil((w + 2 * s) / bs) + 1, bh = Math.ceil((h + 2 * s) / bs) + 1;
    const buckets = Array.from({ length: bw * bh }, () => []);
    for (let i = 0; i < N; i++) buckets[Math.floor((py[i] - by0) / bs) * bw + Math.floor((px[i] - bx0) / bs)].push(i);
    const cells = new Array(N);
    for (let i = 0; i < N; i++){
      const cand = [];
      const cx = Math.floor((px[i] - bx0) / bs), cy = Math.floor((py[i] - by0) / bs);
      for (let v = cy - 3; v <= cy + 3; v++) for (let u = cx - 3; u <= cx + 3; u++){
        if (u < 0 || v < 0 || u >= bw || v >= bh) continue;
        for (const j of buckets[v * bw + u]) if (j !== i){ const d2 = (px[j] - px[i]) ** 2 + (py[j] - py[i]) ** 2; cand.push([d2, j]); }
      }
      cand.sort((a, b) => a[0] - b[0]);
      let xs = stX, ys = stY, ls = stL;
      for (const [d2, j] of cand){
        let maxR2 = 0;
        for (let k = 0; k < xs.length; k++) maxR2 = Math.max(maxR2, (xs[k] - px[i]) ** 2 + (ys[k] - py[i]) ** 2);
        if (d2 / 4 > maxR2) break;
        const d = Math.sqrt(d2), nx = (px[j] - px[i]) / d, ny = (py[j] - py[i]) / d;
        [xs, ys, ls] = clipPoly(xs, ys, ls, nx, ny, nx * (px[i] + px[j]) / 2 + ny * (py[i] + py[j]) / 2, j);
        if (xs.length < 3) break;
      }
      cells[i] = { xs: xs.slice(), ys: ys.slice(), ls: ls.slice() };
    }
    // Near-coincident corners are merged across ALL cells at once, then the edges that collapsed are
    // dropped. Cleaning each cell on its own would drop a corner on one side of a grain boundary and
    // keep it on the other, so two pieces would no longer share exactly the same crack line and
    // reassembled fragments would leave hairline gaps between them.
    {
      const SNAP = 0.12, cs = 0.25, reps = new Map(), rx = [], ry = [];
      const snap = (x, y) => {
        const i0 = Math.floor(x / cs), j0 = Math.floor(y / cs);
        for (let j = j0 - 1; j <= j0 + 1; j++) for (let i = i0 - 1; i <= i0 + 1; i++){
          const list = reps.get(i * 100003 + j);
          if (list) for (const r of list) if (Math.hypot(rx[r] - x, ry[r] - y) < SNAP) return r;
        }
        const r = rx.length; rx.push(x); ry.push(y);
        const k = i0 * 100003 + j0; let l = reps.get(k); if (!l){ l = []; reps.set(k, l); } l.push(r);
        return r;
      };
      for (const c of cells){
        if (!c) continue;
        const ids = c.xs.map((x, k) => snap(x, c.ys[k]));
        for (let k = 0; k < ids.length; k++){ c.xs[k] = rx[ids[k]]; c.ys[k] = ry[ids[k]]; }
        for (let k = 0; k < c.xs.length && c.xs.length > 3; ){
          const kp = (k - 1 + c.xs.length) % c.xs.length;
          if (c.xs[k] === c.xs[kp] && c.ys[k] === c.ys[kp]){ c.ls[kp] = c.ls[k]; c.xs.splice(k, 1); c.ys.splice(k, 1); c.ls.splice(k, 1); }
          else k++;
        }
      }
    }
    const keep = new Int32Array(N).fill(-1);
    const out = [];
    for (let i = 0; i < N; i++){
      const { xs, ys } = cells[i];
      if (xs.length < 3) continue;
      let A = 0;
      for (let k = 0; k < xs.length; k++){ const k2 = (k + 1) % xs.length; A += xs[k] * ys[k2] - xs[k2] * ys[k]; }
      if (A / 2 > 0.3){ keep[i] = out.length; out.push(i); }
    }
    const n = out.length;
    let total = 0;
    for (const i of out) total += cells[i].xs.length;
    const Gr = {
      n, s, off: new Int32Array(n), cnt: new Int32Array(n), vx: new Float64Array(total), vy: new Float64Array(total),
      lab: new Int32Array(total), cx: new Float64Array(n), cy: new Float64Array(n), area: new Float64Array(n),
      inertia: new Float64Array(n), rad: new Float64Array(n), adj: new Array(n),
    };
    let o = 0;
    totalArea0 = 0;
    for (let g = 0; g < n; g++){
      const { xs, ys, ls } = cells[out[g]];
      const m = xs.length;
      Gr.off[g] = o; Gr.cnt[g] = m;
      let A = 0, cx = 0, cy = 0, I = 0;
      for (let k = 0; k < m; k++){
        const k2 = (k + 1) % m;
        const x0 = xs[k], y0 = ys[k], x1 = xs[k2], y1 = ys[k2];
        const cr = x0 * y1 - x1 * y0;
        A += cr; cx += (x0 + x1) * cr; cy += (y0 + y1) * cr;
        I += cr * (x0 * x0 + x0 * x1 + x1 * x1 + y0 * y0 + y0 * y1 + y1 * y1);
        Gr.vx[o + k] = x0; Gr.vy[o + k] = y0;
        Gr.lab[o + k] = ls[k] >= 0 ? keep[ls[k]] : -1;
      }
      A /= 2; cx /= 6 * A; cy /= 6 * A; I /= 12;
      Gr.area[g] = A; Gr.cx[g] = cx; Gr.cy[g] = cy; Gr.inertia[g] = I - A * (cx * cx + cy * cy);
      totalArea0 += A;
      let R = 0;
      for (let k = 0; k < m; k++) R = Math.max(R, Math.hypot(xs[k] - cx, ys[k] - cy));
      Gr.rad[g] = R;
      o += m;
    }
    for (let g = 0; g < n; g++){
      const set = new Set();
      for (let k = Gr.off[g]; k < Gr.off[g] + Gr.cnt[g]; k++){
        const L2 = Gr.lab[k]; if (L2 < 0) continue;
        const k2 = k + 1 < Gr.off[g] + Gr.cnt[g] ? k + 1 : Gr.off[g];
        if (Math.hypot(Gr.vx[k2] - Gr.vx[k], Gr.vy[k2] - Gr.vy[k]) > 0.25) set.add(L2);
      }
      Gr.adj[g] = set;
    }
    for (let g = 0; g < n; g++) for (const h2 of Gr.adj[g]) Gr.adj[h2].add(g);
    for (let g = 0; g < n; g++) Gr.adj[g] = Int32Array.from(Gr.adj[g]);
    GR = Gr;
    lookCell = s; lookX0 = -hw - s; lookY0 = -hh - s;
    lookW = Math.ceil((w + 2 * s) / s) + 1; lookH = Math.ceil((h + 2 * s) / s) + 1;
    const cellsOf = [];
    const counts = new Int32Array(lookW * lookH + 1);
    for (let g = 0; g < n; g++){
      const R = Gr.rad[g];
      const i0 = Math.floor((Gr.cx[g] - R - lookX0) / s), i1 = Math.floor((Gr.cx[g] + R - lookX0) / s);
      const j0 = Math.floor((Gr.cy[g] - R - lookY0) / s), j1 = Math.floor((Gr.cy[g] + R - lookY0) / s);
      for (let j = Math.max(j0, 0); j <= Math.min(j1, lookH - 1); j++) for (let i = Math.max(i0, 0); i <= Math.min(i1, lookW - 1); i++){ cellsOf.push(j * lookW + i, g); counts[j * lookW + i]++; }
    }
    lookStart = new Int32Array(lookW * lookH + 1);
    for (let k = 0; k < lookW * lookH; k++) lookStart[k + 1] = lookStart[k] + counts[k];
    lookList = new Int32Array(lookStart[lookW * lookH]);
    const fill = lookStart.slice(0, lookW * lookH);
    for (let k = 0; k < cellsOf.length; k += 2) lookList[fill[cellsOf[k]]++] = cellsOf[k + 1];
    wX = new Float64Array(total); wY = new Float64Array(total);
    gMinX = new Float64Array(n); gMinY = new Float64Array(n); gMaxX = new Float64Array(n); gMaxY = new Float64Array(n);
    mark = new Int32Array(n);
    own = new Int32Array(n);
    vGrain = new Int32Array(total); vSeen = new Int32Array(total);
    for (let g = 0; g < n; g++) for (let k = Gr.off[g]; k < Gr.off[g] + Gr.cnt[g]; k++) vGrain[k] = g;
  }

  // The label, as the distance inside each letter (CSS px), for a true V-cut profile.
  let labelKey = '';
  let labelInk = null;               // CPU copy of the letter distances, and the ink's box in the rock frame
  const labelText = () => (button.textContent || '').replace(/\s+/g, ' ').trim();
  function buildText(){
    const w = rock.w, h = rock.h;
    const k = 2;
    const cw = Math.max(2, Math.round(w * k)), ch = Math.max(2, Math.round(h * k));
    const c = document.createElement('canvas');
    c.width = cw; c.height = ch;
    const x = c.getContext('2d', { willReadFrequently: true });
    x.fillStyle = '#000'; x.fillRect(0, 0, cw, ch);
    const cs = getComputedStyle(button);
    const text = labelText();
    const size = parseFloat(cs.fontSize) || 16;
    const weight = Math.max(parseInt(cs.fontWeight, 10) || 400, 500);
    const setFont = px => { x.font = `${cs.fontStyle} ${weight} ${px}px ${cs.fontFamily}`; };
    setFont(size * k);
    let fs = size * k;
    const m = x.measureText(text).width;
    if (m > cw * 0.86) fs *= cw * 0.86 / m;
    setFont(fs);
    x.textAlign = 'center'; x.textBaseline = 'middle';
    if ('letterSpacing' in x && cs.letterSpacing && cs.letterSpacing !== 'normal') x.letterSpacing = `${parseFloat(cs.letterSpacing) * k}px`;
    x.fillStyle = '#fff';
    x.fillText(text, cw / 2, ch / 2 + fs * 0.04);
    labelKey = text + '|' + cs.font;
    const img = x.getImageData(0, 0, cw, ch).data;
    // exact Euclidean distance to the letter outline (Felzenszwalb)
    const INF = 1e20, L = Math.max(cw, ch);
    const f = new Float64Array(L), dd = new Float64Array(L), v = new Int32Array(L), zz = new Float64Array(L + 1);
    const grid = new Float64Array(cw * ch);
    const outD = new Float32Array(cw * ch * 2);
    // channel 0: depth inside a letter; channel 1: distance outside one (for the stone's calm halo)
    for (let pass = 0; pass < 2; pass++){
    for (let i = 0; i < cw * ch; i++) grid[i] = (img[i * 4] > 127) === (pass === 0) ? INF : 0;
    const edt = n => {
      let q0 = 0; v[0] = 0; zz[0] = -INF; zz[1] = INF;
      for (let q = 1; q < n; q++){
        let s2 = ((f[q] + q * q) - (f[v[q0]] + v[q0] * v[q0])) / (2 * q - 2 * v[q0]);
        while (s2 <= zz[q0]){ q0--; s2 = ((f[q] + q * q) - (f[v[q0]] + v[q0] * v[q0])) / (2 * q - 2 * v[q0]); }
        q0++; v[q0] = q; zz[q0] = s2; zz[q0 + 1] = INF;
      }
      q0 = 0;
      for (let q = 0; q < n; q++){ while (zz[q0 + 1] < q) q0++; dd[q] = (q - v[q0]) * (q - v[q0]) + f[v[q0]]; }
    };
    for (let xx = 0; xx < cw; xx++){
      for (let yy = 0; yy < ch; yy++) f[yy] = grid[yy * cw + xx];
      edt(ch);
      for (let yy = 0; yy < ch; yy++) grid[yy * cw + xx] = dd[yy];
    }
    for (let yy = 0; yy < ch; yy++){
      for (let xx = 0; xx < cw; xx++) f[xx] = grid[yy * cw + xx];
      edt(cw);
      for (let xx = 0; xx < cw; xx++) outD[(yy * cw + xx) * 2 + pass] = dd[xx] > 0 ? Math.min(Math.max(Math.sqrt(dd[xx]) - 0.5, 0) / k, 60) : 0;
    }
    }
    let ix0 = cw, iy0 = ch, ix1 = -1, iy1 = -1;
    for (let yy = 0; yy < ch; yy++) for (let xx = 0; xx < cw; xx++) if (img[(yy * cw + xx) * 4] > 127){
      if (xx < ix0) ix0 = xx; if (xx > ix1) ix1 = xx; if (yy < iy0) iy0 = yy; if (yy > iy1) iy1 = yy;
    }
    labelInk = ix1 < 0 ? null : { d: outD, cw, ch, k, box: [ix0 / k - w / 2 - 2, iy0 / k - h / 2 - 2, (ix1 + 1) / k - w / 2 + 2, (iy1 + 1) / k - h / 2 + 2] };
    gl.bindTexture(gl.TEXTURE_2D, textTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG16F, cw, ch, 0, gl.RG, gl.FLOAT, outD);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  // ---- bodies
  let bodies = [];
  let nextId = 1;
  let simT = 0, stepNo = 0;
  const thickness = A => Math.min(rock.T, 0.75 * Math.sqrt(A) + 2);

  function makeBody(list, deferOutline){
    const Gr = GR;
    let A = 0, cx = 0, cy = 0;
    for (const g of list){ A += Gr.area[g]; cx += Gr.area[g] * Gr.cx[g]; cy += Gr.area[g] * Gr.cy[g]; }
    cx /= A; cy /= A;
    let I = 0, rbar = 0;
    for (const g of list){
      const dx = Gr.cx[g] - cx, dy = Gr.cy[g] - cy, d2 = dx * dx + dy * dy;
      I += Gr.inertia[g] + Gr.area[g] * d2;
      rbar += Gr.area[g] * (Math.sqrt(d2) + 0.4 * Gr.rad[g]);
    }
    rbar /= A;
    const b = {
      id: nextId++, grains: Int32Array.from(list), bnd: null, area: A,
      mass: A, invMass: 1 / A, I, invI: 1 / I, hx: cx, hy: cy, R: 0, rbar, decel: groundDecel(A), spinDecel: groundDecel(A) * A * rbar / I,
      x: 0, y: 0, a: 0, vx: 0, vy: 0, w: 0, dx: 0, dy: 0, da: 0, rest: 0, wv: false, deep: false,
      minX: 0, minY: 0, maxX: 0, maxY: 0, marg: 0,
      box: [0, 0, 0, 0], row: freeRows.pop() ?? -1, nO: 0, nT: 0,
      T: thickness(A), z: 0, seam: 1, home: false, kin: false, born: simT, bornStep: stepNo, glide: null, dead: false, family: 0, ghost: 0,
    };
    for (const g of list) own[g] = b.id;
    if (!deferOutline) buildOutline(b);
    return b;
  }

  // The outline of a piece: closed loops of grain edges that face another piece (fresh break) or the
  // outside of the stone (original surface). Along a break, the grain-scale zigzag is smoothed for
  // drawing. The smoothing only uses the crack's own vertices and pins every point where the piece on
  // the other side changes, so the two pieces on either side of a crack get the identical curve.
  function buildOutline(b){
    const Gr = GR;
    const stamp = ++markStamp;
    for (const g of b.grains) mark[g] = stamp;
    const outside = k => Gr.lab[k] < 0 || mark[Gr.lab[k]] !== stamp;
    const nextIn = k => { const g = vGrain[k], o = Gr.off[g]; return o + ((k - o + 1) % Gr.cnt[g]); };
    const startingAt = (g, x, y) => {
      const o = Gr.off[g]; let best = o, bd = Infinity;
      for (let k = o; k < o + Gr.cnt[g]; k++){ const d = (Gr.vx[k] - x) ** 2 + (Gr.vy[k] - y) ** 2; if (d < bd){ bd = d; best = k; } }
      return best;
    };
    const nextBoundary = k => {
      let kn = nextIn(k);
      for (let guard = 0; guard < 16; guard++){
        if (outside(kn)) return kn;
        kn = startingAt(Gr.lab[kn], Gr.vx[kn], Gr.vy[kn]);
      }
      return -1;
    };
    const vs = ++vStamp;
    const bnd = [];
    const orig = [], fresh = [];
    let R = 0, x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const g of b.grains){
      let isB = false;
      for (let k = Gr.off[g]; k < Gr.off[g] + Gr.cnt[g]; k++){
        R = Math.max(R, Math.hypot(Gr.vx[k] - b.hx, Gr.vy[k] - b.hy));
        if (!outside(k)) continue;
        isB = true;
        if (vSeen[k] === vs) continue;
        const loop = [];
        let e = k;
        for (let guard = 0; guard < 20000 && e >= 0 && vSeen[e] !== vs; guard++){ vSeen[e] = vs; loop.push(e); e = nextBoundary(e); }
        const n = loop.length;
        const xs = new Float64Array(n), ys = new Float64Array(n), lab = new Int32Array(n), fix = new Uint8Array(n);
        for (let i = 0; i < n; i++){ xs[i] = Gr.vx[loop[i]]; ys[i] = Gr.vy[loop[i]]; lab[i] = Gr.lab[loop[i]]; }
        for (let i = 0; i < n; i++){
          const lp = lab[(i - 1 + n) % n], lc = lab[i];
          fix[i] = lp < 0 || lc < 0 || own[lp] !== own[lc] ? 1 : 0;
        }
        for (let it = 0; it < 3; it++){
          const px = xs.slice(), py = ys.slice();
          for (let i = 0; i < n; i++){
            if (fix[i]) continue;
            const ip = (i - 1 + n) % n, inx = (i + 1) % n;
            xs[i] = 0.5 * px[i] + 0.25 * (px[ip] + px[inx]);
            ys[i] = 0.5 * py[i] + 0.25 * (py[ip] + py[inx]);
          }
        }
        // Drop vertices that don't change the drawn shape (the shader walks every segment for every
        // pixel). Runs are cut wherever the neighbour changes, and a crack's run is simplified from
        // the same points by both of its pieces, so the two sides stay identical.
        const key = i => lab[i] < 0 ? -1 : own[lab[i]];
        const keep = new Uint8Array(n);
        let nb = 0;
        for (let i = 0; i < n; i++) if (key((i - 1 + n) % n) !== key(i)){ keep[i] = 1; nb++; }
        if (nb === 0){
          let far = 0, fd = -1;
          for (let i = 1; i < n; i++){ const d = (xs[i] - xs[0]) ** 2 + (ys[i] - ys[0]) ** 2; if (d > fd){ fd = d; far = i; } }
          keep[0] = 1; keep[far] = 1;
        }
        const dp = (i0, i1, tol) => {
          const len = (i1 - i0 + n) % n;
          if (len < 2) return;
          const ax = xs[i0], ay = ys[i0], ex = xs[i1] - ax, ey = ys[i1] - ay, el = Math.hypot(ex, ey) || 1e-9;
          let best = -1, bd = tol;
          for (let s = 1; s < len; s++){
            const k2 = (i0 + s) % n;
            const d = Math.abs((xs[k2] - ax) * ey - (ys[k2] - ay) * ex) / el;
            if (d > bd){ bd = d; best = k2; }
          }
          if (best < 0) return;
          keep[best] = 1;
          dp(i0, best, tol); dp(best, i1, tol);
        };
        const ks = [];
        for (let i = 0; i < n; i++) if (keep[i]) ks.push(i);
        for (let q = 0; q < ks.length; q++) dp(ks[q], ks[(q + 1) % ks.length], 0.12);
        let first = -1, prev = -1;
        const emit = (i, j) => {
          const ax = xs[i] - b.hx, ay = ys[i] - b.hy, bx = xs[j] - b.hx, by = ys[j] - b.hy;
          (lab[i] < 0 ? orig : fresh).push(ax, ay, bx, by);
          if (ax < x0) x0 = ax; if (ay < y0) y0 = ay; if (ax > x1) x1 = ax; if (ay > y1) y1 = ay;
        };
        for (let i = 0; i < n; i++){
          if (!keep[i]) continue;
          if (prev >= 0) emit(prev, i); else first = i;
          prev = i;
        }
        if (prev >= 0 && first >= 0) emit(prev, first);
      }
      if (isB) bnd.push(g);
    }
    b.bnd = Int32Array.from(bnd);
    b.R = R;
    let fl = 0;
    for (let i = 0; i < fresh.length; i += 4) fl += Math.hypot(fresh[i + 2] - fresh[i], fresh[i + 3] - fresh[i + 1]);
    b.freshLen = fl;
    b.box = [x0, y0, x1, y1];
    const nT = Math.min((orig.length + fresh.length) / 4, SEGW);
    b.nT = nT; b.nO = Math.min(orig.length / 4, nT);
    const data = new Float32Array(nT * 4);
    data.set(orig.slice(0, Math.min(orig.length, nT * 4)));
    if (orig.length < nT * 4) data.set(fresh.slice(0, nT * 4 - orig.length), orig.length);
    b.segs = data;
    if (b.row >= 0){
      gl.bindTexture(gl.TEXTURE_2D, segTex);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, b.row, nT, 1, gl.RGBA, gl.FLOAT, data);
    }
    b.wv = false;
  }
  function freeBody(b){ if (b.row >= 0) freeRows.push(b.row); b.row = -1; b.dead = true; }

  function toRock(b, x, y){
    const c = Math.cos(b.a), s = Math.sin(b.a), dx = x - b.x, dy = y - b.y;
    return [c * dx + s * dy + b.hx, -s * dx + c * dy + b.hy];
  }
  function grainHas(g, x, y){
    const Gr = GR, o = Gr.off[g], m = Gr.cnt[g];
    if (Math.abs(x - Gr.cx[g]) > Gr.rad[g] || Math.abs(y - Gr.cy[g]) > Gr.rad[g]) return false;
    for (let i = 0; i < m; i++){
      const k = o + i, k2 = o + (i + 1) % m;
      if ((Gr.vx[k2] - Gr.vx[k]) * (y - Gr.vy[k]) - (Gr.vy[k2] - Gr.vy[k]) * (x - Gr.vx[k]) < 0) return false;
    }
    return true;
  }
  function grainAt(rx, ry, id){
    const i = Math.floor((rx - lookX0) / lookCell), j = Math.floor((ry - lookY0) / lookCell);
    if (i < 0 || j < 0 || i >= lookW || j >= lookH) return -1;
    const k = j * lookW + i;
    for (let q = lookStart[k]; q < lookStart[k + 1]; q++){
      const g = lookList[q];
      if (own[g] === id && grainHas(g, rx, ry)) return g;
    }
    return -1;
  }
  function bodyContains(b, x, y){
    if ((x - b.x) ** 2 + (y - b.y) ** 2 > (b.R + 1) ** 2) return false;
    const c = Math.cos(b.a), s = Math.sin(b.a), dx = x - b.x, dy = y - b.y;
    return grainAt(c * dx + s * dy + b.hx, -s * dx + c * dy + b.hy, b.id) >= 0;
  }
  function bodyAt(x, y, slack){
    let best = null, bd = Infinity;
    for (let i = bodies.length - 1; i >= 0; i--){
      const b = bodies[i];
      if (bodyContains(b, x, y)) return b;
      if (slack > 0 && b.R < 14){
        const d = Math.hypot(x - b.x, y - b.y) - b.R;
        if (d < slack && d < bd){ bd = d; best = b; }
      }
    }
    return best;
  }

  function nearBody(x, y, r){
    for (const b of bodies){
      if ((x - b.x) ** 2 + (y - b.y) ** 2 > (b.R + r) ** 2) continue;
      for (let k = 0; k < 12; k++){
        const a = k * TAU / 12, rr = k & 1 ? r : r * 0.5;
        if (bodyContains(b, x + Math.cos(a) * rr, y + Math.sin(a) * rr)) return true;
      }
    }
    return false;
  }

  // ---- physics
  // 2D rigid bodies on a floor, seen from above. Sequential impulses with soft contacts (Box2D v3 style),
  // collision detected every substep with speculative margins sized from the relative motion so nothing
  // tunnels, warm starting keyed by grain pair + feature.
  let contactSoft = SOFTS[0].contact, wallSoft = SOFTS[0].wall, dragSoft = SOFTS[0].drag, curH = H_STEP;
  // A soft contact's stiffness is relative to the pair's effective mass, which the lighter piece sets.
  // A heavy slab shoved onto a pebble would sink into it by the mass ratio, so such contacts stiffen
  // in proportion (up to what the substep can resolve).
  const softCache = new Map();
  function pairSoft(ratio){
    const q = Math.min(Math.round(Math.sqrt(ratio) * 4), 400);
    const key = q * 8 + Math.round(H_STEP / curH);
    let s = softCache.get(key);
    if (!s){ s = softness(Math.min(CONTACT_HZ * q / 4, 0.25 / curH), CONTACT_ZETA, curH); softCache.set(key, s); }
    return s;
  }
  const contacts = [];
  let nContacts = 0;
  let cachePrevN = new Map(), cachePrevT = new Map(), cacheN = new Map(), cacheT = new Map();
  const knocks = [];
  let fractureCount = 0, settleUntil = 0;
  function newContact(){ if (nContacts === contacts.length) contacts.push({}); return contacts[nContacts++]; }

  function updateWorld(b){
    const Gr = GR, c = Math.cos(b.a), s = Math.sin(b.a);
    let mnx = Infinity, mny = Infinity, mxx = -Infinity, mxy = -Infinity;
    for (const g of b.bnd){
      let ax = Infinity, ay = Infinity, bx = -Infinity, by = -Infinity;
      const o = Gr.off[g], e = o + Gr.cnt[g];
      for (let k = o; k < e; k++){
        const lx = Gr.vx[k] - b.hx, ly = Gr.vy[k] - b.hy;
        const x = b.x + c * lx - s * ly, y = b.y + s * lx + c * ly;
        wX[k] = x; wY[k] = y;
        if (x < ax) ax = x; if (x > bx) bx = x; if (y < ay) ay = y; if (y > by) by = y;
      }
      gMinX[g] = ax; gMinY[g] = ay; gMaxX[g] = bx; gMaxY[g] = by;
      if (ax < mnx) mnx = ax; if (ay < mny) mny = ay; if (bx > mxx) mxx = bx; if (by > mxy) mxy = by;
    }
    b.bx0 = mnx; b.by0 = mny; b.bx1 = mxx; b.by1 = mxy;
    b.wv = true;
  }
  let msEdge = 0;
  function maxSep(g1, g2){
    const Gr = GR, o1 = Gr.off[g1], n1 = Gr.cnt[g1], o2 = Gr.off[g2], n2 = Gr.cnt[g2];
    let best = -Infinity, bi = 0;
    for (let i = 0; i < n1; i++){
      const k = o1 + i, k2 = o1 + (i + 1 === n1 ? 0 : i + 1);
      const ex = wX[k2] - wX[k], ey = wY[k2] - wY[k];
      const L = Math.hypot(ex, ey);
      if (L < 1e-6) continue;
      const nx = ey / L, ny = -ex / L;
      let mn = Infinity;
      for (let j = o2; j < o2 + n2; j++){
        const d = nx * (wX[j] - wX[k]) + ny * (wY[j] - wY[k]);
        if (d < mn) mn = d;
      }
      if (mn > best){ best = mn; bi = i; }
    }
    msEdge = bi;
    return best;
  }
  let seamNx = 0, seamNy = 0, seamOn = false;
  function addContact(A, B, px, py, nx, ny, sep, key){
    // a pair already driven into each other gets its true contact back, or a sliver could be pushed
    // sideways into its sibling's notch where no single push direction frees it
    if (seamOn && sep > -0.6){ nx = seamNx; ny = seamNy; if (sep < 0) sep = 0; }
    const c = newContact();
    c.A = A; c.B = B; c.nx = nx; c.ny = ny; c.s0 = sep; c.key = key;
    const mA = A ? A.invMass : 0, iA = A ? A.invI : 0, mB = B.invMass, iB = B.invI;
    c.soft = key <= -1e9 ? pairSoft(64) : A ? (A.mass > B.mass * 1.5 || B.mass > A.mass * 1.5 ? pairSoft(Math.max(A.mass / B.mass, B.mass / A.mass)) : contactSoft) : wallSoft;
    c.push = key <= -1e9 ? PUSH_MAX * 2.5 : PUSH_MAX;
    const rAx = A ? px - A.x : 0, rAy = A ? py - A.y : 0, rBx = px - B.x, rBy = py - B.y;
    c.rAx = rAx; c.rAy = rAy; c.rBx = rBx; c.rBy = rBy;
    const rnA = rAx * ny - rAy * nx, rnB = rBx * ny - rBy * nx;
    c.nm = 1 / (mA + mB + iA * rnA * rnA + iB * rnB * rnB);
    const tx = ny, ty = -nx;
    const rtA = rAx * ty - rAy * tx, rtB = rBx * ty - rBy * tx;
    c.tm = 1 / (mA + mB + iA * rtA * rtA + iB * rtB * rtB);
    const vAx = A ? A.vx - A.w * rAy : 0, vAy = A ? A.vy + A.w * rAx : 0;
    const vBx = B.vx - B.w * rBy, vBy = B.vy + B.w * rBx;
    c.relVel = (vBx - vAx) * nx + (vBy - vAy) * ny;
    c.uA = A ? vAx * nx + vAy * ny : 0;
    c.uB = -(vBx * nx + vBy * ny);
    c.px = px; c.py = py;
    const pn = cachePrevN.get(key);
    c.isNew = pn === undefined;
    // Warm-start impulses are keyed by grain pair, so a fragment would inherit the whole impulse its
    // parent was carrying at that contact; on a light fragment that is a launch, not a nudge.
    if ((A && A.bornStep >= stepNo - 1) || B.bornStep >= stepNo - 1){ c.ni = 0; c.ti = 0; }
    else { c.ni = pn || 0; c.ti = cachePrevT.get(key) || 0; }
    c.maxNi = 0;
  }
  function collideGrains(A, B, gA, gB, margin){
    const sA = maxSep(gA, gB); if (sA > margin) return; const eA = msEdge;
    const sB = maxSep(gB, gA); if (sB > margin) return; const eB = msEdge;
    const Gr = GR;
    let ref, inc, e, flip;
    if (sB > sA + 0.02){ ref = gB; inc = gA; e = eB; flip = 1; } else { ref = gA; inc = gB; e = eA; flip = 0; }
    const ro = Gr.off[ref], rn = Gr.cnt[ref];
    const k1 = ro + e, k2 = ro + (e + 1) % rn;
    const v1x = wX[k1], v1y = wY[k1], v2x = wX[k2], v2y = wY[k2];
    const L = Math.hypot(v2x - v1x, v2y - v1y);
    const tx = (v2x - v1x) / L, ty = (v2y - v1y) / L;
    const nx = ty, ny = -tx;
    const io = Gr.off[inc], inn = Gr.cnt[inc];
    let ie = 0, mn = Infinity;
    for (let i = 0; i < inn; i++){
      const a = io + i, b2 = io + (i + 1) % inn;
      const ex = wX[b2] - wX[a], ey = wY[b2] - wY[a], l = Math.hypot(ex, ey);
      if (l < 1e-6) continue;
      const d = (ey * nx - ex * ny) / l;
      if (d < mn){ mn = d; ie = i; }
    }
    const i1 = io + ie, i2 = io + (ie + 1) % inn;
    let p1x = wX[i1], p1y = wY[i1], p2x = wX[i2], p2y = wY[i2];
    let d1 = tx * (p1x - v1x) + ty * (p1y - v1y), d2 = tx * (p2x - v1x) + ty * (p2y - v1y);
    if (d1 < 0 && d2 < 0) return;
    if (d1 < 0){ const t = d1 / (d1 - d2); p1x += (p2x - p1x) * t; p1y += (p2y - p1y) * t; }
    else if (d2 < 0){ const t = d2 / (d2 - d1); p2x += (p1x - p2x) * t; p2y += (p1y - p2y) * t; }
    d1 = tx * (p1x - v2x) + ty * (p1y - v2y); d2 = tx * (p2x - v2x) + ty * (p2y - v2y);
    if (d1 > 0 && d2 > 0) return;
    if (d1 > 0){ const t = d1 / (d1 - d2); p1x += (p2x - p1x) * t; p1y += (p2y - p1y) * t; }
    else if (d2 > 0){ const t = d2 / (d2 - d1); p2x += (p1x - p2x) * t; p2y += (p1y - p2y) * t; }
    const onx = flip ? -nx : nx, ony = flip ? -ny : ny;
    const base = (gA * 4096 + gB) * 1024 + flip * 512 + (e & 15) * 32 + (ie & 15) * 2;
    const s1 = nx * (p1x - v1x) + ny * (p1y - v1y);
    if (s1 <= margin) addContact(A, B, p1x - nx * s1 * 0.5, p1y - ny * s1 * 0.5, onx, ony, s1, base);
    const s2 = nx * (p2x - v1x) + ny * (p2y - v1y);
    if (s2 <= margin) addContact(A, B, p2x - nx * s2 * 0.5, p2y - ny * s2 * 0.5, onx, ony, s2, base + 1);
  }
  // Fragments of one break start out touching along jagged, interlocking seams. While they spread,
  // contacts between grains that were neighbours across a seam would lock them together through the
  // teeth. For a moment those contacts push only along the line between the two fragments' centres.
  function seamPair(ga, gb){
    const adj = GR.adj;
    for (const n of adj[ga]){ if (n === gb) return true; for (const m of adj[n]) if (m === gb) return true; }
    return false;
  }
  const selA = [], selB = [];
  function collideBodies(A, B){
    const ghost = A.family === B.family && simT < A.ghost;
    if (deepPairs.has(pairKey(A, B))){ collideDeep(A, B); collideDeep(B, A); return; }
    const gsA = A.bnd, gsB = B.bnd;
    if (ghost){ const dx = B.x - A.x, dy = B.y - A.y, l = Math.hypot(dx, dy) || 1; seamNx = dx / l; seamNy = dy / l; }
    const margin = SLOP + A.marg + B.marg;
    selA.length = 0; selB.length = 0;
    const x0 = B.bx0 - margin, y0 = B.by0 - margin, x1 = B.bx1 + margin, y1 = B.by1 + margin;
    for (const g of gsA) if (gMaxX[g] >= x0 && gMinX[g] <= x1 && gMaxY[g] >= y0 && gMinY[g] <= y1) selA.push(g);
    if (!selA.length) return;
    const ax0 = A.bx0 - margin, ay0 = A.by0 - margin, ax1 = A.bx1 + margin, ay1 = A.by1 + margin;
    for (const g of gsB) if (gMaxX[g] >= ax0 && gMinX[g] <= ax1 && gMaxY[g] >= ay0 && gMinY[g] <= ay1) selB.push(g);
    for (const ga of selA){
      const gx0 = gMinX[ga] - margin, gy0 = gMinY[ga] - margin, gx1 = gMaxX[ga] + margin, gy1 = gMaxY[ga] + margin;
      for (const gb of selB){
        if (gMaxX[gb] < gx0 || gMinX[gb] > gx1 || gMaxY[gb] < gy0 || gMinY[gb] > gy1) continue;
        seamOn = ghost && seamPair(ga, gb);
        collideGrains(A, B, ga, gb, margin);
        seamOn = false;
      }
    }
  }
  // The stage's edges are the walls, but a piece is drawn past its footprint: its south wall and its
  // shadow fall down and to the right, and its antialiased rim and contact shade reach a little all
  // round. So the walls stand that far inside the canvas and no piece is ever sliced by the canvas edge
  // (never further in than the button itself, which a stage squeezed against the document edge may
  // leave no room for).
  const wallBox = [0, 0, 0, 0];
  function wallsFor(T){
    const run = T * SHADOW_RUN;
    const l = Math.min(6, G.offX), t = Math.min(6, G.offY);
    const r = Math.min(SHADOW_DIR[0] * run + 4, G.stageW - G.offX - G.bw);
    const bt = Math.min(SHADOW_DIR[1] * run + T * LEAN + 4, G.stageH - G.offY - G.bh);
    wallBox[0] = l; wallBox[1] = t; wallBox[2] = G.stageW - Math.max(r, 0); wallBox[3] = G.stageH - Math.max(bt, 0);
    return wallBox;
  }
  function collideWalls(b){
    const m = b.marg + SLOP;
    const [L, Tp, R, B] = wallsFor(b.T);
    if (b.bx0 > L + m && b.by0 > Tp + m && b.bx1 < R - m && b.by1 < B - m) return;
    const Gr = GR;
    for (let w = 0; w < 4; w++){
      const nx = w === 0 ? 1 : w === 1 ? -1 : 0, ny = w === 2 ? 1 : w === 3 ? -1 : 0;
      const c0 = w === 0 ? -L : w === 1 ? R : w === 2 ? -Tp : B;
      let b1 = Infinity, i1 = -1, b2 = Infinity, i2 = -1;
      for (const g of b.bnd){
        for (let k = Gr.off[g]; k < Gr.off[g] + Gr.cnt[g]; k++){
          const s = nx * wX[k] + ny * wY[k] + c0;
          if (s > m) continue;
          if (s < b1){ b2 = b1; i2 = i1; b1 = s; i1 = k; }
          else if (s < b2 && Math.hypot(wX[k] - wX[i1], wY[k] - wY[i1]) > 2){ b2 = s; i2 = k; }
        }
      }
      if (i1 >= 0) addContact(null, b, wX[i1], wY[i1], nx, ny, b1, -(i1 * 4 + w + 1));
      if (i2 >= 0) addContact(null, b, wX[i2], wY[i2], nx, ny, b2, -(i2 * 4 + w + 1));
    }
  }
  // The soft wall contacts can be overpowered for a moment (a piece dragged hard into a wall, a heavy
  // slab shoving gravel); a piece is never let past a wall even then, only stopped at it.
  function confine(b){
    const R0 = b.R * Math.abs(b.da);
    const [L, Tp, R, B] = wallsFor(b.T);
    if (b.bx0 + b.dx - R0 > L && b.by0 + b.dy - R0 > Tp && b.bx1 + b.dx + R0 < R && b.by1 + b.dy + R0 < B) return;
    updateWorld(b);
    let sx = 0, sy = 0;
    if (b.bx1 - b.bx0 > R - L) sx = (L + R - b.bx0 - b.bx1) / 2;
    else if (b.bx0 < L) sx = L - b.bx0;
    else if (b.bx1 > R) sx = R - b.bx1;
    if (b.by1 - b.by0 > B - Tp) sy = (Tp + B - b.by0 - b.by1) / 2;
    else if (b.by0 < Tp) sy = Tp - b.by0;
    else if (b.by1 > B) sy = B - b.by1;
    if (!sx && !sy) return;
    b.x += sx; b.y += sy; b.wv = false; b.rest = 0;
    if (sx > 0 && b.vx < 0 || sx < 0 && b.vx > 0) b.vx = 0;
    if (sy > 0 && b.vy < 0 || sy < 0 && b.vy > 0) b.vy = 0;
  }

  const deepPairs = new Map();
  const jammed = [];
  function pairKey(A, B){ return A.id < B.id ? A.id * 1048576 + B.id : B.id * 1048576 + A.id; }
  // Any boundary grain of one piece whose centre lies inside another piece means they overlap by more
  // than the contact tests can see. Checked once a frame, for every pair whose boxes overlap.
  function findDeepPairs(){
    const Gr = GR;
    for (const [k, t] of deepPairs) if (t < simT) deepPairs.delete(k);
    for (let i = 0; i < jammed.length; i += 2){ deepPairs.set(pairKey(jammed[i], jammed[i + 1]), simT + 0.2); jammed[i].rest = jammed[i + 1].rest = 0; }
    const hadJam = jammed.length > 0;
    jammed.length = 0;
    const list = bodies.filter(b => !b.kin);
    for (const b of list){ if (!b.wv) updateWorld(b); }
    list.sort((p, q) => p.bx0 - q.bx0);
    let changed = hadJam;
    for (let i = 0; i < list.length; i++){
      const A = list[i];
      for (let j = i + 1; j < list.length; j++){
        const B = list[j];
        if (B.bx0 > A.bx1) break;
        if (B.by0 > A.by1 || B.by1 < A.by0) continue;
        const key = pairKey(A, B);
        const [S, O] = A.bnd.length <= B.bnd.length ? [A, B] : [B, A];
        const c = Math.cos(S.a), s = Math.sin(S.a);
        for (const g of S.bnd){
          const lx = Gr.cx[g] - S.hx, ly = Gr.cy[g] - S.hy;
          const x = S.x + c * lx - s * ly, y = S.y + s * lx + c * ly;
          if (x < O.bx0 || x > O.bx1 || y < O.by0 || y > O.by1) continue;
          if (bodyContains(O, x, y)){
            deepPairs.set(key, simT + 0.3);
            A.rest = B.rest = 0;
            changed = true;
            break;
          }
        }
      }
    }
    for (const b of list){
      const want = changed || b.deep ? hasDeep(b) : false;
      if (want !== b.deep) b.deep = want;
    }
  }
  function hasDeep(b){
    for (const k of deepPairs.keys()){ const a = Math.floor(k / 1048576), c = k - a * 1048576; if (a === b.id || c === b.id) return true; }
    return false;
  }
  // A pair overlapping by more than a grain is pushed apart along one direction for the whole pair, the
  // line between their centres (pushes through each one's nearest edge can point opposite ways and
  // cancel, leaving the pair stuck).
  function collideDeep(O, S){
    const Gr = GR, seg = O.segs;
    if (!seg) return;
    let nx = S.x - O.x, ny = S.y - O.y;
    const l = Math.hypot(nx, ny);
    if (l < 1e-3){ nx = 1; ny = 0; } else { nx /= l; ny /= l; }
    const cS = Math.cos(S.a), sS = Math.sin(S.a), cO = Math.cos(O.a), sO = Math.sin(O.a);
    const dlx = cO * nx + sO * ny, dly = -sO * nx + cO * ny;
    for (const g of S.bnd){
      const lx = Gr.cx[g] - S.hx, ly = Gr.cy[g] - S.hy;
      const x = S.x + cS * lx - sS * ly, y = S.y + sS * lx + cS * ly;
      if (x < O.bx0 || x > O.bx1 || y < O.by0 || y > O.by1) continue;
      const dx = x - O.x, dy = y - O.y;
      const px = cO * dx + sO * dy, py = -sO * dx + cO * dy;
      if (grainAt(px + O.hx, py + O.hy, O.id) < 0) continue;
      let tMax = 0;
      for (let i = 0; i < seg.length; i += 4){
        const ax = seg[i], ay = seg[i + 1], ex = seg[i + 2] - ax, ey = seg[i + 3] - ay;
        const den = dlx * ey - dly * ex;
        if (Math.abs(den) < 1e-12) continue;
        const qx = ax - px, qy = ay - py;
        const t = (qx * ey - qy * ex) / den, u = (qx * dly - qy * dlx) / den;
        if (t > tMax && u >= 0 && u <= 1) tMax = t;
      }
      if (tMax <= 0) continue;
      const sep = -(tMax + 0.5 * Gr.rad[g]);
      const key = -(1e9 + (O.id % 4096) * 1e5 + g);
      if (O.id < S.id) addContact(O, S, x, y, nx, ny, sep, key);
      else addContact(S, O, x, y, -nx, -ny, sep, key);
    }
  }

  const dyn = [];
  function collide(h){
    nContacts = 0;
    [cachePrevN, cacheN] = [cacheN, cachePrevN];
    [cachePrevT, cacheT] = [cacheT, cachePrevT];
    cacheN.clear(); cacheT.clear();
    for (const b of dyn){
      if (!b.wv) updateWorld(b);
      b.marg = (Math.hypot(b.vx, b.vy) + Math.abs(b.w) * b.R) * h;
      const m = b.marg + SLOP;
      b.minX = b.bx0 - m; b.maxX = b.bx1 + m; b.minY = b.by0 - m; b.maxY = b.by1 + m;
      b.dx = b.dy = b.da = 0;
    }
    dyn.sort((p, q) => p.minX - q.minX);
    for (let i = 0; i < dyn.length; i++){
      const A = dyn[i];
      for (let j = i + 1; j < dyn.length; j++){
        const B = dyn[j];
        if (B.minX > A.maxX) break;
        if (B.minY > A.maxY || B.maxY < A.minY) continue;
        if (A.rest > 0.25 && B.rest > 0.25 && !A.deep && !B.deep) continue;
        if (A.id < B.id) collideBodies(A, B); else collideBodies(B, A);
      }
    }
    // walls last, so each Gauss-Seidel sweep ends on them: a pebble squeezed against a wall by a heavy
    // slab is then pushed back out rather than left wherever the slab's contact put it
    for (const b of dyn) collideWalls(b);
  }
  function applyImpulse(c, px, py){
    const A = c.A, B = c.B;
    if (A){ A.vx -= A.invMass * px; A.vy -= A.invMass * py; A.w -= A.invI * (c.rAx * py - c.rAy * px); }
    B.vx += B.invMass * px; B.vy += B.invMass * py; B.w += B.invI * (c.rBx * py - c.rBy * px);
  }
  function warmStart(){
    for (let i = 0; i < nContacts; i++){
      const c = contacts[i];
      if (c.ni === 0 && c.ti === 0) continue;
      applyImpulse(c, c.ni * c.nx + c.ti * c.ny, c.ni * c.ny - c.ti * c.nx);
    }
  }
  function solveContacts(useBias, invH){
    for (let i = 0; i < nContacts; i++){
      const c = contacts[i], A = c.A, B = c.B;
      const sft = c.soft;
      let dsx = B.dx - B.da * c.rBy, dsy = B.dy + B.da * c.rBx;
      if (A){ dsx -= A.dx - A.da * c.rAy; dsy -= A.dy + A.da * c.rAx; }
      const s = c.s0 + dsx * c.nx + dsy * c.ny;
      let bias = 0, ms = 1, is = 0;
      if (s > 0) bias = s * invH;
      else if (useBias){ bias = Math.max(sft.biasRate * Math.min(0, s + SLOP), -c.push); ms = sft.massScale; is = sft.impulseScale; }
      let vAx = 0, vAy = 0;
      if (A){ vAx = A.vx - A.w * c.rAy; vAy = A.vy + A.w * c.rAx; }
      let dvx = B.vx - B.w * c.rBy - vAx, dvy = B.vy + B.w * c.rBx - vAy;
      const vn = dvx * c.nx + dvy * c.ny;
      let imp = -c.nm * ms * (vn + bias) - is * c.ni;
      const ni = Math.max(c.ni + imp, 0);
      imp = ni - c.ni; c.ni = ni;
      if (imp > c.maxNi) c.maxNi = imp;
      applyImpulse(c, imp * c.nx, imp * c.ny);
      if (A){ vAx = A.vx - A.w * c.rAy; vAy = A.vy + A.w * c.rAx; }
      dvx = B.vx - B.w * c.rBy - vAx; dvy = B.vy + B.w * c.rBx - vAy;
      const tx = c.ny, ty = -c.nx;
      const vt = dvx * tx + dvy * ty;
      const maxF = MU * c.ni;
      const ti = clamp(c.ti - c.tm * vt, -maxF, maxF);
      const timp = ti - c.ti; c.ti = ti;
      applyImpulse(c, timp * tx, timp * ty);
    }
  }
  // Overlap past `tol` px is also removed by moving the pieces directly (a split impulse), at most
  // `cap` px per pass: the velocity solver alone would push them out by adding speed, which a pile
  // squeezed by a drag or a resize turns into jitter and flung gravel.
  function pushApart(tol, cap, iters){
    for (let it = 0; it < iters; it++){
      let worst = 0;
      for (let i = 0; i < nContacts; i++){
        const c = contacts[i], A = c.A, B = c.B;
        let dsx = B.dx - B.da * c.rBy, dsy = B.dy + B.da * c.rBx;
        if (A){ dsx -= A.dx - A.da * c.rAy; dsy -= A.dy + A.da * c.rAx; }
        const s = c.s0 + dsx * c.nx + dsy * c.ny;
        if (s >= -tol) continue;
        if (s < worst) worst = s;
        const Pm = Math.min(-(s + tol) * 0.8, cap) * c.nm;
        const px = Pm * c.nx, py = Pm * c.ny;
        if (A){
          const ddx = -A.invMass * px, ddy = -A.invMass * py, dda = -A.invI * (c.rAx * py - c.rAy * px);
          A.x += ddx; A.y += ddy; A.a += dda; A.dx += ddx; A.dy += ddy; A.da += dda; A.wv = false; A.rest = 0;
        }
        const ddx = B.invMass * px, ddy = B.invMass * py, dda = B.invI * (c.rBx * py - c.rBy * px);
        B.x += ddx; B.y += ddy; B.a += dda; B.dx += ddx; B.dy += ddy; B.da += dda; B.wv = false; B.rest = 0;
      }
      if (worst === 0) return 0;
    }
    return 1;
  }
  function restitution(){
    for (let i = 0; i < nContacts; i++){
      const c = contacts[i];
      if (c.relVel > -REST_THRESH || c.maxNi === 0) continue;
      const A = c.A, B = c.B;
      let vAx = 0, vAy = 0;
      if (A){ vAx = A.vx - A.w * c.rAy; vAy = A.vy + A.w * c.rAx; }
      const vn = (B.vx - B.w * c.rBy - vAx) * c.nx + (B.vy + B.w * c.rBx - vAy) * c.ny;
      let imp = -c.nm * (vn + RESTITUTION * c.relVel);
      const ni = Math.max(c.ni + imp, 0);
      imp = ni - c.ni; c.ni = ni;
      applyImpulse(c, imp * c.nx, imp * c.ny);
    }
  }

  // ---- pointer joints
  const grabs = new Map();     // pointerId -> { body, lx, ly, tx, ty, ix, iy, ... }
  function grabPoint(gr){
    const b = gr.body, c = Math.cos(b.a), s = Math.sin(b.a);
    return [c * gr.lx - s * gr.ly, s * gr.lx + c * gr.ly];
  }
  function warmJoints(){
    for (const gr of grabs.values()){
      const b = gr.body, [rx, ry] = grabPoint(gr);
      b.vx += b.invMass * gr.ix; b.vy += b.invMass * gr.iy; b.w += b.invI * (rx * gr.iy - ry * gr.ix);
    }
  }
  function solveJoints(h){
    const sft = dragSoft;
    for (const gr of grabs.values()){
      const b = gr.body, [rx, ry] = grabPoint(gr);
      const Cx = b.x + b.dx + rx - gr.tx, Cy = b.y + b.dy + ry - gr.ty;
      const m = b.invMass, i = b.invI;
      const k11 = m + i * ry * ry, k12 = -i * rx * ry, k22 = m + i * rx * rx;
      const det = k11 * k22 - k12 * k12, id = det !== 0 ? 1 / det : 0;
      const vx = b.vx - b.w * ry + sft.biasRate * Cx, vy = b.vy + b.w * rx + sft.biasRate * Cy;
      let jx = -sft.massScale * id * (k22 * vx - k12 * vy) - sft.impulseScale * gr.ix;
      let jy = -sft.massScale * id * (-k12 * vx + k11 * vy) - sft.impulseScale * gr.iy;
      const ox = gr.ix, oy = gr.iy;
      let nx = ox + jx, ny = oy + jy;
      const maxI = DRAG_ACCEL * b.mass * h, l = Math.hypot(nx, ny);
      if (l > maxI){ nx *= maxI / l; ny *= maxI / l; }
      gr.ix = nx; gr.iy = ny; jx = nx - ox; jy = ny - oy;
      b.vx += m * jx; b.vy += m * jy; b.w += i * (rx * jy - ry * jx);
      b.w *= 1 / (1 + h * 0.8);
    }
  }

  // ---- stepping
  const impactEvents = new Map();
  function physicsStep(h){
    stepNo++;
    dyn.length = 0;
    for (const b of bodies) if (!b.kin) dyn.push(b);
    // Coulomb friction with the ground. Sliding and spinning share one friction budget (a cheap stand-in
    // for the plate's limit surface): a piece that is sliding fast turns easily, and vice versa.
    for (const b of dyn){
      const sp = Math.hypot(b.vx, b.vy), rv = Math.abs(b.w) * b.rbar;
      const sum = sp + rv;
      if (sum === 0) continue;
      const dv = b.decel * h * (sp / sum), dw = b.spinDecel * h * (rv / sum);
      if (sp <= dv){ b.vx = 0; b.vy = 0; }
      else {
        const f = (sp > MAX_SPEED ? MAX_SPEED / sp : 1) * (1 - dv / sp);
        b.vx *= f; b.vy *= f;
      }
      if (Math.abs(b.w) <= dw) b.w = 0; else b.w -= Math.sign(b.w) * dw;
      b.w = clamp(b.w, -MAX_SPIN, MAX_SPIN);
      if (sp < 1.5 && rv < 1.5){ b.vx = b.vy = b.w = 0; }
    }
    collide(h);
    warmStart();
    warmJoints();
    const invH = 1 / h;
    for (let it = 0; it < SOLVE_ITERS; it++){ solveJoints(h); solveContacts(true, invH); }
    for (const b of dyn){
      if (b.vx === 0 && b.vy === 0 && b.w === 0){ b.rest += h; continue; }
      b.rest = 0;
      b.dx = b.vx * h; b.dy = b.vy * h; b.da = b.w * h;
      b.x += b.dx; b.y += b.dy; b.a += b.da;
      b.wv = false;
    }
    pushApart(DEEP_PUSH, 0.5, 2);
    for (const b of dyn) confine(b);
    for (let it = 0; it < RELAX_ITERS; it++) solveContacts(false, invH);
    restitution();
    impactEvents.clear();
    for (let i = 0; i < nContacts; i++){
      const c = contacts[i];
      cacheN.set(c.key, c.ni); cacheT.set(c.key, c.ti);
      if (c.s0 < -SLOP - 0.1){ if (c.A) c.A.rest = 0; c.B.rest = 0; }
      if (c.A && c.s0 < -2 && c.key > -1e9 && Math.abs(c.relVel) < 30) jammed.push(c.A, c.B);
      if (c.maxNi === 0) continue;
      if (c.isNew && c.relVel < -45) knocks.push({ v: -c.relVel, m: c.A ? Math.min(c.A.mass, c.B.mass) : c.B.mass, wall: !c.A });
      if (c.A && c.relVel < -BREAK_V * 0.45 && c.s0 > -1 && simT > settleUntil && simT - c.A.born > COOL && simT - c.B.born > COOL){
        const key = c.A.id * 1048576 + c.B.id;
        const ev = impactEvents.get(key);
        if (!ev || ev.v < -c.relVel) impactEvents.set(key, { A: c.A, B: c.B, v: -c.relVel, px: c.px, py: c.py, nx: c.nx, ny: c.ny, uA: c.uA, uB: c.uB });
      }
    }
    // one blow at a time: the strongest two impacts of a step may break something
    if (impactEvents.size){
      const evs = [...impactEvents.values()].sort((p, q) => q.v - p.v);
      for (let i = 0; i < Math.min(evs.length, 2); i++) impact(evs[i]);
    }
  }
  function breakSpeed(area){ return BREAK_V * Math.pow(area / BREAK_AREA, -0.1); }
  // a big slab beds down on the floor's grit and drags harder than a pebble that skates over it
  function groundDecel(area){ return GROUND_DECEL * (0.9 + 0.5 * clamp(Math.sqrt(area) / 90, 0, 1.3)); }

  // One throw into a pile sets off many piece-on-piece impacts within a few milliseconds. They share a
  // single budget of new pieces, set by the energy of the first, so a blow cracks what it hits instead of
  // chaining through the pile.
  const blows = [];
  function blowFor(A, B, x, y, E){
    for (const b of [A, B]) if (b.blow && simT - b.blow.t < BLOW_CARRY) return b.blow;
    for (let i = blows.length - 1; i >= 0; i--) if (simT - blows[i].t > BLOW_T) blows.splice(i, 1);
    for (const bl of blows) if ((x - bl.x) ** 2 + (y - bl.y) ** 2 < BLOW_R * BLOW_R) return bl;
    const bl = { x, y, t: simT, left: Math.floor(clamp(2 + E / BLOW_E0, 2, BLOW_MAX)) };
    blows.push(bl);
    return bl;
  }
  function impact(ev){
    const { A, B, v } = ev;
    if (A.dead || B.dead) return;
    const struck = ev.uA >= ev.uB ? B : A, thrower = struck === B ? A : B;
    const mu = A.mass * B.mass / (A.mass + B.mass);
    const E = 0.5 * mu * v * v;
    const bl = blowFor(A, B, ev.px, ev.py, E);
    A.blow = B.blow = bl;
    if (bl.left < 1) return;
    const vS = breakSpeed(struck.area);
    const made = [];
    let spent = 0;
    if (struck.grains.length >= MIN_SPLIT && v > vS){
      const dir = struck === B ? 1 : -1;
      const kids = fracture(struck, ev.px, ev.py, ev.nx * dir, ev.ny * dir, v / vS, v, bl.left);
      if (kids){ made.push(kids); bl.left -= kids.kids.length - 1; spent += kids.energy; }
    }
    // the thrower breaks too only when what the blow has left after cracking the struck piece is
    // enough for a piece its size
    const vLeft = Math.sqrt(Math.max(0, 2 * (E - spent) / mu));
    const vT = breakSpeed(thrower.area) * 1.3 * Math.sqrt(thrower.mass / (2 * mu));
    if (bl.left >= 1 && thrower.grains.length >= MIN_SPLIT && vLeft > vT * rand(0.9, 1.25)){
      const dir = thrower === B ? 1 : -1;
      const kids = fracture(thrower, ev.px, ev.py, ev.nx * dir, ev.ny * dir, vLeft / vT, vLeft, bl.left);
      if (kids){ made.push(kids); bl.left -= kids.kids.length - 1; spent += kids.energy; }
    }
    if (!made.length) return;
    for (const k of made) for (const ch of k.kids) ch.blow = bl;
    // the new cracks absorb energy: take it out of the motion relative to the pair's centre of mass,
    // which leaves momentum as it was, and the fragments separate on whatever is left
    const set = [];
    for (const k of made) set.push(...k.kids);
    if (made.length < 2) set.push(made[0].parent === struck ? thrower : struck);
    let M = 0, px = 0, py = 0;
    for (const b of set){ M += b.mass; px += b.mass * b.vx; py += b.mass * b.vy; }
    const cvx = px / M, cvy = py / M;
    let Er = 0;
    for (const b of set) Er += 0.5 * b.mass * ((b.vx - cvx) ** 2 + (b.vy - cvy) ** 2) + 0.5 * b.I * b.w * b.w;
    if (Er > 0){
      const f = Math.sqrt(Math.max(0.12, 1 - spent / Er));
      for (const b of set){ b.vx = cvx + (b.vx - cvx) * f; b.vy = cvy + (b.vy - cvy) * f; b.w *= f; }
    }
    snd.crunch(Math.min(v / BREAK_V, 3), mu);
  }

  // ---- fracture
  // Label grains by radial cracks from (cx, cy) (rock frame), plus a crushed zone at the centre.
  // Returns connected grain groups.
  function partition(list, cx, cy, dirX, dirY, nFrag, opt){
    const Gr = GR;
    const full = opt.full;
    const nInner = opt.inner;
    const S = Math.max(2, nFrag - nInner);
    const span = full ? TAU : TAU * 0.62;
    const start = full ? Math.random() * TAU : Math.atan2(dirY, dirX) - span / 2;
    const wts = []; let tot = 0;
    for (let k = 0; k < S; k++){ const w = rand(0.65, 1.35); wts.push(w); tot += w; }
    const cum = [0];
    for (let k = 0; k < S; k++) cum.push(cum[k] + wts[k] / tot * span);
    const wob = [];
    for (let k = 0; k <= S; k++) wob.push({ f: rand(0.02, 0.045), ph: rand(0, TAU), amp: rand(0.02, 0.08), seed: rand(0, 1000) });
    // the crushed zone: the tips of the wedges nearest the blow snap off along a straight chord across
    // each wedge, so every piece stays blocky (a ring round the blow would leave crescents)
    const tipR = [];
    const order = [];
    for (let k = 0; k < S; k++) order.push(k);
    const mid = span / 2;
    order.sort((p, q) => Math.abs((cum[p] + cum[p + 1]) / 2 - mid) - Math.abs((cum[q] + cum[q + 1]) / 2 - mid));
    for (let k = 0; k < S; k++) tipR.push(Infinity);
    for (let i = 0; i < Math.min(nInner, S); i++) tipR[order[i]] = opt.innerR * rand(0.8, 1.25);
    const tipTilt = [];
    for (let k = 0; k < S; k++) tipTilt.push(rand(-0.35, 0.35));
    const lab = new Map();
    for (const g of list){
      const vx = Gr.cx[g] - cx, vy = Gr.cy[g] - cy, d = Math.hypot(vx, vy);
      let th = Math.atan2(vy, vx) - start;
      th = ((th % TAU) + TAU) % TAU;
      const ramp = Math.min(d / 35, 1);
      if (full) th = (((th - ramp * wob[0].amp * Math.sin(d * wob[0].f + wob[0].ph)) % TAU) + TAU) % TAU;
      else if (th > span) th = (th - span < TAU - th) ? span - 1e-6 : 0;
      let k = 0;
      for (let b = 1; b < S; b++){
        const wb = wob[b];
        const bk = cum[b] + ramp * (wb.amp * Math.sin(d * wb.f + wb.ph) + 0.12 * (vn1(d * 0.09 + wb.seed) - 0.5));
        if (th > bk) k = b;
      }
      let L2 = k;
      if (tipR[k] < Infinity){
        const bis = (cum[k] + cum[k + 1]) / 2 + tipTilt[k];
        const proj = d * Math.cos(th - bis) + 1.2 * (vn1(d * 0.2 + k * 7) - 0.5);
        if (proj < tipR[k]) L2 = k + 100;
      }
      lab.set(g, L2);
    }
    const comp = new Map();
    const groups = [];
    for (const g of list){
      if (comp.has(g)) continue;
      const id = groups.length, L2 = lab.get(g), grp = [g];
      comp.set(g, id);
      for (let q = 0; q < grp.length; q++){
        for (const nb of Gr.adj[grp[q]]){
          if (!comp.has(nb) && lab.get(nb) === L2){ comp.set(nb, id); grp.push(nb); }
        }
      }
      groups.push(grp);
    }
    const areas = groups.map(grp => grp.reduce((s, g) => s + Gr.area[g], 0));
    const alive = groups.map(() => true);
    const mergeInto = i => {
      const counts = new Map();
      for (const g of groups[i]) for (const nb of Gr.adj[g]){
        const j = comp.get(nb);
        if (j !== undefined && j !== i && alive[j]) counts.set(j, (counts.get(j) || 0) + 1);
      }
      let best = -1, bc = 0;
      for (const [j, c] of counts) if (c > bc){ bc = c; best = j; }
      if (best < 0) return false;
      for (const g of groups[i]){ comp.set(g, best); groups[best].push(g); }
      areas[best] += areas[i]; alive[i] = false; groups[i] = [];
      return true;
    };
    for (let guard = 0; guard < 400; guard++){
      let worst = -1, wa = Infinity, live = 0;
      for (let i = 0; i < groups.length; i++) if (alive[i]){ live++; if (areas[i] < wa){ wa = areas[i]; worst = i; } }
      if (live <= 1) break;
      if (wa >= opt.minArea && live <= nFrag + 2) break;
      if (!mergeInto(worst)) break;
    }
    return blockyGroups(groups.filter((g, i) => alive[i] && g.length));
  }
  // Broken slabs come apart in blocky pieces. A group that would make a crescent, hook or sliver is
  // folded into whichever neighbour leaves the most compact result.
  function blockyGroups(groups){
    const Gr = GR;
    if (groups.length < 2) return groups;
    const comp = new Map();
    groups.forEach((grp, i) => { for (const g of grp) comp.set(g, i); });
    const info = groups.map(grp => {
      const xs = [], ys = [];
      let area = 0, per = 0;
      for (const g of grp){
        area += Gr.area[g];
        const o = Gr.off[g], m = Gr.cnt[g];
        for (let k = o; k < o + m; k++){
          const L2 = Gr.lab[k];
          if (L2 < 0 || comp.get(L2) !== comp.get(g)){
            const k2 = k + 1 < o + m ? k + 1 : o;
            per += Math.hypot(Gr.vx[k2] - Gr.vx[k], Gr.vy[k2] - Gr.vy[k]);
            xs.push(Gr.vx[k]); ys.push(Gr.vy[k]);
          }
        }
      }
      const hull = hullOf(xs, ys);
      return { area, per, hull, ...hullStats(hull.x, hull.y) };
    });
    const alive = groups.map(() => true);
    const gravel = Gr.s * Gr.s * 3;
    const badness = i => {
      const f = info[i];
      if (f.area < gravel) return 0;
      const sol = f.area / Math.max(f.hullArea || f.area, 1e-6);
      const elong = f.area / Math.max(f.diam2, 1e-6);
      const width = 2 * f.area / Math.max(f.per, 1e-6);
      return Math.max(0, 0.76 - sol) + Math.max(0, 0.12 - elong) * 3 + Math.max(0, 3.2 - width) * 0.2;
    };
    info.forEach(f => { f.hullArea = f.area > 0 ? hullStats(f.hull.x, f.hull.y).area : 0; });
    for (let guard = 0; guard < 40; guard++){
      let worst = -1, wb = 0, live = 0;
      for (let i = 0; i < groups.length; i++) if (alive[i]){ live++; const b = badness(i); if (b > wb){ wb = b; worst = i; } }
      if (worst < 0 || live <= 2) break;
      const shared = new Map();
      for (const g of groups[worst]) for (const nb of Gr.adj[g]){ const j = comp.get(nb); if (j !== undefined && j !== worst && alive[j]) shared.set(j, (shared.get(j) || 0) + 1); }
      let best = -1, bs = -Infinity;
      for (const [j, cnt] of shared){
        const hx = info[worst].hull.x.concat(info[j].hull.x), hy = info[worst].hull.y.concat(info[j].hull.y);
        const hh = hullOf(hx, hy), st = hullStats(hh.x, hh.y);
        const sol = (info[worst].area + info[j].area) / Math.max(st.area, 1e-6);
        const score = sol + 0.02 * Math.min(cnt, 10);
        if (score > bs){ bs = score; best = j; }
      }
      if (best < 0){ info[worst].area = 0; continue; }
      const sl = sharedLen(groups[worst], best, comp);
      for (const g of groups[worst]){ comp.set(g, best); groups[best].push(g); }
      const hx = info[worst].hull.x.concat(info[best].hull.x), hy = info[worst].hull.y.concat(info[best].hull.y);
      const hh = hullOf(hx, hy), st = hullStats(hh.x, hh.y);
      info[best] = { area: info[best].area + info[worst].area, per: info[best].per + info[worst].per - 2 * sl, hull: hh, diam2: st.diam2, hullArea: st.area };
      alive[worst] = false; groups[worst] = [];
    }
    return groups.filter((g, i) => alive[i] && g.length);
  }
  function sharedLen(grp, j, comp){
    const Gr = GR;
    let L = 0;
    for (const g of grp){
      const o = Gr.off[g], m = Gr.cnt[g];
      for (let k = o; k < o + m; k++){
        const L2 = Gr.lab[k];
        if (L2 >= 0 && comp.get(L2) === j){ const k2 = k + 1 < o + m ? k + 1 : o; L += Math.hypot(Gr.vx[k2] - Gr.vx[k], Gr.vy[k2] - Gr.vy[k]); }
      }
    }
    return L;
  }
  // Fold groups smaller than minArea (and then the smallest, while there are more than maxN) into the
  // neighbour they share the most boundary with.
  function mergeGroups(groups, comp, minArea, maxN){
    const Gr = GR;
    const areas = groups.map(grp => grp.reduce((s, g) => s + Gr.area[g], 0));
    const alive = groups.map(() => true), stuck = groups.map(() => false);
    for (let guard = 0; guard < 400; guard++){
      let worst = -1, wa = Infinity, live = 0;
      for (let i = 0; i < groups.length; i++) if (alive[i]){ live++; if (!stuck[i] && areas[i] < wa){ wa = areas[i]; worst = i; } }
      if (worst < 0 || live <= 1 || (wa >= minArea && live <= maxN)) break;
      const counts = new Map();
      for (const g of groups[worst]) for (const nb of Gr.adj[g]){
        const j = comp.get(nb);
        if (j !== undefined && j !== worst && alive[j]) counts.set(j, (counts.get(j) || 0) + 1);
      }
      let best = -1, bc = 0;
      for (const [j, c] of counts) if (c > bc){ bc = c; best = j; }
      if (best < 0){ stuck[worst] = true; continue; }
      alive[worst] = false;
      for (const g of groups[worst]){ comp.set(g, best); groups[best].push(g); }
      areas[best] += areas[worst]; groups[worst] = [];
    }
    return groups.filter((g, i) => alive[i] && g.length);
  }

  function fracture(b, pxw, pyw, nxw, nyw, severity, speed, maxNew = BLOW_MAX){
    const Gr = GR;
    const [cx, cy] = toRock(b, pxw, pyw);
    const c = Math.cos(b.a), s = Math.sin(b.a);
    const dx = c * nxw + s * nyw, dy = -s * nxw + c * nyw;
    const nG = b.grains.length;
    const extra = Math.floor((severity - 1) * 2 + Math.random() * 1.3) + (b.area > 6000 ? 2 : b.area > 2500 ? 1 : 0);
    // every fragment needs a row in the outline texture
    const room = freeRows.length + 1;
    const nFrag = clamp(2 + extra, 2, Math.min(7, Math.floor(nG / 1.5), maxNew + 1, room));
    if (nFrag < 2) return null;
    const inner = nFrag >= 5 ? 2 : nFrag >= 3 ? 1 : 0;
    const opt = { full: false, inner, innerR: Math.max(b.R * 0.24, Gr.s * 1.6), minArea: Gr.s * Gr.s * 0.55 };
    let groups = partition(b.grains, cx, cy, dx, dy, nFrag, opt);
    // a blow landing in a notch can leave every crack on one side; crack the piece right round instead
    for (let tries = 0; groups.length < 2 && tries < 3; tries++) groups = partition(b.grains, cx, cy, dx, dy, nFrag, { ...opt, full: true });
    if (groups.length < 2) return null;
    const cap = Math.min(maxNew + 1, room);
    if (groups.length > cap){
      const comp = new Map();
      groups.forEach((grp, i) => { for (const g of grp) comp.set(g, i); });
      groups = mergeGroups(groups, comp, 0, cap);
      if (groups.length < 2) return null;
    }
    const T = b.T, oldFresh = b.freshLen;
    // fragments ease apart rather than fly: well under any breaking speed
    const kids = splitBody(b, groups, pxw, pyw, Math.min(speed * 0.07, 90) + 20);
    spawnChips(pxw, pyw, Math.round(clamp(2 + severity * 2.5, 2, 8)), 180 + speed * 0.2, cx, cy);
    spawnDust(pxw, pyw, clamp(0.5 + severity * 0.4, 0.5, 1.6));
    fractureCount++;
    // length of the new cracks: each is shared by two fragments, and the parent's old breaks are
    // inherited by its fragments
    let crack = -oldFresh;
    for (const k of kids) crack += k.freshLen;
    crack = Math.max(crack / 2, 0);
    logArea('fracture');
    return { kids, crack, energy: CRACK_G * T * crack, parent: b };
  }

  // Fragments inherit the parent's rigid motion. On top of that they spread: an expansion about the
  // parent's centre of mass (no net linear or angular momentum, and every pair of fragments moves
  // apart so jagged teeth don't lock them together). A click on the intact slab instead spreads
  // everything from the click.
  function splitBody(p, groups, pxw, pyw, kick, fromClick){
    const c = Math.cos(p.a), s = Math.sin(p.a);
    const kids = groups.map(gs => makeBody(gs, true));
    for (const ch of kids) buildOutline(ch);
    const fam = nextId++;
    for (const ch of kids){ ch.family = fam; ch.ghost = simT + 0.35; }
    const kv = [];
    for (const ch of kids){
      const ox = ch.hx - p.hx, oy = ch.hy - p.hy;
      ch.x = p.x + c * ox - s * oy; ch.y = p.y + s * ox + c * oy; ch.a = p.a;
      const rx = ch.x - p.x, ry = ch.y - p.y;
      ch.vx = p.vx - p.w * ry; ch.vy = p.vy + p.w * rx; ch.w = p.w;
      const ddx = ch.x - pxw, ddy = ch.y - pyw, d = Math.hypot(ddx, ddy) || 1;
      if (fromClick){
        const sp = (kick * 0.3 + kick * 0.9 * d / p.R) * rand(0.85, 1.15);
        kv.push(ddx / d * sp, ddy / d * sp, sp);
        continue;
      }
      kv.push(0, 0, kick);
    }
    const kExp = fromClick ? 0 : kick / (0.5 * p.R + 6);
    kids.forEach((ch, i) => {
      const e = kExp * rand(0.85, 1.15);
      ch.vx += kv[i * 3] + e * (ch.x - p.x); ch.vy += kv[i * 3 + 1] + e * (ch.y - p.y);
      ch.w += rand(-1, 1) * kv[i * 3 + 2] / (ch.R + 8) * 0.2;
    });
    // anyone holding the old piece now holds whichever fragment is under their finger
    for (const gr of grabs.values()){
      if (gr.body !== p) continue;
      const rx = gr.lx + p.hx, ry = gr.ly + p.hy;
      let host = null, bd = Infinity;
      for (const ch of kids){
        for (const g of ch.grains){
          if (grainHas(g, rx, ry)){ host = ch; break; }
          const d = Math.hypot(GR.cx[g] - rx, GR.cy[g] - ry);
          if (d < bd){ bd = d; if (!host) host = ch; }
        }
        if (host && grainAt(rx, ry, host.id) >= 0) break;
      }
      gr.body = host; gr.lx = rx - host.hx; gr.ly = ry - host.hy; gr.ix = gr.iy = 0;
    }
    const i = bodies.indexOf(p);
    bodies.splice(i, 1, ...kids);
    freeBody(p);
    dirty = true;
    return kids;
  }

  // ---- particles
  const chips = [];
  const dust = [];
  let tickClock = 0;
  function floorAt(x, y){
    for (const b of bodies) if (!b.kin && bodyContains(b, x, y)) return b.T + b.z;
    return 0;
  }
  function spawnChips(x, y, n, speed, rx, ry){
    for (let i = 0; i < n; i++){
      const a = rand(0, TAU), sp = speed * rand(0.25, 1);
      chips.push({ x, y, z: rock.T * rand(0.5, 1.05), vx: Math.cos(a) * sp, vy: Math.sin(a) * sp, vz: rand(120, 480),
        a: rand(0, TAU), w: rand(-25, 25), size: rand(1.1, 2.6), seed: Math.random(), rest: false, id: nextId++,
        ox: rx + rand(-4, 4), oy: ry + rand(-4, 4), still: 0, life: rand(1.5, 3) });
    }
    dirty = true;
  }
  function spawnDust(x, y, amt){
    if (reduceMotion.matches) return;
    // pale powder over a dark page reads as a smudge long before it reads as dust: there it thins out sooner
    const dustLife = 0.55 + 0.45 * clamp(pageLum * 3, 0, 1);
    const n = Math.round(12 * amt);
    for (let i = 0; i < n; i++){
      const a = rand(0, TAU), sp = rand(20, 120) * amt;
      dust.push({ x: x + rand(-4, 4), y: y + rand(-4, 4), vx: Math.cos(a) * sp, vy: Math.sin(a) * sp - 10, r: rand(4, 10) * Math.sqrt(amt),
        dr: rand(14, 36), life: rand(0.8, 1.8) * dustLife, age: 0, a0: rand(0.12, 0.24) });
    }
    while (dust.length > 300) dust.shift();
    dirty = true;
  }
  function updateParticles(dt){
    const [cL, cT, cR, cB] = wallsFor(3);
    for (const c of chips){
      if (c.fade !== undefined){ const r = c.fadeRate || 2.5; c.fade -= dt * r; c.size *= Math.exp(-dt * r * 1.2); continue; }
      if (c.rest){
        c.still += dt;
        if (c.still > c.life){ c.fade = 1; c.fadeRate = 0.8; }
        continue;
      }
      c.vz -= 2600 * dt;
      c.x += c.vx * dt; c.y += c.vy * dt; c.z += c.vz * dt; c.a += c.w * dt;
      if (c.x < cL + 2){ c.x = cL + 2; c.vx = Math.abs(c.vx) * 0.4; } if (c.x > cR - 2){ c.x = cR - 2; c.vx = -Math.abs(c.vx) * 0.4; }
      if (c.y < cT + 2){ c.y = cT + 2; c.vy = Math.abs(c.vy) * 0.4; } if (c.y > cB - 2){ c.y = cB - 2; c.vy = -Math.abs(c.vy) * 0.4; }
      const fl = floorAt(c.x, c.y);
      if (c.z <= fl){
        c.z = fl;
        if (c.vz < -90){
          c.vz = -c.vz * 0.28; c.vx *= 0.6; c.vy *= 0.6; c.w *= 0.5;
          if (tickClock <= 0){ snd.tick(Math.min(-c.vz / 400, 1) * 0.5); tickClock = 0.02; }
        } else {
          c.vz = 0;
          const sp = Math.hypot(c.vx, c.vy), dv = 1800 * dt;
          if (sp <= dv){ c.vx = c.vy = 0; c.w = 0; c.rest = true; }
          else { c.vx *= 1 - dv / sp; c.vy *= 1 - dv / sp; c.w *= 0.9; }
        }
      }
    }
    for (let i = chips.length - 1; i >= 0; i--) if (chips[i].fade !== undefined && chips[i].fade <= 0) chips.splice(i, 1);
    let live = 0;
    for (let i = chips.length - 1; i >= 0; i--){
      const c = chips[i];
      if (c.fade !== undefined) continue;
      if (++live > MAX_CHIPS){ c.fade = 1; c.fadeRate = 3; }
    }
    // a piece that slides under or over a resting chip carries it along or pushes it aside
    for (const c of chips){
      if (!c.rest || c.fade !== undefined) continue;
      for (const b of bodies){
        if (b.kin || (b.vx === 0 && b.vy === 0 && b.w === 0)) continue;
        if (bodyContains(b, c.x, c.y)){ c.rest = false; c.still = 0; c.vx = b.vx * 1.1 + rand(-40, 40); c.vy = b.vy * 1.1 + rand(-40, 40); c.vz = 60; c.z = Math.min(c.z, b.T * 0.3); break; }
      }
    }
    for (const d of dust){
      d.age += dt;
      const f = Math.exp(-dt * 2.2);
      d.vx *= f; d.vy *= f;
      d.x += d.vx * dt; d.y += d.vy * dt; d.r += d.dr * dt;
    }
    for (let i = dust.length - 1; i >= 0; i--) if (dust[i].age >= dust[i].life) dust.splice(i, 1);
    tickClock -= dt;
  }

  // ---- phases
  let phase = 'intact';        // intact | broken | reassembling
  let lastInput = 0, glideT = 0, sealT = 0, dirty = true;
  const areaLog = [];
  function fragmentArea(){ let a = 0; for (const b of bodies) a += b.area; return a; }
  function logArea(why){
    if (areaLog.length > 400) areaLog.shift();
    areaLog.push({ t: +simT.toFixed(3), why, pieces: bodies.length, area: fragmentArea(), original: totalArea0 });
  }

  function intactBody(){
    const all = [];
    for (let g = 0; g < GR.n; g++) all.push(g);
    const b = makeBody(all);
    b.kin = true; b.home = true; b.seam = 0; b.T = rock.T;
    b.x = pill.cx + b.hx; b.y = pill.cy + b.hy;
    return b;
  }
  function buildRock(){
    for (const b of bodies) freeBody(b);
    bodies = [];
    grabs.clear();
    rock.w = pill.w; rock.h = pill.h; rock.r = pill.r;
    rock.T = clamp(pill.h * 0.3, 8, 20);
    buildGrains(rock.w, rock.h, rock.r);
    buildText();
    bakeStone();
    bodies.push(intactBody());
    chips.length = 0;
    phase = 'intact';
    enablePointer();
    dirty = true;
  }

  // A slab struck from above splits along a few long through-cracks rather than shattering like glass:
  // one crack runs right across along the stone's rift, two or three more run out from the blow, and
  // a branch or two leave the long crack further out. Each crack wanders at low frequency and jags at
  // the grain scale; one that meets another crack stops there. Returns the grain groups between them.
  function crackGroups(list, cx, cy){
    const Gr = GR;
    const hw = rock.w / 2, hh = rock.h / 2;
    const sdf = (x, y) => sdRound(x, y, hw, hh, rock.r);
    const segs = [];
    const CELL = 12, grid = new Map();
    const cellKey = (i, j) => i * 4096 + j;
    const addSeg = sg => {
      segs.push(sg);
      const i0 = Math.floor(Math.min(sg[0], sg[2]) / CELL), i1 = Math.floor(Math.max(sg[0], sg[2]) / CELL);
      const j0 = Math.floor(Math.min(sg[1], sg[3]) / CELL), j1 = Math.floor(Math.max(sg[1], sg[3]) / CELL);
      for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++){
        const k = cellKey(i + 2048, j + 2048);
        let a = grid.get(k); if (!a){ a = []; grid.set(k, a); } a.push(sg);
      }
    };
    const near = (x0, y0, x1, y1, fn) => {
      const i0 = Math.floor(Math.min(x0, x1) / CELL), i1 = Math.floor(Math.max(x0, x1) / CELL);
      const j0 = Math.floor(Math.min(y0, y1) / CELL), j1 = Math.floor(Math.max(y0, y1) / CELL);
      for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++){
        const a = grid.get(cellKey(i + 2048, j + 2048));
        if (a) for (const sg of a) fn(sg);
      }
    };
    // Where a crack crosses the label it takes the line through the least ink (a word gap, or the
    // widest gap between letters), so it doesn't leave a stub of a letter on the far piece.
    const lb = labelInk && labelInk.box;
    const inLabel = (x, y) => lb && x > lb[0] && x < lb[2] && y > lb[1] && y < lb[3];
    const inkNear = (x, y) => {
      const L = labelInk;
      let m = 0;
      for (let q = -1; q <= 1; q++) for (let r = -1; r <= 1; r++){
        const i = Math.floor((x + q * 1.5 + rock.w / 2) * L.k), j = Math.floor((y + r * 1.5 + rock.h / 2) * L.k);
        if (i >= 0 && j >= 0 && i < L.cw && j < L.ch && L.d[(j * L.cw + i) * 2] > 0) m++;
      }
      return m;
    };
    const chooseLane = (x, y, a) => {
      const dx = Math.cos(a), dy = Math.sin(a), nx = -dy, ny = dx;
      let best = 0, bc = Infinity;
      for (let o = -8; o <= 8; o += 0.5){
        let c = 0.06 * Math.abs(o);
        for (let t = 0; t < 60; t += 1){
          const px = x + nx * o + dx * t, py = y + ny * o + dy * t;
          if (t > 4 && !inLabel(px, py)) break;
          c += inkNear(px, py);
        }
        if (c < bc){ bc = c; best = o; }
      }
      return { x: x + nx * best, y: y + ny * best, dx, dy, nx, ny };
    };
    // A crack is walked out first and only kept if it is fit to keep: one running along the line of
    // text, opened by a few px, reads as the label printed twice.
    const walk = (x, y, ang, id, minLen) => {
      const f = rand(0.018, 0.04), ph = rand(0, TAU), amp = rand(0.05, 0.14), sd = rand(0, 1000);
      const step = 3, out = [];
      let len = 0, lane = null, inX = 0, inY = 0, inLen = 0;
      for (let guard = 0; guard < 500; guard++){
        let a = ang + amp * Math.sin(len * f + ph) + 0.55 * (vn1(len * 0.16 + sd) - 0.5);
        const ahead = inLabel(x + Math.cos(a) * step * 2, y + Math.sin(a) * step * 2) || inLabel(x, y);
        if (ahead){
          if (!lane) lane = chooseLane(x, y, a);
          const e = (x - lane.x) * lane.nx + (y - lane.y) * lane.ny;
          a = Math.atan2(lane.dy, lane.dx) - clamp(e * 0.45, -0.6, 0.6);
        } else if (lane && !inLabel(x, y)) lane = null;
        const nx = x + Math.cos(a) * step, ny = y + Math.sin(a) * step;
        let hit = 2;
        if (len >= minLen) near(x, y, nx, ny, sg => { if (sg[4] !== id){ const tt = segHit(x, y, nx, ny, sg[0], sg[1], sg[2], sg[3]); if (tt >= 0 && tt < hit) hit = tt; } });
        const ex = hit <= 1 ? x + (nx - x) * hit : nx, ey = hit <= 1 ? y + (ny - y) * hit : ny;
        out.push([x, y, ex, ey, id]);
        if (inLabel((x + ex) / 2, (y + ey) / 2)){ inX += ex - x; inY += ey - y; inLen += Math.hypot(ex - x, ey - y); }
        if (hit <= 1 || sdf(nx, ny) > 2) break;
        x = nx; y = ny; len += step;
      }
      // how close the crack runs to the text's baseline where it crosses the label, in radians under 35°
      const slope = inLen > 5 ? Math.atan2(Math.abs(inY), Math.abs(inX)) : Math.PI / 2;
      return { segs: out, len, bad: Math.max(0, 0.61 - slope) };
    };
    const commit = r => { for (const sg of r.segs) addSeg(sg); return r.len; };
    // tries(t) gives the t-th try's pair of directions (one, or two walked out of the same point)
    const crack = (x, y, tries, id, minLen, required) => {
      let best = null;
      for (let t = 0; t < 6; t++){
        const rs = tries(t).map(a => walk(x, y, a, id, minLen));
        const bad = rs.reduce((m, r) => m + r.bad, 0);
        if (!best || bad < best.bad) best = { rs, bad };
        if (bad === 0) break;
      }
      if (best.bad > 0 && !required) return;
      for (const r of best.rs) commit(r);
    };
    // a long slab tends to snap across its short dimension
    const across = rock.w > rock.h * 1.6;
    const rift = across ? Math.PI / 2 + rand(-0.5, 0.5) : rand(0, Math.PI);
    let id = 0;
    const main = id++;
    crack(cx, cy, t => { const r = t ? (across ? Math.PI / 2 + rand(-0.5, 0.5) : rand(0, Math.PI)) : rift; return [r + rand(-0.12, 0.12), r + Math.PI + rand(-0.12, 0.12)]; }, main, 1e9, true);
    const mainSegs = segs.slice();
    const nr = Math.random() < 0.55 ? 2 : 3;
    const side0 = Math.random() < 0.5 ? 1 : -1;
    for (let k = 0; k < nr; k++){
      const side = k % 2 ? -side0 : side0;
      crack(cx, cy, () => {
        const dev = across ? (Math.random() < 0.5 ? 1 : -1) * rand(0.5, 1.0) : rand(-0.65, 0.65);
        return [rift + side * (Math.PI / 2 + dev)];
      }, id++, 7, false);
    }
    const inset = Math.min(12, hh * 0.4);
    const branch = () => {
      const far = mainSegs.filter(sg => Math.hypot(sg[0] - cx, sg[1] - cy) > 30 && sdf(sg[0], sg[1]) < -inset);
      if (!far.length) return;
      const sg = far[Math.floor(Math.random() * far.length)];
      const dir = Math.atan2(sg[3] - sg[1], sg[2] - sg[0]);
      crack(sg[0], sg[1], () => [dir + (Math.random() < 0.5 ? 1 : -1) * (across ? rand(0.5, 1.0) : rand(0.8, 1.4))], id++, 7, false);
    };
    const nb = Math.random() < 0.65 ? 2 : 3;
    for (let k = 0; k < nb; k++) branch();

    const minArea = Gr.s * Gr.s * 9;
    const want = clamp(Math.round(totalArea0 / 1400), 5, 10);
    let groups = null;
    for (let attempt = 0; attempt < 7; attempt++){
      const cut = (g, n) => {
        const ax = Gr.cx[g], ay = Gr.cy[g], bx = Gr.cx[n], by = Gr.cy[n];
        let hitAny = false;
        near(ax, ay, bx, by, sg => { if (!hitAny && segHit(ax, ay, bx, by, sg[0], sg[1], sg[2], sg[3]) >= 0) hitAny = true; });
        return hitAny;
      };
      const comp = new Map();
      groups = [];
      for (const g of list){
        if (comp.has(g)) continue;
        const gi = groups.length, grp = [g];
        comp.set(g, gi);
        for (let q = 0; q < grp.length; q++){
          for (const n of Gr.adj[grp[q]]) if (!comp.has(n) && own[n] === own[g] && !cut(grp[q], n)){ comp.set(n, gi); grp.push(n); }
        }
        groups.push(grp);
      }
      groups = blockyGroups(mergeGroups(groups, comp, minArea, 10));
      let big = groups[0], bigA = 0, tot = 0;
      for (const grp of groups){ let a = 0; for (const g of grp) a += Gr.area[g]; tot += a; if (a > bigA){ bigA = a; big = grp; } }
      if (groups.length >= want && bigA < tot * 0.3) break;
      if (groups.length >= 10) break;
      // split the biggest chunk with a crack through it that runs out to its neighbours
      let mx = 0, my = 0;
      for (const g of big){ mx += Gr.cx[g] * Gr.area[g]; my += Gr.cy[g] * Gr.area[g]; }
      mx /= bigA; my /= bigA;
      let sx = Gr.cx[big[0]], sy = Gr.cy[big[0]], sdd = Infinity;
      for (const g of big){ const d = (Gr.cx[g] - mx) ** 2 + (Gr.cy[g] - my) ** 2; if (d < sdd){ sdd = d; sx = Gr.cx[g]; sy = Gr.cy[g]; } }
      const cid = id++;
      crack(sx, sy, () => { const a = (across ? Math.PI / 2 : rift + Math.PI / 2) + rand(-0.7, 0.7); return [a, a + Math.PI]; }, cid, 0, true);
    }
    return groups;
  }

  function breakIntact(x, y){
    if (phase !== 'intact' || !GR) return;
    const b = bodies[0];
    const [cx, cy] = toRock(b, x, y);
    const groups = crackGroups(b.grains, cx, cy);
    const calm = reduceMotion.matches;
    const kids = splitBody(b, groups, x, y, calm ? 45 : 230, true);
    for (const k of kids){ k.kin = false; k.born = simT; }
    phase = 'broken';
    lastInput = simT;
    snd.crack(1.1);
    spawnChips(x, y, calm ? 3 : 10, 260, cx, cy);
    spawnDust(x, y, 1.4);
    fractureCount++;
    areaLog.length = 0;
    logArea('break');
    wake();
  }

  function startReassembly(){
    if (phase !== 'broken') return;
    releaseAllGrabs();
    phase = 'reassembling';
    glideT = 0; sealT = 0;
    for (const b of bodies){
      const tx = pill.cx + b.hx, ty = pill.cy + b.hy;
      const a0 = ((b.a % TAU) + TAU + Math.PI) % TAU - Math.PI;
      const dist = Math.hypot(b.x - tx, b.y - ty) + Math.abs(a0) * b.R;
      b.kin = true; b.vx = b.vy = b.w = 0;
      const quick = reduceMotion.matches;
      b.glide = { x0: b.x, y0: b.y, a0, tx, ty, T0: b.T, delay: quick ? 0 : rand(0, 0.25) + (dist < 1 ? 0 : 0.05), dur: quick ? 0.35 : 0.6 + Math.min(dist / 900, 0.5),
        lift: dist < 1 || quick ? 0 : 0.6 + 1.4 * Math.random(), landed: false };
    }
    // A piece's smoothed crack line is pinned where its neighbour changes, so an outline drawn before
    // that neighbour was broken again no longer matches the neighbour's; redraw them all against the
    // final set of pieces so every crack is the identical curve from both sides.
    for (const b of bodies) buildOutline(b);
    for (const c of chips) if (c.fade === undefined){ c.fade = 1; c.fadeRate = 2.5; }
    logArea('home');
    snd.glide();
    dirty = true;
    wake();
  }

  // Every fragment flies to the exact pose it had in the intact slab (its home is its centroid in the
  // slab's own frame, which never changes however many times it was re-broken), grows back to the
  // slab's full thickness on the way, and once all of them are down the seams close. By then the
  // fragments render pixel-for-pixel as the intact slab, so swapping in the single body is invisible.
  function updateGlide(dt){
    glideT += dt;
    let all = true, landed = 0;
    for (const b of bodies){
      const g = b.glide;
      const u = clamp((glideT - g.delay) / g.dur, 0, 1);
      const e = u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2;
      if (u >= 1){ b.x = g.tx; b.y = g.ty; b.a = 0; b.z = 0; b.T = rock.T; }
      else {
        b.x = g.x0 + (g.tx - g.x0) * e; b.y = g.y0 + (g.ty - g.y0) * e; b.a = g.a0 * (1 - e);
        b.z = g.lift * Math.sin(Math.PI * u);
        b.T = g.T0 + (rock.T - g.T0) * e;
      }
      if (u >= 1 && !g.landed){ g.landed = true; b.home = true; landed++; }
      if (u < 1) all = false;
    }
    if (landed && performance.now() - lastKnock > 25){ snd.settle(); lastKnock = performance.now(); }
    if (all){
      sealT += dt / 0.45;
      for (const b of bodies) b.seam = Math.max(0, 1 - sealT);
      if (sealT >= 1.15) finishReassembly();
    }
    dirty = true;
  }
  // A piece caught on its way home: everything drops back to loose pieces where they are now, since a
  // slab half-rebuilt around a piece the user is holding could never finish.
  function cancelReassembly(){
    if (phase !== 'reassembling') return;
    phase = 'broken';
    for (const b of bodies){
      b.kin = false; b.glide = null; b.home = false; b.seam = 1;
      b.vx = b.vy = b.w = 0; b.rest = 0; b.wv = false;
    }
    relaxPile(4);
    lastInput = simT;
    dirty = true;
  }
  function finishReassembly(){
    for (const b of bodies) freeBody(b);
    bodies = [];
    if (Math.abs(pill.w - rock.w) > 0.5 || Math.abs(pill.h - rock.h) > 0.5 || Math.abs(pill.r - rock.r) > 0.5) buildRock();
    else bodies.push(intactBody());
    chips.length = 0;
    phase = 'intact';
    enablePointer();
    snd.seal();
    dirty = true;
  }

  // ---- input
  function grab(id, x, y){
    if (phase !== 'broken' && phase !== 'reassembling') return false;
    const b = bodyAt(x, y, 10);
    if (!b) return false;
    cancelReassembly();
    const c = Math.cos(b.a), s = Math.sin(b.a), dx = x - b.x, dy = y - b.y;
    let lx = c * dx + s * dy, ly = -s * dx + c * dy;
    if (!bodyContains(b, x, y)){ lx = 0; ly = 0; }
    grabs.set(id, { id, body: b, lx, ly, tx: x, ty: y, ix: 0, iy: 0, px: x, py: y, pt: performance.now(), pvx: 0, pvy: 0, x0: x, y0: y, moved: false });
    b.rest = 0;
    lastInput = simT;
    dirty = true;
    return true;
  }
  function moveGrab(id, x, y){
    const gr = grabs.get(id); if (!gr) return;
    const t = performance.now();
    const dt = Math.max((t - gr.pt) / 1000, 1e-3);
    const vx = (x - gr.px) / dt, vy = (y - gr.py) / dt;
    const k = Math.min(1, dt / 0.04);
    gr.pvx += (vx - gr.pvx) * k; gr.pvy += (vy - gr.pvy) * k;
    gr.px = x; gr.py = y; gr.pt = t;
    const [L, Tp, R, B] = wallsFor(gr.body.T);
    gr.tx = clamp(x, L, R); gr.ty = clamp(y, Tp, B);
    if (Math.hypot(x - gr.x0, y - gr.y0) > 2) gr.moved = true;
    gr.body.rest = 0;
    lastInput = simT;
  }
  function releaseGrab(id, silentRelease){
    const gr = grabs.get(id); if (!gr) return;
    grabs.delete(id);
    const b = gr.body;
    if (!silentRelease && !b.dead && performance.now() - gr.pt < 60){
      // hand the stone the pointer's last velocity at the grab point, so a flick throws it
      const [rx, ry] = grabPoint(gr);
      const ax = b.vx - b.w * ry, ay = b.vy + b.w * rx;
      b.vx += (gr.pvx - ax) * 0.85; b.vy += (gr.pvy - ay) * 0.85;
    }
    lastInput = simT;
    if (!grabs.size) setCursor(null);
  }
  function releaseAllGrabs(){ for (const id of [...grabs.keys()]) releaseGrab(id, true); }

  let pressed = null;                // the pointer pressing the intact slab
  let suppressUntil = 0;             // a drag started on the slab is not a click
  let breakClickUntil = 0;           // the click that belongs to a pointer break is let through as the button's click
  let swallowUntil = 0;              // grabbing or throwing a piece is not a click on whatever is under it
  const toStage = (cx, cy) => { const r = btnRect || button.getBoundingClientRect(); return [cx - r.left + G.offX, cy - r.top + G.offY]; };
  const inStage = (x, y) => x >= 0 && y >= 0 && x <= G.stageW && y <= G.stageH;

  function onBtnDown(e){
    if (lost || e.button !== 0 || !e.isPrimary || pressed || phase !== 'intact' || button.disabled) return;
    audio();
    const [x, y] = toStage(e.clientX, e.clientY);
    pressed = { id: e.pointerId, x0: x, y0: y, x, y, dragging: false };
  }
  function onBtnMove(e){
    if (!pressed || e.pointerId !== pressed.id) return;
    const [x, y] = toStage(e.clientX, e.clientY);
    pressed.x = x; pressed.y = y;
    if (!pressed.dragging && Math.hypot(x - pressed.x0, y - pressed.y0) > DRAG_START) pressed.dragging = true;
  }
  function onBtnUp(e, cancel){
    if (!pressed || e.pointerId !== pressed.id) return;
    const p = pressed; pressed = null;
    if (!cancel && p.dragging){ suppressUntil = performance.now() + 800; return; }
    if (cancel || phase !== 'intact' || lost) return;
    const [x, y] = toStage(e.clientX, e.clientY);
    const lx = x - pill.cx, ly = y - pill.cy;
    if (sdRound(lx, ly, pill.w / 2, pill.h / 2, pill.r) > 1) return;
    breakClickUntil = performance.now() + 1000;
    breakIntact(x, y);
    // on touch the click can arrive a little after the release; the button must still be hit-testable then
    clearTimeout(peTimer); peTimer = setTimeout(disablePointer, 700);
  }
  const onBtnCancel = e => onBtnUp(e, true);
  function onClick(e){
    const now = performance.now();
    if (e.detail > 0 && now < suppressUntil){ suppressUntil = 0; e.preventDefault(); e.stopImmediatePropagation(); return; }
    if (lost || !hidden) return;
    // the click that belongs to a pointer break goes through as the button's click
    if (now < breakClickUntil){ breakClickUntil = 0; disablePointer(); return; }
    if (phase === 'intact'){
      // keyboard, form submission, or button.click()
      audio();
      breakIntact(pill.cx + rand(-0.12, 0.12) * pill.w, pill.cy + rand(-0.15, 0.15) * pill.h);
      disablePointer();
      return;
    }
    // the button isn't there while it lies in pieces: activating it again only brings it back
    e.preventDefault(); e.stopImmediatePropagation();
    if (e.detail === 0 && phase === 'broken'){ lastInput = simT; startReassembly(); }
  }

  // pieces: routed from the document, since the canvas never takes pointer events
  const root = document.documentElement;
  const baseCursor = root.style.cursor;
  let cursor = null;
  function setCursor(c){ if (c === cursor) return; cursor = c; root.style.cursor = c || baseCursor; }
  function docDown(e){
    if (lost || e.button !== 0 || (phase !== 'broken' && phase !== 'reassembling')) return;
    const [x, y] = toStage(e.clientX, e.clientY);
    if (!inStage(x, y)) return;
    if (!grab(e.pointerId, x, y)){
      // a press that only just misses a piece (gravel skitters) must not start selecting the page's text
      if (nearBody(x, y, 8)) e.preventDefault();
      return;
    }
    audio();
    try { root.setPointerCapture(e.pointerId); } catch {}
    setCursor('grabbing');
    e.preventDefault(); e.stopPropagation();
    wake();
  }
  function docMove(e){
    if (grabs.has(e.pointerId)){
      const [x, y] = toStage(e.clientX, e.clientY);
      moveGrab(e.pointerId, x, y);
      wake();
      return;
    }
    if (grabs.size || e.pointerType !== 'mouse') return;
    if (phase === 'broken' || phase === 'reassembling'){
      const [x, y] = toStage(e.clientX, e.clientY);
      setCursor(inStage(x, y) && bodyAt(x, y, 4) ? 'grab' : null);
    } else setCursor(null);
  }
  function docUp(e){
    if (!grabs.has(e.pointerId)) return;
    releaseGrab(e.pointerId, false);
    swallowUntil = performance.now() + 800;
    try { root.releasePointerCapture(e.pointerId); } catch {}
    e.stopPropagation();
    wake();
  }
  function docCancel(e){ if (grabs.has(e.pointerId)){ releaseGrab(e.pointerId, true); wake(); } }
  function docClick(e){
    if (e.detail > 0 && performance.now() < swallowUntil){ swallowUntil = 0; e.preventDefault(); e.stopImmediatePropagation(); }
  }
  // a finger on a piece moves the piece, not the page
  function docTouch(e){ if (grabs.size && e.cancelable) e.preventDefault(); }
  function onWinBlur(){ releaseAllGrabs(); pressed = null; }

  const cap = { capture: true };
  const capNP = { capture: true, passive: false };
  button.addEventListener('pointerdown', onBtnDown);
  button.addEventListener('pointermove', onBtnMove);
  button.addEventListener('pointerup', onBtnUp);
  button.addEventListener('pointercancel', onBtnCancel);
  button.addEventListener('click', onClick, true);
  document.addEventListener('pointerdown', docDown, cap);
  document.addEventListener('pointermove', docMove, cap);
  document.addEventListener('pointerup', docUp, cap);
  document.addEventListener('pointercancel', docCancel, cap);
  document.addEventListener('lostpointercapture', docCancel, cap);
  document.addEventListener('click', docClick, cap);
  document.addEventListener('touchstart', docTouch, capNP);
  document.addEventListener('touchmove', docTouch, capNP);
  document.addEventListener('contextmenu', onWinBlur, cap);
  addEventListener('blur', onWinBlur);

  // ---- layout
  let btnRect = null;
  function measureBounds(){
    G.fixed = inFixedContext(button);
    canvas.style.position = G.fixed ? 'fixed' : 'absolute';
    const de = document.documentElement;
    const disp = canvas.style.display;
    canvas.style.display = 'none';
    bounds = G.fixed ? { w: de.clientWidth, h: de.clientHeight }
      : { w: de.clientWidth, h: Math.max(de.scrollHeight, document.body ? document.body.scrollHeight : 0) };
    canvas.style.display = disp;
    canvas.style.left = '0px'; canvas.style.top = '0px';
    const r = canvas.getBoundingClientRect();
    cbOrigin = { x: r.left + (G.fixed ? 0 : scrollX), y: r.top + (G.fixed ? 0 : scrollY) };
    G.bx = NaN;
    pageLin = toLin(pageColor(button.parentElement));
    pageLum = lumOf(pageLin);
  }
  function layout(r = button.getBoundingClientRect()){
    btnRect = r;
    const bx = r.left + (G.fixed ? 0 : scrollX), by = r.top + (G.fixed ? 0 : scrollY);
    const bw = r.width, bh = r.height;
    if (bw < 8 || bh < 8){ canvas.style.visibility = 'hidden'; G.bx = bx; G.by = by; return; }
    canvas.style.visibility = '';
    const M = opts.margin ?? Math.round(Math.max(110, 1.8 * bh));
    // the stage stays inside the document, so it can never add scrolling
    const x0 = Math.floor(Math.min(bx, Math.max(0, bx - M))), y0 = Math.floor(Math.min(by, Math.max(0, by - M)));
    const x1 = Math.ceil(Math.max(bx + bw, Math.min(bounds.w, bx + bw + M))), y1 = Math.ceil(Math.max(by + bh, Math.min(bounds.h, by + bh + M)));
    const sw = x1 - x0, sh = y1 - y0;
    const sizeChanged = sw !== G.stageW || sh !== G.stageH;
    const oldOffX = G.offX, oldOffY = G.offY;
    Object.assign(G, { bx, by, bw, bh, stageW: sw, stageH: sh, offX: bx - x0, offY: by - y0 });
    Object.assign(canvas.style, { left: `${x0 - cbOrigin.x}px`, top: `${y0 - cbOrigin.y}px` });
    const cs = getComputedStyle(button);
    const rad = Math.min(parseFloat(cs.borderTopLeftRadius) || 0, bw / 2, bh / 2);
    pill.w = bw; pill.h = bh; pill.r = Math.max(rad, 2);
    pill.cx = G.offX + bw / 2; pill.cy = G.offY + bh / 2;
    if (sizeChanged) Object.assign(canvas.style, { width: `${sw}px`, height: `${sh}px` });
    if (sizeChanged || !W) resizeGL();
    const shapeChanged = !GR || Math.abs(pill.w - rock.w) > 0.5 || Math.abs(pill.h - rock.h) > 0.5 || Math.abs(pill.r - rock.r) > 0.5;
    if (shapeChanged){
      // a new size is a new slab; whatever was lying around goes with the old one
      releaseAllGrabs();
      buildRock();
    } else {
      // the pieces keep their place relative to the button
      const mx = G.offX - oldOffX, my = G.offY - oldOffY;
      if (mx || my){
        for (const b of bodies){
          b.x += mx; b.y += my; b.wv = false;
          if (b.glide){ b.glide.x0 += mx; b.glide.y0 += my; b.glide.tx += mx; b.glide.ty += my; }
        }
        for (const c of chips){ c.x += mx; c.y += my; }
        for (const d of dust){ d.x += mx; d.y += my; }
        for (const gr of grabs.values()){ gr.tx += mx; gr.ty += my; gr.px += mx; gr.py += my; }
      }
      if (phase === 'broken' && sizeChanged) keepInStage();
      else if (phase === 'intact' && bodies[0]){ bodies[0].x = pill.cx + bodies[0].hx; bodies[0].y = pill.cy + bodies[0].hy; }
    }
    dirty = true;
    wake();
  }
  function keepInStage(){
    for (const b of bodies){
      const r = Math.min(b.R * 0.6, Math.min(G.stageW, G.stageH) * 0.45);
      b.x = clamp(b.x, r, Math.max(r, G.stageW - r)); b.y = clamp(b.y, r, Math.max(r, G.stageH - r));
      b.wv = false; b.rest = 0;
    }
    relaxPile(30);
    for (const b of bodies) if (!b.kin){ updateWorld(b); b.dx = b.dy = b.da = 0; confine(b); }
    settleUntil = simT + 0.6;
  }
  // After the stage jumps, the pile may be squeezed; walk the pieces apart by position alone (no
  // velocity), so nothing is flung.
  function relaxPile(budgetMs){
    const t0 = performance.now();
    for (const b of bodies){ b.rest = 0; b.wv = false; }
    for (let it = 0; it < 200 && performance.now() - t0 < budgetMs; it++){
      dyn.length = 0;
      for (const b of bodies) if (!b.kin) dyn.push(b);
      findDeepPairs();
      collide(0);
      const left = pushApart(0.2, 2, 4);
      for (const b of dyn){ b.dx = b.dy = b.da = 0; }
      if (!left) break;
    }
    nContacts = 0;
  }
  function place(){
    const r = button.getBoundingClientRect();
    btnRect = r;
    const bx = r.left + (G.fixed ? 0 : scrollX), by = r.top + (G.fixed ? 0 : scrollY);
    if (Math.abs(bx - G.bx) > 0.25 || Math.abs(by - G.by) > 0.25 || Math.abs(r.width - G.bw) > 0.25 || Math.abs(r.height - G.bh) > 0.25) layout(r);
  }

  function resizeGL(){
    if (lost || gl.isContextLost() || !G.stageW) return;
    let d = Math.min(devicePixelRatio || 1, 2);
    const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    d = Math.min(d, maxTex / G.stageW, maxTex / G.stageH, Math.sqrt(6e6 / (G.stageW * G.stageH)));
    canvas.width = Math.max(1, Math.round(G.stageW * d)); canvas.height = Math.max(1, Math.round(G.stageH * d));
    W = gl.drawingBufferWidth; H = gl.drawingBufferHeight;
    const newDpr = W / G.stageW;
    if (Math.abs(newDpr - dpr) > 1e-3) stoneKey = '';
    dpr = newDpr;
    [g0Tex, g1Tex, g2Tex, shTex].forEach(t => t && gl.deleteTexture(t));
    [gFb, shFb].forEach(f => f && gl.deleteFramebuffer(f));
    if (depthRb) gl.deleteRenderbuffer(depthRb);
    g0Tex = makeTex(W, H, gl.RGBA32F, gl.NEAREST);
    g1Tex = makeTex(W, H, gl.RGBA16F, gl.NEAREST);
    depthRb = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, depthRb);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, W, H);
    g2Tex = makeTex(W, H, gl.RGBA16F, gl.LINEAR, Math.floor(Math.log2(Math.max(W, H))) + 1);
    gFb = makeTarget([g0Tex, g1Tex, g2Tex], depthRb);
    shTex = makeTex(W, H, gl.RG8, gl.NEAREST);
    shFb = makeTarget([shTex]);
    if (GR) bakeStone();
    dirty = true;
  }

  function bakeStone(){
    if (!mat || !GR) return;
    const key = rock.w + ':' + rock.h + ':' + dpr + ':' + mat.key;
    if (key === stoneKey) return;
    stoneKey = key;
    const M = 8, texel = 1 / dpr;
    const tw = Math.ceil((rock.w + 2 * M) * dpr), th = Math.ceil((rock.h + 2 * M) * dpr);
    [stoneTex, stoneMTex].forEach(t => t && gl.deleteTexture(t));
    if (stoneFb) gl.deleteFramebuffer(stoneFb);
    stoneTex = makeTex(tw, th, gl.RGBA8, gl.LINEAR, Math.min(5, Math.floor(Math.log2(Math.min(tw, th))) + 1));
    stoneMTex = makeTex(tw, th, gl.RGBA8, gl.LINEAR);
    stoneFb = makeTarget([stoneTex, stoneMTex]);
    const ox = -rock.w / 2 - M, oy = -rock.h / 2 - M;
    stoneBox = [ox, oy, tw * texel, th * texel];
    gl.bindFramebuffer(gl.FRAMEBUFFER, stoneFb);
    gl.viewport(0, 0, tw, th);
    const S = P.stone, u = S.u, m = mat;
    gl.useProgram(S.p);
    gl.uniform2f(u.uOrg, ox, oy);
    gl.uniform1f(u.uTexel, texel);
    gl.uniform1i(u.uKind, m.kind === 0 ? 0 : 1);
    gl.uniform1f(u.uGScale, m.kind === 0 ? 4.8 / (m.grain ?? 1) : 1);
    gl.uniform1f(u.uSeed, opts.seed ?? 0);
    if (m.kind === 0){
      gl.uniform4f(u.uTint, ...(m.tint ? toLin(m.tint) : [0, 0, 0]), m.tint ? 0.85 : 0);
      gl.uniform1f(u.uMeanLum, m.meanLum);
    } else {
      const L = c => toLin(c);
      gl.uniform3f(u.uBase, ...L(m.base)); gl.uniform3f(u.uBase2, ...L(m.base2)); gl.uniform3f(u.uVein, ...L(m.vein));
      gl.uniform3f(u.uSpeck, ...L(m.speckle[0])); gl.uniform3f(u.uSpeck2, ...L(m.speckle[1]));
      gl.uniform4f(u.uGrain, m.grain ?? 1, m.contrast ?? 0.08, m.mottle ?? 0.6, m.mottleScale ?? 60);
      gl.uniform4f(u.uVeinP, m.veinAmount ?? 0, m.veinScale ?? 70, m.veinWidth ?? 0.07, m.veinWarp ?? 1);
      gl.uniform4f(u.uSpeckP, m.speckleAmount ?? 0, m.speckleSize ?? 1, m.bands ?? 0, m.bandSpacing ?? 5);
      gl.uniform4f(u.uSurf, m.pits ?? 0, m.roughness ?? 0.8, m.sheen ?? 0.3, m.sparkle ?? 0.1);
      gl.uniform1f(u.uBandAng, m.bandAngle ?? 0.2);
      gl.uniform4f(u.uTint, 0, 0, 0, 0);
      gl.uniform1f(u.uMeanLum, 1);
      gl.uniform4f(u.uVarP, m.variation ?? 0, m.variationScale ?? 40, 0, 0);
      gl.uniform4f(u.uMin, ...(m.mineral ? L(m.mineral) : [0, 0, 0]), m.mineral ? (m.mineralAmount ?? 0.06) : 0);
      gl.uniform4f(u.uBandP, m.bandMode === 'cleavage' ? 1 : 0, m.crossBedding ?? 0, m.layerStep ?? 0, 0);
      gl.uniform4f(u.uRelP, m.grainRelief ?? 0.3, 0, m.pitDepth ?? 1.2, 0);
    }
    gl.uniform4f(u.uSlabW, rock.w / 2, rock.h / 2, rock.r, m.weather ?? 0.6);
    gl.bindVertexArray(emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    // the label's colour follows what the stone actually came out as, since a tint moved through a
    // preset's structure can end up much lighter or darker than the colour it was given
    if (!mat.labelChecked){
      const px = new Uint8Array(tw * th * 4);
      gl.readBuffer(gl.COLOR_ATTACHMENT0);
      gl.readPixels(0, 0, tw, th, gl.RGBA, gl.UNSIGNED_BYTE, px);
      let sum = 0, n = 0;
      for (let i = 0; i < px.length; i += 4 * 7){ sum += 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2]; n++; }
      mat.lum = sum / n / 255;
      mat.labelChecked = true;
      chooseInlay(mat.lum);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, stoneTex);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    dirty = true;
  }

  // ---- render
  function render(){
    if (!gFb || !stoneTex || !GR) return;
    const sw = G.stageW, sh = G.stageH;
    gl.bindFramebuffer(gl.FRAMEBUFFER, gFb);
    gl.viewport(0, 0, W, H);
    gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, -1]);
    gl.clearBufferfv(gl.COLOR, 1, [0, 0, 0, 0]);
    gl.clearBufferfv(gl.COLOR, 2, [0, 0, 0, 0]);
    gl.clearBufferfv(gl.DEPTH, 0, [1]);
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LESS);

    const live = bodies.filter(b => b.row >= 0);
    const nb = live.length;
    const d = fillInst(pieceInst, nb);
    live.forEach((b, i) => {
      const o = i * 20;
      d[o] = b.x; d[o + 1] = b.y; d[o + 2] = b.a; d[o + 3] = b.z;
      d[o + 4] = b.box[0]; d[o + 5] = b.box[1]; d[o + 6] = b.box[2]; d[o + 7] = b.box[3];
      d[o + 8] = b.row; d[o + 9] = b.nO; d[o + 10] = b.nT; d[o + 11] = b.seam;
      d[o + 12] = b.hx; d[o + 13] = b.hy; d[o + 14] = b.id % 4000000; d[o + 15] = b.T;
      d[o + 16] = b.home ? 1 : 0; d[o + 17] = 0; d[o + 18] = 0; d[o + 19] = 0;
    });
    const pp = P.piece;
    gl.useProgram(pp.p);
    gl.uniform2f(pp.u.uCss, sw, sh);
    gl.uniform1f(pp.u.uLean, LEAN);
    gl.uniform1f(pp.u.uDpr, dpr);
    gl.uniform1f(pp.u.uWear, mat.wear ?? 1);
    gl.uniform3f(pp.u.uSlab, rock.w / 2, rock.h / 2, rock.r);
    bindTex(0, segTex); gl.uniform1i(pp.u.uSegs, 0);
    bindTex(1, textTex); gl.uniform1i(pp.u.uText, 1);
    gl.uniform4f(pp.u.uTextBox, -rock.w / 2, -rock.h / 2, rock.w, rock.h);
    bindTex(2, stoneTex); gl.uniform1i(pp.u.uStone, 2);
    gl.uniform4f(pp.u.uStoneBox, ...stoneBox);
    gl.bindVertexArray(pieceInst.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, pieceInst.buf);
    gl.bufferData(gl.ARRAY_BUFFER, d.subarray(0, nb * 20), gl.DYNAMIC_DRAW);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, nb);

    if (chips.length){
      const cd = fillInst(chipInst, chips.length);
      chips.forEach((c, i) => {
        const o = i * 12;
        cd[o] = c.x; cd[o + 1] = c.y; cd[o + 2] = c.z; cd[o + 3] = c.a;
        cd[o + 4] = c.size; cd[o + 5] = c.seed; cd[o + 6] = c.id % 40000; cd[o + 7] = 0;
        cd[o + 8] = c.ox; cd[o + 9] = c.oy; cd[o + 10] = 0; cd[o + 11] = 0;
      });
      gl.useProgram(P.chip.p);
      gl.uniform2f(P.chip.u.uCss, sw, sh);
      gl.uniform1f(P.chip.u.uDpr, dpr);
      gl.bindVertexArray(chipInst.vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, chipInst.buf);
      gl.bufferData(gl.ARRAY_BUFFER, cd.subarray(0, chips.length * 12), gl.DYNAMIC_DRAW);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, chips.length);
    }
    gl.disable(gl.DEPTH_TEST);

    bindTex(0, g2Tex);
    gl.generateMipmap(gl.TEXTURE_2D);

    gl.bindFramebuffer(gl.FRAMEBUFFER, shFb);
    gl.viewport(0, 0, W, H);
    // the shadow march only runs where stone or its shadow can be
    gl.clearBufferfv(gl.COLOR, 0, [1, 1, 0, 1]);
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    // pieces at home count as the whole slab, so the sealed pieces and the intact slab get the
    // identical shadow region
    const homeR = Math.hypot(rock.w, rock.h) / 2 + 2;
    for (const b of bodies){
      const r = b.home ? homeR : b.R + 2, bx = b.home ? pill.cx : b.x, by = b.home ? pill.cy : b.y;
      x0 = Math.min(x0, bx - r); y0 = Math.min(y0, by - r); x1 = Math.max(x1, bx + r); y1 = Math.max(y1, by + r);
    }
    for (const c of chips){ x0 = Math.min(x0, c.x - 4); y0 = Math.min(y0, c.y - 4); x1 = Math.max(x1, c.x + 4); y1 = Math.max(y1, c.y + 4); }
    const lh = Math.hypot(LIGHT[0], LIGHT[1]), shLen = (rock.T + 16) * lh / LIGHT[2];
    const sx = -LIGHT[0] / lh * shLen, sy = -LIGHT[1] / lh * shLen, pad = 22;
    x0 = Math.min(x0, x0 + sx) - pad; x1 = Math.max(x1, x1 + sx) + pad;
    y0 = Math.min(y0, y0 + sy) - pad; y1 = Math.max(y1, y1 + sy) + pad;
    const sc = [Math.max(0, Math.floor(x0 * dpr)), Math.max(0, Math.floor((sh - y1) * dpr)), 0, 0];
    sc[2] = Math.min(W, Math.ceil(x1 * dpr)) - sc[0]; sc[3] = Math.min(H, Math.ceil((sh - y0) * dpr)) - sc[1];
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(sc[0], sc[1], Math.max(sc[2], 0), Math.max(sc[3], 0));
    gl.useProgram(P.shadow.p);
    gl.uniform1i(P.shadow.u.uH, 0);
    gl.uniform2f(P.shadow.u.uSize, W, H);
    gl.uniform1f(P.shadow.u.uDpr, dpr);
    gl.uniform3f(P.shadow.u.uL, ...LIGHT);
    let top = rock.T;
    for (const b of bodies) if (b.T + b.z > top) top = b.T + b.z;
    for (const c of chips) if (c.z + c.size > top) top = c.z + c.size;
    gl.uniform1f(P.shadow.u.uTop, top + 1);
    gl.bindVertexArray(emptyVao);
    if (sc[2] > 0 && sc[3] > 0) gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.disable(gl.SCISSOR_TEST);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, W, H);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    const c = P.comp;
    gl.useProgram(c.p);
    bindTex(0, g0Tex); gl.uniform1i(c.u.uG0, 0);
    bindTex(1, g1Tex); gl.uniform1i(c.u.uG1, 1);
    bindTex(2, g2Tex); gl.uniform1i(c.u.uG2, 2);
    bindTex(5, shTex); gl.uniform1i(c.u.uSh, 5);
    bindTex(6, stoneTex); gl.uniform1i(c.u.uStone, 6);
    bindTex(7, stoneMTex); gl.uniform1i(c.u.uStoneM, 7);
    gl.uniform4f(c.u.uStoneBox, ...stoneBox);
    gl.uniform1f(c.u.uDpr, dpr);
    gl.uniform3f(c.u.uL, ...LIGHT);
    gl.uniform3f(c.u.uPage, ...pageLin);
    // a black veil hardly shows on a dark page; there the shadow has to be denser to read at all
    gl.uniform1f(c.u.uShadowK, 0.72 + 0.35 * clamp(1 - pageLum * 4, 0, 1));
    gl.uniform1f(c.u.uSpec, mat.sheen ?? 1);
    gl.uniform1f(c.u.uClear, mat.clear ?? 0);
    gl.uniform4f(c.u.uPillC, pill.cx, pill.cy, pill.w, pill.h);
    gl.uniform4f(c.u.uInlay, ...inlay);
    gl.uniform4f(c.u.uFrac, ...toLin(mat.fracture), mat.fractureAmt ?? 0.4);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    if (dust.length){
      // dust is only as visible as it is different from the page it settles on; a pale cloud over a
      // dark page is thinned and pulled toward the page's own colour
      const dk = clamp(0.18 / (Math.abs(lumOf(toLin(mat.dust)) - pageLum) + 0.04), 0.3, 1);
      const dustCol = mat.dust.map((v, i) => v + (Math.pow(pageLin[i], 1 / 2.2) - v) * (1 - dk) * 0.5);
      const dd = fillInst(dustInst, dust.length);
      dust.forEach((p, i) => {
        const t = p.age / p.life;
        dd[i * 4] = p.x; dd[i * 4 + 1] = p.y; dd[i * 4 + 2] = p.r;
        dd[i * 4 + 3] = dk * p.a0 * Math.pow(1 - t, 1.6) * Math.min(1, p.age / 0.05);
      });
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.useProgram(P.dust.p);
      gl.uniform2f(P.dust.u.uCss, sw, sh);
      // dust is only as visible as it is different from the page it settles on; a pale cloud over a dark
      // page is thinned and pulled toward the page's own colour
      gl.uniform3f(P.dust.u.uCol, ...dustCol);
      gl.bindVertexArray(dustInst.vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, dustInst.buf);
      gl.bufferData(gl.ARRAY_BUFFER, dd.subarray(0, dust.length * 4), gl.DYNAMIC_DRAW);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, dust.length);
      gl.disable(gl.BLEND);
    }
    gl.bindVertexArray(null);
    dirty = false;
  }

  // ---- loop
  let acc = 0, lastKnock = 0;
  function update(dt){
    simT += dt;
    if (phase === 'broken'){
      acc += dt;
      for (const b of bodies){
        const nat = thickness(b.area);
        if (b.T !== nat) b.T = Math.abs(b.T - nat) < 0.02 ? nat : b.T + (nat - b.T) * Math.min(1, dt * 8);
        if (b.z) b.z = b.z < 0.02 ? 0 : b.z * Math.max(0, 1 - dt * 10);
      }
      let n = 0, vmax = 0;
      for (const b of bodies) if (!b.kin){ const v = Math.hypot(b.vx, b.vy) + Math.abs(b.w) * b.R; if (v > vmax) vmax = v; }
      findDeepPairs();
      const div = clamp(Math.ceil(vmax * H_STEP / 3), 1, 3), h = H_STEP / div;
      ({ contact: contactSoft, wall: wallSoft, drag: dragSoft } = SOFTS[div - 1]);
      curH = h;
      while (acc >= h - 1e-9 && n < MAX_SUBSTEPS * div){ physicsStep(h); acc -= h; n++; }
      if (n === MAX_SUBSTEPS * div) acc = 0;
      if (knocks.length){
        knocks.sort((p, q) => q.v * Math.sqrt(q.m) - p.v * Math.sqrt(p.m));
        const now = performance.now();
        let played = 0;
        for (const k of knocks){
          if (played >= 2 || now - lastKnock < 28) break;
          snd.knock(k.v, k.m, k.wall); played++; lastKnock = now;
        }
        knocks.length = 0;
      }
      let lvl = 0, bright = 0;
      for (const gr of grabs.values()){
        const sp = Math.hypot(gr.body.vx, gr.body.vy) + Math.abs(gr.body.w) * gr.body.R * 0.5;
        lvl += clamp(sp / 700, 0, 1) * clamp(Math.sqrt(gr.body.area) / 60, 0.35, 1.2);
        bright = Math.max(bright, clamp(sp / 900, 0, 1));
      }
      snd.scrape(0.2 * Math.min(lvl, 1.4), bright);
      const idle = simT - lastInput;
      const wait = reduceMotion.matches ? Math.min(REGROUP, 1.2) : REGROUP;
      if (!grabs.size && idle > wait && (idle > wait + 2.5 || bodies.every(b => Math.hypot(b.vx, b.vy) + Math.abs(b.w) * b.R < 8))) startReassembly();
      dirty = true;
    } else if (phase === 'reassembling'){
      updateGlide(dt);
      snd.scrape(0, 0);
    }
    if (chips.length || dust.length){ updateParticles(dt); dirty = true; }
  }
  function busy(){ return phase !== 'intact' || chips.length > 0 || dust.length > 0 || grabs.size > 0; }

  let raf = 0, inFrame = false, frameCount = 0, stepCount = 0, last = performance.now(), manual = false, lost = false, visible = true;
  function wake(){
    // one animation loop at most: a wake from inside a frame must not start a second loop
    if (!raf && !inFrame && !manual && !lost) raf = requestAnimationFrame(frame);
  }
  function advance(dt){
    stepCount++;
    update(dt);
    if (!lost && !gl.isContextLost() && W && dirty && (visible || phase !== 'intact')) render();
  }
  function frame(now){
    raf = 0;
    if (manual || lost) return;
    const dt = Math.min((now - last) / 1000, 1 / 30);
    last = now;
    place();
    inFrame = true;
    frameCount++;
    try { advance(dt); } finally { inFrame = false; }
    if (busy() || (dirty && visible)) raf = requestAnimationFrame(frame);
  }

  // ---- keep the stage on the button
  const relayout = () => { if (lost) return; measureBounds(); layout(); };
  const ro = new ResizeObserver(relayout);
  ro.observe(button);
  if (document.body) ro.observe(document.body);
  const onScroll = () => { if (!lost) place(); };
  addEventListener('resize', relayout);
  document.addEventListener('scroll', onScroll, { capture: true, passive: true });
  let dprMq = null;
  const onDpr = () => { watchDpr(); relayout(); resizeGL(); render(); wake(); };
  function watchDpr(){
    dprMq?.removeEventListener('change', onDpr);
    dprMq = matchMedia(`(resolution: ${devicePixelRatio}dppx)`);
    dprMq.addEventListener('change', onDpr);
  }
  watchDpr();
  const io = new IntersectionObserver(es => { for (const e of es) visible = e.isIntersecting; if (visible){ dirty = true; wake(); } });
  io.observe(canvas);
  const onFonts = () => { if (!lost && GR){ buildText(); dirty = true; wake(); } };
  document.fonts?.addEventListener?.('loadingdone', onFonts);
  // the label follows text or style changes made to the button while it is attached
  const mo = new MutationObserver(() => {
    if (lost || !GR) return;
    const cs = getComputedStyle(button);
    if (labelText() + '|' + cs.font !== labelKey){ buildText(); dirty = true; wake(); }
  });
  mo.observe(button, { childList: true, characterData: true, subtree: true, attributes: true, attributeFilter: ['class'] });
  // a page theme switch changes the colour the shadows and bounce light work against
  const themeMo = new MutationObserver(() => { if (!lost){ measureBounds(); layout(); dirty = true; wake(); } });
  themeMo.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'data-theme'] });
  if (document.body) themeMo.observe(document.body, { attributes: true, attributeFilter: ['class', 'data-theme'] });

  // A lost context shows the plain button until the GPU comes back, then the stone is rebuilt whole.
  function onLost(e){
    e.preventDefault();
    lost = true;
    cancelAnimationFrame(raf); raf = 0;
    releaseAllGrabs(); pressed = null;
    phase = 'intact'; bodies = []; chips.length = 0; dust.length = 0; GR = null;
    enablePointer(); setCursor(null);
    showButton();
    canvas.style.visibility = 'hidden';
  }
  function onRestored(){
    try { initGL(); } catch (err) { console.warn('ElementalFX rock: could not rebuild after context loss', err); return; }
    lost = false;
    hideButton();
    canvas.style.visibility = '';
    G.stageW = G.stageH = 0; W = 0;
    relayout();
    render();
    wake();
  }
  canvas.addEventListener('webglcontextlost', onLost);
  canvas.addEventListener('webglcontextrestored', onRestored);

  // ---- go
  applyMaterial();
  hideButton();
  document.body.appendChild(canvas);
  relayout();
  render();

  const api = {
    supported: true,
    get state(){ return lost ? 'lost' : phase; },
    get pieces(){
      return bodies.map(b => ({ id: b.id, x: b.x - G.offX, y: b.y - G.offY, angle: b.a, vx: b.vx, vy: b.vy, w: b.w, area: b.area, grains: b.grains.length, R: b.R, T: b.T, z: b.z, seam: b.seam, home: b.home }));
    },
    get material(){ return mat ? mat.name : null; },
    setMaterial(m){
      matSpec = m ?? 'granite';
      applyMaterial();
      if (!lost){ bakeStone(); render(); wake(); }
    },
    reset(now){
      if (lost) return;
      if (now){ if (phase !== 'intact'){ releaseAllGrabs(); for (const b of bodies) freeBody(b); bodies = []; bodies.push(intactBody()); chips.length = 0; dust.length = 0; phase = 'intact'; enablePointer(); dirty = true; render(); } return; }
      lastInput = simT; startReassembly();
    },
    detach(){
      cancelAnimationFrame(raf); raf = 0; manual = true;
      clearTimeout(peTimer);
      ro.disconnect(); io.disconnect(); mo.disconnect(); themeMo.disconnect();
      dprMq?.removeEventListener('change', onDpr);
      document.fonts?.removeEventListener?.('loadingdone', onFonts);
      removeEventListener('resize', relayout); removeEventListener('blur', onWinBlur);
      document.removeEventListener('scroll', onScroll, { capture: true });
      button.removeEventListener('pointerdown', onBtnDown); button.removeEventListener('pointermove', onBtnMove);
      button.removeEventListener('pointerup', onBtnUp); button.removeEventListener('pointercancel', onBtnCancel);
      button.removeEventListener('click', onClick, true);
      document.removeEventListener('pointerdown', docDown, cap); document.removeEventListener('pointermove', docMove, cap);
      document.removeEventListener('pointerup', docUp, cap); document.removeEventListener('pointercancel', docCancel, cap);
      document.removeEventListener('lostpointercapture', docCancel, cap); document.removeEventListener('click', docClick, cap);
      document.removeEventListener('touchstart', docTouch, capNP); document.removeEventListener('touchmove', docTouch, capNP);
      document.removeEventListener('contextmenu', onWinBlur, cap);
      canvas.removeEventListener('webglcontextlost', onLost); canvas.removeEventListener('webglcontextrestored', onRestored);
      for (const id of grabs.keys()){ try { root.releasePointerCapture(id); } catch {} }
      grabs.clear();
      setCursor(null);
      snd.scrape(0, 0);
      showButton();
      canvas.remove();
      gl.getExtension('WEBGL_lose_context')?.loseContext();
    },
    // test hooks
    get _frames(){ return frameCount; },
    get _steps(){ return stepCount; },
    get _raf(){ return raf !== 0; },
    _manual(on){ manual = on; last = performance.now(); if (!on) wake(); else { cancelAnimationFrame(raf); raf = 0; } },
    _advance(sec, fps = 60){ place(); for (let i = 0, n = Math.round(sec * fps); i < n; i++) advance(1 / fps); },
    // total fragment area against the intact slab's, now and at every fracture since the last break
    _area(){ return { original: totalArea0, fragments: fragmentArea(), ratio: fragmentArea() / totalArea0, pieces: bodies.length, gravel: bodies.filter(b => b.grains.length < MIN_SPLIT).length, log: areaLog.slice() }; },
    // stone pixels actually drawn (device px): top faces, walls, and coverage-weighted total
    _gbuf(){
      render();
      gl.bindFramebuffer(gl.FRAMEBUFFER, gFb);
      gl.readBuffer(gl.COLOR_ATTACHMENT0);
      const g0 = new Float32Array(W * H * 4);
      gl.readPixels(0, 0, W, H, gl.RGBA, gl.FLOAT, g0);
      gl.readBuffer(gl.COLOR_ATTACHMENT1);
      const g1 = new Float32Array(W * H * 4);
      gl.readPixels(0, 0, W, H, gl.RGBA, gl.FLOAT, g1);
      gl.readBuffer(gl.COLOR_ATTACHMENT2);
      const g2 = new Float32Array(W * H * 4);
      gl.readPixels(0, 0, W, H, gl.RGBA, gl.FLOAT, g2);
      gl.readBuffer(gl.COLOR_ATTACHMENT0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      const out = new Uint8Array(W * H * 4);
      gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, out);
      return { g0, g1, g2, out, W, H, dpr };
    },
    _coverage(withMask){
      const { g0, g1 } = api._gbuf();
      let top = 0, wall = 0, cov = 0;
      const mask = withMask ? new Array(W * H).fill(0) : null;
      for (let i = 0; i < W * H; i++){
        const w = g0[i * 4 + 3];
        if (w < 0) continue;
        if (w % 2 > 0.5){ wall++; if (mask) mask[i] = 2; } else { top++; cov += g1[i * 4 + 2]; if (mask) mask[i] = 1; }
      }
      return { top, wall, cov, dpr, W, H, mask };
    },
    _bodies(){ return bodies.map(b => ({ id: b.id, row: b.row, x: b.x, y: b.y, a: b.a, hx: b.hx, hy: b.hy, nO: b.nO, nT: b.nT, box: b.box, T: b.T, seam: b.seam, home: b.home, segs: Array.from(b.segs || []) })); },
    _outlines(){ return bodies.map(b => ({ id: b.id, hx: b.hx, hy: b.hy, nO: b.nO, segs: Array.from(b.segs || []) })); },
    _setVelocity(id, vx, vy, w = 0){ const b = bodies.find(q => q.id === id); if (b && !b.kin){ b.vx = vx; b.vy = vy; b.w = w; b.rest = 0; lastInput = simT; wake(); } },
    get _sim(){ return { G, pill, rock, get phase(){ return phase; }, get grabs(){ return grabs.size; }, get chips(){ return chips.length; }, get dust(){ return dust.length; }, fractures: () => fractureCount, get pointerEvents(){ return button.style.pointerEvents; } }; },
  };
  return api;
}

window.ElementalFX = Object.assign(window.ElementalFX || {}, { attachRock, rockAudio, rockMaterials: Object.keys(PRESETS).filter(k => k[0] !== '_') });
})();
