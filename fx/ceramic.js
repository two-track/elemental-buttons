// Ceramic button: turns an existing <button> into a glazed ceramic tile with its label printed under the glaze.
//
//   <script src="fx/ceramic.js"></script>
//   const fx = ElementalFX.attachCeramic(document.querySelector('#buy'), { glaze: 'celadon' });   // fx.detach() restores it
//
// A classic script (not a module) so pages work straight from file:// too.
//
// The <button> stays the real control (focus, keyboard, form submission, click handlers); this only draws
// it, in a transparent canvas laid over the page around it. The click smashes the tile into a handful of
// curved shell shards that ring, spin, skitter and settle flat. While they are down they can be picked
// up and tossed (that never counts as a click). After a quiet moment the shards slide back and fit
// together exactly, the seams fill with molten gold (kintsugi), the gold cools and gleams, then fades,
// leaving the tile clean. The cursor is the studio lamp the glaze reflects. If WebGL2 is unavailable the
// button is left untouched.
//
// Options:
//   glaze: 'porcelain' (default) | 'celadon' | 'cobalt' (blue-and-white) | 'stoneware' (matte, speckled)
//          | 'terracotta' | 'auto' (chosen and tinted from the button's own colours)
//          | { preset, base, pool, rim, body, ink, gloss, craze, speckle, decor }   (CSS colours, 0..1 amounts)
//   color: a CSS colour to re-tint the chosen glaze with.
//   inkColor: CSS colour of the printed label (default: the glaze's own underglaze colour). The label is held
//          to WCAG 4.5:1 or better: an explicit ink or colour keeps its hue and only its lightness moves, while
//          a preset or 'auto' glaze may deepen or pale a little so the author's text colour can stay.
//   keepGold: true leaves the gold seams in place after the mend, until the next break.
//   restoreAfter: seconds of quiet after the shards settle before they mend (default 1.4).
//   margin: CSS px of stage around the button the shards may travel in (default max(100, 1.7 x height)).
//   zIndex: of the stage canvas (default 10).   sound: false to stay silent.
//
// fx.setGlaze(glaze, color?) changes the glaze live; fx.restore() mends now; fx.detach() removes it all.
// ElementalFX.ceramicAudio.muted silences every ceramic button on the page.

(() => {
'use strict';

const TAU = Math.PI * 2;
const rand = (a, b) => a + Math.random() * (b - a);
const clamp = (v, a, b) => v < a ? a : v > b ? b : v;
const ease = v => v * v * (3 - 2 * v);
const easeIO = v => v < 0.5 ? 4 * v * v * v : 1 - Math.pow(-2 * v + 2, 3) / 2;
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');

// ---------------------------------------------------------------- tuning (CSS px, seconds)
const LEAN = 0.8;              // screen px a point rises per px of height (a slightly oblique view from the south)
const EYE_H = 900;             // viewer's distance from the screen
const LAMP_H = 170;            // height of the studio softbox above the page
const SH = [0.26, 0.46];       // shadow run per px of height: the key light is up and to the left
const G = 2400;                // gravity
const FRICTION = 900;          // sliding deceleration on the page once pieces have lost their momentum
const FRICTION_LO = 65;        // ... and in the first moments, while they skate and rattle outward on their glaze
const SLIDE_T = [0.8, 1.3];    // seconds after the blow over which friction climbs from low to full
const TILT_G = 70;             // how fast a tipped shard falls flat (rad/s^2)
const EDGE = 18;               // shards keep this far inside the stage, so their shadows never reach its edge
const EDGE_SOFT = 26;          // inside this band friction climbs steeply, so pieces come to rest well short of EDGE
const GRAB = 12;
const DRAG_START = 5;
const STRIDE = 16;             // floats per vertex
const SEAM_R = 10;             // how far from a crack its distance field reaches (CSS px)
const MODE_RATIOS = [1, 1.52, 2.03, 2.71, 3.36, 4.21];
const MODE_AMPS = [1, 0.8, 0.62, 0.5, 0.36, 0.26];

// ---------------------------------------------------------------- shaders
const COMMON = `#version 300 es
precision highp float;
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*.1031); p3 += dot(p3, p3.yzx+33.33); return fract((p3.x+p3.y)*p3.z); }
vec2 hash22(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*vec3(.1031,.1030,.0973)); p3 += dot(p3, p3.yzx+33.33); return fract((p3.xx+p3.yz)*p3.zy); }
float vnoise(vec2 p){
  vec2 i = floor(p), f = fract(p); vec2 u = f*f*(3.-2.*f);
  return mix(mix(hash12(i), hash12(i+vec2(1,0)), u.x), mix(hash12(i+vec2(0,1)), hash12(i+vec2(1,1)), u.x), u.y);
}
float fbm(vec2 p){ return 0.5*vnoise(p) + 0.3*vnoise(p*2.03+7.1) + 0.2*vnoise(p*4.1+3.7); }
`;

// One stream for everything a shard draws: its shadow, its side walls and its glazed top.
const VS = `#version 300 es
in vec2 aPos; in vec2 aHome; in vec4 aEdge; in vec3 aN; in vec2 aR; in vec3 aK;
uniform vec2 uStage;
out vec2 vStage; out vec2 vHome; out vec4 vEdge; out vec3 vN; out vec2 vR; out vec3 vK;
void main(){
  vStage = aPos; vHome = aHome; vEdge = aEdge; vN = aN; vR = aR; vK = aK;
  gl_Position = vec4(aPos.x / uStage.x * 2. - 1., 1. - aPos.y / uStage.y * 2., 0., 1.);
}`;

const FS = COMMON + `
in vec2 vStage; in vec2 vHome; in vec4 vEdge; in vec3 vN; in vec2 vR; in vec3 vK;
uniform vec2 uSize; uniform float uRad;
uniform vec3 uGlaze, uPool, uRim, uBody, uInk, uCrazeCol;
uniform vec4 uMat;     // gloss, craze, speckle, rim break
uniform vec4 uMat2;    // pooling, mottle, ink strength, decor
uniform sampler2D uText;
uniform sampler2D uSeam;   // distance to the nearest crack (r, /SEAM_R) and the way to it (gb), in home coords
uniform vec2 uSeamUV;
uniform vec2 uEye, uLamp, uLampSize;
uniform vec4 uGold;    // fill radius (home px), molten 0..1, gleam position, visibility
uniform float uCrack;  // how much the break still shows along the seams
uniform vec2 uImpact;
uniform float uFlash, uDisabled;
out vec4 o;

const vec3 KEY = vec3(-0.4264, -0.5876, 0.6876);

float sdRR(vec2 h, out vec2 g){
  vec2 c = uSize * 0.5, p = h - c, s = vec2(p.x < 0. ? -1. : 1., p.y < 0. ? -1. : 1.);
  vec2 q = abs(p) - (c - uRad);
  if (q.x > 0. && q.y > 0.) g = s * normalize(q);
  else g = q.x > q.y ? vec2(s.x, 0.) : vec2(0., s.y);
  return length(max(q, 0.)) + min(max(q.x, q.y), 0.) - uRad;
}
vec2 rot(vec2 v){ return vec2(vR.x * v.x - vR.y * v.y, vR.y * v.x + vR.x * v.y); }

// What a glossy surface with normal n at stage point p reflects: the room (rgb) and how much of the
// studio softbox it sees (a): a rounded window hanging LAMP_H above the page. Because the window is found
// through the surface normal, its reflection bends and squeezes over the rim bevel like a real one.
vec4 envOf(vec3 n, vec2 p, float rough, vec3 V){
  vec3 R = reflect(-V, n);
  vec2 hit = p + R.xy * (${LAMP_H.toFixed(1)} / max(R.z, 0.06));
  float rc = min(uLampSize.y * 0.8, 6.);
  vec2 q = abs(hit - uLamp) - uLampSize + rc;
  float sd = length(max(q, 0.)) + min(max(q.x, q.y), 0.) - rc;
  // antialiased over about a device pixel wherever the reflection lands, however squeezed it is there
  float aa = min(fwidth(sd) * 1.3, 3.) + mix(0.25, 60., rough);
  float box = 1. - smoothstep(-aa, aa, sd);
  // the diffuser is brighter toward its top and centre, so the window has a gradient across it
  float gy = clamp((hit.y - uLamp.y) / uLampSize.y * 0.5 + 0.5, 0., 1.);
  float gx = (hit.x - uLamp.x) / uLampSize.x;
  box *= (1. - 0.2 * gy) * max(1. - 0.18 * gx * gx, 0.) * mix(1., 0.12, rough);
  float room = mix(0.02, 0.5, smoothstep(0.35, 0.98, R.z)) * (0.8 + 0.2 * smoothstep(-0.5, 0.5, -R.y));
  return vec4(vec3(0.92, 0.95, 1.0) * room, box);
}
float fresnel(vec3 n, vec3 V){
  return 0.04 + 0.96 * pow(1. - max(dot(n, V), 0.), 5.);
}
// the glaze has shrunk on the body and crazed into a fine irregular net
float crazeAt(vec2 h){
  float w = 0.;
  for (int s = 0; s < 2; s++){
    float cell = s == 0 ? 13. : 6.;
    vec2 g = h / cell, gi = floor(g);
    float d1 = 9., d2 = 9.;
    for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++){
      vec2 c = gi + vec2(i, j);
      float dd = length(g - c - hash22(c + float(s) * 19.));
      if (dd < d1){ d2 = d1; d1 = dd; } else if (dd < d2) d2 = dd;
    }
    float line = smoothstep(0.42, 0.08, (d2 - d1) * cell * 0.5);
    float gate = smoothstep(0.3, 0.6, vnoise(h * 0.05 + float(s) * 5.));
    w = max(w, line * gate * (s == 0 ? 1. : 0.55));
  }
  return w;
}
float speckAt(vec2 h, out float halo){
  vec2 g = floor(h / 3.);
  float sp = 0.; halo = 0.;
  for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++){
    vec2 c = g + vec2(i, j);
    float r0 = hash12(c + 3.1);
    if (r0 > 0.1 * uMat.z) continue;
    vec2 ctr = (c + 0.2 + 0.6 * hash22(c)) * 3.;
    float rr = 0.3 + 1.0 * pow(hash12(c + 7.), 2.);
    float d = length(h - ctr);
    sp = max(sp, smoothstep(rr, rr * 0.3, d));
    halo = max(halo, smoothstep(rr * 3.5, rr, d));
  }
  return sp;
}
vec3 bodyAt(vec2 h){
  float g = 0.86 + 0.14 * vnoise(h * 1.7) + 0.05 * (hash12(floor(h * 2.)) - 0.5);
  vec3 c = uBody * g;
  float hl; float sp = uMat.z > 0. ? speckAt(h + 31.7, hl) : 0.;
  return mix(c, uBody * 0.4, sp * 0.6);
}

void main(){
  float mode = vK.x;
  if (mode > 1.5){
    // shadow: softened inward from the outline
    o = vec4(0., 0., 0., smoothstep(0., vK.y, vEdge.x) * vK.z);
    return;
  }
  vec2 h = vHome;
  vec2 gr; float dr = -sdRR(h, gr);
  bool crack = vEdge.w > 0.5;
  float gloss = uMat.x, rough = 1. - gloss;
  vec3 col; vec3 n; float spK = 1., t0 = 0.;

  if (mode > 0.5){
    // ---- a side wall: v runs from the top face (0) down to the page (1)
    float v = vK.y;
    vec3 nw = normalize(vN);
    float lam = 0.5 + 0.5 * max(dot(nw, KEY), 0.) / 0.69;
    if (crack){
      // the broken edge: bare white body, with the thin glaze skin along its top
      vec3 b = bodyAt(h * 1.3 + v * 4.);
      // conchoidal ribs rippling out from where it was struck
      float rib = sin(length(h - uImpact) * 0.9 + fbm(h * 0.2) * 5. + v * 2.) * 0.5 + 0.5;
      b *= 0.9 + 0.1 * rib;
      col = b * (0.9 + 0.12 * lam);
      col = mix(col, uGlaze * (0.8 + 0.25 * lam), smoothstep(0.3, 0.12, v));
      col *= 1. - 0.25 * smoothstep(0.6, 1., v);
      spK = 0.25;
    } else {
      // the glazed rim running down to the foot, where the glaze thins and pools
      col = mix(uPool, uGlaze, 0.35) * (0.62 + 0.3 * lam);
      col = mix(col, mix(uRim, uBody, 0.5) * 0.7, smoothstep(0.75, 1., v) * uMat.w);
      col *= 1. - 0.2 * smoothstep(0.7, 1., v);
    }
    n = nw;
  } else {
    // ---- the glazed top
    float bev = clamp(uSize.y * 0.2, 4., 10.);
    float x = clamp(1. - dr / bev, 0., 1.);
    vec2 nh = gr * x * x * 1.7;
    // true distance to the nearest crack, whichever fan triangle this fragment happens to be in
    vec4 sk = texture(uSeam, h * uSeamUV);
    float dc = sk.r * ${SEAM_R.toFixed(1)};
    vec2 sdir = sk.gb * 2. - 1.;
    crack = dc < ${(SEAM_R - 0.5).toFixed(1)};
    // the broken glaze lip: a hard little edge along the break
    if (crack) nh += sdir * smoothstep(1.2, 0., dc) * 0.7 * uCrack;
    vec2 wav = vec2(fbm(h * 0.07), fbm(h * 0.07 + 9.7)) - 0.5;
    // orange peel: barely there on a good glaze, just enough to make reflected edges waver
    nh += wav * (0.012 + 0.1 * rough);
    n = normalize(vec3(rot(nh) + vN.xy / max(vN.z, 0.25), 1.));

    // glaze: mottled, thinning (breaking) over the crest of the rim and pooling just inside it
    float mot = fbm(h * 0.035 + 4.) - 0.5;
    col = uGlaze * (1. + mot * uMat2.y);
    float band = smoothstep(bev * 0.25, bev * 0.95, dr) * (1. - smoothstep(bev * 1.1, bev * 2.6, dr));
    col = mix(col, uPool, band * uMat2.x * (0.75 + 0.5 * fbm(h * 0.09)));
    float crest = 1. - smoothstep(0.4, bev * 0.55, dr);
    col = mix(col, uRim, crest * uMat.w * (0.6 + 0.4 * fbm(h * 0.3)));

    if (uMat.z > 0.){
      float halo; float sp = speckAt(h, halo);
      col = mix(col, col * vec3(0.86, 0.8, 0.72), halo * 0.35);
      col = mix(col, vec3(0.22, 0.14, 0.09), sp * 0.85);
    }
    if (uMat.y > 0.) col = mix(col, uCrazeCol, crazeAt(h) * uMat.y);

    // blue-and-white: a double painted border band
    if (uMat2.w > 0.){
      float wob = (fbm(h * 0.12) - 0.5) * 0.9;
      float l1 = smoothstep(0.9, 0.2, abs(dr - bev * 0.62 - 1.4 + wob));
      float l2 = smoothstep(0.55, 0.05, abs(dr - bev * 0.62 - 4.3 + wob * 0.7));
      col = mix(col, uInk * (0.95 + 0.2 * fbm(h * 0.5)), (l1 * 0.9 + l2 * 0.75) * uMat2.w);
    }

    // the label, printed under the glaze: a slightly soft stroke that pools a touch darker where it is dense
    t0 = texture(uText, h / uSize).r;
    // the stroke's core is fully inked; the blot only varies how far the soft edge spreads
    float blot = 1.08 + 0.12 * fbm(h * 0.55 + 2.);
    vec3 inkC = uInk * mix(1.03, 0.97, t0);
    col = mix(col, inkC, clamp(t0 * uMat2.z * blot, 0., 1.));

    // the break: along the seam the glaze has spalled off in shell-shaped flakes, baring the body
    float bare = 0.;
    if (crack && uCrack > 0.){
      float scar = pow(vnoise(h * 0.42 + 3.), 2.4) * 2.6 + 0.3;
      bare = smoothstep(scar + 0.35, scar - 0.15, dc) * uCrack;
      vec3 bb = bodyAt(h) * (0.85 + 0.2 * clamp(dot(n, KEY), 0., 1.));
      col = mix(col, bb, bare);
      col = mix(col, col * 0.55, smoothstep(0.7, 0., abs(dc - scar - 0.25)) * 0.35 * uCrack);
    }

    // kintsugi: gold laid into the seam and the chipped flakes beside it
    if (crack && uGold.w > 0.){
      float filled = smoothstep(0., 10., uGold.x - length(h - uImpact));
      float w = 1.1 + 0.45 * vnoise(h * 0.25);
      float seam = max(smoothstep(w + 0.5, w - 0.35, dc), bare > 0.5 ? 1. : bare * 2.) * filled;
      // a raised bead of lacquer: its normal rolls across the seam
      float across = clamp(dc / max(w, 0.5), 0., 1.);
      vec3 gn = normalize(vec3(rot(-sdir) * across * 1.1 + wav * 0.2, 1.));
      vec3 gold = vec3(0.95, 0.74, 0.33);
      float gl = 0.5 + 0.5 * max(dot(gn, KEY), 0.) / 0.69;
      vec4 genv = envOf(gn, vStage, 0.25, normalize(vec3(uEye - vStage, ${EYE_H.toFixed(1)})));
      vec3 gc = gold * (0.35 + 0.55 * gl) + gold * (genv.rgb + genv.a * 2.2) * 0.45;
      float gleam = exp(-pow((h.x + h.y * 0.5 - uGold.z) / 10., 2.));
      gc += vec3(1.0, 0.9, 0.62) * gleam * 1.3;
      // molten: white-hot at the core, orange at the edges; kept off the print so the label still reads
      float hotK = uGold.y * (1. - 0.7 * t0);
      vec3 hot = mix(vec3(1.0, 0.45, 0.08), vec3(1.0, 0.93, 0.62), smoothstep(0.9, 0.1, across));
      gc = mix(gc, hot * 1.2, hotK);
      // a narrow halo that follows the seam line itself
      float halo = exp(-max(dc - w, 0.) / 1.6) * smoothstep(6., 2.5, dc);
      col += vec3(1.0, 0.5, 0.12) * halo * filled * hotK * 0.5 * (1. - seam);
      col = mix(col, gc, seam * uGold.w);
      // the lettering shows through the gold, as lacquer laid over a printed mark would let it
      col = mix(col, inkC * 0.9, seam * uGold.w * t0 * 0.55);
      spK *= 1. - seam * uGold.w;
    }
  }

  // lighting: diffuse from the key light (flat top = the glaze's own colour), then the glossy reflection on top
  // walls are seen obliquely from the south, tops from straight in front
  vec3 Vv = mode > 0.5 ? vec3(0., 0.6247, 0.7809) : normalize(vec3(uEye - vStage, ${EYE_H.toFixed(1)}));
  float diff = mode > 0.5 ? 0.82 + 0.28 * max(dot(n, KEY), 0.) / 0.69 : 0.6 + 0.4 * max(dot(n, KEY), 0.) / 0.6876;
  float F = fresnel(n, Vv);
  vec4 E = envOf(n, vStage, rough, Vv);
  col = col * diff * (1. - F * gloss) + E.rgb * F * (0.2 + 0.8 * gloss);
  // The softbox reads as a luminous window, tinted by the glaze it is seen through and never paper white;
  // stronger at grazing angles, where the glaze reflects more.
  vec3 hiC = mix(vec3(1.0, 0.995, 0.985), uGlaze, 0.12) * 0.975;
  float hk = E.a * spK * clamp(0.95 + F * 3., 0., 1.);
  col = mix(col, max(col, hiC), hk);
  col += uFlash * 0.18 * (1. - smoothstep(0., 1., abs(h.x / uSize.x - 0.5 + (h.y / uSize.y - 0.5) * 0.4) * 2.));
  if (uDisabled > 0.) col = mix(col, vec3(dot(col, vec3(0.3, 0.55, 0.15))), 0.55 * uDisabled);
  float a = vK.z * (1. - 0.45 * uDisabled);
  o = vec4(min(col, vec3(1.)) * a, a);
}`;

// crumbs of glaze and body thrown off the break, and a puff of bisque dust
const VS_PART = `#version 300 es
in vec4 aP;   // x, y, size, alpha
in vec4 aC;   // r, g, b, kind (0 crumb, 1 dust)
in float aS;  // seed
uniform vec2 uStage; uniform float uDpr;
out vec4 vC; out float vA; out float vS;
void main(){
  vC = aC; vA = aP.w; vS = aS;
  gl_Position = vec4(aP.x / uStage.x * 2. - 1., 1. - aP.y / uStage.y * 2., 0., 1.);
  gl_PointSize = aP.z * uDpr;
}`;
const FS_PART = COMMON + `
in vec4 vC; in float vA; in float vS;
out vec4 o;
void main(){
  vec2 q = gl_PointCoord * 2. - 1.;
  if (vC.w > 0.5){
    float a = exp(-dot(q, q) * 2.5) * vA;
    o = vec4(vC.rgb * a, a);
    return;
  }
  // a jagged little flake, lit from the upper left
  float ang = atan(q.y, q.x);
  float r = 0.62 + 0.3 * hash12(vec2(floor((ang + 3.1416) * 1.1), vS * 17.));
  float a = smoothstep(r, r - 0.18, length(q)) * vA;
  float lit = 0.8 + 0.35 * clamp(-q.x * 0.5 - q.y * 0.7, -1., 1.);
  float spark = smoothstep(0.35, 0., length(q + vec2(0.3, 0.35))) * step(0.6, fract(vS * 7.3));
  vec3 c = vC.rgb * lit + spark * 0.5;
  o = vec4(min(c, vec3(1.)) * a, a);
}`;

// ---------------------------------------------------------------- glazes
const GLAZES = {
  porcelain: { base: [0.935, 0.94, 0.945], pool: [0.84, 0.885, 0.9], rim: [0.985, 0.98, 0.97], body: [0.975, 0.97, 0.955],
    ink: [0.12, 0.22, 0.56], craze: [0.62, 0.64, 0.66], gloss: 1, crazing: 0.07, speckle: 0, rimBreak: 0.35, pooling: 0.55, mottle: 0.03, inkK: 0.93, decor: 0 },
  celadon: { base: [0.62, 0.75, 0.655], pool: [0.34, 0.52, 0.43], rim: [0.72, 0.68, 0.56], body: [0.86, 0.84, 0.79],
    ink: [0.06, 0.14, 0.1], craze: [0.42, 0.52, 0.45], gloss: 0.95, crazing: 0.32, speckle: 0.04, rimBreak: 0.45, pooling: 0.85, mottle: 0.12, inkK: 0.92, decor: 0 },
  cobalt: { base: [0.945, 0.95, 0.96], pool: [0.8, 0.85, 0.92], rim: [0.985, 0.985, 0.98], body: [0.975, 0.97, 0.96],
    ink: [0.08, 0.17, 0.54], craze: [0.62, 0.65, 0.7], gloss: 1, crazing: 0.05, speckle: 0, rimBreak: 0.3, pooling: 0.5, mottle: 0.03, inkK: 0.95, decor: 1 },
  stoneware: { base: [0.85, 0.81, 0.72], pool: [0.75, 0.68, 0.57], rim: [0.56, 0.38, 0.24], body: [0.71, 0.65, 0.57],
    ink: [0.24, 0.16, 0.11], craze: [0.6, 0.55, 0.48], gloss: 0.16, crazing: 0, speckle: 1, rimBreak: 0.85, pooling: 0.45, mottle: 0.1, inkK: 0.9, decor: 0 },
  terracotta: { base: [0.62, 0.3, 0.18], pool: [0.5, 0.23, 0.14], rim: [0.78, 0.47, 0.31], body: [0.77, 0.43, 0.28],
    ink: [0.99, 0.95, 0.88], craze: [0.5, 0.26, 0.16], gloss: 0.32, crazing: 0, speckle: 0.25, rimBreak: 0.5, pooling: 0.5, mottle: 0.16, inkK: 0.93, decor: 0 },
};

let parseCtx = null;
function parseColor(css){
  if (Array.isArray(css)) return css.slice(0, 3);
  if (typeof css !== 'string' || !css.trim()) return null;
  if (!parseCtx) parseCtx = document.createElement('canvas').getContext('2d');
  parseCtx.fillStyle = '#010203';
  parseCtx.fillStyle = css;
  const v = parseCtx.fillStyle;
  if (v === '#010203' && !/^#?010203$/i.test(css.trim())) return null;
  let m = /^#([0-9a-f]{6})$/i.exec(v);
  if (m){ const n = parseInt(m[1], 16); return [(n >> 16) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]; }
  m = /rgba?\(([^)]+)\)/.exec(v);
  if (m){ const p = m[1].split(/[\s,/]+/).filter(Boolean).map(parseFloat); if (p.length > 3 && p[3] < 0.05) return null; return [p[0] / 255, p[1] / 255, p[2] / 255]; }
  return null;
}
function cssAlpha(css){ const m = /rgba?\(([^)]+)\)/.exec(css || ''); if (!m) return 1; const p = m[1].split(/[\s,/]+/).filter(Boolean).map(parseFloat); return p.length > 3 ? p[3] : 1; }
function pageColor(el){
  const layers = [];
  for (let e = el; e; e = e.parentElement){
    const bg = getComputedStyle(e).backgroundColor, c = parseColor(bg), a = cssAlpha(bg);
    if (c && a > 0.004){ layers.push([...c, a]); if (a >= 0.999) break; }
  }
  let rgb = [1, 1, 1];
  for (let i = layers.length - 1; i >= 0; i--){
    const [r, g, b, a] = layers[i];
    rgb = [r * a + rgb[0] * (1 - a), g * a + rgb[1] * (1 - a), b * a + rgb[2] * (1 - a)];
  }
  return rgb;
}
function rgb2hsl([r, g, b]){
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), l = (mx + mn) / 2;
  if (mx === mn) return [0, 0, l];
  const d = mx - mn, s = l > 0.5 ? d / (2 - mx - mn) : d / (mx + mn);
  const h = mx === r ? (g - b) / d + (g < b ? 6 : 0) : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
  return [h * 60, s, l];
}
function hsl2rgb([h, s, l]){
  const k = n => (n + h / 30) % 12, a = s * Math.min(l, 1 - l);
  const f = n => l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1));
  return [f(0), f(8), f(4)];
}
const mixc = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
// a glaze of any colour: pools deeper and more saturated, thins paler over the rim
function tinted(P, c){
  const [h, s, l] = rgb2hsl(c);
  P.base = c;
  P.pool = hsl2rgb([h, Math.min(1, s * 1.15 + 0.05), l * 0.7]);
  P.rim = mixc(c, P.body, 0.45);
  P.craze = hsl2rgb([h, s * 0.7, l * 0.55]);
  return P;
}
// WCAG contrast. The print is laid a little translucently into the glaze, so it is held to a margin over 4.5:1.
const lin = c => c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
const relLum = c => 0.2126 * lin(c[0]) + 0.7152 * lin(c[1]) + 0.0722 * lin(c[2]);
const contrast = (a, b) => { const x = relLum(a) + 0.05, y = relLum(b) + 0.05; return x > y ? x / y : y / x; };
const INK_CONTRAST = 5.4;
const shiftL = (c, d) => { const [h, s, l] = rgb2hsl(c); return hsl2rgb([h, s, clamp(l + d, 0, 1)]); };
// Make the label readable on the glaze. With moveGlaze (a glaze this code chose, not the author), the
// glaze first deepens or pales a little so the author's own text colour can stay; otherwise, or if that
// is not enough, the ink's lightness walks away from the glaze, keeping its hue.
function fitInk(P, prefer, moveGlaze){
  const ink0 = prefer || P.ink;
  P.ink = ink0;
  if (contrast(ink0, P.base) >= INK_CONTRAST) return P;
  const inkLight = relLum(ink0) > relLum(P.base);
  if (moveGlaze){
    for (let d = 0.02; d <= 0.2001; d += 0.02){
      const dd = inkLight ? -d : d, nb = shiftL(P.base, dd);
      if (contrast(ink0, nb) >= INK_CONTRAST){ P.base = nb; P.pool = shiftL(P.pool, dd); P.craze = shiftL(P.craze, dd); return P; }
    }
  }
  const [h, s, l] = rgb2hsl(ink0);
  for (const dir of inkLight ? [1, -1] : [-1, 1]){
    for (let v = l; v >= -1e-9 && v <= 1 + 1e-9; v += dir * 0.02){
      const c = hsl2rgb([h, s, clamp(v, 0, 1)]);
      if (contrast(c, P.base) >= INK_CONTRAST){ P.ink = c; return P; }
    }
  }
  P.ink = contrast([0, 0, 0], P.base) > contrast([1, 1, 1], P.base) ? [0.04, 0.035, 0.03] : [1, 0.99, 0.97];
  return P;
}
// 'auto': read the button's own colour and pick the pottery it looks most like
function autoGlaze(bg, fg){
  const [h, s, l] = rgb2hsl(bg);
  let name, P;
  if (s < 0.14 && l > 0.78){ name = 'porcelain'; P = Object.assign({}, GLAZES.porcelain); }
  else if (s < 0.2 && l > 0.3){ name = 'stoneware'; P = tinted(Object.assign({}, GLAZES.stoneware), mixc(bg, [0.85, 0.81, 0.72], 0.35)); P.rim = GLAZES.stoneware.rim; }
  else if (h >= 70 && h <= 175 && l > 0.3){ name = 'celadon'; P = tinted(Object.assign({}, GLAZES.celadon), hsl2rgb([h, Math.min(s, 0.35), clamp(l, 0.5, 0.75)])); }
  else if (h >= 5 && h <= 38 && s > 0.3 && l > 0.25 && l < 0.62){ name = 'terracotta'; P = tinted(Object.assign({}, GLAZES.terracotta), bg); P.rim = mixc(bg, [1, 0.9, 0.8], 0.25); }
  else if (l < 0.18){
    // a near-black glaze reads as tenmoku: glossy black that breaks to rust over the rim
    name = 'tenmoku'; P = tinted(Object.assign({}, GLAZES.porcelain), bg); P.rim = [0.5, 0.26, 0.12]; P.rimBreak = 0.9; P.body = [0.72, 0.66, 0.6]; P.crazing = 0;
  } else { name = 'glaze'; P = tinted(Object.assign({}, GLAZES.porcelain), bg); P.crazing = 0.1; }
  fitInk(P, fg, true);
  P.name = name;
  return P;
}
function resolveGlaze(spec, color, inkColor, auto){
  let name = typeof spec === 'string' ? spec : spec && spec.preset;
  const custom = spec && typeof spec === 'object' ? spec : null;
  let P;
  if (name === 'auto') P = autoGlaze(auto.bg, auto.fg);
  else {
    if (!GLAZES[name]) name = 'porcelain';
    P = Object.assign({}, GLAZES[name], { name });
  }
  const c = parseColor(color || (custom && custom.base));
  if (c){ const b = P.body; tinted(P, c); P.body = b; }
  if (custom){
    for (const k of ['pool', 'rim', 'body', 'ink']) { const v = parseColor(custom[k]); if (v) P[k] = v; }
    if (custom.gloss != null) P.gloss = clamp(+custom.gloss, 0, 1);
    if (custom.craze != null) P.crazing = clamp(+custom.craze, 0, 1);
    if (custom.speckle != null) P.speckle = clamp(+custom.speckle, 0, 1);
    if (custom.decor != null) P.decor = custom.decor ? 1 : 0;
  }
  const ic = parseColor(inkColor) || (custom && parseColor(custom.ink));
  // an explicit colour is the author's: only the ink moves for it. A preset or 'auto' glaze may deepen.
  fitInk(P, ic || (name === 'auto' ? auto.fg : P.ink), !c && !(custom && custom.base));
  return P;
}

// ---------------------------------------------------------------- audio (shared by every ceramic button)
let actx = null, actxOverride = null;
const ceramicAudio = { muted: false };
function audio(){
  if (actxOverride) return actxOverride;
  if (ceramicAudio.muted) return null;
  if (!actx){ try { actx = new (window.AudioContext || window.webkitAudioContext)(); } catch { return null; } }
  if (actx.state === 'suspended') actx.resume().catch(() => {});
  return actx;
}
function playBuf(dur, fill, gain, delay = 0){
  const a = audio(); if (!a) return;
  const sr = a.sampleRate, buf = a.createBuffer(1, Math.max(1, Math.round(sr * dur)), sr);
  const d = buf.getChannelData(0);
  fill(d, sr);
  // a ring still sounding when the buffer runs out would stop with a click
  const nf = Math.min(d.length, Math.round(sr * 0.03));
  for (let i = 0; i < nf; i++) d[d.length - 1 - i] *= i / nf;
  const src = a.createBufferSource(); src.buffer = buf;
  const g = a.createGain(); g.gain.value = gain;
  src.connect(g).connect(a.destination); src.start(a.currentTime + delay);
}
// One exponentially damped sine, by recurrence: y[n] = 2r cos(w) y[n-1] - r^2 y[n-2].
function mode(d, sr, at, f, tau, amp){
  const start = Math.round(at * sr); if (start >= d.length || f >= sr * 0.45 || amp <= 0) return;
  const w = TAU * f / sr, r = Math.exp(-1 / (tau * sr)), c = 2 * r * Math.cos(w), r2 = r * r, ph = Math.random() * TAU;
  let y1 = amp * Math.sin(ph - w) / r, y2 = amp * Math.sin(ph - 2 * w) / r2;
  const n = Math.min(d.length - start, Math.ceil(tau * sr * 7)), att = Math.max(1, Math.round(sr * 0.0003));
  for (let i = 0; i < n; i++){ const y = c * y1 - r2 * y2; y2 = y1; y1 = y; d[start + i] += i < att ? y * i / att : y; }
}
// a short burst of high-passed noise: the contact click of hard things meeting
function click(d, sr, at, ms, amp, hp = 2500){
  const start = Math.round(at * sr), n = Math.min(d.length - start, Math.round(ms * sr / 1000));
  if (n <= 0) return;
  const ah = Math.exp(-TAU * hp / sr);
  let hIn = 0, h = 0;
  for (let i = 0; i < n; i++){
    const x = Math.random() * 2 - 1;
    h = ah * (h + x - hIn); hIn = x;
    d[start + i] += h * amp * Math.exp(-5 * i / n);
  }
}
// A struck piece of fired clay rings at a handful of inharmonic plate modes; the higher ones die faster.
function strike(d, sr, at, f0, ratios, amp, tau){
  for (let k = 0; k < ratios.length; k++) mode(d, sr, at, f0 * ratios[k], tau / (1 + 0.45 * k) * rand(0.8, 1.2), amp * MODE_AMPS[k] * rand(0.7, 1.15));
  click(d, sr, at, rand(0.4, 1.1), amp * 0.9, 3000);
}
function normalize(d, peak){ let m = 0; for (let i = 0; i < d.length; i++) m = Math.max(m, Math.abs(d[i])); if (m > 0) for (let i = 0; i < d.length; i++) d[i] *= peak / m; }
const shardRatios = () => MODE_RATIOS.map((r, i) => i ? r * rand(0.95, 1.05) : 1);
let lastSnd = 0, sndBudget = 0;
function allowSnd(){
  const a = actxOverride || actx; if (!a) return false;
  const t = a.currentTime;
  sndBudget = Math.max(0, sndBudget - (t - lastSnd) * 30); lastSnd = t;
  if (sndBudget > 6) return false;
  sndBudget++; return true;
}
const sounds = {
  // a fingertip on the glaze: a dull, damped tock
  tap(){
    playBuf(0.12, (d, sr) => { strike(d, sr, 0, rand(1500, 1800), MODE_RATIOS, 0.5, 0.012); normalize(d, 0.8); }, 0.18);
  },
  // the smash: a sharp crack, the tile's ring cut short, every shard ringing out at its own pitch,
  // a granular crunch of bisque and a patter of crumbs
  smash(shards, size){
    playBuf(1.4, (d, sr) => {
      click(d, sr, 0, 3.5, 1.2, 1800);
      for (let i = 0; i < Math.min(d.length, sr * 0.05); i++){ const t = i / sr; d[i] += Math.sin(TAU * 160 * t) * Math.exp(-t / 0.014) * 0.35 * Math.min(1, t / 0.001); }
      strike(d, sr, 0.0005, rand(1700, 2100) * Math.sqrt(size), MODE_RATIOS, 0.55, 0.03);
      let t = 0.001;
      while (t < 0.07){ t += -Math.log(1 - Math.random()) / 700; click(d, sr, t, rand(0.3, 1.4), rand(0.1, 0.5) * Math.exp(-t / 0.03), rand(2000, 6000)); }
      for (const s of shards) strike(d, sr, rand(0.003, 0.022), s.f0, s.ratios, rand(0.22, 0.4), rand(0.09, 0.2));
      for (let i = 0, n = 10 + Math.floor(Math.random() * 6); i < n; i++){
        const at = 0.03 + Math.pow(Math.random(), 1.7) * 0.5;
        strike(d, sr, at, rand(6000, 11000), shardRatios(), rand(0.03, 0.09) * Math.exp(-at / 0.35), rand(0.02, 0.05));
      }
      normalize(d, 0.95);
    }, 0.55);
  },
  // a shard landing or knocking another: its own ring, louder and longer the harder the hit
  clink(s, v){
    if (!allowSnd()) return;
    const k = clamp(v / 400, 0.08, 1);
    playBuf(0.4, (d, sr) => { strike(d, sr, 0, s.f0 * rand(0.99, 1.01), s.ratios, 1, 0.05 + 0.13 * k); }, 0.09 * k + 0.02);
  },
  // the rattle of a shell shard rocking flat: each time its edge taps down
  tick(s, v){
    if (!allowSnd()) return;
    const k = clamp(v / 6, 0.1, 1);
    playBuf(0.12, (d, sr) => { strike(d, sr, 0, s.f0 * rand(0.98, 1.02), s.ratios, 1, 0.025 + 0.03 * k); }, 0.05 * k + 0.008);
  },
  // grit under a sliding shard
  skitter(s, v){
    if (!allowSnd()) return;
    const k = clamp(v / 250, 0.1, 1);
    playBuf(0.05, (d, sr) => { click(d, sr, 0, rand(0.6, 1.8), 1, 3500); mode(d, sr, 0, s.f0 * rand(1.4, 2.6), 0.006, 0.4); }, 0.035 * k);
  },
  // a press on the footprint while the pieces are down: a dull knock on the page and a rattle of the nearest pieces
  knock(shards){
    playBuf(0.3, (d, sr) => {
      click(d, sr, 0, 2.5, 0.8, 600);
      for (let i = 0; i < d.length && i < sr * 0.06; i++){ const t = i / sr; d[i] += Math.sin(TAU * 190 * t) * Math.exp(-t / 0.012) * 0.6 * Math.min(1, t / 0.001); }
      shards.slice(0, 4).forEach((s, i) => strike(d, sr, 0.03 + i * rand(0.012, 0.03), s.f0 * rand(0.98, 1.02), s.ratios, 0.25, 0.03));
      normalize(d, 0.85);
    }, 0.16);
  },
  // a shard seating back into place
  seat(s){
    if (!allowSnd()) return;
    playBuf(0.1, (d, sr) => { strike(d, sr, 0, s.f0 * 0.55, s.ratios, 1, 0.018); normalize(d, 0.8); }, 0.07);
  },
  // molten gold running into the seams: a soft warm hiss, then a bright bell-like gleam as it sets
  gold(){
    playBuf(0.9, (d, sr) => {
      let l = 0, b = 0;
      for (let i = 0; i < d.length; i++){
        const t = i / sr, env = Math.min(1, t / 0.12) * Math.exp(-Math.max(0, t - 0.15) / 0.22);
        const x = Math.random() * 2 - 1;
        l += (x - l) * 0.18; b += (l - b) * 0.02;
        d[i] += (l - b) * env * 0.5;
      }
    }, 0.1);
  },
  gleam(){
    playBuf(2.2, (d, sr) => {
      const bell = [1, 2.0, 2.76, 4.07, 5.4], amps = [1, 0.35, 0.3, 0.12, 0.08];
      for (const [f0, at, a] of [[1567.98, 0, 1], [2349.32, 0.11, 0.7]])
        for (let k = 0; k < bell.length; k++) mode(d, sr, at, f0 * bell[k] * (k ? rand(0.998, 1.002) : 1), 0.55 / (1 + k * 0.6), a * amps[k]);
      normalize(d, 0.9);
    }, 0.055);
  },
};
const awakeFx = new Set();
let suspendTimer = 0;
function audioAwake(fx, on){
  if (on){ awakeFx.add(fx); clearTimeout(suspendTimer); suspendTimer = 0; return; }
  awakeFx.delete(fx);
  if (awakeFx.size || suspendTimer || !actx) return;
  suspendTimer = setTimeout(() => {
    suspendTimer = 0;
    if (awakeFx.size || !actx || actx.state !== 'running') return;
    actx.suspend().catch(() => {});
  }, 3000);
}
// Render a named sound offline (for inspection): resolves to the Float32Array of samples.
ceramicAudio._render = async (name, args = [], seconds = 1.2, sr = 48000) => {
  const off = new OfflineAudioContext(1, Math.round(seconds * sr), sr);
  actxOverride = off;
  try { sounds[name](...args); } finally { actxOverride = null; }
  lastSnd = 0; sndBudget = 0;
  const buf = await off.startRendering();
  return buf.getChannelData(0);
};

// ---------------------------------------------------------------- geometry
function roundRect(w, h, r){
  const pts = [], seg = 8;
  const cs = [[w - r, r, -Math.PI / 2], [w - r, h - r, 0], [r, h - r, Math.PI / 2], [r, r, Math.PI]];
  for (const [cx, cy, a0] of cs) for (let i = 0; i <= seg; i++){
    const a = a0 + i / seg * Math.PI / 2;
    pts.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]);
  }
  // drop duplicate points where zero-radius corners collapse
  return pts.filter((p, i) => { const q = pts[(i + pts.length - 1) % pts.length]; return Math.hypot(p[0] - q[0], p[1] - q[1]) > 1e-6; });
}
function area(P){ let s = 0; for (let i = 0; i < P.length; i++){ const a = P[i], b = P[(i + 1) % P.length]; s += a[0] * b[1] - b[0] * a[1]; } return s / 2; }
function centroid(P){
  let cx = 0, cy = 0, A = 0;
  for (let i = 0; i < P.length; i++){ const a = P[i], b = P[(i + 1) % P.length], c = a[0] * b[1] - b[0] * a[1]; A += c; cx += (a[0] + b[0]) * c; cy += (a[1] + b[1]) * c; }
  return Math.abs(A) < 1e-9 ? P[0].slice() : [cx / (3 * A), cy / (3 * A)];
}
// Sutherland-Hodgman against the half-plane dot(p - o, n) <= 0. F[i] is the flag of edge P[i] -> P[i+1]:
// 0 for the button's outline, j+1 for the cut shared with Voronoi cell j.
function clip(P, F, o, n, flag){
  const outP = [], outF = [];
  const side = p => (p[0] - o[0]) * n[0] + (p[1] - o[1]) * n[1];
  for (let i = 0; i < P.length; i++){
    const a = P[i], b = P[(i + 1) % P.length], f = F[i];
    const sa = side(a), sb = side(b), ina = sa <= 0, inb = sb <= 0;
    const I = () => { const t = sa / (sa - sb); return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]; };
    if (ina && inb){ outP.push(a); outF.push(f); }
    else if (ina && !inb){ outP.push(a); outF.push(f); outP.push(I()); outF.push(flag); }
    else if (!ina && inb){ outP.push(I()); outF.push(f); }
  }
  return [outP, outF];
}
function voronoi(P, seeds){
  const cells = [];
  for (let i = 0; i < seeds.length; i++){
    let cp = P, cf = P.map(() => 0);
    for (let j = 0; j < seeds.length && cp.length >= 3; j++){
      if (i === j) continue;
      const s = seeds[i], t = seeds[j];
      [cp, cf] = clip(cp, cf, [(s[0] + t[0]) / 2, (s[1] + t[1]) / 2], [t[0] - s[0], t[1] - s[1]], j + 1);
    }
    cells.push(cp.length >= 3 && Math.abs(area(cp)) > 1 ? [cp, cf] : null);
  }
  return cells;
}
// Replace every straight Voronoi cut with a curved, shell-like fracture. Both cells that share a cut
// build the identical curve from the same endpoints, so the pieces still fit together exactly.
function curvePoints(a, b, p){
  const key = q => Math.round(q[0] * 1000) * 1e7 + Math.round(q[1] * 1000);
  const flip = key(a) > key(b), A = flip ? b : a, B = flip ? a : b;
  const ex = B[0] - A[0], ey = B[1] - A[1], L = Math.hypot(ex, ey);
  if (L < 1e-6) return [];
  const nx = -ey / L, ny = ex / L, k = Math.max(1, Math.ceil(L / 2.5)), pts = [];
  for (let s = 1; s < k; s++){
    const t = s / k, env = Math.sin(Math.PI * t);
    const off = p.amp * L * (p.c * env + p.s * Math.sin(TAU * t)) + p.rip * Math.sin(Math.PI * t * p.m + p.ph) * env;
    pts.push([A[0] + ex * t + nx * off, A[1] + ey * t + ny * off]);
  }
  return flip ? pts.reverse() : pts;
}
function curveCells(cells, params){
  return cells.map((cell, i) => {
    if (!cell) return null;
    const [P, F] = cell, out = [], outF = [];
    for (let k = 0; k < P.length; k++){
      const a = P[k], b = P[(k + 1) % P.length], f = F[k];
      out.push(a); outF.push(f ? 1 : 0);
      if (f){
        const j = f - 1, key = Math.min(i, j) + ',' + Math.max(i, j);
        for (const q of curvePoints(a, b, params.get(key))){ out.push(q); outF.push(1); }
      }
    }
    return [out, outF];
  });
}
// every fan triangle from the centroid must keep the polygon's winding, or the fan folds over itself
function fanOK(P){
  const c = centroid(P), s = Math.sign(area(P));
  for (let i = 0; i < P.length; i++){
    const a = P[i], b = P[(i + 1) % P.length];
    if (((a[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (a[1] - c[1])) * s <= 0.05) return false;
  }
  return true;
}
function hull(pts){
  const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const q of p){ while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
  for (let i = p.length - 1; i >= 0; i--){ const q = p[i]; while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}
// minimum translation that pushes convex B out of convex A, or null
function sat(A, B){
  let depth = Infinity, ax = 0, ay = 0;
  for (const P of [A, B]){
    for (let i = 0; i < P.length; i++){
      const a = P[i], b = P[(i + 1) % P.length];
      let nx = b[1] - a[1], ny = a[0] - b[0];
      const l = Math.hypot(nx, ny); if (l < 1e-6) continue;
      nx /= l; ny /= l;
      let minA = Infinity, maxA = -Infinity, minB = Infinity, maxB = -Infinity;
      for (const p of A){ const v = p[0] * nx + p[1] * ny; if (v < minA) minA = v; if (v > maxA) maxA = v; }
      for (const p of B){ const v = p[0] * nx + p[1] * ny; if (v < minB) minB = v; if (v > maxB) maxB = v; }
      const o = Math.min(maxA - minB, maxB - minA);
      if (o <= 0) return null;
      if (o < depth){ depth = o; const sgn = (maxA - minB) < (maxB - minA) ? 1 : -1; ax = nx * sgn; ay = ny * sgn; }
    }
  }
  return [ax, ay, depth];
}
function pointInPoly(P, x, y){
  let inside = false;
  for (let i = 0, j = P.length - 1; i < P.length; j = i++){
    const [xi, yi] = P[i], [xj, yj] = P[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
function distToPoly(P, x, y){
  let best = Infinity;
  for (let i = 0; i < P.length; i++){
    const a = P[i], b = P[(i + 1) % P.length], ex = b[0] - a[0], ey = b[1] - a[1];
    const t = clamp(((x - a[0]) * ex + (y - a[1]) * ey) / (ex * ex + ey * ey || 1), 0, 1);
    best = Math.min(best, Math.hypot(x - a[0] - ex * t, y - a[1] - ey * t));
  }
  return best;
}

// ---------------------------------------------------------------- component
const INTERACTIVE = 'a,button,input,select,textarea,label,summary,[role=button],[tabindex],[contenteditable]';
// what shards must not settle over: controls, labelled widgets and pictures (text is found separately)
const BLOCKING = INTERACTIVE + ',[aria-label],[role],img,svg,video,canvas,iframe,picture';
// Every ceramic button on the page, so the pieces of one keep off another's tile and its debris.
const instances = new Set();

function attachCeramic(button, opts = {}){
  const canvas = document.createElement('canvas');
  canvas.setAttribute('aria-hidden', 'true');
  Object.assign(canvas.style, { position: 'absolute', pointerEvents: 'none', left: '0px', top: '0px' });
  // The stage reaches past the button on every side; clipped to the document by this layer, it can
  // never make the page scroll further than it already does.
  const host = document.createElement('div');
  host.setAttribute('aria-hidden', 'true');
  Object.assign(host.style, { position: 'absolute', left: '0px', top: '0px', width: '0px', height: '0px', overflow: 'clip', pointerEvents: 'none', zIndex: opts.zIndex ?? 10, margin: '0', padding: '0', border: '0' });
  host.appendChild(canvas);
  const gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: true, antialias: true, depth: false, stencil: false });
  if (!gl) return { supported: false, detach(){}, setGlaze(){}, restore(){} };
  const sound = opts.sound !== false;
  const self = {};
  const snd = (name, ...a) => { if (sound) sounds[name](...a); };
  const RESTORE = opts.restoreAfter ?? 1.4;
  const keepGold = !!opts.keepGold;

  const compile = (type, src) => {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS) && !gl.isContextLost()) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  const program = (vs, fs, attrs) => {
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, vs)); gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
    attrs.forEach((name, i) => gl.bindAttribLocation(p, i, name));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS) && !gl.isContextLost()) throw new Error(gl.getProgramInfoLog(p));
    const u = {};
    for (let i = 0, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS) || 0; i < n; i++){
      const info = gl.getActiveUniform(p, i);
      u[info.name.replace(/\[0\]$/, '')] = gl.getUniformLocation(p, info.name);
    }
    return { p, u };
  };
  let P, vao, vbuf, pvao, pbuf, textTex, seamTex;
  function initGL(){
    P = { main: program(VS, FS, ['aPos', 'aHome', 'aEdge', 'aN', 'aR', 'aK']), part: program(VS_PART, FS_PART, ['aP', 'aC', 'aS']) };
    vbuf = gl.createBuffer(); vao = gl.createVertexArray();
    gl.bindVertexArray(vao); gl.bindBuffer(gl.ARRAY_BUFFER, vbuf);
    [2, 2, 4, 3, 2, 3].reduce((off, size, l) => { gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, size, gl.FLOAT, false, STRIDE * 4, off * 4); return off + size; }, 0);
    pbuf = gl.createBuffer(); pvao = gl.createVertexArray();
    gl.bindVertexArray(pvao); gl.bindBuffer(gl.ARRAY_BUFFER, pbuf);
    [4, 4, 1].reduce((off, size, l) => { gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, size, gl.FLOAT, false, 9 * 4, off * 4); return off + size; }, 0);
    gl.bindVertexArray(null);
    textTex = gl.createTexture(); seamTex = gl.createTexture();
    for (const t of [textTex, seamTex]){
      gl.bindTexture(gl.TEXTURE_2D, t);
      for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
    }
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([255, 128, 128, 255]));
  }
  initGL();

  // ---- page
  const saved = button.getAttribute('style');
  const cs0 = getComputedStyle(button);
  const label = button.textContent.trim();
  const font = `${cs0.fontStyle} ${cs0.fontWeight} ${cs0.fontSize} ${cs0.fontFamily}`;
  const autoInfo = { bg: parseColor(cs0.backgroundColor) || pageColor(button.parentElement), fg: parseColor(cs0.color) };
  let mat = resolveGlaze(opts.glaze || 'porcelain', opts.color, opts.inkColor, autoInfo);
  // The tap highlight would paint the UA's flat tint over the tile and its flying pieces.
  const hideButton = () => Object.assign(button.style, { background: 'transparent', borderColor: 'transparent', color: 'transparent', boxShadow: 'none', textShadow: 'none', webkitTapHighlightColor: 'transparent', position: cs0.position === 'static' ? 'relative' : cs0.position });
  const showButton = () => { if (saved === null) button.removeAttribute('style'); else button.setAttribute('style', saved); };
  hideButton();
  document.body.appendChild(host);

  // ---- geometry (CSS px); button-local "home" coords, stage = home + M
  let bw = 0, bh = 0, M = 0, stageW = 0, stageH = 0, dpr = 1, radius = 0, T = 4;
  let slabPoly = [];
  let labelBox = null;              // home px: where the print is, so the lamp's reflection is kept off it
  let lim = [0, 0, 0, 0];           // stage px: x0, y0, x1, y1 of the visible stage
  let dirty = true;
  function buildText(){
    const c = document.createElement('canvas');
    const s = dpr * 2;
    c.width = Math.max(1, Math.round(bw * s)); c.height = Math.max(1, Math.round(bh * s));
    const x = c.getContext('2d');
    x.fillStyle = '#000'; x.fillRect(0, 0, c.width, c.height);
    x.font = font.replace(/(\d+(\.\d+)?)px/, (_, v) => `${parseFloat(v) * s}px`);
    x.textAlign = 'center'; x.textBaseline = 'middle';
    // underglaze colour bleeds a touch into the glaze as it fires
    x.filter = `blur(${0.3 * s}px)`;
    x.fillStyle = '#fff';
    const ty = c.height / 2 + parseFloat(cs0.fontSize) * s * 0.04;
    x.fillText(label, c.width / 2, ty);
    const mt = x.measureText(label), asc = mt.actualBoundingBoxAscent || parseFloat(cs0.fontSize) * s * 0.36, desc = mt.actualBoundingBoxDescent || parseFloat(cs0.fontSize) * s * 0.1;
    labelBox = label ? [bw / 2 - mt.width / s / 2, (ty - asc) / s, bw / 2 + mt.width / s / 2, (ty + desc) / s] : null;
    gl.bindTexture(gl.TEXTURE_2D, textTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, c);
  }

  let placedAt = '';
  let eye = [0, 0];
  let origin = [0, 0];              // the stage's top-left in document coords
  // The button's rect in document coords, dropped whenever something may have moved it, so pointer
  // moves anywhere on the page can be answered without a layout read.
  let docRect = null;
  function btnRect(){
    if (!docRect){ const r = button.getBoundingClientRect(); docRect = [r.left + scrollX, r.top + scrollY, r.width, r.height]; }
    return { left: docRect[0] - scrollX, top: docRect[1] - scrollY, width: docRect[2], height: docRect[3] };
  }
  function place(r = button.getBoundingClientRect()){
    // the viewer sits over the middle of the viewport, but never so far off that the highlight leaves the tile
    const ex = clamp(innerWidth / 2 - (r.left + r.width / 2), -160, 160), ey = clamp(innerHeight / 2 - (r.top + r.height / 2), -50, 50);
    eye = [M + bw / 2 + ex, M + bh / 2 + ey];
    const hr = host.getBoundingClientRect(), de = document.documentElement, body = document.body;
    const docBottom = Math.max(innerHeight, body.getBoundingClientRect().bottom + (parseFloat(getComputedStyle(body).marginBottom) || 0)) + scrollY;
    const left = `${r.left - hr.left - M}px`, top = `${r.top - hr.top - M}px`;
    const hw = `${Math.max(0, de.clientWidth - (hr.left + scrollX))}px`, hh = `${Math.max(0, docBottom - (hr.top + scrollY))}px`;
    const key = left + top + hw + hh;
    if (key !== placedAt){ placedAt = key; canvas.style.left = left; canvas.style.top = top; host.style.width = hw; host.style.height = hh; }
    // the part of the stage that is actually on the page: pieces are kept inside it, not just inside the canvas
    const cl = r.left - hr.left - M, ct = r.top - hr.top - M;
    lim = [Math.max(0, -cl), Math.max(0, -ct), Math.min(stageW, parseFloat(hw) - cl), Math.min(stageH, parseFloat(hh) - ct)];
    origin = [r.left + scrollX - M, r.top + scrollY - M];
    docRect = [r.left + scrollX, r.top + scrollY, r.width, r.height];
    refreshBlockers(r);
    dirty = true;
  }
  // What shards are kept off: other controls, pictures and text inside the stage, so a settled piece never
  // hides a link, a stepper or a line of copy. Found by hit-testing a coarse grid at the break (text by its
  // line boxes, not its whole block), and re-measured whenever the stage moves.
  let blockEls = [], blockText = [], blocks = [], foreign = [], allBlocks = [];
  const range = document.createRange();
  function scanBlockers(){
    const r = button.getBoundingClientRect(), x0 = r.left - M, y0 = r.top - M, found = new Set(), texts = new Set();
    for (let y = 4; y < stageH; y += 10) for (let x = 4; x < stageW; x += 10){
      const px = x0 + x, py = y0 + y;
      if (px < 0 || py < 0 || px >= innerWidth || py >= innerHeight) continue;
      const el = document.elementFromPoint(px, py);
      if (!el || el === button || button.contains(el) || host.contains(el)) continue;
      const hit = el.closest && el.closest(BLOCKING);
      if (hit && !hit.contains(button)){ found.add(hit); continue; }
      for (const n of el.childNodes) if (n.nodeType === 3 && n.data.trim()) texts.add(n);
    }
    blockEls = [...found]; blockText = [...texts];
    refreshBlockers(r);
  }
  function refreshBlockers(r = button.getBoundingClientRect()){
    const x0 = r.left - M, y0 = r.top - M;
    blocks = [];
    const add = (b, pad) => {
      if (!b.width || !b.height) return;
      const q = [b.left - x0 - pad, b.top - y0 - pad, b.right - x0 + pad, b.bottom - y0 + pad];
      if (q[2] > 0 && q[3] > 0 && q[0] < stageW && q[1] < stageH) blocks.push(q);
    };
    for (const el of blockEls) if (el.isConnected) add(el.getBoundingClientRect(), 6);
    for (const t of blockText){
      if (!t.isConnected) continue;
      range.selectNodeContents(t);
      for (const b of range.getClientRects()) add(b, 4);
    }
    allBlocks = foreign.length ? blocks.concat(foreign) : blocks;
  }
  // document rects this button covers: its tile, and while broken each loose piece
  function occupied(){
    const out = [];
    if (docRect) out.push([docRect[0], docRect[1], docRect[0] + docRect[2], docRect[1] + docRect[3]]);
    if (!joined && !lost) for (const s of shards){
      if (!s.world || s.held) continue;
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const p of s.world){ if (p[0] < x0) x0 = p[0]; if (p[0] > x1) x1 = p[0]; if (p[1] < y0) y0 = p[1]; if (p[1] > y1) y1 = p[1]; }
      out.push([origin[0] + x0, origin[1] + y0, origin[0] + x1, origin[1] + y1]);
    }
    return out;
  }
  const peer = { occupied };
  function gatherForeign(){
    foreign = [];
    for (const o of instances){
      if (o === peer) continue;
      for (const q of o.occupied()){
        const b = [q[0] - origin[0] - 3, q[1] - origin[1] - 3, q[2] - origin[0] + 3, q[3] - origin[1] + 3];
        if (b[2] > 0 && b[3] > 0 && b[0] < stageW && b[1] < stageH) foreign.push(b);
      }
    }
    allBlocks = foreign.length ? blocks.concat(foreign) : blocks;
  }
  let lastRect = null;
  function checkPlace(){
    if (lost) return false;
    const r = button.getBoundingClientRect();
    docRect = [r.left + scrollX, r.top + scrollY, r.width, r.height];
    const L = lastRect;
    if (L && L.left === r.left && L.top === r.top && L.width === r.width && L.height === r.height && L.sx === scrollX && L.sy === scrollY) return false;
    lastRect = { left: r.left, top: r.top, width: r.width, height: r.height, sx: scrollX, sy: scrollY };
    if (Math.abs(r.width - bw) > 0.5 || Math.abs(r.height - bh) > 0.5) resize();
    else { place(r); wake(0.05); }
    return true;
  }
  function resize(){
    const r = button.getBoundingClientRect();
    bw = r.width; bh = r.height;
    M = Math.round(opts.margin ?? Math.max(100, bh * 1.7));
    stageW = bw + 2 * M; stageH = bh + 2 * M;
    dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(stageW * dpr); canvas.height = Math.round(stageH * dpr);
    Object.assign(canvas.style, { width: `${stageW}px`, height: `${stageH}px` });
    radius = Math.min(parseFloat(getComputedStyle(button).borderTopLeftRadius) || 0, bw / 2, bh / 2);
    T = clamp(bh * 0.07, 3, 5);
    slabPoly = roundRect(bw, bh, Math.max(radius, 1.5));
    buildText();
    toRest();
    lamp = null; lampT = null;
    place();
    wake(0.1);
  }

  // ---- state
  let state = 'rest';                 // rest | broken | returning | mending
  let lost = false;
  const savedTouch = button.style.touchAction;
  let press = 0, pressTarget = 0, flash = 0;
  let impact = [0, 0];
  let shards = [], parts = [], joined = true, kept = false;
  let idleT = 0, mendT = 0, goldMax = 1, gleamPlayed = false;
  let lamp = null, lampT = null;
  let restoreTimer = 0;
  // where the softbox hangs with no cursor about: its reflection sits on the upper left of the tile
  const defLamp = () => [M + bw * 0.24, M + bh * 0.32];
  const lampSize = () => [clamp(bw * 0.12, 16, 40), clamp(bh * 0.12, 4, 8)];
  // Where the softbox's reflection falls on the flat face is known exactly (the mirror point between the lamp
  // and the eye), so the lamp is moved until the reflection sits above or beside the print, never on it.
  const KL = LAMP_H / EYE_H;
  function fitLamp(L){
    if (!labelBox) return L;
    const [lw, lh] = lampSize(), sx = lw / (1 + KL) + 1.5, sy = lh / (1 + KL) + 1.5;
    let cx = (L[0] + KL * eye[0]) / (1 + KL), cy = (L[1] + KL * eye[1]) / (1 + KL);
    const b = [labelBox[0] + M - 2, labelBox[1] + M - 2, labelBox[2] + M + 2, labelBox[3] + M + 2];
    if (cx + sx > b[0] && cx - sx < b[2] && cy + sy > b[1] && cy - sy < b[3]){
      const up = b[1] - sy, lf = b[0] - sx, rt = b[2] + sx, cand = [];
      if (up - sy >= M - 4) cand.push([cx, up]);
      if (lf - sx >= M + 3) cand.push([lf, cy]);
      if (rt + sx <= M + bw - 3) cand.push([rt, cy]);
      if (!cand.length) cand.push([cx, up]);
      let best = cand[0], bd = Infinity;
      for (const c of cand){ const d = Math.hypot(c[0] - cx, c[1] - cy); if (d < bd){ bd = d; best = c; } }
      [cx, cy] = best;
    }
    return [cx * (1 + KL) - KL * eye[0], cy * (1 + KL) - KL * eye[1]];
  }

  function mkShard(poly, flags){
    const hc = centroid(poly);
    const local = poly.map(p => [p[0] - hc[0], p[1] - hc[1]]);
    const A = Math.abs(area(poly));
    // distance from the centroid to each edge's line, for the fan's per-edge distance field
    const dC = local.map((a, i) => { const b = local[(i + 1) % local.length], ex = b[0] - a[0], ey = b[1] - a[1], l = Math.hypot(ex, ey) || 1; return Math.abs(a[0] * ey - a[1] * ex) / l; });
    const sgn = Math.sign(area(poly)) || 1;
    const nrm = local.map((a, i) => { const b = local[(i + 1) % local.length], ex = b[0] - a[0], ey = b[1] - a[1], l = Math.hypot(ex, ey) || 1; return [ey / l * sgn, -ex / l * sgn]; });
    const hl = hull(local);
    return { hc, local, flags, dC, nrm, sgn, hull: hl, area: A, rad: Math.max(...local.map(p => Math.hypot(p[0], p[1]))),
      px: hc[0] + M, py: hc[1] + M, z: 0, vx: 0, vy: 0, vz: 0, ang: 0, va: 0, tx: 0, ty: 0, wx: 0, wy: 0,
      air: false, held: false, f0: clamp(2700 * Math.sqrt(1400 / Math.max(A, 60)), 2000, 7800) * rand(0.96, 1.04), ratios: shardRatios(),
      lastSnd: 0, skid: 0, ret: null, alpha: 1, world: null };
  }
  function toRest(keepSeams = false){
    state = 'rest'; joined = true; kept = false; idleT = 0; mendT = 0;
    if (!keepSeams) clearSeams();
    clearTimeout(restoreTimer); restoreTimer = 0;
    button.style.touchAction = savedTouch;
    canvas.style.pointerEvents = 'none'; canvas.style.cursor = '';
    impact = [bw / 2, bh / 2];
    shards = [mkShard(slabPoly, slabPoly.map(() => 0))];
    parts = parts.filter(p => p.kind === 1);
  }

  // ---- the break
  // Seeds laid out by angle on widening rings around the blow: the first ring's cells are small wedges
  // meeting at the impact, and each ring further out is sparser, so the pieces grow with distance. On a
  // long tile, seed pairs above and below the axis leave a crack running out along it, as a real strike does.
  function fracture(cx, cy){
    const N = clamp(Math.round(bw * bh / 1500), 5, 12);
    const r0 = clamp(Math.min(bw, bh) * 0.3, 7, 20);
    const far = Math.max(cx, bw - cx, cy, bh - cy);
    const seeds = [];
    const inside = (x, y) => x > 2 && y > 2 && x < bw - 2 && y < bh - 2;
    const free = (x, y, d) => seeds.every(q => Math.hypot(q[0] - x, q[1] - y) > d);
    for (let k = 0, r = r0; r < far + r0 * 0.5 && seeds.length < N; k++, r *= rand(1.75, 2.1)){
      // each ring gets fewer seeds than the last, so there are always some left for the far pieces
      const cap = k === 0 ? 4 : k === 1 ? 3 : 2;
      const m = k === 0 ? cap : 6, off = rand(0, TAU), ring = [];
      for (let i = 0; i < m; i++){
        const a = off + (i + rand(-0.2, 0.2)) * TAU / m, rr = r * rand(0.85, 1.15);
        ring.push([cx + Math.cos(a) * rr, cy + Math.sin(a) * rr * (k ? 1 : 0.85)]);
      }
      // on a long tile most of a ring falls off it: one seed on each side keeps the break going along it,
      // now and then a pair above and below the axis, which leaves a crack running out from the blow
      if (r >= bh * 0.5) for (const side of [-1, 1]){
        const x = cx + side * r * rand(0.9, 1.1);
        if (x < 3 || x > bw - 3) continue;
        const pair = Math.random() < 0.3, y1 = cy + (pair ? rand(0.18, 0.32) * bh * (Math.random() < 0.5 ? -1 : 1) : rand(-0.2, 0.2) * bh);
        ring.unshift([x, clamp(y1, 4, bh - 4)]);
        if (pair) ring.unshift([x + rand(-0.12, 0.12) * r, clamp(2 * cy - y1 + rand(-4, 4), 4, bh - 4)]);
      }
      const side = ring.slice(0, ring.length - m), rest = ring.slice(ring.length - m);
      for (let i = rest.length - 1; i > 0; i--){ const j = Math.floor(Math.random() * (i + 1)); [rest[i], rest[j]] = [rest[j], rest[i]]; }
      let took = 0;
      for (const [x, y] of side.concat(rest)){
        if (took >= Math.max(cap, side.length) || seeds.length >= N) break;
        if (inside(x, y) && free(x, y, Math.max(6, r * 0.45))){ seeds.push([x, y]); took++; }
      }
    }
    for (let tries = 0; seeds.length < 3 && tries < 200; tries++){ const x = rand(3, bw - 3), y = rand(3, bh - 3); if (free(x, y, 8)) seeds.push([x, y]); }
    const cells = voronoi(slabPoly, seeds);
    const params = new Map();
    for (let i = 0; i < seeds.length; i++) for (let j = i + 1; j < seeds.length; j++)
      params.set(i + ',' + j, { amp: 1, c: (Math.random() < 0.5 ? -1 : 1) * rand(0.1, 0.22), s: rand(-0.06, 0.06), rip: rand(0.25, 0.7), m: rand(2, 5), ph: rand(0, TAU) });
    let curved;
    for (let pass = 0; pass < 6; pass++){
      curved = curveCells(cells, params);
      let ok = true;
      curved.forEach((c, i) => {
        if (!c || fanOK(c[0])) return;
        ok = false;
        for (const [k, p] of params){ const [a, b] = k.split(',').map(Number); if (a === i || b === i){ p.amp *= 0.5; p.rip *= 0.5; if (pass >= 4){ p.amp = 0; p.rip = 0; } } }
      });
      if (ok) break;
    }
    const out = curved.filter(Boolean).map(([p, f]) => mkShard(p, f));
    buildSeams(out);
    return out;
  }
  // Distance from every point of the tile to its nearest crack, and the way to it, baked into a texture in
  // home coords. The glaze shader reads the seam's scars, lip and gold from it: interpolating a distance
  // across the fan triangles instead gives wedge-shaped artefacts wherever a triangle is thin.
  let seamW = 1, seamH = 1, seamSc = 1;
  function buildSeams(list){
    const sc = seamSc = Math.min(2, Math.max(1, dpr));
    const W = seamW = Math.max(1, Math.min(2048, Math.ceil(bw * sc))), H = seamH = Math.max(1, Math.min(2048, Math.ceil(bh * sc)));
    const dist = new Float32Array(W * H).fill(SEAM_R), vx = new Float32Array(W * H), vy = new Float32Array(W * H);
    const seen = new Set();
    for (const sh of list){
      const n = sh.local.length;
      for (let i = 0; i < n; i++){
        if (!sh.flags[i]) continue;
        const a = [sh.hc[0] + sh.local[i][0], sh.hc[1] + sh.local[i][1]], b = [sh.hc[0] + sh.local[(i + 1) % n][0], sh.hc[1] + sh.local[(i + 1) % n][1]];
        const key = a[0] < b[0] || (a[0] === b[0] && a[1] < b[1]) ? `${a[0].toFixed(2)},${a[1].toFixed(2)},${b[0].toFixed(2)},${b[1].toFixed(2)}` : `${b[0].toFixed(2)},${b[1].toFixed(2)},${a[0].toFixed(2)},${a[1].toFixed(2)}`;
        if (seen.has(key)) continue;
        seen.add(key);
        const ex = b[0] - a[0], ey = b[1] - a[1], ll = ex * ex + ey * ey || 1;
        const i0 = Math.max(0, Math.floor((Math.min(a[0], b[0]) - SEAM_R) * sc)), i1 = Math.min(W - 1, Math.ceil((Math.max(a[0], b[0]) + SEAM_R) * sc));
        const j0 = Math.max(0, Math.floor((Math.min(a[1], b[1]) - SEAM_R) * sc)), j1 = Math.min(H - 1, Math.ceil((Math.max(a[1], b[1]) + SEAM_R) * sc));
        for (let j = j0; j <= j1; j++){
          const py = (j + 0.5) / sc;
          for (let ii = i0; ii <= i1; ii++){
            const px = (ii + 0.5) / sc, t = clamp(((px - a[0]) * ex + (py - a[1]) * ey) / ll, 0, 1);
            const qx = a[0] + ex * t - px, qy = a[1] + ey * t - py, d = Math.hypot(qx, qy), o = j * W + ii;
            if (d < dist[o]){
              dist[o] = d;
              // right on the line the way to it is undefined: use the segment's normal
              if (d > 1e-3){ vx[o] = qx / d; vy[o] = qy / d; } else { const l = Math.sqrt(ll); vx[o] = -ey / l; vy[o] = ex / l; }
            }
          }
        }
      }
    }
    const px = new Uint8Array(W * H * 4);
    for (let o = 0; o < W * H; o++){
      px[o * 4] = Math.round(dist[o] / SEAM_R * 255); px[o * 4 + 1] = Math.round((vx[o] * 0.5 + 0.5) * 255);
      px[o * 4 + 2] = Math.round((vy[o] * 0.5 + 0.5) * 255); px[o * 4 + 3] = 255;
    }
    gl.bindTexture(gl.TEXTURE_2D, seamTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, W, H, 0, gl.RGBA, gl.UNSIGNED_BYTE, px);
  }
  function clearSeams(){
    seamW = seamH = 1; seamSc = 1;
    gl.bindTexture(gl.TEXTURE_2D, seamTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([255, 128, 128, 255]));
  }
  // How far a piece launched at v would travel, a low hop plus the long early slide, so launches can be
  // matched to the room each piece has.
  const travel = (v, vz) => v * (2 * vz / G) * 1.3 + (v * 0.72) ** 2 / (2 * (FRICTION_LO + 30));
  // room along a direction before the stage edge, another control, text, or another tile's debris
  function room(s, dx, dy){
    const e = EDGE + s.rad + 4;
    let t = Infinity;
    if (dx > 1e-6) t = Math.min(t, (lim[2] - e - s.px) / dx); else if (dx < -1e-6) t = Math.min(t, (lim[0] + e - s.px) / dx);
    if (dy > 1e-6) t = Math.min(t, (lim[3] - e - s.py) / dy); else if (dy < -1e-6) t = Math.min(t, (lim[1] + e - s.py) / dy);
    const pr = s.rad * 0.8;
    for (const [ax, ay, bx, by] of allBlocks){
      // slab test of the ray against the block grown by the piece's size
      let t0 = 0, t1 = Infinity;
      for (const [o, d, lo, hi] of [[s.px, dx, ax - pr, bx + pr], [s.py, dy, ay - pr, by + pr]]){
        if (Math.abs(d) < 1e-9){ if (o < lo || o > hi){ t0 = Infinity; break; } continue; }
        let u0 = (lo - o) / d, u1 = (hi - o) / d; if (u0 > u1) [u0, u1] = [u1, u0];
        t0 = Math.max(t0, u0); t1 = Math.min(t1, u1);
      }
      if (t0 <= t1) t = Math.min(t, t0);
    }
    return Math.max(0, t);
  }
  let smashT = 0, smashWall = -1e9;
  function smashAt(cx, cy){
    impact = [cx, cy];
    shards = fracture(cx, cy);
    joined = false; kept = false; state = 'broken'; idleT = 0; smashT = simT; smashWall = performance.now();
    scanBlockers(); gatherForeign();
    button.style.touchAction = 'none';
    const R = Math.max(bw, bh);
    for (const s of shards){
      const dx = s.hc[0] - cx, dy = s.hc[1] - cy, d = Math.hypot(dx, dy) || 1;
      const base = Math.atan2(dy / d * 0.75 + rand(-0.15, 0.15), dx / d);
      const fall = Math.exp(-d / (R * 0.55));
      s.vz = rand(60, 130) * (0.45 + 0.7 * fall);
      let v = rand(80, 150) * (0.4 + fall) + 25;
      const want = travel(v, s.vz);
      // steer toward open floor: the blow's own direction first, a wider swerve only if it is blocked
      let best = base, bestScore = -Infinity, bestRoom = 0;
      for (const da of [0, 0.3, -0.3, 0.65, -0.65, 1.1, -1.1]){
        const a = base + da, open = room(s, Math.cos(a), Math.sin(a));
        const score = Math.min(open, want) - Math.abs(da) * 14;
        if (score > bestScore){ bestScore = score; best = a; bestRoom = open; }
      }
      const avail = Math.min(bestRoom * 0.6, 34 + s.rad * 1.2);
      while (v > 5 && travel(v, s.vz) > avail) v *= 0.88;
      s.vx = Math.cos(best) * v; s.vy = Math.sin(best) * v;
      s.va = rand(-4, 4) * (0.3 + fall);
      // long pieces rock and wobble, they do not cartwheel up on end
      const tum = clamp(26 / s.rad, 0.25, 1);
      s.wx = rand(-9, 9) * (0.4 + fall) * tum; s.wy = rand(-9, 9) * (0.4 + fall) * tum;
      s.air = true; s.skid = rand(0.02, 0.08);
    }
    // crumbs of glaze and body, and a puff of bisque dust from the point of impact
    const ix = cx + M, iy = cy + M;
    for (let i = 0, n = 10 + Math.floor(Math.random() * 6); i < n; i++){
      const a = rand(0, TAU), sp = rand(40, 170);
      parts.push({ kind: 0, x: ix, y: iy, z: rand(0, 3), vx: Math.cos(a) * sp, vy: Math.sin(a) * sp * 0.8, vz: rand(60, 200), size: rand(1.8, 3.6),
        col: Math.random() < 0.55 ? mat.base : mat.body, age: 0, life: rand(1.1, 2.2), seed: Math.random() });
    }
    for (let i = 0; i < 9; i++){
      const a = rand(0, TAU), sp = rand(10, 45);
      parts.push({ kind: 1, x: ix + rand(-5, 5), y: iy + rand(-4, 4), z: rand(0, 6), vx: Math.cos(a) * sp, vy: Math.sin(a) * sp, vz: rand(8, 30), size: rand(8, 18),
        col: mat.body, age: 0, life: rand(0.45, 0.8), seed: Math.random() });
    }
    snd('smash', shards, clamp(Math.sqrt(1400 / (bw * bh / 6)), 0.8, 1.3));
    wake(1);
  }

  // ---- the mend
  function beginReturn(){
    if (state !== 'broken') return;
    clearTimeout(restoreTimer); restoreTimer = 0;
    state = 'returning';
    canvas.style.pointerEvents = 'none'; canvas.style.cursor = '';
    for (const p of parts) p.out = p.out ?? p.age;
    const order = shards.slice().sort((a, b) => Math.hypot(b.px - b.hc[0] - M, b.py - b.hc[1] - M) - Math.hypot(a.px - a.hc[0] - M, a.py - a.hc[1] - M));
    order.forEach((s, i) => {
      const dist = Math.hypot(s.px - s.hc[0] - M, s.py - s.hc[1] - M);
      let ang = s.ang % TAU; if (ang > Math.PI) ang -= TAU; if (ang < -Math.PI) ang += TAU;
      s.ret = { t: -i * 0.05, dur: clamp(0.5 + dist / 420, 0.5, 0.95), px: s.px, py: s.py, ang, z: s.z, tx: s.tx, ty: s.ty, done: false };
      s.vx = s.vy = s.vz = s.va = s.wx = s.wy = 0; s.held = false; s.air = false;
    });
    wake(1);
  }
  function stepReturn(dt){
    let all = true;
    for (const s of shards){
      const r = s.ret; if (!r) continue;
      if (r.done) continue;
      r.t += dt;
      const u = clamp(r.t / r.dur, 0, 1), e = easeIO(u);
      s.px = r.px + (s.hc[0] + M - r.px) * e;
      s.py = r.py + (s.hc[1] + M - r.py) * e;
      s.ang = r.ang * (1 - e);
      s.tx = r.tx * (1 - e); s.ty = r.ty * (1 - e);
      s.z = r.z * (1 - e) + Math.sin(Math.PI * e) * 7;
      if (u >= 1){ r.done = true; s.px = s.hc[0] + M; s.py = s.hc[1] + M; s.ang = s.tx = s.ty = s.z = 0; snd('seat', s); }
      else all = false;
    }
    if (all){
      joined = true; state = 'mending'; mendT = 0; gleamPlayed = false;
      goldMax = 0;
      for (const s of shards) for (const p of s.local) goldMax = Math.max(goldMax, Math.hypot(p[0] + s.hc[0] - impact[0], p[1] + s.hc[1] - impact[1]));
      snd('gold');
    }
  }
  const FILL_T = 0.75, COOL_T = 1.4, GLEAM_A = 1.05, GLEAM_B = 1.9, HOLD_T = 2.7, FADE_T = 0.8;
  function goldState(){
    if (state === 'mending' || kept){
      const t = kept ? 99 : mendT;
      const fill = goldMax * 1.05 * ease(clamp(t / FILL_T, 0, 1)) + 12;
      const molten = 1 - ease(clamp((t - 0.35) / (COOL_T - 0.35), 0, 1));
      const g = clamp((t - GLEAM_A) / (GLEAM_B - GLEAM_A), 0, 1);
      const gleamX = -40 + (bw + bh * 0.5 + 80) * g;
      const vis = keepGold ? 1 : 1 - ease(clamp((t - HOLD_T) / FADE_T, 0, 1));
      return { fill, molten, gleamX: g > 0 && g < 1 ? gleamX : -999, vis, crack: vis };
    }
    return { fill: 0, molten: 0, gleamX: -999, vis: 0, crack: state === 'rest' ? 0 : 1 };
  }

  // ---- physics
  function mat3(s){
    const ca = Math.cos(s.ang), sa = Math.sin(s.ang), cx = Math.cos(s.tx), sx = Math.sin(s.tx), cy = Math.cos(s.ty), sy = Math.sin(s.ty);
    const r0 = [cy, 0, sy], r1 = [sx * sy, cx, -sx * cy], r2 = [-cx * sy, sx, cx * cy];
    return [ca * r0[0] - sa * r1[0], ca * r0[1] - sa * r1[1], ca * r0[2] - sa * r1[2],
            sa * r0[0] + ca * r1[0], sa * r0[1] + ca * r1[1], sa * r0[2] + ca * r1[2],
            r2[0], r2[1], r2[2]];
  }
  // height of the centroid that puts the lowest corner of the tipped shard on the page
  function zRest(s, m){
    let lo = Infinity;
    for (const p of s.hull){ const v = m[6] * p[0] + m[7] * p[1]; if (v < lo) lo = v; }
    return Math.max(0, T * m[8] - T - lo, -lo - T);
  }
  function worldHull(s){
    const c = Math.cos(s.ang), sn = Math.sin(s.ang);
    return s.hull.map(([x, y]) => [s.px + c * x - sn * y, s.py + sn * x + c * y]);
  }
  function edgeRoom(s){
    let d = Math.min(s.px - lim[0], s.py - lim[1], lim[2] - s.px, lim[3] - s.py) - s.rad - EDGE;
    for (const [ax, ay, bx, by] of allBlocks){
      const dx = Math.max(ax - s.px, s.px - bx, 0), dy = Math.max(ay - s.py, s.py - by, 0);
      d = Math.min(d, Math.hypot(dx, dy) - s.rad);
    }
    return d;
  }
  function confine(s){
    const e = EDGE + s.rad;
    if (s.px < lim[0] + e){ s.px = lim[0] + e; if (s.vx < 0) s.vx *= -0.2; }
    if (s.px > lim[2] - e){ s.px = lim[2] - e; if (s.vx > 0) s.vx *= -0.2; }
    if (s.py < lim[1] + e){ s.py = lim[1] + e; if (s.vy < 0) s.vy *= -0.2; }
    if (s.py > lim[3] - e){ s.py = lim[3] - e; if (s.vy > 0) s.vy *= -0.2; }
    for (const [ax, ay, bx, by] of allBlocks){
      const qx = clamp(s.px, ax, bx), qy = clamp(s.py, ay, by);
      let dx = s.px - qx, dy = s.py - qy, d = Math.hypot(dx, dy);
      if (d >= s.rad * 0.8) continue;
      if (d < 1e-6){ dx = s.px - (ax + bx) / 2; dy = s.py - (ay + by) / 2; d = Math.hypot(dx, dy) || 1; }
      const push = s.rad * 0.8 - (d < 1e-6 ? 0 : d);
      s.px += dx / d * push * 0.5; s.py += dy / d * push * 0.5;
      const vn = s.vx * dx / d + s.vy * dy / d;
      if (vn < 0){ s.vx -= vn * dx / d * 1.2; s.vy -= vn * dy / d * 1.2; }
    }
  }
  function land(s, vi){
    if (vi > 70){
      s.vz = vi * 0.28;
      s.vx *= 0.72; s.vy *= 0.72; s.va *= 0.7;
      s.wx = -s.wx * 0.45 + rand(-3, 3); s.wy = -s.wy * 0.45 + rand(-3, 3);
      snd('clink', s, vi);
    } else {
      s.vz = 0; s.air = false;
      if (vi > 12) snd('tick', s, vi / 20);
    }
  }
  function stepShard(s, dt){
    if (s.held){
      const tx = ptr.x + s.grab[0], ty = ptr.y + s.grab[1];
      const nvx = (tx - s.px) * 22, nvy = (ty - s.py) * 22, sp = Math.hypot(nvx, nvy), cap = 1400;
      s.vx = sp > cap ? nvx / sp * cap : nvx; s.vy = sp > cap ? nvy / sp * cap : nvy;
      s.px += s.vx * dt; s.py += s.vy * dt;
      s.z += (6 - s.z) * Math.min(1, dt * 14);
      s.tx += (clamp(-s.vy * 0.0012, -0.35, 0.35) - s.tx) * Math.min(1, dt * 10);
      s.ty += (clamp(s.vx * 0.0012, -0.35, 0.35) - s.ty) * Math.min(1, dt * 10);
      s.va *= Math.exp(-dt * 8);
      s.ang += s.va * dt;
      confine(s);
      return true;
    }
    const m = mat3(s), zr = zRest(s, m);
    if (s.air || s.z > zr + 0.05 || s.vz > 0){
      s.vz -= G * dt;
      s.z += s.vz * dt; s.px += s.vx * dt; s.py += s.vy * dt; s.ang += s.va * dt;
      s.tx += s.wx * dt; s.ty += s.wy * dt;
      // shell pieces tumble, but never so far over that they land face down
      for (const k of ['tx', 'ty']){ const w = k === 'tx' ? 'wx' : 'wy'; if (Math.abs(s[k]) > 1.05){ s[k] = Math.sign(s[k]) * 1.05; s[w] *= -0.6; } }
      const z2 = zRest(s, mat3(s));
      if (s.z <= z2){ const vi = -s.vz; s.z = z2; s.air = false; if (vi > 0) land(s, vi); }
      else s.air = true;
      confine(s);
      return true;
    }
    // on the page: slide with friction (harder near the edge of the stage), spin down, and rock flat
    let moving = false;
    const sp = Math.hypot(s.vx, s.vy);
    const early = clamp((simT - smashT - SLIDE_T[0]) / (SLIDE_T[1] - SLIDE_T[0]), 0, 1);
    const mu = FRICTION_LO + (FRICTION - FRICTION_LO) * ease(early);
    if (sp > 0){
      const er = edgeRoom(s);
      const band = clamp(1 - er / EDGE_SOFT, 0, 1);
      const dec = (mu * (1 + 3 * band * band) + 900 * band * band) * dt;
      if (sp <= dec){ s.vx = s.vy = 0; }
      else {
        s.vx -= s.vx / sp * dec; s.vy -= s.vy / sp * dec;
        s.px += s.vx * dt; s.py += s.vy * dt; moving = true;
        s.skid -= dt * (3 + sp * 0.045);
        if (s.skid < 0){
          s.skid = rand(0.6, 1.4);
          if (sp > 18){
            snd('skitter', s, sp);
            // a shell piece skating on its rim catches and chatters: a tiny hop, a wobble and a grip of friction
            if (early < 1 && Math.random() < 0.7){
              s.vz = rand(14, 34) * clamp(sp / 90, 0.4, 1); s.air = true;
              s.wx += rand(-2.5, 2.5); s.wy += rand(-2.5, 2.5);
              s.vx *= 0.97; s.vy *= 0.97;
              s.va += rand(-1.2, 1.2);
            }
          }
        }
      }
    }
    if (s.va !== 0){ const d = (6 + 10 * ease(early)) * dt; s.va = Math.abs(s.va) <= d ? 0 : s.va - Math.sign(s.va) * d; s.ang += s.va * dt; moving = true; }
    for (const [k, w] of [['tx', 'wx'], ['ty', 'wy']]){
      if (s[k] === 0 && s[w] === 0) continue;
      const before = s[k];
      s[w] -= Math.sign(s[k] || s[w]) * TILT_G * dt;
      s[w] *= Math.exp(-dt * 1.5);
      s[k] += s[w] * dt;
      if (before !== 0 && Math.sign(s[k]) !== Math.sign(before)){
        // the tipped edge slaps down: it bounces back a little less each time, with a tick
        const v = Math.abs(s[w]);
        s[k] = 0; s[w] = -s[w] * 0.38;
        if (v > 0.6) snd('tick', s, v);
        if (Math.abs(s[w]) < 0.5) s[w] = 0;
      }
      moving = true;
    }
    s.z = zRest(s, mat3(s));
    confine(s);
    return moving;
  }
  function collide(){
    const live = shards.filter(s => s.z < 3 || s.held);
    const W = live.map(worldHull);
    for (let i = 0; i < live.length; i++) for (let j = i + 1; j < live.length; j++){
      const a = live[i], b = live[j];
      if (Math.hypot(a.px - b.px, a.py - b.py) > a.rad + b.rad) continue;
      const r = sat(W[i], W[j]); if (!r) continue;
      const [nx, ny, depth] = r;
      const ma = a.held ? 1e9 : a.area, mb = b.held ? 1e9 : b.area, k = depth / (ma + mb);
      if (!a.held){ a.px -= nx * k * mb; a.py -= ny * k * mb; }
      if (!b.held){ b.px += nx * k * ma; b.py += ny * k * ma; }
      const vn = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny;
      if (vn < 0){
        const j2 = -1.35 * vn / (1 / ma + 1 / mb);
        if (!a.held){ a.vx -= nx * j2 / ma; a.vy -= ny * j2 / ma; a.va += rand(-1, 1) * Math.min(4, -vn * 0.02); }
        if (!b.held){ b.vx += nx * j2 / mb; b.vy += ny * j2 / mb; b.va += rand(-1, 1) * Math.min(4, -vn * 0.02); }
        if (-vn > 40) snd('clink', Math.random() < 0.5 ? a : b, -vn * 0.8);
      }
    }
  }
  function stepParts(dt){
    for (const p of parts){
      p.age += dt;
      if (p.kind === 1){ p.x += p.vx * dt; p.y += p.vy * dt; p.z += p.vz * dt; p.vx *= Math.exp(-dt * 3); p.vy *= Math.exp(-dt * 3); continue; }
      if (p.z > 0 || p.vz > 0){
        p.vz -= G * dt; p.z += p.vz * dt; p.x += p.vx * dt; p.y += p.vy * dt;
        if (p.z <= 0){ p.z = 0; if (p.vz < -60){ p.vz *= -0.3; p.vx *= 0.6; p.vy *= 0.6; } else p.vz = 0; }
      } else {
        const sp = Math.hypot(p.vx, p.vy), dec = FRICTION * 1.3 * dt;
        if (sp <= dec) p.vx = p.vy = 0; else { p.vx -= p.vx / sp * dec; p.vy -= p.vy / sp * dec; p.x += p.vx * dt; p.y += p.vy * dt; }
      }
      const e = EDGE + 4;
      if (p.x < lim[0] + e){ p.x = lim[0] + e; p.vx = 0; } if (p.x > lim[2] - e){ p.x = lim[2] - e; p.vx = 0; }
      if (p.y < lim[1] + e){ p.y = lim[1] + e; p.vy = 0; } if (p.y > lim[3] - e){ p.y = lim[3] - e; p.vy = 0; }
    }
    parts = parts.filter(p => p.age < p.life && !(p.out != null && p.age > p.out + 0.35));
  }

  // ---- interaction
  let ptr = null, lastDown = null, swallowUntil = 0, lastFoot = 0;
  const FOOTPRINT_MS = 450;
  const stageXY = e => { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  function shardAt(x, y, slop = 0){
    let best = null, bd = Infinity;
    for (let i = shards.length - 1; i >= 0; i--){
      const s = shards[i]; if (!s.world) continue;
      if (pointInPoly(s.world, x, y)) return s;
      if (slop > 0){ const d = distToPoly(s.world, x, y); if (d < slop && d < bd){ bd = d; best = s; } }
    }
    return best;
  }
  function onDocDown(e){
    if (e.button !== 0 || ptr || lost) return;
    checkPlace();
    const [x, y] = stageXY(e);
    if (x < 0 || y < 0 || x > stageW || y > stageH) return;
    const onBtn = e.target === button || button.contains(e.target);
    const onCanvas = e.target === canvas;
    if (state === 'rest' || state === 'mending'){
      if (!onBtn || button.disabled) return;
      audio();
      ptr = { id: e.pointerId, x0: x, y0: y, x, y, dragging: false, kind: 'press' };
      lastDown = [x - M, y - M];
      pressTarget = 1; snd('tap'); wake(0.3);
      return;
    }
    if (!onBtn && !onCanvas && e.target.closest && e.target.closest(INTERACTIVE)) return;
    // A deliberate press on the button's own footprint (not on a piece) is still a press of the button: it
    // counts, and the pieces answer it. Right after the blow it is the tail of a double click instead.
    const footprint = onBtn && !button.disabled && performance.now() - smashWall > FOOTPRINT_MS;
    if (state === 'returning'){
      if (!onBtn && !onCanvas) return;
      if (footprint){ ptr = { id: e.pointerId, x0: x, y0: y, x, y, dragging: false, kind: 'press' }; pressTarget = 0.5; audio(); wake(0.3); return; }
      e.preventDefault();
      ptr = { id: e.pointerId, x0: x, y0: y, x, y, dragging: false, kind: 'void', captured: true };
      return;
    }
    // broken: a press on a shard (or on the button's empty footprint) handles the pieces, never the page
    const s = shardAt(x, y, GRAB);
    if (!s && footprint){ ptr = { id: e.pointerId, x0: x, y0: y, x, y, dragging: false, kind: 'press' }; audio(); return; }
    if (!s && !onBtn && !onCanvas) return;
    e.preventDefault();
    audio();
    ptr = { id: e.pointerId, x0: x, y0: y, x, y, lx: x, ly: y, dragging: false, kind: 'shard', shard: s, captured: true };
    if (s){ s.held = true; s.air = false; s.vz = 0; s.grab = [s.px - x, s.py - y]; canvas.style.cursor = 'grabbing'; }
    idleT = 0; clearTimeout(restoreTimer); restoreTimer = 0;
    wake(0.5);
  }
  function onDocMove(e){
    if (lost) return;
    const r = btnRect(), x = e.clientX - r.left + M, y = e.clientY - r.top + M;
    const near = x > -60 && y > -60 && x < stageW + 60 && y < stageH + 60;
    if (!near && !ptr){
      if (lampT){ lampT = null; wake(0.1); }
      if (state === 'broken' && canvas.style.pointerEvents === 'auto'){ canvas.style.pointerEvents = 'none'; canvas.style.cursor = ''; }
      return;
    }
    if (near){
      // the softbox follows the cursor part of the way, so the reflection slides across the glaze
      const d = defLamp();
      lampT = [d[0] + (x - d[0]) * 0.55, d[1] + (y - d[1]) * 0.55];
      wake(0.1);
    } else if (lampT){ lampT = null; wake(0.1); }
    if (ptr && e.pointerId === ptr.id){
      ptr.x = x; ptr.y = y;
      if (!ptr.dragging && Math.hypot(x - ptr.x0, y - ptr.y0) > DRAG_START) ptr.dragging = true;
      if (ptr.kind === 'shard'){ idleT = 0; wake(0.3); }
      return;
    }
    // over a loose shard the stage takes the pointer (and shows it can be picked up); elsewhere it passes through
    if (state === 'broken'){
      const over = x >= 0 && y >= 0 && x < stageW && y < stageH && (e.target === canvas || !(e.target.closest && e.target.closest(INTERACTIVE)) || e.target === button) && shardAt(x, y, 2);
      canvas.style.pointerEvents = over ? 'auto' : 'none';
      canvas.style.cursor = over ? 'grab' : '';
      if (over) idleT = Math.min(idleT, RESTORE * 0.5);
    }
  }
  function onDocUp(e){
    if (!ptr || e.pointerId !== ptr.id) return;
    const p = ptr; ptr = null;
    pressTarget = 0;
    if (p.captured) swallowUntil = performance.now() + 600;
    if (p.kind === 'press'){
      if (state === 'broken' || state === 'returning') lastFoot = performance.now();
      else lastDown = [p.x - M, p.y - M];
      wake(0.2); return;
    }
    if (p.kind !== 'shard') return;
    idleT = 0;
    const s = p.shard;
    if (s){
      s.held = false; s.air = true; s.vz = 30;
      if (!p.dragging){
        // a tap flips the shard up with a clink
        s.vz = rand(120, 180); s.wx = rand(-8, 8); s.wy = rand(-8, 8); s.va = rand(-5, 5);
        snd('clink', s, 180);
      } else {
        const sp = Math.hypot(s.vx, s.vy), cap = 650;
        if (sp > cap){ s.vx *= cap / sp; s.vy *= cap / sp; }
      }
      canvas.style.cursor = 'grab';
    } else jostle(0.6);
    wake(1);
  }
  function onDocClick(e){
    if (e.detail > 0 && performance.now() < swallowUntil){ swallowUntil = 0; e.preventDefault(); e.stopImmediatePropagation(); }
  }
  // the shards all hop where they lie: how a keyboard press (or a stray tap) is felt while broken
  function jostle(k = 1){
    for (const s of shards){ if (s.held) continue; s.air = true; s.vz = rand(60, 120) * k; s.wx += rand(-4, 4) * k; s.wy += rand(-4, 4) * k; s.va += rand(-2, 2) * k; }
    idleT = 0; wake(0.5);
  }
  let touchOwned = false;
  function onTouch(e){
    if (e.type === 'touchend' || e.type === 'touchcancel'){ if (!e.touches.length) touchOwned = false; return; }
    if (!e.cancelable || lost) return;
    if (e.type === 'touchmove'){ if (touchOwned || (ptr && ptr.captured)) e.preventDefault(); return; }
    if (state !== 'broken') return;
    const t = e.touches[0]; if (!t) return;
    const r = canvas.getBoundingClientRect();
    if ((ptr && ptr.captured) || shardAt(t.clientX - r.left, t.clientY - r.top, GRAB)){ touchOwned = true; e.preventDefault(); }
  }
  function onClick(e){
    if (lost) return;
    if (state === 'broken' || state === 'returning'){
      // A pointer on the pieces is handling them, not pressing, and right after the blow a second click is
      // the rest of a double click. Anything else (keyboard, or a later press on the empty footprint) is a
      // real activation: it goes through, and the pieces hop with a dull knock so it is seen and heard.
      const fp = e.detail > 0 && lastFoot && performance.now() - lastFoot < 1500;
      lastFoot = 0;
      if (e.detail > 0 && !fp){ e.preventDefault(); e.stopImmediatePropagation(); return; }
      if (state === 'broken' && !reduceMotion.matches){ jostle(e.detail > 0 ? 0.7 : 1); snd('knock', shards); }
      else snd('tap');
      return;
    }
    audio();
    const at = e.detail > 0 && lastDown ? lastDown : [bw / 2, bh / 2];
    lastDown = null;
    pressTarget = 0;
    if (reduceMotion.matches){
      flash = 1; snd('clink', shards[0], 160); wake(0.6);
      return;
    }
    if (state === 'mending') toRest();
    smashAt(clamp(at[0], 3, bw - 3), clamp(at[1], 3, bh - 3));
  }
  const onCancel = e => { if (ptr && e.pointerId === ptr.id){ if (ptr.shard) ptr.shard.held = false; ptr = null; pressTarget = 0; wake(0.3); } };
  const capNP = { capture: true, passive: false };
  document.addEventListener('touchstart', onTouch, capNP);
  document.addEventListener('touchmove', onTouch, capNP);
  document.addEventListener('touchend', onTouch, true);
  document.addEventListener('touchcancel', onTouch, true);
  document.addEventListener('pointerdown', onDocDown, true);
  document.addEventListener('click', onDocClick, true);
  addEventListener('pointermove', onDocMove, { passive: true });
  addEventListener('pointerup', onDocUp, true);
  addEventListener('pointercancel', onCancel, true);
  button.addEventListener('click', onClick, true);

  // ---- step
  let simT = 0, activeUntil = 0;
  let raf = 0, inFrame = false, last = performance.now();
  // one animation loop at most: a wake from inside a frame must not start a second loop
  function wake(sec){
    if (lost) return;
    activeUntil = Math.max(activeUntil, simT + sec);
    if (!raf && !inFrame){ raf = requestAnimationFrame(frame); audioAwake(self, true); }
  }
  function advance(dt){
    frameCount++;
    checkPlace();
    simT += dt;
    if (state === 'broken' && instances.size > 1) gatherForeign();
    let moving = false;
    if (state === 'broken' || state === 'returning'){
      const n = Math.max(1, Math.ceil(dt / (1 / 120))), h = dt / n;
      for (let i = 0; i < n; i++){
        if (state === 'broken'){
          for (const s of shards) if (stepShard(s, h)) moving = true;
          collide();
        } else { stepReturn(h); moving = true; if (state !== 'returning') break; }
      }
      if (state === 'broken'){
        if (moving || ptr) idleT = 0;
        else {
          idleT += dt;
          if (idleT >= RESTORE) beginReturn();
        }
      }
    }
    if (state === 'mending'){
      mendT += dt; moving = true;
      if (!gleamPlayed && mendT >= GLEAM_A + 0.15){ gleamPlayed = true; snd('gleam'); }
      if (mendT >= HOLD_T + FADE_T){
        if (keepGold){ const keep = shards; toRest(true); shards = keep; kept = true; joined = true; }
        else toRest();
        dirty = true;
      }
    }
    if (parts.length){ stepParts(dt); moving = true; }
    press += (pressTarget - press) * Math.min(1, dt * 25);
    flash = Math.max(0, flash - dt * 2.5);
    const tgt = lampT || defLamp();
    if (!lamp) lamp = tgt.slice();
    const k = Math.min(1, dt * 9);
    lamp[0] += (tgt[0] - lamp[0]) * k; lamp[1] += (tgt[1] - lamp[1]) * k;
    const lampMoving = Math.hypot(tgt[0] - lamp[0], tgt[1] - lamp[1]) > 0.2;
    render();
    dirty = false;
    // broken and settled: sleep, and let a timer bring the pieces home
    if (state === 'broken' && !moving && !ptr && !lampMoving && simT >= activeUntil){
      clearTimeout(restoreTimer);
      restoreTimer = setTimeout(() => { restoreTimer = 0; if (state === 'broken' && !ptr){ beginReturn(); } }, Math.max(0, RESTORE - idleT) * 1000);
      return false;
    }
    return simT < activeUntil || moving || !!ptr || lampMoving || Math.abs(press - pressTarget) > 0.004 || flash > 0 || state === 'returning';
  }
  let manual = false, frameCount = 0;
  function frame(now){
    raf = 0;
    const dt = Math.min((now - last) / 1000, 1 / 30); last = now;
    if (manual || lost || gl.isContextLost()) return;
    inFrame = true;
    let busy;
    try { busy = advance(dt); } finally { inFrame = false; }
    if (busy) raf = requestAnimationFrame(frame);
    else { last = performance.now(); audioAwake(self, false); }
  }

  // ---- render
  let vdata = new Float32Array(STRIDE * 4096), vn = 0;
  function V(x, y, hx, hy, d, nx, ny, kind, Nx, Ny, Nz, rc, rs, mode, v, a){
    if ((vn + 1) * STRIDE > vdata.length){ const b = new Float32Array(vdata.length * 2); b.set(vdata); vdata = b; }
    const o = vn * STRIDE;
    vdata[o] = x; vdata[o + 1] = y; vdata[o + 2] = hx; vdata[o + 3] = hy;
    vdata[o + 4] = d; vdata[o + 5] = nx; vdata[o + 6] = ny; vdata[o + 7] = kind;
    vdata[o + 8] = Nx; vdata[o + 9] = Ny; vdata[o + 10] = Nz; vdata[o + 11] = rc; vdata[o + 12] = rs;
    vdata[o + 13] = mode; vdata[o + 14] = v; vdata[o + 15] = a;
    vn++;
  }
  // a shard posed in 3D and projected: returns top and bottom outline in stage px
  function pose(s){
    const m = mat3(s), n = s.local.length, top = new Array(n), bot = new Array(n), zT = new Array(n);
    const lift = 1 + s.z * 0.003;
    const dy = press * (joined ? 1 : 0);
    for (let i = 0; i < n; i++){
      const [lx, ly] = s.local[i];
      const X = s.px + (m[0] * lx + m[1] * ly) * lift, Y = s.py + (m[3] * lx + m[4] * ly) * lift + dy, Z = s.z + m[6] * lx + m[7] * ly;
      top[i] = [X, Y - Z * LEAN]; zT[i] = Z;
      const Xb = X - m[2] * T, Yb = Y - m[5] * T, Zb = Z - m[8] * T;
      bot[i] = [Xb, Yb - Zb * LEAN];
    }
    return { m, top, bot, zT };
  }
  function emitShadow(s, ps, extraZ, soft, alpha, poly){
    const { m } = ps, n = s.local.length;
    const pts = new Array(n);
    for (let i = 0; i < n; i++){
      const zz = ps.zT[i] + T + extraZ;
      const [lx, ly] = s.local[i];
      pts[i] = [s.px + m[0] * lx + m[1] * ly + zz * SH[0], s.py + m[3] * lx + m[4] * ly + press * 0.5 + zz * SH[1] + T * LEAN * 0.5];
    }
    const cz = s.z + T + extraZ;
    const c = [s.px + cz * SH[0], s.py + cz * SH[1] + T * LEAN * 0.5];
    for (let i = 0; i < n; i++){
      const a = pts[i], b = pts[(i + 1) % n];
      V(c[0], c[1], 0, 0, s.dC[i], 0, 0, 0, 0, 0, 1, 1, 0, 2, soft, alpha);
      V(a[0], a[1], 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, soft, alpha);
      V(b[0], b[1], 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, soft, alpha);
    }
  }
  function emitWalls(s, ps, alpha){
    const { m, top, bot } = ps, n = s.local.length, rc = Math.cos(s.ang), rs = Math.sin(s.ang);
    for (let i = 0; i < n; i++){
      const kind = s.flags[i];
      if (kind && joined) continue;
      const j = (i + 1) % n;
      const a = top[i], b = top[j], bb = bot[j], ab = bot[i];
      // front-facing test on the projected quad
      const qa = (a[0] * b[1] - b[0] * a[1]) + (b[0] * bb[1] - bb[0] * b[1]) + (bb[0] * ab[1] - ab[0] * bb[1]) + (ab[0] * a[1] - a[0] * ab[1]);
      if (qa * s.sgn >= 0) continue;
      const [nx, ny] = s.nrm[i];
      const Nx = m[0] * nx + m[1] * ny, Ny = m[3] * nx + m[4] * ny, Nz = m[6] * nx + m[7] * ny;
      const ha = [s.hc[0] + s.local[i][0], s.hc[1] + s.local[i][1]], hb = [s.hc[0] + s.local[j][0], s.hc[1] + s.local[j][1]];
      V(a[0], a[1], ha[0], ha[1], 0, nx, ny, kind, Nx, Ny, Nz, rc, rs, 1, 0, alpha);
      V(b[0], b[1], hb[0], hb[1], 0, nx, ny, kind, Nx, Ny, Nz, rc, rs, 1, 0, alpha);
      V(bb[0], bb[1], hb[0], hb[1], 0, nx, ny, kind, Nx, Ny, Nz, rc, rs, 1, 1, alpha);
      V(a[0], a[1], ha[0], ha[1], 0, nx, ny, kind, Nx, Ny, Nz, rc, rs, 1, 0, alpha);
      V(bb[0], bb[1], hb[0], hb[1], 0, nx, ny, kind, Nx, Ny, Nz, rc, rs, 1, 1, alpha);
      V(ab[0], ab[1], ha[0], ha[1], 0, nx, ny, kind, Nx, Ny, Nz, rc, rs, 1, 1, alpha);
    }
  }
  function emitTop(s, ps, alpha){
    const { m, top } = ps, n = s.local.length, rc = Math.cos(s.ang), rs = Math.sin(s.ang);
    const Nx = m[2], Ny = m[5], Nz = m[8];
    // centroid on screen
    const zc = s.z, c = [s.px, s.py + press * (joined ? 1 : 0) - zc * LEAN];
    for (let i = 0; i < n; i++){
      const j = (i + 1) % n, kind = s.flags[i], [nx, ny] = s.nrm[i];
      const a = top[i], b = top[j];
      V(c[0], c[1], s.hc[0], s.hc[1], s.dC[i], nx, ny, kind, Nx, Ny, Nz, rc, rs, 0, 0, alpha);
      V(a[0], a[1], s.hc[0] + s.local[i][0], s.hc[1] + s.local[i][1], 0, nx, ny, kind, Nx, Ny, Nz, rc, rs, 0, 0, alpha);
      V(b[0], b[1], s.hc[0] + s.local[j][0], s.hc[1] + s.local[j][1], 0, nx, ny, kind, Nx, Ny, Nz, rc, rs, 0, 0, alpha);
    }
  }
  const pdata = new Float32Array(9 * 512);
  let slabShadow = null;
  function render(){
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    vn = 0;
    const poses = shards.map(s => { const p = pose(s); s.world = p.top; return p; });
    // shadows: while whole, one shadow for the whole tile, so the seams never show through it
    if (joined){
      if (!slabShadow || slabShadow.bw !== bw || slabShadow.bh !== bh){ slabShadow = mkShard(slabPoly, slabPoly.map(() => 0)); slabShadow.bw = bw; slabShadow.bh = bh; }
      const sp = pose(slabShadow);
      emitShadow(slabShadow, sp, 10 - press * 4, 9, 0.16, null);
      emitShadow(slabShadow, sp, 0, 2.2, 0.3, null);
    } else {
      shards.forEach((s, i) => {
        const hA = Math.max(0.3, 1 - s.z / 90);
        emitShadow(s, poses[i], 8, 7 + s.z * 0.15, 0.15 * hA * s.alpha, null);
        emitShadow(s, poses[i], 0, 1.6 + s.z * 0.25, 0.3 * hA * s.alpha, null);
      });
    }
    // pieces on the page: all walls, then all tops; pieces in the air go over them, lowest first
    const flat = [], up = [];
    shards.forEach((s, i) => ((s.z < 0.5 && !s.held && Math.abs(s.tx) + Math.abs(s.ty) < 0.02) ? flat : up).push(i));
    for (const i of flat) emitWalls(shards[i], poses[i], shards[i].alpha);
    for (const i of flat) emitTop(shards[i], poses[i], shards[i].alpha);
    up.sort((a, b) => shards[a].z - shards[b].z);
    for (const i of up){ emitWalls(shards[i], poses[i], shards[i].alpha); emitTop(shards[i], poses[i], shards[i].alpha); }

    const u = P.main.u;
    gl.useProgram(P.main.p);
    gl.uniform2f(u.uStage, stageW, stageH);
    gl.uniform2f(u.uSize, bw, bh); gl.uniform1f(u.uRad, Math.max(radius, 1.5));
    gl.uniform3f(u.uGlaze, ...mat.base); gl.uniform3f(u.uPool, ...mat.pool); gl.uniform3f(u.uRim, ...mat.rim);
    gl.uniform3f(u.uBody, ...mat.body); gl.uniform3f(u.uInk, ...mat.ink); gl.uniform3f(u.uCrazeCol, ...mat.craze);
    gl.uniform4f(u.uMat, mat.gloss, mat.crazing, mat.speckle, mat.rimBreak);
    gl.uniform4f(u.uMat2, mat.pooling, mat.mottle, mat.inkK, mat.decor);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, textTex); gl.uniform1i(u.uText, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, seamTex); gl.uniform1i(u.uSeam, 1);
    gl.uniform2f(u.uSeamUV, seamSc / seamW, seamSc / seamH);
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform2f(u.uEye, ...eye);
    const L = fitLamp(lamp || defLamp()), LS = lampSize();
    gl.uniform2f(u.uLamp, L[0], L[1]);
    gl.uniform2f(u.uLampSize, LS[0], LS[1]);
    const g = goldState();
    gl.uniform4f(u.uGold, g.fill, g.molten, g.gleamX, g.vis);
    gl.uniform1f(u.uCrack, g.crack);
    gl.uniform2f(u.uImpact, impact[0], impact[1]);
    gl.uniform1f(u.uFlash, flash);
    gl.uniform1f(u.uDisabled, button.disabled ? 1 : 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbuf);
    gl.bufferData(gl.ARRAY_BUFFER, vdata.subarray(0, vn * STRIDE), gl.DYNAMIC_DRAW);
    gl.bindVertexArray(vao);
    gl.drawArrays(gl.TRIANGLES, 0, vn);

    if (parts.length){
      let m = 0;
      for (const p of parts){
        if (m >= 512) break;
        const fadeOut = p.out != null ? Math.max(0, 1 - (p.age - p.out) / 0.35) : 1;
        let a;
        if (p.kind === 1) a = 0.22 * (1 - p.age / p.life) * Math.min(1, p.age / 0.05);
        else a = Math.min(1, (p.life - p.age) / 0.4);
        const wf = clamp((Math.min(p.x - lim[0], p.y - lim[1], lim[2] - p.x, lim[3] - p.y) - EDGE) / 12, 0, 1);
        a *= fadeOut * wf;
        if (a <= 0.003) continue;
        const c = p.col;
        pdata.set([p.x, p.y - p.z * LEAN, p.size * (p.kind === 1 ? 1 + p.age * 1.5 : 1), a, c[0], c[1], c[2], p.kind, p.seed], m * 9); m++;
      }
      if (m){
        gl.bindBuffer(gl.ARRAY_BUFFER, pbuf);
        gl.bufferData(gl.ARRAY_BUFFER, pdata.subarray(0, m * 9), gl.DYNAMIC_DRAW);
        gl.useProgram(P.part.p);
        gl.uniform2f(P.part.u.uStage, stageW, stageH); gl.uniform1f(P.part.u.uDpr, dpr);
        gl.bindVertexArray(pvao); gl.drawArrays(gl.POINTS, 0, m);
      }
    }
    gl.bindVertexArray(null);
    gl.disable(gl.BLEND);
  }

  // ---- keep the stage on the button
  const ro = new ResizeObserver(() => { docRect = null; if (!checkPlace() && !lost) place(); });
  ro.observe(button, { box: 'border-box' });
  ro.observe(document.documentElement);
  if (document.body) ro.observe(document.body);
  // Asleep, a page that mutates all the time must not force a layout per mutation, so the check is throttled.
  let moTimer = 0;
  const mo = new MutationObserver(recs => {
    if (recs.some(r => r.target === button && r.attributeName === 'disabled')){ dirty = true; wake(0.05); }
    // our own cursor and pointer-events changes on the stage cannot move the button
    if (recs.every(r => host.contains(r.target))) return;
    docRect = null;
    if (moTimer || raf || inFrame) return;
    moTimer = setTimeout(() => { moTimer = 0; if (!raf) checkPlace(); }, 300);
  });
  mo.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['class', 'style', 'hidden', 'open', 'disabled'] });
  const onFonts = () => { lastRect = null; docRect = null; if (!lost){ buildText(); checkPlace(); wake(0.05); } };
  document.fonts?.addEventListener?.('loadingdone', onFonts);
  const onScrollResize = () => { checkPlace(); place(); wake(0.05); };
  const onVis = () => { if (!document.hidden){ last = performance.now(); checkPlace(); wake(0.1); } };
  addEventListener('resize', onScrollResize); addEventListener('scroll', onScrollResize, { passive: true });
  document.addEventListener('visibilitychange', onVis);

  // A lost context shows the plain button until the GPU comes back; then the tile is rebuilt in place.
  function onLost(e){
    e.preventDefault();
    lost = true;
    cancelAnimationFrame(raf); raf = 0;
    clearTimeout(restoreTimer); restoreTimer = 0;
    ptr = null; pressTarget = 0; audioAwake(self, false);
    showButton();
    canvas.style.visibility = 'hidden'; canvas.style.pointerEvents = 'none';
  }
  function onRestored(){
    try { initGL(); } catch (err) { console.warn('ElementalFX ceramic: could not rebuild after context loss', err); return; }
    lost = false;
    hideButton();
    canvas.style.visibility = '';
    lastRect = null; placedAt = '';
    resize();
  }
  canvas.addEventListener('webglcontextlost', onLost);
  canvas.addEventListener('webglcontextrestored', onRestored);

  resize();
  instances.add(peer);

  // descriptors, not Object.assign: the getters must stay live
  return Object.defineProperties(self, Object.getOwnPropertyDescriptors({
    supported: true,
    get state(){ return kept ? 'mended' : state; },
    get glaze(){ return mat.name; },
    setGlaze(spec, color){ mat = resolveGlaze(spec || 'porcelain', color, opts.inkColor, autoInfo); dirty = true; wake(0.05); },
    restore(){ if (state === 'broken'){ for (const s of shards) s.held = false; ptr = null; beginReturn(); } },
    detach(){
      cancelAnimationFrame(raf); raf = 0; manual = true; ro.disconnect(); mo.disconnect(); clearTimeout(moTimer); clearTimeout(restoreTimer);
      document.fonts?.removeEventListener?.('loadingdone', onFonts);
      document.removeEventListener('touchstart', onTouch, capNP); document.removeEventListener('touchmove', onTouch, capNP);
      document.removeEventListener('touchend', onTouch, true); document.removeEventListener('touchcancel', onTouch, true);
      canvas.removeEventListener('webglcontextlost', onLost); canvas.removeEventListener('webglcontextrestored', onRestored);
      removeEventListener('resize', onScrollResize); removeEventListener('scroll', onScrollResize);
      document.removeEventListener('visibilitychange', onVis);
      document.removeEventListener('pointerdown', onDocDown, true);
      document.removeEventListener('click', onDocClick, true);
      removeEventListener('pointermove', onDocMove); removeEventListener('pointerup', onDocUp, true); removeEventListener('pointercancel', onCancel, true);
      button.removeEventListener('click', onClick, true);
      audioAwake(self, false);
      instances.delete(peer);
      button.style.touchAction = savedTouch;
      showButton();
      host.remove();
      gl.getExtension('WEBGL_lose_context')?.loseContext();
    },
    get _frames(){ return frameCount; },
    get _asleep(){ return !raf && !inFrame; },
    get _shards(){ return shards.length; },
    get _parts(){ return parts.length; },
    get _mendT(){ return mendT; },
    _dump(){ return shards.map(s => ({ z: +s.z.toFixed(2), tx: +s.tx.toFixed(3), ty: +s.ty.toFixed(3), wx: +s.wx.toFixed(2), wy: +s.wy.toFixed(2), air: s.air, v: +Math.hypot(s.vx, s.vy).toFixed(1), rad: +s.rad.toFixed(1), room: +edgeRoom(s).toFixed(1) })); },
    _shardAt(i){
      const s = shards.slice().sort((a, b) => b.area - a.area)[i]; if (!s) return null;
      const r = canvas.getBoundingClientRect(); return [s.px + r.left, s.py - s.z * LEAN + r.top];
    },
    _bounds(){
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const s of shards) for (const p of (s.world || [])){ x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]); x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]); }
      return { x0, y0, x1, y1, stageW, stageH, lim: lim.slice() };
    },
    _manual(on){ manual = on; if (!on) wake(0.1); },
    _advance(sec, fps = 60){ for (let i = 0, n = Math.round(sec * fps); i < n; i++) advance(1 / fps); },
    _lamp(x, y){ lampT = [x, y]; lamp = [x, y]; dirty = true; wake(0.05); },
  }));
}

window.ElementalFX = Object.assign(window.ElementalFX || {}, { attachCeramic, ceramicAudio, ceramicGlazes: Object.keys(GLAZES) });
})();
