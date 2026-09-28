// Water button: turns an existing <button> into a pill of water resting on the page.
//
//   <script src="fx/water.js"></script>
//   const fx = ElementalFX.attachWater(document.querySelector('#cta'));   // fx.detach() restores it
//
// A classic script (not a module) so pages work straight from file:// too.
//
// The <button> stays the real control (focus, keyboard, form submission, click handlers); this only
// draws it, in a transparent canvas laid over the page around it. Pressing dents the water, releasing
// pops it into droplets (that release is the button's click), droplets can be tapped to pop them or
// dragged around to merge them, and after a quiet moment they flow back and the button is there again.
// If WebGL2 with float render targets is unavailable, the button is left untouched.
//
// Options:
//   surface: { image: CanvasImageSource, rect: () => DOMRect }   what the page looks like under the water,
//            e.g. a textured section background (see ElementalFX.concreteCanvas). Without it the water
//            refracts the nearest ancestor's background colour.
//   background: CSS colour to use instead of the detected one.
//   labelColor: CSS colour for the label seen through the water (default: the button's colour, a bit darker).
//   margin: CSS px of stage around the button (default max(140, 1.6 x button height)).
//   area: an element whose box becomes the stage instead, so the drops can roam all of it (e.g. the hero
//         section the button sits on). Its edges are the walls. Overrides margin.
//   zIndex: of the stage canvas (default 10).   sound: false to stay silent.

(() => {
'use strict';

const TAU = Math.PI * 2;
const REGROUP_AFTER = 1.5;
const DRAG_START = 6;          // CSS px of travel before a press becomes a drag instead of a tap
const MAX_PRESS = 0.38;        // deepest squash against a wall, as a fraction of the radius
const DRY_MS = 14000;          // after the last movement, how long wet marks take to fade out completely

// ---------------------------------------------------------------- shaders
const VS = `#version 300 es
void main(){ vec2 p = vec2(float((gl_VertexID<<1)&2), float(gl_VertexID&2)); gl_Position = vec4(p*2.-1., 0., 1.); }`;

const NOISE = `
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*.1031); p3 += dot(p3, p3.yzx+33.33); return fract((p3.x+p3.y)*p3.z); }
float vnoise(vec2 p){
  vec2 i = floor(p), f = fract(p);
  vec2 u = f*f*f*(f*(f*6.-15.)+10.);
  float a = hash12(i), b = hash12(i+vec2(1,0)), c = hash12(i+vec2(0,1)), d = hash12(i+vec2(1,1));
  return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}
float fbm(vec2 p, int oct){
  float s = 0., a = .5, w = 0.;
  mat2 m = mat2(1.62, 1.18, -1.18, 1.62);
  for (int i = 0; i < 8; i++){ if (i >= oct) break; s += a*vnoise(p); w += a; p = m*p + 17.13; a *= .5; }
  return s / w;
}`;

// Procedural concrete / plaster.
const FS_CONCRETE = `#version 300 es
precision highp float;
uniform float uDpr;
uniform vec2 uRes;
out vec4 o;
${NOISE}
float relief(vec2 q){
  float mid  = fbm(q*0.05 + 3.1, 5);
  float fine = fbm(q*0.24 + 7.7, 4);
  float grit = vnoise(q*0.9 + 5.3);
  float pore = fbm(q*0.13 + 21.3, 3);
  float pits = smoothstep(0.40, 0.20, pore);
  return mid*0.50 + fine*0.60 + grit*0.14 - pits*0.20;
}
void main(){
  vec2 q = gl_FragCoord.xy / uDpr;              // CSS px, so texture scale is DPR independent
  float e = 0.6;
  float h0 = relief(q);
  float hx = relief(q + vec2(e,0)) - relief(q - vec2(e,0));
  float hy = relief(q + vec2(0,e)) - relief(q - vec2(0,e));
  vec3 n = normalize(vec3(-hx*4.6, -hy*4.6, 1.));
  vec3 L = normalize(vec3(0.35, 0.65, 0.68));
  float dif = clamp(dot(n, L), 0., 1.) / L.z;

  float blot = fbm(q*0.0055 + 1.7, 5);
  float blot2 = fbm(q*0.018 + 9.2, 4);
  float tone = 0.765 + (blot - 0.5)*0.11 + (blot2 - 0.5)*0.07;
  float cavity = smoothstep(0.48, 0.26, h0);
  vec3 col = vec3(tone) * mix(0.84, 1.045, dif);
  col *= 1.0 - cavity*0.11;
  float g = hash12(floor(q*1.5)) ;
  col *= 0.985 + g*0.03;
  // sparse dark air holes, round and soft
  vec2 cell = floor(q / 9.0);
  vec2 jit = vec2(hash12(cell + 3.1), hash12(cell + 7.9));
  float hole = length(fract(q / 9.0) - (0.2 + 0.6*jit)) * 9.0;
  float holeR = 0.5 + 1.1 * hash12(cell + 1.3);
  col *= 1.0 - step(0.993, hash12(cell + 11.7)) * smoothstep(holeR + 0.6, holeR - 0.4, hole) * 0.2;
  // soft vignette
  vec2 uv = gl_FragCoord.xy / uRes;
  col *= 0.90 + 0.10*smoothstep(1.15, 0.2, length((uv - 0.5)*vec2(1.2, 1.0)));
  col *= vec3(1.0, 0.995, 0.985);
  o = vec4(col, 1.);
}`;

// Liquid field: smooth union of the pill (a capsule) and droplets (stretched capsules),
// turned into a height map with a drop-like cross section.
const fsField = MAX_DROPS => `#version 300 es
precision highp float;
uniform vec4 uPill;          // cx, cy, halfW, halfH  (device px); halfH <= 0 disables
uniform vec4 uModes[4];      // amp, m, phase0, _    surface waves around the rim
uniform vec4 uDent;          // x, y, depth, sigma
uniform int uCount;
uniform vec4 uDrops[${MAX_DROPS}];   // x, y, r, wetting
uniform vec2 uTails[${MAX_DROPS}];   // capsule tail point
uniform float uK;            // smooth-union radius factor
uniform float uHR;           // height / footprint radius
uniform float uP;            // superellipse exponent for the cross section
uniform float uSquash;       // global height multiplier (press)
uniform vec3 uEdge;          // stage width, height, contact inset (device px)
out vec4 o;

float sdCapsule(vec2 p, vec2 a, vec2 b, float r){
  vec2 pa = p - a, ba = b - a;
  float t = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-4), 0., 1.);
  return length(pa - ba*t) - r;
}

void main(){
  vec2 p = gl_FragCoord.xy;
  float d = 1e5, R = 1., wetW = 1.;
  if (uPill.w > 0.5){
    vec2 q = p - uPill.xy;
    float r = uPill.w;
    float L = max(uPill.z - r, 0.);
    float ang = atan(q.y, q.x + 1e-4);
    float w = r * (0.005*sin(2.*ang + 1.3) + 0.004*sin(5.*ang + 4.1) + 0.002*sin(9.*ang + 0.7));
    for (int i = 0; i < 4; i++) w += uModes[i].x * cos(uModes[i].y * (ang - uModes[i].z));
    // gravity slump: the lower half spreads a little wider than the top
    q.x /= 1.0 + 0.025 * clamp(-q.y / r, -1., 1.);
    float dc = sdCapsule(q, vec2(-L, 0.), vec2(L, 0.), r);
    float de = (length(q / vec2(uPill.z, r)) - 1.) * r;
    d = mix(dc, de, 0.25) - w;
    R = r;
  }
  for (int i = 0; i < ${MAX_DROPS}; i++){
    if (i >= uCount) break;
    vec4 D = uDrops[i];
    vec2 T = uTails[i];
    float reach = length(p - D.xy) - D.z - length(T - D.xy);
    float k = uK * D.z + 2.0;
    if (reach > d + k) continue;
    float di = sdCapsule(p, D.xy, T, D.z);
    float hh = clamp(0.5 + 0.5*(di - d)/k, 0., 1.);
    d = mix(di, d, hh) - k*hh*(1. - hh);
    R = mix(D.z, R, hh);
    wetW = mix(D.w, wetW, hh);
  }
  // the stage edge is a wall: water pressed against it gets a flat contact face
  float wall = min(min(p.x, uEdge.x - p.x), min(p.y, uEdge.y - p.y)) - uEdge.z;
  d = max(d, -wall);
  float s = max(-d, 0.);
  float x = clamp(s / max(R, 1e-3), 0., 1.);
  float h = R * uHR * pow(max(1. - pow(1. - x, uP), 0.), 1. / uP);
  if (uDent.z > 0.){
    vec2 dq = p - uDent.xy;
    h -= uDent.z * exp(-dot(dq, dq) / (2.*uDent.w*uDent.w)) * smoothstep(0., R*0.5, s);
  }
  h *= uSquash;
  o = vec4(max(h, 0.), clamp(d, -2048., 2048.), wetW, 1.);
}`;

// Wet stains: water leaves a darkening that evaporates.
const FS_WET = `#version 300 es
precision highp float;
uniform sampler2D uPrev;
uniform highp sampler2D uH;
uniform vec2 uHRes;
uniform vec2 uRes;
uniform float uDecay;
out vec4 o;
void main(){
  vec2 uv = gl_FragCoord.xy / uRes;
  float w = texture(uPrev, uv).r;
  // evaporation is faster at thin edges: decay more where the stain is weak
  w = max(w * uDecay - (1. - uDecay) * 0.35 * (1. - w), 0.);
  ivec2 hp = ivec2(uv * uHRes);
  vec4 f = texelFetch(uH, hp, 0);
  // a lower wetting weight means a drop just sliding through: its film is thinner and narrower than a resting drop's
  float c = smoothstep(3., -6., f.g + (1. - f.b) * 16.) * mix(0.45, 1., f.b);
  o = vec4(max(w, c), 0., 0., 1.);
}`;

const FS_COPY = `#version 300 es
precision highp float;
uniform sampler2D uSrc;
uniform vec2 uRes;
out vec4 o;
void main(){ o = texture(uSrc, gl_FragCoord.xy / uRes); }`;

// Output is premultiplied. The page under the stage is not ours to redraw, so the water colour T we
// want is produced as src + (1 - a) * page: the page shows through wherever the water only tints it
// (so page text under a drop stays readable), and a rises only as far as T needs to depart from the page.
// Where the page matches the colour or surface we were given, the result is exactly T.
const FS_COMPOSITE = `#version 300 es
precision highp float;
uniform highp sampler2D uH;
uniform sampler2D uWet;
uniform sampler2D uLabel;
uniform sampler2D uSurf;
uniform bool uHasSurf;
uniform vec4 uSurfX;        // page surface uv = (p.x*x + y, p.y*z + w)
uniform float uSurfLod;     // log2 of surface texels per device px
uniform vec3 uBgCol;
uniform vec2 uRes;
uniform float uDpr;
uniform float uOs;          // scale of the optical distances (shadow, caustic) to the size of the water
uniform vec4 uLabelRect;    // cx, cy, halfW, halfH of the label area (device px)
uniform float uTime;
out vec4 o;
${NOISE}

vec4 F(vec2 p){ ivec2 q = clamp(ivec2(p), ivec2(0), ivec2(uRes) - 1); return texelFetch(uH, q, 0); }
// premultiplied RGBA; clamp-to-edge reads the canvas's transparent border outside the label
vec4 label(vec2 p){
  return texture(uLabel, (p - uLabelRect.xy) / (uLabelRect.zw * 2.) + 0.5);
}
vec3 page(vec2 p, float lod){
  if (!uHasSurf) return uBgCol;
  vec2 uv = vec2(p.x * uSurfX.x + uSurfX.y, p.y * uSurfX.z + uSurfX.w);
  return textureLod(uSurf, uv, max(lod + uSurfLod, 0.)).rgb;
}

// Environment seen in reflections. z points at the viewer, y is screen-up.
vec3 env(vec3 r){
  float up = r.y;
  // dim room: dark below the horizon of the surface, brighter toward the top of the scene
  vec3 c = mix(vec3(0.03), vec3(0.48), smoothstep(-0.6, 0.9, up));
  c = mix(c, vec3(0.75), smoothstep(0.2, 1.0, r.z) * 0.4);
  // wide window low above the top edge of the screen (HDR, it is what makes the rim streak)
  float bx = smoothstep(0.24, 0.08, abs(r.x - 0.10));
  float el = r.z;                                         // elevation above the surface plane
  float by = smoothstep(0.30, 0.55, up) * smoothstep(0.02, 0.10, el) * smoothstep(0.30, 0.20, el);
  c += vec3(70.0) * bx * by;
  // small hard lamp up and to the right
  vec3 Ls = normalize(vec3(0.55, 0.60, 0.58));
  c += vec3(80.0) * pow(max(dot(r, Ls), 0.), 150.);
  c += vec3(1.2) * pow(max(dot(r, Ls), 0.), 50.);
  return c;
}

void main(){
  vec2 p = gl_FragCoord.xy;
  vec2 uv = p / uRes;
  float px = uDpr;                         // 1 CSS px in device px
  float ox = px * uOs;
  vec4 c0 = F(p);
  float h = c0.r, d = c0.g;
  float mask = clamp(0.5 - d, 0., 1.);
  vec2 Ldir = normalize(vec2(0.45, 1.0));

  // ---------------- marks on the page: a darkening laid over whatever is there
  vec3 B = page(p, 0.);
  float lum = uHasSurf ? dot(B, vec3(0.333)) : 0.76;
  // damp concrete: pores hold water longest, so the wet edge follows the grain
  float wet = texture(uWet, uv).r;
  float thr = max(0.30 + clamp((lum - 0.76) * 2.2, -0.2, 0.3) + (hash12(floor(p / (2.*px))) - 0.5) * 0.06, 0.2);
  float wetA = smoothstep(thr - 0.12, thr + 0.12, wet);
  float ring = smoothstep(thr - 0.12, thr, wet) * smoothstep(thr + 0.25, thr + 0.05, wet);
  float keep = (1.0 - 0.13 * wetA) * (1.0 - 0.05 * ring);
  // light from upper right: tight contact shadow falling down-left
  float sd = F(p + Ldir * 7.0 * ox).g;
  float shadow = smoothstep(10.0*ox, -8.0*ox, sd);
  float sd2 = F(p + Ldir * 2.5 * ox).g;
  float shadow2 = smoothstep(3.0*ox, -2.0*ox, sd2);
  keep *= 1.0 - 0.17*shadow - 0.20*shadow2;
  keep *= 1.0 - 0.20 * exp(-max(d, 0.) / (1.2*px));
  float shA = 1.0 - keep;
  vec3 shC = vec3(0.035, 0.04, 0.05) * shA;
  if (mask <= 0.){ o = vec4(shC, shA); return; }

  // ---------------- water
  float hl = F(p - vec2(1,0)).r, hr = F(p + vec2(1,0)).r;
  float hd = F(p - vec2(0,1)).r, hu = F(p + vec2(0,1)).r;
  vec2 g = vec2(hr - hl, hu - hd) * 0.5;
  vec3 n = normalize(vec3(-g, 1.));
  vec3 I = vec3(0., 0., -1.);
  vec3 T = refract(I, n, 1.0/1.333);
  vec2 off = T.xy / max(-T.z, 0.3) * h * 1.5;
  vec2 sp = p + off;
  // what the water sits on, seen through it: grain softened, contrast raised by the wet surface
  float steep = smoothstep(0.16, 0.92, 1.0 - n.z);
  vec3 base = page(sp, 2.6 + 3.5 * steep);
  float bl = dot(base, vec3(0.333));
  base = mix(vec3(bl), vec3(0.80), 0.25) + (base - bl) * 0.4;
  // the label is only gently lensed so it stays legible as a button face
  vec4 inkW = label(p + off * 0.15);
  base = base * (1.0 - inkW.a * 0.93) + inkW.rgb * 0.92 * 0.93;
  // light entering on the lit side focuses onto the floor on the far side
  float cz = F(sp + Ldir * 12.0 * ox).g;
  float focus = smoothstep(-2.0*ox, -26.0*ox, cz);
  vec3 under = base * (0.93 + 0.20 * focus);
  // the upper half of the body looks into the dim room and reads a little darker
  vec2 lq = (p - uLabelRect.xy) / uLabelRect.zw;
  float inPill = smoothstep(1.25, 0.9, length(lq));
  under *= 1.0 - inPill * 0.10 * smoothstep(-0.3, 0.7, lq.y);
  // total internal reflection: the steep rim goes dark, mostly on the side facing away from the light
  float away = 0.5 - 0.5 * dot(normalize(n.xy + 1e-5), Ldir);
  under *= 1.0 - steep * mix(0.30, 0.88, away);
  float fres = 0.02 + 0.98 * pow(1.0 - clamp(n.z, 0., 1.), 5.0);
  vec3 R = reflect(I, n);
  vec3 water = mix(under, env(R), fres);
  water = water / (1.0 + max(water - 0.85, 0.0) * 0.9);
  water = clamp(water + (hash12(p + fract(uTime)*100.) - 0.5) / 255., 0., 1.);

  vec3 Bs = max(B, vec3(0.02));
  vec3 q = water / Bs;
  vec3 up = (water - B) / max(1.0 - B, vec3(0.02));
  float a = clamp(max(0.12, max(1.0 - min(min(q.r, q.g), q.b), max(max(up.r, up.g), up.b))), 0., 1.);
  vec3 src = clamp(water - (1.0 - a) * B, 0., a);
  o = vec4(src * mask + shC * (1. - mask), a * mask + shA * (1. - mask));
}`;

// ---------------------------------------------------------------- audio (this effect's own)
const waterAudio = { muted: false };
let actx = null;
function audio(){
  if (waterAudio.muted) return null;
  if (!actx){ try { actx = new (window.AudioContext || window.webkitAudioContext)(); } catch { return null; } }
  if (actx.state === 'suspended') actx.resume();
  return actx;
}
let noiseBuf = null;
function noise(a){
  if (noiseBuf && noiseBuf.sampleRate === a.sampleRate) return noiseBuf;
  const len = Math.floor(a.sampleRate * 0.5), b = a.createBuffer(1, len, a.sampleRate), d = b.getChannelData(0);
  for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
  return (noiseBuf = b);
}
function blip(f0, f1, dur, gain, delay = 0){
  const a = audio(); if (!a) return;
  const t = a.currentTime + delay;
  const o = a.createOscillator(), g = a.createGain();
  o.type = 'sine';
  o.frequency.setValueAtTime(f0, t);
  o.frequency.exponentialRampToValueAtTime(f1, t + dur);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(gain, t + 0.004);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  o.connect(g).connect(a.destination);
  o.start(t); o.stop(t + dur + 0.02);
}
// Every pop is the same water sound at a different size: a soft thump and hiss (kept low)
// under a patter of tiny droplet plinks, which is what actually sounds wet.
function waterSound(size){
  const a = audio(); if (!a) return;
  const t = a.currentTime;
  const z = Math.max(0.2, Math.min(size, 1.6));
  const loud = z / 1.6;
  const f = 620 / Math.pow(z, 0.35);
  blip(f, f * 0.3, 0.05 + 0.04 * z, 0.025 + 0.075 * loud);
  const src = a.createBufferSource(); src.buffer = noise(a);
  const bp = a.createBiquadFilter(); bp.type = 'bandpass';
  bp.frequency.setValueAtTime(2600 / Math.sqrt(z), t);
  bp.frequency.exponentialRampToValueAtTime(900, t + 0.1 + 0.25 * loud);
  bp.Q.value = 0.9;
  const g = a.createGain();
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(0.02 + 0.08 * loud, t + 0.008);
  g.gain.exponentialRampToValueAtTime(0.0001, t + 0.12 + 0.3 * loud);
  src.connect(bp).connect(g).connect(a.destination);
  src.start(t); src.stop(t + 0.5);
  const n = Math.max(2, Math.round(2 + 3.6 * z));
  const spread = 0.12 + 0.2 * loud;
  const pitch = 1 + (1 - loud) * 0.5;                  // smaller drops patter a little higher
  for (let i = 0; i < n; i++){
    blip((900 + Math.random() * 1400) * pitch, (400 + Math.random() * 300) * pitch, 0.035, 0.045 + 0.015 * Math.random(), 0.02 + Math.random() * spread);
  }
}
let lastPlip = 0;
function plip(){
  const a = audio(); if (!a) return;
  if (a.currentTime - lastPlip < 0.035) return;
  lastPlip = a.currentTime;
  const f = 500 + Math.random() * 700;
  blip(f * 0.7, f * 1.35, 0.05, 0.06);
}
const sounds = {
  splash(){ waterSound(1.6); },
  // r in reference px (a pill is ~55); the big splash is size 1.6
  dropPop(r, flick){
    if (flick){ blip(1300 + Math.random() * 600, 700, 0.03, 0.04); return; }
    waterSound(r / 12);
  },
  plip,
  reform(){ blip(260, 720, 0.14, 0.22); },
};

// ---------------------------------------------------------------- helpers
const rand = (a, b) => a + Math.random() * (b - a);
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');

let probe = null;
function parseColor(css){
  if (!css) return null;
  let m = css.match(/^rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)(?:\s*[,/]\s*([\d.]+%?))?\s*\)$/);
  if (m){
    const a = m[4] === undefined ? 1 : m[4].endsWith('%') ? parseFloat(m[4]) / 100 : parseFloat(m[4]);
    return [m[1] / 255, m[2] / 255, m[3] / 255, a];
  }
  m = css.match(/^color\(srgb\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s*\/\s*([\d.]+%?))?\s*\)$/);
  if (m){
    const a = m[4] === undefined ? 1 : m[4].endsWith('%') ? parseFloat(m[4]) / 100 : parseFloat(m[4]);
    return [+m[1], +m[2], +m[3], a];
  }
  // any other colour syntax: let a 2D context normalise it
  if (!probe){ probe = document.createElement('canvas').getContext('2d'); }
  probe.fillStyle = '#010203'; probe.fillStyle = css;
  const s = probe.fillStyle;
  if (s === '#010203') return null;
  if (s[0] === '#') return [parseInt(s.slice(1, 3), 16) / 255, parseInt(s.slice(3, 5), 16) / 255, parseInt(s.slice(5, 7), 16) / 255, 1];
  return s === css ? null : parseColor(s);
}

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

function inFixedContext(el){
  for (let e = el; e && e !== document.documentElement; e = e.parentElement){
    const p = getComputedStyle(e).position;
    if (p === 'fixed' || p === 'sticky') return true;
  }
  return false;
}

function compileProgram(gl, fs, vs = VS){
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

// ---------------------------------------------------------------- concrete texture
// The procedural concrete the water was designed on, as a canvas of w x h CSS px (at the given pixel
// ratio). Use it as a section background and pass it to attachWater as { surface: { image, rect } }.
function concreteCanvas(w, h, ratio = Math.min(devicePixelRatio || 1, 2)){
  const out = document.createElement('canvas');
  const pw = Math.max(1, Math.round(w * ratio)), ph = Math.max(1, Math.round(h * ratio));
  out.width = pw; out.height = ph;
  const c = document.createElement('canvas');
  c.width = pw; c.height = ph;
  const gl = c.getContext('webgl2', { alpha: false, antialias: false, depth: false, stencil: false, preserveDrawingBuffer: true });
  const x = out.getContext('2d');
  if (!gl){ x.fillStyle = '#c3c2bf'; x.fillRect(0, 0, pw, ph); return out; }
  try {
    const prog = compileProgram(gl, FS_CONCRETE);
    gl.viewport(0, 0, pw, ph);
    gl.useProgram(prog.p);
    gl.uniform1f(prog.u.uDpr, pw / w);
    gl.uniform2f(prog.u.uRes, pw, ph);
    gl.bindVertexArray(gl.createVertexArray());
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    x.drawImage(c, 0, 0);
  } catch { x.fillStyle = '#c3c2bf'; x.fillRect(0, 0, pw, ph); }
  gl.getExtension('WEBGL_lose_context')?.loseContext();
  return out;
}

// ---------------------------------------------------------------- component
function attachWater(button, opts = {}){
  const noop = () => {};
  const unsupported = { supported: false, state: 'unsupported', detach: noop, refresh: noop, _manual: noop, _advance: noop };
  const canvas = document.createElement('canvas');
  canvas.setAttribute('aria-hidden', 'true');
  Object.assign(canvas.style, { position: 'absolute', left: '0px', top: '0px', margin: '0', padding: '0', border: '0', display: 'block', pointerEvents: 'none', zIndex: String(opts.zIndex ?? 10) });
  const gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: true, antialias: false, depth: false, stencil: false });
  if (!gl || !gl.getExtension('EXT_color_buffer_float')) return unsupported;
  const sound = opts.sound === false ? { splash: noop, dropPop: noop, plip: noop, reform: noop } : sounds;
  const plop = () => sound.plip();

  // ---- GL resources (rebuilt from scratch after a lost context)
  let P, vao, labelTex, surfTex, hTex, hFb, wetTex = [], wetFb = [], wetW = 0, wetH = 0, wetIdx = 0;
  let MAX_DROPS = 32, dropBuf, tailBuf;
  const modeBuf = new Float32Array(16);
  let surfSrc = null, surfW = 0, surfH = 0;
  function initGL(){
    gl.getExtension('EXT_color_buffer_float');
    MAX_DROPS = Math.max(32, Math.min(96, Math.floor((gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_VECTORS) - 32) / 2)));
    dropBuf = new Float32Array(MAX_DROPS * 4); tailBuf = new Float32Array(MAX_DROPS * 2);
    P = { field: compileProgram(gl, fsField(MAX_DROPS)), wet: compileProgram(gl, FS_WET), comp: compileProgram(gl, FS_COMPOSITE), copy: compileProgram(gl, FS_COPY) };
    vao = gl.createVertexArray();
    labelTex = gl.createTexture(); surfTex = gl.createTexture();
    // both are sampled by the composite before they have content; give them one transparent texel
    for (const t of [labelTex, surfTex]){
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    }
    hTex = hFb = null; wetTex = []; wetFb = []; wetIdx = 0; surfSrc = null;
  }
  const makeTex = (w, h, internal, format, type, filter) => {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, null);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, filter], [gl.TEXTURE_MAG_FILTER, filter], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
    return t;
  };
  const makeTarget = t => {
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return fb;
  };
  const draw = (prog, fb, w, h) => {
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.viewport(0, 0, w, h);
    gl.useProgram(prog.p);
    gl.bindVertexArray(vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  };
  const bindTex = (unit, t) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, t); };
  const mipmapped = () => {
    gl.generateMipmap(gl.TEXTURE_2D);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
  };

  try { initGL(); } catch (err) {
    console.warn('ElementalFX water: shaders failed, leaving the button as it is.', err);
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return unsupported;
  }

  // ---- geometry. Physics runs in button-local CSS px (origin at the button's top-left, y down);
  // the stage is the canvas, placed so the button sits at (offX, offY) inside it.
  const G = { fixed: false, bx: NaN, by: NaN, bw: 0, bh: 0, stageW: 0, stageH: 0, offX: 0, offY: 0 };
  let bounds = { w: 0, h: 0 }, cbOrigin = { x: 0, y: 0 };
  let dpr = 1, dprY = 1, W = 0, H = 0, renderScale = 1;
  let bgCol = [1, 1, 1], labelKey = '';

  const pill = {
    cx: 0, cy: 0, hw: 0, hh: 0,
    scale: 1, scaleV: 0, scaleT: 1,
    sx: 1, sxV: 0, sy: 1, syV: 0,
    press: 0, pressV: 0, pressT: 0,
    dentX: 0, dentY: 0,
    modes: [0, 1, 2, 3].map(() => ({ amp: 0, m: 2, phi: 0, w: 0, t: 0 })),
    absorbed: 0,
  };
  const kk = () => pill.hh / 55;

  // ---- the button: invisible but still there
  const saved = button.getAttribute('style');
  const hideStyle = { background: 'transparent', borderColor: 'transparent', color: 'transparent', boxShadow: 'none', textShadow: 'none' };
  let hidden = false;
  function hideButton(){
    hidden = true;
    Object.assign(button.style, hideStyle);
  }
  function showButton(){
    hidden = false;
    if (saved === null) button.removeAttribute('style'); else button.setAttribute('style', saved);
  }
  const savedPE = () => { const t = document.createElement('div'); t.setAttribute('style', saved || ''); return t.style.pointerEvents; };
  const basePE = savedPE();
  let peOff = false;
  function disablePointer(){
    clearTimeout(peTimer);
    if (phase === 'idle' || peOff || !hidden) return;
    peOff = true; button.style.pointerEvents = 'none';
  }
  function enablePointer(){
    clearTimeout(peTimer);
    if (!peOff) return;
    peOff = false; button.style.pointerEvents = basePE;
  }
  let peTimer = 0;

  // ---- layout
  function measureBounds(){
    G.fixed = inFixedContext(button);
    canvas.style.position = G.fixed ? 'fixed' : 'absolute';
    const de = document.documentElement;
    const disp = canvas.style.display;
    canvas.style.display = 'none';
    bounds = G.fixed ? { w: de.clientWidth, h: de.clientHeight }
      : { w: de.clientWidth, h: Math.max(de.scrollHeight, document.body ? document.body.scrollHeight : 0) };
    canvas.style.display = disp;
    // where left:0/top:0 lands (normally the document origin, unless an ancestor is positioned)
    canvas.style.left = '0px'; canvas.style.top = '0px';
    const r = canvas.getBoundingClientRect();
    cbOrigin = { x: r.left + (G.fixed ? 0 : scrollX), y: r.top + (G.fixed ? 0 : scrollY) };
    G.bx = NaN;
  }

  let btnRect = null;
  function layout(r = button.getBoundingClientRect()){
    btnRect = r;
    const bx = r.left + (G.fixed ? 0 : scrollX), by = r.top + (G.fixed ? 0 : scrollY);
    const bw = r.width, bh = r.height;
    if (bw < 2 || bh < 2){ canvas.style.visibility = 'hidden'; G.bx = bx; G.by = by; return; }
    canvas.style.visibility = '';
    let x0, y0, x1, y1;
    if (opts.area){
      const a = opts.area.getBoundingClientRect();
      const ax = a.left + (G.fixed ? 0 : scrollX), ay = a.top + (G.fixed ? 0 : scrollY);
      x0 = Math.floor(Math.min(bx, Math.max(0, ax))); y0 = Math.floor(Math.min(by, Math.max(0, ay)));
      x1 = Math.ceil(Math.max(bx + bw, Math.min(bounds.w, ax + a.width))); y1 = Math.ceil(Math.max(by + bh, Math.min(bounds.h, ay + a.height)));
    } else {
      const M = opts.margin ?? Math.round(Math.max(140, 1.6 * bh));
      // the stage stays inside the document, so it can never add scrolling
      x0 = Math.floor(Math.min(bx, Math.max(0, bx - M))); y0 = Math.floor(Math.min(by, Math.max(0, by - M)));
      x1 = Math.ceil(Math.max(bx + bw, Math.min(bounds.w, bx + bw + M))); y1 = Math.ceil(Math.max(by + bh, Math.min(bounds.h, by + bh + M)));
    }
    const sw = x1 - x0, sh = y1 - y0;
    const sizeChanged = sw !== G.stageW || sh !== G.stageH;
    const btnChanged = Math.abs(bw - G.bw) > 0.25 || Math.abs(bh - G.bh) > 0.25;
    Object.assign(G, { bx, by, bw, bh, stageW: sw, stageH: sh, offX: bx - x0, offY: by - y0 });
    Object.assign(canvas.style, { left: `${x0 - cbOrigin.x}px`, top: `${y0 - cbOrigin.y}px` });
    if (btnChanged){
      const ox = pill.cx, oy = pill.cy;
      pill.hw = bw / 2; pill.hh = bh / 2; pill.cx = bw / 2; pill.cy = bh / 2;
      // drops keep their place relative to the button's centre
      for (const d of drops){ d.x += pill.cx - ox; d.y += pill.cy - oy; }
    }
    if (sizeChanged){
      Object.assign(canvas.style, { width: `${sw}px`, height: `${sh}px` });
      resizeGL();
    } else if (btnChanged) buildLabel();
    dirty = true;
    wake();
  }

  // cheap per-frame / per-scroll check; only a real move re-lays the stage out
  function place(){
    const r = button.getBoundingClientRect();
    btnRect = r;
    const bx = r.left + (G.fixed ? 0 : scrollX), by = r.top + (G.fixed ? 0 : scrollY);
    if (Math.abs(bx - G.bx) > 0.25 || Math.abs(by - G.by) > 0.25 || Math.abs(r.width - G.bw) > 0.25 || Math.abs(r.height - G.bh) > 0.25) layout(r);
  }

  function resizeGL(){
    if (lost || gl.isContextLost() || !G.stageW) return;
    let d = Math.min(devicePixelRatio || 1, 2) * renderScale;
    const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    d = Math.min(d, maxTex / G.stageW, maxTex / G.stageH);
    canvas.width = Math.max(1, Math.round(G.stageW * d)); canvas.height = Math.max(1, Math.round(G.stageH * d));
    // the browser may silently give a smaller buffer than asked for; everything must use the real one
    W = gl.drawingBufferWidth; H = gl.drawingBufferHeight;
    dpr = W / G.stageW; dprY = H / G.stageH;
    // keep the current stains so a quality change or a resize doesn't wipe them
    const oldWet = wetTex[wetIdx] || null;
    [hTex, ...wetTex].forEach(t => t && t !== oldWet && gl.deleteTexture(t));
    [hFb, ...wetFb].forEach(f => f && gl.deleteFramebuffer(f));
    hTex = makeTex(W, H, gl.RGBA32F, gl.RGBA, gl.FLOAT, gl.NEAREST);
    hFb = makeTarget(hTex);
    wetW = Math.max(1, W >> 1); wetH = Math.max(1, H >> 1);
    wetTex = [0, 1].map(() => makeTex(wetW, wetH, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR));
    wetFb = wetTex.map(makeTarget);
    wetFb.forEach(f => { gl.bindFramebuffer(gl.FRAMEBUFFER, f); gl.clearColor(0, 0, 0, 1); gl.clear(gl.COLOR_BUFFER_BIT); });
    wetIdx = 0;
    if (oldWet){
      gl.useProgram(P.copy.p);
      bindTex(0, oldWet); gl.uniform1i(P.copy.u.uSrc, 0);
      gl.uniform2f(P.copy.u.uRes, wetW, wetH);
      draw(P.copy, wetFb[wetIdx], wetW, wetH);
      gl.deleteTexture(oldWet);
    }
    buildLabel();
    fences.forEach(f => gl.deleteSync(f)); fences.length = 0;
    warmup = 10; win = winSlow = 0;
    dirty = true;
  }

  // The label is the button's own text, drawn where the browser lays it out, in its font and colour.
  function labelText(){ return (button.innerText || button.textContent || '').replace(/\s+/g, ' ').trim(); }
  function buildLabel(){
    if (lost || gl.isContextLost() || !G.bw) return;
    const cs = getComputedStyle(button);
    const pad = 4;
    const w = Math.max(1, Math.round((G.bw + pad * 2) * dpr)), h = Math.max(1, Math.round((G.bh + pad * 2) * dpr));
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const x = c.getContext('2d');
    const text = labelText();
    const fsize = parseFloat(cs.fontSize) || 16;
    x.font = `${cs.fontStyle} ${cs.fontWeight} ${fsize * dpr}px ${cs.fontFamily}`;
    if ('letterSpacing' in x && cs.letterSpacing !== 'normal') x.letterSpacing = `${(parseFloat(cs.letterSpacing) || 0) * dpr}px`;
    x.textAlign = 'center'; x.textBaseline = 'alphabetic';
    // centre on the text's laid-out box, so padding and alignment come from the page's CSS
    let cx = G.bw / 2, cy = G.bh / 2;
    const range = document.createRange();
    range.selectNodeContents(button);
    const rr = range.getBoundingClientRect();
    const br = button.getBoundingClientRect();
    if (rr.width > 0 && rr.height > 0){ cx = rr.left + rr.width / 2 - br.left; cy = rr.top + rr.height / 2 - br.top; }
    const m = x.measureText(text);
    const asc = m.fontBoundingBoxAscent ?? fsize * dpr * 0.8, desc = m.fontBoundingBoxDescent ?? fsize * dpr * 0.2;
    const px = (cx + pad) * dpr, py = (cy + pad) * dpr + (asc - desc) / 2;
    const col = parseColor(opts.labelColor || cs.color) || [0.3, 0.3, 0.3, 1];
    const dk = opts.labelColor ? 1 : 0.82;
    const ink = `rgba(${Math.round(col[0] * 255 * dk)},${Math.round(col[1] * 255 * dk)},${Math.round(col[2] * 255 * dk)},${0.97 * Math.max(col[3], 0.6)})`;
    // soft light catch on the lower edge of the glyphs
    x.filter = `blur(${0.6 * dpr}px)`;
    x.fillStyle = 'rgba(235,235,235,0.55)';
    x.fillText(text, px, py + 1.2 * dpr);
    x.fillStyle = ink;
    x.fillText(text, px, py);
    gl.bindTexture(gl.TEXTURE_2D, labelTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, c);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    mipmapped();
    labelKey = text + '|' + cs.font + '|' + cs.color;
    bgCol = opts.background ? (parseColor(opts.background) || [1, 1, 1]).slice(0, 3) : pageColor(button.parentElement);
    dirty = true;
  }

  function syncSurface(){
    const s = opts.surface;
    const img = s && s.image;
    const w = img ? (img.naturalWidth || img.videoWidth || img.displayWidth || img.width || 0) : 0;
    const h = img ? (img.naturalHeight || img.videoHeight || img.displayHeight || img.height || 0) : 0;
    if (!img || !w || !h || (img.complete === false)){ surfSrc = null; return false; }
    if (img === surfSrc && w === surfW && h === surfH) return true;
    try {
      gl.bindTexture(gl.TEXTURE_2D, surfTex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      mipmapped();
      surfSrc = img; surfW = w; surfH = h;
      dirty = true;
      return true;
    } catch (err) {
      // a cross-origin image can't be read: fall back to the flat colour
      console.warn('ElementalFX water: surface image unusable', err);
      opts = Object.assign({}, opts, { surface: null });
      surfSrc = null; return false;
    }
  }

  // ---- simulation
  let phase = 'idle';      // idle | pop | scatter | reform
  let phaseT = 0, simT = 0, lastBurst = 0;
  let drops = [];
  let V0 = 1;

  function kickMode(amp, m){
    const md = pill.modes.reduce((a, b) => (a.amp <= b.amp ? a : b));
    md.amp = amp; md.m = m; md.phi = Math.random() * TAU; md.w = 14 + m * 4; md.t = 0;
  }

  function pillSdf(x, y){
    const r = pill.hh * pill.scale * pill.sy;
    const hw = pill.hw * pill.scale * pill.sx;
    const L = Math.max(hw - r, 0);
    const qx = Math.max(Math.abs(x - pill.cx) - L, 0), qy = y - pill.cy;
    return Math.hypot(qx, qy) - r;
  }

  function pop(ox = pill.cx, oy = pill.cy){
    if (phase !== 'idle') return;
    phase = 'pop'; phaseT = 0;
    pill.pressT = 0;
    sound.splash();
    const k = kk();
    // throw only as far as the stage has room for: the nearest wall sets the scale
    const room = Math.min(G.offX, G.stageW - G.offX - G.bw, G.offY, G.stageH - G.offY - G.bh);
    const splash = Math.max(0.35, Math.min(1, 0.8 * room / (250 * k)));
    const sets = [[10, 13, 19], [22, 7, 11.5], [30, 2.4, 5.5]];
    drops = [];
    const L = pill.hw - pill.hh;
    for (const [count, r0, r1] of sets){
      for (let i = 0; i < count; i++){
        const r = Math.max(rand(r0, r1) * k, r0 < 3 ? 1.4 : 0);
        let x, y;
        for (let tries = 0; tries < 30; tries++){
          x = rand(-pill.hw, pill.hw); y = rand(-pill.hh, pill.hh);
          const qx = Math.max(Math.abs(x) - L, 0);
          if (Math.hypot(qx, y) < pill.hh - r * 0.9) break;
        }
        const dx = x / pill.hw, dy = y / pill.hh;
        const len = Math.hypot(dx, dy) || 1;
        // rupture starts under the finger: fragments are thrown away from it and from the centre
        const ex = pill.cx + x - ox, ey = pill.cy + y - oy, el = Math.hypot(ex, ey) || 1;
        const mx = dx / len + ex / el, my = dy / len + ey / el;
        const ang = Math.atan2(my, mx) + rand(-0.4, 0.4);
        const small = r < 6 * k;
        const speed = splash * rand(170, 430) * k * (reduceMotion.matches ? 0.35 : 1) * (small ? 1.6 : r < 12 * k ? 1.15 : 0.8) * (0.55 + 0.6 * len);
        drops.push({
          x: pill.cx + x, y: pill.cy + y, r, vol: r * r * r,
          vx: Math.cos(ang) * speed, vy: Math.sin(ang) * speed,
          mu: small ? rand(3.2, 5.5) : rand(4.5, 6.5),
          into: null, delay: 0, absorbing: false, born: simT,
        });
      }
    }
    // Conserve water. Drops and pill share the same cross-section profile, so volumes compare in units
    // where a drop of radius r holds r^3: the pill (half-cylinder of radius R and length 2L, plus two
    // quarter-sphere caps) holds 1.5 R^2 L + R^3. The spray keeps its size; the larger pieces take the rest.
    const R = pill.hh, Lp = Math.max(pill.hw - pill.hh, 0);
    // the splash carries 60% of the pill's geometric volume: at full volume the pieces swamp the stage
    const Vpill = 0.6 * (1.5 * R * R * Lp + R * R * R);
    const spray = drops.filter(d => d.r < 6 * k), body = drops.filter(d => d.r >= 6 * k);
    const vSpray = spray.reduce((s, d) => s + d.vol, 0), vBody = body.reduce((s, d) => s + d.vol, 0);
    const grow = Math.cbrt(Math.max(Vpill - vSpray, vBody) / vBody);
    // heavier pieces are thrown harder and slide further, so they separate before they can coalesce
    for (const d of body){
      d.r *= grow; d.vol = d.r * d.r * d.r;
      d.vx *= 1.7; d.vy *= 1.7;
      d.mu = rand(3.4, 4.6);
      d.born = simT + 0.25;
    }
    V0 = drops.reduce((s, d) => s + d.vol, 0);
    pill.absorbed = 0;
    lastBurst = simT;
    wake();
  }

  // While regrouping, the half-formed pill is just another body of water: hand its volume back as a drop.
  function cancelReform(){
    if (phase !== 'reform') return null;
    phase = 'scatter'; phaseT = 0;
    let body = null;
    if (pill.absorbed > 0.5){
      const r = Math.cbrt(pill.absorbed);
      body = { x: pill.cx, y: pill.cy, r, vol: pill.absorbed, vx: 0, vy: 0, mu: 6, into: null, delay: 0, absorbing: false, born: simT - 1 };
      drops.push(body);
    }
    pill.absorbed = 0; pill.scale = 0; pill.scaleV = 0; pill.scaleT = 0;
    for (const d of drops) d.absorbing = false;
    return body;
  }

  // Pop one droplet into smaller ones that behave exactly like the first splash.
  function burstDrop(d, ox, oy){
    const k = kk();
    lastBurst = simT;
    const room = MAX_DROPS - drops.length + 1;
    let n = d.r > 28 * k ? Math.round(rand(6, 9)) : d.r > 10 * k ? Math.round(rand(4, 7)) : d.r > 5 * k ? Math.round(rand(3, 4)) : 2;
    n = Math.min(n, room);
    // too small to split (or no room left): it just gets flicked away from the finger
    if (d.r < Math.max(2.6 * k, 2.2) || n < 2){
      const ex = d.x - ox, ey = d.y - oy, el = Math.hypot(ex, ey) || 1;
      const sp = rand(120, 220) * k;
      d.vx += ex / el * sp; d.vy += ey / el * sp;
      sound.dropPop(d.r / k, true);
      return;
    }
    const w = Array.from({ length: n }, () => Math.pow(rand(0.35, 1), 2));
    const ws = w.reduce((a, b) => a + b, 0);
    const idx = drops.indexOf(d);
    // push the whole burst slightly away from the finger
    const cx = d.x - ox, cy = d.y - oy, cl = Math.hypot(cx, cy) || 1;
    const a0 = rand(0, TAU);
    const parts = w.map((wi, i) => {
      const vol = d.vol * wi / ws;
      const r = Math.cbrt(vol);
      // pieces leave in an even ring so neighbours don't land on each other
      const a = a0 + (i / n) * TAU + rand(-0.25, 0.25) * (TAU / n);
      const off = d.r * rand(0.25, 0.45);
      const x = d.x + Math.cos(a) * off, y = d.y + Math.sin(a) * off;
      const mx = Math.cos(a) + cx / cl * 0.35, my = Math.sin(a) + cy / cl * 0.35;
      const ml = Math.hypot(mx, my) || 1;
      const mu = r < 6 * k ? rand(3.2, 5.0) : rand(3.4, 4.6);
      // with exponential friction a drop slides v/mu, so pick the speed from where it should come to rest:
      // clear of the parent's footprint and of its own neighbours, however big the parent was
      const travel = (d.r * rand(1.2, 2.0) + r * 1.2) * (reduceMotion.matches ? 0.5 : 1);
      const speed = travel * mu;
      return {
        x, y, r, vol, vx: d.vx * 0.2 + mx / ml * speed, vy: d.vy * 0.2 + my / ml * speed,
        mu, into: null, delay: 0, absorbing: false,
        born: simT + Math.min(0.1 + d.r / (120 * k), 0.4),   // bigger bursts need longer to separate
      };
    });
    // anything already sliding into this drop now has nothing to merge with
    for (const o of drops) if (o.into === d) o.into = null;
    drops.splice(idx, 1, ...parts);
    sound.dropPop(d.r / k, false);
  }

  // The drop under a point. Touching anything while it regroups stops the regroup, and the
  // half-formed pill becomes a drop of its own.
  function pickDrop(x, y){
    if (phase !== 'scatter' && phase !== 'reform') return null;
    let target = null, best = Infinity;
    for (const d of drops){
      if (d.into || d.absorbing) continue;
      const g = Math.hypot(d.x - x, d.y - y) - d.r;
      const slop = Math.max(6, 14 - d.r * 0.3);     // small targets get a bigger touch margin
      if (g < slop && g < best){ best = g; target = d; }
    }
    if (!target && phase === 'reform' && pill.scale > 0.05 && pillSdf(x, y) < 4) return cancelReform();
    if (target && phase === 'reform') cancelReform();
    return target;
  }
  function overDrop(x, y){
    if (phase !== 'scatter' && phase !== 'reform') return false;
    if (phase === 'reform' && pill.scale > 0.05 && pillSdf(x, y) < 4) return true;
    return drops.some(d => !d.into && !d.absorbing && Math.hypot(d.x - x, d.y - y) - d.r < Math.max(6, 14 - d.r * 0.3));
  }

  // pointerId -> { drop, x, y, x0, y0, dragging }
  const grabs = new Map();

  // Follow merges: if the held drop slid into a bigger one, the finger is now holding the bigger one.
  function resolveGrab(g){
    let d = g.drop;
    while (d && d.into) d = d.into;
    if (d && (d.dead || !drops.includes(d))) d = null;
    if (d !== g.drop){
      if (g.drop){ g.drop.held = false; g.drop.grab = null; }
      if (d){ d.held = g.dragging; d.grab = g; }
      g.drop = d;
    }
    return d;
  }

  function burstLargest(){
    if (phase === 'reform'){ const body = cancelReform(); if (body){ burstDrop(body, body.x, body.y + 1); return; } }
    let big = null;
    for (const d of drops) if (!d.into && (!big || d.vol > big.vol)) big = d;
    if (big) burstDrop(big, big.x, big.y + 1);
  }

  function startReform(){
    phase = 'reform'; phaseT = 0;
    let maxD = 1;
    for (const d of drops) maxD = Math.max(maxD, Math.hypot(d.x - pill.cx, d.y - pill.cy));
    for (const d of drops){
      const dist = Math.hypot(d.x - pill.cx, d.y - pill.cy);
      d.delay = rand(0, 0.25) + (dist / maxD) * 0.12;
      d.vx = d.vy = 0;
      d.absorbing = false;
    }
    pill.scale = 0; pill.scaleV = 0; pill.scaleT = 0;
  }

  function spring(x, v, target, k, c, dt){
    const a = -k * (x - target) - c * v;
    v += a * dt; x += v * dt;
    return [x, v];
  }

  function step(dt){
    phaseT += dt; simT += dt;
    for (const d of drops){
      if (!d.wob) continue;
      d.wobT += dt;
      d.wob *= Math.exp(-dt / 0.28);
      if (d.wob < 0.004) d.wob = 0;
    }

    [pill.sx, pill.sxV] = spring(pill.sx, pill.sxV, 1 + 0.05 * pill.press, 420, 13, dt);
    [pill.sy, pill.syV] = spring(pill.sy, pill.syV, 1 - 0.05 * pill.press, 420, 13, dt);
    [pill.press, pill.pressV] = spring(pill.press, pill.pressV, pill.pressT, 600, 38, dt);
    for (const m of pill.modes){ m.t += dt; m.amp *= Math.exp(-dt * 3.2); }

    if (phase === 'idle'){
      pill.scale = 1;
    } else if (phase === 'pop'){
      pill.scale = Math.max(0, 1 - phaseT / 0.07);
      if (phaseT > 0.07){ phase = 'scatter'; phaseT = 0; }
    }

    if (phase === 'pop' || phase === 'scatter'){
      const k = kk();
      for (const d of drops){
        if (d.held && d.grab && !d.into){
          // critically damped follow: the drop lags the finger a little, so it stretches like liquid
          const w = 30;
          d.vx += (w * w * (d.grab.x - d.x) - 2 * w * d.vx) * dt;
          d.vy += (w * w * (d.grab.y - d.y) - 2 * w * d.vy) * dt;
          d.x += d.vx * dt; d.y += d.vy * dt;
          wallContact(d, dt);
          continue;
        }
        const f = Math.exp(-d.mu * dt);
        d.vx *= f; d.vy *= f;
        const sp = Math.hypot(d.vx, d.vy);
        if (sp < 22 * k){ d.vx *= Math.exp(-dt * 18); d.vy *= Math.exp(-dt * 18); }
        d.x += d.vx * dt; d.y += d.vy * dt;
        wallContact(d, dt);
      }
      if (phase === 'scatter') mergeDrops(dt);
      for (const g of grabs.values()) resolveGrab(g);
      // nothing regroups while a drop is being held
      if (grabs.size) lastBurst = simT;
      const moving = drops.some(d => Math.hypot(d.vx, d.vy) > 6);
      const quiet = simT - lastBurst;
      if (phase === 'scatter' && ((quiet > REGROUP_AFTER && !moving) || quiet > REGROUP_AFTER + 1)) startReform();
    } else if (phase === 'reform'){
      reformStep(dt);
    }
  }

  // Water meeting the stage edge. It doesn't bounce: the motion into the wall is absorbed, the drop
  // spreads sideways along the wall as far as it is pushed into it, and a hard hit sets it wobbling.
  // A dragged drop can be pressed into the edge and flattens against it instead of leaving the stage.
  function wallContact(d, dt){
    const k = kk(), pad = 2;
    const x0 = -G.offX, x1 = G.stageW - G.offX, y0 = -G.offY, y1 = G.stageH - G.offY;
    let ex = 0, ey = 0;
    const walls = [[d.x - x0, 1, 0], [x1 - d.x, -1, 0], [d.y - y0, 0, 1], [y1 - d.y, 0, -1]];
    for (const [dist, nx, ny] of walls){
      let pen = d.r + pad - dist;                        // how far the drop's round outline crosses the edge
      if (pen <= 0) continue;
      const vn = d.vx * nx + d.vy * ny;                  // < 0: moving into the wall
      if (vn < 0){
        d.vx -= vn * nx; d.vy -= vn * ny;
        // sliding along a wall drags a little
        const f = Math.exp(-dt * 3);
        d.vx = d.vx * (nx ? 1 : f); d.vy = d.vy * (ny ? 1 : f);
        const kick = Math.min(0.2, -vn / (1600 * k));
        if (kick > 0.03 && (!d.wob || kick > d.wob)){ d.wob = kick; d.wobA = Math.atan2(nx, -ny); d.wobT = 0; }
      }
      const maxPen = d.r * MAX_PRESS;
      if (pen > maxPen){ d.x += (pen - maxPen) * nx; d.y += (pen - maxPen) * ny; pen = maxPen; }
      // a free drop relaxes back off the wall; a held one stays pressed as long as it is pushed
      if (!d.held){
        const back = pen * (1 - Math.exp(-dt * 10));
        d.x += back * nx; d.y += back * ny; pen -= back;
      }
      // the renderer cuts the drop flat at the edge; spreading a little along the wall makes up the lost area
      const e = pen * 0.9;
      ex += -ny * e; ey += nx * e;
    }
    d.fx = ex; d.fy = ey;
  }

  function mergeDrops(dt){
    for (let i = 0; i < drops.length; i++){
      const a = drops[i];
      if (a.into) continue;
      if (simT - a.born < 0.12) continue;          // fresh fragments start overlapping; let them fly apart first
      for (let j = i + 1; j < drops.length; j++){
        const b = drops[j];
        if (b.into || simT - b.born < 0.12) continue;
        if (Math.hypot(a.x - b.x, a.y - b.y) < (a.r + b.r) * 1.15){
          const [big, small] = a.vol >= b.vol ? [a, b] : [b, a];
          small.into = big;
          if (phase === 'scatter') plop();
          if (small === a) break;
        }
      }
    }
    // Coalescence: both drops draw together toward their shared centre of mass while the neck between
    // them widens and the smaller one drains into the larger. Bigger drops take longer (capillary
    // time grows with size), and the merged drop is left wobbling along the axis they joined on.
    const k = kk();
    for (const d of drops){
      let t = d.into;
      while (t && t.into) t = t.into;
      if (!t || t.dead){ d.into = null; continue; }
      if (d.into !== t || d.mergeT === undefined){
        d.mergeT = 0; d.vol0 = d.vol;
        // momentum is shared once, as the drops touch; after that the merging drop moves with its partner
        if (!t.held){
          const M0 = t.vol + d.vol;
          t.vx = (t.vx * t.vol + d.vx * d.vol) / M0; t.vy = (t.vy * t.vol + d.vy * d.vol) / M0;
        }
      }
      d.into = t;
      d.mergeT += dt;
      // the smaller drop sets the pace: a droplet touching a big drop is swallowed quickly,
      // two big drops take visibly longer
      const tau = 0.04 + 0.0045 * d.r / k;
      // the neck starts thin, so draining starts slow and speeds up
      const ramp = Math.min(1, d.mergeT / (tau * 1.5));
      const pull = 1 - Math.exp(-dt / tau);
      const M = t.vol + d.vol;
      const angle = Math.atan2(d.y - t.y, d.x - t.x);
      // work in the partner's frame, so a merging drop rides along with a drop that is moving (or being
      // dragged) instead of lagging behind it; the gap closes with the partner giving way by its share of mass
      const rx = d.x - t.x, ry = d.y - t.y;
      const give = t.held ? 0 : d.vol / M;
      t.x += rx * pull * give; t.y += ry * pull * give;
      d.x = t.x + rx * (1 - pull); d.y = t.y + ry * (1 - pull);
      d.vx = t.vx; d.vy = t.vy;
      // exponential drain, floored so the last of it doesn't linger: done within ~3 tau
      const mv = Math.min(Math.max(d.vol * (1 - Math.exp(-dt * ramp / tau)), d.vol0 * dt / (3 * tau) * ramp), d.vol);
      d.vol -= mv; t.vol += mv;
      d.r = Math.cbrt(Math.max(d.vol, 0)); t.r = Math.cbrt(t.vol);
      // the wobble starts when the two surfaces have visibly become one, not when the last volume moves
      if (!d.rang && d.mergeT > tau * 1.1){
        d.rang = true;
        const kick = Math.min(0.16, 0.28 * Math.cbrt(d.vol0 / (t.vol + d.vol)));
        if (!t.wob || kick > t.wob){ t.wob = kick; t.wobA = angle; t.wobT = 0; }
      }
      if (d.vol <= 0.3){ t.vol += Math.max(d.vol, 0); t.r = Math.cbrt(t.vol); d.dead = true; }
    }
    if (drops.some(d => d.dead)) drops = drops.filter(d => !d.dead);
  }

  function reformStep(dt){
    const t = phaseT, k = kk();
    for (const d of drops){
      const lt = t - d.delay;
      if (lt < 0) continue;
      if (d.into) continue;
      const w = Math.min(2.5 + lt * 11, 13);
      const tx = pill.cx, ty = pill.cy;
      let ax = w * w * (tx - d.x) - 1.7 * w * d.vx;
      let ay = w * w * (ty - d.y) - 1.7 * w * d.vy;
      // surface tension pulls a drop toward a larger neighbour, so they chain up on the way in
      let best = null, bestD = Infinity;
      for (const o of drops){
        if (o === d || o.into || o.vol <= d.vol) continue;
        const dd = Math.hypot(o.x - d.x, o.y - d.y) - o.r - d.r;
        if (dd < bestD){ bestD = dd; best = o; }
      }
      if (best && bestD < (d.r + best.r) * 2.5){
        const pull = 2600 * k / (1 + bestD / (d.r + 2));
        const dl = Math.hypot(best.x - d.x, best.y - d.y) || 1;
        ax += pull * (best.x - d.x) / dl; ay += pull * (best.y - d.y) / dl;
      }
      d.vx += ax * dt; d.vy += ay * dt;
      d.x += d.vx * dt; d.y += d.vy * dt;

      if (!d.absorbing){
        const inPill = pill.scale > 0.02 ? pillSdf(d.x, d.y) < -d.r * 0.15 : Math.hypot(d.x - tx, d.y - ty) < d.r + 1;
        if (inPill){ d.absorbing = true; plop(); kickMode(rand(0.6, 1.4) * k * Math.min(1, d.r / 8), 2 + Math.floor(Math.random() * 4)); }
      }
      if (d.absorbing){
        const tauA = 0.06 + 0.0028 * d.r / k;
        const moved = d.vol * (1 - Math.exp(-dt / tauA)) + 0.5;
        const m = Math.min(moved, d.vol);
        d.vol -= m; pill.absorbed += m;
        d.r = Math.cbrt(Math.max(d.vol, 0));
        if (d.vol <= 0.5) d.dead = true;
      }
    }
    drops = drops.filter(d => !d.dead);
    mergeDrops(dt);
    pill.scaleT = Math.cbrt(Math.min(pill.absorbed / V0, 1));
    [pill.scale, pill.scaleV] = spring(pill.scale, pill.scaleV, pill.scaleT, 700, 26, dt);
    pill.scale = Math.max(pill.scale, 0);

    if (drops.length === 0){
      pill.absorbed = V0;
      if (Math.abs(pill.scale - 1) < 0.02 && Math.abs(pill.scaleV) < 0.3){
        phase = 'idle'; phaseT = 0;
        pill.sxV -= 1.1; pill.syV += 1.4;
        kickMode(1.6 * k, 3);
        sound.reform();
        enablePointer();
      }
    }
  }

  // ---- input
  let pressed = null;                // the pointer pressing the pill: { id, x0, y0, dragging }
  let suppressUntil = 0;             // a drag started on the pill is not a click
  let popClickUntil = 0;             // the click that belongs to a pointer pop is let through as the button's click
  let swallowUntil = 0;              // tapping or dragging a drop is not a click on whatever is under it
  const local = (cx, cy) => { const r = button.getBoundingClientRect(); return [cx - r.left, cy - r.top]; };
  const inStage = (x, y) => x >= -G.offX && y >= -G.offY && x <= G.stageW - G.offX && y <= G.stageH - G.offY;
  const scattered = () => phase === 'scatter' || phase === 'reform' || phase === 'pop';

  function onBtnEnter(){
    if (lost || phase !== 'idle' || reduceMotion.matches || pressed) return;
    kickMode(0.9 * kk(), 3); wake();
  }
  function onBtnDown(e){
    if (lost || e.button !== 0 || !e.isPrimary || pressed || phase !== 'idle' || button.disabled) return;
    audio();
    const [x, y] = local(e.clientX, e.clientY);
    pressed = { id: e.pointerId, x0: x, y0: y, dragging: false };
    try { button.setPointerCapture(e.pointerId); } catch {}
    pill.dentX = x; pill.dentY = y; pill.pressT = 1;
    wake();
  }
  function onBtnMove(e){
    if (!pressed || e.pointerId !== pressed.id) return;
    const [x, y] = local(e.clientX, e.clientY);
    if (!pressed.dragging && Math.hypot(x - pressed.x0, y - pressed.y0) > DRAG_START) pressed.dragging = true;
    pill.dentX = x; pill.dentY = y;
    pill.pressT = pillSdf(x, y) < 8 ? 1 : 0;
    wake();
  }
  function onBtnUp(e, cancel){
    if (!pressed || e.pointerId !== pressed.id) return;
    const p = pressed; pressed = null;
    pill.pressT = 0;
    if (!cancel && p.dragging) suppressUntil = performance.now() + 800;
    if (!cancel && !p.dragging && phase === 'idle' && !lost){
      const [x, y] = local(e.clientX, e.clientY);
      popClickUntil = performance.now() + 1000;
      pop(x, y);
      // on touch the click can arrive a little after the release; the button must still be hit-testable then
      clearTimeout(peTimer); peTimer = setTimeout(disablePointer, 700);
    } else { pill.syV += 0.5; pill.sxV -= 0.4; }
    wake();
  }
  const onBtnCancel = e => onBtnUp(e, true);

  function onClick(e){
    const now = performance.now();
    if (e.detail > 0 && now < suppressUntil){ suppressUntil = 0; e.preventDefault(); e.stopImmediatePropagation(); return; }
    if (lost || !hidden) return;
    if (now < popClickUntil){ popClickUntil = 0; disablePointer(); return; }
    audio();
    if (phase === 'idle'){
      // keyboard, form submission, or button.click()
      pop();
      disablePointer();
    } else if (e.detail === 0) burstLargest();
    else { e.preventDefault(); e.stopImmediatePropagation(); }   // a pointer click on the spot the button left
    wake();
  }
  const isActivateKey = e => e.key === ' ' || e.key === 'Enter';
  function onKeyDown(e){
    if (isActivateKey(e) && !e.repeat && phase === 'idle' && !lost){ pill.dentX = pill.cx; pill.dentY = pill.cy; pill.pressT = 1; wake(); }
  }
  function onKeyUp(e){ if (isActivateKey(e) && !pressed){ pill.pressT = 0; wake(); } }
  function onBlur(){ if (!pressed){ pill.pressT = 0; wake(); } }

  // drops: routed from the document, since the canvas never takes pointer events
  const root = document.documentElement;
  const baseCursor = root.style.cursor;
  let cursor = null;
  function setCursor(c){
    if (c === cursor) return;
    cursor = c;
    root.style.cursor = c || baseCursor;
  }
  function docDown(e){
    if (lost || e.button !== 0 || (phase !== 'scatter' && phase !== 'reform')) return;
    const [x, y] = local(e.clientX, e.clientY);
    if (!inStage(x, y)) return;
    const d = pickDrop(x, y);
    if (!d) return;
    audio();
    const g = { drop: d, x, y, x0: x, y0: y, dragging: false };
    for (const o of grabs.values()) if (o.drop === d) o.drop = null;
    d.grab = g;
    grabs.set(e.pointerId, g);
    try { root.setPointerCapture(e.pointerId); } catch {}
    lastBurst = simT;
    e.preventDefault(); e.stopPropagation();
    wake();
  }
  function docMove(e){
    const g = grabs.get(e.pointerId);
    if (g){
      [g.x, g.y] = local(e.clientX, e.clientY);
      if (!g.dragging && Math.hypot(g.x - g.x0, g.y - g.y0) > DRAG_START){
        g.dragging = true;
        const d = resolveGrab(g);
        if (d) d.held = true;
      }
      if (g.dragging) setCursor('grabbing');
      wake();
      return;
    }
    if (grabs.size) return;
    if (scattered()){
      const [x, y] = local(e.clientX, e.clientY);
      setCursor(inStage(x, y) && overDrop(x, y) ? 'grab' : null);
    } else setCursor(null);
  }
  function releaseGrab(id, x, y, cancel){
    const g = grabs.get(id);
    if (!g) return;
    grabs.delete(id);
    const d = resolveGrab(g);
    lastBurst = simT;                                   // regroup waits for a quiet moment after letting go
    if (!grabs.size) setCursor(null);
    if (!d) return;
    d.held = false; d.grab = null;
    if (!g.dragging && !cancel && phase === 'scatter'){ burstDrop(d, x, y); return; }
    // a flick keeps its momentum, within reason
    const k = kk(), sp = Math.hypot(d.vx, d.vy), max = 900 * k;
    if (sp > max){ d.vx *= max / sp; d.vy *= max / sp; }
  }
  function releaseAllGrabs(){ for (const id of [...grabs.keys()]) releaseGrab(id, 0, 0, true); }
  function docUp(e){
    if (!grabs.has(e.pointerId)) return;
    const [x, y] = local(e.clientX, e.clientY);
    releaseGrab(e.pointerId, x, y, false);
    swallowUntil = performance.now() + 800;
    try { root.releasePointerCapture(e.pointerId); } catch {}
    e.stopPropagation();
    wake();
  }
  function docCancel(e){ if (grabs.has(e.pointerId)){ releaseGrab(e.pointerId, 0, 0, true); wake(); } }
  function docClick(e){
    if (e.detail > 0 && performance.now() < swallowUntil){ swallowUntil = 0; e.preventDefault(); e.stopImmediatePropagation(); }
  }
  // a finger on a drop moves the drop, not the page
  function docTouch(e){ if (grabs.size && e.cancelable) e.preventDefault(); }
  function onContextMenu(){ releaseAllGrabs(); if (pressed) onBtnUp({ pointerId: pressed.id, clientX: 0, clientY: 0 }, true); }
  function onWinBlur(){ onContextMenu(); onBlur(); }

  const cap = { capture: true };
  const capNP = { capture: true, passive: false };
  button.addEventListener('pointerenter', onBtnEnter);
  button.addEventListener('pointerdown', onBtnDown);
  button.addEventListener('pointermove', onBtnMove);
  button.addEventListener('pointerup', onBtnUp);
  button.addEventListener('pointercancel', onBtnCancel);
  button.addEventListener('lostpointercapture', onBtnCancel);
  button.addEventListener('click', onClick, true);
  button.addEventListener('keydown', onKeyDown);
  button.addEventListener('keyup', onKeyUp);
  button.addEventListener('blur', onBlur);
  document.addEventListener('pointerdown', docDown, cap);
  document.addEventListener('pointermove', docMove, cap);
  document.addEventListener('pointerup', docUp, cap);
  document.addEventListener('pointercancel', docCancel, cap);
  document.addEventListener('lostpointercapture', docCancel, cap);
  document.addEventListener('click', docClick, cap);
  document.addEventListener('touchstart', docTouch, capNP);
  document.addEventListener('touchmove', docTouch, capNP);
  document.addEventListener('contextmenu', onContextMenu, cap);
  addEventListener('blur', onWinBlur);

  // ---- render
  function render(time, dt){
    const toX = x => (x + G.offX) * dpr, toY = y => (G.stageH - (y + G.offY)) * dprY;
    const k = kk();

    const f = P.field;
    gl.useProgram(f.p);
    const s = pill.scale;
    const r = pill.hh * s * pill.sy * dpr;
    const hw = pill.hw * s * pill.sx * dpr;
    gl.uniform4f(f.u.uPill, toX(pill.cx), toY(pill.cy), hw, s > 0.01 ? r : 0);
    pill.modes.forEach((m, i) => {
      modeBuf[i * 4] = m.amp * Math.sin(m.w * m.t) * dpr;
      modeBuf[i * 4 + 1] = m.m; modeBuf[i * 4 + 2] = m.phi; modeBuf[i * 4 + 3] = 0;
    });
    gl.uniform4fv(f.u.uModes, modeBuf);
    const dentDepth = pill.press * pill.hh * 0.16 * dpr * s;
    gl.uniform4f(f.u.uDent, toX(pill.dentX), toY(pill.dentY), dentDepth, pill.hh * 0.55 * dpr);
    const n = Math.min(drops.length, MAX_DROPS);
    for (let i = 0; i < n; i++){
      const d = drops[i];
      const st = 0.022;
      let tx = d.x - d.vx * st, ty = d.y - d.vy * st;
      const tl = Math.hypot(tx - d.x, ty - d.y);
      const maxT = d.r * 2.2;
      if (tl > maxT){ tx = d.x + (tx - d.x) * maxT / tl; ty = d.y + (ty - d.y) * maxT / tl; }
      // shape oscillation after a merge: alternately stretched along and across the joining axis
      let hx = d.x, hy = d.y, ex = 0, ey = 0;
      if (d.wob){
        const w = TAU * 7 * Math.pow(10 * k / Math.max(d.r, 1), 0.75);
        // starts from round and grows, so a merge never snaps the drop into a stretched shape
        const sw = d.wob * Math.sin(w * d.wobT);
        const a = d.wobA + (sw < 0 ? Math.PI / 2 : 0);
        const e = Math.abs(sw) * d.r;
        ex = Math.cos(a) * e; ey = Math.sin(a) * e;
        hx += ex; hy += ey; tx -= ex; ty -= ey;
      }
      if (d.fx || d.fy){
        hx += d.fx; hy += d.fy; tx -= d.fx; ty -= d.fy;
        ex += d.fx; ey += d.fy;
      }
      const segLen = Math.min(tl, maxT) + 2 * Math.hypot(ex, ey);
      const shrink = 1 / Math.sqrt(1 + segLen / (2 * Math.max(d.r, 0.5)) * 0.8);
      // flung splash drops are partly airborne; dragged and regrouping drops slide and leave a full trail
      const wetting = d.held ? 1 : phase === 'reform' ? 0.6 : 1 - Math.min(Math.hypot(d.vx, d.vy) / (160 * k), 0.85);
      dropBuf[i * 4] = toX(hx); dropBuf[i * 4 + 1] = toY(hy); dropBuf[i * 4 + 2] = d.r * shrink * dpr; dropBuf[i * 4 + 3] = wetting;
      tailBuf[i * 2] = toX(tx); tailBuf[i * 2 + 1] = toY(ty);
    }
    gl.uniform1i(f.u.uCount, n);
    gl.uniform4fv(f.u.uDrops, dropBuf);
    gl.uniform2fv(f.u.uTails, tailBuf);
    gl.uniform1f(f.u.uK, 0.45);
    gl.uniform1f(f.u.uHR, 0.82);
    gl.uniform1f(f.u.uP, 2.0);
    gl.uniform1f(f.u.uSquash, 1 - 0.12 * pill.press);
    gl.uniform3f(f.u.uEdge, W, H, 1.5 * dpr);
    draw(f, hFb, W, H);

    const w = P.wet;
    gl.useProgram(w.p);
    bindTex(0, wetTex[wetIdx]); gl.uniform1i(w.u.uPrev, 0);
    bindTex(1, hTex); gl.uniform1i(w.u.uH, 1);
    gl.uniform2f(w.u.uHRes, W, H);
    gl.uniform2f(w.u.uRes, wetW, wetH);
    gl.uniform1f(w.u.uDecay, Math.exp(-dt / 5.5));
    draw(w, wetFb[1 - wetIdx], wetW, wetH);
    wetIdx = 1 - wetIdx;

    const c = P.comp;
    gl.useProgram(c.p);
    bindTex(1, hTex); gl.uniform1i(c.u.uH, 1);
    bindTex(2, wetTex[wetIdx]); gl.uniform1i(c.u.uWet, 2);
    bindTex(3, labelTex); gl.uniform1i(c.u.uLabel, 3);
    const hasSurf = syncSurface();
    bindTex(4, surfTex); gl.uniform1i(c.u.uSurf, 4);
    gl.uniform1i(c.u.uHasSurf, hasSurf ? 1 : 0);
    if (hasSurf){
      const sr = opts.surface.rect(), br = btnRect || button.getBoundingClientRect();
      const sx = br.left - G.offX, sy = br.top - G.offY;
      const rw = Math.max(sr.width, 1), rh = Math.max(sr.height, 1);
      gl.uniform4f(c.u.uSurfX, 1 / (dpr * rw), (sx - sr.left) / rw, -1 / (dprY * rh), (sy + G.stageH - sr.top) / rh);
      gl.uniform1f(c.u.uSurfLod, Math.log2(Math.max(surfW / (rw * dpr), 1e-3)));
    }
    gl.uniform3f(c.u.uBgCol, bgCol[0], bgCol[1], bgCol[2]);
    gl.uniform2f(c.u.uRes, W, H);
    gl.uniform1f(c.u.uDpr, dpr);
    gl.uniform1f(c.u.uOs, Math.max(0.5, Math.min(1, k / 1.2)));
    gl.uniform4f(c.u.uLabelRect, toX(pill.cx), toY(pill.cy), (pill.hw + 4) * dpr, (pill.hh + 4) * dprY);
    gl.uniform1f(c.u.uTime, time);
    draw(c, null, W, H);
  }

  // ---- loop
  let raf = 0, inFrame = false, frameCount = 0, last = performance.now(), dirty = true, manual = false, lost = false, visible = true;
  let lastActive = performance.now();
  // GPU load is measured with fences: a frame's fence still pending two frames later means the GPU is behind.
  // This is independent of the display's refresh rate, which a frame-interval heuristic cannot separate from load.
  const fences = [];
  let warmup = 10, win = 0, winSlow = 0, winStart = 0, calm = 0;

  function busy(){
    return phase !== 'idle' || !!pressed || grabs.size > 0 || pill.pressT > 0 || Math.abs(pill.press) > 1e-3 ||
      Math.abs(pill.sx - 1) + Math.abs(pill.sy - 1) + Math.abs(pill.sxV) + Math.abs(pill.syV) > 1e-3 ||
      pill.modes.some(m => m.amp > 0.01);
  }
  function wake(){
    lastActive = performance.now();
    // one animation loop at most: a wake from inside a frame must not start a second loop
    if (!raf && !inFrame && !manual && !lost) raf = requestAnimationFrame(frame);
  }
  function advance(dt, time){
    const sub = Math.max(1, Math.ceil(dt / (1 / 240)));
    for (let i = 0; i < sub; i++) step(dt / sub);
    if (!lost && !gl.isContextLost() && W && (visible || dirty)){ render(time, dt); dirty = false; return true; }
    return false;
  }
  function frame(now){
    raf = 0;
    if (manual || lost) return;
    const raw = (now - last) / 1000;
    const dt = Math.min(raw, 1 / 30);
    last = now;
    place();
    if (busy()) lastActive = now;
    adapt(raw);
    inFrame = true;
    frameCount++;
    try { if (advance(dt, now / 1000)) fences.push(gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0)); } finally { inFrame = false; }
    // nothing moves and the stains have dried: the last frame stays on screen
    if (now - lastActive < DRY_MS) raf = requestAnimationFrame(frame);
    else { fences.forEach(f => gl.deleteSync(f)); fences.length = 0; }
  }
  function adapt(raw){
    if (fences.length < 3) return;
    const f = fences.shift();
    // under ~22 fps is too slow on any display (30 Hz screens run at 33 ms); over 1 s is a tab switch, not load
    if (raw > 1){ gl.deleteSync(f); return; }
    const behind = gl.getSyncParameter(f, gl.SYNC_STATUS) !== gl.SIGNALED || raw > 0.045;
    gl.deleteSync(f);
    if (warmup > 0){ warmup--; return; }
    const now = performance.now();
    if (win === 0) winStart = now;
    win++; if (behind) winSlow++;
    // judge every 120 frames, or every 2 s when frames are so slow that 120 would take too long
    if (win < 120 && !(now - winStart > 2000 && win >= 8)) return;
    if (winSlow / win > 0.35 && renderScale > 0.5){
      renderScale = Math.max(0.5, renderScale - 0.15); calm = 0;
      resizeGL();
    } else if (winSlow === 0){
      calm += win;
      if (calm >= 1200 && renderScale < 1 && phase === 'idle'){ renderScale = Math.min(1, renderScale + 0.15); calm = 0; resizeGL(); }
    } else calm = 0;
    win = winSlow = 0;
  }

  // ---- keep the stage on the button
  const relayout = () => { if (lost) return; measureBounds(); layout(); };
  const ro = new ResizeObserver(relayout);
  ro.observe(button);
  if (document.body) ro.observe(document.body);
  if (opts.area) ro.observe(opts.area);
  const onScroll = () => { if (!lost) place(); };
  addEventListener('resize', relayout);
  document.addEventListener('scroll', onScroll, { capture: true, passive: true });
  let dprMq = null;
  const onDpr = () => { watchDpr(); relayout(); resizeGL(); wake(); };
  function watchDpr(){
    dprMq?.removeEventListener('change', onDpr);
    dprMq = matchMedia(`(resolution: ${devicePixelRatio}dppx)`);
    dprMq.addEventListener('change', onDpr);
  }
  watchDpr();
  const io = new IntersectionObserver(es => { for (const e of es) visible = e.isIntersecting; if (visible){ dirty = true; wake(); } });
  io.observe(canvas);
  const onFonts = () => { if (!lost && labelText()){ buildLabel(); wake(); } };
  document.fonts?.addEventListener?.('loadingdone', onFonts);
  // the label follows text or style changes made to the button while it is attached
  const mo = new MutationObserver(() => {
    if (lost) return;
    const cs = getComputedStyle(button);
    if (labelText() + '|' + cs.font + '|' + cs.color !== labelKey){ buildLabel(); wake(); }
  });
  mo.observe(button, { childList: true, characterData: true, subtree: true, attributes: true, attributeFilter: ['class'] });

  // A lost context shows the plain button until the GPU comes back, then the water is rebuilt.
  function onLost(e){
    e.preventDefault();
    lost = true;
    cancelAnimationFrame(raf); raf = 0;
    releaseAllGrabs(); pressed = null;
    phase = 'idle'; drops = []; pill.scale = 1; pill.pressT = pill.press = 0;
    fences.length = 0;
    enablePointer(); setCursor(null);
    showButton();
    canvas.style.visibility = 'hidden';
  }
  function onRestored(){
    try { initGL(); } catch (err) { console.warn('ElementalFX water: could not rebuild after context loss', err); return; }
    lost = false;
    hideButton();
    canvas.style.visibility = '';
    G.stageW = G.stageH = 0;
    relayout();
    advance(0, performance.now() / 1000);
    wake();
  }
  canvas.addEventListener('webglcontextlost', onLost);
  canvas.addEventListener('webglcontextrestored', onRestored);

  // ---- go
  hideButton();
  document.body.appendChild(canvas);
  relayout();
  advance(0, performance.now() / 1000);

  return {
    supported: true,
    get state(){ return lost ? 'lost' : phase; },
    refresh(){ if (!lost){ surfSrc = null; relayout(); buildLabel(); dirty = true; wake(); } },
    detach(){
      cancelAnimationFrame(raf); raf = 0; manual = true;
      clearTimeout(peTimer);
      ro.disconnect(); io.disconnect(); mo.disconnect();
      dprMq?.removeEventListener('change', onDpr);
      document.fonts?.removeEventListener?.('loadingdone', onFonts);
      removeEventListener('resize', relayout); removeEventListener('blur', onWinBlur);
      document.removeEventListener('scroll', onScroll, { capture: true });
      button.removeEventListener('pointerenter', onBtnEnter);
      button.removeEventListener('pointerdown', onBtnDown); button.removeEventListener('pointermove', onBtnMove);
      button.removeEventListener('pointerup', onBtnUp); button.removeEventListener('pointercancel', onBtnCancel);
      button.removeEventListener('lostpointercapture', onBtnCancel); button.removeEventListener('click', onClick, true);
      button.removeEventListener('keydown', onKeyDown); button.removeEventListener('keyup', onKeyUp); button.removeEventListener('blur', onBlur);
      document.removeEventListener('pointerdown', docDown, cap); document.removeEventListener('pointermove', docMove, cap);
      document.removeEventListener('pointerup', docUp, cap); document.removeEventListener('pointercancel', docCancel, cap);
      document.removeEventListener('lostpointercapture', docCancel, cap); document.removeEventListener('click', docClick, cap);
      document.removeEventListener('touchstart', docTouch, capNP); document.removeEventListener('touchmove', docTouch, capNP);
      document.removeEventListener('contextmenu', onContextMenu, cap);
      canvas.removeEventListener('webglcontextlost', onLost); canvas.removeEventListener('webglcontextrestored', onRestored);
      for (const id of grabs.keys()){ try { root.releasePointerCapture(id); } catch {} }
      grabs.clear();
      setCursor(null);
      showButton();
      canvas.remove();
      gl.getExtension('WEBGL_lose_context')?.loseContext();
    },
    // test hooks
    get _frames(){ return frameCount; },
    _manual(on){ manual = on; last = performance.now(); if (!on) wake(); else { cancelAnimationFrame(raf); raf = 0; } },
    _advance(sec, fps = 60){ place(); for (let i = 0, n = Math.round(sec * fps); i < n; i++) advance(1 / fps, performance.now() / 1000); },
    get _sim(){ return { drops, pill, G, pop, burstLargest, get grabs(){ return grabs.size; }, get renderScale(){ return renderScale; }, get pointerEvents(){ return button.style.pointerEvents; } }; },
  };
}

window.ElementalFX = Object.assign(window.ElementalFX || {}, { attachWater, concreteCanvas, waterAudio });
})();
