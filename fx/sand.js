// Sand button: turns an existing <button> into a moulded block of damp kinetic sand.
//
//   <script src="fx/sand.js"></script>
//   const fx = ElementalFX.attachSand(document.querySelector('#subscribe'));   // fx.detach() restores it
//
// A classic script (not a module) so pages work straight from file:// too.
//
// The <button> stays the real control (focus, keyboard, form submission, click handlers); this only
// draws it. Pressing leaves a fingerprint, a click slumps the block and it re-moulds itself, and
// dragging a finger across it carves grooves (a drag is not a click). If WebGL2 with float render
// targets is unavailable, the button is left untouched.

(() => {
'use strict';

const TALUS_LOOSE = 0.75;     // tan of the angle of repose (~37 deg)
const TALUS_PACKED = 40;      // moulded sand holds any wall
const SLIDE_ITERS = 16;

// ---------------------------------------------------------------- shaders
const VS = `#version 300 es
void main(){ vec2 p = vec2(float((gl_VertexID<<1)&2), float(gl_VertexID&2)); gl_Position = vec4(p*2.-1., 0., 1.); }`;

const COMMON = `#version 300 es
precision highp float;
precision highp sampler2D;
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*.1031); p3 += dot(p3, p3.yzx+33.33); return fract((p3.x+p3.y)*p3.z); }
vec2 hash22(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*vec3(.1031,.1030,.0973)); p3 += dot(p3, p3.yzx+33.33); return fract((p3.xx+p3.yz)*p3.zy); }
float vnoise(vec2 p){
  vec2 i = floor(p), f = fract(p);
  vec2 u = f*f*(3.-2.*f);
  return mix(mix(hash12(i), hash12(i+vec2(1,0)), u.x), mix(hash12(i+vec2(0,1)), hash12(i+vec2(1,1)), u.x), u.y);
}
`;

// State (one cell per 1/gs CSS px, y up): r = height (cells), g = packing, b = fluidity, a = spare.
// All lengths below are in cells, so the angle of repose is scale-free.

// The pristine block: a rounded rectangle the size of the button with moulded shoulders and the
// label pressed in. Blended in to reform (uSoft/uLetters stage the press), or fully to initialise.
const FS_TARGET = COMMON + `
uniform sampler2D uSrc;
uniform sampler2D uText;
uniform vec4 uBox;        // cx, cy, halfW, halfH
uniform float uRadius, uHeight, uShoulder, uTextDepth, uAmt, uSoft, uLetters, uLip, uGs;
out vec4 o;
float sdRound(vec2 q, vec2 b, float r){ vec2 d = abs(q) - b + r; return length(max(d, 0.)) + min(max(d.x, d.y), 0.) - r; }
void main(){
  vec2 p = gl_FragCoord.xy;
  vec4 cur = texelFetch(uSrc, ivec2(p), 0);
  float sd = sdRound(p - uBox.xy, uBox.zw, uRadius);
  // a mould edge is never perfect: a little wander and the odd chip
  float sdE = sd + 0.55 * uGs * (vnoise(p / uGs * 0.35) - 0.5) + 0.4 * uGs * smoothstep(0.8, 0.93, vnoise(p / uGs * 0.16 + 3.));
  float sw = uShoulder * mix(1., 5., uSoft);
  float x = clamp(-sdE / sw, 0., 1.);
  float shoulder = mix(sqrt(1. - (1. - x) * (1. - x)), smoothstep(0., 1., x), uSoft);
  float txt = texture(uText, p / vec2(textureSize(uText, 0))).r;
  float h = uHeight * (1. - 0.25 * uSoft) * shoulder - uTextDepth * uLetters * txt * step(0.999, x);
  h += uLip * uHeight * 0.18 * exp(-pow((sd - 2.5 * uGs) / (1.8 * uGs), 2.));
  float packed = smoothstep(0.0, 0.25, clamp((uGs * 0.5 - sdE) / sw, 0., 1.)) * (1. - uSoft);
  float a = clamp(uAmt, 0., 1.);
  o = vec4(mix(cur.r, max(h, 0.), a), mix(cur.g, packed, a), cur.b * (1. - a), 1.);
}`;

// A fingertip pressed into the sand. Sand under the tip is pushed into a ring around it; the map
// t^2 = r^2 (s^2 - a^2) / (b^2 - a^2) from ring radius s to disc radius t has a uniform Jacobian,
// so scaling deposits by it (and by the discrete cell-count correction) moves every bit exactly once.
const FS_STAMP = COMMON + `
uniform sampler2D uSrc;
uniform vec4 uTool;       // cx, cy, radius, depth below the local surface
uniform float uJfix, uSeed;
out vec4 o;
float hAt(vec2 p){
  vec2 f = p - 0.5; vec2 i = floor(f); vec2 t = f - i;
  ivec2 a = ivec2(i), m = textureSize(uSrc, 0) - 1;
  float h00 = texelFetch(uSrc, clamp(a, ivec2(0), m), 0).r;
  float h10 = texelFetch(uSrc, clamp(a + ivec2(1,0), ivec2(0), m), 0).r;
  float h01 = texelFetch(uSrc, clamp(a + ivec2(0,1), ivec2(0), m), 0).r;
  float h11 = texelFetch(uSrc, clamp(a + ivec2(1,1), ivec2(0), m), 0).r;
  return mix(mix(h00, h10, t.x), mix(h01, h11, t.x), t.y);
}
float low;
float tipAt(float t){ float r = uTool.z; return low + r - sqrt(max(r*r - t*t, 0.)); }
void main(){
  vec2 p = gl_FragCoord.xy;
  vec4 c = texelFetch(uSrc, ivec2(p), 0);
  vec2 C = uTool.xy;
  float r = uTool.z;
  // press a fixed depth into whatever is there; every fragment averages the same 8 points
  float ref = 0.;
  for (int k = 0; k < 8; k++){ float a = float(k) * 0.7853982; ref += hAt(C + vec2(cos(a), sin(a)) * r * 1.7); }
  low = max(ref / 8. - uTool.w, 0.);
  float a = r, b = r * 1.55;
  vec2 d = p - C;
  float s = length(d);
  vec2 dir = d / max(s, 1e-4);
  vec2 gsz = vec2(textureSize(uSrc, 0));
  if (s < r){
    vec2 dest = C + dir * sqrt(a*a + s*s * (b*b - a*a) / (r*r));
    if (any(lessThan(dest, vec2(0.))) || any(greaterThanEqual(dest, gsz))){ o = c; return; }
    c.r = min(c.r, tipAt(s));
  } else if (s < b){
    float t = r * sqrt(max((s*s - a*a) / (b*b - a*a), 0.));
    vec2 q = C + dir * t;
    if (any(lessThan(q, vec2(0.))) || any(greaterThanEqual(q, gsz))){ o = c; return; }
    float ex = max(hAt(q) - tipAt(t), 0.) * (r*r) / (b*b - a*a) * uJfix;
    ex *= 0.8 + 0.4 * hash12(floor(p * 0.6) + uSeed);
    c.r += ex;
    c.g = mix(c.g, 0., clamp(ex * 2., 0., 1.));
  }
  c.g *= mix(0.2, 1.0, smoothstep(r * 0.9, r * 1.8, s));
  o = c;
}`;

// Avalanching: neighbours exchange sand in proportion to how far their difference exceeds the angle
// of repose (antisymmetric, so exactly conserved). Fluidised sand also spreads over a few cells per
// step, like a very thick liquid, which is what makes a struck block sag.
const FS_SLIDE = COMMON + `
uniform sampler2D uSrc;
uniform float uTl, uTp, uFluidDecay, uMelt, uSeed, uR1, uR2;
out vec4 o;
float talus(vec4 c, ivec2 cell){
  float n = hash12(vec2(cell) + uSeed);
  return mix(uTl * (0.82 + 0.36 * n) * (1. - 0.9 * c.b), uTp, c.g * c.g * c.g);
}
void main(){
  ivec2 ip = ivec2(gl_FragCoord.xy), m = textureSize(uSrc, 0) - 1;
  vec4 c = texelFetch(uSrc, ip, 0);
  float dh = 0.;
  for (int k = 0; k < 8; k++){
    ivec2 off = k == 0 ? ivec2(1,0) : k == 1 ? ivec2(-1,0) : k == 2 ? ivec2(0,1) : k == 3 ? ivec2(0,-1)
              : k == 4 ? ivec2(1,1) : k == 5 ? ivec2(-1,-1) : k == 6 ? ivec2(1,-1) : ivec2(-1,1);
    ivec2 jp = ip + off;
    if (any(lessThan(jp, ivec2(0))) || any(greaterThan(jp, m))) continue;
    vec4 n = texelFetch(uSrc, jp, 0);
    float diff = n.r - c.r;
    bool upN = diff > 0.;
    float T = talus(upN ? n : c, upN ? jp : ip) * (k < 4 ? 1. : 1.41421356);
    dh += (k < 4 ? 0.08 : 0.04) * sign(diff) * max(abs(diff) - T, 0.);
  }
  if (uMelt > 0.){
    for (int ring = 0; ring < 2; ring++){
      float R = ring == 0 ? uR1 : uR2;
      float kc = ring == 0 ? 0.014 : 0.007;
      for (int k = 0; k < 8; k++){
        float a = float(k) * 0.7853982 + float(ring) * 0.3926991;
        ivec2 jp = ip + ivec2(round(vec2(cos(a), sin(a)) * R));
        if (any(lessThan(jp, ivec2(0))) || any(greaterThan(jp, m))) continue;
        vec4 n = texelFetch(uSrc, jp, 0);
        dh += kc * 0.5 * (c.b + n.b) * (n.r - c.r);
      }
    }
  }
  o = vec4(c.r + dh, c.g, c.b * uFluidDecay, 1.);
}`;

// Striking the block: it cracks into slabs as the failure travels out from the finger, each slab
// giving way after its own delay and keeping a little cohesion, so it slumps in lumps.
const FS_SLUMP = COMMON + `
uniform sampler2D uSrc;
uniform vec2 uFrom;
uniform float uT, uSpeed, uDecay, uSlab;
out vec4 o;
void main(){
  vec2 p = gl_FragCoord.xy;
  vec4 c = texelFetch(uSrc, ivec2(p), 0);
  if (c.g < 0.001){ o = c; return; }
  vec2 g = p / uSlab, gi = floor(g);
  float d1 = 9., d2 = 9.; vec2 id = gi;
  for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++){
    vec2 cc = gi + vec2(i, j);
    float d = length(g - (cc + 0.15 + 0.7 * hash22(cc)));
    if (d < d1){ d2 = d1; d1 = d; id = cc; } else if (d < d2) d2 = d;
  }
  float seam = 1. - smoothstep(0.06, 0.15, d2 - d1);
  float delay = mix(0.05 + 0.4 * hash12(id + 4.2), 0., seam);
  if (uT < length(p - uFrom) / uSpeed + delay){ o = c; return; }
  float keep = mix(0.18 + 0.32 * hash12(id + 7.7), 0., seam);
  float lost = max(c.g - keep, 0.) * (1. - uDecay);
  c.g -= lost;
  c.b = max(c.b, min(1., lost * mix(1.3, 3.5, seam)));
  o = c;
}`;

const FS_SHADE = COMMON + `
uniform sampler2D uSrc;
uniform vec3 uSun;
uniform float uGs;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
float H(ivec2 p){ return texelFetch(uSrc, clamp(p, ivec2(0), textureSize(uSrc, 0) - 1), 0).r; }
float Hb(vec2 p){
  vec2 f = p - 0.5; vec2 i = floor(f); vec2 t = f - i; ivec2 a = ivec2(i);
  return mix(mix(H(a), H(a + ivec2(1,0)), t.x), mix(H(a + ivec2(0,1)), H(a + ivec2(1,1)), t.x), t.y);
}
void main(){
  ivec2 ip = ivec2(gl_FragCoord.xy);
  vec2 pc = vec2(ip) + 0.5;
  float h0 = H(ip);
  float gx = (H(ip + ivec2(1,0)) - H(ip - ivec2(1,0))) * 0.5;
  float gy = (H(ip + ivec2(0,1)) - H(ip - ivec2(0,1))) * 0.5;
  vec2 sd = normalize(uSun.xy);
  float rise = uSun.z / length(uSun.xy);
  float j = hash12(vec2(ip));
  float lit = 1.;
  for (int i = 0; i < 24; i++){
    float fi = float(i) + j;
    float t = (0.8 + fi * 1.0 + fi * fi * 0.04) * uGs;
    float above = Hb(pc + sd * t) - (h0 + t * rise);
    lit = min(lit, clamp(1. - above / ((0.7 + t / uGs * 0.02) * uGs), 0., 1.));
  }
  float occ = 0.;
  for (int i = 0; i < 8; i++){
    vec2 dir = vec2(cos(float(i) * 0.785398), sin(float(i) * 0.785398));
    occ += max(Hb(pc + dir * 3. * uGs) - h0, 0.) * 0.45 + max(Hb(pc + dir * 8. * uGs) - h0, 0.) * 0.2;
  }
  float ao = clamp(1. - occ / 8. * 0.14 / uGs, 0.55, 1.);
  o0 = vec4(-gx, -gy, lit, ao);
  o1 = vec4(h0, 0., 0., 1.);
}`;

// Output is premultiplied: the sand where there is sand, and its shadow on the page around it.
const FS_COMPOSITE = COMMON + `
uniform sampler2D uN;
uniform sampler2D uQ;
uniform vec2 uGrid;
uniform float uDpr, uGs;
uniform vec3 uSun;
out vec4 o;
void main(){
  vec2 fp = gl_FragCoord.xy;
  vec2 cell = fp / uDpr * uGs;
  vec2 uv = cell / uGrid;
  vec4 N = texture(uN, uv);
  float h = texture(uQ, uv).r;
  float cover = smoothstep(0.12 * uGs, 0.7 * uGs, h);

  vec2 css = fp / uDpr;
  vec2 gp = css * 2.0;
  vec2 gcell = floor(gp) + floor(h / uGs * 1.3 + hash12(floor(gp) + 40.1)) * vec2(17.13, 7.71);
  float g0 = hash12(gcell), g1 = hash12(gcell + vec2(1., -1.));
  vec2 gr = hash22(gcell) - 0.5;
  float clump = vnoise(css * 0.7) - 0.5, clump2 = vnoise(css * 2.3 + 5.1) - 0.5;
  vec3 n = normalize(vec3(N.xy + (gr * 0.34 + vec2(g0 - g1) * 0.17 + clump * vec2(0.18, -0.14) + clump2 * vec2(-0.09, 0.11)) * 0.9, 1.));

  vec3 alb = vec3(0.50, 0.41, 0.31) * (0.96 + 0.08 * g0 + 0.05 * clump);
  float hue = hash12(gcell + 21.7);
  alb *= hue < 0.1 ? vec3(1.04, 0.95, 0.93) : hue < 0.2 ? vec3(0.95, 0.97, 1.02) : hue < 0.3 ? vec3(1.03, 1.0, 0.9) : vec3(1.);
  alb = mix(alb, alb * 0.7, step(0.995, g0));

  vec3 L = normalize(uSun);
  float dif = max(dot(n, L), 0.) * N.z;
  vec3 sun = vec3(1.0, 0.92, 0.80) * 1.12;
  vec3 sky = vec3(0.52, 0.62, 0.84) * 0.52;
  vec3 bounce = vec3(0.95, 0.80, 0.60) * 0.12;
  vec3 col = alb * (sun * dif + (sky * (0.55 + 0.45 * n.z) + bounce) * N.w);
  vec3 hv = normalize(L + vec3(0., 0., 1.));
  col += vec3(1.0, 0.95, 0.88) * 0.06 * pow(max(dot(n, hv), 0.), 12.) * N.z;
  col = pow(max(col, 0.), vec3(0.97));
  col = mix(col, col * col * (3. - 2. * col), 0.3);

  // the shadow the block casts on whatever the page is
  float shadowA = clamp((1. - N.z) * 0.34 + (1. - N.w) * 0.5, 0., 0.6);
  vec3 shadowC = vec3(0.16, 0.11, 0.07);
  o = vec4(col * cover + shadowC * shadowA * (1. - cover), cover + shadowA * (1. - cover));
}`;

const VS_GRAIN = `#version 300 es
in vec4 aP;               // x, y (stage CSS px, y down), height, size
in float aShade;
uniform vec2 uStage;
uniform float uDpr, uShadow;
uniform vec2 uShadowDir;
out float vShade;
void main(){
  vec2 p = aP.xy + (uShadow > 0.5 ? uShadowDir * aP.z : vec2(0., -aP.z * 0.35));
  gl_Position = vec4(p.x / uStage.x * 2. - 1., 1. - p.y / uStage.y * 2., 0., 1.);
  gl_PointSize = aP.w * uDpr;
  vShade = aShade;
}`;
const FS_GRAIN = `#version 300 es
precision highp float;
in float vShade;
uniform float uShadow;
out vec4 o;
void main(){
  vec2 c = gl_PointCoord - 0.5;
  float r = length(c);
  if (r > 0.5) discard;
  if (uShadow > 0.5){ float a = 0.25 * smoothstep(0.5, 0.2, r); o = vec4(vec3(0.16, 0.11, 0.07) * a, a); return; }
  vec3 base = mix(vec3(0.40, 0.32, 0.23), vec3(0.62, 0.51, 0.38), vShade) * (0.8 + 0.35 * clamp(-c.x - c.y + 0.3, 0., 1.));
  float a = smoothstep(0.5, 0.35, r);
  o = vec4(base * a, a);
}`;

// ---------------------------------------------------------------- audio (shared)
let actx = null;
const sandAudio = { muted: false };
function audio(){
  if (sandAudio.muted) return null;
  if (!actx){ try { actx = new (window.AudioContext || window.webkitAudioContext)(); } catch { return null; } }
  if (actx.state === 'suspended') actx.resume();
  return actx;
}
let noiseBuf = null;
function noise(a){
  if (noiseBuf && noiseBuf.sampleRate === a.sampleRate) return noiseBuf;
  const len = a.sampleRate * 2, b = a.createBuffer(1, len, a.sampleRate), d = b.getChannelData(0);
  for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
  return (noiseBuf = b);
}
function burst(type, freq, q, gain, dur, delay = 0){
  const a = audio(); if (!a) return;
  const t = a.currentTime + delay;
  const src = a.createBufferSource(); src.buffer = noise(a);
  const f = a.createBiquadFilter(); f.type = type; f.frequency.value = freq; f.Q.value = q;
  const g = a.createGain();
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(gain, t + Math.min(0.01, dur * 0.2));
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  src.connect(f).connect(g).connect(a.destination);
  src.start(t, Math.random() * 1.5); src.stop(t + dur + 0.02);
}
function patter(n, spread, gain, delay = 0){
  for (let i = 0; i < n; i++) burst('highpass', 3500 + Math.random() * 3000, 0.7, gain * (0.5 + Math.random() * 0.5), 0.012 + Math.random() * 0.012, delay + Math.random() * spread);
}
const sounds = {
  press(){ burst('lowpass', 600, 0.7, 0.06, 0.08); patter(3, 0.05, 0.025); },
  slump(){ burst('lowpass', 380, 0.8, 0.14, 0.7); burst('bandpass', 900, 0.6, 0.05, 0.8, 0.04); patter(26, 0.8, 0.04, 0.05); },
  reform(){
    const a = audio(); if (!a) return;
    const t = a.currentTime, o = a.createOscillator(), g = a.createGain();
    o.type = 'sine'; o.frequency.setValueAtTime(120, t); o.frequency.exponentialRampToValueAtTime(60, t + 0.18);
    g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.16, t + 0.01); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.24);
    o.connect(g).connect(a.destination); o.start(t); o.stop(t + 0.26);
    burst('lowpass', 700, 0.7, 0.08, 0.2);
  },
};
let scrape = null;
function scrapeLevel(speed){
  const a = speed > 0 ? audio() : actx; if (!a) return;
  if (!scrape){
    if (speed <= 0) return;
    const src = a.createBufferSource(); src.buffer = noise(a); src.loop = true;
    const f = a.createBiquadFilter(); f.type = 'bandpass'; f.frequency.value = 2000; f.Q.value = 0.6;
    const g = a.createGain(); g.gain.value = 0;
    src.connect(f).connect(g).connect(a.destination); src.start();
    scrape = { src, f, g };
  }
  const lvl = sandAudio.muted ? 0 : Math.min(speed / 700, 1);
  scrape.g.gain.setTargetAtTime(0.1 * lvl, a.currentTime, 0.04);
  scrape.f.frequency.setTargetAtTime(1400 + 1400 * lvl, a.currentTime, 0.05);
}

// ---------------------------------------------------------------- component
const rand = (a, b) => a + Math.random() * (b - a);
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');

function attachSand(button, opts = {}){
  const SUN = opts.sun || [-0.56, 0.56, 0.62];
  const canvas = document.createElement('canvas');
  canvas.setAttribute('aria-hidden', 'true');
  Object.assign(canvas.style, { position: 'absolute', pointerEvents: 'none', zIndex: opts.zIndex ?? 10, left: '0px', top: '0px' });
  const gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: true, antialias: false, depth: false, stencil: false });
  if (!gl || !gl.getExtension('EXT_color_buffer_float')) return { detach(){}, supported: false };

  // ---- GL setup
  const compile = (type, src) => {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  const program = (fs, vs = VS) => {
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, vs)); gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {};
    for (let i = 0, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i < n; i++){
      const info = gl.getActiveUniform(p, i);
      u[info.name] = gl.getUniformLocation(p, info.name);
    }
    return { p, u };
  };
  const P = {
    target: program(FS_TARGET), stamp: program(FS_STAMP), slide: program(FS_SLIDE), slump: program(FS_SLUMP),
    shade: program(FS_SHADE), comp: program(FS_COMPOSITE), grain: program(FS_GRAIN, VS_GRAIN),
  };
  const vao = gl.createVertexArray();
  const tex = (w, h, internal, format, type, filter) => {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, null);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, filter], [gl.TEXTURE_MAG_FILTER, filter], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
    return t;
  };
  const target = (...ts) => {
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    ts.forEach((t, i) => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t, 0));
    if (ts.length > 1) gl.drawBuffers(ts.map((_, i) => gl.COLOR_ATTACHMENT0 + i));
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return fb;
  };
  const bindTex = (unit, t) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, t); };
  const textTex = gl.createTexture();
  const grainBuf = gl.createBuffer(), grainVao = gl.createVertexArray();
  gl.bindVertexArray(grainVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, grainBuf);
  { const l = gl.getAttribLocation(P.grain.p, 'aP'), s = gl.getAttribLocation(P.grain.p, 'aShade');
    gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, 4, gl.FLOAT, false, 20, 0);
    gl.enableVertexAttribArray(s); gl.vertexAttribPointer(s, 1, gl.FLOAT, false, 20, 16); }
  gl.bindVertexArray(null);

  // ---- geometry (CSS px unless noted)
  let bw = 0, bh = 0, M = 0, stageW = 0, stageH = 0, gs = 1, dpr = 1, GW = 0, GH = 0, radius = 0;
  let sand = [], sandFb = [], cur = 0, shadeA, shadeB, shadeFb;
  const toGrid = (x, y) => [(x + M) * gs, GH - (y + M) * gs];   // button-local CSS -> grid (y up)

  const saved = button.getAttribute('style');
  const cs0 = getComputedStyle(button);
  const label = button.textContent.trim();
  const font = `${cs0.fontStyle} ${cs0.fontWeight} ${cs0.fontSize} ${cs0.fontFamily}`;
  Object.assign(button.style, { background: 'transparent', borderColor: 'transparent', color: 'transparent', boxShadow: 'none', textShadow: 'none', position: cs0.position === 'static' ? 'relative' : cs0.position });
  document.body.appendChild(canvas);

  function simPass(prog, setup){
    const dst = 1 - cur;
    gl.bindFramebuffer(gl.FRAMEBUFFER, sandFb[dst]);
    gl.viewport(0, 0, GW, GH);
    gl.useProgram(prog.p);
    bindTex(0, sand[cur]); gl.uniform1i(prog.u.uSrc, 0);
    setup(prog.u);
    gl.bindVertexArray(vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    cur = dst;
  }

  function buildText(){
    const c = document.createElement('canvas');
    c.width = GW; c.height = GH;
    const x = c.getContext('2d');
    x.fillStyle = '#000'; x.fillRect(0, 0, GW, GH);
    x.font = font.replace(/(\d+(\.\d+)?)px/, (_, v) => `${parseFloat(v) * gs}px`);
    x.textAlign = 'center'; x.textBaseline = 'middle';
    x.filter = `blur(${0.9 * gs}px)`;
    x.fillStyle = '#fff';
    x.fillText(label, GW / 2, GH / 2 + parseFloat(cs0.fontSize) * gs * 0.05);
    gl.bindTexture(gl.TEXTURE_2D, textTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, c);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
  }

  function targetUniforms(u, amt, soft = 0, letters = 1, lip = 0){
    bindTex(1, textTex); gl.uniform1i(u.uText, 1);
    gl.uniform4f(u.uBox, GW / 2, GH / 2, bw / 2 * gs, bh / 2 * gs);
    gl.uniform1f(u.uRadius, radius * gs);
    gl.uniform1f(u.uHeight, Math.min(Math.max(bh * 0.3, 10), 24) * gs);
    gl.uniform1f(u.uShoulder, Math.max(3, bh * 0.07) * gs);
    gl.uniform1f(u.uTextDepth, 1.5 * gs);
    gl.uniform1f(u.uGs, gs);
    gl.uniform1f(u.uAmt, amt); gl.uniform1f(u.uSoft, soft); gl.uniform1f(u.uLetters, letters); gl.uniform1f(u.uLip, lip);
  }

  function place(){
    const r = button.getBoundingClientRect();
    Object.assign(canvas.style, { left: `${r.left + scrollX - M}px`, top: `${r.top + scrollY - M}px` });
  }

  function resize(){
    const r = button.getBoundingClientRect();
    bw = r.width; bh = r.height;
    M = Math.round(Math.max(40, bh * 0.9));
    stageW = bw + 2 * M; stageH = bh + 2 * M;
    dpr = Math.min(devicePixelRatio || 1, 2);
    gs = dpr;
    canvas.width = Math.round(stageW * dpr); canvas.height = Math.round(stageH * dpr);
    Object.assign(canvas.style, { width: `${stageW}px`, height: `${stageH}px` });
    radius = Math.min(parseFloat(getComputedStyle(button).borderTopLeftRadius) || 0, bw / 2, bh / 2);
    GW = Math.round(stageW * gs); GH = Math.round(stageH * gs);
    [...sand, shadeA, shadeB].forEach(t => t && gl.deleteTexture(t));
    [...sandFb, shadeFb].forEach(f => f && gl.deleteFramebuffer(f));
    sand = [0, 1].map(() => tex(GW, GH, gl.RGBA32F, gl.RGBA, gl.FLOAT, gl.NEAREST));
    sandFb = sand.map(t => target(t));
    shadeA = tex(GW, GH, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR);
    shadeB = tex(GW, GH, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR);
    shadeFb = target(shadeA, shadeB);
    cur = 0;
    buildText();
    simPass(P.target, u => targetUniforms(u, 1));
    state = 'rest'; slump = null; reform = null; grains.length = 0;
    place();
    wake(0.1);
  }

  // ---- behaviour
  let state = 'rest';          // rest | slumped | carved | reforming
  let slump = null, reform = null, quietAt = 0, simT = 0, activeUntil = 0, talusSeed = 0;
  const stamps = [], grains = [];
  let ptr = null;
  const DRAG_START = 5;
  const finger = () => Math.max(4, bh * 0.13);

  // one animation loop at most: a wake from inside a frame must not start a second loop
  function wake(sec){ activeUntil = Math.max(activeUntil, simT + sec); if (!raf && !inFrame) raf = requestAnimationFrame(frame); }

  function stamp(x, y, r, depth){ stamps.push({ x, y, r, depth }); wake(2); }

  function startSlump(x, y){
    if (reduceMotion.matches){ stamp(x, y, finger() * 1.4, bh * 0.08); state = 'carved'; quietAt = simT + 0.5; return; }
    slump = { x, y, t: 0 };
    state = 'slumped'; quietAt = simT + 0.9;
    sounds.slump();
    // a few grains tumble off the edges as it goes
    for (let i = 0; i < 26; i++){
      const side = Math.random() < 0.5 ? -1 : 1, along = rand(-0.5, 0.5);
      const edgeX = Math.random() < 0.5, gx = edgeX ? side * bw / 2 : along * bw, gy = edgeX ? along * bh : side * bh / 2;
      grains.push({ x: bw / 2 + gx, y: bh / 2 + gy, z: rand(2, 8), vx: Math.sign(gx) * rand(10, 60) * (edgeX ? 1 : 0.3), vy: Math.sign(gy) * rand(10, 60) * (edgeX ? 0.3 : 1), vz: rand(0, 60), size: rand(0.9, 1.8), shade: Math.random() });
    }
    wake(3);
  }

  function layStamps(x0, y0, x1, y1){
    const dx = x1 - x0, dy = y1 - y0, dist = Math.hypot(dx, dy);
    if (dist < 0.5) return;
    const fr = finger(), n = Math.min(40, Math.ceil(dist / (fr * 0.3)));
    for (let i = 1; i <= n; i++) stamp(x0 + dx * i / n, y0 + dy * i / n, fr * rand(0.93, 1.07), fr * 1.1 * rand(0.9, 1.1));
  }

  const local = e => { const r = button.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  let suppressClick = false;
  function onDown(e){
    if (e.button !== 0 || ptr) return;
    audio();
    const [x, y] = local(e);
    ptr = { id: e.pointerId, x0: x, y0: y, x, y, lx: x, ly: y, dragging: false, speed: 0, t: performance.now() };
    try { button.setPointerCapture(e.pointerId); } catch {}
    if (reform){ reform = null; state = 'carved'; }
    // the fingertip sinks in while held
    stamp(x, y, finger() * 1.15, bh * 0.05);
    sounds.press();
  }
  function onMove(e){
    if (!ptr || e.pointerId !== ptr.id) return;
    const [x, y] = local(e);
    ptr.x = x; ptr.y = y;
    if (!ptr.dragging && Math.hypot(x - ptr.x0, y - ptr.y0) > DRAG_START){ ptr.dragging = true; ptr.lx = ptr.x0; ptr.ly = ptr.y0; }
  }
  function onUp(e, cancel){
    if (!ptr || e.pointerId !== ptr.id) return;
    const p = ptr; ptr = null;
    scrapeLevel(0);
    if (p.dragging){
      layStamps(p.lx, p.ly, p.x, p.y);
      suppressClick = true;              // carving the sand is not pressing the button
      state = 'carved'; quietAt = simT + 1.1;
    } else if (!cancel && state !== 'slumped') startSlump(p.x, p.y);
  }
  function onClick(e){
    if (suppressClick){ suppressClick = false; e.preventDefault(); e.stopImmediatePropagation(); return; }
    // keyboard or programmatic activation
    if (e.detail === 0 && state !== 'slumped'){ audio(); startSlump(bw / 2, bh / 2); }
  }
  const onCancel = e => onUp(e, true);
  button.addEventListener('pointerdown', onDown);
  button.addEventListener('pointermove', onMove);
  button.addEventListener('pointerup', onUp);
  button.addEventListener('pointercancel', onCancel);
  button.addEventListener('lostpointercapture', onCancel);
  button.addEventListener('click', onClick, true);

  // ---- frame
  let raf = 0, inFrame = false, last = performance.now(), dirty = true;
  const grainData = new Float32Array(5 * 256);

  function step(dt){
    simT += dt;
    if (ptr && ptr.dragging){
      const d = Math.hypot(ptr.x - ptr.lx, ptr.y - ptr.ly);
      ptr.speed += (d / Math.max(dt, 1e-3) - ptr.speed) * Math.min(1, dt * 12);
      scrapeLevel(ptr.speed);
      layStamps(ptr.lx, ptr.ly, ptr.x, ptr.y);
      ptr.lx = ptr.x; ptr.ly = ptr.y;
      state = 'carved'; quietAt = simT + 1.1;
    }
    if (slump){ slump.t += dt; if (slump.t > 1.3) slump = null; }
    if (!ptr && !reform && (state === 'slumped' || state === 'carved') && simT > quietAt){
      reform = { t: 0 }; wake(2);
    }
    if (reform){
      reform.t += dt;
      if (reform.t > 0.95){ reform = null; state = 'rest'; sounds.reform(); wake(0.3); }
    }
    for (const g of grains){ g.vz -= 900 * dt; g.x += g.vx * dt; g.y += g.vy * dt; g.z += g.vz * dt; }
    for (let i = grains.length - 1; i >= 0; i--) if (grains[i].z <= 0) grains.splice(i, 1);
  }

  function simulate(dt){
    for (const s of stamps){
      const [gx, gy] = toGrid(s.x, s.y), r = s.r * gs, b = r * 1.55;
      let nd = 0, nr = 0;
      for (let j = Math.floor(gy - b); j <= Math.ceil(gy + b); j++) for (let i = Math.floor(gx - b); i <= Math.ceil(gx + b); i++){
        const d = Math.hypot(i + 0.5 - gx, j + 0.5 - gy);
        if (d < r) nd++; else if (d < b) nr++;
      }
      simPass(P.stamp, u => {
        gl.uniform4f(u.uTool, gx, gy, r, s.depth * gs);
        gl.uniform1f(u.uJfix, nr ? (nd / nr) / (r * r / (b * b - r * r)) : 1);
        gl.uniform1f(u.uSeed, Math.random() * 100);
      });
    }
    stamps.length = 0;
    if (slump){
      const [fx, fy] = toGrid(slump.x, slump.y);
      simPass(P.slump, u => {
        gl.uniform2f(u.uFrom, fx, fy); gl.uniform1f(u.uT, slump.t);
        gl.uniform1f(u.uSpeed, Math.max(bw, bh) * gs * 3.5); gl.uniform1f(u.uDecay, Math.exp(-dt * 8));
        gl.uniform1f(u.uSlab, Math.max(9, bh * 0.34) * gs);
      });
    }
    if (reform){
      // gather into a mound, press the mould down (a lip squeezes out), punch the letters in
      const t = reform.t, ease = v => v * v * (3 - 2 * v);
      let amt, soft, letters, lip;
      if (t < 0.45){ amt = (1 - Math.exp(-dt / 0.14)) * Math.min(1, t / 0.12); soft = 1; letters = 0; lip = 0; }
      else if (t < 0.75){ const v = (t - 0.45) / 0.3; amt = 1 - Math.exp(-dt / 0.03); soft = 1 - ease(v); letters = 0; lip = Math.sin(Math.PI * v); }
      else { const v = Math.min(1, (t - 0.75) / 0.12); amt = 1; soft = 0; letters = ease(v); lip = 0; }
      simPass(P.target, u => targetUniforms(u, amt, soft, letters, lip));
    }
    if (Math.random() < dt * 4) talusSeed = Math.random() * 1000;
    const decay = Math.exp(-dt / 0.9 / SLIDE_ITERS);
    const melting = slump || state === 'slumped';
    for (let i = 0; i < SLIDE_ITERS; i++){
      simPass(P.slide, u => {
        gl.uniform1f(u.uTl, TALUS_LOOSE); gl.uniform1f(u.uTp, TALUS_PACKED); gl.uniform1f(u.uFluidDecay, decay);
        gl.uniform1f(u.uMelt, melting ? 1 : 0); gl.uniform1f(u.uSeed, talusSeed);
        gl.uniform1f(u.uR1, Math.round(2 * gs)); gl.uniform1f(u.uR2, Math.round(5 * gs));
      });
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, shadeFb);
    gl.viewport(0, 0, GW, GH);
    gl.useProgram(P.shade.p);
    bindTex(0, sand[cur]); gl.uniform1i(P.shade.u.uSrc, 0);
    gl.uniform3f(P.shade.u.uSun, ...SUN); gl.uniform1f(P.shade.u.uGs, gs);
    gl.bindVertexArray(vao); gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  function render(){
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    const c = P.comp;
    gl.useProgram(c.p);
    bindTex(0, shadeA); gl.uniform1i(c.u.uN, 0);
    bindTex(1, shadeB); gl.uniform1i(c.u.uQ, 1);
    gl.uniform2f(c.u.uGrid, GW, GH); gl.uniform1f(c.u.uDpr, dpr); gl.uniform1f(c.u.uGs, gs);
    gl.uniform3f(c.u.uSun, ...SUN);
    gl.bindVertexArray(vao); gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (grains.length){
      const n = Math.min(grains.length, 256);
      for (let i = 0; i < n; i++){ const g = grains[i]; grainData.set([g.x + M, g.y + M, g.z, g.size, g.shade], i * 5); }
      gl.bindBuffer(gl.ARRAY_BUFFER, grainBuf);
      gl.bufferData(gl.ARRAY_BUFFER, grainData.subarray(0, n * 5), gl.DYNAMIC_DRAW);
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.useProgram(P.grain.p);
      gl.uniform2f(P.grain.u.uStage, stageW, stageH); gl.uniform1f(P.grain.u.uDpr, dpr);
      gl.uniform2f(P.grain.u.uShadowDir, -SUN[0] * 0.9, SUN[1] * 0.9);
      gl.bindVertexArray(grainVao);
      for (const sh of [1, 0]){ gl.uniform1f(P.grain.u.uShadow, sh); gl.drawArrays(gl.POINTS, 0, n); }
      gl.disable(gl.BLEND); gl.bindVertexArray(null);
    }
  }

  let manual = false;
  let frameCount = 0;
  function advance(dt){
    frameCount++;
    step(dt);
    const busy = simT < activeUntil || slump || reform || ptr || stamps.length || grains.length;
    if (busy || dirty){ simulate(dt); render(); dirty = false; }
    return busy;
  }
  function frame(now){
    raf = 0;
    const dt = Math.min((now - last) / 1000, 1 / 30); last = now;
    if (manual || gl.isContextLost()) return;
    inFrame = true;
    let busy;
    try { busy = advance(dt); } finally { inFrame = false; }
    if (busy) raf = requestAnimationFrame(frame);
    else last = performance.now();
  }

  // ---- keep the stage on the button
  const ro = new ResizeObserver(() => { const r = button.getBoundingClientRect(); if (Math.abs(r.width - bw) > 0.5 || Math.abs(r.height - bh) > 0.5) resize(); else place(); });
  ro.observe(button);
  const onScrollResize = () => place();
  addEventListener('resize', onScrollResize); addEventListener('scroll', onScrollResize, { passive: true });
  canvas.addEventListener('webglcontextlost', e => e.preventDefault());
  canvas.addEventListener('webglcontextrestored', () => location.reload());

  resize();

  return {
    supported: true,
    get state(){ return state; },
    get _frames(){ return frameCount; },
    detach(){
      cancelAnimationFrame(raf); ro.disconnect();
      removeEventListener('resize', onScrollResize); removeEventListener('scroll', onScrollResize);
      button.removeEventListener('pointerdown', onDown); button.removeEventListener('pointermove', onMove);
      button.removeEventListener('pointerup', onUp); button.removeEventListener('pointercancel', onCancel);
      button.removeEventListener('lostpointercapture', onCancel); button.removeEventListener('click', onClick, true);
      if (saved === null) button.removeAttribute('style'); else button.setAttribute('style', saved);
      canvas.remove();
      gl.getExtension('WEBGL_lose_context')?.loseContext();
    },
    // test hooks
    _manual(on){ manual = on; if (!on) wake(0.1); },
    _advance(sec, fps = 60){ for (let i = 0, n = Math.round(sec * fps); i < n; i++) advance(1 / fps); },
  };
}

window.ElementalFX = Object.assign(window.ElementalFX || {}, { attachSand, sandAudio });
})();
