// Bubble button: turns an existing <button> into a soap bubble shaped like the pill.
//
//   <script src="fx/bubble.js"></script>
//   const fx = ElementalFX.attachBubble(document.querySelector('#send'));   // fx.detach() restores it
//
// A classic script (not a module) so pages work straight from file:// too.
//
// The <button> stays the real control (focus, keyboard, form submission, click handlers); this only
// draws it, in a transparent canvas laid over the page around it. The film drifts round slowly when the
// cursor stirs the air near it, pressing squeezes it, releasing pops it (that release is the button's click):
// the film tears open from the click point, throwing a fine mist and a few small bubbles that float
// off. Those can be tapped to pop or dragged around; after a quiet moment they drift back, merge, and
// a new bubble is blown in place. If WebGL2 is unavailable, the button is left untouched.
//
// Options:
//   background: CSS colour of the page under the bubble (default: detected from the ancestors).
//   labelColor: CSS colour for the label seen through the film (default: the button's colour).
//   margin: CSS px of stage around the button (default max(170, 2.6 x button height)).
//   zIndex: of the stage canvas (default 10).   sound: false to stay silent.

(() => {
'use strict';

const TAU = Math.PI * 2;
const REGROUP_AFTER = 2.4;
const DRAG_START = 6;
const SLEEP_MS = 9000;          // film keeps flowing this long after the last interaction, then the frame freezes
const MAXB = 24, MAXM = 128;
const MAXPOP = 18;              // most small bubbles one pop can throw

// ---------------------------------------------------------------- shaders
const VS = `#version 300 es
void main(){ vec2 p = vec2(float((gl_VertexID<<1)&2), float(gl_VertexID&2)); gl_Position = vec4(p*2.-1., 0., 1.); }`;

const NOISE = `
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*.1031); p3 += dot(p3, p3.yzx+33.33); return fract((p3.x+p3.y)*p3.z); }
float vnoise(vec2 p){
  vec2 i = floor(p), f = fract(p);
  vec2 u = f*f*(3.-2.*f);
  float a = hash12(i), b = hash12(i+vec2(1,0)), c = hash12(i+vec2(0,1)), d = hash12(i+vec2(1,1));
  return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}
float fbm(vec2 p, int oct){
  float s = 0., a = .5, w = 0.;
  mat2 m = mat2(1.62, 1.18, -1.18, 1.62);
  for (int i = 0; i < 6; i++){ if (i >= oct) break; s += a*vnoise(p); w += a; p = m*p + 17.13; a *= .5; }
  return s / w;
}`;

// Output is premultiplied. The page under the stage is not ours to redraw, so the colour T we want is
// produced as src + (1 - a) * page, with a only as large as T's departure from the page requires.
const FS = `#version 300 es
precision highp float;
uniform vec2 uRes;
uniform float uDpr;
uniform float uTime;
uniform vec3 uBg;
uniform vec4 uPill;          // cx, cy, halfW, r (device px); r <= 0: no pill
uniform vec4 uModes[4];      // amp, m, phase, _   rim waves
uniform vec4 uHole;          // x, y, radius, rim width (device px); radius < 0: intact
uniform vec3 uPillMisc;      // label alpha, seed, shadow strength
uniform vec4 uAir;           // x, y (device px), swirl angle, radius (device px)
uniform int uNB;
uniform vec4 uB[${MAXB}];    // x, y, r, seed
uniform vec4 uBH[${MAXB}];   // hole x, y, radius (< 0: intact), lift above the page (device px)
uniform int uNM;
uniform vec4 uM[${MAXM}];    // x, y, r, alpha
uniform sampler2D uLabel;
uniform vec4 uLabelRect;     // cx, cy, halfW, halfH (device px)
out vec4 o;
${NOISE}

float gs(float x, float m, float s){ float z = (x - m) / s; return exp(-z*z); }

// Reflectance of a soap film of thickness d nm, as RGB, 0 for a black film, ~0.5 for a thick one.
// Two-beam interference sampled across the spectrum, with rough colour-matching curves.
vec3 film(float d, float cosT){
  vec3 acc = vec3(0.), ws = vec3(0.);
  for (int i = 0; i < 10; i++){
    float lam = 395. + float(i) * 33.;
    vec3 w = vec3(gs(lam, 600., 45.) + 0.32*gs(lam, 435., 22.), gs(lam, 548., 42.), gs(lam, 452., 30.));
    acc += w * (1. - cos(12.566 * 1.33 * d * cosT / lam));
    ws += w;
  }
  vec3 f = acc / ws * 0.5;
  float l = dot(f, vec3(0.3333));
  return max(mix(vec3(l), f, 1.9), 0.);
}

// Film thickness in nm at bubble-local u (y up, ~unit radius). Drains downward: thin, with black
// patches, at the top, thick at the bottom; Marangoni swirls ride on that and the cursor stirs it.
float thick(vec2 u, float seed, vec2 au, float aAng, float aRad){
  vec2 dq = u - au;
  float ang = aAng * exp(-dot(dq, dq) / (aRad * aRad));
  float cs = cos(ang), sn = sin(ang);
  u = au + mat2(cs, sn, -sn, cs) * dq;
  float t = uTime;
  float h = clamp(0.5 - 0.5 * u.y, 0., 1.);
  vec2 w = u * vec2(0.55, 0.9) + seed * 7.1;
  vec2 q1 = vec2(fbm(w * 0.9 + vec2(0., t * 0.09), 3), fbm(w * 0.9 + vec2(4.3, 2.1 + t * 0.06), 3));
  float s = fbm(w * 0.8 + 2.2 * (q1 - 0.5) + vec2(t * 0.03, t * 0.12), 3);
  float d = mix(250., 760., pow(h, 1.2)) + (s - 0.5) * mix(300., 700., h);
  float blk = smoothstep(0.55, 0.95, u.y) * smoothstep(0.56, 0.7, fbm(w * 1.6 + vec2(t * 0.04, t * 0.07), 2));
  d = mix(d, 6., blk);
  return max(d, 4.);
}

// What the film reflects. z points at the viewer, y is up: a bright sky, a darker room below, and
// a four-pane window up and to the left (HDR, it becomes the specular highlight).
vec3 env(vec3 r){
  vec3 c = mix(vec3(0.20, 0.22, 0.27), vec3(0.98, 1.0, 1.05), smoothstep(-0.55, 0.65, r.y));
  c = mix(c, vec3(0.55, 0.58, 0.62), smoothstep(0.75, 1.0, r.z) * 0.5);
  vec2 w = vec2(r.x + 0.30, r.y - 0.50);
  vec2 qq = abs(w) - vec2(0.17, 0.10);
  float sdw = length(max(qq, 0.)) + min(max(qq.x, qq.y), 0.) - 0.06;
  c += vec3(2.6) * smoothstep(0.07, -0.03, sdw);
  vec3 Ls = normalize(vec3(0.62, 0.42, 0.66));
  c += vec3(3.0) * pow(max(dot(r, Ls), 0.), 120.);
  return c;
}

void shadeFilm(inout vec3 col, float cov, vec3 n, vec2 u, float seed, vec2 au, float aAng, float aRad){
  if (cov <= 0.) return;
  float cosI = max(n.z, 0.04);
  float cosT = sqrt(1. - (1. - cosI*cosI) / 1.769);
  vec3 f = film(thick(u, seed, au, aAng, aRad), cosT);
  float fres = pow(1. - cosI, 2.5);
  vec3 I = vec3(0., 0., -1.);
  vec3 e = env(reflect(I, n));
  // the far side of the bubble mirrors the room back, flipped: the second, dimmer highlight low right
  vec3 eb = env(reflect(I, vec3(-n.xy, n.z)));
  // the film is seen as reflected colour over a darker room, so it reads even on a white page;
  // a black (very thin) film reflects nothing and shows as a clear, slightly darker patch
  vec3 refl = f * 1.55 * (min(e, vec3(1.1)) * 0.85 + min(eb, vec3(1.1)) * 0.2);
  // a black film is nearly invisible, not grey
  float s = mix(0.035, 0.7, fres) * mix(0.1, 1., smoothstep(0.03, 0.36, dot(f, vec3(0.3333))));
  vec3 c = mix(col, refl, s);
  c += max(e - 1.2, 0.) * (0.35 + 0.65 * dot(f, vec3(0.6))) * 0.5;
  c += max(eb - 1.2, 0.) * 0.05;
  col = mix(col, c, cov);
}

// signed distance to the pill, plus a bubble normal and local coordinates
float sdPill(vec2 p, out vec3 n, out vec2 u){
  vec2 q = p - uPill.xy;
  float r = uPill.w, L = max(uPill.z - r, 0.);
  float ang = atan(q.y, q.x + 1e-4);
  float w = 0.;
  for (int i = 0; i < 4; i++) w += uModes[i].x * cos(uModes[i].y * (ang - uModes[i].z));
  vec2 near = vec2(clamp(q.x, -L, L), 0.);
  float dc = length(q - near) - r;
  float de = (length(q / vec2(uPill.z, r)) - 1.) * r;
  float d = mix(dc, de, 0.2) - w;
  vec2 dir = normalize(mix((q - near) / r, q / vec2(uPill.z, r), 0.6) + 1e-5);
  // round cross-section: the normal tilts in proportion to how far out from the middle ridge this point is
  vec2 nq = dir * (1. - clamp(-d / (r + w), 0., 1.));
  n = vec3(nq, sqrt(max(1. - dot(nq, nq), 0.)));
  u = q / r;
  return d;
}
float sdPillOnly(vec2 p){
  vec2 q = p - uPill.xy;
  float r = uPill.w, L = max(uPill.z - r, 0.);
  vec2 near = vec2(clamp(q.x, -L, L), 0.);
  return mix(length(q - near) - r, (length(q / vec2(uPill.z, r)) - 1.) * r, 0.2);
}

// torn film: gone inside the hole, with a thickened retracting rim just outside it
void hole(vec2 p, vec4 H, float seed, out float keep, out float rim){
  keep = 1.; rim = 0.;
  if (H.z < 0.) return;
  vec2 q = p - H.xy;
  float a = atan(q.y, q.x);
  float jag = (vnoise(vec2(a * 3.0 + seed * 9., seed)) - 0.5) * H.w * 1.6;
  float dd = length(q) - H.z - jag;
  keep = smoothstep(-1., 1., dd);
  rim = exp(-pow((dd - H.w * 0.6) / (H.w * 0.7), 2.));
}

void main(){
  vec2 p = gl_FragCoord.xy;
  float px = uDpr;
  vec3 B = uBg;
  vec3 col = B;

  // ---------------- on the page: faint shadows, each with a brighter caustic where the film focuses light
  if (uPill.w > 0.){
    float sd = sdPillOnly(p + vec2(0., 5. * px));
    float sh = smoothstep(9. * px, -9. * px, sd);
    float ca = exp(-pow((sd + uPill.w * 0.28) / (uPill.w * 0.16), 2.));
    // light passes through the film almost untouched: only the outline of the shadow darkens
    col *= 1. - uPillMisc.z * 0.07 * sh * smoothstep(-uPill.w * 0.6, -uPill.w * 0.05, sd);
    col += uPillMisc.z * 0.05 * ca * vec3(1.0, 0.96, 0.9) * sh;
  }
  for (int i = 0; i < ${MAXB}; i++){
    if (i >= uNB) break;
    vec4 b = uB[i]; float lift = uBH[i].w;
    vec2 sq = p - b.xy + vec2(0., lift);
    float sd = length(sq) - b.z;
    if (sd > b.z * 0.6 + 4. * px) continue;
    float s = (uBH[i].z < 0. ? 1. : 0.) * 0.03 / (1. + lift / (60. * px));
    float blur = 2. * px + b.z * 0.35 + lift * 0.15;
    col *= 1. - s * smoothstep(blur, -blur, sd) * smoothstep(-b.z * 0.7, -b.z * 0.1, sd);
  }

  // ---------------- the button bubble, with its label seen through the film
  if (uPill.w > 0.){
    vec3 n; vec2 u;
    float d = sdPill(p, n, u);
    if (d < 2. * px){
      float keep, rim;
      hole(p, uHole, uPillMisc.y, keep, rim);
      float cov = clamp(0.5 - d, 0., 1.) * keep;
      vec4 lab = texture(uLabel, (p - uLabelRect.xy) / (uLabelRect.zw * 2.) + 0.5);
      float la = uPillMisc.x * cov;
      col = col * (1. - lab.a * la) + lab.rgb * la;
      float r = uPill.w;
      shadeFilm(col, cov, n, u, uPillMisc.y, (uAir.xy - uPill.xy) / r, uAir.z, uAir.w / r);
      col = mix(col, col * 0.8 + 0.28, clamp(rim * clamp(0.5 - d, 0., 1.), 0., 1.) * 0.8);
    }
  }

  // ---------------- small bubbles
  for (int i = 0; i < ${MAXB}; i++){
    if (i >= uNB) break;
    vec4 b = uB[i];
    vec2 q = p - b.xy;
    float d = length(q) - b.z;
    if (d > 1.5 * px) continue;
    float keep, rim;
    hole(p, vec4(uBH[i].xyz, 2.4 * px), b.w, keep, rim);
    vec2 nq = q / max(b.z, 1.);
    nq *= min(1., 0.999 / max(length(nq), 1e-4));
    vec3 n = vec3(nq, sqrt(max(1. - dot(nq, nq), 0.)));
    shadeFilm(col, clamp(0.5 - d, 0., 1.) * keep, n, nq, b.w, (uAir.xy - b.xy) / b.z, uAir.z * 0.6, uAir.w / b.z);
    col = mix(col, col * 0.8 + 0.28, clamp(rim * clamp(0.5 - d, 0., 1.), 0., 1.) * 0.8);
  }

  // ---------------- mist: tiny droplets, dark-rimmed lenses with a pinpoint highlight
  for (int i = 0; i < ${MAXM}; i++){
    if (i >= uNM) break;
    vec4 m = uM[i];
    vec2 q = p - m.xy;
    float rr = m.z;
    float dd = dot(q, q);
    if (dd > (rr + 1.5) * (rr + 1.5)) continue;
    float d = sqrt(dd) - rr;
    float cov = clamp(0.5 - d, 0., 1.) * m.w;
    float rn = sqrt(dd) / max(rr, 0.6);
    vec3 dc = col * mix(0.94, 0.6, rn * rn);
    dc += vec3(0.7) * exp(-dot(q / max(rr, 0.6) - vec2(-0.3, 0.35), q / max(rr, 0.6) - vec2(-0.3, 0.35)) * 6.);
    col = mix(col, dc, cov);
  }

  col = clamp(col + (hash12(p + fract(uTime) * 100.) - 0.5) / 255., 0., 1.);
  vec3 Bs = max(B, vec3(0.02));
  vec3 qd = col / Bs;
  vec3 up = (col - B) / max(1. - B, vec3(0.02));
  float a = clamp(max(1. - min(min(qd.r, qd.g), qd.b), max(max(up.r, up.g), up.b)), 0., 1.);
  if (a < 0.003){ o = vec4(0.); return; }
  o = vec4(clamp(col - (1. - a) * B, 0., a), a);
}`;

// ---------------------------------------------------------------- audio
const bubbleAudio = { muted: false };
let actx = null;
function audio(){
  if (bubbleAudio.muted) return null;
  if (!actx){ try { actx = new (window.AudioContext || window.webkitAudioContext)(); } catch { return null; } }
  if (actx.state === 'suspended') actx.resume();
  return actx;
}
let noiseBuf = null;
function noise(a){
  if (noiseBuf && noiseBuf.sampleRate === a.sampleRate) return noiseBuf;
  const len = Math.floor(a.sampleRate * 0.8), b = a.createBuffer(1, len, a.sampleRate), d = b.getChannelData(0);
  for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
  return (noiseBuf = b);
}
// A burst of the shared noise, from a random point in it so no two sounds are identical.
// filters: [type, frequency, Q] applied in series; env: [[time, gain], ...] after t, ramped exponentially.
function burst(a, t, filters, env){
  const src = a.createBufferSource(); src.buffer = noise(a);
  let node = src;
  for (const [type, f, q] of filters){
    const bq = a.createBiquadFilter(); bq.type = type; bq.frequency.value = f; bq.Q.value = q;
    node.connect(bq); node = bq;
  }
  const g = a.createGain();
  g.gain.setValueAtTime(0.0001, t);
  for (const [dt, v] of env) g.gain.exponentialRampToValueAtTime(Math.max(v, 0.0001), t + dt);
  node.connect(g).connect(a.destination);
  const dur = env[env.length - 1][0];
  src.start(t, Math.random() * (src.buffer.duration - dur - 0.01));
  src.stop(t + dur + 0.01);
  return g;
}
const jit = (v, amt) => v * (1 + (Math.random() * 2 - 1) * amt);
// A soap film pop is nearly all air: a dry snap of a millisecond or two, bright (2-8 kHz), with a faint
// breath of released air behind it and no tone. Smaller bubbles are higher and quieter.
const sounds = {
  pop(size){
    const a = audio(); if (!a) return;
    const z = Math.max(0.15, Math.min(size, 1));
    const t = a.currentTime + 0.002;
    const fc = jit(6200 - 2800 * z, 0.12), lvl = jit(0.07 + 0.26 * z, 0.2);
    burst(a, t, [['highpass', 1800, 0.7], ['bandpass', fc, 0.55]], [[0.0003, lvl], [jit(0.0012 + 0.0016 * z, 0.25), lvl * 0.06], [0.004 + 0.002 * z, 0.0001]]);
    // the rim often snaps twice as it retracts
    if (Math.random() < 0.55){
      const t2 = t + rand(0.0007, 0.0022), l2 = lvl * rand(0.25, 0.55);
      burst(a, t2, [['highpass', 2200, 0.7], ['bandpass', fc * rand(1.05, 1.3), 0.6]], [[0.0002, l2], [0.0012, l2 * 0.05], [0.003, 0.0001]]);
    }
    const pl = jit(0.012 + 0.03 * z, 0.25);
    burst(a, t, [['bandpass', jit(3400 - 1500 * z, 0.15), 0.5], ['lowpass', 7000, 0.5]], [[0.0015, pl], [jit(0.018 + 0.03 * z, 0.2), pl * 0.2], [0.03 + 0.04 * z, 0.0001]]);
  },
  merge(){
    const a = audio(); if (!a) return;
    const t = a.currentTime + 0.002, l = jit(0.028, 0.25);
    burst(a, t, [['bandpass', jit(2600, 0.15), 0.9]], [[0.0004, l], [0.0025, l * 0.1], [0.006, 0.0001]]);
    // the wet part: a very short, quiet upward chirp as the two films join
    const o = a.createOscillator(), g = a.createGain();
    const f = jit(1500, 0.15);
    o.type = 'sine';
    o.frequency.setValueAtTime(f, t); o.frequency.exponentialRampToValueAtTime(f * 1.9, t + 0.018);
    g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(jit(0.009, 0.3), t + 0.002); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.022);
    o.connect(g).connect(a.destination); o.start(t); o.stop(t + 0.03);
  },
  // breath through a wand: soft, low-passed air that swells and fades, with a slight lip flutter
  blow(){
    const a = audio(); if (!a) return;
    const t = a.currentTime + 0.005, d = jit(0.62, 0.1);
    const body = burst(a, t, [['bandpass', jit(700, 0.1), 0.45], ['lowpass', 2200, 0.5]], [[0.2, 0.05], [d * 0.7, 0.04], [d, 0.0001]]);
    burst(a, t, [['highpass', 2500, 0.6], ['lowpass', 7000, 0.5]], [[0.22, 0.012], [d * 0.7, 0.008], [d, 0.0001]]);
    const lfo = a.createOscillator(), depth = a.createGain();
    lfo.frequency.value = rand(9, 13); depth.gain.value = 0.012;
    lfo.connect(depth).connect(body.gain); lfo.start(t); lfo.stop(t + d + 0.02);
  },
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
  if (!probe){ probe = document.createElement('canvas').getContext('2d'); }
  probe.fillStyle = '#010203'; probe.fillStyle = css;
  const s = probe.fillStyle;
  if (s === '#010203') return null;
  if (s[0] === '#') return [parseInt(s.slice(1, 3), 16) / 255, parseInt(s.slice(3, 5), 16) / 255, parseInt(s.slice(5, 7), 16) / 255, 1];
  return s === css ? null : parseColor(s);
}
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

// ---------------------------------------------------------------- component
function attachBubble(button, opts = {}){
  const noop = () => {};
  const unsupported = { supported: false, state: 'unsupported', detach: noop, refresh: noop, _manual: noop, _advance: noop };
  const canvas = document.createElement('canvas');
  canvas.setAttribute('aria-hidden', 'true');
  Object.assign(canvas.style, { position: 'absolute', left: '0px', top: '0px', margin: '0', padding: '0', border: '0', display: 'block', pointerEvents: 'none', zIndex: String(opts.zIndex ?? 10) });
  const gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: true, antialias: false, depth: false, stencil: false });
  if (!gl) return unsupported;
  const sound = opts.sound === false ? { pop: noop, merge: noop, blow: noop } : sounds;

  let P, vao, labelTex;
  const bBuf = new Float32Array(MAXB * 4), bhBuf = new Float32Array(MAXB * 4), mBuf = new Float32Array(MAXM * 4), modeBuf = new Float32Array(16);
  function initGL(){
    P = compileProgram(gl, FS);
    vao = gl.createVertexArray();
    labelTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, labelTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  }
  try { initGL(); } catch (err) {
    console.warn('ElementalFX bubble: shaders failed, leaving the button as it is.', err);
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return unsupported;
  }

  // ---- geometry: physics in button-local CSS px (origin at the button's top-left, y down)
  const G = { fixed: false, bx: NaN, by: NaN, bw: 0, bh: 0, stageW: 0, stageH: 0, offX: 0, offY: 0 };
  let bounds = { w: 0, h: 0 }, cbOrigin = { x: 0, y: 0 };
  let dpr = 1, dprY = 1, W = 0, H = 0;
  let bgCol = [1, 1, 1], labelKey = '';
  const seed = Math.random();

  const pill = {
    cx: 0, cy: 0, hw: 0, hh: 0,
    scale: 1, scaleV: 0,
    sx: 1, sxV: 0, sy: 1, syV: 0,
    press: 0, pressV: 0, pressT: 0,
    modes: [0, 1, 2, 3].map(() => ({ amp: 0, m: 2, phi: 0, w: 0, t: 0 })),
    label: 1,
  };
  const kk = () => pill.hh / 28;
  const bubbleBase = () => Math.sqrt(27 * pill.hh);
  // tx/ty and target are where the cursor puts the swirl; x/y and ang ease after them, so the film drifts
  // round lazily instead of snapping to each mouse movement.
  const air = { x: 0, y: 0, tx: 0, ty: 0, ang: 0, angV: 0, target: 0, lx: NaN, ly: NaN, lt: 0 };

  // ---- the button: invisible but still there
  const saved = button.getAttribute('style');
  let hidden = false;
  function hideButton(){ hidden = true; Object.assign(button.style, { background: 'transparent', borderColor: 'transparent', color: 'transparent', boxShadow: 'none', textShadow: 'none' }); }
  function showButton(){ hidden = false; if (saved === null) button.removeAttribute('style'); else button.setAttribute('style', saved); }
  const basePE = (() => { const t = document.createElement('div'); t.setAttribute('style', saved || ''); return t.style.pointerEvents; })();
  let peOff = false, peTimer = 0;
  function disablePointer(){ clearTimeout(peTimer); if (phase === 'idle' || peOff || !hidden) return; peOff = true; button.style.pointerEvents = 'none'; }
  function enablePointer(){ clearTimeout(peTimer); if (!peOff) return; peOff = false; button.style.pointerEvents = basePE; }

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
    canvas.style.left = '0px'; canvas.style.top = '0px';
    const r = canvas.getBoundingClientRect();
    cbOrigin = { x: r.left + (G.fixed ? 0 : scrollX), y: r.top + (G.fixed ? 0 : scrollY) };
    G.bx = NaN;
  }
  function layout(r = button.getBoundingClientRect()){
    const bx = r.left + (G.fixed ? 0 : scrollX), by = r.top + (G.fixed ? 0 : scrollY);
    const bw = r.width, bh = r.height;
    if (bw < 2 || bh < 2){ canvas.style.visibility = 'hidden'; G.bx = bx; G.by = by; return; }
    canvas.style.visibility = '';
    const M = opts.margin ?? Math.round(Math.max(170, 2.6 * bh));
    // the stage stays inside the document, so it can never add scrolling
    const x0 = Math.floor(Math.min(bx, Math.max(0, bx - M))), y0 = Math.floor(Math.min(by, Math.max(0, by - M)));
    const x1 = Math.ceil(Math.max(bx + bw, Math.min(bounds.w, bx + bw + M))), y1 = Math.ceil(Math.max(by + bh, Math.min(bounds.h, by + bh + M)));
    const sw = x1 - x0, sh = y1 - y0;
    const sizeChanged = sw !== G.stageW || sh !== G.stageH;
    const btnChanged = Math.abs(bw - G.bw) > 0.25 || Math.abs(bh - G.bh) > 0.25;
    Object.assign(G, { bx, by, bw, bh, stageW: sw, stageH: sh, offX: bx - x0, offY: by - y0 });
    Object.assign(canvas.style, { left: `${x0 - cbOrigin.x}px`, top: `${y0 - cbOrigin.y}px` });
    if (btnChanged){
      const ox = pill.cx, oy = pill.cy;
      pill.hw = bw / 2; pill.hh = bh / 2; pill.cx = bw / 2; pill.cy = bh / 2;
      for (const b of bubbles){ b.x += pill.cx - ox; b.y += pill.cy - oy; }
    }
    if (sizeChanged){
      Object.assign(canvas.style, { width: `${sw}px`, height: `${sh}px` });
      resizeGL();
    } else if (btnChanged) buildLabel();
    dirty = true;
    wake();
  }
  function place(){
    const r = button.getBoundingClientRect();
    const bx = r.left + (G.fixed ? 0 : scrollX), by = r.top + (G.fixed ? 0 : scrollY);
    if (Math.abs(bx - G.bx) > 0.25 || Math.abs(by - G.by) > 0.25 || Math.abs(r.width - G.bw) > 0.25 || Math.abs(r.height - G.bh) > 0.25) layout(r);
  }
  function resizeGL(){
    if (lost || gl.isContextLost() || !G.stageW) return;
    let d = Math.min(devicePixelRatio || 1, 2);
    const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    d = Math.min(d, maxTex / G.stageW, maxTex / G.stageH);
    canvas.width = Math.max(1, Math.round(G.stageW * d)); canvas.height = Math.max(1, Math.round(G.stageH * d));
    W = gl.drawingBufferWidth; H = gl.drawingBufferHeight;
    dpr = W / G.stageW; dprY = H / G.stageH;
    buildLabel();
    dirty = true;
  }

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
    let cx = G.bw / 2, cy = G.bh / 2;
    const range = document.createRange();
    range.selectNodeContents(button);
    const rr = range.getBoundingClientRect();
    const br = button.getBoundingClientRect();
    if (rr.width > 0 && rr.height > 0){ cx = rr.left + rr.width / 2 - br.left; cy = rr.top + rr.height / 2 - br.top; }
    const m = x.measureText(text);
    const asc = m.fontBoundingBoxAscent ?? fsize * dpr * 0.8, desc = m.fontBoundingBoxDescent ?? fsize * dpr * 0.2;
    const px = (cx + pad) * dpr, py = (cy + pad) * dpr + (asc - desc) / 2;
    const col = parseColor(opts.labelColor || cs.color) || [0.2, 0.2, 0.25, 1];
    x.fillStyle = `rgba(${Math.round(col[0] * 255)},${Math.round(col[1] * 255)},${Math.round(col[2] * 255)},${Math.max(col[3], 0.7)})`;
    x.fillText(text, px, py);
    gl.bindTexture(gl.TEXTURE_2D, labelTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, c);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.generateMipmap(gl.TEXTURE_2D);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
    labelKey = text + '|' + cs.font + '|' + cs.color;
    bgCol = opts.background ? (parseColor(opts.background) || [1, 1, 1]).slice(0, 3) : pageColor(button.parentElement);
    dirty = true;
  }

  // ---- simulation
  let phase = 'idle';      // idle | pop | scatter | reform | blow
  let phaseT = 0, simT = 0, lastBurst = 0, absorbed = 0, total = 1, settling = false;
  let bubbles = [], mist = [], pending = [];
  let pillHole = null;     // { x, y, R, v, max } while the button's film tears open

  function kickMode(amp, m){
    const md = pill.modes.reduce((a, b) => (a.amp <= b.amp ? a : b));
    md.amp = amp; md.m = m; md.phi = Math.random() * TAU; md.w = 9 + m * 3; md.t = 0;
  }
  // the pill's current outline (CSS px) from its blow-up scale: round while small, stretching into the pill
  function pillShape(){
    const s = pill.scale;
    const over = s > 1 ? 1 + (s - 1) * 0.6 : 1;
    const r = pill.hh * Math.min(1.5 * s, 1) * over;
    const L = Math.max(pill.hw - pill.hh, 0) * Math.max(0, Math.min((s - 0.35) / 0.65, 1)) * over;
    return { r: r * pill.sy, hw: (r + L) * pill.sx };
  }
  function pillSdf(x, y){
    const { r, hw } = pillShape();
    const L = Math.max(hw - r, 0);
    const qx = Math.max(Math.abs(x - pill.cx) - L, 0), qy = y - pill.cy;
    return Math.hypot(qx, qy) - r;
  }

  function mistAt(x, y, vx, vy, delay){
    pending.push({ t: phaseT + delay, kind: 'mist', x, y, vx, vy, r: rand(0.5, 1.5), life: rand(0.5, 1.2) });
  }

  function pop(ox = pill.cx, oy = pill.cy){
    if (phase !== 'idle') return;
    phase = 'pop'; phaseT = 0; settling = false;
    pill.pressT = 0;
    sound.pop(1);
    const k = kk(), L = pill.hw - pill.hh;
    ox = Math.max(pill.cx - pill.hw, Math.min(pill.cx + pill.hw, ox));
    oy = Math.max(pill.cy - pill.hh, Math.min(pill.cy + pill.hh, oy));
    const max = Math.hypot(Math.abs(ox - pill.cx) + pill.hw, Math.abs(oy - pill.cy) + pill.hh) + 4;
    pillHole = { x: ox, y: oy, R: 0, v: max / 0.085, max };
    pending = [];
    const inside = () => {
      for (let t = 0; t < 30; t++){
        const x = rand(-pill.hw, pill.hw), y = rand(-pill.hh, pill.hh);
        if (Math.hypot(Math.max(Math.abs(x) - L, 0), y) < pill.hh * 0.95) return [pill.cx + x, pill.cy + y];
      }
      return [pill.cx, pill.cy];
    };
    const slow = reduceMotion.matches ? 0.4 : 1;
    // the retracting rim sheds its film as mist, thrown the way the rim is travelling
    for (let i = 0; i < 90; i++){
      const [x, y] = inside();
      const ex = x - ox, ey = y - oy, el = Math.hypot(ex, ey) || 1;
      const sp = rand(40, 240) * k * slow;
      const a = Math.atan2(ey, ex) + rand(-0.5, 0.5);
      mistAt(x, y, Math.cos(a) * sp, Math.sin(a) * sp - rand(0, 40) * k, el / pillHole.v);
    }
    // the film becomes the small bubbles: about a fifth of the pill's area goes into them. Their sizes grow
    // only with the square root of the button's height, so a bigger button throws more bubbles, not just
    // bigger ones.
    const rb = bubbleBase();
    let budget = 0.22 * (4 * L * pill.hh + Math.PI * pill.hh * pill.hh);
    for (let i = 0; i < MAXPOP && (i < 2 || budget > 0); i++){
      const [x, y] = inside();
      const ex = x - ox, ey = y - oy, el = Math.hypot(ex, ey) || 1;
      const sp = rand(25, 80) * k * slow;
      const r = rb * (i === 0 ? rand(0.5, 0.62) : i < 3 ? rand(0.32, 0.45) : rand(0.16, 0.4));
      budget -= Math.PI * r * r;
      pending.push({ t: el / pillHole.v + 0.01, kind: 'bubble', x, y, r, vx: ex / el * sp, vy: ey / el * sp - rand(15, 45) * k * slow });
    }
    pending.sort((a, b) => a.t - b.t);
    lastBurst = simT;
    wake();
  }

  function popBubble(b, tx, ty){
    if (b.hole) return;
    const dx = tx - b.x, dy = ty - b.y, dl = Math.hypot(dx, dy);
    const f = dl > b.r * 0.9 ? b.r * 0.9 / dl : 1;
    const hx = b.x + dx * f, hy = b.y + dy * f;
    b.hole = { x: hx, y: hy, R: 0, max: b.r * 2 + 4, v: (b.r * 2 + 4) / 0.05 };
    const k = kk(), n = Math.round(6 + b.r * 0.6);
    for (let i = 0; i < n; i++){
      const a = rand(0, TAU), rr = Math.sqrt(Math.random()) * b.r;
      const x = b.x + Math.cos(a) * rr, y = b.y + Math.sin(a) * rr;
      const ex = x - hx, ey = y - hy, el = Math.hypot(ex, ey) || 1;
      const sp = rand(30, 160) * k;
      mist.push({ x, y, vx: ex / el * sp + b.vx, vy: ey / el * sp + b.vy, r: rand(0.5, 1.3), life: rand(0.4, 0.9), age: -el / b.hole.v });
    }
    sound.pop(Math.min(1, b.r / bubbleBase()) * 0.8);
    lastBurst = simT;
  }

  function cancelReform(){
    if (phase !== 'reform') return;
    phase = 'scatter'; phaseT = 0;
    if (pill.scale > 0.05){
      const { r } = pillShape();
      bubbles.push(newBubble(pill.cx, pill.cy, Math.max(r, 4), 0, 0));
    }
    pill.scale = 0; pill.scaleV = 0; absorbed = 0;
    for (const b of bubbles) b.absorbing = false;
  }
  function newBubble(x, y, r, vx, vy){
    return { x, y, r, r0: r, vx, vy, seed: Math.random(), hole: null, held: false, grab: null, absorbing: false, delay: 0, lift: 8, wob: 0, wobT: 0 };
  }

  function startReform(){
    phase = 'reform'; phaseT = 0;
    absorbed = 0; total = Math.max(1, bubbles.length);
    for (const b of bubbles){ b.delay = rand(0, 0.35); b.absorbing = false; }
    pill.scale = 0; pill.scaleV = 0;
    if (!bubbles.length) startBlow();
  }
  function startBlow(){
    phase = 'blow'; phaseT = 0;
    if (pill.scale < 0.12) pill.scale = 0.12;
    sound.blow();
  }

  function spring(x, v, target, k, c, dt){
    const a = -k * (x - target) - c * v;
    v += a * dt; x += v * dt;
    return [x, v];
  }

  function step(dt){
    phaseT += dt; simT += dt;
    const k = kk();
    air.target *= Math.exp(-dt * 0.35);
    {
      const w = 1.7;
      air.angV += (w * w * (air.target - air.ang) - 2 * w * air.angV) * dt;
      air.ang += air.angV * dt;
      const e = 1 - Math.exp(-dt * 2.5);
      air.x += (air.tx - air.x) * e; air.y += (air.ty - air.y) * e;
    }
    [pill.sx, pill.sxV] = spring(pill.sx, pill.sxV, 1 + 0.06 * pill.press, 260, 9, dt);
    [pill.sy, pill.syV] = spring(pill.sy, pill.syV, 1 - 0.07 * pill.press, 260, 9, dt);
    [pill.press, pill.pressV] = spring(pill.press, pill.pressV, pill.pressT, 300, 26, dt);
    for (const m of pill.modes){ m.t += dt; m.amp *= Math.exp(-dt * 2.2); }

    if (phase === 'pop'){
      pillHole.R += pillHole.v * dt;
      while (pending.length && pending[0].t <= phaseT){
        const e = pending.shift();
        if (e.kind === 'mist') mist.push({ x: e.x, y: e.y, vx: e.vx, vy: e.vy, r: e.r, life: e.life, age: 0 });
        else if (bubbles.length < MAXB){ const b = newBubble(e.x, e.y, e.r, e.vx, e.vy); b.lift = 2; b.wob = 0.12; bubbles.push(b); }
      }
      if (pillHole.R >= pillHole.max && !pending.length){
        phase = 'scatter'; phaseT = 0; pill.scale = 0; pillHole = null; lastBurst = simT;
      }
    } else if (phase === 'blow'){
      [pill.scale, pill.scaleV] = spring(pill.scale, pill.scaleV, 1, 70, 6.5, dt);
      // whole again: it is a button from here on, while the blow-up spring is still ringing
      if (pill.scale >= 0.98){ phase = 'idle'; phaseT = 0; settling = true; enablePointer(); }
    } else if (phase === 'idle' && settling){
      [pill.scale, pill.scaleV] = spring(pill.scale, pill.scaleV, 1, 70, 6.5, dt);
      if (Math.abs(pill.scale - 1) < 0.006 && Math.abs(pill.scaleV) < 0.04){
        pill.scale = 1; pill.scaleV = 0; settling = false;
        kickMode(0.8 * k, 3);
      }
    }
    pill.label = phase === 'idle' || phase === 'pop' ? 1 : phase === 'blow' ? Math.max(0, Math.min(1, (pill.scale - 0.75) / 0.25)) : 0;

    // small bubbles: nearly weightless, carried by the air
    for (const b of bubbles){
      b.wobT += dt; b.wob *= Math.exp(-dt * 2.5);
      b.lift += ((b.held ? 26 : 14) - b.lift) * (1 - Math.exp(-dt * 3));
      if (b.hole){
        b.hole.R += b.hole.v * dt;
        if (b.hole.R >= b.hole.max) b.dead = true;
        b.x += b.vx * dt; b.y += b.vy * dt;
        continue;
      }
      if (b.held && b.grab){
        const w = 14;
        b.vx += (w * w * (b.grab.x - b.x) - 1.6 * w * b.vx) * dt;
        b.vy += (w * w * (b.grab.y - b.y) - 1.6 * w * b.vy) * dt;
      } else if (phase === 'reform' && phaseT > b.delay){
        const lt = phaseT - b.delay;
        const w = Math.min(1.2 + lt * 2.2, 5.5);
        b.vx += (w * w * (pill.cx - b.x) - 1.5 * w * b.vx) * dt;
        b.vy += (w * w * (pill.cy - b.y) - 1.5 * w * b.vy) * dt;
      } else {
        b.vy -= 7 * k * dt;
        b.vx += Math.sin(simT * 0.8 + b.seed * 40) * 9 * k * dt;
        b.vy += Math.cos(simT * 0.6 + b.seed * 23) * 5 * k * dt;
        const f = Math.exp(-dt * 1.3);
        b.vx *= f; b.vy *= f;
      }
      b.x += b.vx * dt; b.y += b.vy * dt;
      // stage edges push back softly; bubbles never leave the canvas
      const x0 = -G.offX + b.r + 2, x1 = G.stageW - G.offX - b.r - 2, y0 = -G.offY + b.r + 2, y1 = G.stageH - G.offY - b.r - 2;
      if (b.x < x0){ b.x = x0; b.vx = Math.abs(b.vx) * 0.4; }
      if (b.x > x1){ b.x = x1; b.vx = -Math.abs(b.vx) * 0.4; }
      if (b.y < y0){ b.y = y0; b.vy = Math.abs(b.vy) * 0.4; }
      if (b.y > y1){ b.y = y1; b.vy = -Math.abs(b.vy) * 0.4; }
      if (phase === 'reform' && !b.absorbing && phaseT > b.delay && Math.hypot(b.x - pill.cx, b.y - pill.cy) < Math.max(8, pill.hh * 0.6)){
        b.absorbing = true; sound.merge();
      }
      if (b.absorbing){
        b.r *= Math.exp(-dt * 14);
        b.x += (pill.cx - b.x) * (1 - Math.exp(-dt * 12)); b.y += (pill.cy - b.y) * (1 - Math.exp(-dt * 12));
        if (b.r < 1.2){ b.dead = true; absorbed++; }
      }
    }
    // bubbles bump off each other instead of overlapping (while regrouping they are allowed to merge)
    for (let i = 0; i < bubbles.length && phase !== 'reform'; i++){
      const a = bubbles[i];
      if (a.hole || a.absorbing) continue;
      for (let j = i + 1; j < bubbles.length; j++){
        const b = bubbles[j];
        if (b.hole || b.absorbing) continue;
        const dx = b.x - a.x, dy = b.y - a.y, dl = Math.hypot(dx, dy) || 1, ov = a.r + b.r + 1 - dl;
        if (ov <= 0) continue;
        const nx = dx / dl, ny = dy / dl;
        const wa = a.held ? 0 : b.held ? 1 : b.r * b.r / (a.r * a.r + b.r * b.r), wb = 1 - wa;
        a.x -= nx * ov * wa; a.y -= ny * ov * wa; b.x += nx * ov * wb; b.y += ny * ov * wb;
        const rv = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny;
        if (rv < 0){ a.vx += nx * rv * wa; a.vy += ny * rv * wa; b.vx -= nx * rv * wb; b.vy -= ny * rv * wb; if (-rv > 30){ a.wob = b.wob = 0.08; a.wobT = b.wobT = 0; } }
      }
    }
    if (bubbles.some(b => b.dead)){
      for (const g of grabs.values()) if (g.bubble && g.bubble.dead) g.bubble = null;
      bubbles = bubbles.filter(b => !b.dead);
    }

    // mist: tiny and light, air drag wins quickly and they evaporate in flight
    for (const m of mist){
      m.age += dt;
      if (m.age < 0) continue;
      const f = Math.exp(-dt * 3.2);
      m.vx *= f; m.vy = m.vy * f + 140 * k * dt;
      m.x += m.vx * dt; m.y += m.vy * dt;
    }
    if (mist.length) mist = mist.filter(m => m.age < m.life);

    if (phase === 'scatter'){
      if (grabs.size || bubbles.some(b => b.hole)) lastBurst = simT;
      if (simT - lastBurst > REGROUP_AFTER) startReform();
    } else if (phase === 'reform'){
      const target = 0.14 + 0.28 * Math.min(absorbed / total, 1);
      [pill.scale, pill.scaleV] = spring(pill.scale, pill.scaleV, absorbed ? target : 0, 160, 14, dt);
      if (!bubbles.length) startBlow();
    }
  }

  // ---- input
  let pressed = null, suppressUntil = 0, popClickUntil = 0, swallowUntil = 0;
  const local = (cx, cy) => { const r = button.getBoundingClientRect(); return [cx - r.left, cy - r.top]; };
  const inStage = (x, y) => x >= -G.offX && y >= -G.offY && x <= G.stageW - G.offX && y <= G.stageH - G.offY;
  const grabs = new Map();

  function onBtnDown(e){
    if (lost || e.button !== 0 || !e.isPrimary || pressed || phase !== 'idle' || button.disabled) return;
    audio();
    const [x, y] = local(e.clientX, e.clientY);
    pressed = { id: e.pointerId, x0: x, y0: y, dragging: false };
    try { button.setPointerCapture(e.pointerId); } catch {}
    pill.pressT = 1;
    wake();
  }
  function onBtnMove(e){
    if (!pressed || e.pointerId !== pressed.id) return;
    const [x, y] = local(e.clientX, e.clientY);
    if (!pressed.dragging && Math.hypot(x - pressed.x0, y - pressed.y0) > DRAG_START) pressed.dragging = true;
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
    } else { pill.syV += 0.6; pill.sxV -= 0.5; kickMode(0.6 * kk(), 2); }
    wake();
  }
  const onBtnCancel = e => onBtnUp(e, true);
  function onClick(e){
    const now = performance.now();
    if (e.detail > 0 && now < suppressUntil){ suppressUntil = 0; e.preventDefault(); e.stopImmediatePropagation(); return; }
    if (lost || !hidden) return;
    if (now < popClickUntil){ popClickUntil = 0; disablePointer(); return; }
    audio();
    if (phase === 'idle'){ pop(); disablePointer(); }
    else if (e.detail === 0){ const b = bubbles.find(b => !b.hole && !b.absorbing); if (b) popBubble(b, b.x, b.y - b.r); }
    else { e.preventDefault(); e.stopImmediatePropagation(); }
    wake();
  }
  const isActivateKey = e => e.key === ' ' || e.key === 'Enter';
  function onKeyDown(e){ if (isActivateKey(e) && !e.repeat && phase === 'idle' && !lost){ pill.pressT = 1; wake(); } }
  function onKeyUp(e){ if (isActivateKey(e) && !pressed){ pill.pressT = 0; wake(); } }
  function onBlur(){ if (!pressed){ pill.pressT = 0; wake(); } }

  const root = document.documentElement;
  const baseCursor = root.style.cursor;
  let cursor = null;
  function setCursor(c){ if (c === cursor) return; cursor = c; root.style.cursor = c || baseCursor; }
  function pickBubble(x, y){
    if (phase !== 'scatter' && phase !== 'reform') return null;
    let best = null, bd = Infinity;
    for (const b of bubbles){
      if (b.hole || b.absorbing) continue;
      const g = Math.hypot(b.x - x, b.y - y) - b.r;
      if (g < Math.max(5, 12 - b.r * 0.3) && g < bd){ bd = g; best = b; }
    }
    return best;
  }
  function docDown(e){
    if (lost || e.button !== 0 || (phase !== 'scatter' && phase !== 'reform')) return;
    const [x, y] = local(e.clientX, e.clientY);
    if (!inStage(x, y)) return;
    const b = pickBubble(x, y);
    if (!b) return;
    audio();
    if (phase === 'reform') cancelReform();
    const g = { bubble: b, x, y, x0: x, y0: y, dragging: false };
    b.grab = g;
    grabs.set(e.pointerId, g);
    try { root.setPointerCapture(e.pointerId); } catch {}
    lastBurst = simT;
    e.preventDefault(); e.stopPropagation();
    wake();
  }
  function docMove(e){
    const [x, y] = local(e.clientX, e.clientY);
    const g = grabs.get(e.pointerId);
    if (g){
      g.x = x; g.y = y;
      if (!g.dragging && Math.hypot(x - g.x0, y - g.y0) > DRAG_START){ g.dragging = true; if (g.bubble) g.bubble.held = true; }
      if (g.dragging) setCursor('grabbing');
      wake();
      return;
    }
    if (!inStage(x, y)){ air.lx = NaN; setCursor(null); return; }
    stir(x, y);
    if (!grabs.size) setCursor(pickBubble(x, y) ? 'grab' : null);
  }
  // Moving air: the cursor slowly drags the film round with it. Floating bubbles are left alone.
  function stir(x, y){
    const now = performance.now();
    if (!Number.isNaN(air.lx)){
      const dt = Math.max((now - air.lt) / 1000, 1 / 240);
      const mx = x - air.lx, my = y - air.ly, ml = Math.hypot(mx, my);
      if (ml > 0.5 && !reduceMotion.matches){
        const rx = x - pill.cx, ry = y - pill.cy;
        const near = Math.exp(-Math.max(pillSdf(x, y), 0) / 60);
        const spin = Math.sign(rx * my - ry * mx) || 1;
        if (phase === 'idle' || phase === 'blow'){
          air.target = Math.max(-2.2, Math.min(2.2, air.target + spin * Math.min(ml, 25) * 0.01 * near));
          if (ml / dt > 700 && near > 0.4 && Math.random() < 0.25) kickMode(0.35 * kk() * near, 2 + Math.floor(Math.random() * 3));
        } else air.target = Math.max(-2.2, Math.min(2.2, air.target + spin * Math.min(ml, 25) * 0.006));
        wake();
      }
    }
    if (Number.isNaN(air.lx) && Math.abs(air.ang) < 0.02 && Math.abs(air.target) < 0.02){ air.x = x; air.y = y; }
    air.tx = x; air.ty = y; air.lx = x; air.ly = y; air.lt = now;
  }
  function releaseGrab(id, x, y, cancel){
    const g = grabs.get(id);
    if (!g) return;
    grabs.delete(id);
    lastBurst = simT;
    if (!grabs.size) setCursor(null);
    const b = g.bubble;
    if (!b) return;
    b.held = false; b.grab = null;
    if (!g.dragging && !cancel){ popBubble(b, x, y); return; }
    const sp = Math.hypot(b.vx, b.vy), max = 500;
    if (sp > max){ b.vx *= max / sp; b.vy *= max / sp; }
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
  function docClick(e){ if (e.detail > 0 && performance.now() < swallowUntil){ swallowUntil = 0; e.preventDefault(); e.stopImmediatePropagation(); } }
  function docTouch(e){ if (grabs.size && e.cancelable) e.preventDefault(); }
  function onContextMenu(){ releaseAllGrabs(); if (pressed) onBtnUp({ pointerId: pressed.id, clientX: 0, clientY: 0 }, true); }
  function onWinBlur(){ onContextMenu(); onBlur(); }

  const cap = { capture: true }, capNP = { capture: true, passive: false };
  const listeners = [
    [button, 'pointerdown', onBtnDown], [button, 'pointermove', onBtnMove], [button, 'pointerup', onBtnUp],
    [button, 'pointercancel', onBtnCancel], [button, 'lostpointercapture', onBtnCancel], [button, 'click', onClick, true],
    [button, 'keydown', onKeyDown], [button, 'keyup', onKeyUp], [button, 'blur', onBlur],
    [document, 'pointerdown', docDown, cap], [document, 'pointermove', docMove, cap], [document, 'pointerup', docUp, cap],
    [document, 'pointercancel', docCancel, cap], [document, 'lostpointercapture', docCancel, cap], [document, 'click', docClick, cap],
    [document, 'touchstart', docTouch, capNP], [document, 'touchmove', docTouch, capNP], [document, 'contextmenu', onContextMenu, cap],
    [window, 'blur', onWinBlur],
  ];
  for (const [t, ev, fn, o] of listeners) t.addEventListener(ev, fn, o);

  // ---- render
  function render(time){
    const toX = x => (x + G.offX) * dpr, toY = y => (G.stageH - (y + G.offY)) * dprY;
    const u = P.u;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, W, H);
    gl.useProgram(P.p);
    gl.bindVertexArray(vao);
    const showPill = phase === 'idle' || phase === 'pop' || ((phase === 'reform' || phase === 'blow') && pill.scale > 0.02);
    const { r, hw } = pillShape();
    const breathe = phase === 'idle' && !reduceMotion.matches ? Math.sin(time * 1.9) * 0.012 : 0;
    gl.uniform4f(u.uPill, toX(pill.cx), toY(pill.cy), hw * (1 + breathe * 0.5) * dpr, showPill ? r * (1 - breathe) * dpr : 0);
    pill.modes.forEach((m, i) => {
      modeBuf[i * 4] = m.amp * Math.sin(m.w * m.t) * dpr;
      modeBuf[i * 4 + 1] = m.m; modeBuf[i * 4 + 2] = m.phi; modeBuf[i * 4 + 3] = 0;
    });
    // an idle bubble is never perfectly still: a slow travelling wave on the rim
    if (phase === 'idle' && !reduceMotion.matches) modeBuf[0] += Math.sin(time * 1.3) * 0.35 * dpr * kk();
    gl.uniform4fv(u.uModes, modeBuf);
    if (pillHole) gl.uniform4f(u.uHole, toX(pillHole.x), toY(pillHole.y), pillHole.R * dpr, 3.2 * dpr);
    else gl.uniform4f(u.uHole, 0, 0, -1, 1);
    gl.uniform3f(u.uPillMisc, pill.label, seed, pillHole ? Math.max(0, 1 - pillHole.R / pillHole.max * 1.5) : Math.min(pill.scale, 1));
    gl.uniform4f(u.uAir, toX(air.x), toY(air.y), air.ang, 70 * dpr);
    const nb = Math.min(bubbles.length, MAXB);
    for (let i = 0; i < nb; i++){
      const b = bubbles[i];
      const wob = b.wob * Math.sin(b.wobT * 22);
      bBuf[i * 4] = toX(b.x); bBuf[i * 4 + 1] = toY(b.y); bBuf[i * 4 + 2] = b.r * (1 + wob * 0.3) * dpr; bBuf[i * 4 + 3] = b.seed;
      if (b.hole){ bhBuf[i * 4] = toX(b.hole.x); bhBuf[i * 4 + 1] = toY(b.hole.y); bhBuf[i * 4 + 2] = b.hole.R * dpr; }
      else { bhBuf[i * 4 + 2] = -1; }
      bhBuf[i * 4 + 3] = b.lift * dpr;
    }
    gl.uniform1i(u.uNB, nb);
    gl.uniform4fv(u.uB, bBuf); gl.uniform4fv(u.uBH, bhBuf);
    let nm = 0;
    for (const m of mist){
      if (m.age < 0) continue;
      if (nm >= MAXM) break;
      const fade = 1 - m.age / m.life;
      mBuf[nm * 4] = toX(m.x); mBuf[nm * 4 + 1] = toY(m.y); mBuf[nm * 4 + 2] = m.r * Math.sqrt(Math.max(fade, 0)) * dpr; mBuf[nm * 4 + 3] = Math.min(1, fade * 2.5);
      nm++;
    }
    gl.uniform1i(u.uNM, nm);
    gl.uniform4fv(u.uM, mBuf);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, labelTex); gl.uniform1i(u.uLabel, 0);
    gl.uniform4f(u.uLabelRect, toX(pill.cx), toY(pill.cy), (pill.hw + 4) * dpr, (pill.hh + 4) * dprY);
    gl.uniform3f(u.uBg, bgCol[0], bgCol[1], bgCol[2]);
    gl.uniform2f(u.uRes, W, H);
    gl.uniform1f(u.uDpr, dpr);
    gl.uniform1f(u.uTime, reduceMotion.matches ? 0 : time);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  // ---- loop
  let raf = 0, inFrame = false, frameCount = 0, last = performance.now(), dirty = true, manual = false, lost = false, visible = true;
  let lastActive = performance.now();
  const t0 = performance.now();
  function busy(){
    return phase !== 'idle' || settling || !!pressed || grabs.size > 0 || mist.length > 0 || pill.pressT > 0 || Math.abs(pill.press) > 1e-3 ||
      Math.abs(pill.sx - 1) + Math.abs(pill.sy - 1) + Math.abs(pill.sxV) + Math.abs(pill.syV) > 1e-3 ||
      pill.modes.some(m => m.amp > 0.01) || Math.abs(air.ang) + Math.abs(air.target) + Math.abs(air.angV) > 0.02;
  }
  function wake(){
    lastActive = performance.now();
    // one animation loop at most: a wake from inside a frame must not start a second loop
    if (!raf && !inFrame && !manual && !lost) raf = requestAnimationFrame(frame);
  }
  function advance(dt, time){
    const sub = Math.max(1, Math.ceil(dt / (1 / 240)));
    for (let i = 0; i < sub; i++) step(dt / sub);
    if (!lost && !gl.isContextLost() && W && (visible || dirty)){ render(time); dirty = false; }
  }
  function frame(now){
    raf = 0;
    if (manual || lost) return;
    const dt = Math.min((now - last) / 1000, 1 / 30);
    last = now;
    place();
    if (busy()) lastActive = now;
    inFrame = true;
    frameCount++;
    try { advance(dt, (now - t0) / 1000); } finally { inFrame = false; }
    // nothing moving and nobody near it for a while: the last frame stays on screen
    if (visible && now - lastActive < SLEEP_MS) raf = requestAnimationFrame(frame);
  }

  const relayout = () => { if (lost) return; measureBounds(); layout(); };
  const ro = new ResizeObserver(relayout);
  ro.observe(button);
  if (document.body) ro.observe(document.body);
  const onScroll = () => { if (!lost) place(); };
  addEventListener('resize', relayout);
  document.addEventListener('scroll', onScroll, { capture: true, passive: true });
  const io = new IntersectionObserver(es => { for (const e of es) visible = e.isIntersecting; if (visible){ dirty = true; wake(); } });
  io.observe(canvas);
  const onFonts = () => { if (!lost && labelText()){ buildLabel(); wake(); } };
  document.fonts?.addEventListener?.('loadingdone', onFonts);
  const mo = new MutationObserver(() => {
    if (lost) return;
    const cs = getComputedStyle(button);
    if (labelText() + '|' + cs.font + '|' + cs.color !== labelKey){ buildLabel(); wake(); }
  });
  mo.observe(button, { childList: true, characterData: true, subtree: true, attributes: true, attributeFilter: ['class'] });

  function onLost(e){
    e.preventDefault();
    lost = true;
    cancelAnimationFrame(raf); raf = 0;
    releaseAllGrabs(); pressed = null;
    phase = 'idle'; bubbles = []; mist = []; pending = []; pillHole = null; pill.scale = 1; pill.scaleV = 0; settling = false; pill.pressT = pill.press = 0;
    enablePointer(); setCursor(null);
    showButton();
    canvas.style.visibility = 'hidden';
  }
  function onRestored(){
    try { initGL(); } catch (err) { console.warn('ElementalFX bubble: could not rebuild after context loss', err); return; }
    lost = false;
    hideButton();
    canvas.style.visibility = '';
    G.stageW = G.stageH = 0; G.bw = G.bh = 0;
    relayout();
    wake();
  }
  canvas.addEventListener('webglcontextlost', onLost);
  canvas.addEventListener('webglcontextrestored', onRestored);

  hideButton();
  document.body.appendChild(canvas);
  relayout();
  advance(0, 0);

  return {
    supported: true,
    get state(){ return lost ? 'lost' : phase; },
    refresh(){ if (!lost){ relayout(); buildLabel(); dirty = true; wake(); } },
    detach(){
      cancelAnimationFrame(raf); raf = 0; manual = true;
      clearTimeout(peTimer);
      ro.disconnect(); io.disconnect(); mo.disconnect();
      document.fonts?.removeEventListener?.('loadingdone', onFonts);
      removeEventListener('resize', relayout);
      document.removeEventListener('scroll', onScroll, { capture: true });
      for (const [t, ev, fn, o] of listeners) t.removeEventListener(ev, fn, o);
      canvas.removeEventListener('webglcontextlost', onLost); canvas.removeEventListener('webglcontextrestored', onRestored);
      for (const id of grabs.keys()){ try { root.releasePointerCapture(id); } catch {} }
      grabs.clear();
      setCursor(null);
      enablePointer();
      showButton();
      canvas.remove();
      gl.getExtension('WEBGL_lose_context')?.loseContext();
    },
    // test hooks
    get _frames(){ return frameCount; },
    _manual(on){ manual = on; last = performance.now(); if (!on) wake(); else { cancelAnimationFrame(raf); raf = 0; } },
    _advance(sec, fps = 60){ place(); for (let i = 0, n = Math.round(sec * fps); i < n; i++) advance(1 / fps, (performance.now() - t0) / 1000); },
    get _sim(){ return { bubbles, mist, pill, G, pop, get phase(){ return phase; }, get grabs(){ return grabs.size; } }; },
  };
}

window.ElementalFX = Object.assign(window.ElementalFX || {}, { attachBubble, bubbleAudio });
})();
