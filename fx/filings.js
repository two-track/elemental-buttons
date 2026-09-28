// Iron filings button: a thin layer of magnetic filings lies on the face of an existing <button>, combed
// into the field lines of a bar magnet hidden under the label.
//
//   <script src="fx/filings.js"></script>
//   const fx = ElementalFX.attachFilings(document.querySelector('#buy'));   // fx.detach() restores it
//
// A classic script (not a module) so pages work straight from file:// too.
//
// The <button> stays the real control (focus, keyboard, form submission, click handlers) and keeps its
// own face and label; this only draws the filings, in a transparent canvas laid over the page around it
// that never takes pointer input. The cursor is a magnet held above the card: filings near it swing round
// into its field, stand up and creep toward it, and swing back into the hidden magnet's pattern once it
// has passed. Holding the button down brings the magnet close, and the filings leap into bristling tufts.
// A click releases them: they spray off the card across the page and settle, and a moment later the
// magnet sweeps them back into place along its field lines. The label is a window the filings never
// cover while they lie on the card. If WebGL2 is unavailable, the button is left untouched.
//
// Options:
//   color: 'auto' (default: iron on a pale button, steel on a dark one) | 'iron' | 'copper' | 'gold' | 'steel'
//          | { base: CSS colour, spec: CSS colour }
//   density: filings per unit area, relative to the default (default 1; 0.3 to 2.5).
//   restoreAfter: seconds of quiet after the filings land before they are swept back (default 0.35).
//   margin: CSS px of page around the button the filings may land on (default max(64, 1.25 x height)).
//   zIndex: of the stage canvas (default 10).   sound: false to stay silent.
//
// fx.setColor(c) and fx.setDensity(d) change it live; fx.restore() sweeps the filings home now.

(() => {
'use strict';

const TAU = Math.PI * 2;
const rand = (a, b) => a + Math.random() * (b - a);
const clamp = (v, a, b) => v < a ? a : v > b ? b : v;
const smooth = (a, b, v) => { const t = clamp((v - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const wrapHalf = a => a - Math.PI * Math.round(a / Math.PI);
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');

// ---------------------------------------------------------------- tuning (CSS px, seconds)
const H_HOVER = 20;          // magnet height above the card while hovering
const H_PRESS = 8;           // ... while the button is held down
const Q_HOVER = 1.5;         // pole strength while hovering
const Q_PRESS = 1.9;         // ... when pressed
const PARTNER = 55;          // the magnet's far pole sits this far above the near one
const ROT_K = 150;           // rotational mobility of a filing on card
const ROT_MIN = 0.006;       // squared in-plane field below which a filing won't turn (static friction)
const PULL_K = 40;           // slide speed per unit of field-gradient force
const MU_S = 0.045;          // static friction, in field-gradient units: a hovering magnet turns filings, a pressed one drags
const OBLIQUE = 0.55;        // screen px a point rises per px of height (the page is seen slightly from below)
const SHADOW = [0.5, 0.72];  // shadow offset per px of height (light from the upper left)
const GRAV = 1100;           // px/s^2 for airborne filings
const RELAX_DELAY = 0.28;    // quiet time before a disturbed filing is drawn back into the pattern
const LABEL_PAD = [9, 5];    // clear margin round the label, x and y
const SPOKES = 16, SPOKE_W = TAU / SPOKES;
const TUFT_R = 25;           // filings this close to a pressed magnet leap into its tuft
const TUFT_IN = 12;          // the tuft stays at least this far inside the card's rim
const SETTLE_N = 10;         // frames a filing must hold still before it is left alone
const SETTLE_D = 0.6, SETTLE_A = 0.06;
const FREEZE_T = 0.9;        // a magnet held still this long: whatever still twitches is done

const COLORS = {
  iron:   { base: [0.16, 0.16, 0.175], spec: [0.9, 0.92, 0.96], shadow: 0.34 },
  copper: { base: [0.44, 0.19, 0.08], spec: [1.0, 0.74, 0.52], shadow: 0.3 },
  gold:   { base: [0.5, 0.36, 0.1], spec: [1.0, 0.9, 0.58], shadow: 0.3 },
  steel:  { base: [0.64, 0.66, 0.7], spec: [1.0, 1.0, 1.0], shadow: 0.55 },
};

// ---------------------------------------------------------------- shaders
// One instanced capsule per filing, drawn twice: its shadow on the card, then the needle itself.
const VS = `#version 300 es
layout(location=0) in vec2 aQ;
layout(location=1) in vec4 aA;   // x y z angle
layout(location=2) in vec4 aB;   // tilt length alpha seed
uniform vec2 uStage;
uniform float uDpr, uShadow, uOblique;
uniform vec2 uSh;
out vec2 vST;
flat out float vL, vHW, vAlpha, vSeed, vTilt, vZ;
flat out vec2 vU, vMid;
void main(){
  // a negative length marks a filing in the air, whose shadow falls away from it
  float th = aA.w, tl = aB.x, len = abs(aB.y);
  vec3 ax = vec3(cos(th) * cos(tl), sin(th) * cos(tl), sin(tl));
  vec3 p0 = aA.xyz - ax * len * 0.5, p1 = aA.xyz + ax * len * 0.5;
  float hw = 0.2 + 0.13 * fract(aB.w * 7.13);
  vec2 s0, s1;
  if (uShadow > 0.5){
    // a contact shadow: it stays under the filing however high it stands, so a tuft does not cast a blot
    float zc = aB.y < 0. ? 1e4 : min(aA.z, 2.5);
    s0 = p0.xy + uSh * min(p0.z, zc); s1 = p1.xy + uSh * min(p1.z, zc);
    hw += 0.12;
  } else {
    s0 = p0.xy - vec2(0., uOblique * p0.z); s1 = p1.xy - vec2(0., uOblique * p1.z);
    // seen from almost straight above, a standing needle would shrink to a dot; drawn at (a little more
    // than) its full length along its lean it reads as a bristle, which is what the eye expects of one
    vec2 dd = s1 - s0; float l0 = length(dd);
    vec2 dir = l0 > 1e-3 ? dd / l0 : vec2(cos(th), sin(th));
    vec2 mid = (s0 + s1) * 0.5;
    float lw = len * (1. + 0.35 * smoothstep(0.15, 1.1, tl));
    s0 = mid - dir * lw * 0.5; s1 = mid + dir * lw * 0.5;
    hw += 0.12 * smoothstep(0.2, 1.0, tl);
  }
  s0 *= uDpr; s1 *= uDpr; hw *= uDpr;
  vec2 d = s1 - s0; float L = length(d);
  vec2 u = L > 1e-3 ? d / L : vec2(cos(th), sin(th));
  vec2 n = vec2(-u.y, u.x);
  float pad = hw + 1.5;
  float s = mix(-pad, L + pad, aQ.x), t = aQ.y * pad;
  vec2 pos = s0 + u * s + n * t;
  vST = vec2(s, t); vL = L; vHW = hw; vAlpha = aB.z; vSeed = aB.w; vTilt = tl; vZ = aA.z; vU = u; vMid = (s0 + s1) * 0.5 / uDpr;
  vec2 clip = pos / (uStage * uDpr) * 2. - 1.;
  gl_Position = vec4(clip.x, -clip.y, 0., 1.);
}`;

const FS = `#version 300 es
precision highp float;
in vec2 vST;
flat in float vL, vHW, vAlpha, vSeed, vTilt, vZ;
flat in vec2 vU, vMid;
uniform float uShadow, uShadowA, uDpr, uDesat, uFlash;
uniform vec3 uBase, uSpec;
uniform vec3 uLamp;      // stage px of the cursor, and how much it lights
out vec4 o;
float h11(float p){ p = fract(p * .1031); p *= p + 33.33; p *= p + p; return fract(p); }
void main(){
  float s = vST.x, t = vST.y;
  float dx = max(max(-s, s - vL), 0.);
  float d = length(vec2(dx, t));
  if (uShadow > 0.5){
    float soft = vHW + 0.5 * uDpr;
    float a = smoothstep(soft, 0., d) * uShadowA * vAlpha * (0.35 + 0.65 * smoothstep(0., 1.5, vZ)) / (1. + 0.45 * vZ);
    o = vec4(0., 0., 0., a);
    return;
  }
  // thin needles: coverage by area once they are under a device pixel wide
  float hw = max(vHW, 0.55);
  float cov = clamp(hw - d + 0.5, 0., 1.) * min(1., vHW / 0.55);
  if (cov <= 0.) discard;
  // a cylinder across its width, rough along its length (filings are chips, not wire)
  float k = clamp(t / hw, -1., 1.);
  vec2 n2 = vec2(-vU.y, vU.x);
  vec3 N = normalize(vec3(n2 * k * 0.9, sqrt(max(0., 1. - k * k)) + 0.15));
  float seg = floor(s / (1.3 * uDpr) + vSeed * 50.);
  float rough = h11(seg + vSeed * 91.);
  N = normalize(N + vec3((rough - 0.5) * 0.5, (h11(seg * 1.7 + 3.) - 0.5) * 0.5, 0.));
  vec3 L1 = normalize(vec3(-0.45, -0.62, 0.64));
  vec3 V = vec3(0., 0., 1.);
  float diff = max(dot(N, L1), 0.);
  float sp = pow(max(dot(N, normalize(L1 + V)), 0.), 18.) * (0.35 + 0.65 * rough);
  // the cursor is a lamp as well as a magnet: facets flash as it passes
  vec3 L2 = normalize(vec3(uLamp.xy - vMid, 70.));
  float sp2 = pow(max(dot(N, normalize(L2 + V)), 0.), 40.) * uLamp.z * step(0.55, rough);
  // a standing filing shows its bright broken end, and its shaft darkens as it turns away from the light
  float stand = smoothstep(0.12, 0.7, vTilt);
  float tip = stand * smoothstep(vL - 1.0 * uDpr, vL - 0.25 * uDpr, s);
  vec3 col = uBase * (0.42 + 0.75 * diff) * (0.85 + 0.3 * h11(vSeed * 13.)) * (1. - 0.35 * stand) + uSpec * (sp * 0.55 + sp2 * 0.9) * (1. - 0.6 * stand);
  col = mix(col, uSpec, tip * 0.85);
  col = mix(col, vec3(dot(col, vec3(0.3, 0.55, 0.15))), uDesat);
  col = mix(col, uSpec, uFlash * 0.6);
  float a = cov * vAlpha * (1. - 0.45 * uFlash);
  o = vec4(col * a, a);
}`;

// ---------------------------------------------------------------- audio (shared)
let actx = null;
const filingsAudio = { muted: false };
function audio(){
  if (filingsAudio.muted) return null;
  // test hook: an OfflineAudioContext to render the sounds into
  if (filingsAudio._ctx) return filingsAudio._ctx;
  if (!actx){ try { actx = new (window.AudioContext || window.webkitAudioContext)(); } catch { return null; } }
  if (actx.state === 'suspended') actx.resume();
  return actx;
}
// Everything is synthesised at one fixed rate into plain arrays while the page is idle, and only mixed
// at event time: building ~1s of grains on the click frame stalled the first release by 150ms.
const SR = 44100;
// A noise-excited two-pole resonator: a struck grain of steel rings at its own pitch for a few ms.
function resonate(d, sr, at, f, decay, amp, excMs = 0.3){
  const start = Math.round(at * sr); if (start < 0 || start >= d.length || f >= sr * 0.45) return;
  const w = TAU * f / sr, r = Math.exp(-1 / (decay * sr)), a1 = 2 * r * Math.cos(w), a2 = -r * r;
  const n = Math.min(d.length - start, Math.ceil(decay * sr * 7)), exc = Math.max(1, Math.round(excMs * sr / 1000));
  let y1 = 0, y2 = 0;
  const g = amp * (1 - r) * 2.2;
  for (let i = 0; i < n; i++){
    const x = i < exc ? (Math.random() * 2 - 1) : 0;
    const y = x * g + a1 * y1 + a2 * y2; y2 = y1; y1 = y;
    d[start + i] += y;
  }
}
// One grain hitting card: a short noise tick, coloured by a one-pole low- and high-pass.
function grain(d, sr, at, ms, amp, lp, hp){
  const start = Math.round(at * sr), n = Math.min(d.length - start, Math.round(ms * sr / 1000));
  if (start < 0 || n <= 0) return;
  const al = 1 - Math.exp(-TAU * lp / sr), ah = Math.exp(-TAU * hp / sr);
  let l = 0, l2 = 0, hIn = 0, h = 0;
  for (let i = 0; i < n; i++){
    l += (Math.random() * 2 - 1 - l) * al; l2 += (l - l2) * al;
    h = ah * (h + l2 - hIn); hIn = l2;
    d[start + i] += h * Math.exp(-5 * i / n) * amp;
  }
}
function normalize(d, to){
  let peak = 0;
  for (let i = 0; i < d.length; i++) peak = Math.max(peak, Math.abs(d[i]));
  if (peak > 0) for (let i = 0; i < d.length; i++) d[i] *= to / peak;
  return d;
}
const tail = () => 0.15 + 1.4 * Math.pow(Math.random(), 3.2);
// A little clump of filings landing together: a few dry ticks inside ~6ms with the body of card under
// them, and now and then the thin ring of steel.
function clumpEvent(){
  const d = new Float32Array(Math.round(SR * 0.03));
  const k = 1 + Math.floor(Math.random() * 4);
  for (let j = 0; j < k; j++){
    const t = j ? rand(0.0005, 0.006) : 0.0003, amp = j ? rand(0.3, 0.8) : 1;
    grain(d, SR, t, rand(0.5, 1.4), amp, rand(2800, 7000), rand(600, 1400));
    resonate(d, SR, t, rand(1200, 3200), rand(0.0008, 0.002), amp * 0.6);
  }
  if (Math.random() < 0.5) resonate(d, SR, 0.0003, rand(4000, 7500), rand(0.002, 0.004), 0.2);
  return normalize(d, 1);
}
// the card itself taking the weight of the first of them
function thumpEvent(){
  const d = new Float32Array(Math.round(SR * 0.09));
  resonate(d, SR, 0, rand(300, 420), 0.007, 1, 1.2);
  resonate(d, SR, 0, rand(560, 800), 0.005, 0.6, 1.2);
  grain(d, SR, 0, 3, 0.4, 1800, 200);
  return normalize(d, 1);
}
let pool = null;
function buildPool(){
  if (pool) return pool;
  const ev = Array.from({ length: 28 }, clumpEvent), thumps = Array.from({ length: 3 }, thumpEvent);
  // the magnet letting go: a low knock and a short breath of air as the filings leave the card
  const knock = new Float32Array(Math.round(SR * 0.18));
  let ph = 0;
  const f0 = rand(150, 190);
  for (let i = 0; i < SR * 0.08; i++){ const t = i / SR; ph += TAU * f0 * (1 - 0.3 * t / 0.08) / SR; knock[i] += Math.sin(ph) * Math.min(1, t / 0.001) * Math.exp(-t / 0.018) * 0.55; }
  resonate(knock, SR, 0.0005, rand(900, 1150), 0.009, 0.5);
  resonate(knock, SR, 0.0005, rand(2400, 2900), 0.004, 0.3);
  let l = 0;
  for (let i = 0; i < SR * 0.14; i++){
    const t = i / SR, f = 900 + 3500 * (t / 0.14);
    l += (Math.random() * 2 - 1 - l) * (1 - Math.exp(-TAU * f / SR));
    knock[i] += l * Math.sin(Math.PI * Math.min(1, t / 0.14)) * 0.1;
  }
  normalize(knock, 1);
  // the magnet clacking down on the card
  const clack = new Float32Array(Math.round(SR * 0.05));
  resonate(clack, SR, 0.001, rand(1700, 2100), 0.004, 0.4);
  resonate(clack, SR, 0.001, rand(3600, 4300), 0.0025, 0.25);
  resonate(clack, SR, 0.001, rand(500, 700), 0.006, 0.3, 0.8);
  normalize(clack, 1);
  // filings sliding over card, band-limited so it sits under the ticks rather than hissing over them
  const hiss = new Float32Array(Math.round(SR * 1.2));
  let a = 0, b = 0, hIn = 0, h = 0;
  const al = 1 - Math.exp(-TAU * 4500 / SR), ah = Math.exp(-TAU * 1200 / SR);
  for (let i = 0; i < hiss.length; i++){ a += (Math.random() * 2 - 1 - a) * al; b += (a - b) * al; h = ah * (h + b - hIn); hIn = b; hiss[i] = h; }
  normalize(hiss, 1);
  // the rustle bed: sparse single ticks, so a moving magnet sounds like grains shifting, not static
  const rustleD = new Float32Array(Math.round(SR * 1.6));
  for (let t = 0; t < 1.58;){
    t += -Math.log(1 - Math.random()) / 140;
    grain(rustleD, SR, t, rand(0.3, 1.0), tail(), rand(1800, 4500), rand(500, 1200));
    if (Math.random() < 0.12) resonate(rustleD, SR, t, rand(2500, 5000), 0.0015, 0.3 * tail());
  }
  normalize(rustleD, 0.9);
  pool = { ev, thumps, knock, clack, hiss, rustle: rustleD };
  return pool;
}
// compose a sound out of pool pieces into one buffer: only additions, well under a millisecond
function mixIn(d, src, at, amp){
  const s0 = Math.round(at * SR);
  if (s0 < 0 || s0 >= d.length) return;
  const n = Math.min(src.length, d.length - s0);
  for (let i = 0; i < n; i++) d[s0 + i] += src[i] * amp;
}
function playArr(a, d, gain){
  const buf = a.createBuffer(1, d.length, SR);
  buf.copyToChannel(d, 0);
  const src = a.createBufferSource(); src.buffer = buf;
  const g = a.createGain(); g.gain.value = gain;
  src.connect(g).connect(a.destination); src.start();
}
const pick = arr => arr[(Math.random() * arr.length) | 0];
// Landings are drawn from the flight times the sim has already chosen, so what you hear is the scatter
// you are watching, but as a few dozen audible clumps: one grain per filing summed to a flat hiss.
function patter(d, times, offset, count, amp, P){
  if (!times.length) return;
  for (let k = 0; k < count; k++){
    const t = times[(Math.random() * times.length) | 0];
    const r = Math.random();
    mixIn(d, pick(P.ev), t + offset + rand(-0.004, 0.004), amp * (0.06 + 1.5 * Math.pow(r, 6)));
  }
}
const sounds = {
  press(){
    const a = audio(); if (!a) return;
    const P = buildPool(), d = new Float32Array(Math.round(SR * 0.2));
    mixIn(d, P.clack, 0, 1);
    let t = 0.006;
    while (t < 0.13){ t += -Math.log(1 - Math.random()) / (60 + 500 * (t / 0.13)); mixIn(d, pick(P.ev), t, 0.4 * tail() * (1 - t / 0.16)); }
    playArr(a, normalize(d, 0.8), 0.16);
  },
  release(lands, count){
    const a = audio(); if (!a) return;
    const P = buildPool();
    const end = lands.reduce((m, t) => Math.max(m, t), 0);
    const d = new Float32Array(Math.round(SR * (end + 0.35)));
    mixIn(d, P.knock, 0, 0.9);
    const first = lands.reduce((m, t) => Math.min(m, t), 9);
    mixIn(d, pick(P.thumps), first + 0.01, 0.5);
    mixIn(d, pick(P.thumps), first + 0.07 + rand(0, 0.04), 0.3);
    const n = Math.round(clamp(count / 100, 26, 48));
    patter(d, lands, 0.004, n, 0.5, P);
    // a few bounce off the card and land again
    patter(d, lands.map(t => t + rand(0.06, 0.2)), 0, Math.round(n * 0.25), 0.18, P);
    playArr(a, normalize(d, 0.9), 0.3 + 0.08 * Math.min(1, count / 5000));
  },
  // the sweep home: a slide that swells as more filings are caught, each clump arriving with a tick,
  // then a single dry snap as the pattern locks
  restore(arrivals, end){
    const a = audio(); if (!a) return;
    const P = buildPool(), d = new Float32Array(Math.round(SR * (end + 0.25)));
    const n = Math.min(Math.round(SR * end), d.length, P.hiss.length);
    for (let i = 0; i < n; i++) d[i] += P.hiss[i] * 0.07 * Math.pow(i / n, 1.6);
    patter(d, arrivals, 0, Math.round(clamp(arrivals.length / 50, 40, 80)), 0.38, P);
    resonate(d, SR, end, rand(2300, 2700), 0.006, 1.0);
    resonate(d, SR, end, rand(600, 800), 0.01, 0.8, 1);
    grain(d, SR, end, 1.2, 1.2, 6000, 700);
    playArr(a, normalize(d, 0.9), 0.2);
  },
};
// The rustle under a moving magnet is a looped bed of ticks whose level follows how much the filings
// are actually moving; a silent looping source would keep the audio thread busy, so it is torn down.
let rustle = null, rustleStop = 0;
function rustleLevel(lvl){
  const a = lvl > 0 ? audio() : (filingsAudio._ctx || actx); if (!a) return;
  if (!rustle){
    if (lvl <= 0.002) return;
    const d = buildPool().rustle, buf = a.createBuffer(1, d.length, SR);
    buf.copyToChannel(d, 0);
    const src = a.createBufferSource(); src.buffer = buf; src.loop = true;
    const g = a.createGain(); g.gain.value = 0;
    src.connect(g).connect(a.destination); src.start(0, Math.random() * 1.5);
    rustle = { src, g };
  }
  const v = filingsAudio.muted ? 0 : Math.min(1, lvl);
  rustle.g.gain.setTargetAtTime(0.16 * v * rand(0.7, 1.2), a.currentTime, 0.03);
  clearTimeout(rustleStop); rustleStop = 0;
  if (v <= 0.002){
    const r = rustle;
    rustleStop = setTimeout(() => {
      rustleStop = 0;
      if (rustle !== r) return;
      try { r.src.stop(); } catch { /* already stopped */ }
      r.src.disconnect(); rustle = null;
    }, 400);
  }
}
const idle = window.requestIdleCallback || (f => setTimeout(f, 200));
// Starting the audio device is a long synchronous stall the first time (a quarter second in some
// builds), so it is done while the pointer is merely hovering whenever the page already has the user
// activation that lets a context run; creating one without it would only log an autoplay warning.
let warming = false;
function prewarmAudio(){
  if (actx || warming || filingsAudio.muted || filingsAudio._ctx) return;
  const ua = navigator.userActivation;
  if (!ua || !ua.hasBeenActive) return;
  warming = true;
  idle(() => { warming = false; if (!actx && !filingsAudio.muted) audio(); }, { timeout: 400 });
}
const awakeFx = new Set();
let suspendTimer = 0;
function audioAwake(fx, on){
  if (on){ awakeFx.add(fx); clearTimeout(suspendTimer); suspendTimer = 0; return; }
  awakeFx.delete(fx);
  if (awakeFx.size || suspendTimer || !actx) return;
  suspendTimer = setTimeout(() => {
    suspendTimer = 0;
    if (awakeFx.size || !actx || actx.state !== 'running' || rustle) return;
    actx.suspend().catch(() => {});
  }, 2500);
}

// ---------------------------------------------------------------- page helpers
function parseColor(css){
  const m = /rgba?\(([^)]+)\)/.exec(css || '');
  if (!m) return null;
  const p = m[1].split(/[\s,/]+/).filter(Boolean).map(parseFloat);
  return [p[0] / 255, p[1] / 255, p[2] / 255, p.length > 3 ? p[3] : 1];
}
function cssColor(c){
  const x = document.createElement('canvas').getContext('2d');
  x.fillStyle = '#000'; x.fillStyle = c;
  const v = x.fillStyle;
  if (v[0] === '#') return [1, 3, 5].map(i => parseInt(v.slice(i, i + 2), 16) / 255);
  return (parseColor(v) || [0, 0, 0]).slice(0, 3);
}
// what colour the button face is, compositing its background over whatever lies behind it
function faceColor(el){
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
const lum = c => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
function inFixedContext(el){
  for (let e = el; e && e !== document.documentElement; e = e.parentElement){
    const p = getComputedStyle(e).position;
    if (p === 'fixed' || p === 'sticky') return true;
  }
  return false;
}
// signed distance to a rounded box centred at (cx, cy) with half-size (hx, hy) and corner radius r
function sdBox(x, y, cx, cy, hx, hy, r){
  const qx = Math.abs(x - cx) - (hx - r), qy = Math.abs(y - cy) - (hy - r);
  return Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0) - r;
}

// ---------------------------------------------------------------- component
function attachFilings(button, opts = {}){
  const noop = () => {};
  const unsupported = { supported: false, state: 'unsupported', detach: noop, restore: noop, setColor: noop, setDensity: noop };
  const canvas = document.createElement('canvas');
  canvas.setAttribute('aria-hidden', 'true');
  Object.assign(canvas.style, { position: 'absolute', left: '0px', top: '0px', margin: '0', padding: '0', border: '0', display: 'block', pointerEvents: 'none', zIndex: String(opts.zIndex ?? 10) });
  const gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: true, antialias: false, depth: false, stencil: false });
  if (!gl) return unsupported;
  const sound = opts.sound !== false;
  const self = {};
  const snd = (name, ...a) => { if (sound) sounds[name](...a); };
  const RESTORE_AFTER = opts.restoreAfter ?? 0.35;
  let density = clamp(opts.density ?? 1, 0.3, 2.5);
  let colorOpt = opts.color || 'auto';

  // ---- GL
  let prog, U, vao, instBuf, quadBuf;
  function compile(type, src){
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS) && !gl.isContextLost()) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }
  function initGL(){
    prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS) && !gl.isContextLost()) throw new Error(gl.getProgramInfoLog(prog));
    U = {};
    for (const n of ['uStage', 'uDpr', 'uShadow', 'uOblique', 'uSh', 'uShadowA', 'uBase', 'uSpec', 'uLamp', 'uDesat', 'uFlash']) U[n] = gl.getUniformLocation(prog, n);
    vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    quadBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, -1, 1, -1, 0, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    instBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, instBuf);
    for (let i = 0; i < 2; i++){ gl.enableVertexAttribArray(1 + i); gl.vertexAttribPointer(1 + i, 4, gl.FLOAT, false, 32, i * 16); gl.vertexAttribDivisor(1 + i, 1); }
    gl.bindVertexArray(null);
  }
  initGL();

  // ---- geometry: stage px, origin at the stage's top-left; the button face sits at (offX, offY)
  const G = { fixed: false, stageW: 0, stageH: 0, offX: 0, offY: 0, bw: 0, bh: 0, rad: 0, bx: NaN, by: NaN };
  const win = { on: false, cx: 0, cy: 0, hx: 0, hy: 0, r: 0 };
  const poles = [[0, 0], [0, 0]];
  let dpr = 1, cbOrigin = { x: 0, y: 0 }, bounds = { w: 0, h: 0 };
  let pal = COLORS.iron;

  function resolveColor(){
    if (typeof colorOpt === 'object' && colorOpt){
      const base = cssColor(colorOpt.base || '#2a2a2c'), spec = cssColor(colorOpt.spec || '#ffffff');
      pal = { base, spec, shadow: lum(faceColor(button)) > 0.45 ? 0.32 : 0.55 };
    } else if (COLORS[colorOpt]) pal = COLORS[colorOpt];
    else pal = lum(faceColor(button)) > 0.45 ? COLORS.iron : COLORS.steel;
  }

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

  // The label window: the text's own box, found with a Range so any padding or alignment is respected.
  function measureLabel(r){
    const rg = document.createRange();
    rg.selectNodeContents(button);
    const t = rg.getBoundingClientRect();
    if (!t.width || !t.height){ win.on = false; return; }
    win.on = true;
    win.cx = t.left + t.width / 2 - r.left + G.offX;
    win.cy = t.top + t.height / 2 - r.top + G.offY;
    win.hx = Math.min(t.width / 2 + LABEL_PAD[0], G.bw / 2 - 6);
    win.hy = Math.min(t.height / 2 + LABEL_PAD[1] - 2, G.bh / 2 - 5);
    win.r = Math.min(win.hy, 8);
  }

  function layout(r = button.getBoundingClientRect()){
    const bx = r.left + (G.fixed ? 0 : scrollX), by = r.top + (G.fixed ? 0 : scrollY);
    const bw = r.width, bh = r.height;
    if (bw < 16 || bh < 12){ canvas.style.visibility = 'hidden'; G.bx = bx; G.by = by; G.bw = bw; G.bh = bh; return; }
    canvas.style.visibility = lost ? 'hidden' : '';
    const M = opts.margin ?? Math.round(Math.max(64, 1.25 * bh));
    // the stage stays inside the document, so it can never add scrolling
    const x0 = Math.floor(Math.min(bx, Math.max(0, bx - M))), y0 = Math.floor(Math.min(by, Math.max(0, by - M)));
    const x1 = Math.ceil(Math.max(bx + bw, Math.min(bounds.w, bx + bw + M))), y1 = Math.ceil(Math.max(by + bh, Math.min(bounds.h, by + bh + M)));
    const sw = x1 - x0, sh = y1 - y0;
    const sizeChanged = sw !== G.stageW || sh !== G.stageH;
    const shapeChanged = Math.abs(bw - G.bw) > 0.5 || Math.abs(bh - G.bh) > 0.5;
    const mx = (bx - x0) - G.offX, my = (by - y0) - G.offY;
    Object.assign(G, { bx, by, bw, bh, stageW: sw, stageH: sh, offX: bx - x0, offY: by - y0 });
    Object.assign(canvas.style, { left: `${x0 - cbOrigin.x}px`, top: `${y0 - cbOrigin.y}px` });
    if (sizeChanged){
      Object.assign(canvas.style, { width: `${sw}px`, height: `${sh}px` });
      dpr = Math.min(devicePixelRatio || 1, 2, Math.sqrt(5e6 / (sw * sh)));
      canvas.width = Math.max(1, Math.round(sw * dpr)); canvas.height = Math.max(1, Math.round(sh * dpr));
      dpr = canvas.width / sw;
    }
    G.rad = Math.min(parseFloat(getComputedStyle(button).borderTopLeftRadius) || 0, bw / 2, bh / 2);
    if (shapeChanged || !N) build(r);
    else if (mx || my){
      // the filings keep their place relative to the button
      for (let i = 0; i < N; i++){ X[i] += mx; Y[i] += my; HX[i] += mx; HY[i] += my; FX[i] += mx; FY[i] += my; CX[i] += mx; CY[i] += my; }
      win.cx += mx; win.cy += my;
      for (const p of poles){ p[0] += mx; p[1] += my; }
      if (sizeChanged) for (let i = 0; i < N; i++) if (MODE[i] === 1 || MODE[i] === 3){ X[i] = clamp(X[i], 4, sw - 4); Y[i] = clamp(Y[i], 4, sh - 4); }
    }
    dirty = true;
    wake(0.1);
  }
  let lastRect = null;
  function checkPlace(){
    if (lost) return false;
    const r = button.getBoundingClientRect();
    const L = lastRect;
    if (L && L.left === r.left && L.top === r.top && L.width === r.width && L.height === r.height && L.sx === scrollX && L.sy === scrollY) return false;
    lastRect = { left: r.left, top: r.top, width: r.width, height: r.height, sx: scrollX, sy: scrollY };
    const bx = r.left + (G.fixed ? 0 : scrollX), by = r.top + (G.fixed ? 0 : scrollY);
    // negated so a NaN (bounds just re-measured) always counts as moved
    if (!(Math.abs(bx - G.bx) <= 0.25 && Math.abs(by - G.by) <= 0.25 && Math.abs(r.width - G.bw) <= 0.25 && Math.abs(r.height - G.bh) <= 0.25)) layout(r);
    return true;
  }

  // ---- filings, structure of arrays
  let N = 0, capN = 0;
  let X, Y, Z, VX, VY, VZ, TH, TL, LEN, HX, HY, HTH, SEED, AL, SPIN, TPH, FX, FY, CX, CY, T0, T1, TOUCH, RX, RY, RTH, RT, MODE, ST, SPK;
  let tuftIdx = new Int32Array(0), tuftKey = new Float32Array(0);
  let inst = new Float32Array(0);
  function alloc(n){
    capN = n;
    const f = () => new Float32Array(n);
    [X, Y, Z, VX, VY, VZ, TH, TL, LEN, HX, HY, HTH, SEED, AL, SPIN, TPH, FX, FY, CX, CY, T0, T1, TOUCH, RX, RY, RTH, RT] = Array.from({ length: 27 }, f);
    MODE = new Uint8Array(n); ST = new Uint8Array(n); SPK = new Uint8Array(n).fill(255);
    tuftIdx = new Int32Array(n); tuftKey = new Float32Array(n);
    inst = new Float32Array(n * 8);
  }
  const faceSD = (x, y) => sdBox(x, y, G.offX + G.bw / 2, G.offY + G.bh / 2, G.bw / 2, G.bh / 2, Math.max(G.rad, 1));
  const winSD = (x, y) => win.on ? sdBox(x, y, win.cx, win.cy, win.hx, win.hy, win.r) : 1e9;
  // the hidden bar magnet: two opposite poles in the card plane at the ends of the label. In 2D its field
  // lines are circular arcs through both poles, and psi (the angle subtended) is constant along each one.
  function baseField(x, y){
    const ax = x - poles[0][0], ay = y - poles[0][1], bx = x - poles[1][0], by = y - poles[1][1];
    const a2 = ax * ax + ay * ay + 1, b2 = bx * bx + by * by + 1;
    return [ax / a2 - bx / b2, ay / a2 - by / b2, Math.atan2(ay, ax) - Math.atan2(by, bx)];
  }

  function build(r){
    measureLabel(r);
    const cx = G.offX + G.bw / 2, cy = G.offY + G.bh / 2;
    if (win.on){
      const inset = Math.min(win.hy, win.hx * 0.5);
      poles[0] = [win.cx - win.hx + inset, win.cy]; poles[1] = [win.cx + win.hx - inset, win.cy];
    } else { poles[0] = [cx - G.bw * 0.22, cy]; poles[1] = [cx + G.bw * 0.22, cy]; }
    // free area of the face, estimated on a coarse grid
    let free = 0;
    for (let y = G.offY; y < G.offY + G.bh; y += 2) for (let x = G.offX; x < G.offX + G.bw; x += 2) if (faceSD(x, y) < -2 && winSD(x, y) > 1) free += 4;
    const n = clamp(Math.round(free * 0.26 * density), 300, 20000);
    if (n > capN) alloc(n);
    N = n;
    // striation: filings gather along field lines, so candidates are accepted by where they fall
    // between adjacent contours of psi, with the contour spacing chosen to be ~3px in the bands
    const half = Math.max(8, (poles[1][0] - poles[0][0]) / 2), d0 = (win.on ? win.hy : G.bh * 0.25) + 4;
    const E0 = 2 * half / (half * half + d0 * d0);
    const K = TAU / (3.1 * E0);
    let i = 0, tries = 0;
    while (i < n && tries < n * 60){
      tries++;
      const x = G.offX + Math.random() * G.bw, y = G.offY + Math.random() * G.bh;
      if (faceSD(x, y) > -2.2 || winSD(x, y) < 1.4) continue;
      const [ex, ey, psi] = baseField(x, y);
      const E = Math.hypot(ex, ey);
      const spacing = TAU / (K * E + 1e-6);
      const contrast = smooth(1.6, 2.6, spacing) * smooth(9, 5, spacing) + smooth(9, 14, spacing) * 0.5;
      const band = Math.pow(0.5 + 0.5 * Math.cos(psi * K), 2.2);
      const dens = 0.55 + 0.45 * smooth(0.3 * E0, 2 * E0, E);
      if (Math.random() > dens * (1 - contrast + contrast * band)) continue;
      // filings lie in short chains, end to end along the field: that is what draws the lines
      const k = 1 + Math.floor(Math.pow(Math.random(), 1.4) * 5);
      let px = x, py = y, prev = 0, sgn = Math.random() < 0.5 ? 1 : -1;
      for (let j = 0; j < k && i < n; j++){
        const len = clamp(1.2 + 1.4 * Math.exp((Math.random() + Math.random() - 1) * 0.7) - 0.3, 1, 4);
        if (j){
          const [fx, fy] = baseField(px, py), fl = Math.hypot(fx, fy) || 1;
          const step = (prev + len) / 2 + rand(0.1, 0.7);
          px += fx / fl * step * sgn; py += fy / fl * step * sgn;
          if (faceSD(px, py) > -2.2 || winSD(px, py) < 1.4) break;
        }
        const [fx, fy] = baseField(px, py);
        X[i] = HX[i] = px; Y[i] = HY[i] = py; Z[i] = 0; VX[i] = VY[i] = VZ[i] = 0;
        const stray = k === 1 && Math.random() < 0.25;
        HTH[i] = TH[i] = stray ? Math.random() * Math.PI : Math.atan2(fy, fx) + (Math.random() + Math.random() - 1) * 0.2;
        TL[i] = 0; LEN[i] = len;
        SEED[i] = Math.random(); AL[i] = 1; MODE[i] = 0; TOUCH[i] = -10; ST[i] = 0; RX[i] = px; RY[i] = py; RTH[i] = TH[i]; RT[i] = 0;
        prev = len; i++;
      }
    }
    N = i;
    phase = 'rest';
    dirty = true;
  }

  // ---- state
  let phase = 'rest';                 // rest | flying | scattered | restoring
  let lost = false, visible = true;
  let simT = 0, phaseT = 0, quietT = 0, dirty = true;
  const mag = { x: 0, y: 0, tx: 0, ty: 0, on: 0, onT: 0, press: 0, pressT: 0, lagx: 0, lagy: 0, lamp: 0 };
  let ptr = null;                      // the pointer pressing the button
  let motion = 0, pageMotion = 0;
  let relabel = false;
  // under reduced motion a click is acknowledged by a brief brightening of the filings, nothing moves
  let flashAt = -9;
  const flash = () => { const t = (simT - flashAt) / 0.22; return t >= 0 && t < 1 ? Math.sin(Math.PI * Math.sqrt(t)) : 0; };

  const disabled = () => button.disabled || button.getAttribute('aria-disabled') === 'true';
  const calm = () => reduceMotion.matches;

  // Distance from a stage point to the nearest place debris must not reach: the stage's own edge.
  const wallDist = (x, y) => Math.min(x, y, G.stageW - x, G.stageH - y);
  // Other controls inside the stage (a "Save for later" link beside the button): filings are not thrown
  // onto them, so the page around the button stays usable and legible. Found by hit-testing a coarse grid
  // when the filings are released.
  const INTERACTIVE = 'a,button,input,select,textarea,label,summary,[role=button],[tabindex],[contenteditable]';
  let blocks = [];
  function scanBlockers(){
    const r = canvas.getBoundingClientRect(), found = new Set();
    blocks = [];
    for (let y = 4; y < G.stageH; y += 10) for (let x = 4; x < G.stageW; x += 10){
      const px = r.left + x, py = r.top + y;
      if (px < 0 || py < 0 || px >= innerWidth || py >= innerHeight) continue;
      const el = document.elementFromPoint(px, py), hit = el && el.closest && el.closest(INTERACTIVE);
      if (hit && hit !== button && !button.contains(hit) && !hit.contains(button)) found.add(hit);
    }
    for (const el of found){
      const b = el.getBoundingClientRect();
      if (b.width && b.height) blocks.push([b.left - r.left, b.top - r.top, b.right - r.left, b.bottom - r.top]);
    }
  }
  function blockDist(x, y){
    let d = 1e9;
    for (const [ax, ay, bx, by] of blocks){
      const dx = Math.max(ax - x, x - bx), dy = Math.max(ay - y, y - by);
      d = Math.min(d, dx > 0 || dy > 0 ? Math.hypot(Math.max(dx, 0), Math.max(dy, 0)) : Math.max(dx, dy));
    }
    return d;
  }

  // Released, the filings hop off the card in the clumps they were chained in: iron is heavy, so most land
  // within about half a card height of it, a thin fringe goes further and fades out, and a few fall back.
  function scatter(ox, oy){
    scanBlockers();
    const lands = [];
    let flung = 0;
    const bw = G.bw, bh = G.bh, core = 0.6 * bh;
    const left = G.offX, right = G.offX + bw, top = G.offY, bot = G.offY + bh;
    // a narrow margin (a phone column) sends the throw up and down instead of into the stage's side
    const narrowL = left < 40, narrowR = G.stageW - right < 40;
    const CELLS = 7, clumps = new Map();
    function clump(x, y){
      let dx = x - ox, dy = y - oy;
      const dl = Math.hypot(dx, dy);
      if (dl < 0.5){ const a = Math.random() * TAU; dx = Math.cos(a); dy = Math.sin(a); } else { dx /= dl; dy /= dl; }
      const fringe = Math.random() < 0.08;
      for (let k = 0; k < 8; k++){
        const b = (k < 2 ? rand(-0.45, 0.45) : rand(-1.2, 1.2));
        let ux = dx * Math.cos(b) - dy * Math.sin(b), uy = dx * Math.sin(b) + dy * Math.cos(b);
        if ((ux < 0 && narrowL) || (ux > 0 && narrowR)){ ux *= 0.3; if (Math.abs(uy) < 0.2) uy = uy < 0 ? -0.6 : 0.6; }
        const ul = Math.hypot(ux, uy); ux /= ul; uy /= ul;
        // how far along this direction the card ends
        const ex = ux > 1e-3 ? (right - x) / ux : ux < -1e-3 ? (x - left) / -ux : 1e9;
        const ey = uy > 1e-3 ? (bot - y) / uy : uy < -1e-3 ? (y - top) / -uy : 1e9;
        const exit = Math.max(0, Math.min(ex, ey));
        let s;
        if (Math.random() < 0.18) s = exit * rand(0.35, 0.9);
        else s = exit + 2 + (fringe ? core + Math.pow(Math.random(), 1.5) * 0.7 * bh : core * Math.pow(Math.random(), 0.9));
        const tx = x + ux * s, ty = y + uy * s;
        if (tx < 3 || ty < 3 || tx > G.stageW - 3 || ty > G.stageH - 3) continue;
        if (Math.random() > 0.05 + 0.95 * smooth(-4, 20, blockDist(tx, ty))) continue;
        const tf = 0.19 + s / 650 + Math.pow(Math.random(), 1.5) * 0.14;
        return { dx: tx - x, dy: ty - y, tf, spin: rand(-7, 7) };
      }
      return { dx: rand(-4, 4), dy: rand(-4, 4), tf: rand(0.16, 0.22), spin: rand(-4, 4) };
    }
    for (let i = 0; i < N; i++){
      const m = MODE[i];
      if (m === 1) continue;
      if (m === 3){
        // already on the page: the jolt makes them hop a little
        VZ[i] = rand(40, 120); MODE[i] = 1; SPIN[i] = rand(-6, 6); TPH[i] = Math.random() * TAU;
        VX[i] = rand(-15, 15); VY[i] = rand(-15, 15); Z[i] = Math.max(Z[i], 0.3);
        continue;
      }
      const key = ((X[i] / CELLS) | 0) + ((Y[i] / CELLS) | 0) * 8192;
      let c = clumps.get(key);
      if (!c){ c = clump(X[i], Y[i]); clumps.set(key, c); }
      const tf = c.tf * rand(0.97, 1.03);
      const tx = clamp(X[i] + c.dx + rand(-1.2, 1.2), 3, G.stageW - 3), ty = clamp(Y[i] + c.dy + rand(-1.2, 1.2), 3, G.stageH - 3);
      VZ[i] = (0.5 * GRAV * tf * tf - Z[i]) / tf;
      VX[i] = (tx - X[i]) / tf; VY[i] = (ty - Y[i]) / tf;
      SPIN[i] = c.spin + rand(-1.5, 1.5); TPH[i] = Math.random() * 0.4;
      MODE[i] = 1;
      lands.push(tf);
      flung++;
    }
    phase = 'flying'; phaseT = 0; quietT = 0;
    snd('release', lands, flung);
    wake(1);
  }

  function beginRestore(){
    if (phase === 'restoring' || phase === 'rest') return;
    const arrivals = [];
    const t0 = simT + 0.04;
    let end = 0;
    const left = G.offX, w = Math.max(1, G.bw);
    for (let i = 0; i < N; i++){
      if (MODE[i] === 1){ MODE[i] = 3; Z[i] = 0; VZ[i] = 0; }
      const dx = X[i] - HX[i], dy = Y[i] - HY[i], dist = Math.hypot(dx, dy);
      // the sweep runs along the magnet from its north pole to its south: each filing arrives when it
      // passes, and comes in along its own field line (the curve's last tangent is the home direction)
      const u = clamp((HX[i] - left) / w, 0, 1);
      const arrive = t0 + 0.14 + 0.36 * u + rand(-0.04, 0.04);
      const dur = clamp(0.2 + dist / 520, 0.2, 0.5);
      T0[i] = Math.max(t0, arrive - dur); T1[i] = T0[i] + dur;
      FX[i] = X[i]; FY[i] = Y[i];
      const hx = Math.cos(HTH[i]), hy = Math.sin(HTH[i]);
      const sgn = (dx * hx + dy * hy) >= 0 ? 1 : -1;
      const lead = Math.min(dist * 0.55, 30);
      CX[i] = HX[i] + hx * sgn * lead; CY[i] = HY[i] + hy * sgn * lead;
      MODE[i] = 2;
      arrivals.push(T1[i] - t0);
      end = Math.max(end, T1[i] - t0);
    }
    phase = 'restoring'; phaseT = 0;
    snd('restore', arrivals, end);
    wake(1);
  }

  // ---- magnetic field of the cursor magnet: a near pole at height H over the cursor and its far pole
  // above it, trailing the way the hand came from. Returns [bx, by, bz] (bz toward the card is negative).
  let fq = 0, fH = H_HOVER;
  function cursorB(x, y, out){
    let dx = x - mag.x, dy = y - mag.y, dz = -fH;
    let r2 = dx * dx + dy * dy + dz * dz, k = fq / (r2 * Math.sqrt(r2));
    let bx = -dx * k, by = -dy * k, bz = -dz * k;
    dx = x - mag.x - mag.lagx; dy = y - mag.y - mag.lagy; dz = -fH - PARTNER;
    r2 = dx * dx + dy * dy + dz * dz; k = fq / (r2 * Math.sqrt(r2));
    bx += dx * k; by += dy * k; bz += dz * k;
    out[0] = bx; out[1] = by; out[2] = bz;
    return r2;
  }
  const Bv = [0, 0, 0];
  const spokeJit = Array.from({ length: SPOKES }, () => rand(-0.35, 0.35) * SPOKE_W);


  // The tuft under a held magnet: filings near it chain into spikes along a fixed set of spokes, each
  // spike a bundle three filings wide, standing tallest at the middle. Placed explicitly rather than
  // grown from a density sim, which heaped them into a blot.
  const spokeLen = Array.from({ length: SPOKES }, () => rand(9, 15)), spokeRoom = new Float32Array(SPOKES);
  const tuft = { x: 0, y: 0 };
  function tuftCentre(){
    let x = mag.x, y = mag.y;
    const e = 0.5;
    for (let k = 0; k < 3; k++){
      const ws = winSD(x, y);
      if (ws < 6){ const gx = winSD(x + e, y) - winSD(x - e, y), gy = winSD(x, y + e) - winSD(x, y - e), gl2 = Math.hypot(gx, gy) || 1; x += gx / gl2 * (6 - ws); y += gy / gl2 * (6 - ws); }
      const fs = faceSD(x, y);
      if (fs > -TUFT_IN){ const gx = faceSD(x + e, y) - faceSD(x - e, y), gy = faceSD(x, y + e) - faceSD(x, y - e), gl2 = Math.hypot(gx, gy) || 1; x -= gx / gl2 * (fs + TUFT_IN); y -= gy / gl2 * (fs + TUFT_IN); }
    }
    tuft.x = x; tuft.y = y;
  }
  const pageAlpha = (x, y) => {
    // debris fades out toward the stage edge and thins out away from the card, so there is no rim to see
    const fd = faceSD(x, y);
    return smooth(0, 16, wallDist(x, y)) * (1 - 0.8 * smooth(0.55 * G.bh, 1.2 * G.bh + 10, fd));
  };

  function stepSurface(dt, magStill){
    const active = mag.on > 0.01 && !disabled();
    const H = H_HOVER + (H_PRESS - H_HOVER) * mag.press;
    fH = H;
    fq = mag.on * H_HOVER * H_HOVER * (Q_HOVER + (Q_PRESS - Q_HOVER) * mag.press);
    const R = H * 4.6, R2 = R * R;
    // a magnet that has sat still long enough: everything left twitching is taken as settled
    if (magStill && stillT > FREEZE_T && mag.on > 0){ motion = 0; pageMotion = 0; return false; }
    let moving = false, mot = 0, pmot = 0;
    const tufting = active && mag.press > 0.3;
    let nt = 0;
    const TR = TUFT_R * mag.press, TR2 = TR * TR;
    if (tufting) tuftCentre();
    const relaxRate = Math.min(1, dt * 7);
    for (let i = 0; i < N; i++){
      const m = MODE[i];
      if (m !== 0 && m !== 3) continue;
      // on the page only the cursor magnet can move a filing
      if (m === 3 && !active) continue;
      let x = X[i], y = Y[i];
      let influenced = false, vx = 0, vy = 0, tlT = 0, zLift = 0;
      if (tufting && m === 0){
        const tx = x - tuft.x, ty = y - tuft.y, t2 = tx * tx + ty * ty;
        if (t2 < TR2){
          // placed in the second pass below
          TOUCH[i] = simT;
          tuftIdx[nt] = i;
          // a filing keeps the spike it joined: judged afresh each frame, the lanes either side of a
          // spike's root would flip between neighbouring spikes
          if (SPK[i] === 255){ const phi = Math.atan2(ty, tx); SPK[i] = ((Math.round(phi / SPOKE_W) % SPOKES) + SPOKES) % SPOKES; }
          tuftKey[nt] = SPK[i] * 1000 + Math.sqrt(t2);
          nt++;
          continue;
        }
      }
      SPK[i] = 255;
      if (magStill && ST[i] >= SETTLE_N) continue;
      if (active){
        const ddx = x - mag.x, ddy = y - mag.y;
        if (ddx * ddx + ddy * ddy < R2){
          cursorB(x, y, Bv);
          const bh2 = Bv[0] * Bv[0] + Bv[1] * Bv[1], B = Math.sqrt(bh2 + Bv[2] * Bv[2]);
          if (bh2 > ROT_MIN || B > 0.3){
            influenced = true; TOUCH[i] = simT;
            const rr = Math.sqrt(ddx * ddx + ddy * ddy) || 1e-3;
            // align: an induced needle feels a torque ~ B^2 sin(2 delta), and turns either way round
            const del = wrapHalf(Math.atan2(Bv[1], Bv[0]) - TH[i]);
            let dth = clamp(ROT_K * bh2 * Math.sin(2 * del) * dt, -0.7, 0.7);
            if (Math.abs(dth) < 0.5 * dt) dth = 0;
            TH[i] += dth;
            // stand up toward the pole once the pull beats the filing's weight
            tlT = Math.min(0.95, Math.atan2(Math.abs(Bv[2]), Math.sqrt(bh2)) * smooth(0.25, 1.0, B));
            // a hovering magnet only turns and lifts them; a pressed one also drags the loose ones at the
            // edge of the tuft a little way in (static friction holds the rest)
            if (mag.press > 0.05 && rr > TR){
              const r2 = rr * rr + H * H;
              const g = 4 * fq * fq / (r2 * r2 * r2) * mag.press;
              if (g > MU_S){ const sp = Math.min(60, PULL_K * (g - MU_S) * 60); if (sp > 5){ vx = -ddx / rr * sp; vy = -ddy / rr * sp; } }
            }
            zLift = 0.8 * smooth(0.4, 1.2, B);
          }
        }
      }
      let homing = false;
      if (!influenced && m === 0 && simT - TOUCH[i] > RELAX_DELAY){
        // the hidden magnet draws a disturbed filing back into its line
        const hx = HX[i] - x, hy = HY[i] - y, hd = Math.hypot(hx, hy);
        const dth = wrapHalf(HTH[i] - TH[i]);
        if (hd > 0.2 || Math.abs(dth) > 0.02 || TL[i] > 0.02){
          const sp = Math.min(hd * 10 + 3, 110);
          if (hd > 0.2){ vx = hx / hd * Math.min(sp, hd / dt); vy = hy / hd * Math.min(sp, hd / dt); }
          TH[i] += Math.abs(dth) < 0.02 ? dth : dth * relaxRate;
          homing = true;
        } else { x = HX[i]; y = HY[i]; TH[i] = HTH[i]; TL[i] = 0; }
      }
      if (Math.abs(tlT - TL[i]) > 0.02) TL[i] += (tlT - TL[i]) * Math.min(1, dt * 16);
      else TL[i] = tlT;
      const zT = zLift + 0.5 * LEN[i] * Math.sin(TL[i]);
      if (Math.abs(VZ[i]) > 0.5 || Math.abs(zT - Z[i]) > 0.05){
        VZ[i] += ((zT - Z[i]) * 900 - VZ[i] * 34) * dt;
        Z[i] = Math.max(0, Z[i] + VZ[i] * dt);
      } else { VZ[i] = 0; Z[i] = zT; }
      if (vx || vy){ x += vx * dt; y += vy * dt; }
      x = confine(i, m, x, y); y = confY;
      X[i] = x; Y[i] = y;
      const s = settle(i);
      if (homing) ST[i] = 0;
      if (!s || homing) moving = true;
      mot += lastMove; if (m === 3) pmot += lastMove;
    }
    if (nt){
      // second pass: order each spoke's members by radius and hand out slots from the middle outward
      const order = Array.from({ length: nt }, (_, j) => j).sort((p, q) => tuftKey[p] - tuftKey[q]);
      // each spike runs out until it would cross the card's rim or the label window
      for (let k = 0; k < SPOKES; k++){
        const sa = k * SPOKE_W + spokeJit[k], ca = Math.cos(sa), sn = Math.sin(sa);
        let r = 2;
        while (r < spokeLen[k]){ const x = tuft.x + ca * (r + 1), y = tuft.y + sn * (r + 1); if (faceSD(x, y) > -3 || winSD(x, y) < 2) break; r++; }
        spokeRoom[k] = r;
      }
      const rise = Math.min(1, dt * 18);
      let prevK = -1, j = 0;
      for (const o of order){
        const i = tuftIdx[o], k = Math.floor(tuftKey[o] / 1000);
        if (k !== prevK){ prevK = k; j = 0; } else j++;
        // a settled one still holds its slot, so the rest do not crowd onto it
        if (magStill && ST[i] >= SETTLE_N) continue;
        const sa = k * SPOKE_W + spokeJit[k];
        const L = spokeRoom[k] * smooth(0.3, 0.9, mag.press);
        const lane = j % 3, slot = (j / 3) | 0;
        const rt = 2.2 + slot * 2.2;
        const ca = Math.cos(sa), sa2 = Math.sin(sa);
        let tx, ty, tlT, zT;
        const cand = rt <= L;
        const px = tuft.x + ca * rt - sa2 * (lane - 1) * 0.9, py = tuft.y + sa2 * rt + ca * (lane - 1) * 0.9;
        if (cand && faceSD(px, py) < -2 && winSD(px, py) > 1.4){
          const f = 1 - rt / Math.max(L, 1);
          tx = px; ty = py;
          tlT = 0.3 + 0.8 * f; zT = 1 + 5 * f * mag.press;
          TH[i] += wrapHalf(sa - TH[i]) * rise;
        } else {
          // no room on the spike: it lies at the foot of the tuft, pointing in
          const dx = X[i] - tuft.x, dy = Y[i] - tuft.y, dl = Math.hypot(dx, dy) || 1;
          const rr = Math.max(dl, L + 2);
          tx = tuft.x + dx / dl * rr; ty = tuft.y + dy / dl * rr;
          tlT = 0.1; zT = 0.3;
          TH[i] += wrapHalf(Math.atan2(dy, dx) - TH[i]) * rise;
        }
        let x = X[i] + (tx - X[i]) * rise, y = Y[i] + (ty - Y[i]) * rise;
        TL[i] += (tlT - TL[i]) * rise;
        // a leap, not a rise: the height springs, and overshoots a little
        VZ[i] += ((zT - Z[i]) * 1400 - VZ[i] * 30) * dt;
        Z[i] = Math.max(0, Z[i] + VZ[i] * dt);
        if (Math.abs(VZ[i]) < 0.5 && Math.abs(zT - Z[i]) < 0.05){ VZ[i] = 0; Z[i] = zT; }
        x = confine(i, 0, x, y); y = confY;
        X[i] = x; Y[i] = y;
        if (!settle(i) || VZ[i]) moving = true;
        mot += lastMove;
      }
    }
    motion = mot; pageMotion = pmot;
    return moving;
  }
  // keep on the card and out of the label window; on the page, inside the stage
  let confY = 0;
  function confine(i, m, x, y){
    if (m === 0){
      const e = 0.5, fs = faceSD(x, y);
      if (fs > -2){ const gx = faceSD(x + e, y) - faceSD(x - e, y), gy = faceSD(x, y + e) - faceSD(x, y - e), gl2 = Math.hypot(gx, gy) || 1; x -= gx / gl2 * (fs + 2); y -= gy / gl2 * (fs + 2); }
      const ws = winSD(x, y);
      if (ws < 1.4){ const gx = winSD(x + e, y) - winSD(x - e, y), gy = winSD(x, y + e) - winSD(x, y - e), gl2 = Math.hypot(gx, gy) || 1; x += gx / gl2 * (1.4 - ws); y += gy / gl2 * (1.4 - ws); }
    } else {
      x = clamp(x, 2, G.stageW - 2); y = clamp(y, 2, G.stageH - 2);
      AL[i] = pageAlpha(x, y);
    }
    confY = y;
    return x;
  }
  // Settled means it has stayed within a hair of where it was SETTLE_N frames running, judged on where
  // the constraints actually left it: a filing pushed back against the label window every frame, or
  // trading places at a spike's foot, is not moving however hard it is being pushed.
  let lastMove = 0;
  function settle(i){
    const dx = X[i] - RX[i], dy = Y[i] - RY[i];
    lastMove = 0;
    if (Math.abs(dx) < SETTLE_D && Math.abs(dy) < SETTLE_D && Math.abs(wrapHalf(TH[i] - RTH[i])) < SETTLE_A && Math.abs(TL[i] - RT[i]) < 0.05 && !VZ[i]){
      if (ST[i] < 255) ST[i]++;
    } else {
      lastMove = Math.hypot(dx, dy) + Math.abs(wrapHalf(TH[i] - RTH[i])) * LEN[i];
      RX[i] = X[i]; RY[i] = Y[i]; RTH[i] = TH[i]; RT[i] = TL[i]; ST[i] = 0;
    }
    return ST[i] >= SETTLE_N;
  }

  function stepAir(dt){
    let any = false;
    for (let i = 0; i < N; i++){
      if (MODE[i] !== 1) continue;
      any = true;
      VZ[i] -= GRAV * dt;
      X[i] += VX[i] * dt; Y[i] += VY[i] * dt; Z[i] += VZ[i] * dt;
      TH[i] += SPIN[i] * dt;
      TPH[i] += Math.abs(SPIN[i]) * 0.5 * dt;
      TL[i] = Math.abs(Math.sin(TPH[i])) * 0.5;
      AL[i] = pageAlpha(X[i], Y[i]);
      if (Z[i] <= 0){
        Z[i] = 0;
        if (VZ[i] < -140 && Math.random() < 0.35){
          // a small dead hop off the card; friction takes most of the sideways speed
          VZ[i] = -VZ[i] * rand(0.1, 0.2); VX[i] *= 0.2; VY[i] *= 0.2; SPIN[i] *= 0.3;
        } else {
          VZ[i] = 0; VX[i] = 0; VY[i] = 0; TL[i] = 0;
          const onFace = faceSD(X[i], Y[i]) < -2 && winSD(X[i], Y[i]) > 1.4;
          MODE[i] = onFace ? 0 : 3; ST[i] = 0;
          // one that falls back on the card is caught by the hidden magnet like the rest
          if (onFace) TOUCH[i] = simT;
          X[i] = clamp(X[i], 2, G.stageW - 2); Y[i] = clamp(Y[i], 2, G.stageH - 2);
          AL[i] = onFace ? 1 : pageAlpha(X[i], Y[i]);
        }
      }
    }
    return any;
  }

  function stepHoming(dt){
    let any = false;
    for (let i = 0; i < N; i++){
      if (MODE[i] !== 2) continue;
      any = true;
      if (simT < T0[i]) continue;
      const t = Math.min(1, (simT - T0[i]) / (T1[i] - T0[i]));
      // accelerating into the magnet, arriving with a little speed and stopping dead: a snap, not a glide
      const s = t * t * (2.2 - 1.2 * t);
      const a = 1 - s;
      const x = a * a * FX[i] + 2 * a * s * CX[i] + s * s * HX[i];
      const y = a * a * FY[i] + 2 * a * s * CY[i] + s * s * HY[i];
      const tx = 2 * a * (CX[i] - FX[i]) + 2 * s * (HX[i] - CX[i]), ty = 2 * a * (CY[i] - FY[i]) + 2 * s * (HY[i] - CY[i]);
      if (tx * tx + ty * ty > 1e-4){
        const ang = Math.atan2(ty, tx);
        TH[i] = t < 0.9 ? ang : HTH[i] + wrapHalf(ang - HTH[i]) * (1 - t) * 10;
      }
      const dist = Math.hypot(FX[i] - HX[i], FY[i] - HY[i]);
      Z[i] = Math.sin(Math.PI * s) * Math.min(5, 1 + dist * 0.05);
      TL[i] = Math.sin(Math.PI * s) * 0.5;
      X[i] = x; Y[i] = y;
      AL[i] = Math.max(AL[i], smooth(0, 0.3, t));
      if (t >= 1){ MODE[i] = 0; ST[i] = 0; X[i] = HX[i]; Y[i] = HY[i]; TH[i] = HTH[i]; Z[i] = 0; TL[i] = 0; VZ[i] = 0; AL[i] = 1; TOUCH[i] = -10; }
    }
    return any;
  }

  let stillT = 0;
  const anchor = { x: NaN, y: NaN, on: -1, press: -1 };
  function stepMagnet(dt){
    const k = Math.min(1, dt * 30);
    const px = mag.x, py = mag.y;
    mag.x += (mag.tx - mag.x) * k; mag.y += (mag.ty - mag.y) * k;
    if (Math.abs(mag.x - mag.tx) < 0.05 && Math.abs(mag.y - mag.ty) < 0.05){ mag.x = mag.tx; mag.y = mag.ty; }
    // the far pole trails the hand a little, so a moving magnet combs the filings into curves
    const lx = -(mag.x - px) / Math.max(dt, 1e-3) * 0.06, ly = -(mag.y - py) / Math.max(dt, 1e-3) * 0.06;
    mag.lagx += (clamp(lx, -25, 25) - mag.lagx) * Math.min(1, dt * 12);
    mag.lagy += (clamp(ly, -25, 25) - mag.lagy) * Math.min(1, dt * 12);
    if (Math.abs(mag.lagx) < 0.25 && Math.abs(mag.lagy) < 0.25 && mag.x === mag.tx && mag.y === mag.ty){ mag.lagx = 0; mag.lagy = 0; }
    const onT = mag.onT * (phase === 'flying' ? 0 : 1);
    mag.on += (onT - mag.on) * Math.min(1, dt * (onT > mag.on ? 10 : 6));
    mag.press += (mag.pressT - mag.press) * Math.min(1, dt * (mag.pressT > mag.press ? 22 : 9));
    mag.lamp += ((mag.onT > 0 ? 1 : 0) - mag.lamp) * Math.min(1, dt * 6);
    if (Math.abs(mag.on - onT) < 0.02) mag.on = onT;
    if (Math.abs(mag.press - mag.pressT) < 0.004) mag.press = mag.pressT;
    const mm = Math.abs(mag.x - mag.tx) > 0.05 || Math.abs(mag.y - mag.ty) > 0.05 || Math.abs(mag.on - onT) > 0.004 || mag.press !== mag.pressT
      || Math.abs(mag.lagx) > 0.05 || Math.abs(mag.lagy) > 0.05;
    // a real change of where or how hard the magnet is: every filing has to look again
    if (!(Math.abs(mag.tx - anchor.x) <= 1 && Math.abs(mag.ty - anchor.y) <= 1) || onT !== anchor.on || mag.pressT !== anchor.press){
      anchor.x = mag.tx; anchor.y = mag.ty; anchor.on = onT; anchor.press = mag.pressT;
      if (ST) ST.fill(0);
      stillT = 0;
    }
    stillT = mm ? 0 : stillT + dt;
    return mm;
  }

  function stepPhase(dt){
    phaseT += dt;
    if (phase === 'flying'){
      let air = 0;
      for (let i = 0; i < N; i++) if (MODE[i] === 1){ air = 1; break; }
      if (!air){ phase = 'scattered'; phaseT = 0; quietT = 0; }
    } else if (phase === 'scattered'){
      // stirring the fallen filings with the magnet holds the sweep off, but never for long
      quietT = ptr || pageMotion > 60 * dt ? Math.min(quietT, RESTORE_AFTER * 0.5) : quietT + dt;
      if ((quietT > RESTORE_AFTER || phaseT > 2.5) && !ptr) beginRestore();
    } else if (phase === 'restoring'){
      let left = 0;
      for (let i = 0; i < N; i++) if (MODE[i] === 2){ left = 1; break; }
      if (!left){
        phase = 'rest'; phaseT = 0;
        // a label changed mid-scatter is combed in now that the card is whole again
        if (relabel){ relabel = false; build(button.getBoundingClientRect()); }
      }
    }
  }

  // ---- render
  function render(){
    if (lost || gl.isContextLost()) return;
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    if (!N) return;
    const off = disabled(), dis = off ? 0.32 : 1;
    let n = 0;
    for (let i = 0; i < N; i++){
      const a = AL[i] * dis;
      if (a < 0.01) continue;
      const o = n * 8;
      inst[o] = X[i]; inst[o + 1] = Y[i]; inst[o + 2] = Z[i]; inst[o + 3] = TH[i];
      inst[o + 4] = TL[i]; inst[o + 5] = MODE[i] === 1 || MODE[i] === 2 ? -LEN[i] : LEN[i]; inst[o + 6] = a; inst[o + 7] = SEED[i];
      n++;
    }
    gl.useProgram(prog);
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, instBuf);
    gl.bufferData(gl.ARRAY_BUFFER, inst.subarray(0, n * 8), gl.DYNAMIC_DRAW);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.uniform2f(U.uStage, G.stageW, G.stageH);
    gl.uniform1f(U.uDpr, dpr);
    gl.uniform1f(U.uOblique, OBLIQUE);
    gl.uniform2f(U.uSh, SHADOW[0], SHADOW[1]);
    gl.uniform1f(U.uShadowA, pal.shadow);
    gl.uniform3fv(U.uBase, pal.base); gl.uniform3fv(U.uSpec, pal.spec);
    gl.uniform3f(U.uLamp, mag.x, mag.y, mag.lamp);
    gl.uniform1f(U.uDesat, off ? 0.8 : 0);
    gl.uniform1f(U.uFlash, flash());
    gl.uniform1f(U.uShadow, 1);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, n);
    gl.uniform1f(U.uShadow, 0);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, n);
    gl.bindVertexArray(null);
  }

  // ---- loop
  let raf = 0, inFrame = false, last = performance.now(), activeUntil = 0, manual = false, frameCount = 0;
  // one animation loop at most: a wake from inside a frame must not start a second loop
  function wake(sec){
    if (lost || manual) { activeUntil = Math.max(activeUntil, simT + sec); return; }
    activeUntil = Math.max(activeUntil, simT + sec);
    if (!raf && !inFrame){ last = performance.now(); raf = requestAnimationFrame(frame); audioAwake(self, true); }
  }
  function advance(dt){
    frameCount++;
    checkPlace();
    simT += dt;
    let busy = false;
    if (calm()){
      // reduced motion: no physics at all, everything lies where it belongs
      if (phase !== 'rest' || mag.on){
        for (let i = 0; i < N; i++){ MODE[i] = 0; X[i] = HX[i]; Y[i] = HY[i]; TH[i] = HTH[i]; Z[i] = 0; TL[i] = 0; AL[i] = 1; }
        phase = 'rest'; mag.on = mag.onT = mag.press = mag.pressT = 0; dirty = true;
      }
      if (simT - flashAt < 0.26){ busy = true; dirty = true; }
    } else {
      const sub = dt > 1 / 50 ? 2 : 1, h = dt / sub;
      for (let s = 0; s < sub; s++){
        const mm = stepMagnet(h);
        const air = stepAir(h);
        const home = stepHoming(h);
        const surf = stepSurface(h, !mm);
        busy = busy || mm || air || home || surf;
      }
      stepPhase(dt);
      rustleLevel(sound && phase !== 'restoring' ? Math.min(1, motion / dt / 11000) : 0);
    }
    busy = busy || phase !== 'rest' || simT < activeUntil || mag.on > 0.004 && mag.onT === 0;
    if (busy || dirty){ render(); dirty = false; }
    return busy;
  }
  function frame(now){
    raf = 0;
    const dt = Math.min((now - last) / 1000, 1 / 30); last = now;
    if (manual || lost || gl.isContextLost()) return;
    if (!visible && phase === 'rest'){ audioAwake(self, false); rustleLevel(0); return; }
    inFrame = true;
    let busy;
    try { busy = advance(Math.max(dt, 1 / 240)); } finally { inFrame = false; }
    if (busy) raf = requestAnimationFrame(frame);
    else { rustleLevel(0); audioAwake(self, false); }
  }

  // ---- input. The canvas never takes the pointer: the button, and everything around it, keeps working.
  const stageXY = (cx, cy) => { const r = canvas.getBoundingClientRect(); return [cx - r.left, cy - r.top]; };
  const onBtn = t => t === button || button.contains(t);
  function onMove(e){
    if (lost) return;
    checkPlace();
    const [x, y] = stageXY(e.clientX, e.clientY);
    const inStage = x > -30 && y > -30 && x < G.stageW + 30 && y < G.stageH + 30;
    // a touch has no hover: the magnet only comes down while a finger holds the button
    const hover = e.pointerType !== 'touch';
    if (ptr && e.pointerId === ptr.id){ setMag(x, y, 1); return; }
    if (!hover) return;
    if (inStage && x > 2 && y > 2 && x < G.stageW - 2 && y < G.stageH - 2 && !calm() && !disabled()){ setMag(x, y, 1); if (sound) prewarmAudio(); }
    else if (mag.onT){ mag.onT = 0; wake(0.4); }
  }
  function setMag(x, y, on){
    if (!mag.onT && !mag.on){ mag.x = x; mag.y = y; }
    // the magnet stays over the stage, well clear of its edge, so filings on the page are never
    // dragged into a heap along it
    mag.tx = clamp(x, 22, G.stageW - 22); mag.ty = clamp(y, 22, G.stageH - 22);
    mag.onT = on;
    wake(0.3);
  }
  function onLeave(e){ if (!e.relatedTarget && mag.onT){ mag.onT = 0; wake(0.4); } }
  function onDown(e){
    if (lost || e.button !== 0 || !onBtn(e.target) || disabled() || calm()) return;
    audio();
    const [x, y] = stageXY(e.clientX, e.clientY);
    ptr = { id: e.pointerId, x, y };
    setMag(x, y, 1);
    mag.pressT = 1;
    snd('press');
    wake(0.5);
  }
  function onUp(e){
    if (!ptr || e.pointerId !== ptr.id) return;
    const [x, y] = stageXY(e.clientX, e.clientY);
    ptr = null; mag.pressT = 0;
    if (e.pointerType === 'touch'){ mag.onT = 0; }
    lastUp = [x, y];
    wake(0.5);
  }
  const onCancel = e => { if (ptr && e.pointerId === ptr.id){ ptr = null; mag.pressT = 0; if (e.pointerType === 'touch') mag.onT = 0; wake(0.3); } };
  let lastUp = null;
  // Space or Enter held on the focused button brings the magnet down over the middle of it.
  let keyHeld = false;
  function onKeyDown(e){
    if (e.target !== button || (e.key !== ' ' && e.key !== 'Enter') || e.repeat || disabled() || calm() || lost) return;
    audio();
    keyHeld = true;
    setMag(G.offX + G.bw / 2, G.offY + G.bh / 2, 1);
    mag.pressT = 1; snd('press');
  }
  function onKeyUp(e){
    if (!keyHeld || (e.key !== ' ' && e.key !== 'Enter')) return;
    keyHeld = false; mag.pressT = 0; mag.onT = 0; wake(0.4);
  }
  function onBlur(){ if (keyHeld){ keyHeld = false; mag.pressT = 0; mag.onT = 0; wake(0.3); } }
  // The button's own click (pointer, keyboard, or form code calling click()) is what releases the
  // filings. It is only observed, never cancelled or repeated.
  function onClick(e){
    if (lost || !N) return;
    if (calm()){ flashAt = simT; dirty = true; wake(0.3); return; }
    audio();
    let o = e.detail > 0 && lastUp ? lastUp : [G.offX + G.bw / 2, G.offY + G.bh / 2];
    lastUp = null;
    o = [clamp(o[0], G.offX, G.offX + G.bw), clamp(o[1], G.offY, G.offY + G.bh)];
    mag.pressT = 0;
    if (e.detail === 0) mag.onT = 0;
    scatter(o[0], o[1]);
  }
  const cap = { capture: true, passive: true };
  addEventListener('pointermove', onMove, { passive: true });
  document.addEventListener('pointerleave', onLeave, { passive: true });
  document.documentElement.addEventListener('mouseleave', onLeave, { passive: true });
  document.addEventListener('pointerdown', onDown, cap);
  addEventListener('pointerup', onUp, cap);
  addEventListener('pointercancel', onCancel, cap);
  button.addEventListener('keydown', onKeyDown);
  button.addEventListener('keyup', onKeyUp);
  button.addEventListener('blur', onBlur);
  button.addEventListener('click', onClick);

  // ---- keep the stage on the button
  const ro = new ResizeObserver(() => { measureBounds(); lastRect = null; checkPlace(); });
  ro.observe(button, { box: 'border-box' });
  ro.observe(document.documentElement);
  if (document.body) ro.observe(document.body);
  let moTimer = 0;
  const mo = new MutationObserver(() => {
    if (moTimer || raf || inFrame) return;
    moTimer = setTimeout(() => { moTimer = 0; if (!raf) checkPlace(); }, 300);
  });
  mo.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['class', 'style', 'hidden', 'open'] });
  // the label changing (say "Added!") moves the window the filings keep clear of: re-comb the card
  let labelText = button.textContent;
  const bmo = new MutationObserver(() => {
    if (button.textContent !== labelText){
      labelText = button.textContent;
      setTimeout(() => { if (!lost && phase === 'rest'){ build(button.getBoundingClientRect()); render(); } else relabel = true; }, 0);
    }
    dirty = true; wake(0.1);
  });
  bmo.observe(button, { childList: true, characterData: true, subtree: true, attributes: true, attributeFilter: ['disabled', 'aria-disabled'] });
  const onFonts = () => { lastRect = null; G.bw = 0; checkPlace(); };
  document.fonts?.addEventListener?.('loadingdone', onFonts);
  const io = new IntersectionObserver(es => { visible = es[es.length - 1].isIntersecting; if (visible){ checkPlace(); wake(0.1); } });
  io.observe(button);
  const onScrollResize = () => { measureBounds(); lastRect = null; checkPlace(); };
  const onScroll = () => { checkPlace(); };
  const onVis = () => { if (!document.hidden){ last = performance.now(); checkPlace(); wake(0.1); } };
  addEventListener('resize', onScrollResize); addEventListener('scroll', onScroll, { passive: true });
  document.addEventListener('visibilitychange', onVis);
  const onMotionPref = () => { dirty = true; wake(0.1); };
  reduceMotion.addEventListener?.('change', onMotionPref);

  // A lost context shows the plain button until the GPU comes back; then the filings are laid again.
  function onLost(e){
    e.preventDefault();
    lost = true;
    cancelAnimationFrame(raf); raf = 0;
    ptr = null; mag.pressT = 0; rustleLevel(0); audioAwake(self, false);
    canvas.style.visibility = 'hidden';
  }
  function onRestored(){
    try { initGL(); } catch (err) { console.warn('ElementalFX filings: could not rebuild after context loss', err); return; }
    lost = false;
    canvas.style.visibility = '';
    measureBounds(); lastRect = null; G.bw = 0; G.stageW = 0;
    checkPlace();
    dirty = true; wake(0.2);
  }
  canvas.addEventListener('webglcontextlost', onLost);
  canvas.addEventListener('webglcontextrestored', onRestored);

  if (sound) idle(buildPool);
  resolveColor();
  document.body.appendChild(canvas);
  measureBounds();
  checkPlace();
  render();

  // getters must be copied as getters (Object.assign would freeze their current values)
  return Object.defineProperties(self, Object.getOwnPropertyDescriptors({
    supported: true,
    get state(){ return phase; },
    get count(){ return N; },
    setColor(c){ colorOpt = c || 'auto'; resolveColor(); dirty = true; wake(0.05); },
    setDensity(d){ density = clamp(+d || 1, 0.3, 2.5); build(button.getBoundingClientRect()); dirty = true; wake(0.05); },
    restore(){ if (phase === 'flying'){ for (let i = 0; i < N; i++) if (MODE[i] === 1){ MODE[i] = 3; Z[i] = 0; } } beginRestore(); },
    detach(){
      cancelAnimationFrame(raf); raf = 0; manual = true; ro.disconnect(); io.disconnect(); mo.disconnect(); bmo.disconnect(); clearTimeout(moTimer);
      document.fonts?.removeEventListener?.('loadingdone', onFonts);
      canvas.removeEventListener('webglcontextlost', onLost); canvas.removeEventListener('webglcontextrestored', onRestored);
      removeEventListener('resize', onScrollResize); removeEventListener('scroll', onScroll);
      document.removeEventListener('visibilitychange', onVis);
      reduceMotion.removeEventListener?.('change', onMotionPref);
      removeEventListener('pointermove', onMove);
      document.removeEventListener('pointerleave', onLeave); document.documentElement.removeEventListener('mouseleave', onLeave);
      document.removeEventListener('pointerdown', onDown, cap);
      removeEventListener('pointerup', onUp, cap); removeEventListener('pointercancel', onCancel, cap);
      button.removeEventListener('keydown', onKeyDown); button.removeEventListener('keyup', onKeyUp);
      button.removeEventListener('blur', onBlur); button.removeEventListener('click', onClick);
      rustleLevel(0);
      audioAwake(self, false);
      canvas.remove();
      gl.getExtension('WEBGL_lose_context')?.loseContext();
    },
    // test hooks
    get _frames(){ return frameCount; },
    get _asleep(){ return !raf && !inFrame; },
    get _lost(){ return lost; },
    get _canvas(){ return canvas; },
    get _modes(){ const c = [0, 0, 0, 0]; for (let i = 0; i < N; i++) c[MODE[i]]++; return c; },
    // filings lying inside the label window (should be 0 whenever they are on the card)
    get _onLabel(){ let c = 0; for (let i = 0; i < N; i++) if (MODE[i] === 0 && winSD(X[i], Y[i]) < 0) c++; return c; },
    get _edge(){ let c = 0; for (let i = 0; i < N; i++) if (AL[i] > 0.3 && wallDist(X[i], Y[i]) < 6) c++; return c; },
    _manual(on){ manual = on; if (!on) wake(0.1); },
    _advance(sec, fps = 60){ for (let i = 0, n = Math.round(sec * fps); i < n; i++) advance(1 / fps); },
  }));
}

window.ElementalFX = Object.assign(window.ElementalFX || {}, { attachFilings, filingsAudio });
})();
