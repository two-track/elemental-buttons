// Ferrofluid button: turns an existing <button> into a pool of glossy black magnetic liquid.
//
//   <script src="fx/ferro.js"></script>
//   const fx = ElementalFX.attachFerro(document.querySelector('#buy'));   // fx.detach() restores it
//
// A classic script (not a module) so pages work straight from file:// too.
//
// The <button> stays the real control (focus, keyboard, form submission, click handlers); this only
// draws it. The cursor is a magnet. Spikes are not placed: they grow out of a pattern-forming height
// field (Swift-Hohenberg dynamics, the textbook model of the Rosensweig instability) whose drive is the
// local field strength, so crowns emerge, re-pack, split and merge as the magnet moves. The magnet also
// drags a lobe of fluid out of the pill; fast moves tear it into droplets that grow their own crowns and
// race home. Holding the button is a strong field: the pool erupts into a spike forest that turns into
// a labyrinth. Releasing is the click. A drag out of the button is not a click. The label is a bright
// inlay the fluid only covers while spikes stand over it. Without WebGL2 the button is left untouched.
//
// Options: zIndex (default 10), sound (default true; a soft wet slosh on release, after a user gesture).

(() => {
'use strict';

const MAX_BLOBS = 8;          // slot 0 is the lobe still attached to the pool, the rest are free droplets

const VS = `#version 300 es
void main(){ vec2 p = vec2(float((gl_VertexID<<1)&2), float(gl_VertexID&2)); gl_Position = vec4(p*2.-1., 0., 1.); }`;

// Everything is in stage CSS px with y pointing down the page; z is height toward the viewer.
const FS = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uPsi;      // spike height, one texel per sim cell
uniform sampler2D uText;
uniform vec2 uPsiSpan;       // CSS px covered by the pattern texture
uniform vec2 uStage;
uniform float uDpr;
uniform vec4 uBox;           // cx, cy, halfW, halfH
uniform float uRad, uH, uMen, uK, uFlat, uSpread, uZmax, uA, uLam, uCover;
uniform vec2 uPtr;
uniform float uLeanAmt, uRf, uMound;
uniform vec4 uBlob[${MAX_BLOBS}];   // x, y, radius, _
uniform vec4 uNeck;          // lobe anchor xy, neck radius, _
uniform vec4 uRip[4];        // x, y, age, amplitude
out vec4 o;

float sdRound(vec2 q, vec2 b, float r){ vec2 d = abs(q) - b + r; return length(max(d, 0.)) + min(max(d.x, d.y), 0.) - r; }
float smin(float a, float b, float k){ float h = clamp(.5 + .5 * (b - a) / k, 0., 1.); return mix(b, a, h) - k * h * (1. - h); }

float fluidSd(vec2 p){
  float sd = sdRound(p - uBox.xy, uBox.zw, uRad) - uSpread;
  vec4 L = uBlob[0];
  if (L.z > 0.5){
    vec2 a = uNeck.xy, ba = L.xy - a;
    float t = clamp(dot(p - a, ba) / max(dot(ba, ba), 1e-3), 0., 1.);
    float dn = length(p - a - ba * t) - mix(uNeck.z * 1.4, uNeck.z, t);
    sd = smin(sd, dn, 6.);
    sd = smin(sd, length(p - L.xy) - L.z, 8.);
  }
  for (int i = 1; i < ${MAX_BLOBS}; i++){
    vec4 b = uBlob[i];
    if (b.z < 0.5) continue;
    sd = smin(sd, length(p - b.xy) - b.z, b.z * .7);
  }
  return sd;
}

// cubic B-spline through four bilinear taps: C2-smooth, so the spike flanks shade without grid facets
float psiAt(vec2 p){
  vec2 ts = vec2(textureSize(uPsi, 0));
  vec2 g = p / uPsiSpan * ts - .5;
  vec2 i = floor(g), f = g - i;
  vec2 f2 = f * f, f3 = f2 * f;
  vec2 w0 = (1. - 3. * f + 3. * f2 - f3) / 6., w1 = (4. - 6. * f2 + 3. * f3) / 6., w2 = (1. + 3. * f + 3. * f2 - 3. * f3) / 6., w3 = f3 / 6.;
  vec2 s0 = w0 + w1, s1 = w2 + w3;
  vec2 c0 = (i - .5 + w1 / s0) / ts, c1 = (i + 1.5 + w3 / s1) / ts;
  return s0.y * (s0.x * texture(uPsi, vec2(c0.x, c0.y)).r + s1.x * texture(uPsi, vec2(c1.x, c0.y)).r)
       + s1.y * (s0.x * texture(uPsi, vec2(c0.x, c1.y)).r + s1.x * texture(uPsi, vec2(c1.x, c1.y)).r);
}
float heightAt(vec2 p, out float sd, out float spk){
  sd = fluidSd(p);
  spk = 0.;
  if (sd > 0.) return -1.;
  float x = clamp(-sd / uMen, 0., 1.);
  float h = uH * uFlat * sqrt(1. - (1. - x) * (1. - x));
  vec2 dp = uPtr - p;
  float d2 = dot(dp, dp);
  h += uMound * exp(-d2 / (uRf * uRf * .5)) * x;
  // spikes lean toward the magnet: the cross-section at height fraction f is shifted by lean * f
  // directly under the magnet the spikes stand straight; they tilt toward it from the sides
  float dl = sqrt(d2);
  vec2 lean = dp / max(dl, 1.) * uLeanAmt * smoothstep(0., uRf * .6, dl) * exp(-d2 / (uRf * uRf * 2.));
  float inner = smoothstep(0., uLam * .55, -sd);
  float s1 = psiAt(p);
  float s2 = psiAt(p - lean * clamp(s1 / uA, 0., 1.));
  spk = s2 * inner;
  h += spk;
  for (int k = 0; k < 4; k++){
    vec4 r = uRip[k];
    if (r.w <= 0.) continue;
    float w = length(p - r.xy) - r.z * 150.;
    h += r.w * exp(-r.z * 2.4) * sin(w * 0.36) * exp(-w * w / 500.) * x;
  }
  return max(h, 0.02 * x);
}
float heightAt(vec2 p){ float a, b; return heightAt(p, a, b); }

// A photo studio seen in a black mirror: pitch-dark room, thin strip lights in a ring just above the
// horizon with wide dark gaps (a straight cone flank catches one needle-thin streak per strip), and a
// small key softbox overhead. Only these sharp sources may show; everything else must reflect black.
vec3 env(vec3 r){
  float el = asin(clamp(r.z, -1., 1.)) * 57.29578, az = atan(r.y, r.x) * 57.29578;
  vec3 c = vec3(0.);
  // tall strips: a steep flank reflects well below the horizon, and the streak must still reach it
  float band = smoothstep(-50., -40., el) * (1. - smoothstep(18., 24., el));
  float strips = 0.;
  strips += 14. * (1. - smoothstep(4., 6.5, abs(az + 125.)));
  strips += 10. * (1. - smoothstep(2.5, 4.5, abs(az - 5.)));
  strips += 8. * (1. - smoothstep(2.5, 4.5, abs(abs(az) - 172.)));
  strips += 8. * (1. - smoothstep(2.5, 4.5, abs(az + 55.)));
  strips += 9. * (1. - smoothstep(2.5, 4.5, abs(az - 70.)));
  strips += 6. * (1. - smoothstep(2.5, 4.5, abs(az - 125.)));
  c += band * strips * vec3(0.97, 0.98, 1.0);
  if (r.z > 0.3){
    vec2 s = r.xy / r.z;
    vec2 f = (s - vec2(-0.12, -0.585)) / vec2(0.1, 0.028);
    c += vec3(3.) * (1. - smoothstep(0.8, 1.0, max(abs(f.x), abs(f.y))));
  }
  return c;
}

void main(){
  vec2 p = vec2(gl_FragCoord.x, float(textureSize(uText, 0).y) - gl_FragCoord.y) / uDpr;

  // tilted camera (toward the bottom of the page): a surface point at height z shows up k*z higher,
  // so march the view ray down through the height field
  float sd0 = fluidSd(p), sd1 = fluidSd(p + vec2(0., uK * uZmax)), sdm = fluidSd(p + vec2(0., uK * uZmax * .5));
  vec2 q = p;
  bool hit = false;
  if (min(sd0, min(sd1, sdm)) < uK * uZmax * .5){
    const int N = 26;
    float zPrev = uZmax, z = uZmax;
    for (int i = 1; i <= N; i++){
      z = uZmax * (1. - float(i) / float(N));
      if (heightAt(p + vec2(0., uK * z)) >= z){ hit = true; break; }
      zPrev = z;
    }
    if (hit){
      float lo = z, hi = zPrev;
      for (int i = 0; i < 6; i++){
        float m = (lo + hi) * .5;
        if (heightAt(p + vec2(0., uK * m)) >= m) lo = m; else hi = m;
      }
      q = p + vec2(0., uK * lo);
    }
  }

  float sd, spk;
  float h0 = heightAt(q, sd, spk);

  // contact shadow on the page, offset away from the key light
  float sdS = fluidSd(p - vec2(2.5, 5.));
  float shadow = 0.34 * (1. - smoothstep(-8., 18., sdS)) + 0.35 * (1. - smoothstep(-2., 3.5, sd0));
  shadow = clamp(shadow, 0., 0.62);

  if (!hit){ o = vec4(0., 0., 0., shadow); return; }

  float e = 0.4;
  float hx = max(heightAt(q + vec2(e, 0.)), 0.) - max(heightAt(q - vec2(e, 0.)), 0.);
  float hy = max(heightAt(q + vec2(0., e)), 0.) - max(heightAt(q - vec2(0., e)), 0.);
  vec3 N = normalize(vec3(-hx / (2. * e), -hy / (2. * e), 1.));
  // a perspective eye so the flat top sweeps across the lights instead of mirroring one flat colour
  vec3 eye = vec3(uStage.x * .5, uStage.y * .5 + 520. * uK, 520.);
  vec3 V = normalize(eye - vec3(q, h0));
  vec3 R = reflect(-V, N);
  float ndv = max(dot(N, V), 0.);
  // Schlick Fresnel for an oily dielectric (n ~ 1.5, F0 = 0.04) and zero albedo: faces toward the
  // camera go black, only sharp sources and grazing angles light up
  float F = 0.04 + 0.96 * pow(1. - ndv, 5.);
  vec3 col = env(R) * F;
  vec3 L1 = normalize(vec3(-0.62, -0.7, 0.55)), L2 = normalize(vec3(0.85, 0.2, 0.45));
  col += vec3(1.0, 0.97, 0.93) * 60. * pow(max(dot(N, normalize(L1 + V)), 0.), 900.) * F;
  col += vec3(0.82, 0.9, 1.0) * 30. * pow(max(dot(N, normalize(L2 + V)), 0.), 600.) * F;
  // the hairline rim: only the last few degrees before the silhouette
  col += vec3(0.85, 0.9, 1.0) * 0.8 * smoothstep(0.7, 0.95, 1. - ndv) * smoothstep(0.2, 0.5, 1. - N.z);

  col = 1. - exp(-col * 1.2);
  col = pow(col, vec3(1. / 2.2));

  // the label is a bright metal inlay flush with the pool; only standing spikes may cover it
  float txt = texture(uText, q / uStage).r;
  float vis = 1. - smoothstep(uCover * .15, uCover * .6, spk);
  vec3 ink = vec3(0.93, 0.94, 0.955) + 0.05 * (1. - q.y / uStage.y);
  col = mix(col, ink, txt * vis * smoothstep(0., 2., -sd));

  // antialias the silhouette against the page
  float a = clamp(-sd * uDpr * 1.5 + 0.5, 0., 1.);
  o = vec4(col * a, a + shadow * (1. - a));
}`;

// ---------------------------------------------------------------- component
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
const smooth = (a, b, x) => { const t = Math.min(Math.max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t); };
const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
const sdRound = (qx, qy, bx, by, r) => {
  const dx = Math.abs(qx) - bx + r, dy = Math.abs(qy) - by + r;
  return Math.hypot(Math.max(dx, 0), Math.max(dy, 0)) + Math.min(Math.max(dx, dy), 0) - r;
};
const smin = (a, b, k) => { const h = clamp(0.5 + 0.5 * (b - a) / k, 0, 1); return b + (a - b) * h - k * h * (1 - h); };
// semi-implicit spring; returns true while still moving
function spring(s, target, w, z, dt){
  s.v += (w * w * (target - s.x) - 2 * z * w * s.v) * dt;
  s.x += s.v * dt;
  return Math.abs(s.v) > 0.01 || Math.abs(target - s.x) > 0.004;
}

function attachFerro(button, opts = {}){
  const canvas = document.createElement('canvas');
  canvas.setAttribute('aria-hidden', 'true');
  Object.assign(canvas.style, { position: 'absolute', pointerEvents: 'none', zIndex: opts.zIndex ?? 10, left: '0px', top: '0px' });
  const gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: true, antialias: false, depth: false, stencil: false });
  if (!gl) return { detach(){}, supported: false };

  const compile = (type, src) => {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  const U = {};
  for (let i = 0, n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS); i < n; i++){
    const info = gl.getActiveUniform(prog, i);
    U[info.name.replace(/\[0\]$/, '')] = gl.getUniformLocation(prog, info.name);
  }
  const vao = gl.createVertexArray();
  const mkTex = filter => {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, filter], [gl.TEXTURE_MAG_FILTER, filter], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
    return t;
  };
  // R16F rather than R32F: half floats are filterable without an extension, which the B-spline taps rely on
  const psiTex = mkTex(gl.LINEAR), textTex = mkTex(gl.LINEAR);

  // ---- geometry (CSS px, stage space: button's top-left sits at (M, M))
  let bw = 0, bh = 0, M = 0, stageW = 0, stageH = 0, dpr = 1, radius = 0;
  let H = 10, A = 24, Rf = 50, lam = 13, cell = 2, gx = 0, gy = 0, k0 = 1, reach = 60, rLobe = 14;
  let psi = null, lap = null, rF = null, gF = null, pres = null, amp = null, tmp = null, hOut = null;

  const saved = button.getAttribute('style');
  const cs0 = getComputedStyle(button);
  const label = button.textContent.trim();
  const font = `${cs0.fontStyle} ${cs0.fontWeight} ${cs0.fontSize} ${cs0.fontFamily}`;
  Object.assign(button.style, { background: 'transparent', borderColor: 'transparent', color: 'transparent', boxShadow: 'none', textShadow: 'none', position: cs0.position === 'static' ? 'relative' : cs0.position });
  document.body.appendChild(canvas);

  function buildText(){
    const c = document.createElement('canvas');
    c.width = canvas.width; c.height = canvas.height;
    const x = c.getContext('2d');
    x.fillStyle = '#000'; x.fillRect(0, 0, c.width, c.height);
    x.font = font.replace(/(\d+(\.\d+)?)px/, (_, v) => `${parseFloat(v) * dpr}px`);
    x.textAlign = 'center'; x.textBaseline = 'middle';
    x.fillStyle = '#fff';
    x.fillText(label, c.width / 2, c.height / 2 + parseFloat(cs0.fontSize) * dpr * 0.05);
    gl.bindTexture(gl.TEXTURE_2D, textTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, c);
  }

  function place(){
    const r = button.getBoundingClientRect();
    Object.assign(canvas.style, { left: `${r.left + scrollX - M}px`, top: `${r.top + scrollY - M}px` });
  }

  function resize(){
    const r = button.getBoundingClientRect();
    bw = r.width; bh = r.height;
    M = Math.round(Math.max(96, bh * 1.7));
    stageW = bw + 2 * M; stageH = bh + 2 * M;
    dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(stageW * dpr); canvas.height = Math.round(stageH * dpr);
    Object.assign(canvas.style, { width: `${stageW}px`, height: `${stageH}px` });
    radius = Math.min(parseFloat(getComputedStyle(button).borderTopLeftRadius) || 0, bw / 2, bh / 2);
    H = clamp(bh * 0.17, 6, 12);
    lam = clamp(bh * 0.24, 10, 16);
    A = clamp(bh * 0.5, 18, 34);
    Rf = Math.max(46, bh * 0.85);
    reach = bh * 0.95;
    rLobe = bh * 0.4;
    // six cells per spike spacing: enough for the pattern to be smooth, few enough to run on the CPU
    cell = lam / 6;
    k0 = 2 * Math.PI / 6;
    gx = Math.ceil(stageW / cell); gy = Math.ceil(stageH / cell);
    const n = gx * gy;
    amp = new Float32Array(n); tmp = new Float32Array(n); hOut = new Float32Array(n);
    psi = new Float32Array(n); lap = new Float32Array(n); rF = new Float32Array(n); gF = new Float32Array(n); pres = new Float32Array(n);
    gl.bindTexture(gl.TEXTURE_2D, psiTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, gx, gy, 0, gl.RED, gl.FLOAT, psi);
    buildText();
    place();
    dirty = true;
    wake(0.1);
  }

  // ---- behaviour
  const hover = { x: 0, v: 0 }, flat = { x: 1, v: 0 }, spread = { x: 0, v: 0 }, hold = { x: 0, v: 0 }, lobeAmt = { x: 0, v: 0 };
  const lobe = { x: 0, y: 0, vx: 0, vy: 0 };
  const blobs = [];             // free droplets: { x, y, vx, vy, r, mag }
  const ripples = [];
  let pointer = null;           // last known pointer, client coords
  let press = null;             // { id, x0, y0, x, y, dragging, t } in stage coords
  let hoverHoldUntil = 0, simT = 0, activeUntil = 0, tearCool = 0, anchorX = 0, anchorY = 0;
  let suppressClick = false, psiMax = 0, patternLive = false;
  let ppx = -1e4, ppy = -1e4, pvx = 0, pvy = 0;

  function wake(sec){ activeUntil = Math.max(activeUntil, simT + sec); if (!raf && !inFrame) raf = requestAnimationFrame(frame); }
  const stagePt = (cx, cy) => { const r = button.getBoundingClientRect(); return [cx - r.left + M, cy - r.top + M]; };
  const boxSd = (x, y) => sdRound(x - stageW / 2, y - stageH / 2, bw / 2, bh / 2, radius);
  const anchorOf = (x, y) => {
    const hw = Math.max(bw / 2 - bh * 0.4, 0), hh = Math.max(bh / 2 - bh * 0.32, 0);
    return [stageW / 2 + clamp(x - stageW / 2, -hw, hw), stageH / 2 + clamp(y - stageH / 2, -hh, hh)];
  };

  function ripple(x, y, amp){ if (ripples.length >= 4) ripples.shift(); ripples.push({ x, y, t: 0, amp }); wake(2); }

  // ---- sound: a short low-passed noise gurgle, created lazily after a gesture
  let actx = null;
  function slosh(gain){
    if (opts.sound === false) return;
    try {
      actx = actx || new (window.AudioContext || window.webkitAudioContext)();
      const t = actx.currentTime, len = 0.32, buf = actx.createBuffer(1, Math.ceil(actx.sampleRate * len), actx.sampleRate);
      const d = buf.getChannelData(0);
      let b = 0;
      for (let i = 0; i < d.length; i++){ b = b * 0.97 + (Math.random() * 2 - 1) * 0.03; d[i] = b * 6 * Math.pow(1 - i / d.length, 2); }
      const src = actx.createBufferSource(); src.buffer = buf;
      const f = actx.createBiquadFilter(); f.type = 'bandpass'; f.Q.value = 5;
      f.frequency.setValueAtTime(260, t); f.frequency.exponentialRampToValueAtTime(900, t + 0.08); f.frequency.exponentialRampToValueAtTime(180, t + len);
      const g = actx.createGain(); g.gain.setValueAtTime(gain, t); g.gain.exponentialRampToValueAtTime(0.001, t + len);
      src.connect(f).connect(g).connect(actx.destination); src.start(t);
    } catch {}
  }

  function onWinMove(e){
    if (e.pointerType === 'touch' && !press) return;
    pointer = { x: e.clientX, y: e.clientY };
    const [x, y] = stagePt(e.clientX, e.clientY);
    if (boxSd(x, y) < Rf * 2.4 || hover.x > 0.01) wake(0.6);
  }
  function onWinLeave(){ pointer = null; wake(0.6); }

  function onDown(e){
    if (e.button !== 0 || press) return;
    const [x, y] = stagePt(e.clientX, e.clientY);
    press = { id: e.pointerId, x0: x, y0: y, x, y, dragging: false };
    try { button.setPointerCapture(e.pointerId); } catch {}
    wake(1);
  }
  function onMove(e){
    if (!press || e.pointerId !== press.id) return;
    const [x, y] = stagePt(e.clientX, e.clientY);
    press.x = x; press.y = y;
    if (!press.dragging && Math.hypot(x - press.x0, y - press.y0) > 6) press.dragging = true;
  }
  function onUp(e, cancel){
    if (!press || e.pointerId !== press.id) return;
    const p = press; press = null;
    hoverHoldUntil = simT + 0.45;
    if (p.dragging){
      suppressClick = true;           // pulling the fluid around is not pressing the button
      tear(1);
    } else if (!cancel){
      ripple(p.x, p.y, H * 0.5);
      ripple(stageW / 2, stageH / 2, H * 0.3);
      flat.v -= 2.2; spread.v += 26;
      slosh(0.5);
    }
    // collapse: drain the pattern hard for a moment
    for (let k = 0; k < psi.length; k++) psi[k] *= 0.55;
    wake(2.5);
  }
  function onClick(e){
    if (suppressClick){ suppressClick = false; e.preventDefault(); e.stopImmediatePropagation(); return; }
    if (e.detail === 0){ ripple(stageW / 2, stageH / 2, H * 0.45); spread.v += 30; flat.v -= 2; hold.v += 6; wake(2); }
  }
  const onCancel = e => onUp(e, true);
  button.addEventListener('pointerdown', onDown);
  button.addEventListener('pointermove', onMove);
  button.addEventListener('pointerup', onUp);
  button.addEventListener('pointercancel', onCancel);
  button.addEventListener('lostpointercapture', onCancel);
  button.addEventListener('click', onClick, true);
  addEventListener('pointermove', onWinMove, { passive: true });
  document.documentElement.addEventListener('pointerleave', onWinLeave);

  // the attached lobe pinches off into a string of droplets that keep its momentum
  function tear(strength){
    const r = rLobe * lobeAmt.x;
    if (r < 4) return;
    const n = strength > 0.7 ? 3 : 2;
    for (let i = 0; i < n && blobs.length < MAX_BLOBS - 1; i++){
      const t = 1 - 0.22 * i;
      const bx = anchorX + (lobe.x - anchorX) * t, by = anchorY + (lobe.y - anchorY) * t;
      blobs.push({ x: bx, y: by, vx: lobe.vx * t + (Math.random() - 0.5) * 60, vy: lobe.vy * t + (Math.random() - 0.5) * 60,
        r: Math.max(4, r * (i === 0 ? 0.85 : 0.62 - 0.1 * i)), mag: 1, age: 0 });
    }
    lobeAmt.x = 0; lobeAmt.v = 0;
    lobe.x = anchorX; lobe.y = anchorY; lobe.vx = lobe.vy = 0;
    tearCool = simT + 0.35;
    slosh(0.18);
    wake(2);
  }

  // ---- frame
  let raf = 0, inFrame = false, last = performance.now(), dirty = true, frameCount = 0, manual = false;
  let px = -1e4, py = -1e4;

  function fluidSd(x, y){
    let sd = boxSd(x, y) - spread.x;
    const lr = rLobe * lobeAmt.x;
    if (lr > 0.5){
      const bax = lobe.x - anchorX, bay = lobe.y - anchorY;
      const t = clamp(((x - anchorX) * bax + (y - anchorY) * bay) / Math.max(bax * bax + bay * bay, 1e-3), 0, 1);
      const nr = neckR();
      sd = smin(sd, Math.hypot(x - anchorX - bax * t, y - anchorY - bay * t) - nr * (1.4 - 0.4 * t), 6);
      sd = smin(sd, Math.hypot(x - lobe.x, y - lobe.y) - lr, 8);
    }
    for (const b of blobs) sd = smin(sd, Math.hypot(x - b.x, y - b.y) - b.r, b.r * 0.7);
    return sd;
  }
  const neckR = () => {
    const len = Math.hypot(lobe.x - anchorX, lobe.y - anchorY);
    return rLobe * lobeAmt.x * clamp(0.95 - 0.5 * len / reach, 0.38, 0.9);
  };

  function stepBlobs(dt){
    let moving = false;
    const pressing = !!press;
    const mx = pressing ? press.x : px, my = pressing ? press.y : py;
    const magOn = (pointer || pressing) && !reduceMotion.matches && simT > hoverHoldUntil;
    const out = boxSd(mx, my);
    // lobe: follows the magnet when it is off the pool (or dragging), with a limited reach
    [anchorX, anchorY] = anchorOf(mx, my);
    let want = 0, tx = anchorX, ty = anchorY;
    if (magOn && (out > -bh * 0.15 || (pressing && press.dragging)) && out < Rf * 1.6){
      want = pressing && press.dragging ? 1.15 : 1 - smooth(Rf * 0.9, Rf * 1.6, out) * 0.8;
      const dx = mx - anchorX, dy = my - anchorY, d = Math.hypot(dx, dy);
      const lim = pressing && press.dragging ? reach * 1.35 : reach;
      const s = d > lim ? lim / d : 1;
      tx = anchorX + dx * s; ty = anchorY + dy * s;
      if (pressing && press.dragging && d > lim * 1.25 && simT > tearCool) tear(1);
    }
    moving |= spring(lobeAmt, want, 7, 0.75, dt);
    if (lobeAmt.x < 0.03 && want === 0){ lobe.x = anchorX; lobe.y = anchorY; lobe.vx = lobe.vy = 0; }
    const wl = 8, zl = 0.5;
    lobe.vx += (wl * wl * (tx - lobe.x) - 2 * zl * wl * lobe.vx) * dt;
    lobe.vy += (wl * wl * (ty - lobe.y) - 2 * zl * wl * lobe.vy) * dt;
    lobe.x += lobe.vx * dt; lobe.y += lobe.vy * dt;
    if (Math.hypot(lobe.vx, lobe.vy) > 2) moving = true;
    // a flick leaves the lobe far behind its target: past a point the neck cannot follow and it tears
    const sp = Math.hypot(pvx, pvy), lag = Math.hypot(lobe.x - tx, lobe.y - ty);
    if (lag > reach * 0.5 && sp > 300 && lobeAmt.x > 0.3 && simT > tearCool) tear(clamp(lag / reach, 0, 1));

    // free droplets: pulled by the magnet while it is close, always pulled home, merge on contact
    for (let i = blobs.length - 1; i >= 0; i--){
      const b = blobs[i];
      const [hx, hy] = anchorOf(b.x, b.y);
      const home = 25 + 60 * smooth(0.2, 0.9, b.age);
      let ax = (hx - b.x) * home, ay = (hy - b.y) * home;
      if (magOn){
        const dx = mx - b.x, dy = my - b.y, d = Math.hypot(dx, dy) + 1;
        const f = 5200 * Math.exp(-d * d / (Rf * Rf * 1.2));
        ax += dx / d * f; ay += dy / d * f;
      }
      b.vx += (ax - 5.5 * b.vx) * dt; b.vy += (ay - 5.5 * b.vy) * dt;
      b.x += b.vx * dt; b.y += b.vy * dt;
      b.mag = Math.max(0, b.mag - dt * 0.9);
      b.age += dt;
      if (b.age > 0.2 && boxSd(b.x, b.y) < -b.r * 0.2 - spread.x){
        blobs.splice(i, 1);
        ripple(b.x, b.y, H * 0.28 * clamp(b.r / 8, 0.5, 1.3));
        flat.v -= 0.4;
        continue;
      }
      for (let j = 0; j < i; j++){
        const c = blobs[j];
        if (Math.hypot(b.x - c.x, b.y - c.y) < (b.r + c.r) * 0.55){
          const a1 = b.r * b.r, a2 = c.r * c.r;
          c.x = (c.x * a2 + b.x * a1) / (a1 + a2); c.y = (c.y * a2 + b.y * a1) / (a1 + a2);
          c.vx = (c.vx * a2 + b.vx * a1) / (a1 + a2); c.vy = (c.vy * a2 + b.vy * a1) / (a1 + a2);
          c.r = Math.sqrt(a1 + a2); c.mag = Math.max(c.mag, b.mag);
          blobs.splice(i, 1);
          break;
        }
      }
      moving = true;
    }
    return moving;
  }

  // Swift-Hohenberg: dpsi/dt = r psi - (k0^2 + lap)^2 psi + g psi^2 - psi^3. r is the local field
  // drive; g > 0 favours hexagonal spikes, g -> 0 favours stripes (the labyrinth of a strong field).
  function stepPattern(){
    const pressing = !!press;
    const mx = pressing ? press.x : px, my = pressing ? press.y : py;
    const holdAmt = clamp(hold.x, 0, 1.3);
    const hv = hover.x;
    // drive and presence per cell, plus the active bounding box so the quiet margin costs nothing
    let x0 = gx, x1 = -1, y0 = gy, y1 = -1;
    const Rf2 = Rf * Rf, anyMag = hv > 0.01 || holdAmt > 0.01 || blobs.length > 0 || lobeAmt.x > 0.05;
    for (let j = 0, k = 0; j < gy; j++){
      const y = (j + 0.5) * cell;
      for (let i = 0; i < gx; i++, k++){
        const x = (i + 0.5) * cell;
        let pr = 0, drive = 0;
        if (anyMag || psi[k] !== 0){
          const sd = fluidSd(x, y);
          pr = smooth(0, lam * 0.45, -sd);
          if (pr > 0){
            const dx = x - mx, dy = y - my;
            drive = hv * Math.exp(-(dx * dx + dy * dy) / Rf2) * 1.15 + holdAmt;
            if (lobeAmt.x > 0.05){ const ex = x - lobe.x, ey = y - lobe.y, lr = rLobe * lobeAmt.x; drive += lobeAmt.x * 0.6 * Math.exp(-(ex * ex + ey * ey) / (lr * lr * 1.5)); }
            for (const b of blobs) if (b.mag > 0){ const ex = x - b.x, ey = y - b.y; drive += b.mag * 0.9 * Math.exp(-(ex * ex + ey * ey) / (b.r * b.r * 2)); }
          }
        }
        pres[k] = pr;
        // without a held field the drive stays in the hexagon regime; past about 1 the pattern turns to stripes
        rF[k] = pr > 0 ? -0.5 + 1.25 * Math.min(drive, 0.95 + 0.65 * smooth(0.3, 1, holdAmt)) : -1.5;
        gF[k] = 1.25 - 0.95 * smooth(0.5, 1.1, holdAmt);
        if (rF[k] > 0 || Math.abs(psi[k]) > 1e-4){ if (i < x0) x0 = i; if (i > x1) x1 = i; if (j < y0) y0 = j; if (j > y1) y1 = j; }
      }
    }
    if (x1 < 0){ if (patternLive){ psi.fill(0); hOut.fill(0); patternLive = false; return true; } return false; }
    x0 = Math.max(2, x0 - 3); x1 = Math.min(gx - 3, x1 + 3); y0 = Math.max(2, y0 - 3); y1 = Math.min(gy - 3, y1 + 3);
    const k2 = k0 * k0, k4 = k2 * k2, dt = 0.035, subs = 16;
    let maxD = 0;
    for (let s = 0; s < subs; s++){
      for (let j = y0 - 1; j <= y1 + 1; j++){
        let k = j * gx + x0 - 1;
        for (let i = x0 - 1; i <= x1 + 1; i++, k++) lap[k] = psi[k - 1] + psi[k + 1] + psi[k - gx] + psi[k + gx] - 4 * psi[k];
      }
      for (let j = y0; j <= y1; j++){
        let k = j * gx + x0;
        for (let i = x0; i <= x1; i++, k++){
          const p = psi[k], l2 = lap[k - 1] + lap[k + 1] + lap[k - gx] + lap[k + gx] - 4 * lap[k];
          let d = rF[k] * p - (k4 * p + 2 * k2 * lap[k] + l2) + gF[k] * p * p - p * p * p;
          // seed noise where the pool is unstable, so crowns nucleate instead of waiting on round-off
          if (rF[k] > 0 && p * p < 0.0025) d += (Math.random() - 0.5) * 0.25;
          const nv = p + dt * d;
          if (s === subs - 1){ const c = Math.abs(nv - p); if (c > maxD) maxD = c; }
          psi[k] = nv;
        }
      }
    }
    let m = 0;
    for (let j = y0; j <= y1; j++) for (let i = x0, k = j * gx + x0; i <= x1; i++, k++){ if (psi[k] > m) m = psi[k]; if (pres[k] === 0 && rF[k] < 0) psi[k] *= 0.9; }
    psiMax = m;
    patternLive = true;
    toHeight(holdAmt);
    return maxD > 2e-4;
  }

  // Each crest is normalised by the tallest crest within about one spacing, then 1 - sqrt(1 - t) turns
  // its rounded maximum into a point: the pattern keeps its organic layout but every crest is a cone.
  function toHeight(holdAmt){
    const R = 3;
    for (let j = 0; j < gy; j++){
      const row = j * gx;
      for (let i = 0; i < gx; i++){
        let m = 0;
        for (let d = Math.max(0, i - R), e = Math.min(gx - 1, i + R); d <= e; d++){ const v = psi[row + d]; if (v > m) m = v; }
        tmp[row + i] = m;
      }
    }
    for (let j = 0; j < gy; j++) for (let i = 0; i < gx; i++){
      let m = 0;
      for (let d = Math.max(0, j - R), e = Math.min(gy - 1, j + R); d <= e; d++){ const v = tmp[d * gx + i]; if (v > m) m = v; }
      amp[j * gx + i] = m;
    }
    // the windowed max switches between crests abruptly; a box blur turns those steps into ramps
    for (let pass = 0; pass < 2; pass++){
      const src = pass ? tmp : amp, dst = pass ? amp : tmp, sx = pass ? gx : 1, nMax = pass ? gy : gx;
      for (let k = 0; k < psi.length; k++){
        const c = pass ? (k / gx) | 0 : k % gx;
        let sum = 0, cnt = 0;
        for (let d = -2; d <= 2; d++){ const cc = c + d; if (cc >= 0 && cc < nMax){ sum += src[k + d * sx]; cnt++; } }
        dst[k] = sum / cnt;
      }
    }
    // a held field packs the forest denser, so each spike stands a little shorter
    const Ah = A * (1 - 0.28 * smooth(0.6, 1.2, holdAmt));
    for (let k = 0; k < psi.length; k++){
      const p = psi[k], a = amp[k];
      if (a < 0.02){ hOut[k] = 0; continue; }
      const scale = Ah * Math.min(a / 1.6, 1.1) * smooth(0.02, 0.4, a);
      if (p > 0){ const t = Math.min(p / a, 1); hOut[k] = scale * (1 - Math.sqrt(1 - t)); }
      else hOut[k] = scale * 0.06 * Math.max(p / a, -1);
    }
  }

  function step(dt){
    simT += dt;
    let moving = false;
    if (pointer){
      const [nx, ny] = stagePt(pointer.x, pointer.y);
      if (ppx > -1e3){ const a = Math.min(1, dt * 20); pvx += ((nx - ppx) / Math.max(dt, 1e-3) - pvx) * a; pvy += ((ny - ppy) / Math.max(dt, 1e-3) - pvy) * a; }
      ppx = px = nx; ppy = py = ny;
    } else { pvx = pvy = 0; ppx = -1e4; }
    const pressing = !!press;
    const magNear = pointer && boxSd(px, py) < Rf * 1.6;
    const hoverT = magNear && !pressing && simT > hoverHoldUntil && !reduceMotion.matches ? 1 : 0;
    moving |= spring(hover, hoverT, 7, 0.9, dt);
    // holding ramps the field up over ~0.6 s: a spike forest first, then the labyrinth
    moving |= spring(hold, pressing && !press.dragging && !reduceMotion.matches ? 1.25 : 0, pressing ? 4.5 : 10, pressing ? 1 : 0.9, dt);
    moving |= spring(flat, 1, 15, 0.25, dt);
    moving |= spring(spread, 0, 11, 0.25, dt);
    moving |= stepBlobs(dt);
    moving |= stepPattern();
    for (const r of ripples) r.t += dt;
    while (ripples.length && ripples[0].t > 1.8) ripples.shift();
    return moving || ripples.length > 0 || pressing;
  }

  const ripData = new Float32Array(16), blobData = new Float32Array(MAX_BLOBS * 4);
  function render(){
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(prog);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, psiTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gx, gy, gl.RED, gl.FLOAT, hOut);
    gl.uniform1i(U.uPsi, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, textTex); gl.uniform1i(U.uText, 1);
    gl.uniform2f(U.uPsiSpan, gx * cell, gy * cell);
    gl.uniform2f(U.uStage, stageW, stageH);
    gl.uniform1f(U.uDpr, canvas.width / stageW);
    gl.uniform4f(U.uBox, stageW / 2, stageH / 2, bw / 2, bh / 2);
    gl.uniform1f(U.uRad, radius);
    gl.uniform1f(U.uH, H); gl.uniform1f(U.uMen, clamp(bh * 0.22, 7, 14));
    gl.uniform1f(U.uK, 0.5);
    gl.uniform1f(U.uFlat, flat.x); gl.uniform1f(U.uSpread, spread.x);
    const hk = clamp(hold.x, 0, 1.3);
    // a held field makes the forest denser but each spike a little shorter
    gl.uniform1f(U.uA, A);
    gl.uniform1f(U.uLam, lam);
    gl.uniform1f(U.uCover, A * 0.12);
    gl.uniform2f(U.uPtr, press ? press.x : px, press ? press.y : py);
    gl.uniform1f(U.uLeanAmt, lam * 0.4 * hover.x);
    gl.uniform1f(U.uRf, Rf);
    gl.uniform1f(U.uMound, H * 0.35 * hover.x);
    gl.uniform1f(U.uZmax, H * 2 + A * 1.5 + 6);
    blobData.fill(0);
    const lr = rLobe * lobeAmt.x;
    if (lr > 0.5) blobData.set([lobe.x, lobe.y, lr, 0], 0);
    blobs.forEach((b, i) => blobData.set([b.x, b.y, b.r, 0], (i + 1) * 4));
    gl.uniform4fv(U.uBlob, blobData);
    gl.uniform4f(U.uNeck, anchorX, anchorY, neckR(), 0);
    ripData.fill(0);
    ripples.forEach((r, i) => ripData.set([r.x, r.y, r.t, r.amp], i * 4));
    gl.uniform4fv(U.uRip, ripData);
    gl.bindVertexArray(vao); gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  function advance(dt){
    frameCount++;
    const moving = step(dt);
    const busy = moving || simT < activeUntil;
    if (busy || dirty){ render(); dirty = false; }
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

  const ro = new ResizeObserver(() => { const r = button.getBoundingClientRect(); if (Math.abs(r.width - bw) > 0.5 || Math.abs(r.height - bh) > 0.5) resize(); else place(); });
  ro.observe(button);
  const onScrollResize = () => place();
  addEventListener('resize', onScrollResize); addEventListener('scroll', onScrollResize, { passive: true });
  canvas.addEventListener('webglcontextlost', e => e.preventDefault());
  canvas.addEventListener('webglcontextrestored', () => location.reload());

  resize();
  if (document.fonts && document.fonts.status !== 'loaded') document.fonts.ready.then(() => { buildText(); dirty = true; wake(0.1); });

  return {
    supported: true,
    get _frames(){ return frameCount; },
    get _sleeping(){ return !raf; },
    get _psiMax(){ return psiMax; },
    get _blobs(){ return blobs.length; },
    detach(){
      cancelAnimationFrame(raf); ro.disconnect();
      removeEventListener('resize', onScrollResize); removeEventListener('scroll', onScrollResize);
      removeEventListener('pointermove', onWinMove); document.documentElement.removeEventListener('pointerleave', onWinLeave);
      button.removeEventListener('pointerdown', onDown); button.removeEventListener('pointermove', onMove);
      button.removeEventListener('pointerup', onUp); button.removeEventListener('pointercancel', onCancel);
      button.removeEventListener('lostpointercapture', onCancel); button.removeEventListener('click', onClick, true);
      if (saved === null) button.removeAttribute('style'); else button.setAttribute('style', saved);
      canvas.remove();
      try { actx && actx.close(); } catch {}
      gl.getExtension('WEBGL_lose_context')?.loseContext();
    },
    _manual(on){ manual = on; if (!on) wake(0.1); },
    _advance(sec, fps = 60){ for (let i = 0, n = Math.round(sec * fps); i < n; i++) advance(1 / fps); },
  };
}

window.ElementalFX = Object.assign(window.ElementalFX || {}, { attachFerro });
})();
