// Ferrofluid button: turns an existing <button> into a pool of jet-black magnetic liquid.
//
//   <script src="fx/ferro.js"></script>
//   const fx = ElementalFX.attachFerro(document.querySelector('#buy'));   // fx.detach() restores it
//   ElementalFX.ferroAudio.muted = true;                                  // silences every ferro button
//
// A classic script (not a module) so pages work straight from file:// too.
//
// The <button> stays the real control (focus, keyboard, form submission, click handlers); this only
// draws it. The cursor (or a finger) is a magnet. Spikes are not placed: they sit on the crests of a
// pattern-forming field (Swift-Hohenberg dynamics, the textbook model of the Rosensweig instability)
// whose drive is the local field strength, so crowns emerge, re-pack, split and merge as the magnet
// moves. Each crest becomes an analytic needle cone, baked into a height texture on the GPU and ray
// marched from a tilted camera. The magnet pulls a lobe out of the pill; dragging it far tears it into
// droplets that chase the pointer before draining home. Holding the button erupts the whole pool into
// a spike forest (a held Space or Enter does too); releasing is the click. The label is a silver inlay:
// while the magnet only hovers the fluid keeps its slot flat so the words stay readable, and a forest
// swallows the whole word. The button gets touch-action: pan-y, so on a phone a vertical swipe still
// scrolls while a sideways drag pulls the fluid out. Without WebGL2 + EXT_color_buffer_float the button
// is left untouched.
//
// Options: zIndex (default 10), sound (default true; a soft wet slosh on release, after a user gesture).

(() => {
'use strict';

const MAX_BLOBS = 8;          // slot 0 is the lobe still attached to the pool, the rest are free droplets
const MAX_GLYPHS = 64;
const TIP_P = 2.0;            // spike profile h * t^p: concave flanks, bounded slope, sharp tip
const LEAN = 0.8;             // how far (in pattern wavelengths) a spike beside the magnet tips toward it
const SLOT0 = 0.3, SLOT1 = 0.75;   // hold amounts over which a rising forest takes over the label's slot
const CAM_K = 0.5;            // tilted camera: a point at height z shows up CAM_K * z higher on screen

const VS = `#version 300 es
void main(){ vec2 p = vec2(float((gl_VertexID<<1)&2), float(gl_VertexID&2)); gl_Position = vec4(p*2.-1., 0., 1.); }`;

// Everything is in stage CSS px with y pointing down the page; z is height toward the viewer.
const COMMON = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform vec4 uBox;           // cx, cy, halfW, halfH
uniform float uRad, uSpread;
uniform vec4 uBlob[${MAX_BLOBS}];   // x, y, radius, _
uniform vec4 uNeck;          // lobe anchor xy, neck radius, _
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
`;

// Height bake: pool meniscus + mound + ripples, plus the needle cones read from the spike grid.
// Output: height, dh/dx, dh/dy, spike height alone. -1 marks "no fluid".
const FS_BAKE = COMMON + `
uniform sampler2D uSpk;      // one texel per grid cell: spike x, y, height (0 = empty), base radius
uniform sampler2D uLean;     // the same cell: how far the spike's tip leans, x and y
uniform float uCs, uBake;
uniform float uH, uMen, uFlat, uMound, uRf, uRt, uSwell, uSmK;
uniform vec2 uPtr;
uniform vec4 uRip[4];        // x, y, age, amplitude
layout(location = 0) out vec4 o;
layout(location = 1) out vec4 oV;   // r: how much this point lies in the crease where two cone skirts meet

float baseH(vec2 p, out float sd){
  sd = fluidSd(p);
  if (sd > 0.) return -1.;
  float x = clamp(-sd / uMen, 0., 1.);
  float h = uH * uFlat * sqrt(1. - (1. - x) * (1. - x));
  vec2 dp = uPtr - p;
  h += (uMound * exp(-dot(dp, dp) / (uRf * uRf * .35)) + uSwell) * x;
  for (int k = 0; k < 4; k++){
    vec4 r = uRip[k];
    if (r.w <= 0.) continue;
    float w = length(p - r.xy) - r.z * 150.;
    h += r.w * exp(-r.z * 2.4) * sin(w * 0.36) * exp(-w * w / 500.) * x;
  }
  return max(h, 0.02 * x);
}

void main(){
  vec2 p = gl_FragCoord.xy / uBake;
  float sd, sdx, sdy;
  float b = baseH(p, sd);
  oV = vec4(0.);
  if (b < 0.){ o = vec4(-1., 0., 0., 0.); return; }
  float e = .5;
  float bx = baseH(p + vec2(e, 0.), sdx), by = baseH(p + vec2(0., e), sdy);
  vec2 g = (bx < 0. || by < 0.) ? vec2(0.) : vec2(bx - b, by - b) / e;

  ivec2 n = textureSize(uSpk, 0);
  ivec2 c = ivec2(floor(p / uCs));
  float best = 0., vly = 0.;
  vec2 gs = vec2(0.);
  bool any = false;
  for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++){
    ivec2 cc = c + ivec2(dx, dy);
    if (cc.x < 0 || cc.y < 0 || cc.x >= n.x || cc.y >= n.y) continue;
    vec4 s = texelFetch(uSpk, cc, 0);
    if (s.z <= 0.) continue;
    vec2 L = texelFetch(uLean, cc, 0).xy;
    // the cross-section at height z is shifted by L * z / h: find z by fixed-point iteration
    float z = 0., rr = 1., t = 0.;
    vec2 dd = vec2(0.);
    for (int it = 0; it < 3; it++){
      dd = p - s.xy - L * (z / s.z);
      float dist = length(dd);
      rr = sqrt(dist * dist + uRt * uRt);
      t = clamp(1. - (rr - uRt) / s.w, 0., 1.);
      z = s.z * pow(t, ${TIP_P.toFixed(2)});
    }
    if (z <= 0.) continue;
    vec2 gz = -s.z * ${TIP_P.toFixed(2)} * pow(t, ${(TIP_P - 1).toFixed(2)}) / s.w * dd / rr;
    if (!any){ best = z; gs = gz; any = true; continue; }
    // Overlapping skirts meet in a smooth concave fillet rather than a seam. The blend width shrinks to
    // zero as either cone does, so the field stays continuous where a neighbour's skirt ends.
    float k = min(uSmK, 2. * min(z, best));
    // where two skirts sloping opposite ways meet, the normal flips within a pixel or two: sampled there
    // it sweeps across the lights and draws a line of single bright pixels
    float opp = clamp(-dot(gs, gz) / max(length(gs) * length(gz), 1e-4), 0., 1.);
    vly = max(vly, opp * (1. - smoothstep(0., 1.5 * uSmK, abs(z - best))) * smoothstep(0., 1., min(z, best)));
    float hh = clamp(.5 + .5 * (z - best) / max(k, 1e-4), 0., 1.);
    best = mix(best, z, hh) + k * hh * (1. - hh);
    gs = mix(gs, gz, hh);
  }
  // cone bases must not hang off the edge of the pool
  float edge = clamp(-sd * .5, 0., 1.);
  o = vec4(b + best * edge, g + gs * edge, best * edge);
  oV = vec4(vly * edge, 0., 0., 1.);
}`;

const FS_MAIN = COMMON + `
uniform sampler2D uHt;       // baked height (see FS_BAKE)
uniform sampler2D uText;     // r: label ink, g: glyph index / 255
uniform sampler2D uVly;      // crease mask from the bake
uniform vec2 uStage;
uniform float uDpr, uCanvasH, uBake, uZmax, uSlopeK, uPress, uKey, uMen, uZpool, uSlopeLo;
uniform vec4 uSpkBox;         // screen-space box (stage px) the spike cones can occupy: x0, y0, x1, y1
uniform vec4 uGlyph[${MAX_GLYPHS / 4}];
out vec4 o;

vec4 ht(vec2 q){ return texture(uHt, q * uBake / vec2(textureSize(uHt, 0))); }

// A black studio seen in a black mirror. The floor and walls are black, so the steep spike flanks
// (which mirror the floor) stay black. Only a faint sky and two small lights overhead show; every
// needle tip, being rounded, catches them as a pinpoint.
// pts: the two overhead pinpoints; on the smooth mound and ripples they would smear into a grey sheen
vec3 env(vec3 r, float men, float pts){
  vec3 c = vec3(0.012 * smoothstep(0.05, 0.5, r.z));
  if (r.z > 0.3){
    vec2 s = r.xy / r.z;
    vec2 f = (s - vec2(-0.12, -0.585)) / vec2(0.1, 0.028);
    c += vec3(3. * uKey * men) * (1. - smoothstep(0.35, 1.0, abs(f.x))) * (1. - smoothstep(0.8, 1.0, abs(f.y)));
  }
  vec3 k1 = normalize(vec3(-0.55, -0.35, 0.76));
  c += vec3(1., .97, .93) * 5. * pts * smoothstep(0.9955, 0.9985, dot(r, k1));
  vec3 k2 = normalize(vec3(0.7, 0.1, 0.7));
  c += vec3(.85, .92, 1.) * 2.5 * pts * smoothstep(0.9965, 0.999, dot(r, k2));
  return c;
}
// Steep flanks mirror the floor 40 to 60 degrees below the horizon, whatever way they face, and the
// exact angle depends on the spike's slope. Two lamps down there, hairline-thin in azimuth but tall in
// elevation, give every needle one thin streak on its lower-left and lower-right flank at any height.
// Anything wide here (a ring, a broad strip) silvers every flank.
// fw: how many degrees the reflected ray sweeps across one screen pixel. A lamp thinner than that would
// be point-sampled into dashes, so it is widened to the footprint with its energy kept.
float lamp(vec3 r, float az0, float halfW, float el0, float el1, float fw){
  float el = degrees(asin(clamp(r.z, -1., 1.)));
  float d = abs(mod(degrees(atan(r.y, r.x)) - az0 + 540., 360.) - 180.) * cos(radians(el));
  float w = halfW + fw;
  return halfW / w * (1. - smoothstep(w * .35, w, d)) * smoothstep(el0 - 3. - fw, el0, el) * (1. - smoothstep(el1, el1 + 3. + fw, el));
}
vec3 floorLamps(vec3 r, float fw){
  return vec3(1., .97, .93) * 13. * lamp(r, 166., 2.8, -60., -40., fw)
       + vec3(.85, .92, 1.) * 5. * lamp(r, 56., 2.2, -58., -42., fw);
}

vec3 reflAt(vec2 q){
  vec4 h = ht(q);
  vec3 N = normalize(vec3(-h.y, -h.z, 1.));
  vec3 eye = vec3(uBox.x, uBox.y + 520. * ${CAM_K.toFixed(2)}, 520.);
  return reflect(-normalize(eye - vec3(q, h.x)), N);
}

// men: 1 on the pool's meniscus, 0 on its flat top, the mound, ripples and spikes. The overhead softbox
// and the rim line belong to the meniscus alone; on a wide flat top the softbox would mirror as a grey
// bar, and on ripple or mound slopes the rim term would draw grey shards.
vec3 shade(vec2 q, vec4 h, float men){
  vec3 N = normalize(vec3(-h.y, -h.z, 1.));
  vec3 eye = vec3(uBox.x, uBox.y + 520. * ${CAM_K.toFixed(2)}, 520.);
  vec3 V = normalize(eye - vec3(q, h.x));
  vec3 R = reflect(-V, N);
  float ndv = max(dot(N, V), 0.);
  // oily dielectric (F0 = 0.04), zero albedo; steep flanks get no extra Fresnel lift, or they turn grey
  float F = 0.04 + 0.96 * pow(1. - ndv, 5.);
  F = mix(min(F, 0.25), F, smoothstep(0.35, 0.6, N.z));
  float spike = smoothstep(0.5, 3., h.w);
  float vly = texture(uVly, q * uBake / vec2(textureSize(uVly, 0))).r;
  vec3 col = env(R, men, max(men, spike) * (1. - vly)) * F;
  if (spike > 0.){
    float px = 1. / uDpr;
    float fw = degrees(max(acos(clamp(dot(R, reflAt(q + vec2(px, 0.))), -1., 1.)), acos(clamp(dot(R, reflAt(q + vec2(0., px))), -1., 1.))));
    // low bumps (a collapsing forest, a ripple over a cone skirt) stay black; only real needles get streaks
    col += floorLamps(R, min(fw, 9.)) * smoothstep(1.5, 5., h.w) * F * (1. - vly);
  }
  col *= 1. - 0.7 * vly;
  // the pool's hairline rim (the meniscus at rest); spikes don't get it
  col += vec3(0.85, 0.9, 1.0) * 0.8 * smoothstep(0.7, 0.95, 1. - ndv) * smoothstep(0.2, 0.5, 1. - N.z) * (1. - spike) * men;
  col = 1. - exp(-col * 1.2);
  return pow(col, vec3(1. / 2.2));
}

void main(){
  vec2 p = vec2(gl_FragCoord.x, uCanvasH - gl_FragCoord.y) / uDpr;
  const float K = ${CAM_K.toFixed(2)};
  float sd0 = fluidSd(p), sd1 = fluidSd(p + vec2(0., K * uZmax)), sdm = fluidSd(p + vec2(0., K * uZmax * .5));

  // contact shadow on the page, offset away from the key light
  float sdS = fluidSd(p - vec2(2.5, 5.));
  float shadow = 0.34 * (1. - smoothstep(-8., 18., sdS)) + 0.35 * (1. - smoothstep(-2., 3.5, sd0));
  shadow = clamp(shadow, 0., 0.62);

  bool hit = false;
  vec2 q = p;
  float missBest = 1e9;
  vec2 qMiss = p;
  if (min(sd0, min(sd1, sdm)) < K * uZmax * .5){
    // Conservative march down the view ray: the gap f = z - H can shrink by at most
    // (1 + K * maxSlope) per unit of z, so stepping f / uSlopeK can never jump over a spike.
    // a ray that never passes over the spike box only has the pool, mound and ripples to hit: start it
    // lower and let it take the long steps that their gentle slopes allow
    bool spikes = p.x >= uSpkBox.x && p.x <= uSpkBox.z && p.y <= uSpkBox.w && p.y + K * uZmax >= uSpkBox.y;
    float slopeK = spikes ? uSlopeK : uSlopeLo;
    float z = spikes ? uZmax : uZpool, f = z - ht(p + vec2(0., K * z)).x;
    float zPrev = z, fPrev = f;
    for (int i = 0; i < 110; i++){
      if (f <= 0.){ hit = true; break; }
      if (z <= 0.) break;
      zPrev = z; fPrev = f;
      z = max(z - max(f / slopeK, 0.25), 0.);
      vec2 qq = p + vec2(0., K * z);
      vec4 h = ht(qq);
      f = z - h.x;
      if (f > 0. && h.w > 0.3){
        float m = f / max(length(h.yz), 0.5);
        if (m < missBest){ missBest = m; qMiss = qq; }
      }
    }
    if (hit){
      for (int i = 0; i < 4; i++){
        float zs = z + (zPrev - z) * (-f) / max(fPrev - f, 1e-4);
        float fs = zs - ht(p + vec2(0., K * zs)).x;
        if (fs > 0.){ zPrev = zs; fPrev = fs; } else { z = zs; f = fs; }
      }
      q = p + vec2(0., K * z);
    }
  }

  if (!hit){
    // antialias spike silhouettes against the page from the ray's closest approach
    float cov = clamp(1. - missBest * uDpr, 0., 1.);
    if (cov > 0.){
      vec3 col = shade(qMiss, ht(qMiss), 0.);
      o = vec4(col * cov, cov + shadow * (1. - cov));
      return;
    }
    o = vec4(0., 0., 0., shadow);
    return;
  }

  vec4 h = ht(q);
  float sd = fluidSd(q);
  vec3 col = shade(q, h, 1. - smoothstep(uMen * .45, uMen * .95, -sd));

  // the label is a silver inlay flush with the pool; whole glyphs fade out where spikes stand
  vec2 tsz = vec2(textureSize(uText, 0));
  vec2 tuv = q * uDpr / tsz;
  float txt = texture(uText, tuv).r;
  if (txt > 0.){
    int gi = int(texelFetch(uText, clamp(ivec2(tuv * tsz), ivec2(0), ivec2(tsz) - 1), 0).g * 255. + .5);
    float vis = 1.;
    if (gi > 0){ int j = min(gi - 1, ${MAX_GLYPHS - 1}); vis = uGlyph[j >> 2][j & 3]; }
    vis *= 1. - smoothstep(0.3, 2., h.w);
    vec3 ink = (vec3(0.93, 0.94, 0.955) + 0.05 * (1. - q.y / uStage.y)) * (1. - 0.3 * uPress);
    col = mix(col, ink, txt * vis * smoothstep(0., 2., -sd));
  }

  float a = clamp(-sd * uDpr * 1.5 + 0.5, 0., 1.);
  if (h.w > 0.5) a = 1.;
  o = vec4(col * a, a + shadow * (1. - a));
}`;

// ---------------------------------------------------------------- shared
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
const mkSpk = n => ({ x: new Float32Array(n), y: new Float32Array(n), h: new Float32Array(n), r: new Float32Array(n),
  hp: new Float32Array(n), lx: new Float32Array(n), ly: new Float32Array(n), n: 0 });
function inFixedContext(el){
  for (let e = el; e && e !== document.documentElement; e = e.parentElement){
    const p = getComputedStyle(e).position;
    if (p === 'fixed' || p === 'sticky') return true;
  }
  return false;
}

// ---- sound: a short low-passed noise gurgle, created lazily after a gesture, shared by all buttons
const ferroAudio = { muted: false };
let actx = null;
function slosh(gain){
  if (ferroAudio.muted) return;
  try {
    actx = actx || new (window.AudioContext || window.webkitAudioContext)();
    if (actx.state === 'suspended') actx.resume();
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

// ---------------------------------------------------------------- component
function attachFerro(button, opts = {}){
  const noop = () => {};
  const unsupported = { supported: false, detach: noop, _manual: noop, _advance: noop };
  const canvas = document.createElement('canvas');
  canvas.setAttribute('aria-hidden', 'true');
  Object.assign(canvas.style, { position: 'absolute', left: '0px', top: '0px', margin: '0', padding: '0', border: '0', display: 'block', pointerEvents: 'none', zIndex: String(opts.zIndex ?? 10) });
  const gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: true, antialias: false, depth: false, stencil: false });
  if (!gl || !gl.getExtension('EXT_color_buffer_float')) return unsupported;
  const sound = opts.sound === false ? noop : slosh;

  // ---- GL resources, rebuilt from scratch after a lost context
  let P = null, vao = null, htTex = null, vlyTex = null, htFb = null, spkTex = null, leanTex = null, textTex = null, htW = 0, htH = 0;
  function program(fs){
    const sh = (type, src) => {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS) && !gl.isContextLost()) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS) && !gl.isContextLost()) throw new Error(gl.getProgramInfoLog(p));
    const u = {};
    for (let i = 0, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS) || 0; i < n; i++){
      const info = gl.getActiveUniform(p, i);
      u[info.name.replace(/\[0\]$/, '')] = gl.getUniformLocation(p, info.name);
    }
    return { p, u };
  }
  const mkTex = filter => {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, filter], [gl.TEXTURE_MAG_FILTER, filter], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
    return t;
  };
  function initGL(){
    gl.getExtension('EXT_color_buffer_float');
    P = { bake: program(FS_BAKE), main: program(FS_MAIN) };
    vao = gl.createVertexArray();
    htTex = mkTex(gl.LINEAR); vlyTex = mkTex(gl.LINEAR); spkTex = mkTex(gl.NEAREST); leanTex = mkTex(gl.NEAREST); textTex = mkTex(gl.LINEAR);
    htFb = gl.createFramebuffer();
    htW = htH = 0;
  }
  try { initGL(); } catch (err) {
    console.warn('ElementalFX ferro: shaders failed, leaving the button as it is.', err);
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return unsupported;
  }

  // ---- geometry. Stage = the canvas, in CSS px; the button's top-left sits at (offX, offY) in it.
  let bw = 0, bh = 0, offX = 0, offY = 0, bcx = 0, bcy = 0, stageW = 0, stageH = 0, dpr = 1, bake = 1, radius = 0;
  let docX = NaN, docY = NaN, fixedCtx = false, bounds = { w: 0, h: 0 }, cbOrigin = { x: 0, y: 0 };
  let H = 10, A = 24, Rf = 50, lam = 13, cell = 2, gx = 0, gy = 0, k0 = 1, rLobe = 14, rb = 8, cs = 10, sgx = 0, sgy = 0;
  let psi = null, lap = null, rF = null, pres = null, prev = null, spkGrid = null, leanGrid = null;
  let spk = mkSpk(0), tmpl = null, textBox = null;
  let glyphs = [], labelKey = '', lost = false;
  const glyphVis = new Float32Array(MAX_GLYPHS).fill(1), glyphCov = new Float32Array(MAX_GLYPHS);

  const saved = button.getAttribute('style');
  function hideButton(){
    const cs0 = getComputedStyle(button);
    // pan-y: a vertical swipe still scrolls the page, but a sideways drag stays with the button and can
    // pull the fluid out, which a touch-action of auto would hand to the browser as a pan
    Object.assign(button.style, { background: 'transparent', borderColor: 'transparent', color: 'transparent', boxShadow: 'none', textShadow: 'none', touchAction: 'pan-y pinch-zoom' });
    if (cs0.position === 'static') button.style.position = 'relative';
  }
  function showButton(){ if (saved === null) button.removeAttribute('style'); else button.setAttribute('style', saved); }

  const readRadius = () => Math.min(parseFloat(getComputedStyle(button).borderTopLeftRadius) || 0, bw / 2, bh / 2);
  const labelText = () => (button.innerText || button.textContent || '').replace(/\s+/g, ' ').trim();
  const fontKey = () => { const c = getComputedStyle(button); return `${labelText()}|${c.font}|${c.letterSpacing}|${c.borderTopLeftRadius}|${bw}x${bh}|${dpr}`; };

  // The label: ink in red, and each glyph's cell painted with its index in green, so the shader can
  // fade whole glyphs instead of leaving letter fragments between spikes.
  function buildText(){
    if (lost || !stageW) return;
    radius = readRadius();
    const cs0 = getComputedStyle(button);
    const c = document.createElement('canvas');
    c.width = canvas.width; c.height = canvas.height;
    const x = c.getContext('2d');
    x.fillStyle = '#000'; x.fillRect(0, 0, c.width, c.height);
    const text = labelText();
    const fsize = parseFloat(cs0.fontSize) || 16;
    x.font = `${cs0.fontStyle} ${cs0.fontWeight} ${fsize * dpr}px ${cs0.fontFamily}`;
    if ('letterSpacing' in x && cs0.letterSpacing !== 'normal') x.letterSpacing = `${(parseFloat(cs0.letterSpacing) || 0) * dpr}px`;
    x.textAlign = 'left'; x.textBaseline = 'alphabetic';
    let cx = bw / 2, cy = bh / 2;
    const range = document.createRange();
    range.selectNodeContents(button);
    const rr = range.getBoundingClientRect(), br = button.getBoundingClientRect();
    if (rr.width > 0 && rr.height > 0){ cx = rr.left + rr.width / 2 - br.left; cy = rr.top + rr.height / 2 - br.top; }
    const m = x.measureText(text);
    const asc = m.fontBoundingBoxAscent ?? fsize * dpr * 0.8, desc = m.fontBoundingBoxDescent ?? fsize * dpr * 0.2;
    const x0 = (cx + offX) * dpr - m.width / 2, y0 = (cy + offY) * dpr + (asc - desc) / 2;
    const top = y0 - (m.actualBoundingBoxAscent ?? asc) - 2 * dpr, bot = y0 + (m.actualBoundingBoxDescent ?? desc) + 2 * dpr;
    glyphs = [];
    const chars = Array.from(text), nonSpace = chars.filter(ch => ch.trim()).length;
    let prefix = '', gi = 0;
    for (const ch of chars){
      const a = x.measureText(prefix).width;
      prefix += ch;
      if (!ch.trim()) continue;
      const b = x.measureText(prefix).width;
      const slot = nonSpace > MAX_GLYPHS ? Math.floor(gi * MAX_GLYPHS / nonSpace) : gi;
      gi++;
      x.fillStyle = `rgb(0,${slot + 1},0)`;
      x.fillRect(Math.floor(x0 + a - dpr), Math.floor(top), Math.ceil(b - a + 2 * dpr), Math.ceil(bot - top));
      const g = glyphs[slot] || (glyphs[slot] = { x0: Infinity, x1: -Infinity, y0: top / dpr, y1: bot / dpr });
      g.x0 = Math.min(g.x0, (x0 + a) / dpr); g.x1 = Math.max(g.x1, (x0 + b) / dpr);
    }
    textBox = null;
    for (const g of glyphs) if (g){
      textBox = textBox || { x0: Infinity, x1: -Infinity, y0: g.y0, y1: g.y1 };
      textBox.x0 = Math.min(textBox.x0, g.x0); textBox.x1 = Math.max(textBox.x1, g.x1);
    }
    x.globalCompositeOperation = 'lighter';
    x.fillStyle = '#f00';
    x.fillText(text, x0, y0);
    gl.bindTexture(gl.TEXTURE_2D, textTex);
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, c);
    labelKey = fontKey();
    dirty = true;
  }

  function measureBounds(){
    fixedCtx = inFixedContext(button);
    canvas.style.position = fixedCtx ? 'fixed' : 'absolute';
    const de = document.documentElement;
    const disp = canvas.style.display;
    canvas.style.display = 'none';
    bounds = fixedCtx ? { w: de.clientWidth, h: de.clientHeight }
      : { w: de.clientWidth, h: Math.max(de.scrollHeight, document.body ? document.body.scrollHeight : 0) };
    canvas.style.display = disp;
    // where left:0/top:0 lands (the document origin, unless an ancestor is positioned)
    canvas.style.left = '0px'; canvas.style.top = '0px';
    const r = canvas.getBoundingClientRect();
    cbOrigin = { x: r.left + (fixedCtx ? 0 : scrollX), y: r.top + (fixedCtx ? 0 : scrollY) };
    docX = NaN;
  }

  function layout(){
    if (lost) return;
    const r = button.getBoundingClientRect();
    const bx = r.left + (fixedCtx ? 0 : scrollX), by = r.top + (fixedCtx ? 0 : scrollY);
    if (r.width < 2 || r.height < 2){ canvas.style.visibility = 'hidden'; docX = bx; docY = by; return; }
    canvas.style.visibility = '';
    const M = Math.round(Math.max(96, r.height * 1.7));
    // the stage stays inside the document (or viewport), so it can never add scrolling
    const x0 = Math.floor(Math.min(bx, Math.max(0, bx - M))), y0 = Math.floor(Math.min(by, Math.max(0, by - M)));
    const x1 = Math.ceil(Math.max(bx + r.width, Math.min(bounds.w, bx + r.width + M))), y1 = Math.ceil(Math.max(by + r.height, Math.min(bounds.h, by + r.height + M)));
    const sw = x1 - x0, sh = y1 - y0, nOffX = bx - x0, nOffY = by - y0;
    // a big stage is a lot of march per pixel; past ~600k device px the main pass drops to DPR 1.5
    let nd = Math.min(devicePixelRatio || 1, 2);
    if (sw * sh * nd * nd > 600000) nd = Math.min(nd, 1.5);
    const changed = sw !== stageW || sh !== stageH || Math.abs(r.width - bw) > 0.25 || Math.abs(r.height - bh) > 0.25
      || Math.abs(nOffX - offX) > 0.25 || Math.abs(nOffY - offY) > 0.25 || nd !== dpr;
    docX = bx; docY = by;
    Object.assign(canvas.style, { left: `${x0 - cbOrigin.x}px`, top: `${y0 - cbOrigin.y}px` });
    if (!changed) return;
    for (const rp of ripples){ rp.x += nOffX - offX; rp.y += nOffY - offY; }
    bw = r.width; bh = r.height; offX = nOffX; offY = nOffY; stageW = sw; stageH = sh; dpr = nd;
    bcx = offX + bw / 2; bcy = offY + bh / 2;
    Object.assign(canvas.style, { width: `${sw}px`, height: `${sh}px` });
    canvas.width = Math.max(1, Math.round(sw * dpr)); canvas.height = Math.max(1, Math.round(sh * dpr));
    radius = readRadius();
    H = clamp(bh * 0.17, 6, 12);
    lam = clamp(bh * 0.24, 10, 16);
    A = clamp(bh * 0.8, 24, 52);
    Rf = Math.max(46, bh * 0.85);
    rLobe = bh * 0.4;
    // six cells per pattern wavelength: enough for the pattern to be smooth, few enough for the CPU
    cell = lam / 6;
    k0 = 2 * Math.PI / 6;
    // hexagonal crests of a wavelength-lam pattern sit 2 lam / sqrt 3 apart; a grid cell of 0.7 of that
    // holds at most one crest, and a cone of base radius <= cell only reaches its 3x3 neighbourhood
    const spacing = lam * 1.1547;
    // neighbouring skirts overlap a little so the smooth max in the bake can fillet the valleys
    cs = spacing * 0.7; rb = spacing * 0.56;
    gx = Math.ceil(stageW / cell); gy = Math.ceil(stageH / cell);
    const n = gx * gy;
    psi = new Float32Array(n); lap = new Float32Array(n); rF = new Float32Array(n).fill(-1.5); pres = new Float32Array(n); prev = new Float32Array(n);
    activeBox = null;
    spk = mkSpk(Math.ceil(n / 12) + 16 + MAX_BLOBS * 3);
    tmpl = new Float32Array(n); newLattice();
    sgx = Math.ceil(stageW / cs); sgy = Math.ceil(stageH / cs);
    spkGrid = new Float32Array(sgx * sgy * 4);
    gl.bindTexture(gl.TEXTURE_2D, spkTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, sgx, sgy, 0, gl.RGBA, gl.FLOAT, spkGrid);
    leanGrid = new Float32Array(sgx * sgy * 2);
    gl.bindTexture(gl.TEXTURE_2D, leanTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG32F, sgx, sgy, 0, gl.RG, gl.FLOAT, leanGrid);
    // the spike bake needs sub-pixel tips, but not a full DPR-sized texture
    bake = Math.min(dpr, 1.5);
    htW = Math.max(1, Math.round(stageW * bake)); htH = Math.max(1, Math.round(stageH * bake));
    gl.bindTexture(gl.TEXTURE_2D, htTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, htW, htH, 0, gl.RGBA, gl.HALF_FLOAT, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, htFb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, htTex, 0);
    gl.bindTexture(gl.TEXTURE_2D, vlyTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, htW, htH, 0, gl.RED, gl.UNSIGNED_BYTE, null);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, vlyTex, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    blobs.length = 0; lobeAmt.x = lobeAmt.v = 0;
    buildText();
    dirty = true;
    wake(0.1);
  }
  function relayout(){ if (lost) return; measureBounds(); layout(); }
  // cheap per-scroll check; only a real move re-lays the stage out
  function place(){
    if (lost) return;
    const r = button.getBoundingClientRect();
    const bx = r.left + (fixedCtx ? 0 : scrollX), by = r.top + (fixedCtx ? 0 : scrollY);
    if (Math.abs(bx - docX) > 0.25 || Math.abs(by - docY) > 0.25 || Math.abs(r.width - bw) > 0.25 || Math.abs(r.height - bh) > 0.25) layout();
  }

  // ---- behaviour
  const hover = { x: 0, v: 0 }, flat = { x: 1, v: 0 }, spread = { x: 0, v: 0 }, hold = { x: 0, v: 0 }, lobeAmt = { x: 0, v: 0 };
  const lobe = { x: 0, y: 0, vx: 0, vy: 0 };
  const blobs = [];             // free droplets: { x, y, vx, vy, r, mag, age, chase }
  const ripples = [];
  let pointer = null;           // the magnet, client coords
  let pointerTouch = false, touchMagnetUntil = 0;
  let press = null;             // { id, touch, x0, y0, x, y, dragging } in stage coords
  let hoverHoldUntil = 0, simT = 0, activeUntil = 0, tearCool = 0, anchorX = 0, anchorY = 0, lastStir = 0, simAcc = 0;
  let suppressUntil = 0, psiMax = 0, patternLive = false, pressVis = 0, activeBox = null, lastMaxD = 0;
  let ppx = -1e4, ppy = -1e4, pvx = 0, pvy = 0, px = -1e4, py = -1e4;
  let raf = 0, inFrame = false, last = performance.now(), dirty = true, frameCount = 0, manual = false;
  let keyHold = false, pulseUntil = 0;
  // Keep-out: 1 keeps the label's slot clear of cones. It drops as a hold's forest rises, but only comes
  // back once the forest has sunk, so a release never cuts a rectangular hole into a standing forest.
  let keepLin = 1;
  const trail = [];             // recent fast magnet positions { t, x, y }: the wake a flick leaves behind
  const held = () => !!press || keyHold;
  // pressed without dragging: only this (or a keyboard click's pulse) feeds the forest
  const holdStill = () => keyHold || (!!press && (!press.dragging || boxSd(press.x, press.y) < 0)) || simT < pulseUntil;

  // A hexagonal template (three cosines at the pattern wavenumber) that newly driven cells are seeded
  // from, so a crown stands up within a frame or two and Swift-Hohenberg only has to re-pack it. The
  // lattice gets a fresh orientation and phase whenever the pattern dies out completely.
  function newLattice(){
    if (!tmpl) return;
    const th = Math.random() * Math.PI / 3, ox = Math.random() * 12, oy = Math.random() * 12;
    const ks = [0, 1, 2].map(m => [Math.cos(th + m * 2.0944) * k0, Math.sin(th + m * 2.0944) * k0]);
    for (let j = 0, k = 0; j < gy; j++) for (let i = 0; i < gx; i++, k++){
      const x = i - ox, y = j - oy;
      tmpl[k] = (Math.cos(ks[0][0] * x + ks[0][1] * y) + Math.cos(ks[1][0] * x + ks[1][1] * y) + Math.cos(ks[2][0] * x + ks[2][1] * y)) / 3;
    }
  }

  function wake(sec = 0.1){
    // with reduced motion nothing animates: a wake only has to paint the new static state
    if (!motionOK()) sec = Math.min(sec, 0.05);
    activeUntil = Math.max(activeUntil, simT + sec);
    if (!raf && !inFrame && !manual && !lost){ last = performance.now(); raf = requestAnimationFrame(frame); }
  }
  const stagePt = (cx, cy) => { const r = button.getBoundingClientRect(); return [cx - r.left + offX, cy - r.top + offY]; };
  const boxSd = (x, y) => sdRound(x - bcx, y - bcy, bw / 2, bh / 2, radius);
  const anchorOf = (x, y) => {
    const hw = Math.max(bw / 2 - bh * 0.4, 0), hh = Math.max(bh / 2 - bh * 0.32, 0);
    return [bcx + clamp(x - bcx, -hw, hw), bcy + clamp(y - bcy, -hh, hh)];
  };
  const motionOK = () => !reduceMotion.matches;

  function ripple(x, y, amp){ if (ripples.length >= 4) ripples.shift(); ripples.push({ x, y, t: 0, amp }); wake(2); }

  function onWinMove(e){
    if (e.pointerType === 'touch') return;
    pointer = { x: e.clientX, y: e.clientY }; pointerTouch = false;
    const [x, y] = stagePt(e.clientX, e.clientY);
    if (boxSd(x, y) < Rf * 2.4 || hover.x > 0.01 || lobeAmt.x > 0.01) wake(0.6);
  }
  function onWinLeave(){ if (!pointerTouch){ pointer = null; wake(0.6); } }

  function onDown(e){
    if (e.button !== 0 || press) return;
    suppressUntil = 0;
    const [x, y] = stagePt(e.clientX, e.clientY);
    const touch = e.pointerType === 'touch' || e.pointerType === 'pen';
    press = { id: e.pointerId, touch, x0: x, y0: y, x, y, dragging: false };
    // a finger has no hover: the magnet appears where it lands, so the fluid reacts before any scroll
    if (touch && motionOK()){ pointer = { x: e.clientX, y: e.clientY }; pointerTouch = true; touchMagnetUntil = Infinity; hover.v += 14; lastStir = simT; }
    try { button.setPointerCapture(e.pointerId); } catch {}
    if (!motionOK()){ pressVis = 1; dirty = true; }
    wake(1);
  }
  function onMove(e){
    if (!press || e.pointerId !== press.id) return;
    const [x, y] = stagePt(e.clientX, e.clientY);
    press.x = x; press.y = y;
    if (press.touch) pointer = { x: e.clientX, y: e.clientY };
    if (!press.dragging && Math.hypot(x - press.x0, y - press.y0) > 14) press.dragging = true;
    wake(0.6);
  }
  function endPress(e, real){
    const p = press; press = null;
    pressVis = 0; dirty = true;
    // a cancelled touch (the page took over to scroll) leaves the crown standing a moment longer
    if (p.touch) touchMagnetUntil = real ? simT : simT + 0.7;
    if (real) hoverHoldUntil = simT + 0.45;
    const [x, y] = stagePt(e.clientX, e.clientY);
    const inside = real && boxSd(x, y) <= 1;
    // pointer capture sends the click to the button even when released far outside it
    if (real && !inside) suppressUntil = performance.now() + 600;
    if (inside && motionOK()){
      ripple(x, y, H * 0.24);
      ripple(bcx, bcy, H * 0.14);
      flat.v -= 2.2; spread.v += 26;
    }
    if (inside) sound(0.5);
    if (real && !inside && p.dragging && motionOK()) tear(1);
    // The page took a swipe over to scroll. The finger was still pulling on the fluid, so throw a lobe
    // out along the swipe and let it tear instead of just going flat (and never click).
    else if (!real && p.touch && motionOK() && Math.hypot(p.x - p.x0, p.y - p.y0) > 6){ touchMagnetUntil = simT; fling(p.x0, p.y0, p.x, p.y); }
    lastStir = simT;
    wake(2.5);
  }
  function onUp(e){ if (press && e.pointerId === press.id) endPress(e, true); }
  function onCancel(e){ if (press && e.pointerId === press.id) endPress(e, false); }
  // a held Space or Enter gets the same eruption a held pointer does, with the magnet at the centre
  function onKeyDown(e){
    if ((e.key !== ' ' && e.key !== 'Enter') || e.repeat || press || keyHold || !motionOK()) return;
    keyHold = true; hover.v += 14; lastStir = simT;
    wake(1);
  }
  function endKey(){
    if (!keyHold) return;
    keyHold = false;
    hoverHoldUntil = simT + 0.45;
    lastStir = simT;
    wake(2.5);
  }
  function onKeyUp(e){ if (e.key === ' ' || e.key === 'Enter') endKey(); }
  function onClick(e){
    if (e.detail > 0 && performance.now() < suppressUntil){ suppressUntil = 0; e.preventDefault(); e.stopImmediatePropagation(); return; }
    suppressUntil = 0;
    if (e.detail === 0){
      if (motionOK()){ ripple(bcx, bcy, H * 0.22); spread.v += 30; flat.v -= 2; hold.v += 6; pulseUntil = simT + 0.2; wake(2); }
      else { pressVis = 1; dirty = true; wake(0.05); setTimeout(() => { pressVis = 0; dirty = true; wake(0.05); }, 140); }
      sound(0.4);
    }
  }
  button.addEventListener('pointerdown', onDown);
  button.addEventListener('pointermove', onMove);
  button.addEventListener('pointerup', onUp);
  button.addEventListener('pointercancel', onCancel);
  button.addEventListener('lostpointercapture', onCancel);
  button.addEventListener('click', onClick, true);
  button.addEventListener('keydown', onKeyDown);
  button.addEventListener('keyup', onKeyUp);
  button.addEventListener('blur', endKey);
  addEventListener('pointermove', onWinMove, { passive: true });
  document.documentElement.addEventListener('pointerleave', onWinLeave);

  function fling(x0, y0, x1, y1){
    const dx = x1 - x0, dy = y1 - y0, l = Math.hypot(dx, dy) || 1, ux = dx / l, uy = dy / l;
    [anchorX, anchorY] = anchorOf(x1, y1);
    lobeAmt.x = Math.max(lobeAmt.x, 0.9); lobeAmt.v = 0;
    lobe.x = anchorX + ux * bh * 0.6; lobe.y = anchorY + uy * bh * 0.6;
    lobe.vx = pvx = ux * 900; lobe.vy = pvy = uy * 900;
    tear(0.6);
  }
  // the lobe pinches off into a string of droplets that chase the pointer, then drain home
  function tear(strength){
    const r = rLobe * lobeAmt.x;
    if (r < 4) return;
    const n = strength > 0.7 ? 4 : 2;
    const dx = lobe.x - anchorX, dy = lobe.y - anchorY;
    // a string laid back along the neck from the lobe, but only where it is outside the pool
    // a fast lobe leaves its string behind along its path; a slow pull leaves it along the neck
    const pv = Math.hypot(pvx, pvy), l = Math.hypot(dx, dy) || 1, fast = !press && pv > 300;
    const ux = fast ? pvx / pv : dx / l, uy = fast ? pvy / pv : dy / l;
    let ahead = null, along = 0;
    for (let i = 0; i < n && blobs.length < MAX_BLOBS - 1; i++){
      const br = Math.max(3.5, r * (i === 0 ? 0.8 : 0.6 - 0.07 * i));
      if (ahead) along += (ahead.r + br) * 1.2;
      const x = lobe.x - ux * along, y = lobe.y - uy * along;
      if (i > 0 && boxSd(x, y) < -br * 0.3) break;
      const b = { x, y, vx: fast ? pvx * 0.6 : lobe.vx, vy: fast ? pvy * 0.6 : lobe.vy, r: br, mag: 1, age: 0, chase: 0.45 + 0.06 * i, ahead };
      blobs.push(b);
      ahead = b;
    }
    lobeAmt.x = 0; lobeAmt.v = 0;
    lobe.x = anchorX; lobe.y = anchorY; lobe.vx = lobe.vy = 0;
    tearCool = simT + 0.6;
    sound(0.18);
    wake(2);
  }

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
  // the neck thins as the lobe is pulled further out
  const neckR = () => {
    const len = Math.hypot(lobe.x - anchorX, lobe.y - anchorY);
    return rLobe * lobeAmt.x * clamp(0.95 - 0.42 * len / bh, 0.22, 0.9);
  };

  function magnet(){
    if (press) return [press.x, press.y];
    if (keyHold) return [bcx, bcy];
    return [px, py];
  }

  // The region a body of radius r can occupy without its cones or its soft shadow running off the stage
  // (the canvas clips them with a hard straight edge). Tips stand CAM_K * height above the centre; the
  // shadow reaches 18 px past the body, offset 2.5 px right and 5 px down.
  function fenceBox(r){
    const top = Math.max(r + 17, CAM_K * A * 0.95 + LEAN * lam + 4), bot = r + 27, left = r + 20, right = r + 25;
    let x0 = left, x1 = stageW - right, y0 = top, y1 = stageH - bot;
    if (x1 < x0) x0 = x1 = (x0 + x1) / 2;
    if (y1 < y0) y0 = y1 = (y0 + y1) / 2;
    return [x0, x1, y0, y1];
  }
  // a soft wall that slows a body well before the stage edge, then a hard stop
  function fence(o, r, dt){
    const [x0, x1, y0, y1] = fenceBox(r), band = 18, k = 2400;
    const wall = (p, v, lo, hi) => {
      if (p < lo + band){ const d = Math.min((lo + band - p) / band, 1.5); v += k * d * d * dt; if (v < 0) v *= Math.max(0, 1 - 12 * d * dt); }
      if (p > hi - band){ const d = Math.min((p - hi + band) / band, 1.5); v -= k * d * d * dt; if (v > 0) v *= Math.max(0, 1 - 12 * d * dt); }
      if (p < lo){ p = lo; v = Math.max(v, 0); }
      if (p > hi){ p = hi; v = Math.min(v, 0); }
      return [p, v];
    };
    [o.x, o.vx] = wall(o.x, o.vx, x0, x1);
    [o.y, o.vy] = wall(o.y, o.vy, y0, y1);
  }

  function stepBlobs(dt){
    let moving = false;
    const pressing = held(), dragging = !!press && press.dragging;
    const [mx, my] = magnet();
    const magOn = (pointer || pressing) && motionOK() && (pressing || simT > hoverHoldUntil);
    const out = boxSd(mx, my);
    [anchorX, anchorY] = anchorOf(mx, my);
    let want = 0, tx = anchorX, ty = anchorY;
    const dx = mx - anchorX, dy = my - anchorY, d = Math.hypot(dx, dy);
    if (magOn && dragging && out > -bh * 0.15){
      // pulled by hand: the lobe follows far out on a thinning neck, and tears past a length
      want = 1.1;
      const lim = bh * 1.5, s = d > lim ? lim / d : 1;
      tx = anchorX + dx * s; ty = anchorY + dy * s;
      if (d > bh * 1.85 && simT > tearCool && lobeAmt.x > 0.6) tear(1);
    } else if (magOn && out > -bh * 0.15 && out < Rf * 1.6){
      // hovering: a modest bulge toward the magnet that stays close to the button
      want = 1 - smooth(Rf * 0.6, Rf * 1.6, out) * 0.85;
      const lim = bh * 0.5, s = d > lim ? lim / d : 1;
      tx = anchorX + dx * s; ty = anchorY + dy * s;
    }
    if (want > 0){ const [x0, x1, y0, y1] = fenceBox(rLobe * want); tx = clamp(tx, x0, x1); ty = clamp(ty, y0, y1); }
    moving |= spring(lobeAmt, want, 7, 0.75, dt);
    if (lobeAmt.x < 0.03 && want === 0){ lobe.x = anchorX; lobe.y = anchorY; lobe.vx = lobe.vy = 0; }
    const wl = 8, zl = 0.5;
    lobe.vx += (wl * wl * (tx - lobe.x) - 2 * zl * wl * lobe.vx) * dt;
    lobe.vy += (wl * wl * (ty - lobe.y) - 2 * zl * wl * lobe.vy) * dt;
    lobe.x += lobe.vx * dt; lobe.y += lobe.vy * dt;
    if (lobeAmt.x > 0.03) fence(lobe, rLobe * lobeAmt.x, dt);
    if (Math.hypot(lobe.vx, lobe.vy) > 2) moving = true;
    // a flick leaves the lobe behind its target: past a point the neck cannot follow and it tears
    const sp = Math.hypot(pvx, pvy), lag = Math.hypot(lobe.x - tx, lobe.y - ty);
    if (!pressing && lag > bh * 0.3 && sp > 650 && lobeAmt.x > 0.4 && simT > tearCool) tear(clamp(lag / bh, 0, 1));

    // free droplets: chase the magnet for a moment, then drain home; merge once settled
    for (let i = blobs.length - 1; i >= 0; i--){
      const b = blobs[i];
      const [hx, hy] = anchorOf(b.x, b.y);
      const chasing = magOn && b.age < b.chase;
      const home = chasing ? 0 : 30 + 60 * smooth(0.4, 1.2, b.age);
      let ax = (hx - b.x) * home, ay = (hy - b.y) * home;
      if (chasing){
        // the lead droplet chases the magnet; the rest trail it in a string, one spacing apart
        const f = b.ahead && blobs.includes(b.ahead) ? b.ahead : null;
        let gx2 = mx, gy2 = my;
        if (f){
          const ex = b.x - f.x, ey = b.y - f.y, l = Math.hypot(ex, ey) || 1, gap = (f.r + b.r) * 1.25;
          gx2 = f.x + ex / l * gap; gy2 = f.y + ey / l * gap;
        }
        ax += (gx2 - b.x) * 160; ay += (gy2 - b.y) * 160;
      } else if (magOn){
        const ex = mx - b.x, ey = my - b.y, e2 = Math.hypot(ex, ey) + 1;
        const f = 2600 * Math.exp(-e2 * e2 / (Rf * Rf));
        ax += ex / e2 * f; ay += ey / e2 * f;
      }
      const damp = chasing ? 14 : 7;
      b.vx += (ax - damp * b.vx) * dt; b.vy += (ay - damp * b.vy) * dt;
      b.x += b.vx * dt; b.y += b.vy * dt;
      // droplets never stray far from the button
      const far = boxSd(b.x, b.y) - bh * (pressing ? 1.9 : 1.3);
      if (far > 0){ const [ax2, ay2] = anchorOf(b.x, b.y), l = Math.hypot(b.x - ax2, b.y - ay2) || 1; b.x -= (b.x - ax2) / l * far; b.y -= (b.y - ay2) / l * far; }
      fence(b, b.r, dt);
      b.mag = Math.max(0, b.mag - dt * 0.9);
      b.age += dt;
      if (b.age > Math.max(0.3, b.chase) && boxSd(b.x, b.y) < -b.r * 0.2 - spread.x){
        blobs.splice(i, 1);
        ripple(b.x, b.y, H * 0.18 * clamp(b.r / 8, 0.5, 1.3));
        flat.v -= 0.4;
        continue;
      }
      if (b.age > b.chase + 0.15){
        for (let j = 0; j < i; j++){
          const c = blobs[j];
          if (c.age > c.chase + 0.15 && Math.hypot(b.x - c.x, b.y - c.y) < (b.r + c.r) * 0.55){
            const a1 = b.r * b.r, a2 = c.r * c.r;
            c.x = (c.x * a2 + b.x * a1) / (a1 + a2); c.y = (c.y * a2 + b.y * a1) / (a1 + a2);
            c.vx = (c.vx * a2 + b.vx * a1) / (a1 + a2); c.vy = (c.vy * a2 + b.vy * a1) / (a1 + a2);
            c.r = Math.sqrt(a1 + a2); c.mag = Math.max(c.mag, b.mag);
            blobs.splice(i, 1);
            break;
          }
        }
      }
      moving = true;
    }
    return moving;
  }

  // Swift-Hohenberg: dpsi/dt = r psi - (k0^2 + lap)^2 psi + g psi^2 - psi^3. r is the local field
  // drive; g > 0 keeps the pattern hexagonal (spikes) rather than striped.
  const SUB_DT = 1 / 960;       // one pattern substep per 1/960 s of real time, whatever the frame rate
  const SUB_BUDGET = 3;         // ms of pattern stepping per frame at most
  let subCost = 2e-5;           // measured ms per cell per substep
  function stepPattern(dt){
    simAcc = Math.min(simAcc + dt, SUB_DT * 32);
    let subs = Math.floor(simAcc / SUB_DT);
    simAcc -= subs * SUB_DT;
    const pressing = held(), still = holdStill();
    const [mx, my] = magnet();
    const holdAmt = clamp(hold.x, 0, 1.3);
    const hv = hover.x;
    const Rf2 = Rf * Rf, anyMag = hv > 0.01 || holdAmt > 0.01 || blobs.length > 0 || lobeAmt.x > 0.05 || trail.length > 0;
    const stirring = simT - lastStir < 0.35;
    // A static magnet over a settled pattern is static fluid: stop integrating so the loop can sleep.
    if (!stirring && !pressing && patternLive && simT - lastStir > 3) return false;

    // only the region the fluid can occupy, plus wherever pattern is still decaying
    let fx0 = bcx - bw / 2 - spread.x, fx1 = bcx + bw / 2 + spread.x, fy0 = bcy - bh / 2 - spread.x, fy1 = bcy + bh / 2 + spread.x;
    const lr = rLobe * lobeAmt.x;
    if (lr > 0.5){ fx0 = Math.min(fx0, lobe.x - lr); fx1 = Math.max(fx1, lobe.x + lr); fy0 = Math.min(fy0, lobe.y - lr); fy1 = Math.max(fy1, lobe.y + lr); }
    const TR = 0.15;
    while (trail.length && simT - trail[0].t > TR) trail.shift();
    for (const b of blobs){ fx0 = Math.min(fx0, b.x - b.r); fx1 = Math.max(fx1, b.x + b.r); fy0 = Math.min(fy0, b.y - b.r); fy1 = Math.max(fy1, b.y + b.r); }
    let i0 = clamp(Math.floor(fx0 / cell) - 2, 2, gx - 3), i1 = clamp(Math.ceil(fx1 / cell) + 2, 2, gx - 3);
    let j0 = clamp(Math.floor(fy0 / cell) - 2, 2, gy - 3), j1 = clamp(Math.ceil(fy1 / cell) + 2, 2, gy - 3);
    if (activeBox){ i0 = Math.min(i0, activeBox[0]); i1 = Math.max(i1, activeBox[1]); j0 = Math.min(j0, activeBox[2]); j1 = Math.max(j1, activeBox[3]); }

    let x0 = gx, x1 = -1, y0 = gy, y1 = -1;
    for (let j = j0; j <= j1; j++){
      const y = (j + 0.5) * cell;
      for (let i = i0, k = j * gx + i0; i <= i1; i++, k++){
        const x = (i + 0.5) * cell;
        let pr = 0, drive = 0;
        if (anyMag || psi[k] !== 0){
          const sd = fluidSd(x, y);
          pr = smooth(0, lam * 0.45, -sd);
          if (pr > 0){
            const dx = x - mx, dy = y - my;
            let near = Math.exp(-(dx * dx + dy * dy) / (Rf2 * 0.45));
            for (const t of trail){
              const ex = x - t.x, ey = y - t.y, d2 = ex * ex + ey * ey;
              if (d2 < Rf2 * 0.5) near = Math.max(near, (1 - (simT - t.t) / TR) * Math.exp(-d2 / (Rf2 * 0.12)));
            }
            drive = hv * near * 1.45 + holdAmt * (still ? 1 : 0.6);
            if (lobeAmt.x > 0.05){ const ex = x - lobe.x, ey = y - lobe.y, r2 = lr * lr; drive += lobeAmt.x * 0.7 * Math.exp(-(ex * ex + ey * ey) / (r2 * 1.5)); }
            for (const b of blobs) if (b.mag > 0){ const ex = x - b.x, ey = y - b.y; drive += b.mag * 0.9 * Math.exp(-(ex * ex + ey * ey) / (b.r * b.r * 2)); }
          }
        }
        pres[k] = pr;
        const was = rF[k];
        rF[k] = pr > 0 ? -0.5 + 1.25 * Math.min(drive, 1.55) : -1.5;
        // newly driven fluid snaps straight to the lattice instead of waiting for noise to grow
        if (rF[k] > 0 && (was <= 0 || (stirring && Math.abs(psi[k]) < 0.02))){
          const sv = 0.5 * tmpl[k] * smooth(0, 0.35, rF[k]);
          if (Math.abs(sv) > Math.abs(psi[k])) psi[k] = sv;
        }
        if (rF[k] > 0 || Math.abs(psi[k]) > 1e-4){ if (i < x0) x0 = i; if (i > x1) x1 = i; if (j < y0) y0 = j; if (j > y1) y1 = j; }
      }
    }
    if (x1 < 0){
      activeBox = null;
      if (patternLive){ psi.fill(0); spk.n = 0; patternLive = false; newLattice(); return true; }
      return false;
    }
    x0 = Math.max(2, x0 - 3); x1 = Math.min(gx - 3, x1 + 3); y0 = Math.max(2, y0 - 3); y1 = Math.min(gy - 3, y1 + 3);
    activeBox = [x0, x1, y0, y1];
    const k2 = k0 * k0, k4 = k2 * k2, sdt = 0.035, g = 1.25, drain = Math.exp(-dt / 0.07);
    // A moving magnet drags the crests near it along (semi-Lagrangian advection), so a crown rolls and
    // ripples after the cursor rather than fading out behind it and re-forming ahead.
    const sp = Math.hypot(pvx, pvy);
    if (hv > 0.05 && sp > 15 && !still){
      let ux = pvx * 0.45 * dt / cell, uy = pvy * 0.45 * dt / cell;
      const um = Math.hypot(ux, uy);
      if (um > 1.5){ ux *= 1.5 / um; uy *= 1.5 / um; }
      const R2 = Rf * Rf * 0.6;
      for (let j = y0; j <= y1; j++) for (let i = x0, k = j * gx + x0; i <= x1; i++, k++){
        const ex = (i + 0.5) * cell - mx, ey = (j + 0.5) * cell - my, w = hv * Math.exp(-(ex * ex + ey * ey) / R2);
        if (w < 0.02){ lap[k] = psi[k]; continue; }
        const sx = clamp(i - ux * w, 0, gx - 1.001), sy = clamp(j - uy * w, 0, gy - 1.001);
        const ia = Math.floor(sx), ja = Math.floor(sy), fx = sx - ia, fy = sy - ja, a = ja * gx + ia;
        lap[k] = (psi[a] * (1 - fx) + psi[a + 1] * fx) * (1 - fy) + (psi[a + gx] * (1 - fx) + psi[a + gx + 1] * fx) * fy;
      }
      for (let j = y0; j <= y1; j++) for (let i = x0, k = j * gx + x0; i <= x1; i++, k++) psi[k] = lap[k];
    }
    for (let j = y0; j <= y1; j++) for (let i = x0, k = j * gx + x0; i <= x1; i++, k++) prev[k] = psi[k];
    // On a slow CPU the pattern runs slower than real time rather than eating the frame: substeps are
    // capped to a per-frame budget from the measured cost per cell.
    const cells = (x1 - x0 + 3) * (y1 - y0 + 3);
    subs = Math.min(subs, Math.max(3, Math.floor(SUB_BUDGET / (subCost * cells))));
    const t0 = performance.now();
    for (let s = 0; s < subs; s++){
      for (let j = y0 - 1; j <= y1 + 1; j++){
        let k = j * gx + x0 - 1;
        for (let i = x0 - 1; i <= x1 + 1; i++, k++) lap[k] = psi[k - 1] + psi[k + 1] + psi[k - gx] + psi[k + gx] - 4 * psi[k];
      }
      for (let j = y0; j <= y1; j++){
        let k = j * gx + x0;
        for (let i = x0; i <= x1; i++, k++){
          const p = psi[k], l2 = lap[k - 1] + lap[k + 1] + lap[k - gx] + lap[k + gx] - 4 * lap[k];
          let d = rF[k] * p - (k4 * p + 2 * k2 * lap[k] + l2) + g * p * p - p * p * p;
          // seed noise only while the field is changing, so crowns nucleate but a still magnet settles
          if (stirring && rF[k] > 0 && p * p < 0.0025) d += (Math.random() - 0.5) * 0.25;
          psi[k] = p + sdt * d;
        }
      }
    }
    if (subs > 0){ const c = (performance.now() - t0) / (subs * cells); if (c > 0) subCost += (Math.min(c, 2e-4) - subCost) * 0.2; }
    let m = 0, maxD = 0;
    for (let j = y0; j <= y1; j++) for (let i = x0, k = j * gx + x0; i <= x1; i++, k++){
      // Undriven pattern drains fast, so a released forest sinks as one instead of lingering as a
      // subcritical hexagon lattice (with g > 0 it would otherwise persist for a while below threshold).
      if (rF[k] < 0) psi[k] *= pres[k] === 0 ? 0.9 : (still ? 1 : drain);
      const v = psi[k];
      if (v > m) m = v;
      if (Math.abs(v) > 0.05 || Math.abs(prev[k]) > 0.05){ const c = Math.abs(v - prev[k]); if (c > maxD) maxD = c; }
    }
    psiMax = m;
    patternLive = true;
    extractSpikes(holdAmt, mx, my, hv);
    lastMaxD = maxD;
    return maxD > 0.004;
  }

  // Every crest of the pattern becomes one needle cone; its height follows the crest amplitude and the
  // local field, so the spike under the magnet stands tallest with a ring of smaller ones around it.
  // While the magnet only hovers, the label's slot stays flat: crests whose cone would stand on or be
  // seen across the text are lowered away, so the crown parts into a row above and a row below the
  // words, like fluid over a magnet with a slot cut in it. Holding lifts the slot too.
  function extractSpikes(holdAmt, mx, my, hv){
    const Ah = A * (1 - 0.2 * smooth(0.6, 1.2, holdAmt));
    const b = activeBox, cap = spk.x.length;
    let n = 0;
    const sx2 = Math.max(lam * 2.1, 28) ** 2, sy2 = Rf * Rf * 0.7;
    const Rc2 = Rf * Rf * 0.35, forest = 0.55 * smooth(0.3, 1.2, holdAmt);
    // after a hold every crest the magnet is not holding up sinks together as the field lets go
    const sink = holdStill() ? 1 : smooth(0.25, 0.95, holdAmt) + (1 - smooth(0, 0.3, holdAmt));
    const slot = smooth(0, 1, keepLin);
    // a quick look ahead, so glyph cover can hide a letter just before its spike arrives
    const holdAhead = clamp(hold.x + hold.v * 0.1, 0, 1.3), slotAhead = holdStill() ? Math.min(slot, 1 - smooth(SLOT0, SLOT1, holdAhead)) : slot;
    const forestAhead = 0.55 * smooth(0.3, 1.2, holdAhead), hvAhead = clamp(hover.x + hover.v * 0.1, 0, 1);
    const push = (x, y, h, r, hp) => {
      if (n >= cap) return;
      let [lx, ly] = leanAt(x, y, mx, my);
      // A lean past half the base radius makes the flank overhang: the bake's height field can't hold
      // that, and the steeper slope bound would starve the ray march of steps.
      const lm = Math.hypot(lx, ly), lcap = r * 0.5;
      if (lm > lcap){ lx *= lcap / lm; ly *= lcap / lm; }
      // beside the label, a tip may lean along the rows but never over the words
      const T = textBox;
      if (T && x + r > T.x0 - 2 && x - r < T.x1 + 2){
        if (y < T.y0) ly = Math.min(ly, Math.max(0, T.y0 - 0.5 - y));
        else if (y > T.y1) ly = Math.max(ly, 0);
      }
      const c = capInSlot(x, y, h, r, lx, ly), cp = capInSlot(x, y, hp, r, lx, ly);
      let hh = h + (c - h) * slot;
      // a cone half-lowered into the slot would read as a stub or a dimple: past a point it just goes
      if (c < h) hh *= smooth(0.15 * h, 0.4 * h, hh);
      const hpp = Math.max(hh, hp + (cp - hp) * slotAhead);
      if (hh < 0.4 && hpp < 0.4) return;
      // a lowered cone keeps its needle shape rather than turning into a stub
      spk.x[n] = x; spk.y[n] = y; spk.h[n] = hh; spk.r[n] = hh < h ? r * Math.max(hh / h, 0.3) : r; spk.hp[n] = hpp; spk.lx[n] = lx; spk.ly[n] = ly; n++;
    };
    if (b) for (let j = b[2] + 1; j < b[3]; j++) for (let i = b[0] + 1, k = j * gx + i; i < b[1]; i++, k++){
      const c = psi[k];
      if (c < 0.1 || pres[k] <= 0) continue;
      if (c < psi[k - 1] || c <= psi[k + 1] || c < psi[k - gx] || c <= psi[k + gx]
        || c < psi[k - gx - 1] || c < psi[k - gx + 1] || c <= psi[k + gx - 1] || c <= psi[k + gx + 1]) continue;
      const l = psi[k - 1], r = psi[k + 1], u = psi[k - gx], d = psi[k + gx];
      const ddx = l - 2 * c + r, ddy = u - 2 * c + d;
      const ox = ddx < 0 ? clamp(0.5 * (l - r) / ddx, -0.5, 0.5) : 0, oy = ddy < 0 ? clamp(0.5 * (u - d) / ddy, -0.5, 0.5) : 0;
      const x = (i + 0.5 + ox) * cell, y = (j + 0.5 + oy) * cell;
      const ex = x - mx, ey = y - my, g2 = Math.exp(-(ex * ex + ey * ey) / Rc2);
      // a crest is a full needle well before the pattern saturates (real spikes snap up), and a small
      // one is a thin needle, not a wide lump
      const grown = smooth(0.18, 0.75, c), pr = smooth(0.1, 0.8, pres[k]);
      // under a hovering magnet the height falls off sharply sideways, so the rows above and below the
      // label peak right over the cursor and follow it instead of standing as an even fringe
      const gxy = Math.exp(-ex * ex / sx2 - ey * ey / sy2);
      const hovS = 0.62 + (0.16 + 1.08 * gxy - 0.62) * hv;
      let h = Ah * grown * Math.max(hovS, 0.62 + 0.55 * forest) * pr * Math.max(sink, hv * g2);
      // a low, wide crest draws as a dimple-like bump, not a spike: short ones go before they get there
      h *= smooth(0.2 * A, 0.36 * A, h);
      if (h < 1.5) continue;
      const hp = Ah * smooth(0.18, 0.75, c * 1.8) * (0.62 + 0.55 * Math.max(hvAhead * g2, forestAhead)) * pr;
      push(x, y, h, rb * (0.4 + 0.6 * grown) * clamp(h / (0.4 * A), 0.35, 1), hp);
    }
    // The lobe and every magnetised droplet grow their own cones: they move too fast for the pattern
    // to form on them. Taller cones win the spike grid, so these displace any crest they overlap.
    const conesOn = (x, y, rad, amt) => {
      const rr = Math.min(rb, cs * 0.95, rad * 0.75);
      if (amt <= 0.05 || rr < 2) return;
      const hc = A * amt * clamp(rad / (rLobe * 0.9), 0.45, 1.05);
      push(x, y, hc, rr, hc);
      if (rad > rb * 1.4){
        // two smaller cones flanking the first, across the direction to the magnet
        const dx = mx - x, dy = my - y, l = Math.hypot(dx, dy) || 1, o = rad * 0.55;
        push(x - dy / l * o, y + dx / l * o, hc * 0.6, rr * 0.8, hc * 0.6);
        push(x + dy / l * o, y - dx / l * o, hc * 0.6, rr * 0.8, hc * 0.6);
      }
    };
    const lr = rLobe * lobeAmt.x;
    if (lr > 4) conesOn(lobe.x, lobe.y, lr, 0.85 * smooth(0.25, 0.8, lobeAmt.x));
    for (const bl of blobs) conesOn(bl.x, bl.y, bl.r, bl.mag * 0.8);
    spk.n = n;
  }
  // the same lean the bake applies: spikes to the side of the magnet tilt toward it
  function leanAt(x, y, mx, my){
    const dx = mx - x, dy = my - y, d2 = dx * dx + dy * dy, dl = Math.sqrt(d2);
    const f = LEAN * lam * hover.x * smooth(0, Rf * 0.6, dl) * Math.exp(-d2 / (Rf * Rf * 2)) / Math.max(dl, 1);
    return [dx * f, dy * f];
  }
  // the tallest a cone at (x, y) can stand without being seen across the label
  function capInSlot(x, y, h, r, lx, ly){
    const T = textBox;
    if (!T || h <= 0) return h;
    const pad = 1.5;
    if (Math.max(x + r, x + lx) < T.x0 - pad || Math.min(x - r, x + lx) > T.x1 + pad) return h;
    if (y + r * 0.7 < T.y0 && y + ly < T.y0) return h;
    if (y - r * 0.7 > T.y1){ const c = (y + ly - T.y1 - pad) / CAM_K; return c < h * 0.3 ? 0 : Math.min(c, h); }
    return 0;
  }

  // A glyph hides when the cones actually seen in front of it cover a quarter of it. Hiding is instant
  // (and uses where spikes are about to be), showing takes ~200 ms, so no half-faded letter is
  // ever left standing between spikes.
  const GS = 4;
  function stepGlyphs(dt){
    let moving = false;
    const aShow = 1 - Math.exp(-dt / 0.07);
    for (let gi = 0; gi < glyphs.length; gi++){
      const g = glyphs[gi];
      if (!g) continue;
      const x0 = g.x0 + 1, x1 = g.x1 - 1, y0 = g.y0 + 1, y1 = g.y1 - 1;
      let hit = 0;
      if (x1 > x0 && y1 > y0){
        // cones whose screen footprint can reach the glyph at all
        let cand = 0;
        for (let s = 0; s < spk.n && cand < 32; s++){
          const h = spk.hp[s], r = spk.r[s], x = spk.x[s], y = spk.y[s];
          if (h < 0.4) continue;
          if (Math.max(x + r, x + spk.lx[s]) < x0 || Math.min(x - r, x + spk.lx[s]) > x1) continue;
          if (y + r < y0 || Math.min(y - r, y + spk.ly[s] - CAM_K * h) > y1) continue;
          candList[cand++] = s;
        }
        if (cand) for (let a = 0; a < GS; a++) for (let c = 0; c < GS; c++){
          const qx = x0 + (x1 - x0) * (a + 0.5) / GS, qy = y0 + (y1 - y0) * (c + 0.5) / GS;
          let inside = false;
          for (let m = 0; m < cand && !inside; m++){
            const s = candList[m], h = spk.hp[s], r = spk.r[s];
            // the cone's silhouette: its cross-section at height z, shifted up the screen by CAM_K * z
            for (let f = 0; f < 5; f++){
              const zf = f * 0.2, rad = r * (1 - Math.sqrt(zf));
              const ex = qx - spk.x[s] - spk.lx[s] * zf, ey = qy - spk.y[s] - spk.ly[s] * zf + CAM_K * h * zf;
              if (ex * ex + ey * ey < rad * rad){ inside = true; break; }
            }
          }
          if (inside) hit++;
        }
      }
      const frac = hit / (GS * GS);
      if (frac > 0.25) glyphCov[gi] = 1; else if (frac < 0.12) glyphCov[gi] = 0;
    }
    // While a forest rises or subsides the label goes and comes back as a whole word: never a word with
    // bare holes in it, never one grey letter at a time.
    let nCov = 0, nG = 0;
    for (let gi = 0; gi < glyphs.length; gi++) if (glyphs[gi]){ nG++; nCov += glyphCov[gi]; }
    const swallowed = (held() || hold.x > 0.1) && nCov > 0;
    for (let gi = 0; gi < glyphs.length; gi++){
      if (!glyphs[gi]) continue;
      const target = swallowed ? 0 : 1 - glyphCov[gi];
      const v = target < glyphVis[gi] ? target : glyphVis[gi] + (target - glyphVis[gi]) * aShow;
      if (Math.abs(v - target) > 0.003){ moving = true; glyphVis[gi] = v; } else glyphVis[gi] = target;
    }
    return moving;
  }
  const candList = new Int32Array(32);

  function step(dt){
    simT += dt;
    let moving = false;
    if (pointerTouch && !press && simT > touchMagnetUntil){ pointer = null; pointerTouch = false; }
    if (pointer){
      const [nx, ny] = stagePt(pointer.x, pointer.y);
      if (ppx > -1e3){ const a = Math.min(1, dt * 20); pvx += ((nx - ppx) / Math.max(dt, 1e-3) - pvx) * a; pvy += ((ny - ppy) / Math.max(dt, 1e-3) - pvy) * a; }
      if (Math.abs(nx - px) > 0.5 || Math.abs(ny - py) > 0.5) lastStir = simT;
      ppx = px = nx; ppy = py = ny;
    } else { pvx = pvy = 0; ppx = -1e4; }
    const pressing = held();
    // the forest only reaches the words ~0.17 s into a hold, so a quick click never loses its label
    if (holdStill()) keepLin = Math.min(keepLin, 1 - smooth(SLOT0, SLOT1, hold.x));
    else if (hold.x < 0.45 && keepLin < 1){ keepLin = Math.min(1, keepLin + dt / 0.22); lastStir = simT; }
    const [mx, my] = magnet();
    const magNear = (pointer || pressing) && boxSd(mx, my) < Rf * 1.6;
    const hoverT = magNear && (pressing || simT > hoverHoldUntil) && motionOK() ? 1 : 0;
    const holdT = pressing && holdStill() && motionOK() ? 1.25 : 0;
    // a fast pass over the pool leaves a line of spikes behind it for a moment
    if (hoverT && !press && Math.hypot(pvx, pvy) > 250 && boxSd(mx, my) < bh * 0.3){
      const t = trail[trail.length - 1];
      if (!t || Math.hypot(t.x - mx, t.y - my) > lam * 0.8){ trail.push({ t: simT, x: mx, y: my }); if (trail.length > 16) trail.shift(); }
    }
    const sm = [spring(hover, hoverT, 14, 0.9, dt),
      // holding ramps the field up over ~0.6 s: a crown under the finger first, then the whole forest
      spring(hold, holdT, pressing ? 5.5 : 10, pressing ? 1 : 0.9, dt),
      spring(flat, 1, 15, 0.25, dt), spring(spread, 0, 11, 0.25, dt), stepBlobs(dt)];
    if (sm[0] || sm[1] || sm[4] || trail.length) lastStir = simT;
    moving = sm.some(Boolean);
    moving |= stepPattern(dt);
    moving |= stepGlyphs(dt);
    for (const r of ripples) r.t += dt;
    while (ripples.length && ripples[0].t > 1.8) ripples.shift();
    return moving || ripples.length > 0 || (pressing && motionOK());
  }

  let spkOrder = new Uint32Array(0);
  const ripData = new Float32Array(16), blobData = new Float32Array(MAX_BLOBS * 4), glyphData = new Float32Array(MAX_GLYPHS);
  function setShape(u){
    gl.uniform4f(u.uBox, bcx, bcy, bw / 2, bh / 2);
    gl.uniform1f(u.uRad, radius);
    gl.uniform1f(u.uSpread, spread.x);
    blobData.fill(0);
    const lr = rLobe * lobeAmt.x;
    if (lr > 0.5) blobData.set([lobe.x, lobe.y, lr, 0], 0);
    blobs.forEach((b, i) => blobData.set([b.x, b.y, b.r, 0], (i + 1) * 4));
    gl.uniform4fv(u.uBlob, blobData);
    gl.uniform4f(u.uNeck, anchorX, anchorY, neckR(), 0);
  }
  function render(){
    // spike grid: at most one crest per cell (the taller wins; a close pair reads as one spike anyway)
    // tallest first; a crest closer than 0.6 spacing to one already placed is the same spike splitting
    // or merging, and two cones there would read as a forked tip
    spkGrid.fill(0); leanGrid.fill(0);
    let hmax = 0, lmax = 0, bx0 = Infinity, by0 = Infinity, bx1 = -Infinity, by1 = -Infinity;
    const order = spkOrder.length >= spk.n ? spkOrder : (spkOrder = new Uint32Array(spk.x.length));
    for (let s = 0; s < spk.n; s++) order[s] = s;
    const ord = order.subarray(0, spk.n).sort((a, b) => spk.h[b] - spk.h[a]);
    const minD2 = (cs / 0.7 * 0.6) ** 2;
    for (const s of ord){
      // a cone lowered to nothing is kept only for the glyph look-ahead; it has no shape to draw
      if (spk.h[s] <= 0 || spk.r[s] <= 0) break;
      const x = spk.x[s], y = spk.y[s];
      const ci = Math.floor(x / cs), cj = Math.floor(y / cs);
      if (ci < 0 || cj < 0 || ci >= sgx || cj >= sgy) continue;
      let clash = false;
      for (let dj = -1; dj <= 1 && !clash; dj++) for (let di = -1; di <= 1; di++){
        const ni = ci + di, nj = cj + dj;
        if (ni < 0 || nj < 0 || ni >= sgx || nj >= sgy) continue;
        const o = (nj * sgx + ni) * 4;
        if (spkGrid[o + 2] > 0 && (spkGrid[o] - x) ** 2 + (spkGrid[o + 1] - y) ** 2 < minD2){ clash = true; break; }
      }
      const o = (cj * sgx + ci) * 4;
      if (clash || spkGrid[o + 2] > 0) continue;
      spkGrid[o] = x; spkGrid[o + 1] = y; spkGrid[o + 2] = spk.h[s]; spkGrid[o + 3] = spk.r[s];
      leanGrid[o / 2] = spk.lx[s]; leanGrid[o / 2 + 1] = spk.ly[s];
      lmax = Math.max(lmax, Math.hypot(spk.lx[s], spk.ly[s]) / spk.r[s]);
      if (spk.h[s] / spk.r[s] > hmax) hmax = spk.h[s] / spk.r[s];
      const rr = spk.r[s] + 1;
      bx0 = Math.min(bx0, x - rr, x + spk.lx[s] - 1); bx1 = Math.max(bx1, x + rr, x + spk.lx[s] + 1);
      by0 = Math.min(by0, y - rr, y + spk.ly[s] - 1); by1 = Math.max(by1, y + rr, y + spk.ly[s] + 1);
    }
    gl.bindVertexArray(vao);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, spkTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, sgx, sgy, gl.RGBA, gl.FLOAT, spkGrid);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, leanTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, sgx, sgy, gl.RG, gl.FLOAT, leanGrid);
    const [mx, my] = magnet();

    // pass 1: bake the height field
    let u = P.bake.u;
    gl.bindFramebuffer(gl.FRAMEBUFFER, htFb);
    gl.viewport(0, 0, htW, htH);
    gl.useProgram(P.bake.p);
    setShape(u);
    gl.uniform1i(u.uSpk, 0); gl.uniform1i(u.uLean, 3);
    gl.uniform1f(u.uCs, cs); gl.uniform1f(u.uBake, htW / stageW);
    gl.uniform1f(u.uH, H); gl.uniform1f(u.uMen, clamp(bh * 0.22, 7, 14));
    gl.uniform1f(u.uFlat, flat.x);
    const mound = H * 1.2 * hover.x * (1 - 0.5 * clamp(hold.x, 0, 1)), swell = H * 0.45 * clamp(hold.x, 0, 1.3);
    gl.uniform1f(u.uMound, mound);
    gl.uniform1f(u.uSwell, swell);
    gl.uniform1f(u.uSmK, rb * 0.3);
    gl.uniform1f(u.uRf, Rf);
    gl.uniform1f(u.uRt, 0.9);
    gl.uniform2f(u.uPtr, mx, my);
    ripData.fill(0);
    ripples.forEach((r, i) => ripData.set([r.x, r.y, r.t, r.amp], i * 4));
    gl.uniform4fv(u.uRip, ripData);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // pass 2: march and shade
    u = P.main.u;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(P.main.p);
    setShape(u);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, htTex); gl.uniform1i(u.uHt, 1);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, textTex); gl.uniform1i(u.uText, 2);
    gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, vlyTex); gl.uniform1i(u.uVly, 4);
    gl.uniform2f(u.uStage, stageW, stageH);
    gl.uniform1f(u.uDpr, canvas.width / stageW);
    gl.uniform1f(u.uCanvasH, canvas.height);
    gl.uniform1f(u.uBake, htW / stageW);
    const zPool = H * 2 + mound + swell + 4;
    gl.uniform1f(u.uZmax, zPool + A * 1.3);
    gl.uniform1f(u.uZpool, zPool);
    gl.uniform1f(u.uSlopeLo, 1 + CAM_K * 3 * 1.15);
    gl.uniform1f(u.uMen, clamp(bh * 0.22, 7, 14));
    if (bx1 > bx0) gl.uniform4f(u.uSpkBox, bx0, by0, bx1, by1); else gl.uniform4f(u.uSpkBox, 1e6, 1e6, -1e6, -1e6);
    // the steepest slope anywhere: cone flanks (sheared by the lean) or the meniscus / ripples
    const slope = Math.max(hmax * TIP_P * (1 + 0.8 * lmax), 3);
    gl.uniform1f(u.uSlopeK, 1 + CAM_K * slope * 1.15);
    gl.uniform1f(u.uPress, pressVis);
    // the overhead softbox that draws the meniscus line at rest would smear grey across a spike forest's valleys
    gl.uniform1f(u.uKey, 1 - 0.65 * clamp(Math.max(hover.x, hold.x), 0, 1));
    glyphData.set(glyphVis);
    gl.uniform4fv(u.uGlyph, glyphData);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  function advance(dt){
    frameCount++;
    place();
    const moving = step(dt);
    const busy = moving || simT < activeUntil;
    if (busy || dirty){ render(); dirty = false; }
    return busy;
  }
  function frame(now){
    raf = 0;
    // A rAF timestamp is the frame's start, which can precede the performance.now() taken when wake()
    // scheduled it; a negative dt would run every spring and decay backwards and blow the pattern up.
    const dt = clamp((now - last) / 1000, 0, 1 / 30); last = Math.max(last, now);
    if (manual || lost || gl.isContextLost()) return;
    inFrame = true;
    let busy = false;
    try { busy = advance(dt); } finally { inFrame = false; }
    if (busy) raf = requestAnimationFrame(frame);
  }

  const ro = new ResizeObserver(() => relayout());
  ro.observe(button);
  if (document.body) ro.observe(document.body);
  const onResize = () => relayout();
  const onScroll = () => place();
  addEventListener('resize', onResize);
  document.addEventListener('scroll', onScroll, { capture: true, passive: true });
  // the label follows text or style changes made to the button while it is attached
  const mo = new MutationObserver(() => { if (!lost && fontKey() !== labelKey){ buildText(); wake(0.1); } });
  mo.observe(button, { childList: true, characterData: true, subtree: true, attributes: true, attributeFilter: ['class', 'style'] });
  const onFonts = () => { if (!lost){ buildText(); wake(0.1); } };
  document.fonts?.addEventListener?.('loadingdone', onFonts);
  const onMotionPref = () => { dirty = true; wake(0.5); };
  reduceMotion.addEventListener?.('change', onMotionPref);

  // A lost context shows the plain button until the GPU comes back; then everything is rebuilt in place.
  function onLost(e){
    e.preventDefault();
    lost = true;
    cancelAnimationFrame(raf); raf = 0;
    press = null; keyHold = false; blobs.length = 0; ripples.length = 0; trail.length = 0;
    showButton();
    canvas.style.visibility = 'hidden';
  }
  function onRestored(){
    try { initGL(); } catch (err) { console.warn('ElementalFX ferro: could not rebuild after context loss', err); return; }
    lost = false;
    hideButton();
    canvas.style.visibility = '';
    stageW = stageH = 0;
    relayout();
    dirty = true;
    wake(0.2);
  }
  canvas.addEventListener('webglcontextlost', onLost);
  canvas.addEventListener('webglcontextrestored', onRestored);

  hideButton();
  document.body.appendChild(canvas);
  relayout();
  if (document.fonts && document.fonts.status !== 'loaded') document.fonts.ready.then(onFonts);

  return {
    supported: true,
    get _frames(){ return frameCount; },
    get _sleeping(){ return !raf; },
    get _psiMax(){ return psiMax; },
    get _blobs(){ return blobs.length; },
    get _spikes(){ return spk.n; },
    get _glyphVis(){ return Array.from(glyphVis.slice(0, glyphs.length)); },
    get _stage(){ return { stageW, stageH, offX, offY, bw, bh, lastMaxD, lastStir, simT }; },
    detach(){
      cancelAnimationFrame(raf); raf = 0; manual = true;
      ro.disconnect(); mo.disconnect();
      removeEventListener('resize', onResize); document.removeEventListener('scroll', onScroll, { capture: true });
      document.fonts?.removeEventListener?.('loadingdone', onFonts);
      reduceMotion.removeEventListener?.('change', onMotionPref);
      removeEventListener('pointermove', onWinMove); document.documentElement.removeEventListener('pointerleave', onWinLeave);
      button.removeEventListener('pointerdown', onDown); button.removeEventListener('pointermove', onMove);
      button.removeEventListener('pointerup', onUp); button.removeEventListener('pointercancel', onCancel);
      button.removeEventListener('lostpointercapture', onCancel); button.removeEventListener('click', onClick, true);
      button.removeEventListener('keydown', onKeyDown); button.removeEventListener('keyup', onKeyUp); button.removeEventListener('blur', endKey);
      canvas.removeEventListener('webglcontextlost', onLost); canvas.removeEventListener('webglcontextrestored', onRestored);
      showButton();
      canvas.remove();
      gl.getExtension('WEBGL_lose_context')?.loseContext();
    },
    _manual(on){ manual = on; if (on){ cancelAnimationFrame(raf); raf = 0; } else wake(0.1); },
    _advance(sec, fps = 60){ for (let i = 0, n = Math.round(sec * fps); i < n; i++) advance(1 / fps); },
  };
}

window.ElementalFX = Object.assign(window.ElementalFX || {}, { attachFerro, ferroAudio });
})();
