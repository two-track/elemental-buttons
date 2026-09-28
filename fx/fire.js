// Fire button: turns an existing <button> into a match-striker strip you strike to submit.
//
//   <script src="fx/fire.js"></script>
//   const fx = ElementalFX.attachFire(document.querySelector('#deploy'), { pending: () => fetch('/deploy') });
//
// A classic script (not a module) so pages work straight from file:// too.
//
// The <button> stays the real control (focus, keyboard, form submission, click handlers); this only
// draws it, in a transparent canvas laid over the page around it. A click strikes it: a flare bursts
// from the finger with sparks, then a flame settles along the top edge and burns while the action is
// pending (the button is aria-busy and swallows clicks meanwhile). Success snuffs it into a curl of
// smoke and briefly brands the label; failure chokes it into soot. Air pushed by the pointer bends the
// flame; a deliberate gust across it can blow it out (a cancel). The flame is a real 2D fluid simulation:
// velocity with pressure projection and vorticity confinement, temperature-driven buoyancy, fuel that
// burns, and soot that becomes smoke. If WebGL2 with float render targets is unavailable, the button
// is left untouched.
//
// It can also burn the whole strip away: the flame catches the strip itself, and a glowing front of
// embers eats down through it (the strip is simulated as material with fuel, char and heat, and feeds
// the flames riding that front), charring the label, opening holes, shedding ash and sparks, until
// nothing is left; then the strip grows back, fresh, from its base, and is clickable again.
//
// Options:
//   pending: (event, { signal }) => Promise   called on each click; the flame burns until it settles
//            (resolve = success, reject = failure). Without it a click burns ~1.6 s and then succeeds.
//            The button stays aria-busy and swallows clicks until the promise settles, even if the
//            flame was blown out, so a request can never be submitted twice.
//   onCancel: () => void          called when the flame is blown out, after signal is aborted.
//   cancelable: true              let a gust blow the flame out (a cancel) without an onCancel; for a
//            pending that honours its AbortSignal. Implied by onCancel or a pending declaring two
//            parameters. When nothing can be cancelled a gust only bends the flame.
//   scale: 'match' | 'candle' (or 'small') | 'campfire' | 'bonfire' (default) | 0..1
//            the size and character of the fire, from a match's calm teardrop flame to a fire along the
//            whole edge. It changes the physics (how much of the edge feeds it and how hard, buoyancy,
//            turbulence, the room and resolution it is simulated at), the sparks and the sound.
//   consume: 'never' (default) | 'onSuccess' | 'always'
//            'onSuccess' burns the strip away when the action succeeds, instead of snuffing it;
//            'always' lets every strike catch and burn it away (a failure still reads afterwards).
//            Whatever this is, pressing and holding a burning button (pointer, or Space / Enter held for
//            a moment) fans the fire until the strip catches (let go sooner and it settles back), and
//            consume() burns it away on demand. With prefers-reduced-motion there is no fanning; the
//            strip still burns away through consume and the consume option, faster and without embers.
//   successText (default "Done"), failText (default "Try again"): what the label reads afterwards.
//   consumedText (default "Done"): announced to screen readers once a burnt-away strip is back.
//   onConsumed: (outcome) => void   called once a burnt-away strip has grown back and its action has settled
//            ('success' or 'fail'): the moment to change layout around it, which mid-burn would move it.
//   zIndex: of the stage canvas (default 10).   sound: false to stay silent.
//
// Returned: { supported, state, scale, detach(), refresh(), burn(), succeed(), fail(), consume(),
// setScale(scale) }. burn() lights it without a click and keeps it burning until succeed() or fail().
// state is rest | burning | success | fail | smoulder (blown out, request still settling) | out |
// consume | regrow | settling (grown back, request still settling). A disabled button is drawn dimmed,
// and outcomes are announced through a polite live region.

(() => {
'use strict';

const MIN_BURN = 0.7;          // s a strike burns at least, however fast the action settles
const DEFAULT_BURN = 1.6;      // s without opts.pending
const GUTTER = 0.45;           // s the flame gutters before it snuffs out
const SPUTTER = 0.85;          // s a failing flame sputters before it dies
const SHOW = 1.25;             // s the success / failure label stays up
const BLOW_SPEED = 3000;       // CSS px/s: only a gust this fast across the flame can blow it out
const BLOW_GRACE = 600;        // ms after a click, or after the pointer left the button, when nothing blows it out
const JACOBI = 28;
const PW = 128, PH = 8, NPART = PW * PH;
const REGROW = 0.95;           // s the strip takes to grow back after burning away
const ASH_HOLD = 0.2;          // s between the last of the strip burning out and it growing back
const HEAL = 0.35;             // s scorching from a fire that never caught takes to clear once it is out
const TIGN = 0.34;             // ignition temperature (sim units; 1 is a hot flame)

// ---------------------------------------------------------------- shaders
const VS = `#version 300 es
void main(){ vec2 p = vec2(float((gl_VertexID<<1)&2), float(gl_VertexID&2)); gl_Position = vec4(p*2.-1., 0., 1.); }`;

const HEAD = `#version 300 es
precision highp float;
precision highp sampler2D;
#define TIGN ${TIGN.toFixed(3)}
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*.1031); p3 += dot(p3, p3.yzx+33.33); return fract((p3.x+p3.y)*p3.z); }
vec2 hash22(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*vec3(.1031,.1030,.0973)); p3 += dot(p3, p3.yzx+33.33); return fract((p3.xx+p3.yz)*p3.zy); }
float vnoise(vec2 p){
  vec2 i = floor(p), f = fract(p);
  vec2 u = f*f*(3.-2.*f);
  return mix(mix(hash12(i), hash12(i+vec2(1,0)), u.x), mix(hash12(i+vec2(0,1)), hash12(i+vec2(1,1)), u.x), u.y);
}
// How much harder the fire burns for what the strip still has to feed it: the strip's coarsest mip is
// the mean of its volatiles, so no readback is needed. a is 0 unless the strip is burning away.
float feedOf(sampler2D m, float lod, float a){ return a > 0. ? 1. + a * min(1., textureLod(m, vec2(0.5), lod).a / 0.48) : 1.; }
float sdRR(vec2 p, vec2 b, float r){ vec2 d = abs(p) - b + r; return length(max(d, 0.)) + min(max(d.x, d.y), 0.) - r; }
// Light given off by the gas: incandescent soot glowing on a blackbody-like ramp (red -> orange ->
// yellow-white), plus the blue of the reaction zone where fuel is actually burning. Soot has not formed
// yet where the fuel is still rich and has only just lit, so the base of the flame shows the blue.
vec3 flameEmit(vec4 s){
  float T = s.r, F = s.g;
  float t = max(T - 0.26, 0.);
  vec3 c = vec3(1.00, 0.20, 0.03) * (t*t*2.6)
         + vec3(0.30, 0.90, 0.12) * (t*t*t*1.3)
         + vec3(0.30, 0.55, 0.75) * (t*t*t*t*2.4);
  float premix = smoothstep(0.04, 0.22, F) * (1. - smoothstep(0.62, 0.95, T));
  c *= 1. - 0.9 * premix;
  float burning = min(F * 4., 1.) * smoothstep(TIGN, TIGN + 0.12, T) * (1. - smoothstep(0.75, 1.15, T));
  c += vec3(0.16, 0.30, 1.0) * burning * 1.2;
  return c;
}
`;

// ---- fluid (velocity grid, cells and cells/s, y up)
const FS_ADVECT_V = HEAD + `
uniform sampler2D uVel;
uniform vec2 uRes;
uniform float uDt, uDamp;
out vec4 o;
void main(){
  vec2 p = gl_FragCoord.xy;
  vec2 v0 = texelFetch(uVel, ivec2(p), 0).xy;
  vec2 v1 = texture(uVel, (p - 0.5*uDt*v0) / uRes).xy;
  vec2 v = texture(uVel, (p - uDt*v1) / uRes).xy;
  o = vec4(v * uDamp, 0., 1.);
}`;

const FS_CURL = HEAD + `
uniform sampler2D uVel;
out vec4 o;
vec2 V(ivec2 p){ return texelFetch(uVel, clamp(p, ivec2(0), textureSize(uVel, 0) - 1), 0).xy; }
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  o = vec4(0.5*((V(p + ivec2(1,0)).y - V(p - ivec2(1,0)).y) - (V(p + ivec2(0,1)).x - V(p - ivec2(0,1)).x)), 0., 0., 1.);
}`;

// Buoyancy, vorticity confinement, pointer wind.
const FS_FORCE = HEAD + `
uniform sampler2D uVel, uCurl, uS, uM;
uniform vec2 uRes;
uniform float uDt, uBuoy, uWeight, uEps, uJit, uSeed, uBreeze, uBreezeY, uFeedA, uTopLod, uCalm, uLift;
uniform int uNW;
uniform vec4 uW[8];       // x, y, radius (cells), strength
uniform vec2 uWV[8];      // air velocity there (cells/s)
out vec4 o;
float C(ivec2 p){ return texelFetch(uCurl, clamp(p, ivec2(0), textureSize(uCurl, 0) - 1), 0).x; }
void main(){
  ivec2 ip = ivec2(gl_FragCoord.xy);
  vec2 p = gl_FragCoord.xy;
  vec2 v = texelFetch(uVel, ip, 0).xy;
  vec4 s = texture(uS, p / uRes);
  float c = C(ip);
  vec2 g = 0.5*vec2(abs(C(ip + ivec2(1,0))) - abs(C(ip - ivec2(1,0))), abs(C(ip + ivec2(0,1))) - abs(C(ip - ivec2(0,1))));
  vec2 N = g / (length(g) + 1e-3);
  float heat = clamp(s.r, 0., 1.4);
  // A small flame's smoke (uCalm 1) is a thin thread still warm from the flame: it rises and is not curled
  // up into eddies of its own, which on a thread that fine read as rings hanging in the air.
  vec2 f = uEps * (0.35 + heat) * vec2(N.y, -N.x) * c * mix(1., smoothstep(0.08, 0.4, heat), uCalm);
  f.y += uBuoy * (0.6 + 0.4 * feedOf(uM, uTopLod, uFeedA)) * s.r - uWeight * s.b * (1. - uCalm) + uLift * uCalm * min(s.b, 1.);
  // the hot gas is never perfectly still: a little stirring for the plume's instability to feed on
  vec2 j = hash22(floor(p * 0.5) + uSeed) - 0.5;
  f += uJit * j * smoothstep(0.05, 0.5, s.r + s.b * (1. - 0.9 * uCalm));
  f.x += uBreeze * smoothstep(uBreezeY, uBreezeY + 20., p.y);
  v += f * uDt;
  for (int i = 0; i < 8; i++){
    if (i >= uNW) break;
    vec2 d = p - uW[i].xy;
    float w = exp(-dot(d, d) / (uW[i].z * uW[i].z)) * uW[i].w;
    // air is dragged along with the pointer, never pulled against it
    float sp = length(uWV[i]);
    if (sp > 1e-3){ vec2 dir = uWV[i] / sp; v += dir * max(sp - dot(v, dir), 0.) * clamp(w, 0., 1.); }
  }
  o = vec4(v, 0., 1.);
}`;

// Divergence, less the expansion of gas where fuel is burning (that is what makes the strike whoomp).
const FS_DIV = HEAD + `
uniform sampler2D uVel, uS;
uniform vec2 uRes;
uniform float uExpand, uKBurn;
out vec4 o;
vec2 V(ivec2 p){ return texelFetch(uVel, clamp(p, ivec2(0), textureSize(uVel, 0) - 1), 0).xy; }
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  float d = 0.5*(V(p + ivec2(1,0)).x - V(p - ivec2(1,0)).x + V(p + ivec2(0,1)).y - V(p - ivec2(0,1)).y);
  vec4 s = texture(uS, gl_FragCoord.xy / uRes);
  float burn = s.g * uKBurn * smoothstep(TIGN, TIGN + 0.25, s.r);
  o = vec4(d - uExpand * burn, 0., 0., 1.);
}`;

// Open boundaries: pressure is zero outside the stage, so gas leaves freely.
const FS_JACOBI = HEAD + `
uniform sampler2D uP, uDiv;
out vec4 o;
float Pr(ivec2 p){ ivec2 m = textureSize(uP, 0); return any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, m)) ? 0. : texelFetch(uP, p, 0).x; }
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  o = vec4((Pr(p + ivec2(1,0)) + Pr(p - ivec2(1,0)) + Pr(p + ivec2(0,1)) + Pr(p - ivec2(0,1)) - texelFetch(uDiv, p, 0).x) * 0.25, 0., 0., 1.);
}`;

const FS_GRAD = HEAD + `
uniform sampler2D uP, uVel;
out vec4 o;
float Pr(ivec2 p){ ivec2 m = textureSize(uP, 0); return any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, m)) ? 0. : texelFetch(uP, p, 0).x; }
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec2 v = texelFetch(uVel, p, 0).xy - 0.5*vec2(Pr(p + ivec2(1,0)) - Pr(p - ivec2(1,0)), Pr(p + ivec2(0,1)) - Pr(p - ivec2(0,1)));
  o = vec4(v, 0., 1.);
}`;

// ---- scalars at twice the velocity resolution: r = temperature, g = fuel, b = soot.
// Advected with clamped Catmull-Rom so the flame keeps its edges, then sources, combustion and cooling.
const FS_ADVECT_S = HEAD + `
uniform sampler2D uS, uVel;
uniform vec2 uRes;
uniform float uDt, uVtoD, uTime;
uniform float uKBurn, uHeat, uSootY, uKOx, uCool, uRad, uSootDecay, uSootH, uFuelDecay, uQuench;
uniform vec4 uWick;       // x0, x1, y, half thickness (cells)
uniform vec4 uWick2;      // fuel rate, pilot heat, noise scale, noise speed
uniform float uSteady;    // 0: a fire's ragged, gusting fuel; 1: a match's even feed
uniform sampler2D uM;     // the strip's material: r solid, g char, b heat, a volatiles
uniform vec4 uMRect;      // offX, offY, w, h of the strip (stage CSS px, y down)
uniform vec4 uMat;        // CSS px per cell, stage height, fuel rate, pilot heat
uniform float uMatOn, uMLift, uFeedA, uTopLod;
uniform int uNS;
uniform vec4 uSp[8];      // x, y, rx, ry (cells)
uniform vec4 uSa[8];      // dT, dFuel, dSoot, max T
out vec4 o;
vec4 cubic(vec2 q){
  vec2 f = q - 0.5, i = floor(f), t = f - i, t2 = t*t, t3 = t2*t;
  vec2 w0 = -0.5*t3 + t2 - 0.5*t, w1 = 1.5*t3 - 2.5*t2 + 1., w2 = -1.5*t3 + 2.*t2 + 0.5*t, w3 = 0.5*t3 - 0.5*t2;
  float wx[4] = float[4](w0.x, w1.x, w2.x, w3.x), wy[4] = float[4](w0.y, w1.y, w2.y, w3.y);
  ivec2 b = ivec2(i) - 1, m = textureSize(uS, 0) - 1;
  vec4 acc = vec4(0.), lo = vec4(1e9), hi = vec4(-1e9);
  for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++){
    // outside the stage is plain cool air
    ivec2 c = b + ivec2(x, y);
    vec4 v = any(lessThan(c, ivec2(0))) || any(greaterThan(c, m)) ? vec4(0.) : texelFetch(uS, c, 0);
    acc += v * wx[x] * wy[y];
    if ((x == 1 || x == 2) && (y == 1 || y == 2)){ lo = min(lo, v); hi = max(hi, v); }
  }
  return clamp(acc, lo, hi);
}
void main(){
  vec2 p = gl_FragCoord.xy;
  vec2 v0 = texture(uVel, p / uRes).xy * uVtoD;
  vec2 v1 = texture(uVel, (p - 0.5*uDt*v0) / uRes).xy * uVtoD;
  vec4 s = cubic(p - uDt*v1);

  // where the strip itself burns, it gives off fuel and keeps it alight; where the top edge is gone,
  // there is nothing left for the flame along it to feed on
  vec2 bl = vec2(p.x * uMat.x - uMRect.x, uMat.y - p.y * uMat.x - uMRect.y);
  float topLeft = 1.;
  if (uMatOn > 0.5){
    // The gas a burning patch gives off rises off it, so its flames stand just above the glowing edge
    // rather than on top of it. Gas comes off in proportion to the volatiles still there, so it is the
    // leading edge of the front that flames; the spent char behind it only glows, and must not keep a pale
    // sheet of flame over it. Sampled along the path this gas travelled during the step, like the edge below.
    float burn = 0., pilot = 0.;
    for (int k = 0; k < 4; k++){
      vec2 q = p - v1 * uDt * (float(k) + 0.5) / 4.;
      vec2 bq = vec2(q.x * uMat.x - uMRect.x, uMat.y - q.y * uMat.x - uMRect.y);
      vec2 mu = vec2(bq.x / uMRect.z, 1. - (bq.y + uMLift) / uMRect.w);
      if (all(greaterThan(mu, vec2(0.))) && all(lessThan(mu, vec2(1.)))){
        vec4 m = textureLod(uM, mu, 0.);
        // hotter than anything the gas alone can preheat it to (see FS_MAT): it is actually alight
        float hot = smoothstep(0.6, 0.9, m.b);
        burn += 0.25 * hot * smoothstep(0.02, 0.6, m.a);
        pilot = max(pilot, hot * smoothstep(0.25, 0.6, m.a));
      }
    }
    if (burn + pilot > 0.){
      float n = vnoise(p * 0.21 + vec2(0., uTime * 4.));
      // fed unevenly along the front, drifting, so it throws up separate tongues with the char showing
      // between them instead of one sheet of flame
      float tongue = smoothstep(0.3, 0.75, vnoise(vec2(bl.x * 0.045, uTime * 0.9)) * 0.7 + vnoise(vec2(bl.x * 0.13 + 5., uTime * 2.1)) * 0.3);
      s.g += uDt * uMat.z * feedOf(uM, uTopLod, uFeedA) * burn * (0.3 + 1.4 * n) * (0.12 + 1.75 * tongue);
      s.r = max(s.r, uMat.w * pilot * (0.75 + 0.25 * tongue));
    }
    topLeft = smoothstep(0.03, 0.45, textureLod(uM, vec2(clamp(bl.x / uMRect.z, 0., 1.), 1. - 2.5 / uMRect.w), 0.).a);
  }

  // the burning edge: fuel boils off it unevenly, and the edge itself keeps what is above it alight
  // Averaged along the path this gas travelled during the step: stamped once per step, a thin source
  // would leave one stripe per step wherever the gas moves faster than the source is thick.
  if (uWick2.x > 0. || uWick2.y > 0.){
    float prof = 0., hot = 0.;
    for (int k = 0; k < 4; k++){
      float dy = (p.y - v1.y * uDt * (float(k) + 0.5) / 4. - uWick.z) / uWick.w;
      prof += exp(-dy*dy) * 0.25;
      hot = max(hot, exp(-dy*dy*0.6));
    }
    if (prof > 1e-4 || hot > 1e-4){
      // fed less towards the ends, so the flame narrows there instead of fanning past the corners
      float taper = max(2.*uWick.w, 0.22*(uWick.y - uWick.x));
      float edge = smoothstep(uWick.x - uWick.w, uWick.x + taper, p.x) * smoothstep(uWick.y + uWick.w, uWick.y - taper, p.x) * topLeft;
      float n1 = vnoise(vec2(p.x * uWick2.z, uTime * uWick2.w));
      float n2 = vnoise(vec2(p.x * uWick2.z * 2.9 + 11.3, uTime * uWick2.w * 1.9 + 3.7));
      s.g += uDt * uWick2.x * edge * prof * mix(0.25 + 1.3*n1*n1 + 0.45*n2, 0.72 + 0.12*n1, uSteady);
      // the pilot holds its heat right to the ends, or a short edge (a match) would never light at all
      float edgeP = smoothstep(uWick.x - uWick.w, uWick.x, p.x) * smoothstep(uWick.y + uWick.w, uWick.y, p.x) * topLeft;
      s.r = max(s.r, uWick2.y * max(edge, edgeP) * hot);
    }
  }
  for (int i = 0; i < 8; i++){
    if (i >= uNS) break;
    vec2 d = (p - uSp[i].xy) / uSp[i].zw;
    float w = exp(-dot(d, d));
    s.r = max(s.r, min(s.r + uSa[i].x * w, uSa[i].w));
    s.g += uSa[i].y * w;
    s.b += uSa[i].z * w;
  }

  float T = s.r, F = s.g, K = s.b;
  float b = F * (1. - exp(-uKBurn * smoothstep(TIGN, TIGN + 0.25, T) * uDt));
  F -= b; T += b * uHeat; K += b * uSootY;
  // soot burns away while it is hot; it only survives as smoke once the gas cools
  K -= K * (1. - exp(-uKOx * smoothstep(0.55, 1.0, T) * uDt));
  T -= uDt * (uCool * T + uRad * T*T*T*T + uQuench * T);
  // smoke thins out as it rises: a wisp is gone a short way above the button, never a hanging cloud
  K *= exp(-(uSootDecay + uSootH * max(p.y - uWick.z, 0.)) * uDt);
  F *= exp(-uFuelDecay * uDt);
  o = vec4(clamp(T, 0., 2.2), max(F, 0.), clamp(K, 0., 3.), 1.);
}`;

// Carries a field over to a new layout of the stage (the button moved inside it, or the stage or its grid
// changed size): each new cell takes what was at the same place relative to the button. Outside the old
// stage there was nothing. uMul rescales velocities, which are in cells/s, to the new cell size.
const FS_REMAP = HEAD + `
uniform sampler2D uOld;
uniform vec2 uOldRes;
uniform vec4 uNew0;       // new: cell size (CSS px), stage height, offX, offY
uniform vec4 uOld0;       // old: the same
uniform vec4 uMul;
out vec4 o;
void main(){
  vec2 g = gl_FragCoord.xy;
  vec2 b = vec2(g.x * uNew0.x - uNew0.z, uNew0.y - g.y * uNew0.x - uNew0.w);
  vec2 q = vec2((b.x + uOld0.z) / uOld0.x, (uOld0.y - (b.y + uOld0.w)) / uOld0.x);
  o = any(lessThan(q, vec2(0.))) || any(greaterThan(q, uOldRes)) ? vec4(0., 0., 0., 1.) : texture(uOld, q / uOldRes) * uMul;
}`;
// and the sparks and ash, which live in stage CSS px
const FS_PSHIFT = HEAD + `
uniform sampler2D uA, uB;
uniform vec2 uD;
layout(location = 0) out vec4 oA;
layout(location = 1) out vec4 oB;
void main(){
  ivec2 ip = ivec2(gl_FragCoord.xy);
  oA = texelFetch(uA, ip, 0) + vec4(uD, 0., 0.);
  oB = texelFetch(uB, ip, 0);
}`;

const FS_EMIT = HEAD + `
uniform sampler2D uS;
uniform vec2 uRes;
out vec4 o;
void main(){ o = vec4(flameEmit(texture(uS, gl_FragCoord.xy / uRes)), 1.); }`;

// ---- the strip as material, on its own grid over the button (y up): r how much of it is still there,
// g how charred it is, b heat, a the volatile fuel it has left to flame with. Fresh, it is solid and
// unburnt wherever the button's rounded shape is.
const FS_MINIT = HEAD + `
uniform vec3 uMGeom;      // w, h (CSS px), radius
uniform vec2 uCell;       // CSS px per cell
out vec4 o;
void main(){
  vec2 p = gl_FragCoord.xy * uCell, hs = uMGeom.xy * 0.5;
  float in_ = step(sdRR(p - hs, hs, uMGeom.z), 0.5);
  o = vec4(in_, 0., 0., in_);
}`;

// Combustion of the strip: an excitable medium. Heat conducts through what is left of it (more readily
// upwards), and the flame's gas heats what it touches. Hotter than its own ignition point, a cell flames:
// it gives off heat and burns through its volatiles, blackening. The char left behind smoulders where it
// is hot enough, eating itself away to a hole and keeping itself hot while it does, or goes out and stays
// black until heat from its neighbours relights it. The grain makes conductivity, ignition point and how
// readily the char burns vary from place to place, so the front advances raggedly and holes open in
// patches behind it instead of the strip vanishing as one clean line.
const FS_MAT = HEAD + `
uniform sampler2D uM, uS;
uniform vec2 uMRes, uGridPx, uOff;
uniform vec3 uMGeom;      // w, h (CSS px), stage height
uniform vec2 uCell;
uniform float uDt, uCatch, uD, uGen, uCool, uBurn, uIgn, uGasK, uAge, uOx, uSmoulder, uKMin, uKVar;
uniform vec4 uLight;      // x0, x1 (CSS px), heat rate, depth from the top edge (CSS px)
out vec4 o;
vec4 M(ivec2 q){ return any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, ivec2(uMRes))) ? vec4(0.) : texelFetch(uM, q, 0); }
float h0;
// heat of a neighbour; across a hole or the edge of the shape there is nothing to conduct it
float N(ivec2 q){ vec4 n = M(q); return n.r > 0.01 ? n.b : h0; }
void main(){
  ivec2 ip = ivec2(gl_FragCoord.xy);
  vec4 m = M(ip);
  float S = m.r, C = m.g, h = m.b, V = m.a;
  h0 = h;
  vec2 px = gl_FragCoord.xy * uCell;
  float gA = vnoise(px * 0.05) * 0.5 + vnoise(px * vec2(0.16, 0.07) + 3.1) * 0.35 + hash12(floor(px * 0.7) + 9.) * 0.15;
  float gB = vnoise(px * 0.12 + 17.) * 0.65 + hash12(floor(px * 0.9) + 2.) * 0.35;
  float gC = vnoise(px * 0.09 + 41.) * 0.7 + vnoise(px * 0.3 + 7.) * 0.3;
  float hsum = N(ip + ivec2(1, 0)) + N(ip - ivec2(1, 0)) + 0.9 * N(ip + ivec2(0, 1)) + 1.1 * N(ip - ivec2(0, 1))
             + 0.5 * (N(ip + ivec2(1, 1)) + N(ip + ivec2(-1, 1)) + N(ip + ivec2(1, -1)) + N(ip + ivec2(-1, -1)));
  float k = uD * (uKMin + uKVar * gA * gA * gA) / (uCell.x * uCell.y) * smoothstep(0., 0.05, S);
  h += uDt * k * (hsum - 6. * h);
  // until it has fully caught (uCatch 1) nothing ignites, though it browns and blackens. uAge, late in a
  // burn, lowers the ignition point of whatever is already warm, so a front that is dying still finishes
  // the strip, but only by spreading, never by the whole strip lighting at once.
  float age = uAge * smoothstep(0.02, 0.15, h);
  float ign0 = uIgn * (0.6 + 0.8 * gB) - age, ign = ign0 + (1. - uCatch) * 3.;
  vec2 sp = vec2(px.x + uOff.x, uMGeom.z - uOff.y - uMGeom.y + px.y);
  float Tg = texture(uS, sp / uGridPx).r;
  // The flame's gas scorches what it touches, but only up to just short of lighting it: the front moves by
  // heat conducted from what is already alight. Letting the gas light the strip too closes a loop (a bigger
  // flame lights more strip, which feeds a bigger flame) that sets the whole strip off at once. The cap is
  // hard, so no step, however long, can carry a cell over its ignition point on the gas's heat alone.
  float hGas = h;
  h += uDt * uGasK * uCatch * max(Tg - 0.3, 0.) * step(0.02, V) * (1. - smoothstep(0.55, 0.9, h / max(ign0, 0.05)));
  h = min(h, max(hGas, 0.85 * ign0));
  // where it was lit: the top edge under the flame, for a moment
  float fromTop = uMGeom.y - px.y;
  h += uDt * uLight.z * step(fromTop, uLight.w) * step(uLight.x, px.x) * step(px.x, uLight.y) * step(0.02, V);
  float burning = smoothstep(ign, ign + 0.12, h) * smoothstep(0., 0.08, V);
  h += uDt * uGen * burning;
  // a cell only uses up its volatiles once it is properly alight: smouldering at its ignition point would
  // spend them without the heat that carries the front on
  V = max(V - uDt * uBurn * (0.6 + 0.8 * gB) * burning * smoothstep(ign + 0.1, ign + 0.5, h), 0.);
  // Past 1, g counts how long a cell has been spent char (1 to 2 over ~0.4 s): it holds together that long
  // before it can go to ash, so a band of black, curling char trails the front instead of the strip opening
  // the moment its fuel is gone.
  float Cn = C < 1. ? min(1., C + uDt * (6. * burning + 2.2 * uCatch * smoothstep(0.03, max(ign0, 0.1), h) * step(0.02, V))) : C;
  Cn = min(2., Cn + uDt * 2.5 * step(0.999, C) * step(V, 0.3) * step(0.5, uCatch));
  C = Cn;
  float ox = smoothstep(0.3 - age, 0.75 - age, h) * smoothstep(1.05 - 0.3 * gC, 1.55, C) * step(V, 0.3) * step(0.5, uCatch);
  S = max(S - uDt * (uOx * (0.25 + 2.1 * gC * gC) + 0.8 * age) * ox, 0.);
  h += uDt * uSmoulder * (0.4 + 1.2 * gC) * ox * smoothstep(0., 0.15, S);
  // and char that went out flares back up, so no black island is left standing
  h += uDt * uAge * 3. * step(0.9, C) * step(0.02, S);
  // with nothing left to hold it, the heat of a hole is gone at once
  h -= uDt * (uCool + 10. * (1. - smoothstep(0., 0.03, S))) * h;
  o = vec4(S, C, clamp(h, 0., 1.6), V);
}`;

// Passes a fragment for every block of the strip (a few px square, from a mip) with enough left of it to
// see, or still glowing; an occlusion query over it says, without reading anything back, whether the burn
// is over. Blocks, not cells, so a speck too small to notice does not hold the regrow back.
const FS_MLEFT = HEAD + `
uniform sampler2D uM;
uniform vec2 uRes;
uniform float uLod;
out vec4 o;
bool leftAt(vec4 m){ return m.r > 0.35 || m.r * smoothstep(0.3, 0.8, m.b) > 0.08; }
void main(){
  if (!leftAt(textureLod(uM, gl_FragCoord.xy / uRes, uLod))) discard;
  o = vec4(0.);
}`;
const LEFT_LOD = 2;

// ---- sparks and ash: state in two float textures, carried by the gas. Sparks fall and cool; ash (heat
// below zero) is a flake of burnt strip: nearly weightless, fluttering, its edge glowing out.
const FS_PART = HEAD + `
uniform sampler2D uA, uB, uVel;
uniform vec2 uGridPx;     // grid extent, CSS px
uniform vec2 uStage;
uniform float uCs, uDt, uSeedP;
uniform sampler2D uM;
uniform vec4 uMRect;      // offX, offY, w, h of the strip (stage CSS px, y down)
layout(location = 0) out vec4 oA;
layout(location = 1) out vec4 oB;
void main(){
  ivec2 ip = ivec2(gl_FragCoord.xy);
  vec4 a = texelFetch(uA, ip, 0), b = texelFetch(uB, ip, 0);   // a: x, y (stage CSS px, y down), vx, vy   b: life, life0, heat, size
  if (b.x <= 0.){ oA = a; oB = vec4(0.); return; }
  // A candidate (life0 < 0) thrown at a random spot of a burning strip lives only if the strip there is
  // doing what it would come from: an ember off the flaming front, a flake off char going to ash.
  if (b.y < 0.){
    vec2 mu = vec2((a.x - uMRect.x) / uMRect.z, 1. - (a.y - uMRect.y) / uMRect.w);
    vec4 m = textureLod(uM, clamp(mu, 0., 1.), 0.);
    float p = b.z < -0.5
      ? smoothstep(0.6, 0.9, m.g) * step(0.02, m.r) * max(0.6 - m.r, 0.) * (0.3 + m.b) * 2.
      : smoothstep(0.4, 0.7, m.b) * smoothstep(0., 0.3, m.a);
    if (hash12(vec2(ip) * 1.37 + uSeedP) >= p){ oA = a; oB = vec4(0.); return; }
    b.y = -b.y;
  }
  vec2 fv = texture(uVel, vec2(a.x, uStage.y - a.y) / uGridPx).xy * uCs;
  fv.y = -fv.y;
  if (b.z < -0.5){
    float ph = hash12(vec2(ip) + 3.7) * 6.28;
    a.zw += (fv - a.zw) * (1. - exp(-4.5 * uDt));
    a.w += 42. * uDt;
    a.z += cos(b.x * (5. + 3. * fract(ph)) + ph) * 160. * uDt;
    b.z = min(b.z + uDt * 1.2, -1.);
  } else {
    float light = smoothstep(2.4, 0.8, b.w);                    // small embers ride the gas; big sparks fly
    a.zw += (fv - a.zw) * (1. - exp(-mix(1.6, 9., light) * uDt));
    a.w += mix(520., 60., light) * uDt;
    b.z = max(b.z - uDt * mix(1.1, 0.7, light), 0.);
  }
  a.xy += a.zw * uDt;
  b.x -= uDt;
  oA = a; oB = b;
}`;

const VS_SPARK = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uA, uB;
uniform vec2 uStage;
uniform int uPW;
out vec2 vC;
out float vHeat, vFade, vThin, vAsh, vSeed;
void main(){
  int id = gl_VertexID / 6, c = gl_VertexID - id * 6;
  ivec2 ip = ivec2(id % uPW, id / uPW);
  vec4 a = texelFetch(uA, ip, 0), b = texelFetch(uB, ip, 0);
  if (b.x <= 0.){ gl_Position = vec4(2., 2., 2., 1.); return; }
  vec2 k = c == 0 ? vec2(-1,-1) : c == 1 ? vec2(1,-1) : c == 2 ? vec2(1,1) : c == 3 ? vec2(-1,-1) : c == 4 ? vec2(1,1) : vec2(-1,1);
  vFade = clamp(b.x / max(b.y * 0.35, 0.05), 0., 1.);
  if (b.z < -0.5){
    // a tumbling flake: its outline turns edge-on and back as it spins
    float r = float(id) * 2.39 + b.x * (2. + fract(float(id) * 0.37) * 3.);
    vec2 u = vec2(cos(r), sin(r)), v = vec2(-u.y, u.x);
    float w = b.w, t = w * (0.25 + 0.75 * abs(sin(b.x * 3.1 + float(id))));
    vec2 p = a.xy + u * k.x * w + v * k.y * t;
    gl_Position = vec4(p.x / uStage.x * 2. - 1., 1. - p.y / uStage.y * 2., 0., 1.);
    vC = k; vHeat = -1. - b.z; vThin = 0.; vAsh = 1.; vSeed = fract(float(id) * 0.7548);
    return;
  }
  float sp = length(a.zw);
  vec2 dir = sp > 1. ? a.zw / sp : vec2(0., -1.);
  vec2 nrm = vec2(-dir.y, dir.x);
  // a short exposure's streak; each spark gets its own, so the burst is not a clip-art starburst
  float ex = 0.008 + 0.022 * fract(float(id) * 0.61803 + 0.17);
  float w = b.w, len = min(sp * ex, 24.);
  vec2 ctr = a.xy - dir * len * 0.5;
  vec2 p = ctr + dir * k.x * (len * 0.5 + w) + nrm * k.y * w;
  gl_Position = vec4(p.x / uStage.x * 2. - 1., 1. - p.y / uStage.y * 2., 0., 1.);
  vC = vec2(k.x * (len * 0.5 + w) / w, k.y);
  vThin = w / (len + 2. * w);
  vHeat = b.z;
  vAsh = 0.; vSeed = 0.;
}`;
const FS_SPARK = `#version 300 es
precision highp float;
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*.1031); p3 += dot(p3, p3.yzx+33.33); return fract((p3.x+p3.y)*p3.z); }
float vnoise(vec2 p){
  vec2 i = floor(p), f = fract(p);
  vec2 u = f*f*(3.-2.*f);
  return mix(mix(hash12(i), hash12(i+vec2(1,0)), u.x), mix(hash12(i+vec2(0,1)), hash12(i+vec2(1,1)), u.x), u.y);
}
in vec2 vC;
in float vHeat, vFade, vThin, vAsh, vSeed;
out vec4 o;
void main(){
  if (vAsh > 0.5){
    // a torn, curled scrap: an irregular outline (no two flakes alike, and never a regular polygon), soft
    // and ragged at the edge, charcoal-black in the middle going to pale grey ash at the rims, shaded across
    // the curl; the last of the glow creeps along the rim and dies
    float ang = atan(vC.y, vC.x), sd = vSeed * 57.3;
    float wob = 0.22 * sin(ang * 2. + sd) + 0.13 * sin(ang * 5. + sd * 1.7) + 0.07 * sin(ang * 9. + sd * 2.9)
              + 0.12 * (vnoise(vec2(ang * 2.2 + sd, sd)) - 0.5);
    float r = length(vC * vec2(1., 0.8 + 0.4 * fract(sd))) * (1. + wob);
    float a = (1. - smoothstep(0.55, 0.9, r)) * vFade;
    a *= 0.75 + 0.25 * hash12(floor(vC * 5.) + sd);
    float curl = smoothstep(-0.9, 0.9, vC.x * cos(sd) + vC.y * sin(sd));
    vec3 inner = vec3(0.09, 0.085, 0.08), rimC = mix(vec3(0.24, 0.235, 0.225), vec3(0.45, 0.44, 0.43), fract(sd * 3.1));
    vec3 col = mix(inner, rimC, smoothstep(0.2, 0.8, r) * (0.5 + 0.5 * fract(sd * 1.3))) * (0.7 + 0.5 * curl);
    col += vec3(1.0, 0.3, 0.04) * vHeat * vHeat * smoothstep(0.45, 0.8, r) * (0.6 + 0.8 * vnoise(vec2(ang * 3. + sd, vHeat * 4.))) * 1.8;
    o = vec4(col * a, a * 0.92);
    return;
  }
  float across = exp(-vC.y*vC.y*3.2);
  float h = vHeat;
  vec3 col = mix(vec3(0.75, 0.08, 0.01), vec3(1.0, 0.45, 0.08), smoothstep(0.15, 0.55, h));
  col = mix(col, vec3(1.0, 0.92, 0.62), smoothstep(0.6, 1.0, h));
  float I = across * vFade * (0.25 + 1.6 * vThin) * (0.35 + 1.1 * h);
  o = vec4(col * I, I * 0.45);
}`;

// ---- the stage: striker strip, its shadow and the flame's light on the page, smoke, flame.
// Output is premultiplied; rgb above alpha is light added to whatever the page is.
const FS_COMPOSITE = HEAD + `
uniform sampler2D uS, uEmis, uLab0, uLab1, uM;
uniform vec2 uStage, uGridPx;
uniform float uMatOn, uHeal, uRegrow, uPreLod;
uniform float uDpr, uDprY;
uniform vec4 uBtn;        // offX, offY, w, h (stage CSS px, y down)
uniform float uRadius;
uniform vec4 uLabRect;    // x, y, w, h of the label textures (stage CSS px)
uniform float uFlameOn, uWarm, uEmber, uTime, uChar, uAlt, uAltStyle, uRim, uSmokeDark, uDis;
uniform vec2 uShake;
uniform vec3 uScorch;     // button-local x, y, amount
out vec4 o;

vec4 over(vec4 a, vec4 b){ return a + b * (1. - a.a); }
float lum(vec3 c){ return dot(c, vec3(0.3, 0.59, 0.11)); }

vec3 glowAt(vec2 uv){
  return textureLod(uEmis, uv, 1.5).rgb * 0.22 + textureLod(uEmis, uv, 3.).rgb * 0.35 + textureLod(uEmis, uv, 4.5).rgb * 0.45;
}

void main(){
  vec2 fc = gl_FragCoord.xy;
  vec2 css = vec2(fc.x / uDpr, uStage.y - fc.y / uDprY);
  vec2 bp = css - uBtn.xy - uShake;
  vec2 hs = uBtn.zw * 0.5;
  vec2 guv = vec2(css.x, uStage.y - css.y) / uGridPx;

  // ---- light from the flame: its blurred emission, and the direction it comes from
  vec3 glow = vec3(0.), lightDir = vec3(0., -0.7, 0.7);
  if (uFlameOn > 0.5){
    // fades out before the stage edge so the light never shows the stage's rectangle
    float win = smoothstep(0., 40., min(min(css.x, uStage.x - css.x), min(css.y, uStage.y - css.y)));
    glow = glowAt(guv) * win;
    vec2 e = vec2(10., 0.) / uGridPx;
    float gx = lum(textureLod(uEmis, guv + e.xy, 3.5).rgb) - lum(textureLod(uEmis, guv - e.xy, 3.5).rgb);
    float gy = lum(textureLod(uEmis, guv + e.yx, 3.5).rgb) - lum(textureLod(uEmis, guv - e.yx, 3.5).rgb);
    vec2 gd = vec2(gx, -gy);
    lightDir = normalize(vec3(gd / (length(gd) + 1e-4) * 0.8 + vec2(0., -0.35), 0.55));
  }

  // ---- the page: contact shadow, and the flame's light pool with the strip's shadow in it
  float sd = sdRR(bp - hs, hs, uRadius);
  // what is left of the strip, and how much of it has grown back
  vec2 mu = vec2(bp.x / uBtn.z, 1. - bp.y / uBtn.w);
  vec4 mt = vec4(1., 0., 0., 1.);
  float solid = 1.;
  if (uMatOn > 0.5){
    mt = mix(textureLod(uM, mu, 0.), mt, uHeal);
    solid = mix(clamp(textureLod(uM, clamp(mu, 0., 1.), 3.).r * 1.2, 0., 1.), 1., uHeal);
  }
  float rl = uBtn.w * (1. - uRegrow * 1.2) + 7. * (vnoise(vec2(bp.x * 0.07, uTime * 0.7)) - 0.5);
  float grown = uRegrow >= 1. ? 1. : smoothstep(rl - 1.2, rl + 1.2, bp.y) * smoothstep(0., 0.3, uRegrow);
  solid *= uRegrow >= 1. ? 1. : smoothstep(0.1, 0.9, uRegrow);
  float contact = (1. - smoothstep(-2., 6., sdRR(bp - hs - vec2(0., 1.6), hs, uRadius))) * 0.30
                + (1. - smoothstep(-6., 18., sdRR(bp - hs - vec2(0., 6.), hs - 3., uRadius))) * 0.16;
  float castS = (1. - smoothstep(-4., 8., sdRR(bp - hs - vec2(0., 4.), hs, uRadius))) * solid;
  contact *= solid;
  vec3 pool = glow * 0.35 * (1. - 0.7 * castS);
  pool = 1. - exp(-pool * 1.2);
  vec4 page = vec4(0., 0., 0., contact);
  page.rgb += pool;
  page.a += lum(pool) * 0.25 * (1. - page.a);
  // faint warmth at the edge when a pointer lingers near it, or embers while a blown-out request settles
  float heatE = max(uWarm, uEmber) * (1. - uDis);
  float warmTop = heatE * exp(-max(-bp.y, 0.) / 5.) * step(bp.y, 0.5) * smoothstep(-2., 8., min(bp.x, uBtn.z - bp.x));
  float flick = 0.6 + 0.8 * vnoise(vec2(bp.x * 0.07, uTime * 1.7)) * vnoise(vec2(bp.x * 0.19 + 5., uTime * 2.3));
  page.rgb += vec3(1.0, 0.36, 0.08) * warmTop * flick * 0.22;

  // ---- the strip
  vec4 strip = vec4(0.);
  float cover = clamp(0.5 - sd * uDpr, 0., 1.);
  if (cover > 0.){
    vec2 q = bp;
    // heat shimmer just under the warm edge
    q.x += uWarm * 0.7 * (vnoise(vec2(bp.y * 0.45, uTime * 7.)) - 0.5) * exp(-max(bp.y, 0.) / 7.);
    vec2 cell = floor(q * 1.7);
    float g0 = hash12(cell), g1 = vnoise(q * 0.8), g2 = vnoise(q * 0.21 + 7.1), g3 = vnoise(q * 2.2 + 3.3);
    vec2 gn = hash22(cell + 1.7) - 0.5;
    // bevelled shoulders
    float bw = min(3.2, uRadius * 0.6 + 1.6);
    vec2 e = vec2(0.5, 0.);
    vec2 sg = vec2(sdRR(bp - hs + e.xy, hs, uRadius) - sdRR(bp - hs - e.xy, hs, uRadius), sdRR(bp - hs + e.yx, hs, uRadius) - sdRR(bp - hs - e.yx, hs, uRadius));
    float x = clamp(-sd / bw, 0., 1.);
    float slope = 1.5 * (1. - x) * (1. - x);
    vec2 nxy = sg * slope;
    // gritty surface: glass and phosphorus grains
    nxy += gn * 0.32 + (vec2(vnoise(q * 1.3 + 9.), vnoise(q * 1.3 + 4.)) - 0.5) * 0.35;
    // the label, pressed in: r is the sharp glyph coverage, g a blurred copy that shapes the bevel
    vec2 lu = (css - uShake - uLabRect.xy) / uLabRect.zw;
    lu.y = 1. - lu.y;
    vec2 le = vec2(0.8 / uLabRect.z, 0.);
    vec2 le2 = vec2(0., 0.8 / uLabRect.w);
    float m0 = texture(uLab0, lu).r, m1 = texture(uLab1, lu).r;
    float m0x = texture(uLab0, lu + le).g - texture(uLab0, lu - le).g, m0y = texture(uLab0, lu - le2).g - texture(uLab0, lu + le2).g;
    float m1x = texture(uLab1, lu + le).g - texture(uLab1, lu - le).g, m1y = texture(uLab1, lu - le2).g - texture(uLab1, lu + le2).g;
    // the fades overlap, so some legible text is always there
    float a1 = smoothstep(0.3, 0.7, uAlt), a0 = 1. - a1;
    // a strip growing back is blank until it is nearly whole, and then its label comes up through it
    float labIn = uRegrow >= 1. ? 1. : smoothstep(0.7, 1., uRegrow);
    a0 *= labIn; a1 *= labIn;
    nxy += 1.1 * (vec2(m0x, m0y) * a0 + vec2(m1x, m1y) * a1);
    vec3 n = normalize(vec3(-nxy, 1.));

    vec3 alb = vec3(0.40, 0.115, 0.07) * (0.80 + 0.30 * g1 + 0.12 * (g2 - 0.5)) * (0.9 + 0.2 * g3);
    alb = mix(alb, vec3(0.16, 0.055, 0.04), step(g0, 0.14) * 0.55);
    float glass = step(0.965, g0);
    alb = mix(alb, vec3(0.52, 0.38, 0.33), glass * 0.55);
    // scorching: along the burnt edge, and where the head was struck
    float ch = uChar * clamp(1. - (bp.y - 1.) / (uBtn.w * 0.34 + 9. * (vnoise(bp * 0.09) - 0.5) + 4. * (g1 - 0.5)), 0., 1.);
    ch = max(ch, uScorch.z * (1. - smoothstep(3., 11. + 5. * g2, length((bp - uScorch.xy) * vec2(0.6, 1.)))));
    ch = clamp(ch, 0., 1.);
    alb = mix(alb, vec3(0.045, 0.03, 0.025) * (0.7 + 0.6 * g1), ch * 0.92);

    vec3 ink0 = vec3(0.90, 0.82, 0.68) * (0.93 + 0.1 * g3);
    // Ahead of the front the heat that has crept into the strip, spread wider than the front itself, scorches
    // it: the label yellows, browns and blisters (its sharp edges soften) and is black before the front arrives.
    float pre = 0.;
    if (uMatOn > 0.5){
      float hb = textureLod(uM, mu, uPreLod).b;
      pre = smoothstep(0.04, 0.45, hb) * (1. - smoothstep(0.35, 0.95, mt.g)) * (1. - uHeal);
      pre = max(pre, smoothstep(0.05, 0.5, mt.g));
      m0 = mix(m0, texture(uLab0, lu).g * 1.15, 0.55 * smoothstep(0.2, 0.8, pre));
      ink0 = mix(ink0, vec3(0.62, 0.40, 0.17) * (0.9 + 0.2 * g2), smoothstep(0., 0.35, pre));
      ink0 = mix(ink0, vec3(0.20, 0.09, 0.04), smoothstep(0.3, 0.7, pre));
      ink0 = mix(ink0, vec3(0.035, 0.025, 0.02), smoothstep(0.65, 1., pre));
      alb = mix(alb, vec3(0.24, 0.085, 0.045) * (0.8 + 0.4 * g1), 0.5 * smoothstep(0.1, 0.8, pre));
    }
    vec3 ink1 = uAltStyle < 0.5 ? vec3(0.05, 0.025, 0.018) : vec3(0.63, 0.61, 0.58) * (0.85 + 0.25 * g3);
    alb = mix(alb, ink0, m0 * a0 * (1. - 0.25 * ch));
    alb = mix(alb, ink1, m1 * a1);

    // burning: it browns ahead of the front, blackens as it burns, and goes to grey ash at the edges of
    // the holes it leaves; the label goes with it
    float fe = mt.r + (g1 - 0.5) * 0.16 + (g3 - 0.5) * 0.06;
    float brown = smoothstep(0., 0.45, mt.g), black = smoothstep(0.35, 0.95, mt.g);
    alb = mix(alb, vec3(0.19, 0.07, 0.035) * (0.8 + 0.4 * g1), brown * 0.85 * (1. - black));
    alb = mix(alb, vec3(0.034, 0.027, 0.023) * (0.7 + 0.6 * g1), black);
    float ashy = smoothstep(0.45, 0.16, fe) * black * (1. - smoothstep(0.5, 1.1, mt.b));
    alb = mix(alb, vec3(0.42, 0.40, 0.385) * (0.5 + 0.7 * g3), ashy * 0.8);
    if (uMatOn > 0.5 && black > 0.){
      // the burnt edge curls up off the page
      vec2 me = vec2(1.2 / uBtn.z, 0.), me2 = vec2(0., 1.2 / uBtn.w);
      vec2 fg = vec2(textureLod(uM, mu + me, 0.).r - textureLod(uM, mu - me, 0.).r, textureLod(uM, mu - me2, 0.).r - textureLod(uM, mu + me2, 0.).r);
      nxy -= fg * 2.2 * black;
      n = normalize(vec3(-nxy, 1.));
    }

    vec3 L = normalize(vec3(-0.32, -0.62, 0.72));
    float dif = max(dot(n, L), 0.);
    float amb = 0.50 + 0.22 * n.z - 0.12 * n.y;
    vec3 col = alb * (vec3(1.0, 0.97, 0.92) * 0.78 * dif + vec3(0.92, 0.94, 1.0) * amb * 0.62);
    vec3 H = normalize(L + vec3(0., 0., 1.));
    col += vec3(1.0, 0.97, 0.9) * (0.05 * pow(max(dot(n, H), 0.), 14.) + glass * 0.28 * pow(max(dot(n, H), 0.), 90.)) * (1. - ch * 0.8);
    // lit by the flame above it; the face just under the burning edge catches the most
    vec3 fl = glow * 1.1;
    if (uFlameOn > 0.5) fl += glowAt(guv + vec2(0., 10.) / uGridPx) * 1.1 * exp(-max(bp.y, 0.) / (uBtn.w * 1.2));
    float fd = max(dot(n, lightDir), 0.);
    col += alb * fl * (0.35 + 1.4 * fd);
    vec3 Hf = normalize(lightDir + vec3(0., 0., 1.));
    col += fl * glass * 0.9 * pow(max(dot(n, Hf), 0.), 70.);
    // hover warmth or embers: the top edge starts to glow
    col += vec3(1.0, 0.33, 0.07) * heatE * exp(-max(bp.y, 0.) / 2.2) * flick * 0.35 * smoothstep(-2., 8., min(bp.x, uBtn.z - bp.x));
    // the branded label: glowing, cooling to char
    if (uAltStyle < 0.5){
      float glowT = uRim * (0.75 + 0.5 * vnoise(q * 0.4 + uTime * 0.6));
      col += m1 * a1 * mix(vec3(0.9, 0.16, 0.02), vec3(1.0, 0.62, 0.22), glowT) * (0.25 + 1.2 * glowT);
      col += vec3(1.0, 0.3, 0.05) * ch * uRim * 0.35 * step(0.9, hash12(floor(q * 0.9) + 5.)) * vnoise(q * 0.6 + uTime);
    }
    // embers: a thin glowing line along the flaming front and the rims of the holes, and a dull glow in
    // the cracks of the smouldering char behind it
    if (mt.b > 0.05){
      float hot = smoothstep(0.3, 1.5, mt.b);
      float front = smoothstep(0., 0.1, mt.a) * (1. - smoothstep(0.35, 0.8, mt.a));
      float rimH = smoothstep(0.02, 0.1, fe) * (1. - smoothstep(0.12, 0.3, fe));
      float cr = smoothstep(0.6, 0.78, vnoise(q * 0.5 + 3.) * 0.6 + vnoise(q * 1.6 + 11.) * 0.4);
      float fk = 0.7 + 0.6 * vnoise(q * 0.25 + vec2(0., uTime * 1.7));
      float Ie = hot * (0.06 * black + 0.95 * front + 0.8 * rimH + 0.45 * cr * black) * fk;
      vec3 bb = mix(vec3(0.5, 0.035, 0.), vec3(1.0, 0.3, 0.035), smoothstep(0.08, 0.55, Ie));
      bb = mix(bb, vec3(1.0, 0.7, 0.28), smoothstep(0.6, 1.2, Ie));
      col += bb * Ie * 1.2;
    }
    // growing back: a faint sheen along the line it re-forms at
    if (uRegrow < 1.) col += vec3(0.9, 0.75, 0.6) * 0.18 * exp(-abs(bp.y - rl) / 1.6) * (1. - uRegrow);
    // disabled: a dead, greyed strip
    col = mix(col, vec3(lum(col)) * vec3(0.95, 0.9, 0.86), 0.7 * uDis) * (1. - 0.3 * uDis);
    col = pow(max(col, 0.), vec3(0.96));
    strip = vec4(col, 1.) * cover * smoothstep(0.05, 0.2, fe) * smoothstep(0., 0.04, mt.r) * grown;
  }
  vec4 c = over(strip, page);

  // ---- smoke and flame
  if (uFlameOn > 0.5){
    vec4 s = texture(uS, guv);
    // the stage's top edge must never show as a line across a tall fire
    s *= smoothstep(0., 30., css.y) * smoothstep(0., 16., min(css.x, uStage.x - css.x));
    float K = s.b * smoothstep(0.9, 0.35, s.r);
    float sa = 1. - exp(-K * 1.35);
    vec3 sc = mix(vec3(0.58, 0.57, 0.56), vec3(0.36, 0.35, 0.33), uSmokeDark) * (0.85 + 0.2 * vnoise(css * 0.15));
    sc += glow * 0.4;
    c = over(vec4(sc * sa, sa), c);
    vec3 E = flameEmit(s);
    vec3 F = 1. - exp(-E * 1.7);
    // opaque where it is bright, so the core is not a ghost over the page; the blue base stays thin
    float fa = clamp(max(F.r, max(F.g, F.b * 0.6)) * 1.1, 0., 1.);
    c = vec4(F, fa) + c * (1. - fa);
    // the bright air around the flame
    vec3 halo = glow * 0.12 * (1. - strip.a * 0.6);
    c.rgb += halo;
    c.a += lum(halo) * 0.35 * (1. - c.a);
  }
  c.rgb += (hash12(fc + fract(uTime) * 91.) - 0.5) / 255.;
  o = clamp(c, 0., 1.);
}`;

// ---------------------------------------------------------------- audio (this effect's own)
const fireAudio = { muted: false };
let actx = null;
function audio(){
  if (fireAudio.muted) return null;
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
function hiss(type, f0, f1, q, gain, dur, delay = 0, attack = 0.004){
  const a = audio(); if (!a) return;
  const t = a.currentTime + delay;
  const src = a.createBufferSource(); src.buffer = noise(a);
  const f = a.createBiquadFilter(); f.type = type; f.Q.value = q;
  f.frequency.setValueAtTime(f0, t);
  if (f1 !== f0) f.frequency.exponentialRampToValueAtTime(f1, t + dur);
  const g = a.createGain();
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(gain, t + attack);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  src.connect(f).connect(g).connect(a.destination);
  src.start(t, Math.random() * 1.5); src.stop(t + dur + 0.05);
}
function tone(f0, f1, dur, gain, delay = 0){
  const a = audio(); if (!a) return;
  const t = a.currentTime + delay;
  const o = a.createOscillator(), g = a.createGain();
  o.type = 'sine';
  o.frequency.setValueAtTime(f0, t); o.frequency.exponentialRampToValueAtTime(f1, t + dur);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(gain, t + 0.012);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  o.connect(g).connect(a.destination); o.start(t); o.stop(t + dur + 0.03);
}
const rand = (a, b) => a + Math.random() * (b - a);
const sounds = {
  // the head dragged over glass grit, then the flare catching; k is the fire's scale, 0 a match
  strike(k = 1){
    const n = Math.round(3 + 3 * k);
    for (let i = 0; i < n; i++) hiss('bandpass', rand(2400, 5200), rand(1500, 2400), 1.3, rand(0.07, 0.13) * (0.6 + 0.4 * k), rand(0.018, 0.035), i * 0.013);
    hiss('highpass', 4500, 4500, 0.7, 0.04, 0.1);
    if (k < 0.5){
      // a match head flaring: a bright fizz that drops as it settles
      hiss('bandpass', 3200, 1100, 0.9, 0.11 * (1 - k), 0.32, 0.03, 0.02);
      hiss('lowpass', 500, 900, 0.8, 0.08 * (1 - k), 0.25, 0.05, 0.04);
    }
    if (k > 0.05){
      hiss('lowpass', 180, 1500, 0.8, 0.32 * k, 0.55, 0.05, 0.07);
      hiss('bandpass', 700, 300, 0.6, 0.12 * k, 0.7, 0.07, 0.12);
      tone(90, 42, 0.38, 0.2 * k, 0.05);
    }
    for (let i = 0, m = Math.round(1 + 6 * k); i < m; i++) sounds.crackle(1, 0.08 + Math.random() * 0.45);
  },
  scratch(){ hiss('bandpass', rand(2200, 4200), 1800, 1.2, 0.045, 0.028); },
  snuff(){ hiss('bandpass', 1500, 260, 0.8, 0.13, 0.45, 0, 0.025); tone(170, 85, 0.12, 0.04); },
  fizz(){
    for (let i = 0; i < 30; i++) hiss('highpass', rand(3500, 7000), rand(3000, 6000), 0.8, rand(0.015, 0.045), rand(0.015, 0.06), Math.random() * 0.8);
    hiss('bandpass', 5200, 2600, 0.5, 0.05, 0.85, 0, 0.1);
  },
  puff(){ hiss('lowpass', 520, 140, 0.8, 0.22, 0.32, 0, 0.01); tone(120, 50, 0.22, 0.16); },
  blow(){ hiss('bandpass', 950, 320, 0.6, 0.15, 0.42, 0, 0.03); },
  crackle(level, delay = 0){ hiss('highpass', rand(1400, 5000), rand(1200, 4000), 0.9, rand(0.02, 0.08) * level, rand(0.004, 0.014), delay, 0.001); },
  // air pushed into the fire
  fan(k = 1){ hiss('lowpass', 260, 700, 0.7, 0.1 + 0.08 * k, 0.28, 0, 0.06); hiss('bandpass', 1200, 700, 0.7, 0.03, 0.22, 0.02, 0.05); },
  // the whole strip catching
  whoomp(k = 1){ hiss('lowpass', 140, 900, 0.8, 0.12 + 0.3 * k, 0.9, 0, 0.18); tone(70, 40, 0.6, 0.05 + 0.15 * k, 0.05); },
  // a soft rising breath as it re-forms
  regrow(){ hiss('bandpass', 380, 1500, 1.4, 0.05, 0.7, 0, 0.35); tone(330, 495, 0.5, 0.018, 0.25); },
};
const silent = Object.fromEntries(Object.keys(sounds).map(k => [k, () => {}]));

// ---------------------------------------------------------------- helpers
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');

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

// ---------------------------------------------------------------- scale
// 0 is a match, 1 the full fire along the whole edge. Everything about the fire follows from it: how
// wide a stretch of the edge feeds it and how hard, how buoyant and turbulent the gas is, how much
// room it gets, how finely it is simulated, its sparks, its sound, and how fast the strip burns away.
const SCALES = { match: 0, candle: 0.22, small: 0.22, campfire: 0.6, bonfire: 1 };
function scaleOf(v){
  if (typeof v === 'number' && Number.isFinite(v)) return Math.max(0, Math.min(1, v));
  if (typeof v === 'string' && Object.hasOwn(SCALES, v)) return SCALES[v];
  return 1;
}
function scaleParams(k){
  const L = (a, b, e = 1) => a + (b - a) * Math.pow(k, e);
  // The small end is two different fires, not one: a match (m) is a short, round flame that wavers and
  // trails a thread of smoke, after a sulphur flare bigger than the flame it leaves; a candle (c) is a
  // tall, narrow, steady one that barely smokes. Both fade out well before a campfire.
  const m = Math.max(0, 1 - k / 0.12), c = Math.max(0, 1 - Math.abs(k - 0.22) / 0.2);
  return {
    k, match: m,
    halfW: full => Math.min(full, 3.8 + 0.7 * m - 1.1 * c + Math.max(0, full - 3.8) * Math.pow(k, 1.7)),
    kBurn: L(60, 14, 0.5), heat: 1.05,
    fuel: L(32, 12, 0.5) * (1 - 0.3 * m), buoy: L(420, 1300) * (1 - 0.18 * m) * (1 + 0.3 * c), jit: L(60, 1400, 1.3) + 240 * m - 30 * c,
    eps: L(2, 14, 1.2) + 3 * m, flick: L(2.4, 5.5) + 5 * m,
    steady: Math.max(0, 1 - k / 0.45) * (1 - 0.65 * m), flare: L(0.55, 1) + 0.5 * m, flareOver: L(1.2, 0.35) + 0.9 * m, flareT: 0.25 - 0.08 * m,
    wisp: m * (1 - 0.5 * c), calm: Math.max(m, c, 1 - k / 0.3, 0), sootDecay: 0.45 + 1.5 * Math.max(m, c),
    sparkN: Math.round(L(9, 46) - 6 * m), sparkV: L(0.4, 1), sparkSz: L(0.55, 1) + 0.35 * m, ambient: L(0.25, 2.5),
    mTopMin: L(90, 170), mTopK: L(1.7, 3.4), mSideMin: L(64, 120), mSideK: L(1.2, 2.2),
    csMin: L(1.35, 2.5), csDiv: L(240, 170),
    flameHMin: L(14, 30), flameHK: L(0.45, 1.1),
    roar: L(0.14, 1), roarLp: L(950, 380), roarBp: L(2600, 1600), crackle: L(0.15, 1),
    // the strip burning away: diffusion (px^2/s), heat release, cooling, fuel use, ignition point
    mD: L(20, 40), mGen: L(9, 12), mCool: L(1.1, 1), mBurn: L(2.2, 3.4), mIgn: L(0.5, 0.42), mGasK: L(20, 30),
    feedK: L(0.3, 1.0), mOx: L(1.0, 1.8), mSmoulder: L(2.4, 2.8), mFuel: L(1.6, 2.8, 1.5), mPilot: L(0.72, 1.35, 1.5), mLift: L(3, 6), expect: L(4.6, 1.7), fanCatch: L(1.4, 1.0),
    emberRate: L(90, 500), ashRate: L(50, 260), ashSz: L(0.7, 1.1),
  };
}

// ---------------------------------------------------------------- component
function attachFire(button, opts = {}){
  const noop = () => {};
  const unsupported = { supported: false, state: 'unsupported', scale: scaleOf(opts.scale), detach: noop, refresh: noop, burn: noop, succeed: noop, fail: noop, consume: noop, setScale: noop, _manual: noop, _advance: noop };
  const canvas = document.createElement('canvas');
  canvas.setAttribute('aria-hidden', 'true');
  Object.assign(canvas.style, { position: 'absolute', left: '0px', top: '0px', margin: '0', padding: '0', border: '0', display: 'block', pointerEvents: 'none', zIndex: String(opts.zIndex ?? 10) });
  const gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: true, antialias: false, depth: false, stencil: false });
  if (!gl || !gl.getExtension('EXT_color_buffer_float')) return unsupported;
  const sound = opts.sound === false ? silent : sounds;

  // ---- GL resources (rebuilt from scratch after a lost context)
  let P, vao, lab0, lab1, pA = [], pB = [], pFb = [], pCur = 0;
  let vel = [], velFb = [], S = [], SFb = [], pr = [], prFb = [], divT, divFb, curlT, curlFb, emis, emisFb;
  let vc = 0, sc = 0, pc = 0;
  // the strip's material grid, and a coarse mip of it read back asynchronously
  let Mt = [], MFb = [], MLvFb = [], mc = 0, mW = 0, mH = 0, mCellX = 1, mCellY = 1, mLw = 1, mLh = 1, mTop = 0;
  let rbBuf = null, leftQ = null, leftPending = false;
  function initGL(){
    gl.getExtension('EXT_color_buffer_float');
    P = {
      advV: compileProgram(gl, FS_ADVECT_V), curl: compileProgram(gl, FS_CURL), force: compileProgram(gl, FS_FORCE),
      div: compileProgram(gl, FS_DIV), jacobi: compileProgram(gl, FS_JACOBI), grad: compileProgram(gl, FS_GRAD),
      advS: compileProgram(gl, FS_ADVECT_S), emit: compileProgram(gl, FS_EMIT), part: compileProgram(gl, FS_PART),
      spark: compileProgram(gl, FS_SPARK, VS_SPARK), comp: compileProgram(gl, FS_COMPOSITE),
      minit: compileProgram(gl, FS_MINIT), mat: compileProgram(gl, FS_MAT), mleft: compileProgram(gl, FS_MLEFT),
      remap: compileProgram(gl, FS_REMAP), pshift: compileProgram(gl, FS_PSHIFT),
    };
    vao = gl.createVertexArray();
    lab0 = gl.createTexture(); lab1 = gl.createTexture();
    for (const t of [lab0, lab1]){
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    }
    pA = [0, 1].map(() => makeTex(PW, PH, gl.RGBA32F, gl.RGBA, gl.FLOAT, gl.NEAREST));
    pB = [0, 1].map(() => makeTex(PW, PH, gl.RGBA32F, gl.RGBA, gl.FLOAT, gl.NEAREST));
    pFb = [0, 1].map(i => makeTarget(pA[i], pB[i]));
    pCur = 0;
    vel = []; S = []; pr = []; divT = curlT = emis = null;
    Mt = []; MFb = []; MLvFb = []; leftQ = gl.createQuery(); leftPending = false;
  }
  const makeTex = (w, h, internal, format, type, filter) => {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, null);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, filter], [gl.TEXTURE_MAG_FILTER, filter === gl.LINEAR_MIPMAP_LINEAR ? gl.LINEAR : filter], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
    return t;
  };
  const makeTarget = (...ts) => {
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    ts.forEach((t, i) => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t, 0));
    if (ts.length > 1) gl.drawBuffers(ts.map((_, i) => gl.COLOR_ATTACHMENT0 + i));
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return fb;
  };
  const bindTex = (unit, t) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, t); };
  const pass = (prog, fb, w, h, setup) => {
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.viewport(0, 0, w, h);
    gl.useProgram(prog.p);
    setup(prog.u);
    gl.bindVertexArray(vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  };

  try { initGL(); } catch (err) {
    console.warn('ElementalFX fire: shaders failed, leaving the button as it is.', err);
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return unsupported;
  }

  // ---- geometry. The stage is the canvas, placed so the button sits at (offX, offY) inside it; there is
  // more room above the button than below, since that is where flame and smoke go.
  const G = { fixed: false, bx: NaN, by: NaN, bw: 0, bh: 0, stageW: 0, stageH: 0, offX: 0, offY: 0 };
  let bounds = { w: 0, h: 0 }, cbOrigin = { x: 0, y: 0 };
  let dpr = 1, dprY = 1, W = 0, H = 0;
  let cs = 3, vW = 0, vH = 0, dW = 0, dH = 0, radius = 0, labelKey = '';
  let labRect = [0, 0, 1, 1];

  // ---- the button: invisible but still there
  const saved = button.getAttribute('style');
  const savedBusy = button.getAttribute('aria-busy');
  const hideStyle = { background: 'transparent', borderColor: 'transparent', color: 'transparent', boxShadow: 'none', textShadow: 'none' };
  function hideButton(){ Object.assign(button.style, hideStyle); }
  function showButton(){
    // setting it first: removing a style attribute only ever written through CSSOM can leave style=""
    button.setAttribute('style', saved ?? '');
    if (saved === null) button.removeAttribute('style');
  }
  function setBusy(on){
    if (on) button.setAttribute('aria-busy', 'true');
    else if (savedBusy === null) button.removeAttribute('aria-busy'); else button.setAttribute('aria-busy', savedBusy);
  }

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

  let btnRect = null;
  function layout(r = button.getBoundingClientRect()){
    btnRect = r;
    const bx = r.left + (G.fixed ? 0 : scrollX), by = r.top + (G.fixed ? 0 : scrollY);
    const bw = r.width, bh = r.height;
    if (bw < 2 || bh < 2){ canvas.style.visibility = 'hidden'; G.bx = bx; G.by = by; return; }
    canvas.style.visibility = lost ? 'hidden' : '';
    const mTop = Math.round(Math.max(SP.mTopMin, SP.mTopK * bh)), mSide = Math.round(Math.max(SP.mSideMin, SP.mSideK * bh)), mBot = Math.round(Math.max(56, 1.1 * bh));
    // the stage stays inside the document, so it can never add scrolling
    const x0 = Math.floor(Math.min(bx, Math.max(0, bx - mSide))), y0 = Math.floor(Math.min(by, Math.max(0, by - mTop)));
    const x1 = Math.ceil(Math.max(bx + bw, Math.min(bounds.w, bx + bw + mSide))), y1 = Math.ceil(Math.max(by + bh, Math.min(bounds.h, by + bh + mBot)));
    const sw = x1 - x0, sh = y1 - y0;
    const sizeChanged = sw !== G.stageW || sh !== G.stageH;
    const btnChanged = Math.abs(bw - G.bw) > 0.25 || Math.abs(bh - G.bh) > 0.25;
    const offChanged = Math.abs(bx - x0 - G.offX) > 0.25 || Math.abs(by - y0 - G.offY) > 0.25;
    // A relayout mid-fire (a rotation, a resize, content shifting above the button) keeps the fire: the gas,
    // sparks and ash are carried over to the new layout, and the strip's material never depended on it.
    const prev = { offX: G.offX, offY: G.offY, stageH: G.stageH, cs, vW, vH, dW, dH };
    Object.assign(G, { bx, by, bw, bh, stageW: sw, stageH: sh, offX: bx - x0, offY: by - y0 });
    Object.assign(canvas.style, { left: `${x0 - cbOrigin.x}px`, top: `${y0 - cbOrigin.y}px` });
    radius = Math.min(parseFloat(getComputedStyle(button).borderTopLeftRadius) || 0, bw / 2, bh / 2);
    if (sizeChanged){
      Object.assign(canvas.style, { width: `${sw}px`, height: `${sh}px` });
      resizeGL(prev);
    } else if (btnChanged || offChanged){ buildLabels(); if (offChanged) shiftSim(prev); if (btnChanged) allocMat(); }
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

  function resizeGL(prev = null){
    if (lost || gl.isContextLost() || !G.stageW) return;
    // up to 3x while the stage is small enough that the per-pixel composite stays cheap
    let d = Math.min(devicePixelRatio || 1, Math.max(2, Math.sqrt(2.4e6 / (G.stageW * G.stageH))), 3);
    const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    d = Math.min(d, maxTex / G.stageW, maxTex / G.stageH);
    canvas.width = Math.max(1, Math.round(G.stageW * d)); canvas.height = Math.max(1, Math.round(G.stageH * d));
    W = gl.drawingBufferWidth; H = gl.drawingBufferHeight;
    dpr = W / G.stageW; dprY = H / G.stageH;
    // the flame is simulated coarser than the screen: velocity on ~2.5 px cells, heat, fuel and soot on
    // cells half that size, and the composite reconstructs it smoothly at full resolution
    cs = Math.max(SP.csMin, Math.max(G.stageW, G.stageH) / SP.csDiv);
    vW = Math.ceil(G.stageW / cs); vH = Math.ceil(G.stageH / cs);
    dW = vW * 2; dH = vH * 2;
    const oldT = [...vel, ...S, ...pr, divT, curlT, emis], oldF = [...velFb, ...SFb, ...prFb, divFb, curlFb, emisFb];
    const oldVel = vel[vc], oldS = S[sc];
    const half = (w, h, f = gl.LINEAR) => makeTex(w, h, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, f);
    const red = (w, h) => makeTex(w, h, gl.R16F, gl.RED, gl.HALF_FLOAT, gl.NEAREST);
    vel = [half(vW, vH), half(vW, vH)]; velFb = vel.map(t => makeTarget(t));
    S = [half(dW, dH), half(dW, dH)]; SFb = S.map(t => makeTarget(t));
    pr = [red(vW, vH), red(vW, vH)]; prFb = pr.map(t => makeTarget(t));
    divT = red(vW, vH); divFb = makeTarget(divT);
    curlT = red(vW, vH); curlFb = makeTarget(curlT);
    emis = half(vW, vH, gl.LINEAR_MIPMAP_LINEAR); gl.generateMipmap(gl.TEXTURE_2D); emisFb = makeTarget(emis);
    vc = sc = pc = 0;
    clearSim();
    if (prev && simHot && oldVel && oldS && prev.vW){
      const k = prev.cs / cs;
      remapField(oldS, prev.dW, prev.dH, SFb[0], dW, dH, cs / 2, prev.cs / 2, prev, [1, 1, 1, 1]);
      remapField(oldVel, prev.vW, prev.vH, velFb[0], vW, vH, cs, prev.cs, prev, [k, k, 1, 1]);
      shiftSparks(prev);
    }
    oldT.forEach(t => t && gl.deleteTexture(t));
    oldF.forEach(f => f && gl.deleteFramebuffer(f));
    // the material follows only the button, so a change of stage alone leaves a burning strip burning
    if (!Mt.length || matFor.bw !== G.bw || matFor.bh !== G.bh) allocMat();
    buildLabels();
    dirty = true;
  }
  let matFor = { bw: 0, bh: 0 };
  function remapField(src, sw, sh, dstFb, w, h, cNew, cOld, prev, mul){
    pass(P.remap, dstFb, w, h, u => {
      bindTex(0, src); gl.uniform1i(u.uOld, 0); gl.uniform2f(u.uOldRes, sw, sh);
      gl.uniform4f(u.uNew0, cNew, G.stageH, G.offX, G.offY);
      gl.uniform4f(u.uOld0, cOld, prev.stageH, prev.offX, prev.offY);
      gl.uniform4f(u.uMul, ...mul);
    });
  }
  function shiftSparks(prev){
    const dx = G.offX - prev.offX, dy = G.offY - prev.offY;
    if (!dx && !dy) return;
    pass(P.pshift, pFb[1 - pCur], PW, PH, u => {
      bindTex(0, pA[pCur]); gl.uniform1i(u.uA, 0);
      bindTex(1, pB[pCur]); gl.uniform1i(u.uB, 1);
      gl.uniform2f(u.uD, dx, dy);
    });
    pCur = 1 - pCur;
  }
  // the button moved inside a stage of the same size
  function shiftSim(prev){
    if (lost || gl.isContextLost() || !vel.length) return;
    shiftSparks(prev);
    if (!simHot) return;
    remapField(S[sc], dW, dH, SFb[1 - sc], dW, dH, cs / 2, cs / 2, prev, [1, 1, 1, 1]); sc = 1 - sc;
    remapField(vel[vc], vW, vH, velFb[1 - vc], vW, vH, cs, cs, prev, [1, 1, 1, 1]); vc = 1 - vc;
    gl.clearColor(0, 0, 0, 0);
    for (const f of prFb){ gl.bindFramebuffer(gl.FRAMEBUFFER, f); gl.clear(gl.COLOR_BUFFER_BIT); }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  function allocMat(){
    if (lost || gl.isContextLost() || !G.bw) return;
    Mt.forEach(t => gl.deleteTexture(t));
    [...MFb, ...MLvFb].forEach(f => gl.deleteFramebuffer(f));
    leftPending = false;
    // about a CSS pixel per cell: fine enough for a ragged front and charred glyphs, cheap enough to ignore
    const c = Math.max(1, G.bw / 256, G.bh / 96);
    mW = Math.max(2, Math.round(G.bw / c)); mH = Math.max(2, Math.round(G.bh / c));
    mCellX = G.bw / mW; mCellY = G.bh / mH;
    matFor = { bw: G.bw, bh: G.bh };
    const L = Math.min(LEFT_LOD, Math.floor(Math.log2(Math.max(mW, mH))));
    mTop = Math.floor(Math.log2(Math.max(mW, mH)));
    mLw = Math.max(1, mW >> L); mLh = Math.max(1, mH >> L);
    Mt = [0, 1].map(() => makeTex(mW, mH, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR_MIPMAP_LINEAR));
    MFb = Mt.map(t => makeTarget(t));
    MLvFb = Mt.map(t => {
      gl.bindTexture(gl.TEXTURE_2D, t); gl.generateMipmap(gl.TEXTURE_2D);
      const fb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, L);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return fb;
    });
    rbBuf = new Float32Array(mLw * mLh * 4);
    initMat();
    // a burn-away cut short by a new layout is over: there is nothing of the old strip left to burn
    if (state === 'consume') startRegrow();
  }
  function initMat(){
    if (!Mt.length) return;
    for (const i of [0, 1]){
      pass(P.minit, MFb[i], mW, mH, u => { gl.uniform3f(u.uMGeom, G.bw, G.bh, radius); gl.uniform2f(u.uCell, mCellX, mCellY); });
      gl.bindTexture(gl.TEXTURE_2D, Mt[i]); gl.generateMipmap(gl.TEXTURE_2D);
    }
    matDirty = false; left = true;
  }

  function clearSim(){
    if (!vel.length) return;
    gl.clearColor(0, 0, 0, 0);
    for (const f of [...velFb, ...SFb, ...prFb, divFb, curlFb, emisFb]){ gl.bindFramebuffer(gl.FRAMEBUFFER, f); gl.clear(gl.COLOR_BUFFER_BIT); }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // The label is the button's own text, drawn where the browser lays it out, in its font; the success
  // and failure texts are drawn the same way so they sit where the label was.
  function labelText(){ return (button.innerText || button.textContent || '').replace(/\s+/g, ' ').trim(); }
  // Two channels: r is the glyphs drawn sharp (their colour), g a blurred copy (the pressed-in bevel), so
  // the text reads as crisply as the page's own. Drawn at the screen's full density even when the stage
  // is not.
  function drawLabel(tex, text){
    const cs0 = getComputedStyle(button);
    const pad = 4, ld = Math.max(dpr, Math.min(devicePixelRatio || 1, 3));
    const w = Math.max(1, Math.round((G.bw + pad * 2) * ld)), h = Math.max(1, Math.round((G.bh + pad * 2) * ld));
    const fsize = parseFloat(cs0.fontSize) || 16;
    let cx = G.bw / 2, cy = G.bh / 2;
    const range = document.createRange();
    range.selectNodeContents(button);
    const rr = range.getBoundingClientRect();
    const br = button.getBoundingClientRect();
    if (rr.width > 0 && rr.height > 0){ cx = rr.left + rr.width / 2 - br.left; cy = rr.top + rr.height / 2 - br.top; }
    const layer = blur => {
      const c = document.createElement('canvas');
      c.width = w; c.height = h;
      const x = c.getContext('2d', { willReadFrequently: true });
      x.font = `${cs0.fontStyle} ${cs0.fontWeight} ${fsize * ld}px ${cs0.fontFamily}`;
      if ('letterSpacing' in x && cs0.letterSpacing !== 'normal') x.letterSpacing = `${(parseFloat(cs0.letterSpacing) || 0) * ld}px`;
      x.textAlign = 'center'; x.textBaseline = 'alphabetic';
      const m = x.measureText(text);
      const asc = m.fontBoundingBoxAscent ?? fsize * ld * 0.8, desc = m.fontBoundingBoxDescent ?? fsize * ld * 0.2;
      if (blur) x.filter = `blur(${blur * ld}px)`;
      x.fillStyle = '#fff';
      x.fillText(text, (cx + pad) * ld, (cy + pad) * ld + (asc - desc) / 2);
      return x.getImageData(0, 0, w, h).data;
    };
    const sharp = layer(0), soft = layer(0.6);
    const px = new Uint8Array(w * h * 4);
    // flipped rows: GL textures start at the bottom
    for (let y = 0; y < h; y++){
      const src = (h - 1 - y) * w * 4, dst = y * w * 4;
      for (let i = 0; i < w * 4; i += 4){ px[dst + i] = sharp[src + i + 3]; px[dst + i + 1] = soft[src + i + 3]; px[dst + i + 3] = 255; }
    }
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.generateMipmap(gl.TEXTURE_2D);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
    labRect = [G.offX - pad, G.offY - pad, G.bw + pad * 2, G.bh + pad * 2];
  }
  let altText = '';
  function buildLabels(){
    if (lost || gl.isContextLost() || !G.bw) return;
    drawLabel(lab0, labelText());
    drawLabel(lab1, altText);
    const cs0 = getComputedStyle(button);
    labelKey = labelText() + '|' + cs0.font;
    dirty = true;
  }
  function setAlt(text){ if (text !== altText){ altText = text; if (!lost && G.bw) drawLabel(lab1, altText); } }

  // ---- behaviour
  let state = 'rest';          // rest | burning | success | fail | smoulder | out | consume | regrow | settling
  let lit = false;             // the fire is running (or the strip is growing back): clicks are swallowed
  let req = null;              // the action of the current strike: { settled, outcome, ctl }
  let simT = 0, phaseT = 0, strikeT = 0, strikeX = 0, strikeY = 0, strikeWall = -1e9;
  let settleReq = null, consumeAt = Infinity;
  let simUntil = 0, simHot = false, sparksUntil = 0;
  let sick = 0, charA = 0, altA = 0, altStyle = 0, rim = 0, scorch = 0, warm = 0, ember = 0;
  let shakeT = -1;
  let smokeDark = 0, quench = 0, breeze = 0;
  let flareNext = 0, flareK = 1;
  let wickAmp = 0, wickC = 0, litX0 = 0, litX1 = 0, sputterHold = 0, sputterOn = 1;
  const splS = [], splV = [], wind = [];
  let near = false, nearT = 0, insideWall = -1e9;
  let steps = 0;
  // burning away
  let fireK = scaleOf(opts.scale), SP = scaleParams(fireK), relayoutWhenCold = false;
  const consumeMode = opts.consume === 'onSuccess' || opts.consume === 'always' ? opts.consume : 'never';
  const cancelable = opts.cancelable ?? (!!opts.onCancel || (typeof opts.pending === 'function' && opts.pending.length >= 2));
  let catchA = 0, fan = null, matDirty = false, healT = -1, regrowA = 1, ashT = -1, matAge = 0, left = true, feedA = 0, lightUntil = 0;

  const wickY = () => 1.5;                 // button-local y of the burning edge, just inside the top
  const wickSpan = () => { const inset = Math.min(G.bw * 0.3, radius * 0.55 + 7); return [inset, G.bw - inset]; };
  const flameH = () => Math.max(SP.flameHMin, G.bh * SP.flameHK);

  // ---- announcements for screen readers
  const live = document.createElement('div');
  live.setAttribute('role', 'status'); live.setAttribute('aria-live', 'polite');
  Object.assign(live.style, { position: 'absolute', width: '1px', height: '1px', margin: '-1px', padding: '0', overflow: 'hidden', clip: 'rect(0 0 0 0)', whiteSpace: 'nowrap', border: '0' });
  function announce(text){ live.textContent = ''; if (text) setTimeout(() => { live.textContent = text; }, 30); }

  // ---- sparks
  const spawnQ = [];
  let pNext = 0;
  function spark(x, y, vx, vy, life, heat, size){
    if (reduceMotion.matches) return;
    // a negative life marks a candidate the particle pass still has to accept (see FS_PART)
    spawnQ.push([x + G.offX, y + G.offY, vx, vy, Math.abs(life), life, heat, size]);
    sparksUntil = Math.max(sparksUntil, simT + Math.abs(life) + 0.1);
  }
  const bufA = new Float32Array(PW * 4), bufB = new Float32Array(PW * 4);
  function flushSparks(){
    if (!spawnQ.length) return;
    if (spawnQ.length > NPART) spawnQ.splice(0, spawnQ.length - NPART);
    // contiguous slots of the ring buffer, one upload per texture row
    let i = 0;
    while (i < spawnQ.length){
      const slot = pNext, row = (slot / PW) | 0, col = slot % PW;
      const n = Math.min(spawnQ.length - i, PW - col);
      for (let k = 0; k < n; k++){
        const s = spawnQ[i + k];
        bufA.set([s[0], s[1], s[2], s[3]], k * 4);
        bufB.set([s[4], s[5], s[6], s[7]], k * 4);
      }
      gl.bindTexture(gl.TEXTURE_2D, pA[pCur]); gl.texSubImage2D(gl.TEXTURE_2D, 0, col, row, n, 1, gl.RGBA, gl.FLOAT, bufA, 0);
      gl.bindTexture(gl.TEXTURE_2D, pB[pCur]); gl.texSubImage2D(gl.TEXTURE_2D, 0, col, row, n, 1, gl.RGBA, gl.FLOAT, bufB, 0);
      pNext = (pNext + n) % NPART;
      i += n;
    }
    spawnQ.length = 0;
  }

  function updateBusy(){ setBusy(lit || !!(req && !req.settled)); }
  const isBusy = () => lit || !!(req && !req.settled);

  // ---- the strike and what follows
  function strike(x, y){
    state = 'burning'; lit = true;
    strikeT = phaseT = simT; strikeX = x; strikeY = y; strikeWall = performance.now();
    settleReq = null; consumeAt = Infinity; fan = null;
    sick = 0; quench = 0; smokeDark = 0; charA = 0; altA = 0; rim = 0; shakeT = -1; ember = 0;
    scorch = 1;
    breeze = rand(-1, 1) * 12;
    const [a, b] = wickSpan();
    const hw = SP.halfW((b - a) / 2);
    wickC = Math.max(a + hw, Math.min(b - hw, x));
    litX0 = litX1 = Math.max(a, Math.min(b, x));
    simUntil = Infinity; simHot = true;
    const k = SP.k;
    // the head flares where it was struck
    splS.push({ x, y, r: 3 + 2 * k, dT: 1.4, dF: (reduceMotion.matches ? 0.6 : 1.3) * SP.flare, dK: 0, maxT: 1.5 });
    if (!reduceMotion.matches){
      splV.push({ x, y: y - 6, r: 8 + 8 * k, vx: 0, vy: -(110 + 110 * k), k: 1 });
      for (let i = 0; i < SP.sparkN; i++){
        const a = rand(-Math.PI, 0) + rand(-0.35, 0.35), sp = rand(90, 560) * SP.sparkV * (Math.random() < 0.2 ? 0.4 : 1);
        spark(x + rand(-3, 3), y + rand(-2, 2), Math.cos(a) * sp, Math.sin(a) * sp, rand(0.25, 0.95) * (0.6 + 0.4 * k), rand(0.75, 1), rand(0.8, 1.9) * SP.sparkSz);
      }
    }
    sound.strike(k);
    updateBusy();
    wake();
  }

  function begin(kind){
    phaseT = simT;
    settleReq = null;
    if (kind === 'success'){ state = 'success'; }
    else if (kind === 'fail'){ state = 'fail'; sputterHold = 0; sound.fizz(); }
    wake();
  }

  function flameOut(){
    lit = false;
    wickAmp = 0;
    simUntil = simT + 4.5;
    updateBusy();
  }

  function blowOut(){
    if (!cancelable || !lit || state !== 'burning' && !(state === 'success' && simT - phaseT < GUTTER) && !(state === 'fail' && simT - phaseT < SPUTTER)) return;
    // a gust right after the click, or from the finger that just left the button, is not a cancel
    const now = performance.now();
    if (now - strikeWall < BLOW_GRACE || now - insideWall < BLOW_GRACE) return;
    state = req && !req.settled ? 'smoulder' : 'out'; phaseT = simT;
    fan = null; catchA = 0;
    flameOut();
    quench = 7;
    smokeDark = 0;
    const [a, b] = [litX0, litX1];
    for (let i = 0; i < 3; i++) splS.push({ x: rand(a, b), y: wickY() - 3, r: 6, dT: 0.2, dF: 0, dK: 0.5, maxT: 0.3 });
    sound.blow();
    try { req?.ctl?.abort(); } catch {}
    try { opts.onCancel?.(); } catch (err) { console.error(err); }
    announce('Cancelled');
  }

  function requestSettle(kind, r){
    if (!r || r !== req || r.settled) return;
    r.settled = true; r.outcome = kind;
    if (state === 'burning') settleReq = kind;
    else if (state === 'smoulder'){ state = 'out'; phaseT = simT; }
    else if (state === 'settling') afterRegrow();
    updateBusy();
    wake();
  }

  // ---- burning the strip away
  function startConsume(){
    if (lost || state === 'consume' || state === 'regrow' || !Mt.length) return;
    if (!lit) strike(G.bw / 2, G.bh * 0.6);
    state = 'consume'; phaseT = simT;
    fan = null; catchA = 1; matDirty = true; healT = -1; ashT = -1; matAge = 0; left = true; feedA = 0; lightUntil = simT + 0.3;
    consumeAt = Infinity; settleReq = null;
    simUntil = Infinity; simHot = true;
    splV.push({ x: (litX0 + litX1) / 2, y: wickY() + 4, r: Math.max(14, (litX1 - litX0) * 0.4), vx: 0, vy: -200 * (0.5 + 0.5 * SP.k), k: 0.7 });
    sound.whoomp(SP.k);
    updateBusy();
    wake();
  }
  function startRegrow(){
    state = 'regrow'; phaseT = simT;
    regrowA = 0; catchA = 0; fan = null; healT = -1; ashT = -1; wickAmp = 0;
    initMat();
    simUntil = simT + 3;
    sound.regrow();
    wake();
  }
  function afterRegrow(){
    regrowA = 1;
    if (req && !req.settled){ state = 'settling'; return; }
    try { opts.onConsumed?.(req?.outcome ?? 'success'); } catch (err) { console.error(err); }
    if (req && req.outcome === 'fail'){
      state = 'fail'; phaseT = simT - SPUTTER; sick = 1;
      altStyle = 1; setAlt(opts.failText ?? 'Try again'); shakeT = simT;
      announce(opts.failText ?? 'Try again');
      return;
    }
    announce(opts.consumedText ?? 'Done');
    toRest();
  }

  function step(dt){
    simT += dt;
    // hover warmth
    const warmT = near && simT - nearT > 0.25 && !lit && state === 'rest' && !reduceMotion.matches ? 1 : 0;
    warm += (warmT - warm) * (1 - Math.exp(-dt / (warmT ? 0.45 : 0.3)));
    if (warm < 1e-3 && !warmT) warm = 0;
    const emberT = state === 'smoulder' ? 1 : 0;
    ember += (emberT - ember) * (1 - Math.exp(-dt / (emberT ? 0.3 : 0.5)));
    if (ember < 1e-3 && !emberT) ember = 0;

    const t = simT - phaseT;
    quench = Math.max(0, quench - dt * 12);
    let amp = 0;
    const [wx0, wx1] = wickSpan();
    if (state === 'burning'){
      const tS = simT - strikeT;
      const spread = 16 + 360 * tS, hw = SP.halfW((wx1 - wx0) / 2);
      const t0 = Math.max(wx0, wickC - hw), t1 = Math.min(wx1, wickC + hw);
      litX0 = Math.max(t0, Math.min(litX0, strikeX - spread)); litX1 = Math.min(t1, Math.max(litX1, strikeX + spread));
      litX0 = Math.min(litX0, t1); litX1 = Math.max(litX1, t0);
      amp = Math.min(1, tS / 0.12) * (1 + SP.flareOver * Math.exp(-tS / SP.flareT));
      if (tS < 0.14) splS.push({ x: strikeX, y: strikeY - 2, r: 3 + 1.5 * SP.k, dT: 0.5, dF: 6 * dt * SP.flare, dK: 0, maxT: 1.4 });
      if (SP.match > 0){
        // a match head: the sulphur flares and fizzes in spurts before the wood takes over; then a steady
        // thread of smoke leaves the flame's tip
        if (tS < 0.4){
          if (simT >= flareNext){ flareNext = simT + rand(0.03, 0.07); flareK = rand(0.45, 1.35); }
          amp *= 1 + (flareK - 1) * SP.match * (1 - tS / 0.4);
        }
        if (tS > 0.45 && state === 'burning'){
          const tipY = wickY() - flameH() * rand(0.85, 1.05);
          splS.push({ x: wickC + rand(-0.8, 0.8), y: tipY, r: 1.3, dT: 0.22, dF: 0, dK: 4.5 * dt * SP.wisp, maxT: 0.22 });
        }
      }
      if (Math.random() < dt * SP.ambient) spark(rand(litX0, litX1), wickY() - rand(4, 14) * (0.4 + 0.6 * SP.k), rand(-20, 20), rand(-90, -40), rand(0.7, 1.5), rand(0.55, 0.8), rand(0.7, 1.1) * SP.sparkSz);
      // held down on a burning button: air is fed in and the fire grows until the strip itself catches
      if (fan){
        catchA = Math.min(1, (simT - fan.t0) / SP.fanCatch);
        matDirty = true;
        if (simT >= fan.next){
          fan.next = simT + rand(0.2, 0.3);
          splV.push({ x: rand(litX0, litX1), y: wickY() + 5, r: 12 + 8 * SP.k, vx: rand(-30, 30), vy: -(150 + 130 * SP.k) * (0.6 + 0.6 * catchA), k: 0.6 });
          sound.fan(SP.k);
        }
        if (catchA >= 1){ startConsume(); amp = 1; }
      } else catchA = Math.max(0, catchA - dt / 0.35);
      if (state === 'burning'){
        amp *= 1 + 0.9 * catchA;
        if (!fan && settleReq && tS >= MIN_BURN){
          if (settleReq === 'success' && consumeMode !== 'never' || consumeMode === 'always') startConsume();
          else begin(settleReq);
        } else if (!fan && consumeMode === 'always' && tS >= MIN_BURN + 0.3) startConsume();
        else if (simT >= consumeAt) startConsume();
        else if (req && !req.settled && req.autoAt <= simT) requestSettle('success', req);
      }
    } else if (state === 'consume'){
      // The fire grows with what it has to eat: while most of the strip is still there to feed the front
      // it burns up to (1 + feedK) times as hard as the edge alone did, and falls away as the fuel runs out
      // (feedOf, on the GPU). The edge's own flame keeps going for as long as there is edge left to feed it.
      feedA = SP.feedK * Math.min(1, t / 0.35);
      amp = 1 + feedA;
      matAge = Math.max(0, t - SP.expect) * 1.5 + (reduceMotion.matches ? t * 1.2 : 0);
      if (ashT < 0 && (!left && t > 0.5 || t > SP.expect * 2.5 + 4)){ ashT = simT; }
      if (ashT >= 0 && simT - ashT > ASH_HOLD) startRegrow();
      if (ashT < 0){
        if (Math.random() < dt * 9 * SP.crackle) sound.crackle(0.8);
        // candidates for embers and ash all over the strip; the particle pass keeps the ones that land
        // where the strip is doing the right thing
        if (!reduceMotion.matches){
          for (let n = poisson(SP.emberRate * dt); n > 0; n--)
            spark(rand(0, G.bw), rand(0, G.bh), rand(-25, 25), rand(-130, -40) * (0.5 + 0.5 * SP.k), -rand(0.5, 1.3), rand(0.6, 0.95), rand(0.5, 1.0) * SP.sparkSz);
          for (let n = poisson(SP.ashRate * dt); n > 0; n--)
            spark(rand(0, G.bw), rand(0, G.bh), rand(-20, 20), rand(-70, -20), -rand(1.3, 2.6), -rand(1.0, 1.9), (0.8 + 2.6 * Math.pow(Math.random(), 2)) * SP.ashSz);
        }
      }
    } else if (state === 'regrow'){
      wickAmp = 0;
      regrowA = Math.min(1, t / REGROW);
      if (regrowA >= 1){ lit = false; wickAmp = 0; afterRegrow(); updateBusy(); }
    } else if (state === 'success'){
      if (t < GUTTER){
        const k = 1 - t / GUTTER;
        amp = k * (Math.random() < 0.3 ? 0.3 : 1);
      } else {
        if (lit){
          flameOut(); sound.snuff();
          altStyle = 0; setAlt(opts.successText ?? 'Done');
          announce(opts.successText ?? 'Done');
        }
        const ts = t - GUTTER;
        // the snuffed edge smoulders: two thin wisps of smoke
        if (ts < 1.1){
          const k = 1 - ts / 1.1;
          const mid = (litX0 + litX1) / 2, span = (litX1 - litX0) * 0.28;
          splS.push({ x: mid - span, y: wickY() - 2, r: 2.2, dT: 0.34 * k, dF: 0, dK: 1.6 * k * dt * 10, maxT: 0.34 });
          if (span > 4) splS.push({ x: mid + span * 0.8, y: wickY() - 2, r: 1.8, dT: 0.3 * k, dF: 0, dK: 1.1 * k * dt * 10, maxT: 0.3 });
        }
        charA = Math.min(1, ts / 0.2);
        altA = ts < SHOW + 0.2 ? Math.min(1, ts / 0.2) : Math.max(0, 1 - (ts - SHOW - 0.2) / 0.45);
        rim = Math.max(0, 1 - ts / (SHOW + 0.4));
        if (ts > SHOW + 0.2) charA = altA;
        scorch = Math.min(scorch, charA);
        if (ts > SHOW + 0.65) toRest();
      }
    } else if (state === 'fail'){
      sick = Math.min(1, t / 0.3);
      if (t < SPUTTER){
        // sputtering: the fuel comes in fits, each weaker than the last
        sputterHold -= dt;
        if (sputterHold <= 0){
          const on = sputterOn < 0.05;
          sputterOn = on ? rand(0.7, 1.3) * (1 - 0.6 * t / SPUTTER) : 0;
          sputterHold = on ? rand(0.07, 0.16) : rand(0.03, 0.09);
        }
        amp = sputterOn;
        if (Math.random() < dt * 20) sound.crackle(0.6);
      } else {
        if (lit){
          flameOut(); sound.puff();
          smokeDark = 1;
          const a = litX0, b = litX1;
          for (let i = 0; i < 3; i++) splS.push({ x: a + (b - a) * rand(0.15, 0.85), y: wickY() - rand(3, 8), r: rand(6, 12) * (0.5 + 0.5 * SP.k), dT: 0.45, dF: 0, dK: rand(1.0, 1.6), maxT: 0.45 });
          splV.push({ x: (a + b) / 2 + rand(-10, 10), y: wickY() - 6, r: Math.max(20, (b - a) * 0.35), vx: rand(-40, 40), vy: -170, k: 0.8 });
          shakeT = simT;
          altStyle = 1; setAlt(opts.failText ?? 'Try again');
          announce(opts.failText ?? 'Try again');
        }
        const ts = t - SPUTTER;
        charA = Math.min(1, ts / 0.15);
        altA = ts < SHOW ? Math.min(1, ts / 0.15) : Math.max(0, 1 - (ts - SHOW) / 0.45);
        if (ts > SHOW) charA = altA;
        scorch = Math.min(scorch, charA);
        if (ts > SHOW + 0.45) toRest();
      }
    } else if (state === 'out'){
      scorch = Math.max(0, scorch - dt * 2);
      if (t > 0.6) toRest();
    } else {
      scorch = Math.max(0, scorch - dt * 2);
    }
    if (state !== 'burning' && state !== 'consume') catchA = Math.max(0, catchA - dt / 0.35);
    if (healT >= 0 && simT - healT >= HEAL){ healT = -1; initMat(); }
    wickAmp += (amp - wickAmp) * (1 - Math.exp(-dt / 0.05));
    // a strip growing back has no burning edge, even on the step its burn-away ended
    if (!lit || state === 'regrow') wickAmp = 0;
    if (state !== 'fail') sick = Math.max(0, sick - dt * 3);
    if (simT >= simUntil && simHot){
      clearSim(); simUntil = 0; simHot = false;
      if (relayoutWhenCold){ relayoutWhenCold = false; G.stageW = G.stageH = 0; relayout(); }
    }
    const level = state === 'consume' ? (ashT < 0 ? 0.6 + 0.5 * SP.k : Math.max(0, 1 - (simT - ashT) / ASH_HOLD) * 0.5) : lit ? wickAmp * (1 - 0.5 * sick) : 0;
    roarLevel(level);
    if (lit && state !== 'consume' && Math.random() < dt * 9 * SP.crackle * wickAmp) sound.crackle(0.5 + wickAmp * 0.5);
  }

  function toRest(){
    state = 'rest';
    altA = 0; charA = 0; rim = 0; sick = 0;
    // scorching from a fire that was fed but never caught fades away
    if (matDirty && healT < 0) healT = simT;
  }

  function poisson(m){
    let n = 0, p = Math.exp(-Math.min(m, 30)), q = Math.random();
    while (q > p && n < 60){ q *= Math.random(); n++; }
    return n;
  }
  // is any of the strip left? An occlusion query, polled on later frames so nothing ever waits on the GPU
  function queryLeft(){
    if (leftPending || !leftQ) return;
    gl.colorMask(false, false, false, false);
    gl.beginQuery(gl.ANY_SAMPLES_PASSED_CONSERVATIVE, leftQ);
    pass(P.mleft, MFb[1 - mc], mLw, mLh, u => { bindTex(0, Mt[mc]); gl.uniform1i(u.uM, 0); gl.uniform2f(u.uRes, mLw, mLh); gl.uniform1f(u.uLod, LEFT_LOD); });
    gl.endQuery(gl.ANY_SAMPLES_PASSED_CONSERVATIVE);
    gl.colorMask(true, true, true, true);
    leftPending = true;
  }
  function pollLeft(){
    if (!leftPending || !gl.getQueryParameter(leftQ, gl.QUERY_RESULT_AVAILABLE)) return;
    leftPending = false;
    if (state === 'consume') left = !!gl.getQueryParameter(leftQ, gl.QUERY_RESULT);
  }
  // the test hook's path: _advance runs in one task, where a query result never becomes available; the
  // same test as FS_MLEFT, on the same mip
  function readLeftNow(){
    if (!MLvFb.length) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, MLvFb[mc]);
    gl.readPixels(0, 0, mLw, mLh, gl.RGBA, gl.FLOAT, rbBuf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const ss = (a, b, x) => { const t = Math.max(0, Math.min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t); };
    let any = false;
    for (let i = 0; i < rbBuf.length; i += 4) if (rbBuf[i] > 0.35 || rbBuf[i] * ss(0.3, 0.8, rbBuf[i + 2]) > 0.08){ any = true; break; }
    left = any;
  }

  // ---- roar: a steady filtered rumble and hiss that follows the flame
  let roar = null, roarStop = 0;
  function roarLevel(l){
    const a = l > 0 ? audio() : actx;
    if (!a || opts.sound === false) return;
    if (!roar){
      if (l <= 0) return;
      const src = a.createBufferSource(); src.buffer = noise(a); src.loop = true;
      const lp = a.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = 380; lp.Q.value = 0.7;
      const bp = a.createBiquadFilter(); bp.type = 'bandpass'; bp.frequency.value = 1600; bp.Q.value = 0.6;
      const g1 = a.createGain(), g2 = a.createGain(); g1.gain.value = 0; g2.gain.value = 0;
      src.connect(lp).connect(g1).connect(a.destination);
      src.connect(bp).connect(g2).connect(a.destination);
      src.start();
      roar = { src, g1, g2, lp, bp };
    }
    const v = fireAudio.muted ? 0 : Math.max(0, Math.min(l, 1.5));
    // a small flame is a quiet, higher hiss; a big one a low roar
    roar.lp.frequency.setTargetAtTime(SP.roarLp, a.currentTime, 0.1);
    roar.bp.frequency.setTargetAtTime(SP.roarBp, a.currentTime, 0.1);
    roar.g1.gain.setTargetAtTime(0.16 * v * SP.roar, a.currentTime, 0.06);
    roar.g2.gain.setTargetAtTime(0.018 * v * (0.6 + 0.4 * SP.k), a.currentTime, 0.06);
    if (v > 0){ clearTimeout(roarStop); roarStop = 0; }
    else if (!roarStop){
      roarStop = setTimeout(() => { roarStop = 0; if (roar){ try { roar.src.stop(); } catch {} roar = null; } }, 600);
    }
  }

  // ---- simulation
  const wU = new Float32Array(32), wV = new Float32Array(16), sU = new Float32Array(32), sA = new Float32Array(32);
  let seed = 0;
  function simulate(dt){
    steps++;
    const toV = (x, y) => [(x + G.offX) / cs, (G.stageH - (y + G.offY)) / cs];
    // pointer wind and impulses
    let nw = 0;
    for (const w of [...splV, ...wind]){
      if (nw >= 8) break;
      const [gx, gy] = toV(w.x, w.y);
      wU.set([gx, gy, w.r / cs, w.k], nw * 4);
      wV.set([w.vx / cs, -w.vy / cs], nw * 2);
      nw++;
    }
    splV.length = 0; wind.length = 0;

    pass(P.advV, velFb[1 - vc], vW, vH, u => {
      bindTex(0, vel[vc]); gl.uniform1i(u.uVel, 0);
      gl.uniform2f(u.uRes, vW, vH); gl.uniform1f(u.uDt, dt); gl.uniform1f(u.uDamp, Math.exp(-dt * 2.2));
    });
    vc = 1 - vc;
    pass(P.curl, curlFb, vW, vH, u => { bindTex(0, vel[vc]); gl.uniform1i(u.uVel, 0); });
    seed = (seed + 17.31) % 1000;
    pass(P.force, velFb[1 - vc], vW, vH, u => {
      bindTex(0, vel[vc]); gl.uniform1i(u.uVel, 0);
      bindTex(1, curlT); gl.uniform1i(u.uCurl, 1);
      bindTex(2, S[sc]); gl.uniform1i(u.uS, 2);
      gl.uniform2f(u.uRes, vW, vH); gl.uniform1f(u.uDt, dt);
      gl.uniform1f(u.uBuoy, SP.buoy / cs);
      bindTex(3, Mt[mc] || null); gl.uniform1i(u.uM, 3); gl.uniform1f(u.uFeedA, state === 'consume' ? feedA : 0); gl.uniform1f(u.uTopLod, mTop); gl.uniform1f(u.uWeight, 120 / cs);
      gl.uniform1f(u.uEps, SP.eps);
      gl.uniform1f(u.uCalm, SP.calm); gl.uniform1f(u.uLift, 90 / cs);
      gl.uniform1f(u.uJit, Math.min(SP.jit, reduceMotion.matches ? 300 : Infinity) / cs);
      gl.uniform1f(u.uSeed, seed);
      gl.uniform1f(u.uBreeze, (lit ? 0 : breeze) / cs);
      gl.uniform1f(u.uBreezeY, (G.stageH - G.offY + 20) / cs);
      gl.uniform1i(u.uNW, nw); gl.uniform4fv(u.uW, wU); gl.uniform2fv(u.uWV, wV);
    });
    vc = 1 - vc;
    pass(P.div, divFb, vW, vH, u => {
      bindTex(0, vel[vc]); gl.uniform1i(u.uVel, 0);
      bindTex(1, S[sc]); gl.uniform1i(u.uS, 1);
      gl.uniform2f(u.uRes, vW, vH); gl.uniform1f(u.uExpand, 1.6); gl.uniform1f(u.uKBurn, 14);
    });
    for (let i = 0; i < JACOBI; i++){
      pass(P.jacobi, prFb[1 - pc], vW, vH, u => {
        bindTex(0, pr[pc]); gl.uniform1i(u.uP, 0);
        bindTex(1, divT); gl.uniform1i(u.uDiv, 1);
      });
      pc = 1 - pc;
    }
    pass(P.grad, velFb[1 - vc], vW, vH, u => {
      bindTex(0, pr[pc]); gl.uniform1i(u.uP, 0);
      bindTex(1, vel[vc]); gl.uniform1i(u.uVel, 1);
    });
    vc = 1 - vc;

    // scalars
    const toD = (x, y) => [(x + G.offX) / cs * 2, (G.stageH - (y + G.offY)) / cs * 2];
    let ns = 0;
    for (const s of splS){
      if (ns >= 8) break;
      const [gx, gy] = toD(s.x, s.y);
      sU.set([gx, gy, s.r / cs * 2, s.r / cs * 2], ns * 4);
      sA.set([s.dT, s.dF, s.dK, s.maxT], ns * 4);
      ns++;
    }
    splS.length = 0;
    const [wx0] = toD(litX0, wickY()), [wx1, wy] = toD(litX1, wickY());
    pass(P.advS, SFb[1 - sc], dW, dH, u => {
      bindTex(0, S[sc]); gl.uniform1i(u.uS, 0);
      bindTex(1, vel[vc]); gl.uniform1i(u.uVel, 1);
      gl.uniform2f(u.uRes, dW, dH); gl.uniform1f(u.uDt, dt); gl.uniform1f(u.uVtoD, 2); gl.uniform1f(u.uTime, simT);
      gl.uniform1f(u.uKBurn, SP.kBurn); gl.uniform1f(u.uHeat, SP.heat); gl.uniform1f(u.uSootY, 0.06); gl.uniform1f(u.uKOx, 8);
      gl.uniform1f(u.uCool, 3.2); gl.uniform1f(u.uRad, 2.2); gl.uniform1f(u.uSootDecay, SP.sootDecay); gl.uniform1f(u.uFuelDecay, 1.5);
      gl.uniform1f(u.uQuench, quench);
      const wick = lit && wickAmp > 0.01;
      gl.uniform4f(u.uWick, wx0, wx1, wy, Math.max(2.5, 3.2 / cs * 2));
      gl.uniform4f(u.uWick2, wick ? SP.fuel * wickAmp : 0, wick ? 0.62 * Math.min(1, wickAmp * 1.5) : 0, 0.09 * cs / 2, SP.flick);
      gl.uniform1f(u.uSteady, SP.steady);
      bindTex(2, Mt[mc] || null); gl.uniform1i(u.uM, 2);
      gl.uniform1f(u.uMatOn, matOn() ? 1 : 0);
      gl.uniform4f(u.uMRect, G.offX, G.offY, G.bw, G.bh);
      gl.uniform4f(u.uMat, cs / 2, G.stageH, SP.mFuel, SP.mPilot);
      gl.uniform1f(u.uMLift, SP.mLift); gl.uniform1f(u.uFeedA, state === 'consume' ? feedA : 0); gl.uniform1f(u.uTopLod, mTop);
      gl.uniform1i(u.uNS, ns); gl.uniform4fv(u.uSp, sU); gl.uniform4fv(u.uSa, sA);
    });
    sc = 1 - sc;
    pass(P.emit, emisFb, vW, vH, u => {
      bindTex(0, S[sc]); gl.uniform1i(u.uS, 0);
      gl.uniform2f(u.uRes, vW, vH);
    });
    gl.bindTexture(gl.TEXTURE_2D, emis); gl.generateMipmap(gl.TEXTURE_2D);
    if (matOn()) burnStrip(dt);
  }

  const matOn = () => matDirty && state !== 'regrow' && Mt.length > 0;
  let mReads = 0;
  function burnStrip(dt){
    // Explicit diffusion, with the grid's worst-case conductivity (FS_MAT: uD * (uKMin + uKVar)): the stencil
    // is stable for h * k <= 1/4, and a long frame taken in too few substeps blows the heat up and lights the
    // whole strip at once. A frame too long even for the most substeps allowed runs the strip slower instead.
    const kMax = SP.mD * ((SP.kMin ?? 0.06) + (SP.kVar ?? 3.2)) / (mCellX * mCellY);
    const hMax = 0.18 / kMax;
    const n = Math.min(24, Math.max(1, Math.ceil(dt / hMax)));
    const h = Math.min(dt / n, hMax);
    for (let i = 0; i < n; i++){
      pass(P.mat, MFb[1 - mc], mW, mH, u => {
        bindTex(0, Mt[mc]); gl.uniform1i(u.uM, 0);
        bindTex(1, S[sc]); gl.uniform1i(u.uS, 1);
        gl.uniform2f(u.uMRes, mW, mH); gl.uniform2f(u.uGridPx, vW * cs, vH * cs); gl.uniform2f(u.uOff, G.offX, G.offY);
        gl.uniform3f(u.uMGeom, G.bw, G.bh, G.stageH); gl.uniform2f(u.uCell, mCellX, mCellY);
        gl.uniform1f(u.uDt, h); gl.uniform1f(u.uCatch, catchA);
        gl.uniform1f(u.uD, SP.mD); gl.uniform1f(u.uGen, SP.mGen); gl.uniform1f(u.uCool, SP.mCool); gl.uniform1f(u.uBurn, SP.mBurn);
        gl.uniform1f(u.uIgn, SP.mIgn); gl.uniform1f(u.uGasK, SP.mGasK); gl.uniform1f(u.uAge, matAge);
        gl.uniform4f(u.uLight, litX0 - 2, litX1 + 2, state === 'consume' && simT < lightUntil ? 8 : 0, 2.5);
        gl.uniform1f(u.uOx, SP.mOx); gl.uniform1f(u.uSmoulder, SP.mSmoulder); gl.uniform1f(u.uKMin, SP.kMin ?? 0.06); gl.uniform1f(u.uKVar, SP.kVar ?? 3.2);
      });
      mc = 1 - mc;
    }
    gl.bindTexture(gl.TEXTURE_2D, Mt[mc]); gl.generateMipmap(gl.TEXTURE_2D);
    if (state === 'consume' && ++mReads % 4 === 0){ if (manual) readLeftNow(); else queryLeft(); }
  }

  function stepSparks(dt){
    flushSparks();
    pass(P.part, pFb[1 - pCur], PW, PH, u => {
      bindTex(0, pA[pCur]); gl.uniform1i(u.uA, 0);
      bindTex(1, pB[pCur]); gl.uniform1i(u.uB, 1);
      bindTex(2, vel[vc]); gl.uniform1i(u.uVel, 2);
      gl.uniform2f(u.uGridPx, vW * cs, vH * cs); gl.uniform2f(u.uStage, G.stageW, G.stageH);
      gl.uniform1f(u.uCs, cs); gl.uniform1f(u.uDt, dt);
      bindTex(3, Mt[mc] || null); gl.uniform1i(u.uM, 3);
      gl.uniform4f(u.uMRect, G.offX, G.offY, G.bw, G.bh);
      gl.uniform1f(u.uSeedP, Math.random() * 100);
    });
    pCur = 1 - pCur;
  }

  // ---- render
  function render(time){
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, W, H);
    const c = P.comp;
    gl.useProgram(c.p);
    bindTex(0, S[sc]); gl.uniform1i(c.u.uS, 0);
    bindTex(1, emis); gl.uniform1i(c.u.uEmis, 1);
    bindTex(2, lab0); gl.uniform1i(c.u.uLab0, 2);
    bindTex(3, lab1); gl.uniform1i(c.u.uLab1, 3);
    gl.uniform2f(c.u.uStage, G.stageW, G.stageH);
    gl.uniform2f(c.u.uGridPx, vW * cs, vH * cs);
    gl.uniform1f(c.u.uDpr, dpr); gl.uniform1f(c.u.uDprY, dprY);
    gl.uniform4f(c.u.uBtn, G.offX, G.offY, G.bw, G.bh);
    gl.uniform1f(c.u.uRadius, radius);
    gl.uniform4f(c.u.uLabRect, ...labRect);
    gl.uniform1f(c.u.uFlameOn, simHot ? 1 : 0);
    gl.uniform1f(c.u.uWarm, warm);
    gl.uniform1f(c.u.uEmber, ember);
    gl.uniform1f(c.u.uDis, button.disabled ? 1 : 0);
    bindTex(4, Mt[mc] || null); gl.uniform1i(c.u.uM, 4);
    gl.uniform1f(c.u.uMatOn, matDirty && Mt.length ? 1 : 0);
    gl.uniform1f(c.u.uHeal, healT >= 0 ? Math.min(1, (simT - healT) / HEAL) : 0);
    gl.uniform1f(c.u.uRegrow, state === 'regrow' ? regrowA : 1);
    gl.uniform1f(c.u.uPreLod, 1.5 + 1.8 * SP.k);
    gl.uniform1f(c.u.uTime, time % 1000);
    gl.uniform1f(c.u.uChar, charA);
    gl.uniform1f(c.u.uAlt, altA);
    gl.uniform1f(c.u.uAltStyle, altStyle);
    gl.uniform1f(c.u.uRim, rim);
    gl.uniform1f(c.u.uSmokeDark, smokeDark);
    let sx = 0;
    if (shakeT >= 0){
      const ts = simT - shakeT;
      if (ts < 0.5) sx = 3.2 * Math.exp(-ts * 7) * Math.sin(ts * 55); else shakeT = -1;
    }
    gl.uniform2f(c.u.uShake, sx, 0);
    gl.uniform3f(c.u.uScorch, strikeX, strikeY, scorch * (reduceMotion.matches ? 0 : 0.3));
    gl.bindVertexArray(vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (simT < sparksUntil){
      const s = P.spark;
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.useProgram(s.p);
      bindTex(0, pA[pCur]); gl.uniform1i(s.u.uA, 0);
      bindTex(1, pB[pCur]); gl.uniform1i(s.u.uB, 1);
      gl.uniform2f(s.u.uStage, G.stageW, G.stageH); gl.uniform1i(s.u.uPW, PW);
      gl.drawArrays(gl.TRIANGLES, 0, NPART * 6);
      gl.disable(gl.BLEND);
    }
  }

  // ---- input
  let pressed = null;                // { id, x, y, lx, ly }
  let lastUp = null;                 // where the last pointer release was, for the click that follows
  const local = (cx, cy) => { const r = btnRect || button.getBoundingClientRect(); return [cx - r.left, cy - r.top]; };

  function onBtnDown(e){
    if (lost || e.button !== 0 || !e.isPrimary || button.disabled) return;
    audio();
    const [x, y] = local(e.clientX, e.clientY);
    pressed = { id: e.pointerId, x, y, lx: x, ly: y, travel: 0 };
    // pressing on a burning button fans it: held long enough, the strip itself catches
    if (lit && state === 'burning' && !reduceMotion.matches){ fan = { id: e.pointerId, t0: simT, next: simT + 0.05 }; matDirty = true; healT = -1; wake(); }
  }
  function onBtnMove(e){
    if (!pressed || e.pointerId !== pressed.id || lit) return;
    const [x, y] = local(e.clientX, e.clientY);
    const d = Math.hypot(x - pressed.lx, y - pressed.ly);
    pressed.travel += d;
    // dragging a finger along the striker scratches sparks off it
    if (pressed.travel > 7 && d > 0.5){
      pressed.travel = 0;
      const vx = (x - pressed.lx) * 25, vy = (y - pressed.ly) * 25;
      for (let i = 0; i < 2; i++) spark(x, y, vx * rand(0.3, 0.9) + rand(-60, 60), vy * 0.3 + rand(-160, -30), rand(0.1, 0.3), rand(0.7, 1), rand(0.6, 1));
      sound.scratch();
      wake();
    }
    pressed.lx = x; pressed.ly = y;
  }
  function onBtnUp(e){
    if (!pressed || e.pointerId !== pressed.id) return;
    const [x, y] = local(e.clientX, e.clientY);
    pressed = null;
    if (fan && fan.id === e.pointerId) fan = null;
    lastUp = { x: Math.max(4, Math.min(G.bw - 4, x)), y: Math.max(4, Math.min(G.bh - 4, y)), t: performance.now() };
  }
  // Holding Space or Enter on a burning button fans it too, once the key has been down a moment: a fresh
  // press, so the Enter that struck it and is still held down never feeds it by accident.
  let keyHold = null;
  const feedKey = e => e.key === ' ' || e.key === 'Enter';
  function onBtnKeyDown(e){
    if (!feedKey(e) || lost || button.disabled || reduceMotion.matches) return;
    if (!e.repeat){ keyHold = lit && state === 'burning' ? { key: e.key, t0: performance.now() } : null; return; }
    if (!keyHold || keyHold.key !== e.key || fan || !(lit && state === 'burning')) return;
    if (performance.now() - keyHold.t0 > 400){ fan = { id: 'key', t0: simT, next: simT + 0.05 }; matDirty = true; healT = -1; wake(); }
  }
  function endKeyFan(){ keyHold = null; if (fan && fan.id === 'key') fan = null; }
  function onBtnKeyUp(e){ if (keyHold && e.key === keyHold.key) endKeyFan(); }
  function onBtnCancel(e){ if (pressed && e.pointerId === pressed.id) pressed = null; if (fan && fan.id === e.pointerId) fan = null; }
  // a press that wanders off the button still ends when it is released
  function docUp(e){ if (fan && fan.id === e.pointerId) fan = null; if (pressed && pressed.id === e.pointerId) pressed = null; }

  function onClick(e){
    // one click per strike: nothing gets through while the fire or its action is still running
    if (isBusy()){ e.preventDefault(); e.stopImmediatePropagation(); return; }
    if (lost) return;
    audio();
    let x = G.bw / 2, y = G.bh * 0.6;
    if (e.detail > 0 && lastUp && performance.now() - lastUp.t < 1000){ x = lastUp.x; y = lastUp.y; }
    lastUp = null;
    const r = req = { settled: false, outcome: null, ctl: typeof AbortController === 'function' ? new AbortController() : null, autoAt: Infinity };
    strike(x, y);
    if (opts.pending){
      let p;
      try { p = opts.pending(e, { signal: r.ctl?.signal }); } catch (err) { p = Promise.reject(err); }
      Promise.resolve(p).then(() => requestSettle('success', r), err => { requestSettle('fail', r); if (err) console.debug('ElementalFX fire: pending action failed', err); });
    } else r.autoAt = strikeT + DEFAULT_BURN;
  }

  // pointer near the flame: wind, blow-outs and hover warmth, from the document since the canvas never takes events
  let lastMove = null;
  function docMove(e){
    if (lost) return;
    const r = btnRect || button.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    const dx = Math.max(-x, 0, x - r.width), dy = Math.max(-y, 0, y - r.height);
    const isNear = Math.hypot(dx, dy) < 26;
    if (dx === 0 && dy === 0) insideWall = performance.now();
    if (isNear !== near){ near = isNear; if (near) nearT = simT; wake(); }
    const now = e.timeStamp;
    const prev = lastMove && lastMove.id === e.pointerId && now - lastMove.t < 100 ? lastMove : null;
    lastMove = { id: e.pointerId, x, y, t: now };
    if (!prev || !simHot) return;
    const dtm = Math.max((now - prev.t) / 1000, 1 / 240);
    const vx = (x - prev.x) / dtm, vy = (y - prev.y) / dtm;
    const sp = Math.hypot(vx, vy);
    if (sp < 5) return;
    const inStage = x > -G.offX && y > -G.offY && x < G.stageW - G.offX && y < G.stageH - G.offY;
    if (!inStage) return;
    // air moves with the pointer, a little behind it
    const k = Math.min(1, 0.45 + sp / 3000);
    wind.push({ x, y, r: 14, vx: vx * 0.55, vy: vy * 0.55, k });
    if (wind.length > 6) wind.splice(0, wind.length - 6);
    if (sp > BLOW_SPEED && lit && crossesFlame(prev.x, prev.y, x, y)){
      wind.push({ x: (prev.x + x) / 2, y: (prev.y + y) / 2, r: 26, vx: vx * 0.6, vy: vy * 0.6, k: 1 });
      blowOut();
    }
    wake();
  }
  // does the segment pass through the zone the flame occupies above the burning edge?
  function crossesFlame(x0, y0, x1, y1){
    const zx0 = litX0 - 8, zx1 = litX1 + 8, zy0 = wickY() - flameH(), zy1 = wickY() + 6;
    let t0 = 0, t1 = 1;
    const dx = x1 - x0, dy = y1 - y0;
    for (const [p, q] of [[-dx, x0 - zx0], [dx, zx1 - x0], [-dy, y0 - zy0], [dy, zy1 - y0]]){
      if (p === 0){ if (q < 0) return false; continue; }
      const r = q / p;
      if (p < 0){ if (r > t1) return false; if (r > t0) t0 = r; }
      else { if (r < t0) return false; if (r < t1) t1 = r; }
    }
    return (t1 - t0) * Math.hypot(dx, dy) > 10;
  }
  function docLeave(){ if (near){ near = false; wake(); } lastMove = null; }

  const cap = { capture: true, passive: true };
  button.addEventListener('pointerdown', onBtnDown);
  button.addEventListener('pointermove', onBtnMove);
  button.addEventListener('pointerup', onBtnUp);
  button.addEventListener('pointercancel', onBtnCancel);
  button.addEventListener('keydown', onBtnKeyDown);
  button.addEventListener('keyup', onBtnKeyUp);
  button.addEventListener('blur', endKeyFan);
  button.addEventListener('click', onClick, true);
  document.addEventListener('pointermove', docMove, cap);
  document.addEventListener('pointerup', docUp, cap);
  document.addEventListener('pointercancel', docUp, cap);
  document.documentElement.addEventListener('pointerleave', docLeave);

  // ---- loop
  let raf = 0, inFrame = false, frameCount = 0, last = performance.now(), dirty = true, manual = false, lost = false, visible = true;
  function busy(){
    return state !== 'rest' && state !== 'settling' || lit || simHot || simT < sparksUntil || near || warm > 0 || ember > 0 || scorch > 0
      || shakeT >= 0 || healT >= 0 || !!fan;
  }
  function wake(){
    // one animation loop at most: a wake from inside a frame must not start a second loop
    if (!raf && !inFrame && !manual && !lost) raf = requestAnimationFrame(frame);
  }
  function advance(dt, time){
    if (!manual && leftQ) pollLeft();
    step(dt);
    if (lost || gl.isContextLost() || !W) return false;
    if (simHot && dt > 0) simulate(dt);
    if (simT < sparksUntil + 0.05 && dt > 0) stepSparks(dt);
    if (visible || dirty){ render(time); dirty = false; }
    return true;
  }
  function frame(now){
    raf = 0;
    if (manual || lost) return;
    const dt = Math.min((now - last) / 1000, 1 / 30);
    last = now;
    place();
    inFrame = true;
    frameCount++;
    let more = false;
    try { advance(dt, now / 1000); more = busy(); } finally { inFrame = false; }
    if (more) raf = requestAnimationFrame(frame);
    else { dirty = true; last = performance.now(); }
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
  const onDpr = () => { watchDpr(); relayout(); resizeGL(); wake(); };
  function watchDpr(){
    dprMq?.removeEventListener('change', onDpr);
    dprMq = matchMedia(`(resolution: ${devicePixelRatio}dppx)`);
    dprMq.addEventListener('change', onDpr);
  }
  watchDpr();
  const io = new IntersectionObserver(es => { for (const e of es) visible = e.isIntersecting; if (visible){ dirty = true; wake(); } });
  io.observe(canvas);
  const onFonts = () => { if (!lost && labelText()){ buildLabels(); wake(); } };
  document.fonts?.addEventListener?.('loadingdone', onFonts);
  const mo = new MutationObserver(() => {
    if (lost) return;
    const cs0 = getComputedStyle(button);
    if (labelText() + '|' + cs0.font !== labelKey){ buildLabels(); wake(); }
    else { dirty = true; wake(); }
  });
  mo.observe(button, { childList: true, characterData: true, subtree: true, attributes: true, attributeFilter: ['class', 'disabled'] });

  // A lost context shows the plain button until the GPU comes back, then the stage is rebuilt.
  function onLost(e){
    e.preventDefault();
    lost = true;
    cancelAnimationFrame(raf); raf = 0;
    lit = false; fan = null; catchA = 0; healT = -1; regrowA = 1; matDirty = false; leftPending = false;
    state = 'rest'; simHot = false; roarLevel(0);
    updateBusy();
    showButton();
    canvas.style.visibility = 'hidden';
  }
  function onRestored(){
    try { initGL(); } catch (err) { console.warn('ElementalFX fire: could not rebuild after context loss', err); return; }
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
  document.body.appendChild(live);
  relayout();
  advance(0, performance.now() / 1000);

  return {
    supported: true,
    get state(){ return lost ? 'lost' : state; },
    get scale(){ return fireK; },
    refresh(){ if (!lost){ relayout(); buildLabels(); dirty = true; wake(); } },
    burn(){
      if (lost || isBusy()) return;
      audio();
      req = { settled: false, outcome: null, ctl: null, autoAt: Infinity };
      strike(G.bw / 2, G.bh * 0.6);
    },
    succeed(){ requestSettle('success', req); },
    fail(){ requestSettle('fail', req); },
    // burn the strip away now; it grows back once the fire is out (and any pending action has settled)
    consume(){
      if (lost || state === 'consume' || state === 'regrow') return false;
      if (lit && state === 'burning'){ startConsume(); return true; }
      if (isBusy()) return false;
      audio();
      req = { settled: true, outcome: 'success', ctl: null, autoAt: Infinity };
      strike(G.bw / 2, G.bh * 0.6);
      consumeAt = simT + 0.3;
      return true;
    },
    setScale(v){
      const k = scaleOf(v);
      if (k === fireK) return;
      fireK = k; SP = scaleParams(k);
      // the stage and grid follow the scale, but rebuilding them would wipe a live fire, so that waits
      if (simHot) relayoutWhenCold = true;
      else if (!lost){ G.stageW = G.stageH = 0; relayout(); }
      wake();
    },
    detach(){
      cancelAnimationFrame(raf); raf = 0; manual = true;
      roarLevel(0);
      live.remove();
      document.removeEventListener('pointerup', docUp, cap); document.removeEventListener('pointercancel', docUp, cap);
      leftPending = false;
      ro.disconnect(); io.disconnect(); mo.disconnect();
      dprMq?.removeEventListener('change', onDpr);
      document.fonts?.removeEventListener?.('loadingdone', onFonts);
      removeEventListener('resize', relayout);
      document.removeEventListener('scroll', onScroll, { capture: true });
      button.removeEventListener('pointerdown', onBtnDown); button.removeEventListener('pointermove', onBtnMove);
      button.removeEventListener('pointerup', onBtnUp); button.removeEventListener('pointercancel', onBtnCancel);
      button.removeEventListener('keydown', onBtnKeyDown); button.removeEventListener('keyup', onBtnKeyUp); button.removeEventListener('blur', endKeyFan);
      button.removeEventListener('click', onClick, true);
      document.removeEventListener('pointermove', docMove, cap);
      document.documentElement.removeEventListener('pointerleave', docLeave);
      canvas.removeEventListener('webglcontextlost', onLost); canvas.removeEventListener('webglcontextrestored', onRestored);
      setBusy(false);
      showButton();
      canvas.remove();
      gl.getExtension('WEBGL_lose_context')?.loseContext();
    },
    // test hooks
    get _frames(){ return frameCount; },
    get _steps(){ return steps; },
    _manual(on){ manual = on; last = performance.now(); if (!on) wake(); else { cancelAnimationFrame(raf); raf = 0; } },
    _advance(sec, fps = 60){ place(); for (let i = 0, n = Math.round(sec * fps); i < n; i++) advance(1 / fps, simT + 1 / fps); },
    _read(which = 'S'){
      const fb = which === 'S' ? SFb[sc] : which === 'V' ? velFb[vc] : emisFb, w = which === 'S' ? dW : vW, h = which === 'S' ? dH : vH;
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      const px = new Float32Array(w * h * 4);
      gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, px);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      const mx = [0, 0, 0, 0], sum = [0, 0, 0, 0];
      for (let i = 0; i < px.length; i++){ mx[i & 3] = Math.max(mx[i & 3], Math.abs(px[i])); sum[i & 3] += px[i]; }
      return { w, h, max: mx.map(v => +v.toFixed(3)), mean: sum.map(v => +(v / (w * h)).toFixed(4)) };
    },
    _readS(){
      gl.bindFramebuffer(gl.FRAMEBUFFER, SFb[sc]);
      const px = new Float32Array(dW * dH * 4);
      gl.readPixels(0, 0, dW, dH, gl.RGBA, gl.FLOAT, px);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return { w: dW, h: dH, cell: cs / 2, px: Array.from(px) };
    },
    _tune(o){ Object.assign(SP, o); return SP; },
    _readMat(){
      gl.bindFramebuffer(gl.FRAMEBUFFER, MFb[mc]);
      const px = new Float32Array(mW * mH * 4);
      gl.readPixels(0, 0, mW, mH, gl.RGBA, gl.FLOAT, px);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      return { w: mW, h: mH, px: Array.from(px) };
    },
    get _mat(){ return { left, matDirty, catchA, regrowA, fan: !!fan, grid: [mW, mH, mLw, mLh] }; },
    get _sim(){ return { G, get lit(){ return lit; }, get simHot(){ return simHot; }, get wickAmp(){ return wickAmp; }, get cs(){ return cs; }, grid: () => [vW, vH, dW, dH], strike, blowOut, wind }; },
  };
}

window.ElementalFX = Object.assign(window.ElementalFX || {}, { attachFire, fireAudio });
})();
