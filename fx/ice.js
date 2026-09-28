// Ice button: turns an existing <button> into a slab of clear ice with a light snowfall settling on it.
//
//   <script src="fx/ice.js"></script>
//   const fx = ElementalFX.attachIce(document.querySelector('#freeze'));   // fx.detach() restores it
//
// A classic script (not a module) so pages work straight from file:// too.
//
// The <button> stays the real control (focus, keyboard, form submission, click handlers); this only
// draws it, in a transparent canvas laid over the page around it. Holding the button down strains the
// slab (white stress crazing spreads under the finger); a click breaks it in place into big chunks,
// with the frozen-lake "pew" ringing through the sheet. Clicking a chunk then crushes that one flat into
// the page, leaving glittering ice chips and crushed powder frozen onto the surface. Dragging through
// the ice scrapes it: chunks shove along with friction, stuck chips resist, then break loose and skid
// leaving frost trails, and powder smears. After a quiet moment the pieces draw back together and the
// slab regrows as frost crystals. Pointer presses on the broken ice never click anything. The cursor is
// the light: glints follow it.
// If WebGL2 is unavailable, the button is left untouched.
//
// Options:
//   surface: { image: CanvasImageSource, rect: () => DOMRect }   what the page looks like under the ice
//            (refracted through it). Without it the ice refracts the nearest ancestor's background colour.
//   snow: false to stop the snowfall.   snowIdle: seconds without any page activity before the snowfall
//   pauses and the loop sleeps (default 8).   sound: false to stay silent.
//   refreezeAfter: seconds of quiet before it refreezes (default 3.2).   zIndex: stage canvas (default 10).

(() => {
'use strict';

const TAU = Math.PI * 2;
const CS = 2;                  // CSS px per powder-grid cell
const SNOW_PX = 4;             // powder height (CSS px) that maps to a full R8 texel
const DRAG_START = 6;
const PUSHER_R = 10;
const MU = 650;                // kinetic friction of a skidding chip, CSS px/s^2
const CRUSH_T = 0.22, REGROW_T = 1.15;
const LIGHT_H = 150;           // height of the cursor "lamp" above the page, CSS px

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
// a tiny facet catching the cursor lamp: sharp enough that only a few of many random facets fire at once
float glintOf(vec3 n, vec2 at, vec2 lamp, float sharp){
  vec3 L = normalize(vec3(lamp - at, ${LIGHT_H.toFixed(1)}));
  vec3 H = normalize(L + vec3(0., 0., 1.));
  return pow(max(dot(n, H), 0.), sharp);
}
`;

// One vertex stream serves both the chunk bodies and their shadows. Every fan triangle belongs to one
// polygon edge, so aEdge.x interpolates to the exact distance from that edge's line.
const VS_SHARD = `#version 300 es
in vec2 aPos;      // stage CSS px, y down (already lifted by the chunk's height)
in vec2 aHome;     // button-local CSS px of the unbroken slab
in vec4 aEdge;     // distance to edge line, outward normal xy, kind (0 rim, 1 crack)
in vec4 aInfo;     // crack reveal distance, height, seed, ground offset y (lift)
in vec4 aX;        // crush (0..1), fade, unused, unused
uniform vec2 uStage;
uniform float uShadow;
uniform vec2 uShadowDir;
out vec2 vHome; out vec4 vEdge; out vec4 vInfo; out vec4 vX; out vec2 vStage;
void main(){
  vec2 p = aPos;
  if (uShadow > 0.5) p += vec2(0., aInfo.w) + uShadowDir * (4. + aInfo.y * 0.9) * (1. - aX.x);
  vStage = p; vHome = aHome; vEdge = aEdge; vInfo = aInfo; vX = aX;
  gl_Position = vec4(p.x / uStage.x * 2. - 1., 1. - p.y / uStage.y * 2., 0., 1.);
}`;

const FS_SHADOW = COMMON + `
in vec2 vHome; in vec4 vEdge; in vec4 vInfo; in vec4 vX; in vec2 vStage;
uniform float uPress, uFlat;
out vec4 o;
void main(){
  float soft = 5. + vInfo.y * 0.3;
  float a = smoothstep(0., soft, vEdge.x) * (0.42 - vInfo.y * 0.004);
  // light focused through the slab lands as a wobbly bright seam inside the cold shadow
  float c = pow(vnoise(vHome * vec2(0.09, 0.14) + vInfo.z * 5.), 3.) * smoothstep(0., 7., vEdge.x) * smoothstep(20., 6., vEdge.x);
  float k = (1. - uPress * 0.3) * (1. - uFlat) * (1. - vX.x) * vX.y;
  o = vec4((vec3(0.02, 0.07, 0.16) * a + vec3(0.55, 0.85, 1.0) * c * 0.35) * k, a * k);
}`;

const FS_ICE = COMMON + `
in vec2 vHome; in vec4 vEdge; in vec4 vInfo; in vec4 vX; in vec2 vStage;
uniform sampler2D uBgTex; uniform sampler2D uText;
uniform float uHasBg; uniform vec3 uBgCol;
uniform vec4 uBg;          // stage->surface: offset xy, size zw (CSS px)
uniform vec2 uSize;        // button CSS px
uniform float uCrackR, uPress, uTime, uCraze, uGrow, uGrowOn, uReach;
uniform vec2 uImpact;      // home px
uniform vec2 uLamp;        // stage px
uniform vec4 uPlanes[4];   // fracture planes: point xy, direction zw (home px)
uniform vec2 uSeeds[3];    // regrowth nuclei (home px)
out vec4 o;
vec3 bgAt(vec2 sp){
  if (uHasBg < 0.5) return uBgCol;
  return texture(uBgTex, (sp + uBg.xy) / uBg.zw).rgb;
}
void main(){
  vec2 h = vHome;
  float d = vEdge.x;
  vec2 N = vEdge.yz;
  bool rim = vEdge.w < 0.5;
  float crush = vX.x;
  float bev = rim ? 9. : 3.2;
  float x = clamp(1. - d / bev, 0., 1.);
  float slope = (rim ? x * x * 1.35 : smoothstep(0., 1., x) * 0.75) * (1. - crush * 0.7);
  vec2 wav = vec2(fbm(h * 0.045 + vInfo.z), fbm(h * 0.045 + 11.3 + vInfo.z)) - 0.5;
  vec3 n = normalize(vec3(N * slope + wav * 0.12, 1.));

  vec2 tilt = (hash22(vec2(vInfo.z * 91.7)) - 0.5) * 3. * step(0.001, vInfo.z);
  vec2 off = n.xy * 16. + wav * 5. + tilt;
  vec3 bg = bgAt(vStage - off);
  float thick = (0.55 + 0.45 * smoothstep(0., 22., d)) * (1. - crush * 0.5);
  vec3 tint = mix(vec3(1.), vec3(0.62, 0.84, 1.0), thick);
  vec3 col = bg * tint * 0.9 + vec3(0.07, 0.19, 0.30) * thick;
  col += vec3(0.05, 0.15, 0.24) * smoothstep(0.2, 1.1, (h.x / uSize.x + h.y / uSize.y) * 0.6) * thick;

  // internal fracture veils: planes in the ice seen at a slant, sharp on one side and fading away
  for (int i = 0; i < 4; i++){
    vec4 P = uPlanes[i];
    vec2 q = h - P.xy;
    float s = dot(q, vec2(-P.w, P.z));
    float along = dot(q, P.zw);
    float gate = smoothstep(0.35, 0.7, vnoise(vec2(along * 0.035, float(i) * 3.1))) * step(0., s);
    float veil = exp(-s / (4. + 5. * vnoise(vec2(along * 0.08, 1.7 + float(i))))) * gate;
    float edgeLine = smoothstep(1.1, 0., s) * gate;
    vec3 irid = 0.5 + 0.5 * cos(6.2831 * (along * 0.012 + vec3(0., 0.33, 0.67)));
    col = mix(col, vec3(0.85, 0.95, 1.0), veil * 0.22);
    col += (vec3(0.8, 0.93, 1.0) + irid * 0.15) * edgeLine * 0.35;
  }

  // trapped bubbles
  {
    vec2 g = h / 9., gi = floor(g);
    for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++){
      vec2 c = gi + vec2(i, j);
      float r0 = hash12(c + 17.3);
      if (r0 > 0.33) continue;
      vec2 ctr = (c + 0.15 + 0.7 * hash22(c)) * 9.;
      float rad = 0.6 + 1.5 * r0 * 3.;
      float dd = length(h - ctr);
      float ring = smoothstep(rad, rad - 0.55, dd) * smoothstep(rad - 1.3, rad - 0.6, dd);
      float hi = smoothstep(rad * 0.5, 0., length(h - ctr + vec2(rad * 0.35)));
      col = mix(col, vec3(0.9, 0.97, 1.), ring * 0.5 + hi * 0.6);
    }
  }

  // the label, etched into the top face as frosted strokes with a lit and a shaded wall
  float t0 = texture(uText, h / uSize).r;
  float tS = texture(uText, (h + vec2(0.7, 0.9)) / uSize).r;
  float tL = texture(uText, (h - vec2(0.7, 0.9)) / uSize).r;
  col = mix(col, vec3(0.94, 0.98, 1.0), t0 * (0.86 + 0.14 * hash12(floor(h * 2.))) * 0.9);
  col = mix(col, vec3(0.03, 0.10, 0.20), clamp(tS - t0, 0., 1.) * 0.55);
  col += vec3(0.6, 0.8, 1.) * clamp(tL - t0, 0., 1.) * 0.15;

  // frosted rim and a light dusting of rime
  float fr = fbm(h * 0.35 + vInfo.z * 3.);
  if (rim) col = mix(col, vec3(0.93, 0.97, 1.0), smoothstep(5.5, 0.4, d + (fr - 0.5) * 5.) * (0.55 + 0.45 * fbm(h * 1.3)));
  float rime = smoothstep(0.55, 0.85, fbm(h * 0.09 + 2.)) * 0.22 + step(0.93, hash12(floor(h * 1.6))) * 0.25 * fbm(h * 0.05);
  col = mix(col, vec3(0.95, 0.98, 1.), rime);

  // the cracks between chunks
  if (!rim){
    float vis = step(vInfo.x, uCrackR);
    col = mix(col, vec3(0.97, 0.99, 1.0), smoothstep(1.4, 0.2, d) * vis * 0.85);
    col = mix(col, col * 0.7, smoothstep(3.5, 1.2, d) * (1. - smoothstep(1.2, 0.6, d)) * vis * 0.5);
  }

  // stress crazing: under a held finger on the whole slab (radial splits too), or through a chunk
  // being crushed; crushed ice goes opaque white and grainy
  {
    float inside = crush, spoke = 0.;
    if (uCraze > 0.){
      vec2 q = h - uImpact;
      float r = length(q), ang = atan(q.y, q.x);
      float reach = uCraze * length(uSize) * (0.7 + 0.5 * vnoise(vec2(ang * 2.2, 3.)));
      inside = max(inside, smoothstep(reach, reach - 8., r));
      float sa = ang * 14. / 6.2831 + (fbm(q * 0.06) - 0.5) * 1.2;
      float spokePx = abs(fract(sa + 0.5) - 0.5) * 6.2831 / 14. * r;
      float spokeLen = reach * (0.5 + 0.6 * hash12(vec2(floor(sa + 0.5), 4.)));
      spoke = smoothstep(1.0, 0.25, spokePx) * smoothstep(spokeLen, spokeLen * 0.7, r) * step(2., r);
    }
    if (inside > 0. || spoke > 0.){
      float cell = 4.5;
      vec2 g = h / cell, gi = floor(g);
      float d1 = 9., d2 = 9.;
      for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++){
        vec2 c = gi + vec2(i, j);
        float dd = length(g - c - hash22(c));
        if (dd < d1){ d2 = d1; d1 = dd; } else if (dd < d2) d2 = dd;
      }
      float wall = smoothstep(0.7, 0.1, (d2 - d1) * cell);
      col = mix(col, vec3(0.97, 0.99, 1.0), clamp(wall * 0.6 * inside + spoke * 0.95, 0., 1.));
      col = mix(col, vec3(0.88, 0.94, 1.0) * (0.9 + 0.12 * hash12(floor(h * 1.5))), crush * 0.75);
    }
  }

  // cursor-lit glitter in the rime
  {
    vec2 gc = floor(h * 1.2);
    float gl = 0.;
    if (hash12(gc + 5.5 + vInfo.z) > 0.9){
      vec3 nr = normalize(vec3((hash22(gc) - 0.5) * 1.5, 1.));
      gl = glintOf(nr, vStage, uLamp, 500.) * smoothstep(0.5, 0.05, length(fract(h * 1.2) - 0.5));
    }
    col += vec3(0.95, 0.99, 1.) * gl * 2.2;
  }

  // light: Fresnel rim glow, a crisp specular from a fixed sun (a lamp straight overhead would blow the
  // flat top face out), a broad glint across the top face
  vec3 Ls = normalize(vec3(-0.45, -0.6, 0.66));
  float rs = max(dot(reflect(-Ls, n), vec3(0., 0., 1.)), 0.);
  float spec = pow(rs, 90.) * 1.6 + pow(rs, 14.) * 0.18;
  float fres = pow(1. - n.z, 2.);
  float glint = smoothstep(0.1, 0., abs(fract((h.x * 0.6 + h.y * 1.3) / (uSize.x * 0.9) + 0.18) - 0.5) - 0.05) * 0.08;
  col += vec3(0.9, 0.97, 1.0) * (spec + glint * (1. - uPress)) + vec3(0.5, 0.75, 1.) * fres * 0.5;
  // a hairline of darker ice where a chunk meets the page, so it reads on light backgrounds too
  col *= 1. - smoothstep(0.9, 0., d) * 0.25 * float(rim);
  col *= 1. - uPress * 0.06;
  float a = smoothstep(0., 0.7, d + 0.35) * vX.y;

  // regrowth: frost crystals race out from a few nuclei; the fresh ice behind the front is still rimed
  if (uGrowOn > 0.5){
    float g = 1e9;
    for (int i = 0; i < 3; i++) g = min(g, length(h - uSeeds[i]));
    g += (fbm(h * 0.07) - 0.5) * 22.;
    float band = uGrow * uReach - g;
    float fresh = smoothstep(16., 0., band);
    float den = 0.;
    for (int k = 0; k < 3; k++){
      float an = float(k) * 1.0472;
      den = max(den, pow(0.5 + 0.5 * sin(dot(h, vec2(cos(an), sin(an))) * 1.7 + fbm(h * 0.25) * 6.), 16.));
    }
    col = mix(col, vec3(0.93, 0.98, 1.), clamp(fresh * (0.3 + den * 0.9), 0., 1.));
    col += vec3(0.8, 0.95, 1.) * smoothstep(3., 0., abs(band)) * 0.5;
    a *= smoothstep(-1., 0.8, band);
  }
  o = vec4(col * a, a);
}`;

const VS_FULL = `#version 300 es
void main(){ vec2 p = vec2(float((gl_VertexID<<1)&2), float(gl_VertexID&2)); gl_Position = vec4(p*2.-1., 0., 1.); }`;

// Settled snow and crushed-ice powder, from the powder height grid.
const FS_SNOW = COMMON + `
uniform sampler2D uSnow;
uniform vec2 uGridPx;      // grid size in CSS px
uniform float uDpr, uStageH, uTime;
uniform vec2 uLamp;
out vec4 o;
float S(vec2 css){ return texture(uSnow, css / uGridPx).r * ${SNOW_PX.toFixed(1)}; }
void main(){
  vec2 css = vec2(gl_FragCoord.x, uStageH * uDpr - gl_FragCoord.y) / uDpr;
  float h = S(css);
  if (h < 0.01){ o = vec4(0.); return; }
  float gx = S(css + vec2(1.5, 0.)) - S(css - vec2(1.5, 0.));
  float gy = S(css + vec2(0., 1.5)) - S(css - vec2(0., 1.5));
  vec3 n = normalize(vec3(-gx, -gy, 1.6));
  float grain = hash12(floor(css * 2.));
  float cover = smoothstep(0.05, 0.55, h + (fbm(css * 0.6) - 0.5) * 0.5 + (grain - 0.5) * 0.25);
  vec3 L = normalize(vec3(-0.45, -0.6, 0.66));
  float dif = max(dot(n, L), 0.);
  vec3 col = vec3(0.82, 0.9, 0.98) * (0.55 + 0.5 * dif) + vec3(0.10, 0.16, 0.26) * (1. - dif) * 0.35;
  col = min(col * (0.93 + 0.1 * grain), vec3(1.));
  // crushed ice is full of tiny facets: a few of them catch the lamp at any moment
  float gl = 0.;
  vec2 gc = floor(css * 1.5);
  if (hash12(gc + 7.7) > 0.7){
    vec3 nr = normalize(vec3((hash22(gc) - 0.5) * 1.7, 1.));
    gl = glintOf(nr, css, uLamp, 700.) * smoothstep(0.55, 0.05, length(fract(css * 1.5) - 0.5)) * smoothstep(0.15, 0.6, h);
    gl *= 0.8 + 0.2 * sin(uTime * 11. + hash12(gc) * 40.);
  }
  o = vec4(col * cover + vec3(0.95, 0.99, 1.) * gl * 2.5, cover);
}`;

// Ice chips frozen onto the page, one point sprite each: an irregular convex fleck, clear or frosted,
// with two top facets that glint independently and a star flare when one lines up with the lamp.
const VS_CHIP = `#version 300 es
in vec4 aP;        // x, y (stage CSS px), size, alpha
in vec4 aQ;        // angle, seed, hop height, frost
uniform vec2 uStage; uniform float uDpr;
out vec4 vP; out vec4 vQ;
void main(){
  vec2 p = aP.xy - vec2(0., aQ.z * 0.9);
  gl_Position = vec4(p.x / uStage.x * 2. - 1., 1. - p.y / uStage.y * 2., 0., 1.);
  gl_PointSize = (aP.z + 12.) * uDpr;
  vP = vec4(p, aP.zw); vQ = aQ;
}`;
const FS_CHIP = COMMON + `
in vec4 vP; in vec4 vQ;
uniform sampler2D uBgTex; uniform float uHasBg; uniform vec3 uBgCol; uniform vec4 uBg;
uniform vec2 uLamp; uniform float uTime;
out vec4 o;
vec3 bgAt(vec2 sp){
  if (uHasBg < 0.5) return uBgCol;
  return texture(uBgTex, (sp + uBg.xy) / uBg.zw).rgb;
}
float star(vec2 p, float s){
  float r = length(p);
  float a = exp(-abs(p.y) * 2.2) * exp(-abs(p.x) / (0.9 * s)) + exp(-abs(p.x) * 2.2) * exp(-abs(p.y) / (0.9 * s));
  vec2 d = vec2(p.x + p.y, p.x - p.y) * 0.7071;
  a += 0.4 * (exp(-abs(d.y) * 2.6) * exp(-abs(d.x) / (0.5 * s)) + exp(-abs(d.x) * 2.6) * exp(-abs(d.y) / (0.5 * s)));
  return a + exp(-r * 1.1) * 1.2;
}
void main(){
  float S = vP.z + 12.;
  vec2 pc = (gl_PointCoord - 0.5) * S;
  vec2 sp = vP.xy + pc;
  float seed = vQ.y;
  float c = cos(vQ.x), s = sin(vQ.x);
  vec2 q = vec2(c * pc.x + s * pc.y, -s * pc.x + c * pc.y);
  float R = vP.z * 0.5;
  float dm = 1e9; vec2 nn = vec2(0.);
  for (int k = 0; k < 6; k++){
    float a = float(k) * 1.0472 + (hash12(vec2(seed * 97.1, float(k) * 1.3)) - 0.5) * 0.9;
    vec2 dir = vec2(cos(a), sin(a));
    float d = R * (0.55 + 0.45 * hash12(vec2(float(k) * 7.7, seed * 53.3))) - dot(q, dir);
    if (d < dm){ dm = d; nn = dir; }
  }
  vec2 nS = vec2(c * nn.x - s * nn.y, s * nn.x + c * nn.y);
  vec2 tA = (hash22(vec2(seed * 13.1, 2.)) - 0.5) * 1.2;
  vec2 tB = (hash22(vec2(seed * 29.7, 5.)) - 0.5) * 1.2;
  vec2 sd = normalize(hash22(vec2(seed * 3.3, 9.)) - 0.5 + 1e-3);
  float gA = glintOf(normalize(vec3(tA, 1.)), vP.xy, uLamp, 260.);
  float gB = glintOf(normalize(vec3(tB, 1.)), vP.xy, uLamp, 260.);
  float tw = 0.8 + 0.2 * sin(uTime * 9. + seed * 60.);
  float flare = clamp(max(gA, gB) * tw * vP.w, 0., 1.5) * star(pc, max(vP.z, 3.)) * smoothstep(S * 0.5, S * 0.3, length(pc));
  vec3 flareC = vec3(0.92, 0.98, 1.) * flare;
  if (dm < -1.8){
    if (flare < 0.004) discard;
    o = vec4(flareC, flare * 0.2);
    return;
  }
  float inside = smoothstep(-0.45, 0.35, dm);
  float shadow = smoothstep(-1.8, -0.2, dm) * (1. - inside) * 0.32;
  vec2 tilt = dot(q, sd) > (hash12(vec2(seed, 1.)) - 0.5) * R ? tA : tB;
  vec3 n = normalize(vec3(dm < 1.1 ? nS * 0.9 + tilt * 0.3 : tilt, 1.));
  vec3 bg = bgAt(sp - n.xy * 7.);
  vec3 clear = bg * vec3(0.7, 0.87, 1.0) * 0.95 + vec3(0.07, 0.17, 0.28);
  vec3 frosted = vec3(0.86, 0.93, 1.0) * (0.86 + 0.14 * hash12(floor(q * 2.) + seed * 40.));
  vec3 col = mix(clear, frosted, vQ.w);
  // a lit bevel on the lamp side, a dark wet contact line where the chip is frozen to the page
  vec2 toL = normalize(uLamp - vP.xy + 1e-3);
  col += vec3(0.75, 0.9, 1.) * smoothstep(1.3, 0.3, dm) * (0.2 + 0.45 * max(dot(nS, toL), 0.));
  col *= 1. - smoothstep(0.55, 0., dm) * 0.4;
  col += vec3(1.) * (glintOf(n, sp, uLamp, 260.) * 2.4 * tw + glintOf(n, sp, uLamp, 24.) * 0.12);
  float a = inside * vP.w;
  o = vec4(col * a + vec3(0.02, 0.07, 0.16) * shadow * vP.w + flareC, a + shadow * vP.w + flare * 0.2);
}`;

const VS_PART = `#version 300 es
in vec4 aP;        // screen x, y (stage CSS px), size, alpha
in vec2 aK;        // kind, twinkle
uniform vec2 uStage; uniform float uDpr;
out vec2 vK; out float vA;
void main(){
  gl_Position = vec4(aP.x / uStage.x * 2. - 1., 1. - aP.y / uStage.y * 2., 0., 1.);
  gl_PointSize = max(aP.z * uDpr, 1.);
  vK = aK; vA = aP.w;
}`;
const FS_PART = `#version 300 es
precision highp float;
in vec2 vK; in float vA;
out vec4 o;
void main(){
  vec2 c = gl_PointCoord - 0.5;
  float r = length(c);
  if (r > 0.5) discard;
  if (vK.x > 2.5){ float a = smoothstep(0.5, 0.1, r) * vA * 0.28; o = vec4(vec3(0.02, 0.07, 0.16) * a, a); return; }
  float a = smoothstep(0.5, vK.x > 0.5 ? 0.3 : 0.12, r) * vA;
  vec3 col = vK.x > 1.5 ? vec3(1.) : vec3(0.93, 0.97, 1.0) * (0.9 + 0.1 * vK.y);
  if (vK.x > 1.5){ float cr = max(smoothstep(0.08, 0., abs(c.x)), smoothstep(0.08, 0., abs(c.y))) * smoothstep(0.5, 0., r); a = max(a * 0.6, cr * vA); }
  o = vec4(col * a, a);
}`;

// ---------------------------------------------------------------- audio (shared)
let actx = null;
const iceAudio = { muted: false };
function audio(){
  if (iceAudio.muted) return null;
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
  g.gain.exponentialRampToValueAtTime(gain, t + Math.min(0.004, dur * 0.2));
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  src.connect(f).connect(g).connect(a.destination);
  src.start(t, Math.random() * 1.5); src.stop(t + dur + 0.02);
}
function ping(freq, gain, dur, delay = 0){
  const a = audio(); if (!a) return;
  const t = a.currentTime + delay;
  const o = a.createOscillator(), g = a.createGain();
  o.type = 'sine'; o.frequency.setValueAtTime(freq, t); o.frequency.exponentialRampToValueAtTime(freq * 0.985, t + dur);
  g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(gain, t + 0.003); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  o.connect(g).connect(a.destination); o.start(t); o.stop(t + dur + 0.02);
}

// The frozen-lake "pew". Flexural waves in a plate are dispersive (speed ~ sqrt(frequency)), so a hit
// arrives as a descending chirp: over a path, arrival time ~ 1/sqrt(f), i.e. f(t) = f0 / (1 + t/tau)^2.
// Each echo off the far shore has travelled further, so it comes later and more stretched out.
// Rendered once per sample rate into buffers; a comb-ish feedback delay adds the metallic, spacey ring.
let chirpSet = null;
function chirps(a){
  if (chirpSet && chirpSet.sr === a.sampleRate) return chirpSet;
  const sr = a.sampleRate;
  const make = (dur, fHi, arrivals, voices) => {
    const n = Math.round(sr * dur), buf = a.createBuffer(1, n, sr), d = buf.getChannelData(0);
    for (const [t0, tau, amp] of arrivals) for (const det of voices){
      const f0 = fHi * det, start = Math.round(t0 * sr);
      let ph = Math.random() * TAU;
      for (let i = start; i < n; i++){
        const t = (i - start) / sr, f = f0 / ((1 + t / tau) * (1 + t / tau));
        if (f < 160) break;
        ph += TAU * f / sr;
        const env = amp * Math.min(1, t / 0.0015) * Math.pow(f / f0, 0.22) * Math.exp(-t * 2.6) * Math.min(1, (f - 160) / 120);
        d[i] += Math.sin(ph) * env / voices.length;
      }
    }
    let peak = 0;
    for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(d[i]));
    if (peak > 0) for (let i = 0; i < n; i++) d[i] *= 0.9 / peak;
    return buf;
  };
  chirpSet = {
    sr,
    big: make(1.3, 7500, [[0, 0.03, 1], [0.1, 0.042, 0.55], [0.23, 0.058, 0.34], [0.4, 0.075, 0.2], [0.62, 0.095, 0.1]], [1, 1.023, 0.982]),
    small: make(0.45, 6000, [[0, 0.016, 1], [0.07, 0.024, 0.4]], [1, 1.017]),
  };
  return chirpSet;
}
let bus = null;
function laserBus(a){
  if (bus && bus.ctx === a) return bus.inp;
  const inp = a.createGain(), dl = a.createDelay(0.5), fb = a.createGain(), hp = a.createBiquadFilter(), wet = a.createGain();
  dl.delayTime.value = 0.041; fb.gain.value = 0.45; hp.type = 'highpass'; hp.frequency.value = 650; wet.gain.value = 0.32;
  inp.connect(a.destination);
  inp.connect(dl); dl.connect(hp); hp.connect(fb); fb.connect(dl); hp.connect(wet); wet.connect(a.destination);
  bus = { ctx: a, inp };
  return inp;
}
function pew(kind, gain, rate, delay = 0){
  const a = audio(); if (!a) return;
  const src = a.createBufferSource(); src.buffer = chirps(a)[kind]; src.playbackRate.value = rate;
  const g = a.createGain(); g.gain.value = gain;
  src.connect(g).connect(laserBus(a)); src.start(a.currentTime + delay);
}
const rand = (a, b) => a + Math.random() * (b - a);
let lastTink = 0, lastLoose = 0, lastGrit = 0;
const sounds = {
  press(){ burst('bandpass', 1300, 1.4, 0.05, 0.09); burst('highpass', 5200, 0.8, 0.02, 0.03, 0.02); },
  // the first break: the slab cracks through, and the whole sheet rings with the frozen-lake "pew"
  smash(){
    burst('lowpass', 700, 0.9, 0.22, 0.22); burst('bandpass', 2600, 0.7, 0.16, 0.3, 0.01);
    for (let i = 0; i < 18; i++) burst('highpass', rand(3000, 7000), 1, rand(0.02, 0.07), rand(0.01, 0.03), rand(0, 0.25));
    pew('big', 0.34, 1, 0.02); pew('big', 0.2, 0.93, 0.055); pew('big', 0.13, 1.08, 0.12);
  },
  // a chunk crushed into the page: a thump and a fast packed crackle, like a boot in crusted snow
  crush(size = 1){
    burst('lowpass', 240, 0.8, 0.12 + 0.14 * size, 0.14);
    for (let i = 0; i < 14 + 26 * size; i++) burst('bandpass', rand(900, 4200), rand(1, 2.5), rand(0.03, 0.09), rand(0.01, 0.035), 0.2 * Math.pow(Math.random(), 1.6));
    burst('highpass', 3800, 0.7, 0.05, 0.18, 0.02);
    for (let i = 0; i < 4; i++) ping(rand(3000, 7000), 0.012, rand(0.04, 0.12), rand(0.02, 0.2));
  },
  tink(v){
    const a = actx; if (!a || a.currentTime - lastTink < 0.04) return;
    lastTink = a.currentTime;
    ping(rand(3000, 6500), Math.min(0.03, 0.006 + v * 0.0002), rand(0.05, 0.14));
  },
  loose(v){
    const a = actx; if (!a || a.currentTime - lastLoose < 0.05) return;
    lastLoose = a.currentTime;
    burst('bandpass', rand(2200, 4800), 1.6, Math.min(0.08, 0.025 + v * 0.0001), rand(0.012, 0.03));
    ping(rand(3500, 7500), Math.min(0.025, 0.008 + v * 0.00005), rand(0.03, 0.09));
  },
  grit(lvl){
    const a = actx; if (!a || a.currentTime - lastGrit < 0.025) return;
    lastGrit = a.currentTime;
    burst('bandpass', rand(1800, 5200), rand(1.5, 3), 0.015 + 0.05 * lvl * Math.random(), rand(0.008, 0.02));
  },
  tap(){ burst('bandpass', rand(1600, 3000), 1.2, 0.07, 0.05); for (let i = 0; i < 3; i++) ping(rand(3500, 7000), 0.015, rand(0.04, 0.1), rand(0, 0.06)); },
  refreeze(){
    for (let i = 0; i < 26; i++) burst('highpass', rand(5000, 9000), 1.2, rand(0.008, 0.025), 0.012, Math.pow(Math.random(), 0.7) * 0.9);
    for (let i = 0; i < 6; i++) ping(1400 * Math.pow(1.19, i) * rand(0.99, 1.01), 0.012, 0.25, 0.12 + i * 0.12);
    ping(1760, 0.035, 0.9, 1.0); ping(2637, 0.016, 0.7, 1.02);
  },
};
let scrape = null;
function scrapeLevel(lvl){
  const a = lvl > 0 ? audio() : actx; if (!a) return;
  if (!scrape){
    if (lvl <= 0) return;
    const src = a.createBufferSource(); src.buffer = noise(a); src.loop = true;
    const f = a.createBiquadFilter(); f.type = 'bandpass'; f.frequency.value = 3000; f.Q.value = 1.1;
    const g = a.createGain(); g.gain.value = 0;
    src.connect(f).connect(g).connect(a.destination); src.start();
    scrape = { src, f, g };
  }
  // jittered every frame so it grates instead of hissing
  const v = iceAudio.muted ? 0 : Math.min(lvl, 1) * (lvl > 0 ? rand(0.45, 1.25) : 0);
  scrape.g.gain.setTargetAtTime(0.07 * v, a.currentTime, 0.012);
  scrape.f.frequency.setTargetAtTime(1800 + 3000 * v * rand(0.7, 1.1), a.currentTime, 0.02);
}

// ---------------------------------------------------------------- geometry
function parseColor(css){
  const m = /rgba?\(([^)]+)\)/.exec(css || '');
  if (!m) return null;
  const p = m[1].split(/[\s,/]+/).filter(Boolean).map(parseFloat);
  return [p[0] / 255, p[1] / 255, p[2] / 255, p.length > 3 ? p[3] : 1];
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
function roundRect(w, h, r){
  const pts = [], seg = 6;
  const cs = [[w - r, r, -Math.PI / 2], [w - r, h - r, 0], [r, h - r, Math.PI / 2], [r, r, Math.PI]];
  for (const [cx, cy, a0] of cs) for (let i = 0; i <= seg; i++){
    const a = a0 + i / seg * Math.PI / 2;
    pts.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]);
  }
  return pts;
}
function centroid(P){
  let cx = 0, cy = 0, A = 0;
  for (let i = 0; i < P.length; i++){ const a = P[i], b = P[(i + 1) % P.length], c = a[0] * b[1] - b[0] * a[1]; A += c; cx += (a[0] + b[0]) * c; cy += (a[1] + b[1]) * c; }
  return Math.abs(A) < 1e-9 ? P[0].slice() : [cx / (3 * A), cy / (3 * A)];
}
function area(P){ let s = 0; for (let i = 0; i < P.length; i++){ const a = P[i], b = P[(i + 1) % P.length]; s += a[0] * b[1] - b[0] * a[1]; } return s / 2; }
// Sutherland-Hodgman against the half-plane dot(p - o, n) <= 0, tracking which edges are the original
// outline (flag 0) and which are new cuts (flag 1). F[i] is the flag of the edge P[i] -> P[i+1].
function clip(P, F, o, n){
  const outP = [], outF = [];
  const side = p => (p[0] - o[0]) * n[0] + (p[1] - o[1]) * n[1];
  for (let i = 0; i < P.length; i++){
    const a = P[i], b = P[(i + 1) % P.length], f = F[i];
    const sa = side(a), sb = side(b), ina = sa <= 0, inb = sb <= 0;
    const I = () => { const t = sa / (sa - sb); return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]; };
    if (ina && inb){ outP.push(a); outF.push(f); }
    else if (ina && !inb){ outP.push(a); outF.push(f); outP.push(I()); outF.push(1); }
    else if (!ina && inb){ outP.push(I()); outF.push(f); }
  }
  return [outP, outF];
}
function voronoi(P, F, seeds){
  const cells = [];
  for (let i = 0; i < seeds.length; i++){
    let cp = P, cf = F;
    for (let j = 0; j < seeds.length && cp.length >= 3; j++){
      if (i === j) continue;
      const s = seeds[i], t = seeds[j];
      [cp, cf] = clip(cp, cf, [(s[0] + t[0]) / 2, (s[1] + t[1]) / 2], [t[0] - s[0], t[1] - s[1]]);
    }
    if (cp.length >= 3 && Math.abs(area(cp)) > 3) cells.push([cp, cf]);
  }
  return cells;
}
function pointInPoly(P, x, y){
  let inside = false;
  for (let i = 0, j = P.length - 1; i < P.length; j = i++){
    const [xi, yi] = P[i], [xj, yj] = P[j];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
function closestOnPoly(P, x, y){
  let best = Infinity, bx = 0, by = 0;
  for (let i = 0; i < P.length; i++){
    const a = P[i], b = P[(i + 1) % P.length];
    const ex = b[0] - a[0], ey = b[1] - a[1];
    const t = Math.max(0, Math.min(1, ((x - a[0]) * ex + (y - a[1]) * ey) / (ex * ex + ey * ey || 1)));
    const px = a[0] + ex * t, py = a[1] + ey * t, d = (x - px) ** 2 + (y - py) ** 2;
    if (d < best){ best = d; bx = px; by = py; }
  }
  return [bx, by, Math.sqrt(best)];
}
// minimum translation that separates convex polygons A and B (pushes B away from A), or null
function sat(A, B){
  let depth = Infinity, ax = 0, ay = 0;
  for (const [P, Q] of [[A, B], [B, A]]){
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
const ease = v => v * v * (3 - 2 * v);

// ---------------------------------------------------------------- component
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');

function attachIce(button, opts = {}){
  const canvas = document.createElement('canvas');
  canvas.setAttribute('aria-hidden', 'true');
  Object.assign(canvas.style, { position: 'absolute', pointerEvents: 'none', zIndex: opts.zIndex ?? 10, left: '0px', top: '0px' });
  const gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: true, antialias: true, depth: false, stencil: false });
  if (!gl) return { detach(){}, supported: false };
  const sound = opts.sound !== false;
  const snd = (name, ...a) => { if (sound) sounds[name](...a); };
  const REFREEZE = opts.refreezeAfter ?? 3.2;
  const SNOW_IDLE = (opts.snowIdle ?? 8) * 1000;

  const compile = (type, src) => {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS) && !gl.isContextLost()) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  const program = (vs, fs) => {
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, vs)); gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
    // the shadow and ice programs share one VAO, so their attributes need fixed locations
    ['aPos', 'aHome', 'aEdge', 'aInfo', 'aX'].forEach((name, i) => gl.bindAttribLocation(p, i, name));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS) && !gl.isContextLost()) throw new Error(gl.getProgramInfoLog(p));
    const u = {};
    for (let i = 0, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS) || 0; i < n; i++){
      const info = gl.getActiveUniform(p, i);
      u[info.name.replace(/\[0\]$/, '')] = gl.getUniformLocation(p, info.name);
    }
    return { p, u };
  };
  const P = { shadow: program(VS_SHARD, FS_SHADOW), ice: program(VS_SHARD, FS_ICE), snow: program(VS_FULL, FS_SNOW), part: program(VS_PART, FS_PART), chip: program(VS_CHIP, FS_CHIP) };
  const fullVao = gl.createVertexArray();

  // chunk vertex stream: pos2 home2 edge4 info4 x4 (stride 16 floats); both chunk programs share a layout
  const shardBuf = gl.createBuffer(), shardVao = gl.createVertexArray();
  gl.bindVertexArray(shardVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, shardBuf);
  [2, 2, 4, 4, 4].reduce((off, size, l) => { gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, size, gl.FLOAT, false, 64, off * 4); return off + size; }, 0);
  const pointVao = (prog, buf, names, stride) => {
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    let off = 0;
    for (const [name, size] of names){ const l = gl.getAttribLocation(prog.p, name); gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, size, gl.FLOAT, false, stride, off); off += size * 4; }
    return vao;
  };
  const partBuf = gl.createBuffer(), partVao = pointVao(P.part, partBuf, [['aP', 4], ['aK', 2]], 24);
  const chipBuf = gl.createBuffer(), chipVao = pointVao(P.chip, chipBuf, [['aP', 4], ['aQ', 4]], 32);
  gl.bindVertexArray(null);
  const mkTex = (filter) => {
    const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, filter], [gl.TEXTURE_MAG_FILTER, filter], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
    return t;
  };
  const textTex = mkTex(gl.LINEAR), bgTex = mkTex(gl.LINEAR), snowTex = mkTex(gl.LINEAR);
  const bindTex = (unit, t) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, t); };

  // ---- page
  const saved = button.getAttribute('style');
  const cs0 = getComputedStyle(button);
  const label = button.textContent.trim();
  const font = `${cs0.fontStyle} ${cs0.fontWeight} ${cs0.fontSize} ${cs0.fontFamily}`;
  Object.assign(button.style, { background: 'transparent', borderColor: 'transparent', color: 'transparent', boxShadow: 'none', textShadow: 'none', position: cs0.position === 'static' ? 'relative' : cs0.position });
  document.body.appendChild(canvas);
  const bgCol = opts.background ? (parseColor(opts.background) || [1, 1, 1]).slice(0, 3) : pageColor(button.parentElement);
  const surface = opts.surface || null;
  if (surface){
    gl.bindTexture(gl.TEXTURE_2D, bgTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, surface.image);
  }

  // ---- geometry (CSS px); button-local "home" coords, stage = home + M
  let bw = 0, bh = 0, M = 0, stageW = 0, stageH = 0, dpr = 1, radius = 0, GW = 0, GH = 0;
  let slabPoly = [], planes = new Float32Array(16);
  let snow = new Float32Array(0), slabMask = new Uint8Array(0), snowBytes = new Uint8Array(0), snowDirty = true;
  let bgOff = [0, 0, 1, 1];
  let dirty = true;

  function buildText(){
    const c = document.createElement('canvas');
    const s = dpr * 2;
    c.width = Math.max(1, Math.round(bw * s)); c.height = Math.max(1, Math.round(bh * s));
    const x = c.getContext('2d');
    x.fillStyle = '#000'; x.fillRect(0, 0, c.width, c.height);
    x.font = font.replace(/(\d+(\.\d+)?)px/, (_, v) => `${parseFloat(v) * s}px`);
    x.textAlign = 'center'; x.textBaseline = 'middle';
    x.filter = `blur(${0.35 * s}px)`;
    x.fillStyle = '#fff';
    x.fillText(label, c.width / 2, c.height / 2 + parseFloat(cs0.fontSize) * s * 0.04);
    gl.bindTexture(gl.TEXTURE_2D, textTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, c);
  }

  function place(){
    const r = button.getBoundingClientRect();
    Object.assign(canvas.style, { left: `${r.left + scrollX - M}px`, top: `${r.top + scrollY - M}px` });
    if (surface){
      const s = surface.rect();
      bgOff = [r.left - M - s.left, r.top - M - s.top, s.width, s.height];
    }
    dirty = true;
  }

  // signed distance to the button outline, home coords
  const slabSD = (x, y) => {
    const qx = Math.abs(x - bw / 2) - (bw / 2 - radius), qy = Math.abs(y - bh / 2) - (bh / 2 - radius);
    return Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0) - radius;
  };

  function resize(){
    const r = button.getBoundingClientRect();
    bw = r.width; bh = r.height;
    M = Math.round(Math.max(60, bh * 1.2));
    stageW = bw + 2 * M; stageH = bh + 2 * M;
    dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(stageW * dpr); canvas.height = Math.round(stageH * dpr);
    Object.assign(canvas.style, { width: `${stageW}px`, height: `${stageH}px` });
    radius = Math.min(parseFloat(getComputedStyle(button).borderTopLeftRadius) || 0, bw / 2, bh / 2);
    slabPoly = roundRect(bw, bh, Math.max(radius, 2));
    for (let i = 0; i < 4; i++){
      const a = rand(-0.5, 0.5) + (i % 2 ? Math.PI / 2 : 0);
      planes.set([rand(0.1, 0.9) * bw, rand(0.15, 0.85) * bh, Math.cos(a), Math.sin(a)], i * 4);
    }
    GW = Math.ceil(stageW / CS); GH = Math.ceil(stageH / CS);
    snow = new Float32Array(GW * GH); slabMask = new Uint8Array(GW * GH); snowBytes = new Uint8Array(GW * GH);
    for (let j = 0; j < GH; j++) for (let i = 0; i < GW; i++) slabMask[j * GW + i] = slabSD((i + 0.5) * CS - M, (j + 0.5) * CS - M) < -1.5 ? 1 : 0;
    // it has been snowing for a while already: a light dusting, heavier toward the top edge
    for (let n = Math.round(bw * bh / 22); n > 0; n--){
      const x = rand(4, bw - 4), y = bh * Math.pow(Math.random(), 2.6);
      deposit(x + M, y + M, rand(0.03, 0.12) * (1.4 - y / bh), rand(1.5, 3));
    }
    snowDirty = true;
    buildText();
    toRest();
    place();
    wake(0.2);
  }

  // ---- state
  let state = 'rest';                 // rest | smashed | regrowing
  let phaseT = 0, idleT = 0, press = 0, pressTarget = 0, holdT = 0;
  let craze = 0, crackR = 1e9, grow = 1, reach = 1;
  let impact = [0, 0];
  const seeds = new Float32Array(6);
  let shards = [], chips = [], growSlab = null;
  let lamp = null;                    // stage px of the cursor, the light the glints answer to

  function mkShard(poly, flags, click){
    const hc = centroid(poly);
    const local = poly.map(p => [p[0] - hc[0], p[1] - hc[1]]);
    const A = Math.abs(area(poly));
    const mids = flags.map((f, i) => {
      const a = poly[i], b = poly[(i + 1) % poly.length];
      return click ? Math.hypot((a[0] + b[0]) / 2 - click[0], (a[1] + b[1]) / 2 - click[1]) : 0;
    });
    return { hc, local, flags, mids, area: A, rad: Math.sqrt(A / Math.PI), off: [0, 0], ang: 0, scale: 1, z: 0, vz: 0, vx: 0, vy: 0, va: 0,
      delay: 0, seed: Math.random() * 0.9 + 0.05, world: null, gone: false, crushT: -1, crush: 0, fade: 1, strain: 0 };
  }
  function worldOf(s){
    const c = Math.cos(s.ang), sn = Math.sin(s.ang), k = s.scale * (1 + s.z * 0.004);
    const ox = s.hc[0] + s.off[0] + M, oy = s.hc[1] + s.off[1] + M;
    return s.local.map(([x, y]) => [ox + (c * x - sn * y) * k, oy + (sn * x + c * y) * k]);
  }
  // stage point -> the chunk's position in the unbroken slab (home coords), so crushed bits know where
  // to gather back to
  function homeOf(s, x, y){
    const c = Math.cos(-s.ang), sn = Math.sin(-s.ang), k = s.scale * (1 + s.z * 0.004);
    const lx = x - (s.hc[0] + s.off[0] + M), ly = y - (s.hc[1] + s.off[1] + M);
    return [(c * lx - sn * ly) / k + s.hc[0], (sn * lx + c * ly) / k + s.hc[1]];
  }
  const live = () => shards.filter(s => !s.gone);

  function toRest(){
    state = 'rest'; craze = 0; grow = 1; crackR = 1e9; chips = []; growSlab = null;
    impact = [bw / 2, bh / 2];
    shards = [mkShard(slabPoly, slabPoly.map(() => 0), null)];
    shards[0].seed = 0;
    shards[0].world = worldOf(shards[0]);
  }

  // Step one: the slab cracks through in place into big chunks that hop and settle where they were.
  function smashAt(cx, cy){
    impact = [cx, cy];
    const seedsV = [[cx + rand(-1, 1), cy + rand(-1, 1)]];
    const rings = [[9, 4], [22, 5], [40, 6], [66, 7], [100, 8], [150, 9], [220, 10]];
    const far = Math.hypot(Math.max(cx, bw - cx), Math.max(cy, bh - cy)) + 10;
    for (const [r, n] of rings){
      if (r > far) break;
      const a0 = rand(0, TAU);
      for (let i = 0; i < n; i++){
        const a = a0 + i / n * TAU + rand(-0.25, 0.25), rr = r * rand(0.8, 1.2);
        const x = cx + Math.cos(a) * rr, y = cy + Math.sin(a) * rr;
        if (x > -2 && x < bw + 2 && y > -2 && y < bh + 2) seedsV.push([x, y]);
      }
    }
    shards = voronoi(slabPoly, slabPoly.map(() => 0), seedsV).map(([p, f]) => {
      const s = mkShard(p, f, [cx, cy]);
      const dx = s.hc[0] - cx, dy = s.hc[1] - cy, d = Math.hypot(dx, dy) || 1;
      const fall = Math.exp(-d / (Math.max(bw, bh) * 0.45));
      s.delay = d / 900;
      s.vx = dx / d * rand(6, 22) * fall; s.vy = dy / d * rand(6, 22) * fall;
      s.vz = rand(40, 110) * (0.25 + fall); s.va = rand(-0.8, 0.8) * fall;
      s.world = worldOf(s);
      return s;
    });
    crackR = 0; state = 'smashed'; phaseT = 0; idleT = 0;
    snd('smash');
    for (let i = 0; i < 40; i++){
      const a = rand(0, TAU), sp = rand(15, 90);
      parts.push({ x: cx + M, y: cy + M, z: rand(0, 4), vx: Math.cos(a) * sp, vy: Math.sin(a) * sp, vz: rand(40, 140), size: rand(1.2, 3), kind: 1, age: 0, life: rand(0.6, 1.3), dep: 0.05 });
    }
    for (let i = 0; i < 8; i++) parts.push({ x: cx + M + rand(-8, 8), y: cy + M + rand(-8, 8), z: rand(2, 10), vx: rand(-30, 30), vy: rand(-30, 30), vz: rand(20, 80), size: rand(4, 7), kind: 2, age: 0, life: rand(0.25, 0.5), dep: 0 });
    // the snow that was sitting on the slab jumps
    for (let j = 0; j < GH; j++) for (let i = 0; i < GW; i++){
      const k = j * GW + i;
      if (snow[k] > 0.15 && slabMask[k] && Math.random() < 0.25){
        const x = (i + 0.5) * CS, y = (j + 0.5) * CS;
        const dx = x - M - cx, dy = y - M - cy, d = Math.hypot(dx, dy) || 1;
        parts.push({ x, y, z: 1, vx: dx / d * rand(5, 30), vy: dy / d * rand(5, 30), vz: rand(30, 90), size: rand(1.2, 2.4), kind: 1, age: 0, life: 2, dep: snow[k] * 0.9 });
      }
    }
    for (let k = 0; k < snow.length; k++) if (slabMask[k]) snow[k] *= 0.1;
    snowDirty = true;
    wake(3);
  }

  // Step two: one chunk is crushed down into the page.
  function crushShard(s, px, py){
    if (s.crushT >= 0 || s.gone) return;
    s.crushT = 0; s.crushAt = [px, py]; s.vx = s.vy = s.va = 0; s.vz = 0; s.z = 0;
    idleT = 0;
    snd('crush', Math.min(1, s.area / 1800));
    wake(1);
  }
  // A pressed-flat layer of chips and powder over the chunk's footprint, squeezed outward from the fist.
  function spawnChipsIn(s){
    const W = s.world, [cx, cy] = s.crushAt;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const [x, y] of W){ x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
    const A = s.area * s.scale * s.scale, far = Math.max(8, Math.sqrt(A));
    const n = Math.round(Math.min(280, Math.max(6, A / 26)));
    let made = 0;
    for (let tries = 0; made < n && tries < n * 8; tries++){
      const x = rand(x0, x1), y = rand(y0, y1);
      if (!pointInPoly(W, x, y)) continue;
      const dx = x - cx, dy = y - cy, d = Math.hypot(dx, dy) || 1;
      const near = Math.exp(-d / (far * 0.5));
      const v = Math.sqrt(2 * MU * rand(0, 1) * (2 + 14 * near));
      const bondMax = rand(5, 16);
      const [hx, hy] = homeOf(s, x, y);
      chips.push({
        x, y, hx: hx + M, hy: hy + M, ang: rand(0, TAU), va: rand(-6, 6) * near,
        size: 1.6 + 6.5 * Math.pow(Math.random(), 2.4) * (1 - 0.4 * near), seed: Math.random(),
        frost: Math.random() < 0.35 ? rand(0.6, 1) : rand(0, 0.25),
        stuck: false, bond: bondMax, bondMax, strain: 0, rest: 0,
        vx: dx / d * v * rand(0.8, 1.2), vy: dy / d * v * rand(0.8, 1.2),
        z: 0, vz: Math.random() < 0.1 * near ? rand(25, 70) : 0, jx: 0, jy: 0, a: 1,
      });
      made++;
    }
    for (let k = Math.round(A / 5); k > 0; k--){
      const x = rand(x0, x1), y = rand(y0, y1);
      if (!pointInPoly(W, x, y)) continue;
      deposit(x, y, 0.12 + 0.3 * Math.exp(-Math.hypot(x - cx, y - cy) / (far * 0.5)) * Math.random(), rand(2, 3.5));
    }
    for (let i = 0; i < 6 + n / 6; i++){
      const a = rand(0, TAU), sp = rand(15, 60);
      parts.push({ x: cx, y: cy, z: rand(0, 3), vx: Math.cos(a) * sp, vy: Math.sin(a) * sp, vz: rand(30, 80), size: rand(1.2, 2.4), kind: 1, age: 0, life: rand(0.5, 1), dep: 0.05 });
    }
    parts.push({ x: cx, y: cy, z: 3, vx: 0, vy: 0, vz: 10, size: rand(5, 7), kind: 2, age: 0, life: 0.3, dep: 0 });
    snowDirty = true;
  }

  function loosen(c, vx, vy){
    c.stuck = false; c.vx = vx; c.vy = vy; c.va = rand(-9, 9); c.rest = 0; c.strain = 0; c.jx = c.jy = 0;
  }

  function tapAt(x, y){
    const R = PUSHER_R * 1.5;
    let hit = 0;
    for (const c of chips){
      const dx = c.x - x, dy = c.y - y, d = Math.hypot(dx, dy);
      if (d > R + c.size * 0.4) continue;
      const k = 1 - d / (R + c.size * 0.4), n = d || 1;
      loosen(c, dx / n * rand(60, 170) * k + rand(-15, 15), dy / n * rand(60, 170) * k + rand(-15, 15));
      if (Math.random() < 0.4) c.vz = rand(20, 60) * k;
      hit++;
    }
    for (let i = 0; i < 8; i++){
      const a = rand(0, TAU), sp = rand(10, 45);
      parts.push({ x, y, z: 1, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp, vz: rand(20, 60), size: rand(1, 2.2), kind: 1, age: 0, life: 1, dep: 0.04 });
    }
    if (hit) snd('tap');
  }

  // ---- powder grid
  function deposit(x, y, amount, r){
    const i0 = Math.max(0, Math.floor((x - r) / CS)), i1 = Math.min(GW - 1, Math.ceil((x + r) / CS));
    const j0 = Math.max(0, Math.floor((y - r) / CS)), j1 = Math.min(GH - 1, Math.ceil((y + r) / CS));
    let wsum = 0;
    const ws = [];
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++){
      const d = Math.hypot((i + 0.5) * CS - x, (j + 0.5) * CS - y) / r;
      if (d < 1){ const w = (1 - d * d); ws.push(j * GW + i, w); wsum += w; }
    }
    if (!wsum) return;
    for (let k = 0; k < ws.length; k += 2) snow[ws[k]] += amount * ws[k + 1] / wsum * (CS * CS);
    snowDirty = true;
  }
  // a finger shoving powder: most of what is under it travels with it and banks up at the leading edge,
  // the rest is left behind as a smear
  function plow(x, y, dx, dy, r, take = 0.85){
    const L = Math.hypot(dx, dy); if (L < 0.01) return 0;
    const i0 = Math.max(0, Math.floor((x - r) / CS)), i1 = Math.min(GW - 1, Math.ceil((x + r) / CS));
    const j0 = Math.max(0, Math.floor((y - r) / CS)), j1 = Math.min(GH - 1, Math.ceil((y + r) / CS));
    const moves = [];
    let moved = 0;
    const ux = dx / L, uy = dy / L;
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++){
      const k = j * GW + i, h = snow[k];
      if (h < 0.005) continue;
      const cx = (i + 0.5) * CS - x, cy = (j + 0.5) * CS - y;
      if (cx * cx + cy * cy > r * r) continue;
      const along = cx * ux + cy * uy, side = -cx * uy + cy * ux;
      const ahead = Math.sqrt(Math.max(r * r - side * side, 0)) + CS * rand(0.3, 1.5);
      const tx = x + ux * Math.max(ahead, along + L) - uy * side * rand(0.9, 1.25), ty = y + uy * Math.max(ahead, along + L) + ux * side * rand(0.9, 1.25);
      const ti = Math.floor(tx / CS), tj = Math.floor(ty / CS);
      const t = h * take;
      snow[k] -= t; moved += t;
      if (ti >= 0 && tj >= 0 && ti < GW && tj < GH) moves.push(tj * GW + ti, t);
    }
    for (let m = 0; m < moves.length; m += 2) snow[moves[m]] += moves[m + 1];
    if (moved) snowDirty = true;
    return moved;
  }
  function relaxSnow(dt){
    const T = 0.9;           // steepest a pile holds, CSS px per cell
    for (let pass = 0; pass < 2; pass++){
      for (let j = 1; j < GH - 1; j++) for (let i = 1; i < GW - 1; i++){
        const k = j * GW + i, h = snow[k];
        if (h < 0.05) continue;
        for (const n of [k + 1, k - 1, k + GW, k - GW]){
          const d = h - snow[n] - T;
          if (d > 0){ const m = d * 0.2; snow[k] -= m; snow[n] += m; }
        }
      }
    }
    // at rest, snow that misses the slab melts into the page and the slab holds only a dusting;
    // crushed powder stays frozen where it lies until the regrowth draws it back in
    const rest = state === 'rest';
    const melt = Math.exp(-dt * (rest ? 1.6 : state === 'regrowing' ? 5 : 0.02));
    for (let k = 0; k < snow.length; k++){
      if (snow[k] <= 0) continue;
      if (!rest || !slabMask[k]) snow[k] *= melt;
      else if (snow[k] > 1) snow[k] += (1 - snow[k]) * Math.min(1, dt * 2);
      if (snow[k] < 0.003) snow[k] = 0;
    }
    snowDirty = true;
  }

  // ---- particles (snowflakes, powder puffs, sparkles); ground x, y in stage coords, z height
  const parts = [];
  let flakeClock = 0, visible = true, lastActivity = performance.now();
  const snowWanted = () => opts.snow !== false && !reduceMotion.matches && visible && !document.hidden;
  const snowing = () => snowWanted() && performance.now() - lastActivity < SNOW_IDLE;
  function spawnFlake(){
    const x = rand(4, stageW - 4), y = rand(stageH * 0.25, stageH - 4);
    const z = y / 0.9 + rand(4, 16);
    parts.push({ x, y, z, vx: rand(-4, 4), vy: 0, vz: -rand(14, 22), size: rand(2.2, 4.6), kind: 0, age: 0, life: 99, dep: 0.35, ph: rand(0, TAU) });
  }
  function nearChip(x, y, r){
    for (const c of chips) if (Math.abs(c.x - x) < r + c.size * 0.5 && Math.abs(c.y - y) < r + c.size * 0.5) return true;
    return false;
  }
  function onShard(x, y){
    for (let i = shards.length - 1; i >= 0; i--){ const s = shards[i]; if (!s.gone && s.crushT < 0 && s.world && pointInPoly(s.world, x, y)) return s; }
    return null;
  }

  // ---- interaction
  let ptr = null, hoverLast = null, lastDown = null, swallowUntil = 0;
  const stageXY = e => { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  function onActivity(){
    const wasIdle = performance.now() - lastActivity >= SNOW_IDLE;
    lastActivity = performance.now();
    if (wasIdle && snowWanted()) wake(0.2);
  }
  function onDocDown(e){
    if (e.button !== 0 || ptr) return;
    const [x, y] = stageXY(e);
    if (x < 0 || y < 0 || x > stageW || y > stageH) return;
    const onBtn = e.target === button || button.contains(e.target);
    audio();
    if (state === 'rest'){
      if (!onBtn) return;
      ptr = { id: e.pointerId, x0: x, y0: y, x, y, lx: x, ly: y, vx: 0, vy: 0, speed: 0, dragging: false, onBtn };
      lastDown = [x - M, y - M]; impact = lastDown.slice(); holdT = 0;
      pressTarget = 1; snd('press'); wake(0.5);
      return;
    }
    // broken: presses on the chunks or the chip layer work the ice and never reach the page or the button
    const hit = onShard(x, y);
    const k = Math.floor(y / CS) * GW + Math.floor(x / CS);
    const onChip = !hit && nearChip(x, y, 6);
    if (!hit && !onBtn && !onChip && !(snow[k] > 0.12)) return;
    // a thin film of powder smeared over some other control must not take that control's clicks
    if (!hit && !onBtn && !onChip && e.target.closest && e.target.closest('a,button,input,select,textarea,label,summary,[role=button],[tabindex]')) return;
    e.preventDefault();
    ptr = { id: e.pointerId, x0: x, y0: y, x, y, lx: x, ly: y, vx: 0, vy: 0, speed: 0, dragging: false, onBtn, captured: true, hit };
    idleT = 0;
    wake(1);
  }
  function onDocMove(e){
    onActivity();
    const [x, y] = stageXY(e);
    const near = x > -40 && y > -40 && x < stageW + 40 && y < stageH + 40;
    if (near || lamp){ lamp = [x, y]; if (near) wake(0.25); }
    if (ptr && e.pointerId === ptr.id){
      ptr.x = x; ptr.y = y;
      if (!ptr.dragging && Math.hypot(x - ptr.x0, y - ptr.y0) > DRAG_START){
        ptr.dragging = true;
        if (state === 'rest' && ptr.onBtn){ pressTarget = 0; ptr.captured = true; }
      }
      wake(0.5);
      return;
    }
    // hovering brushes settled snow off the slab
    if (state !== 'rest' || !near){ hoverLast = null; return; }
    if (hoverLast){
      const moved = plow(x, y, x - hoverLast[0], y - hoverLast[1], 9);
      if (moved > 0.2 && Math.random() < 0.5){
        parts.push({ x: x + rand(-6, 6), y: y + rand(-6, 6), z: rand(1, 4), vx: (x - hoverLast[0]) * rand(4, 10), vy: (y - hoverLast[1]) * rand(4, 10), vz: rand(10, 40), size: rand(1, 2.2), kind: 1, age: 0, life: 1.2, dep: 0 });
      }
      if (moved) wake(0.6);
    }
    hoverLast = [x, y];
  }
  function onDocUp(e){
    if (!ptr || e.pointerId !== ptr.id) return;
    const p = ptr; ptr = null;
    scrapeLevel(0);
    pressTarget = 0;
    // anything we took hold of (a drag, or any press on the broken ice) must not become a click
    if (p.captured) swallowUntil = performance.now() + 600;
    if (state === 'rest'){ lastDown = [p.x - M, p.y - M]; return; }
    idleT = 0;
    if (!p.dragging && state === 'smashed'){
      const s = onShard(p.x, p.y);
      if (s) crushShard(s, p.x, p.y); else tapAt(p.x, p.y);
    }
    wake(1);
  }
  function onDocClick(e){
    if (e.detail > 0 && performance.now() < swallowUntil){ swallowUntil = 0; e.preventDefault(); e.stopImmediatePropagation(); }
  }
  function onClick(e){
    if (state !== 'rest'){
      // a real pointer on the broken ice is working the ice, not pressing; keyboard activation still works
      if (e.detail > 0){ e.preventDefault(); e.stopImmediatePropagation(); }
      return;
    }
    audio();
    const at = e.detail > 0 && lastDown ? lastDown : [bw / 2, bh / 2];
    lastDown = null;
    if (reduceMotion.matches){ pressTarget = 0; return; }
    smashAt(Math.max(2, Math.min(bw - 2, at[0])), Math.max(2, Math.min(bh - 2, at[1])));
  }
  const onCancel = e => { if (ptr && e.pointerId === ptr.id){ ptr = null; pressTarget = 0; scrapeLevel(0); } };
  document.addEventListener('pointerdown', onDocDown, true);
  document.addEventListener('click', onDocClick, true);
  addEventListener('pointermove', onDocMove, { passive: true });
  addEventListener('pointerup', onDocUp, true);
  addEventListener('pointercancel', onCancel, true);
  addEventListener('keydown', onActivity, true);
  addEventListener('wheel', onActivity, { passive: true });
  button.addEventListener('click', onClick, true);

  // ---- step
  let simT = 0, activeUntil = 0;
  let raf = 0, inFrame = false, last = performance.now();
  // one animation loop at most: a wake from inside a frame must not start a second loop
  function wake(sec){ activeUntil = Math.max(activeUntil, simT + sec); if (!raf && !inFrame) raf = requestAnimationFrame(frame); }

  // The pointer dragged through the broken ice, in small sub-steps so fast strokes don't tunnel.
  // Chunks are shoved and scrape along (friction, no fling); chips stick-slip; powder smears.
  function dragStep(dt){
    const dx = ptr.x - ptr.lx, dy = ptr.y - ptr.ly, L = Math.hypot(dx, dy);
    const f = Math.min(1, dt * 25);
    ptr.vx += (dx / dt - ptr.vx) * f; ptr.vy += (dy / dt - ptr.vy) * f;
    ptr.speed = Math.hypot(ptr.vx, ptr.vy);
    let contacts = 0, shove = 0;
    if (L > 0.01){
      const steps = Math.ceil(L / 2.5), ux = dx / L, uy = dy / L;
      for (let st = 1; st <= steps; st++){
        const px = ptr.lx + dx * st / steps, py = ptr.ly + dy * st / steps, sl = L / steps, sdx = ux * sl, sdy = uy * sl;
        for (const s of shards){
          if (s.gone || s.crushT >= 0) continue;
          const W = s.world;
          const inside = pointInPoly(W, px, py);
          const [qx, qy, dist] = closestOnPoly(W, px, py);
          if (!inside && dist > PUSHER_R) continue;
          // a big slab frozen to the page drags; small bits go more readily
          const give = Math.min(1, 0.55 + 260 / Math.max(s.area * s.scale * s.scale, 1));
          let mx, my;
          if (inside){ mx = sdx * give; my = sdy * give; }
          else { const nx = (qx - px) / (dist || 1), ny = (qy - py) / (dist || 1), pen = PUSHER_R - dist; mx = (nx * pen * 0.6 + sdx * 0.4) * give; my = (ny * pen * 0.6 + sdy * 0.4) * give; }
          s.off[0] += mx; s.off[1] += my;
          const c = centroid(W);
          s.ang += ((qx - c[0]) * my - (qy - c[1]) * mx) / Math.max(s.rad * s.rad * 3, 40);
          s.world = worldOf(s);
          shove += Math.hypot(mx, my);
          if (Math.random() < 0.08) deposit(qx, qy, 0.04, 1.5);
        }
        for (const c of chips){
          const reachC = PUSHER_R + c.size * 0.4;
          const ex = c.x - px, ey = c.y - py;
          if (Math.abs(ex) > reachC || Math.abs(ey) > reachC) continue;
          const d = Math.hypot(ex, ey);
          if (d >= reachC) continue;
          const nx = d > 1e-3 ? ex / d : ux, ny = d > 1e-3 ? ey / d : uy, pen = reachC - d;
          contacts++;
          if (c.stuck){
            // frozen on: it takes strain (and wobbles a little) until the bond lets go
            c.strain += sl * (0.5 + pen / reachC);
            const j = Math.min(1.5, c.strain * 0.2);
            c.jx = ux * j; c.jy = uy * j;
            if (c.strain > c.bond){
              const kick = rand(1.05, 1.6);
              loosen(c, ptr.vx * kick + nx * rand(10, 40), ptr.vy * kick + ny * rand(10, 40));
              snd('loose', ptr.speed);
              deposit(c.x, c.y, 0.05 * c.size, 1.5 + c.size * 0.3);
            }
          } else {
            c.x += nx * pen * 0.9; c.y += ny * pen * 0.9;
            const along = c.vx * ux + c.vy * uy;
            if (along < ptr.speed){ c.vx = ptr.vx * rand(0.95, 1.25) + nx * 15; c.vy = ptr.vy * rand(0.95, 1.25) + ny * 15; }
            c.rest = 0;
          }
        }
        plow(px, py, sdx, sdy, PUSHER_R * 0.8, 0.55);
      }
    }
    const lvl = Math.min(1, contacts / 25 + shove / Math.max(L, 1) * 0.6) * Math.min(1, ptr.speed / 300);
    scrapeLevel(lvl * 0.9 + Math.min(0.15, ptr.speed / 2500));
    if ((contacts || shove) && Math.random() < 0.6) snd('grit', lvl);
    ptr.lx = ptr.x; ptr.ly = ptr.y;
    idleT = 0;
  }

  function stepShards(dt){
    if (state === 'rest') return false;
    let moving = false;
    const regrowing = state === 'regrowing';
    for (const s of shards){
      if (s.gone) continue;
      if (regrowing){
        const k = 1 - Math.exp(-dt * 5.5);
        s.off[0] -= s.off[0] * k; s.off[1] -= s.off[1] * k; s.ang -= s.ang * k; s.z -= s.z * k;
        s.vx = s.vy = s.va = s.vz = 0;
        s.world = worldOf(s);
        moving = true;
        continue;
      }
      if (s.crushT >= 0){
        s.crushT += dt;
        s.crush = ease(Math.min(1, s.crushT / 0.12));
        s.scale = 1 + 0.05 * s.crush;
        if (!s.spawned && s.crushT >= 0.1){ s.spawned = true; s.world = worldOf(s); spawnChipsIn(s); }
        s.fade = 1 - ease(Math.min(1, Math.max(0, (s.crushT - 0.1) / (CRUSH_T - 0.1))));
        if (s.crushT >= CRUSH_T) s.gone = true;
        s.world = worldOf(s);
        moving = true;
        continue;
      }
      if (phaseT < s.delay){ s.world = worldOf(s); moving = true; continue; }
      s.vz -= 1100 * dt;
      s.z += s.vz * dt;
      if (s.z < 0){
        s.z = 0;
        if (s.vz < -60){ snd('tink', -s.vz); s.vz *= -0.2; s.vx *= 0.5; s.vy *= 0.5; } else s.vz = 0;
      }
      // on the ground a chunk is half frozen to the page: it scrapes to a stop almost at once
      const damp = Math.exp(-dt * (s.z > 0 ? 1.5 : 14));
      s.vx *= damp; s.vy *= damp; s.va *= damp;
      s.off[0] += s.vx * dt; s.off[1] += s.vy * dt; s.ang += s.va * dt;
      const cx = s.hc[0] + s.off[0] + M, cy = s.hc[1] + s.off[1] + M;
      const pad = Math.min(s.rad, 20);
      if (cx < pad) s.off[0] += pad - cx; if (cx > stageW - pad) s.off[0] -= cx - (stageW - pad);
      if (cy < pad) s.off[1] += pad - cy; if (cy > stageH - pad) s.off[1] -= cy - (stageH - pad);
      if (Math.abs(s.vx) + Math.abs(s.vy) + Math.abs(s.va) * 10 + Math.abs(s.vz) > 0.5 || s.z > 0) moving = true;
      s.world = worldOf(s);
    }
    // chunks shouldering each other apart
    if ((moving || (ptr && ptr.dragging)) && !regrowing){
      const L = live().filter(s => s.crushT < 0);
      for (let it = 0; it < 2; it++){
        for (let i = 0; i < L.length; i++) for (let j = i + 1; j < L.length; j++){
          const a = L[i], b = L[j];
          if (Math.abs(a.z - b.z) > 6) continue;
          const dx = (b.hc[0] + b.off[0]) - (a.hc[0] + a.off[0]), dy = (b.hc[1] + b.off[1]) - (a.hc[1] + a.off[1]);
          const rr = (a.rad + b.rad) * 1.6;
          if (dx * dx + dy * dy > rr * rr) continue;
          const m = sat(a.world, b.world);
          if (!m || m[2] < 0.35) continue;
          const [nx, ny, depth] = m;
          const ma = a.area * a.scale * a.scale, mb = b.area * b.scale * b.scale, wa = mb / (ma + mb), wb = ma / (ma + mb);
          const corr = (depth - 0.3) * 0.6;
          a.off[0] -= nx * corr * wa; a.off[1] -= ny * corr * wa;
          b.off[0] += nx * corr * wb; b.off[1] += ny * corr * wb;
          a.world = worldOf(a); b.world = worldOf(b);
          if (depth > 2 && ptr && ptr.dragging) snd('tink', depth * 20);
        }
      }
    }
    return moving;
  }

  const hash = new Map();
  function stepChips(dt){
    if (!chips.length) return false;
    let moving = false;
    if (state === 'regrowing'){
      const k = 1 - Math.exp(-dt * 7);
      for (const c of chips){
        c.x += (c.hx - c.x) * k; c.y += (c.hy - c.y) * k; c.z = 0; c.jx = c.jy = 0;
        c.a = Math.max(0, 1 - phaseT / 0.6);
      }
      return true;
    }
    for (const c of chips){
      if (c.vz || c.z > 0){
        c.vz -= 900 * dt; c.z += c.vz * dt;
        if (c.z <= 0){ c.z = 0; c.vz = c.vz < -40 ? -c.vz * 0.2 : 0; }
        moving = true;
      }
      if (c.stuck){
        c.bond = Math.min(c.bondMax, c.bond + dt * 5);
        c.strain *= Math.exp(-dt * 1.5);
        if (c.jx || c.jy){ const k = Math.exp(-dt * 30); c.jx *= k; c.jy *= k; if (Math.abs(c.jx) + Math.abs(c.jy) < 0.02) c.jx = c.jy = 0; moving = true; }
        continue;
      }
      const sp = Math.hypot(c.vx, c.vy);
      let ns = 0;
      if (sp > 0){
        // Coulomb friction: a constant deceleration, so skids stop short instead of coasting
        ns = Math.max(0, sp - MU * dt * (c.z > 0 ? 0.1 : 1));
        c.vx *= ns / sp; c.vy *= ns / sp;
        c.x += c.vx * dt; c.y += c.vy * dt;
        c.ang += c.va * dt; c.va *= Math.exp(-dt * 5);
        if (ns > 25 && c.z === 0) deposit(c.x, c.y, ns * dt * 0.0006 * c.size, 0.9 + c.size * 0.15);
        moving = true;
      }
      if (ns < 4){
        c.vx = c.vy = 0; c.rest += dt;
        // wet ice refreezes onto the page almost at once, weakly at first
        if (c.rest > 0.1){ c.stuck = true; c.bond = rand(1.5, 4); c.bondMax = rand(5, 12); c.strain = 0; }
        else moving = true;
      } else c.rest = 0;
      c.x = Math.max(3, Math.min(stageW - 3, c.x)); c.y = Math.max(3, Math.min(stageH - 3, c.y));
    }
    // skidding chips shunt into their neighbours: a frozen one takes the blow as strain
    hash.clear();
    const H = 8, key = (i, j) => i * 4096 + j;
    for (const c of chips){ const k = key(Math.floor(c.x / H), Math.floor(c.y / H)); let b = hash.get(k); if (!b) hash.set(k, b = []); b.push(c); }
    for (const c of chips){
      if (c.stuck || (!c.vx && !c.vy)) continue;
      const ci = Math.floor(c.x / H), cj = Math.floor(c.y / H);
      for (let j = cj - 1; j <= cj + 1; j++) for (let i = ci - 1; i <= ci + 1; i++){
        const b = hash.get(key(i, j)); if (!b) continue;
        for (const o of b){
          if (o === c) continue;
          const dx = o.x - c.x, dy = o.y - c.y, d = Math.hypot(dx, dy) || 1e-3;
          const ov = (c.size + o.size) * 0.38 - d;
          if (ov <= 0) continue;
          const nx = dx / d, ny = dy / d, closing = (c.vx - (o.vx || 0)) * nx + (c.vy - (o.vy || 0)) * ny;
          if (closing <= 0) continue;
          const mc = c.size * c.size, mo = o.size * o.size;
          if (o.stuck){
            o.strain += closing * (mc / (mc + mo)) * 0.06;
            if (o.strain > o.bond){ loosen(o, nx * closing * 0.7, ny * closing * 0.7); c.vx *= 0.5; c.vy *= 0.5; }
            else { c.x -= nx * ov; c.y -= ny * ov; c.vx -= nx * closing * 0.8; c.vy -= ny * closing * 0.8; }
          } else {
            const t = closing * mc / (mc + mo);
            o.vx += nx * t; o.vy += ny * t; c.vx -= nx * t; c.vy -= ny * t;
            o.x += nx * ov * 0.5; o.y += ny * ov * 0.5; c.x -= nx * ov * 0.5; c.y -= ny * ov * 0.5;
          }
        }
      }
    }
    return moving;
  }

  function stepPhases(dt){
    if (state === 'rest'){
      if (ptr && pressTarget > 0){ holdT += dt; craze = Math.min(0.16, holdT * 0.35) * press; }
      else craze *= Math.exp(-dt * 10);
      if (craze < 0.002) craze = 0;
      return;
    }
    phaseT += dt;
    craze = 0;
    crackR = Math.max(crackR, phaseT * 1400);
    if (state === 'smashed'){
      if (!ptr) idleT += dt;
      if (idleT > REFREEZE){
        state = 'regrowing'; phaseT = 0; grow = 0;
        const j = () => [rand(-0.1, 0.1) * bw, rand(-0.2, 0.2) * bh];
        const pts = [[bw * 0.5, bh * 0.5], [bw * 0.18, bh * 0.5], [bw * 0.82, bh * 0.5]].map(([x, y]) => { const [a, b] = j(); return [x + a, y + b]; });
        pts.forEach((p, i) => seeds.set(p, i * 2));
        reach = 0;
        for (const [x, y] of slabPoly) reach = Math.max(reach, Math.min(...pts.map(p => Math.hypot(x - p[0], y - p[1]))));
        reach += 14;
        growSlab = mkShard(slabPoly, slabPoly.map(() => 0), null);
        growSlab.seed = 0; growSlab.world = worldOf(growSlab);
        snd('refreeze');
        wake(REGROW_T + 0.3);
      }
    } else if (state === 'regrowing'){
      grow = ease(Math.min(1, Math.max(0, (phaseT - 0.15) / (REGROW_T - 0.3))));
      if (phaseT > REGROW_T){ toRest(); for (let k = 0; k < snow.length; k++) if (!slabMask[k]) snow[k] = 0; snowDirty = true; }
    }
  }

  function stepParts(dt){
    if (snowing()){
      flakeClock += dt * (opts.snowRate ?? 1) * stageW / 70;
      while (flakeClock > 1){ flakeClock -= 1; spawnFlake(); }
    }
    const paused = !snowing();
    for (const p of parts){
      p.age += dt;
      if (p.kind === 0 && paused && !p.melt && !p.out) p.out = p.age;
      if (p.out && p.age - p.out > 1.2){ p.dead = true; continue; }
      if (p.kind === 0){
        p.x += (p.vx + Math.sin(p.age * 1.3 + p.ph) * 6) * dt;
        p.z += p.vz * dt;
      } else {
        p.vz -= (p.kind === 1 ? 420 : 200) * dt;
        const drag = Math.exp(-dt * (p.kind === 1 ? 2.5 : 5));
        p.vx *= drag; p.vy *= drag;
        p.x += p.vx * dt; p.y += p.vy * dt; p.z += p.vz * dt;
      }
      if (p.z <= 0 || p.age > p.life){
        p.dead = true;
        if (p.z <= 0 && p.dep > 0){
          const i = Math.floor(p.x / CS), j = Math.floor(p.y / CS);
          const inGrid = i >= 0 && j >= 0 && i < GW && j < GH, k = j * GW + i;
          // snow settles on the whole slab, or on crushed powder; on a loose chunk (drawn over the grid) it melts
          const onIce = inGrid && (state === 'rest' ? slabMask[k] : snow[k] > 0.15 && !onShard(p.x, p.y));
          if (p.kind === 0){
            if (onIce) deposit(p.x, p.y, p.dep * p.size * 0.3, 1.5 + p.size * 0.3);
            else if (inGrid) parts.push({ x: p.x, y: p.y, z: 0.01, vx: 0, vy: 0, vz: 0, size: p.size, kind: 0, age: 0, life: 0.6, dep: 0, melt: 1 });
          } else if (state === 'rest' || !onShard(p.x, p.y)) deposit(p.x, p.y, p.dep * 4, 3);
        }
      }
    }
    for (let i = parts.length - 1; i >= 0; i--) if (parts[i].dead) parts.splice(i, 1);
    for (const p of parts) if (p.melt){ p.vz = 0; p.z = 0.01; }
  }

  // ---- render
  let vbuf = new Float32Array(16 * 3 * 256);
  function buildVerts(list){
    let n = 0;
    const need = list.reduce((a, s) => a + (s.gone ? 0 : s.local.length * 3), 0) * 16;
    if (vbuf.length < need) vbuf = new Float32Array(need * 2);
    const B = vbuf;
    for (const s of list){
      if (s.gone) continue;
      const W = s.world || worldOf(s);
      const lift = -s.z * 0.6;
      const c = centroid(W);
      const hcx = s.hc[0], hcy = s.hc[1];
      for (let i = 0; i < W.length; i++){
        const a = W[i], b = W[(i + 1) % W.length];
        let nx = b[1] - a[1], ny = a[0] - b[0];
        const l = Math.hypot(nx, ny) || 1; nx /= l; ny /= l;
        if ((a[0] - c[0]) * nx + (a[1] - c[1]) * ny < 0){ nx = -nx; ny = -ny; }
        const h = Math.max(0.01, (a[0] - c[0]) * nx + (a[1] - c[1]) * ny);
        const la = s.local[i], lb = s.local[(i + 1) % W.length];
        const kind = s.flags[i], mid = s.mids[i];
        for (const [p, hx, hy, d] of [[c, hcx, hcy, h], [a, la[0] + hcx, la[1] + hcy, 0], [b, lb[0] + hcx, lb[1] + hcy, 0]]){
          B[n++] = p[0]; B[n++] = p[1] + lift; B[n++] = hx; B[n++] = hy;
          B[n++] = d; B[n++] = nx; B[n++] = ny; B[n++] = kind;
          B[n++] = mid; B[n++] = s.z; B[n++] = s.seed; B[n++] = -lift;
          B[n++] = s.crush; B[n++] = s.fade; B[n++] = 0; B[n++] = 0;
        }
      }
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, shardBuf);
    gl.bufferData(gl.ARRAY_BUFFER, B.subarray(0, n), gl.DYNAMIC_DRAW);
    return n / 16;
  }
  function drawIce(list, L, growOn){
    const nv = buildVerts(list);
    if (!nv) return;
    gl.bindVertexArray(shardVao);
    gl.useProgram(P.shadow.p);
    gl.uniform2f(P.shadow.u.uStage, stageW, stageH);
    gl.uniform1f(P.shadow.u.uShadow, 1);
    gl.uniform2f(P.shadow.u.uShadowDir, 0.55 * (1 - press * 0.5), 0.8 * (1 - press * 0.5));
    gl.uniform1f(P.shadow.u.uPress, press);
    gl.uniform1f(P.shadow.u.uFlat, growOn ? 1 - Math.pow(grow, 8) : 0);
    gl.drawArrays(gl.TRIANGLES, 0, nv);
    const u = P.ice.u;
    gl.useProgram(P.ice.p);
    gl.uniform2f(u.uStage, stageW, stageH);
    gl.uniform1f(u.uShadow, 0);
    bindTex(0, bgTex); gl.uniform1i(u.uBgTex, 0);
    bindTex(1, textTex); gl.uniform1i(u.uText, 1);
    gl.uniform1f(u.uHasBg, surface ? 1 : 0);
    gl.uniform3f(u.uBgCol, ...bgCol);
    gl.uniform4f(u.uBg, ...bgOff);
    gl.uniform2f(u.uSize, bw, bh);
    gl.uniform1f(u.uCrackR, crackR); gl.uniform1f(u.uPress, press); gl.uniform1f(u.uTime, simT);
    gl.uniform1f(u.uCraze, craze); gl.uniform1f(u.uGrow, grow); gl.uniform1f(u.uGrowOn, growOn ? 1 : 0); gl.uniform1f(u.uReach, reach);
    gl.uniform2f(u.uImpact, ...impact); gl.uniform2f(u.uLamp, ...L);
    gl.uniform4fv(u.uPlanes, planes); gl.uniform2fv(u.uSeeds, seeds);
    gl.drawArrays(gl.TRIANGLES, 0, nv);
  }
  function drawPowder(L){
    if (snowDirty){
      for (let k = 0; k < snow.length; k++) snowBytes[k] = Math.min(255, snow[k] / SNOW_PX * 255 + 0.5);
      gl.bindTexture(gl.TEXTURE_2D, snowTex);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, GW, GH, 0, gl.RED, gl.UNSIGNED_BYTE, snowBytes);
      snowDirty = false;
    }
    gl.useProgram(P.snow.p);
    bindTex(2, snowTex); gl.uniform1i(P.snow.u.uSnow, 2);
    gl.uniform2f(P.snow.u.uGridPx, GW * CS, GH * CS);
    gl.uniform1f(P.snow.u.uDpr, dpr); gl.uniform1f(P.snow.u.uStageH, stageH); gl.uniform1f(P.snow.u.uTime, simT);
    gl.uniform2f(P.snow.u.uLamp, ...L);
    gl.bindVertexArray(fullVao); gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
  const pdata = new Float32Array(6 * 2048);
  let cdata = new Float32Array(8 * 256);
  function render(){
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    press += (pressTarget - press) * 0.35;
    const L = lamp || [stageW * 0.2, -stageH * 0.8];
    if (state === 'rest'){
      // snow sits on top of the whole slab
      drawIce(shards, L, false);
      drawPowder(L);
    } else {
      // crushed powder and chips lie on the page, under the chunks
      drawPowder(L);
      if (chips.length){
        if (cdata.length < chips.length * 8) cdata = new Float32Array(chips.length * 16);
        let m = 0;
        for (const c of chips){
          if (c.a <= 0.001) continue;
          const shrink = state === 'regrowing' ? 0.4 + 0.6 * c.a : 1;
          cdata.set([c.x + c.jx, c.y + c.jy, c.size * shrink, c.a, c.ang, c.seed, c.z, c.frost], m * 8); m++;
        }
        if (m){
          gl.bindBuffer(gl.ARRAY_BUFFER, chipBuf);
          gl.bufferData(gl.ARRAY_BUFFER, cdata.subarray(0, m * 8), gl.DYNAMIC_DRAW);
          const u = P.chip.u;
          gl.useProgram(P.chip.p);
          gl.uniform2f(u.uStage, stageW, stageH); gl.uniform1f(u.uDpr, dpr);
          bindTex(0, bgTex); gl.uniform1i(u.uBgTex, 0);
          gl.uniform1f(u.uHasBg, surface ? 1 : 0); gl.uniform3f(u.uBgCol, ...bgCol); gl.uniform4f(u.uBg, ...bgOff);
          gl.uniform2f(u.uLamp, ...L); gl.uniform1f(u.uTime, simT);
          gl.bindVertexArray(chipVao); gl.drawArrays(gl.POINTS, 0, m);
        }
      }
      drawIce(shards, L, false);
      if (state === 'regrowing' && growSlab && grow > 0.001) drawIce([growSlab], L, true);
    }
    // particles: shadows then bodies
    if (parts.length){
      const n = Math.min(parts.length, 1000);
      let m = 0;
      for (let i = 0; i < n; i++){
        const p = parts[i];
        if (p.kind === 2) continue;
        const fadeIn = (p.kind === 0 ? Math.min(1, p.age / 0.8) : 1) * (p.out ? Math.max(0, 1 - (p.age - p.out) / 1.2) : 1);
        const a = (p.melt ? 1 - p.age / p.life : 1) * fadeIn * Math.max(0, 1 - p.z / 140);
        pdata.set([p.x + p.z * 0.35, p.y + p.z * 0.1, p.size * 1.3, a * 0.8, 3, 0], m * 6); m++;
      }
      for (let i = 0; i < n; i++){
        const p = parts[i];
        const fadeIn = (p.kind === 0 ? Math.min(1, p.age / 0.8) : 1) * (p.out ? Math.max(0, 1 - (p.age - p.out) / 1.2) : 1);
        const life = p.kind === 2 ? 1 - p.age / p.life : p.kind === 1 ? Math.min(1, (p.life - p.age) / 0.3) : p.melt ? 1 - p.age / p.life : 1;
        pdata.set([p.x, p.y - p.z * 0.9, p.size * (p.kind === 2 ? 1.6 : 1), Math.max(0, life) * fadeIn * (p.kind === 0 ? 0.92 : 0.95), p.kind, Math.random()], m * 6); m++;
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, partBuf);
      gl.bufferData(gl.ARRAY_BUFFER, pdata.subarray(0, m * 6), gl.DYNAMIC_DRAW);
      gl.useProgram(P.part.p);
      gl.uniform2f(P.part.u.uStage, stageW, stageH); gl.uniform1f(P.part.u.uDpr, dpr);
      gl.bindVertexArray(partVao); gl.drawArrays(gl.POINTS, 0, m);
    }
    gl.bindVertexArray(null);
    gl.disable(gl.BLEND);
  }

  let manual = false, frameCount = 0, renderAcc = 0;
  function advance(dt){
    frameCount++;
    simT += dt;
    stepPhases(dt);
    if (ptr && ptr.dragging && state === 'smashed') dragStep(dt);
    const moving = stepShards(dt) | stepChips(dt);
    stepParts(dt);
    if (snowDirty || state !== 'rest') relaxSnow(dt);
    const busy = simT < activeUntil || moving || state !== 'rest' || ptr || Math.abs(press - pressTarget) > 0.01 || craze > 0;
    // snowfall alone renders at half rate; anything the user is doing gets every frame
    renderAcc += dt;
    if (busy || dirty || renderAcc > 1 / 31){ render(); dirty = false; renderAcc = 0; }
    return busy || parts.length > 0 || snowing();
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
  const io = new IntersectionObserver(es => { visible = es[es.length - 1].isIntersecting; if (visible) wake(0.2); });
  io.observe(button);
  const onScrollResize = () => { place(); onActivity(); };
  const onVis = () => { if (!document.hidden){ last = performance.now(); wake(0.2); } };
  addEventListener('resize', onScrollResize); addEventListener('scroll', onScrollResize, { passive: true });
  document.addEventListener('visibilitychange', onVis);
  canvas.addEventListener('webglcontextlost', e => e.preventDefault());
  canvas.addEventListener('webglcontextrestored', () => location.reload());

  resize();

  return {
    supported: true,
    get state(){ return state; },
    get _frames(){ return frameCount; },
    get _chips(){ return chips.length; },
    get _shards(){ return shards.filter(s => !s.gone).length; },
    _shardAt(i){ const s = shards.filter(s => !s.gone && s.crushT < 0).sort((a, b) => b.area - a.area)[i]; if (!s) return null; const c = centroid(s.world); const r = canvas.getBoundingClientRect(); return [c[0] + r.left, c[1] + r.top]; },
    get _stuck(){ return chips.filter(c => c.stuck).length; },
    get _snowing(){ return snowing(); },
    get _asleep(){ return !raf && !inFrame; },
    detach(){
      cancelAnimationFrame(raf); ro.disconnect(); io.disconnect();
      removeEventListener('resize', onScrollResize); removeEventListener('scroll', onScrollResize);
      document.removeEventListener('visibilitychange', onVis);
      document.removeEventListener('pointerdown', onDocDown, true);
      document.removeEventListener('click', onDocClick, true);
      removeEventListener('pointermove', onDocMove); removeEventListener('pointerup', onDocUp, true); removeEventListener('pointercancel', onCancel, true);
      removeEventListener('keydown', onActivity, true); removeEventListener('wheel', onActivity);
      button.removeEventListener('click', onClick, true);
      scrapeLevel(0);
      if (saved === null) button.removeAttribute('style'); else button.setAttribute('style', saved);
      canvas.remove();
      gl.getExtension('WEBGL_lose_context')?.loseContext();
    },
    _manual(on){ manual = on; if (!on) wake(0.1); },
    _advance(sec, fps = 60){ for (let i = 0, n = Math.round(sec * fps); i < n; i++) advance(1 / fps); },
    _lamp(x, y){ lamp = [x, y]; dirty = true; },
  };
}

window.ElementalFX = Object.assign(window.ElementalFX || {}, { attachIce, iceAudio });
})();
