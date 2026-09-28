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
// the ice scrapes it: chunks are frozen down and only creep a few px once their bond gives, stuck chips
// resist, then break loose and skid a short way leaving frost trails, and powder smears. After a quiet moment the pieces draw back together and the
// slab regrows as frost crystals. Pointer presses on the broken ice never click anything. The cursor is
// the light: glints follow it.
// If WebGL2 is unavailable, the button is left untouched.
//
// Options:
//   surface: { image: CanvasImageSource, rect: () => DOMRect }   what the page looks like under the ice
//            (refracted through it). Without it the ice refracts the nearest ancestor's background colour.
//   snow: false to stop the snowfall.   snowIdle: seconds without any page activity before the snowfall
//   pauses and the loop sleeps (default 8).   sound: false to stay silent.
//   refreezeAfter: seconds of quiet before it refreezes (default 5).   zIndex: stage canvas (default 10).

(() => {
'use strict';

const TAU = Math.PI * 2;
const CS = 2;                  // CSS px per powder-grid cell
const SNOW_PX = 4;             // powder height (CSS px) that maps to a full R8 texel
const DRAG_START = 6;
const PUSHER_R = 10;
const VSTRIDE = 20;            // floats per chunk vertex
const GRAB = 16;               // how far off the ice a press still grabs it, CSS px
const MU = 1100;               // kinetic friction of a skidding chip, CSS px/s^2
const SKID_MAX = 230;          // fastest a chip breaks loose, so a hard shove skids it ~24px, not across the page
const CRUSH_T = 0.22, REGROW_T = 1.15;
const LIGHT_H = 150;           // height of the cursor "lamp" above the page, CSS px
const EYE_H = 900;             // viewer's distance from the screen, CSS px
const CHIP_SHARP = 380;        // specular lobe of a chip facet
const BLOOMS = 5;              // how many of the brightest chip glints flare into full stars per frame

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
// a tiny facet catching the cursor lamp: sharp enough that only a few of many random facets fire at once.
// The eye sits over the middle of the viewport, not over every pixel, so scrolling shifts which facets
// line up and the ice shimmers as it moves past.
uniform vec2 uEye;
vec3 halfOf(vec2 at, vec2 lamp){
  vec3 L = normalize(vec3(lamp - at, ${LIGHT_H.toFixed(1)}));
  vec3 V = normalize(vec3(uEye - at, ${EYE_H.toFixed(1)}));
  return normalize(L + V);
}
float glintOf(vec3 n, vec2 at, vec2 lamp, float sharp){
  return pow(max(dot(n, halfOf(at, lamp)), 0.), sharp);
}
// On a pale page a clear facet is seen by what it sends back: tilted one way it passes the white page
// through, a few degrees further it totally internally reflects the dim room instead and goes grey-blue.
// A light moving over the pile flips facets between the two, which is the shimmer that reads on white.
float winkOf(vec3 n, vec2 at, vec2 lamp, float phase){
  float ang = acos(clamp(dot(n, halfOf(at, lamp)), -1., 1.));
  return smoothstep(0.3, 0.7, 0.5 + 0.5 * cos(ang * 24. + phase * 6.2831));
}
`;

// One vertex stream serves both the chunk bodies and their shadows. Every fan triangle belongs to one
// polygon edge, so aEdge.x interpolates to the exact distance from that edge's line.
const VS_SHARD = `#version 300 es
in vec2 aPos;      // stage CSS px, y down (already lifted by the chunk's height)
in vec2 aHome;     // button-local CSS px of the unbroken slab
in vec4 aEdge;     // distance to edge line, outward normal xy, kind (0 rim, 1 crack)
in vec4 aInfo;     // crack reveal distance, height, seed, ground offset y (lift)
in vec4 aX;        // crush (0..1), fade, settled snow, unused
in vec4 aY;        // crush origin xy (home px), chunk radius, time since the crush began
uniform vec2 uStage;
uniform float uShadow;
uniform vec2 uShadowDir;
out vec2 vHome; out vec4 vEdge; out vec4 vInfo; out vec4 vX; out vec2 vStage; out vec4 vY;
void main(){
  vec2 p = aPos; vY = aY;
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
in vec2 vHome; in vec4 vEdge; in vec4 vInfo; in vec4 vX; in vec2 vStage; in vec4 vY;
uniform sampler2D uBgTex; uniform sampler2D uText;
uniform float uHasBg; uniform vec3 uBgCol;
uniform vec4 uBg;          // stage->surface: offset xy, size zw (CSS px)
uniform vec2 uSize;        // button CSS px
uniform float uCrackR, uPress, uTime, uCraze, uGrow, uGrowOn, uReach, uLight;
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
  // white frost reads against dark ice; on a pale page the ice is pale too, so the cut goes deep blue-grey
  vec3 stroke = mix(vec3(0.94, 0.98, 1.0), vec3(0.15, 0.29, 0.45), uLight);
  col = mix(col, stroke, t0 * (0.86 + 0.14 * hash12(floor(h * 2.))) * mix(0.9, 0.95, uLight));
  col = mix(col, vec3(0.03, 0.10, 0.20), clamp(tS - t0, 0., 1.) * 0.55 * (1. - uLight));
  col += vec3(0.6, 0.8, 1.) * clamp(tL - t0, 0., 1.) * 0.15 * (1. - uLight);
  col = mix(col, vec3(1.), clamp(tL - t0, 0., 1.) * 0.7 * uLight);

  // frosted rim and a light dusting of rime
  float fr = fbm(h * 0.35 + vInfo.z * 3.);
  if (rim) col = mix(col, vec3(0.93, 0.97, 1.0), smoothstep(5.5, 0.4, d + (fr - 0.5) * 5.) * (0.55 + 0.45 * fbm(h * 1.3)));
  float rime = smoothstep(0.55, 0.85, fbm(h * 0.09 + 2.)) * 0.22 + step(0.93, hash12(floor(h * 1.6))) * 0.25 * fbm(h * 0.05);
  col = mix(col, vec3(0.95, 0.98, 1.), rime);
  // snow that has settled on a loose chunk since the break
  if (vX.z > 0.){
    float sn = fbm(h * 0.21 + vInfo.z * 13.) + 0.35 * vnoise(h * 1.1);
    float m = smoothstep(1.3 - vX.z, 1.38 - vX.z, sn) * smoothstep(0.5, 3., d);
    col = mix(col, vec3(0.93, 0.97, 1.0) * (0.92 + 0.08 * hash12(floor(h * 2.))), m * 0.92);
  }

  // the cracks between chunks
  if (!rim){
    float vis = step(vInfo.x, uCrackR);
    col = mix(col, vec3(0.97, 0.99, 1.0), smoothstep(1.4, 0.2, d) * vis * 0.85);
    col = mix(col, col * 0.7, smoothstep(3.5, 1.2, d) * (1. - smoothstep(1.2, 0.6, d)) * vis * 0.5);
  }

  // Stress crazing: under a held finger on the whole slab, or racing out from the fist through a chunk
  // being crushed. Crushed ice whitens along a network of cracks and radial splits, flashes as it
  // fails, and stays partly clear between them rather than turning into a flat white tile.
  {
    float inside = 0., spoke = 0.;
    if (uCraze > 0.){
      vec2 q = h - uImpact;
      float r = length(q), ang = atan(q.y, q.x);
      float reach = uCraze * length(uSize) * (0.7 + 0.5 * vnoise(vec2(ang * 2.2, 3.)));
      inside = smoothstep(reach, reach - 8., r);
      float sa = ang * 14. / 6.2831 + (fbm(q * 0.06) - 0.5) * 1.2;
      float spokePx = abs(fract(sa + 0.5) - 0.5) * 6.2831 / 14. * r;
      float spokeLen = reach * (0.5 + 0.6 * hash12(vec2(floor(sa + 0.5), 4.)));
      spoke = smoothstep(1.0, 0.25, spokePx) * smoothstep(spokeLen, spokeLen * 0.7, r) * step(2., r);
    }
    float inC = 0.;
    if (crush > 0.){
      vec2 q = h - vY.xy;
      float r = length(q), ang = atan(q.y, q.x);
      float front = crush * (vY.z * 2.4 + 6.) * (0.75 + 0.5 * vnoise(vec2(ang * 2., 5.)));
      inC = smoothstep(front, front - 7., r);
      inside = max(inside, inC);
      float sa = ang * 11. / 6.2831 + (fbm(q * 0.08 + vInfo.z * 7.) - 0.5) * 1.4;
      float spokePx = abs(fract(sa + 0.5) - 0.5) * 6.2831 / 11. * r;
      spoke = max(spoke, smoothstep(0.9, 0.2, spokePx) * inC * step(1.5, r));
    }
    if (inside > 0. || spoke > 0.){
      float wall = 0.;
      for (int s2 = 0; s2 < 2; s2++){
        float cell = s2 == 0 ? 4.5 : 2.2;
        vec2 g = h / cell, gi = floor(g);
        float d1 = 9., d2 = 9.;
        for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++){
          vec2 c = gi + vec2(i, j);
          float dd = length(g - c - hash22(c + float(s2) * 17.));
          if (dd < d1){ d2 = d1; d1 = dd; } else if (dd < d2) d2 = dd;
        }
        wall = max(wall, smoothstep(0.7, 0.1, (d2 - d1) * cell) * (s2 == 0 ? 1. : crush * 0.5));
      }
      // white fracture planes read on dark ice; on pale ice they show by their grey-blue shadowed edge
      vec3 crackC = mix(vec3(0.97, 0.99, 1.0), vec3(0.42, 0.55, 0.7), uLight * 0.85);
      col = mix(col, crackC, clamp(wall * (0.6 + 0.3 * crush) * inside + spoke * 0.95, 0., 1.));
      col = mix(col, vec3(0.86, 0.93, 1.0) * (0.9 + 0.12 * hash12(floor(h * 1.5))), inC * crush * mix(0.3, 0.15, uLight));
      float flash = inC * exp(-vY.w / 0.05);
      col += vec3(0.8, 0.92, 1.0) * flash * 0.45;
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
    vec3 spect = 0.75 + 0.25 * cos(6.2831 * (hash12(gc) + vec3(0., 0.33, 0.67)));
    col += mix(vec3(0.95, 0.99, 1.), spect, 0.5) * gl * 2.2;
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

// Settled snow and crushed-ice powder, from the powder grid: R is the height, G how much of it is
// crushed ice rather than snowfall. Snow is soft and fluffy. Crushed ice is packed and hard-edged, a
// mosaic of separate grains: clear ones let the page show through (bent), frosted ones scatter white,
// and the seams between them scatter light too, so they read bright on a dark page and cool blue on a
// pale one rather than as dark grout.
const FS_SNOW = COMMON + `
uniform sampler2D uSnow;
uniform sampler2D uBgTex; uniform float uHasBg; uniform vec3 uBgCol; uniform vec4 uBg;
uniform vec2 uGridPx;      // grid size in CSS px
uniform vec2 uStageSz;     // stage size in CSS px
uniform float uDpr, uStageH, uTime, uLight;
uniform vec2 uLamp;
out vec4 o;
float S(vec2 css){ return texture(uSnow, css / uGridPx).r * ${SNOW_PX.toFixed(1)}; }
vec3 bgAt(vec2 sp){
  if (uHasBg < 0.5) return uBgCol;
  return texture(uBgTex, (sp + uBg.xy) / uBg.zw).rgb;
}
void main(){
  vec2 css = vec2(gl_FragCoord.x, uStageH * uDpr - gl_FragCoord.y) / uDpr;
  vec2 tx = texture(uSnow, css / uGridPx).rg;
  float h = tx.r * ${SNOW_PX.toFixed(1)};
  if (h < 0.01){ o = vec4(0.); return; }
  float ice = smoothstep(0.25, 0.7, tx.g);
  float gx = S(css + vec2(1.5, 0.)) - S(css - vec2(1.5, 0.));
  float gy = S(css + vec2(0., 1.5)) - S(css - vec2(0., 1.5));
  vec3 n = normalize(vec3(-gx, -gy, 1.6));
  vec3 L = normalize(vec3(-0.45, -0.6, 0.66));
  float grain = hash12(floor(css * 2.));

  // snow
  float coverS = smoothstep(0.05, 0.55, h + (fbm(css * 0.6) - 0.5) * 0.5 + (grain - 0.5) * 0.25);
  float dif = max(dot(n, L), 0.);
  vec3 colS = vec3(0.82, 0.9, 0.98) * (0.55 + 0.5 * dif) + vec3(0.10, 0.16, 0.26) * (1. - dif) * 0.35;
  colS = min(colS * (0.93 + 0.1 * grain), vec3(1.));
  {
    vec2 gc = floor(css * 0.55);
    if (hash12(gc + 7.7) > 0.72){
      vec3 nr = normalize(vec3((hash22(gc) - 0.5) * 1.2, 1.));
      float sg = glintOf(nr, (gc + 0.5) / 0.55, uLamp, 400.) * smoothstep(0.15, 0.6, h);
      colS += vec3(1.) * sg * smoothstep(0.25, 0.03, length(fract(css * 0.55) - 0.5)) * 2.;
    }
  }

  // crushed ice
  vec2 g = css / 2.6, gi = floor(g);
  float d1 = 9., d2 = 9.; vec2 c1 = gi;
  for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++){
    vec2 c = gi + vec2(i, j);
    float dd = length(g - c - hash22(c));
    if (dd < d1){ d2 = d1; d1 = dd; c1 = c; } else if (dd < d2) d2 = dd;
  }
  float seamW = smoothstep(0.13, 0.03, d2 - d1);
  float hA = hash12(c1 + 9.), hF = hash12(c1 + 4.4), hG = hash12(c1 + 21.7);
  vec3 fn = normalize(vec3((hash22(c1 + 3.1) - 0.5) * 0.7, 1.) + n * 0.3);
  float fd = max(dot(fn, L), 0.);
  float edge = h + (vnoise(css * 0.45) - 0.5) * 0.35 + (hash12(c1) - 0.5) * 0.25;
  float coverI = smoothstep(0.26, 0.28, edge);
  float rimL = 1. - smoothstep(0.33, 0.38, edge);
  // a thin scatter is separate grains with bare page between them; only a deep pile packs solid
  float present = step(mix(0.5, 0.0, smoothstep(0.35, 1.6, h)), hG);
  float fr = smoothstep(mix(0.62, 0.5, uLight), 0.92, hF);
  vec3 bgc = bgAt(css - fn.xy * 6.);
  vec3 clearG = bgc * mix(vec3(0.9), vec3(0.88, 0.94, 0.99), uLight) + mix(vec3(0.05, 0.065, 0.08), vec3(0.01, 0.02, 0.04), uLight);
  vec3 frostG = mix(vec3(0.78, 0.82, 0.87), vec3(0.91, 0.95, 1.0), uLight);
  // each grain is a little solid, not a flat tile: its side toward the light is brighter
  vec2 toC = g - c1 - hash22(c1);
  float side = clamp(dot(toC, normalize(L.xy)) * 1.4 + 0.5, 0., 1.);
  vec3 colI = mix(clearG, frostG, fr) * (0.84 + 0.28 * hA) * (0.82 + 0.22 * fd) * mix(0.8 + 0.35 * side, 0.9 + 0.2 * side, uLight);
  // grey-blue darks on a pale page (facets turned away by total internal reflection)
  vec2 cc0 = (c1 + hash22(c1)) * 2.6;
  colI = mix(colI, vec3(0.46, 0.55, 0.66) * (0.9 + 0.2 * hA), mix(step(hF, 0.12), step(hF, 0.4) * winkOf(fn, cc0, uLamp, hA * 5.1), step(0.12, hF)) * uLight * 0.95);
  float aG = mix(0.5 + 0.3 * hA, 0.9, fr);
  vec3 seam = mix(vec3(0.7, 0.75, 0.8), vec3(0.55, 0.64, 0.74), uLight);
  colI = mix(colI, seam, seamW * 0.45);
  aG = mix(aG, max(aG, 0.75), seamW * present);
  aG *= mix(present, 1., rimL * 0.3);
  colI = mix(colI, seam, rimL * 0.6); aG = max(aG, rimL * 0.8 * present);
  vec2 cc = (c1 + hash22(c1)) * 2.6;
  float spk = glintOf(fn, cc, uLamp, mix(260., 100., uLight)) * (0.8 + 0.2 * sin(uTime * 9. + hA * 50.)) * present;
  vec3 fire = clamp(0.5 + 0.9 * cos(6.2831 * (hA * 3. + vec3(0., 0.33, 0.67))), 0., 1.);
  colI += vec3(1.) * spk * 1.4 * (1. - uLight);
  colI = mix(colI, mix(vec3(1.), fire, 0.35), smoothstep(0.1, 0.6, spk) * uLight);
  aG = max(aG, min(spk, 1.) * present);
  float aI = coverI * aG;

  // nothing is drawn right up to the stage edge, so the canvas rectangle never shows
  vec2 toE = min(css, uStageSz - css);
  float fade = smoothstep(1., 13., min(toE.x, toE.y));
  o = vec4(mix(colS * coverS, colI * aI, ice), mix(coverS, aI, ice)) * fade;
}`;

// Ice chips frozen onto the page, one point sprite each: an irregular convex fleck, clear or frosted,
// cut into three facets. The CPU lights the facets (aR.xyz, one specular per facet) so it can pick the
// few brightest glints in the pile each frame and flare only those into full stars (aR.w).
const VS_CHIP = `#version 300 es
in vec4 aP;        // x, y (stage CSS px), size, alpha
in vec4 aQ;        // angle, seed, hop height, frost
in vec4 aR;        // facet glints 0..2, star bloom
uniform vec2 uStage; uniform float uDpr;
out vec4 vP; out vec4 vQ; out vec4 vR;
void main(){
  vec2 p = aP.xy - vec2(0., aQ.z * 0.9);
  gl_Position = vec4(p.x / uStage.x * 2. - 1., 1. - p.y / uStage.y * 2., 0., 1.);
  gl_PointSize = (aP.z + (aR.w > 0.01 ? 32. : 7.)) * uDpr;
  vP = vec4(p, aP.zw); vQ = aQ; vR = aR;
}`;
const FS_CHIP = COMMON + `
in vec4 vP; in vec4 vQ; in vec4 vR;
uniform sampler2D uBgTex; uniform float uHasBg; uniform vec3 uBgCol; uniform vec4 uBg;
uniform vec2 uLamp; uniform float uTime, uLight;
out vec4 o;
vec3 bgAt(vec2 sp){
  if (uHasBg < 0.5) return uBgCol;
  return texture(uBgTex, (sp + uBg.xy) / uBg.zw).rgb;
}
void main(){
  float S = vP.z + (vR.w > 0.01 ? 32. : 7.);
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

  // three facets meeting at an off-centre apex
  vec2 apex = (hash22(vec2(seed * 7.3, 4.)) - 0.5) * R * 0.6;
  float a0 = hash12(vec2(seed * 11.1, 3.)) * 6.2831;
  vec2 v = q - apex;
  float fi = floor(mod(atan(v.y, v.x) - a0, 6.2831) / 2.0944);
  float g = fi < 0.5 ? vR.x : fi < 1.5 ? vR.y : vR.z;
  float ridge = 1e9;
  for (int k = 0; k < 3; k++){
    float b = a0 + float(k) * 2.0944;
    vec2 dir = vec2(cos(b), sin(b));
    ridge = min(ridge, dot(v, dir) > 0. ? abs(dot(v, vec2(-dir.y, dir.x))) : length(v));
  }

  // the spark sits on the brightest facet
  float gm = max(vR.x, max(vR.y, vR.z));
  float bi = vR.x >= gm ? 0. : vR.y >= gm ? 1. : 2.;
  float bb = a0 + (bi + 0.5) * 2.0944;
  vec2 sq = apex + vec2(cos(bb), sin(bb)) * R * 0.3;
  vec2 starAt = vec2(c * sq.x - s * sq.y, s * sq.x + c * sq.y);
  float amp = gm * vP.w;
  vec3 starC = vec3(0.); float starA = 0., ring = 0.;
  if (amp > 0.3){
    float len = (2.6 + vR.w * 8.) * (1. + 0.45 * uLight);
    vec2 p = pc - starAt, ap = abs(p);
    float r = length(p);
    float k = smoothstep(0.3, 0.9, amp);
    float core = exp(-r * r * 1.6) * k;
    float tx = clamp(1. - ap.x / len, 0., 1.), ty = clamp(1. - ap.y / len, 0., 1.);
    // hairline spikes; red spreads widest across an arm and blue least, so the edges split into colour
    vec3 wid = mix(vec3(1.6, 2.1, 2.8), vec3(1.5, 1.8, 2.2), uLight);
    vec3 arms = exp(-ap.y * wid) * tx * tx + exp(-ap.x * wid) * ty * ty;
    // along each arm the colour runs out through the spectrum, white at the core, then cyan, green,
    // yellow to red at the tip, as light split by a prism would
    float sp = clamp(0.3 + (r - 1.2) / len * 0.85, 0., 1.);
    vec3 hue = clamp(vec3(abs(sp - 0.85) < 0.35 ? 1. - abs(sp - 0.85) / 0.35 : 0., 1. - abs(sp - 0.55) / 0.3, 1. - abs(sp - 0.2) / 0.3), 0., 1.) * 1.1;
    vec3 hueD = 0.5 + 0.5 * cos(6.2831 * (r / len * 0.8 + seed + vec3(0., 0.33, 0.67)));
    arms *= mix(mix(vec3(1.), hueD * 1.2, smoothstep(0.8, 3., r) * 0.45), mix(vec3(1.), hue, smoothstep(1.0, 2.2, r) * clamp(vR.w * 1.5, 0., 1.)), uLight) * k;
    starC = arms + vec3(core);
    starA = clamp(max(core, max(arms.r, max(arms.g, arms.b)) * mix(0.9, 1.1, uLight)), 0., 1.);
    ring = uLight * 0.12 * k * exp(-pow((r - 1.5) / 0.5, 2.));
  }
  vec3 ringC = mix(vec3(0.10, 0.20, 0.34), vec3(0.28, 0.36, 0.48), uLight);
  float starOp = mix(0.35, 1., uLight);
  if (dm < -1.8){
    if (starA < 0.004 && ring < 0.004) discard;
    o = vec4(starC + ringC * ring * (1. - starA), clamp(starA * starOp + ring * (1. - starA), 0., 1.));
    return;
  }

  float inside = smoothstep(-0.45, 0.35, dm);
  float shadow = smoothstep(-1.8, -0.2, dm) * (1. - inside) * mix(0.3, 0.12, uLight);
  vec2 tilt = (hash22(vec2(seed * 13.1, fi + 2.)) - 0.5) * 0.9;
  vec3 n = normalize(vec3(dm < 1.1 ? nS * 0.9 + tilt * 0.3 : tilt, 1.));
  vec3 bg = bgAt(sp - n.xy * 7.);
  // Crushed ice is mostly clear: it shows the page (bent), with white scatter on its fractured faces.
  // Only the odd thick, flawless chip is deep enough to look blue. Seen on a white page, some facets
  // turn light away by total internal reflection and show up as grey-blue darks; that, not brightness,
  // is what makes clear ice read against white.
  float fh = hash12(vec2(seed * 19.7, fi + 5.));
  float thick = step(hash12(vec2(seed * 3.3, 8.)), 0.13) * step(2.2, R);
  float fracture = clamp(vQ.w + step(0.72, fh) * 0.7, 0., 1.);
  vec3 clearD = bg * 0.9 + vec3(0.045, 0.06, 0.075);
  vec3 clearL = bg * vec3(0.84, 0.9, 0.96) + vec3(0.01, 0.015, 0.03);
  vec3 clear = mix(mix(clearD, clearL, uLight), bg * vec3(0.62, 0.82, 1.0) + mix(vec3(0.05, 0.14, 0.26), vec3(0.0, 0.03, 0.08), uLight), thick);
  vec3 frosted = mix(bg * 0.35 + vec3(0.5, 0.54, 0.58), vec3(0.93, 0.96, 1.0), uLight) * (0.88 + 0.16 * hash12(floor(q * 2.) + seed * 40.));
  vec3 col = mix(clear, frosted, fracture * (1. - thick * 0.7)) * (0.92 + 0.16 * hash12(vec2(seed * 5.3, fi)));
  float tir = mix(step(fh, 0.14), step(fh, 0.5) * winkOf(n, sp, uLamp, fh * 3.7), step(0.14, fh)) * uLight * (1. - fracture * 0.6);
  col = mix(col, vec3(0.44, 0.53, 0.64) * (0.9 + 0.2 * hash12(vec2(seed, fi * 3.1))), tir * 0.95);
  // edges: a bright bevel on the lamp side, a crisp cool edge on the far side
  vec2 toL = normalize(uLamp - vP.xy + 1e-3);
  float lit = dot(nS, toL), rimZ = smoothstep(1.3, 0.3, dm);
  col += mix(vec3(0.8, 0.88, 0.95), vec3(1.), uLight) * rimZ * (0.12 + 0.45 * max(lit, 0.)) * (1. - uLight * 0.5);
  col = mix(col, mix(vec3(0.42, 0.5, 0.6), vec3(0.34, 0.45, 0.58), uLight), rimZ * max(-lit, 0.) * mix(0.35, 0.6, uLight));
  col += vec3(0.85, 0.92, 1.) * smoothstep(0.55, 0.1, ridge) * mix(0.22, 0.1, uLight) * step(1.2, R);
  col *= 1. - smoothstep(0.55, 0., dm) * mix(0.35, 0.25, uLight);
  // a facet lined up with the lamp: white on a dark page; on a pale one white would vanish, so the flash
  // splits into saturated spectral colour, the "fire" of a cut stone
  // the lit facet disperses: its colour runs across it in bands, and its brightest point stays white
  float band = dot(q, vec2(cos(a0 + fi), sin(a0 + fi))) / max(R, 1.) * 0.8 + seed * 3. + fi * 0.37;
  vec3 fire = clamp(0.5 + 0.8 * cos(6.2831 * (band + vec3(0., 0.33, 0.67))), 0., 1.);
  float gg = clamp(g * vP.w, 0., 1.5);
  col += vec3(1.) * gg * 1.3 * (1. - uLight);
  col = mix(col, mix(vec3(1.), fire, 0.3 + 0.4 * vR.w), smoothstep(0.12, 0.7, gg) * uLight);
  col = mix(col, vec3(1.), clamp(gg - 0.6, 0., 1.) * uLight * smoothstep(R * 0.5, 0., length(q - sq)));
  col += vec3(0.85, 0.95, 1.) * glintOf(n, sp, uLamp, 24.) * 0.08 * (1. - uLight);
  col = mix(col, ringC, ring);
  float a = inside * vP.w, sh = shadow * vP.w;
  vec3 body = col * a + mix(vec3(0.02, 0.07, 0.16), vec3(0.20, 0.40, 0.65), uLight) * sh;
  float ba = clamp(a + sh, 0., 1.);
  float ra = ring * (1. - inside);
  body = body * (1. - ra) + ringC * ra; ba += ra * (1. - ba);
  o = vec4(starC + body * (1. - starA), clamp(starA * starOp + ba * (1. - starA), 0., 1.));
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

// Offline-rendered sounds. A noise-excited two-pole resonator is the building block: a struck grain of
// ice rings at its own pitch for a few ms (a crunch grain) or tens of ms (a chip tinkle).
function resonate(d, sr, at, f, decay, amp, excMs = 0.4){
  const start = Math.round(at * sr); if (start >= d.length || f >= sr * 0.45) return;
  const w = TAU * f / sr, r = Math.exp(-1 / (decay * sr)), a1 = 2 * r * Math.cos(w), a2 = -r * r;
  const n = Math.min(d.length - start, Math.ceil(decay * sr * 7)), exc = Math.max(1, Math.round(excMs * sr / 1000));
  const tmp = new Float32Array(n);
  let y1 = 0, y2 = 0, peak = 1e-9;
  for (let i = 0; i < n; i++){
    const x = i < exc ? (Math.random() * 2 - 1) * (1 - i / exc) : 0;
    const y = x + a1 * y1 + a2 * y2; y2 = y1; y1 = y;
    tmp[i] = y; const m = Math.abs(y); if (m > peak) peak = m;
  }
  const k = amp / peak;
  for (let i = 0; i < n; i++) d[start + i] += tmp[i] * k;
}
function playBuf(a, fill, dur, gain, delay = 0){
  const sr = a.sampleRate, buf = a.createBuffer(1, Math.round(sr * dur), sr), d = buf.getChannelData(0);
  fill(d, sr);
  const src = a.createBufferSource(); src.buffer = buf;
  const g = a.createGain(); g.gain.value = gain;
  src.connect(g).connect(a.destination); src.start(a.currentTime + delay);
}
// A chip ringing. A lone sine reads as an electronic beep, so a small noise burst excites a few
// inharmonic modes at random ratios that die away within ~60-120ms (t60: time to fall 60dB).
function chipRing(d, sr, at, f, amp, t60 = rand(0.06, 0.12)){
  const ratios = [1, rand(1.3, 1.75), rand(2.05, 2.8), rand(3.1, 4.3)], amps = [1, 0.65, 0.4, 0.25];
  for (let k = 0; k < 4; k++) resonate(d, sr, at, f * ratios[k], t60 / 6.9 * (1 - k * 0.18), amp * amps[k] * rand(0.7, 1.1), rand(1.2, 3));
}
function ring(freq, gain, dur, delay = 0){
  const a = audio(); if (!a) return;
  playBuf(a, (d, sr) => chipRing(d, sr, 0, freq, 1, Math.max(0.06, Math.min(0.12, dur))), 0.14, gain, delay);
}
// One fracture event in crushed ice: a 1-3ms burst of noise, shaped by a fast attack and a decaying
// tail, and coloured by a one-pole low-pass and high-pass so grains vary from dull to bright. It is
// broadband: no resonance, so hundreds of them read as a crunch rather than a cloud of pitched tinks.
function noiseGrain(d, sr, at, ms, amp, lp, hp){
  const start = Math.round(at * sr), n = Math.min(d.length - start, Math.round(ms * sr / 1000));
  if (n <= 0) return;
  const al = 1 - Math.exp(-TAU * lp / sr), ah = Math.exp(-TAU * hp / sr);
  let l = 0, hIn = 0, h = 0;
  const att = Math.max(1, Math.round(sr * 0.0002));
  for (let i = 0; i < n; i++){
    const x = Math.random() * 2 - 1;
    l += (x - l) * al;
    h = ah * (h + l - hIn); hIn = l;
    const env = Math.min(1, i / att) * Math.exp(-4 * i / n);
    d[start + i] += h * env * amp;
  }
}
// Crushing a chunk: a body thud and knock under a dense granular crunch. Grains are single fracture
// events, arriving as a Poisson process whose rate collapses over ~200ms so the tail thins into audible
// gaps; a couple of freed chips ring well below the crunch.
function crunch(a, size){
  playBuf(a, (d, sr) => {
    const n = d.length;
    const f0 = rand(95, 160);
    let ph = 0;
    for (let i = 0; i < Math.min(n, sr * 0.2); i++){
      const t = i / sr, f = f0 * (0.65 + 0.35 * Math.exp(-t / 0.03));
      ph += TAU * f / sr;
      d[i] += Math.sin(ph) * Math.min(1, t / 0.0012) * Math.exp(-t / 0.032) * (0.5 + 0.2 * size);
    }
    resonate(d, sr, 0.001, rand(420, 780), 0.008, 0.4, 1.2);
    let t = 0.001;
    const r0 = 2600 + 1800 * size;
    while (t < 0.28){
      const rate = r0 * Math.exp(-t / 0.06) + 40;
      t += -Math.log(1 - Math.random()) / rate;
      if (t >= 0.28) break;
      // heavy-tailed loudness: a few big fractures stand out of the patter, so it crunches, not hisses
      const loud = Math.exp(-t / 0.085) * (0.12 + 1.5 * Math.pow(Math.random(), 3));
      const bright = Math.random();
      noiseGrain(d, sr, t, rand(1, 3), loud * 1.6, 2500 + 13000 * bright, 300 + 2500 * bright * Math.random());
    }
    for (let i = 0, m = 1 + Math.round(Math.random()); i < m; i++) chipRing(d, sr, rand(0.06, 0.22), rand(2600, 6000), rand(0.05, 0.09) * 0.3);
    let peak = 0;
    for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(d[i]));
    if (peak > 0) for (let i = 0; i < n; i++) d[i] *= 0.9 / peak;
  }, 0.5, 0.3 + 0.12 * size);
}

// The frozen-lake "pew". Flexural waves in a plate are dispersive (speed ~ sqrt(frequency)), so a hit
// arrives as a descending chirp: over a path of length D, arrival time ~ D/sqrt(f), i.e.
// f(t) = f0 / (1 + t/tau)^2 with tau growing in proportion to D. One strike gives one dominant arrival
// (the direct path) and a few echoes off the far shore: each has travelled further, so it arrives later
// on a roughly geometric schedule, sweeps more slowly (longer tau), and has lost more of its top end and
// level. A uniform train of identical chirps reads as a laser volley, not a hammer on a lake.
const PEW_RMS = 0.1;
function makeChirp(a){
  const sr = a.sampleRate, dur = 1.35;
  const n = Math.round(sr * dur), buf = a.createBuffer(1, n, sr), d = buf.getChannelData(0);
  const arr = [[0, 0.024 * rand(0.9, 1.1), 1, rand(7200, 8400)]];
  let D = 1;
  for (let i = 0, E = 3 + Math.floor(Math.random() * 4); i < E; i++){
    D *= rand(1.35, 1.9);
    const t0 = 0.055 * (D - 1) * rand(0.85, 1.15);
    if (t0 > 0.9) break;
    arr.push([t0, 0.024 * D * rand(0.85, 1.15), Math.pow(D, -0.85) * rand(0.7, 1), 8000 * Math.pow(D, -0.55) * rand(0.9, 1.05)]);
  }
  for (const [t0, tau, amp, f0] of arr){
    const start = Math.round(t0 * sr);
    let ph = Math.random() * TAU;
    for (let i = start; i < n; i++){
      const t = (i - start) / sr, f = f0 / ((1 + t / tau) * (1 + t / tau));
      if (f < 160) break;
      ph += TAU * f / sr;
      d[i] += Math.sin(ph) * amp * Math.min(1, t / 0.0015) * Math.pow(f / f0, 0.22) * Math.exp(-t * 2.6) * Math.min(1, (f - 160) / 120);
    }
  }
  // level by energy, not by peak, so every variant sounds equally loud
  let e2 = 0, peak = 0;
  for (let i = 0; i < n; i++){ e2 += d[i] * d[i]; peak = Math.max(peak, Math.abs(d[i])); }
  const rms = Math.sqrt(e2 / Math.round(sr * 0.8));
  const k = rms > 0 ? Math.min(PEW_RMS / rms, 0.95 / peak) : 0;
  for (let i = 0; i < n; i++) d[i] *= k;
  return buf;
}
let chirpSet = null;
function chirps(a){
  if (chirpSet && chirpSet.sr === a.sampleRate) return chirpSet;
  chirpSet = { sr: a.sampleRate, big: [makeChirp(a)] };
  // more variants off the click path, so the first smash doesn't pay for them
  const more = () => { if (chirpSet && chirpSet.big.length < 4){ chirpSet.big.push(makeChirp(a)); setTimeout(more, 250); } };
  setTimeout(more, 250);
  return chirpSet;
}
function pew(gain, rate, delay = 0){
  const a = audio(); if (!a) return;
  const set = chirps(a).big;
  const src = a.createBufferSource(); src.buffer = set[Math.floor(Math.random() * set.length)]; src.playbackRate.value = rate;
  const g = a.createGain(); g.gain.value = gain;
  src.connect(g).connect(a.destination); src.start(a.currentTime + delay);
}
const rand = (a, b) => a + Math.random() * (b - a);
const PEW_GAIN = 0.8;
let lastTink = 0, lastLoose = 0, lastGrit = 0;
const sounds = {
  press(){ burst('bandpass', 1300, 1.4, 0.05, 0.09); burst('highpass', 5200, 0.8, 0.02, 0.03, 0.02); },
  // the first break: the slab cracks through, and the whole sheet rings with the frozen-lake "pew"
  smash(){
    const a = audio(); if (!a) return;
    burst('lowpass', 700, 0.9, 0.22, 0.22); burst('bandpass', 2600, 0.7, 0.16, 0.3, 0.01);
    for (let i = 0; i < 7; i++) burst('highpass', rand(3000, 7000), 1, rand(0.02, 0.06), rand(0.01, 0.025), rand(0, 0.12) * Math.random());
    pew(PEW_GAIN, rand(0.94, 1.06), 0.02);
  },
  // a chunk crushed into the page
  crush(size = 1){ const a = audio(); if (a) crunch(a, size); },
  tink(v){
    const a = actx; if (!a || a.currentTime - lastTink < 0.04) return;
    lastTink = a.currentTime;
    ring(rand(3000, 6500), Math.min(0.09, 0.02 + v * 0.0006), rand(0.05, 0.14));
  },
  loose(v){
    const a = actx; if (!a || a.currentTime - lastLoose < 0.05) return;
    lastLoose = a.currentTime;
    burst('bandpass', rand(2200, 4800), 1.6, Math.min(0.08, 0.025 + v * 0.0001), rand(0.012, 0.03));
    ring(rand(3500, 7500), Math.min(0.08, 0.025 + v * 0.00015), rand(0.03, 0.09));
  },
  grit(lvl){
    const a = actx; if (!a || a.currentTime - lastGrit < 0.025) return;
    lastGrit = a.currentTime;
    burst('bandpass', rand(1800, 5200), rand(1.5, 3), 0.015 + 0.05 * lvl * Math.random(), rand(0.008, 0.02));
  },
  tap(){ burst('bandpass', rand(1600, 3000), 1.2, 0.07, 0.05); for (let i = 0; i < 3; i++) ring(rand(3500, 7000), 0.05, rand(0.04, 0.1), rand(0, 0.06)); },
  refreeze(){
    for (let i = 0; i < 26; i++) burst('highpass', rand(5000, 9000), 1.2, rand(0.008, 0.025), 0.012, Math.pow(Math.random(), 0.7) * 0.9);
    for (let i = 0; i < 6; i++) ping(1400 * Math.pow(1.19, i) * rand(0.99, 1.01), 0.012, 0.25, 0.12 + i * 0.12);
    ping(1760, 0.035, 0.9, 1.0); ping(2637, 0.016, 0.7, 1.02);
  },
};
let scrape = null, scrapeStop = 0;
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
  // a silent looping source still keeps the audio thread rendering, so it is torn down once it has faded
  clearTimeout(scrapeStop); scrapeStop = 0;
  if (v <= 0){
    const sc = scrape;
    scrapeStop = setTimeout(() => {
      scrapeStop = 0;
      if (scrape !== sc) return;
      try { sc.src.stop(); } catch { /* already stopped */ }
      sc.src.disconnect(); scrape = null;
    }, 350);
  }
}
// The shared context is suspended once every ice button on the page has gone to sleep (after anything
// still ringing has finished); audio() resumes it on the next sound.
const awakeFx = new Set();
let suspendTimer = 0;
function audioAwake(fx, on){
  if (on){ awakeFx.add(fx); clearTimeout(suspendTimer); suspendTimer = 0; return; }
  awakeFx.delete(fx);
  if (awakeFx.size || suspendTimer || !actx) return;
  suspendTimer = setTimeout(() => {
    suspendTimer = 0;
    if (awakeFx.size || !actx || actx.state !== 'running' || scrape) return;
    actx.suspend().catch(() => {});
  }, 2500);
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
// smooth 1D value noise in 0..1, for things that should wander along a stroke
const h1 = i => { const x = Math.sin(i * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };
const wob = t => { const i = Math.floor(t), f = t - i; return h1(i) + (h1(i + 1) - h1(i)) * ease(f); };
// Three top facets of a chip lying on the page: mostly near flat, tilted up to ~20 degrees, so across a
// pile many facets sit close to the lamp's mirror angle wherever the cursor is.
function facets(){
  const f = new Float32Array(9);
  for (let k = 0; k < 3; k++){
    const a = Math.random() * TAU, m = 0.36 * Math.sqrt(Math.random());
    const x = Math.cos(a) * m, y = Math.sin(a) * m, l = Math.hypot(x, y, 1);
    f[k * 3] = x / l; f[k * 3 + 1] = y / l; f[k * 3 + 2] = 1 / l;
  }
  return f;
}

// ---------------------------------------------------------------- component
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');

function attachIce(button, opts = {}){
  const canvas = document.createElement('canvas');
  canvas.setAttribute('aria-hidden', 'true');
  Object.assign(canvas.style, { position: 'absolute', pointerEvents: 'none', zIndex: opts.zIndex ?? 10, left: '0px', top: '0px' });
  const gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: true, antialias: true, depth: false, stencil: false });
  if (!gl) return { detach(){}, supported: false };
  const sound = opts.sound !== false;
  const self = {};
  const snd = (name, ...a) => { if (sound) sounds[name](...a); };
  const REFREEZE = opts.refreezeAfter ?? 5;
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
    ['aPos', 'aHome', 'aEdge', 'aInfo', 'aX', 'aY'].forEach((name, i) => gl.bindAttribLocation(p, i, name));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS) && !gl.isContextLost()) throw new Error(gl.getProgramInfoLog(p));
    const u = {};
    for (let i = 0, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS) || 0; i < n; i++){
      const info = gl.getActiveUniform(p, i);
      u[info.name.replace(/\[0\]$/, '')] = gl.getUniformLocation(p, info.name);
    }
    return { p, u };
  };
  let P, fullVao, shardBuf, shardVao, partBuf, partVao, chipBuf, chipVao, textTex, bgTex, snowTex;
  const pointVao = (prog, buf, names, stride) => {
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    let off = 0;
    for (const [name, size] of names){ const l = gl.getAttribLocation(prog.p, name); gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, size, gl.FLOAT, false, stride, off); off += size * 4; }
    return vao;
  };
  const mkTex = (filter) => {
    const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, filter], [gl.TEXTURE_MAG_FILTER, filter], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
    return t;
  };
  // everything GPU-side; run again after a lost context comes back
  function initGL(){
    P = { shadow: program(VS_SHARD, FS_SHADOW), ice: program(VS_SHARD, FS_ICE), snow: program(VS_FULL, FS_SNOW), part: program(VS_PART, FS_PART), chip: program(VS_CHIP, FS_CHIP) };
    fullVao = gl.createVertexArray();
    // chunk vertex stream: pos2 home2 edge4 info4 x4 y4 (stride 20 floats); both chunk programs share a layout
    shardBuf = gl.createBuffer(); shardVao = gl.createVertexArray();
    gl.bindVertexArray(shardVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, shardBuf);
    [2, 2, 4, 4, 4, 4].reduce((off, size, l) => { gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, size, gl.FLOAT, false, VSTRIDE * 4, off * 4); return off + size; }, 0);
    partBuf = gl.createBuffer(); partVao = pointVao(P.part, partBuf, [['aP', 4], ['aK', 2]], 24);
    chipBuf = gl.createBuffer(); chipVao = pointVao(P.chip, chipBuf, [['aP', 4], ['aQ', 4], ['aR', 4]], 48);
    gl.bindVertexArray(null);
    textTex = mkTex(gl.LINEAR); bgTex = mkTex(gl.LINEAR); snowTex = mkTex(gl.LINEAR);
  }
  initGL();
  const bindTex = (unit, t) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, t); };

  // ---- page
  const saved = button.getAttribute('style');
  const cs0 = getComputedStyle(button);
  const label = button.textContent.trim();
  const font = `${cs0.fontStyle} ${cs0.fontWeight} ${cs0.fontSize} ${cs0.fontFamily}`;
  const hideButton = () => Object.assign(button.style, { background: 'transparent', borderColor: 'transparent', color: 'transparent', boxShadow: 'none', textShadow: 'none', position: cs0.position === 'static' ? 'relative' : cs0.position });
  const showButton = () => { if (saved === null) button.removeAttribute('style'); else button.setAttribute('style', saved); };
  hideButton();
  document.body.appendChild(canvas);
  const bgCol = opts.background ? (parseColor(opts.background) || [1, 1, 1]).slice(0, 3) : pageColor(button.parentElement);
  const surface = opts.surface || null;
  // 0 on a dark page, 1 on a pale one: pale pages need dark label strokes and dark halos round the glints
  let lightK = 0;
  function uploadSurface(){
    let rgb = bgCol;
    if (surface && surface.image){
      gl.bindTexture(gl.TEXTURE_2D, bgTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, surface.image);
      try {
        const c = document.createElement('canvas'); c.width = c.height = 8;
        const x = c.getContext('2d'); x.drawImage(surface.image, 0, 0, 8, 8);
        const d = x.getImageData(0, 0, 8, 8).data; const m = [0, 0, 0];
        for (let i = 0; i < d.length; i += 4) for (let k = 0; k < 3; k++) m[k] += d[i + k] / 255 / 64;
        rgb = m;
      } catch { /* a cross-origin image can't be read back; the ancestor colour will do */ }
    }
    const lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    lightK = ease(Math.min(1, Math.max(0, (lum - 0.45) / 0.35)));
  }
  uploadSurface();

  // ---- geometry (CSS px); button-local "home" coords, stage = home + M
  let bw = 0, bh = 0, M = 0, stageW = 0, stageH = 0, dpr = 1, radius = 0, GW = 0, GH = 0;
  let slabPoly = [], planes = new Float32Array(16);
  let snow = new Float32Array(0), crushed = new Float32Array(0), slabMask = new Uint8Array(0), snowBytes = new Uint8Array(0), snowDirty = true;
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

  let placedAt = '';
  let eye = [0, 0];
  function place(r = button.getBoundingClientRect()){
    eye = [innerWidth / 2 - (r.left - M), innerHeight / 2 - (r.top - M)];
    const left = `${r.left + scrollX - M}px`, top = `${r.top + scrollY - M}px`;
    if (left + top !== placedAt){ placedAt = left + top; canvas.style.left = left; canvas.style.top = top; }
    if (surface){
      const s = surface.rect();
      bgOff = [r.left - M - s.left, r.top - M - s.top, s.width, s.height];
    }
    refreshBlockers(r);
    dirty = true;
  }

  // Other controls inside the stage (a "Cancel" link beside the button). Debris is kept off them: chips
  // stall and fade out at their edge, and no powder lies on them, so the stage never hides their text.
  // Found by hit-testing a coarse grid when the slab breaks; their rects are re-read whenever it moves.
  const INTERACTIVE = 'a,button,input,select,textarea,label,summary,[role=button],[tabindex],[contenteditable]';
  let blockEls = [], blocks = [], blockMask = new Uint8Array(0);
  function scanBlockers(){
    const r = button.getBoundingClientRect(), x0 = r.left - M, y0 = r.top - M, found = new Set();
    for (let y = 4; y < stageH; y += 10) for (let x = 4; x < stageW; x += 10){
      const px = x0 + x, py = y0 + y;
      if (px < 0 || py < 0 || px >= innerWidth || py >= innerHeight) continue;
      const el = document.elementFromPoint(px, py), hit = el && el.closest && el.closest(INTERACTIVE);
      if (hit && hit !== button && !button.contains(hit) && !hit.contains(button)) found.add(hit);
    }
    blockEls = [...found];
    refreshBlockers(r);
  }
  function refreshBlockers(r = button.getBoundingClientRect()){
    const x0 = r.left - M, y0 = r.top - M, PAD = 3;
    blocks = [];
    for (const el of blockEls){
      if (!el.isConnected) continue;
      const b = el.getBoundingClientRect();
      if (!b.width || !b.height) continue;
      const q = [b.left - x0 - PAD, b.top - y0 - PAD, b.right - x0 + PAD, b.bottom - y0 + PAD];
      if (q[2] > 0 && q[3] > 0 && q[0] < stageW && q[1] < stageH) blocks.push(q);
    }
    if (blockMask.length !== GW * GH) blockMask = new Uint8Array(GW * GH); else blockMask.fill(0);
    for (const [ax, ay, bx, by] of blocks){
      for (let j = Math.max(0, Math.floor(ay / CS)); j < Math.min(GH, Math.ceil(by / CS)); j++)
        for (let i = Math.max(0, Math.floor(ax / CS)); i < Math.min(GW, Math.ceil(bx / CS)); i++) blockMask[j * GW + i] = 1;
    }
  }
  // Distance (CSS px) from a stage point to the nearest place debris must not reach: the stage's own edge
  // or another control. Negative inside a control.
  const EDGE_SOFT = 24;
  function wallDist(x, y){
    let d = Math.min(x, y, stageW - x, stageH - y);
    for (const [ax, ay, bx, by] of blocks){
      const dx = Math.max(ax - x, x - bx), dy = Math.max(ay - y, y - by);
      d = Math.min(d, dx > 0 || dy > 0 ? Math.hypot(Math.max(dx, 0), Math.max(dy, 0)) : Math.max(dx, dy));
    }
    return d;
  }
  // Anything can move the button without resizing it or scrolling (content inserted above, a font
  // loading, an accordion opening), so the rect is compared rather than trusted: every awake frame,
  // on pointer activity near it, and after DOM changes while asleep.
  let lastRect = null;
  function checkPlace(){
    if (lost) return false;
    const r = button.getBoundingClientRect();
    const L = lastRect;
    if (L && L.left === r.left && L.top === r.top && L.width === r.width && L.height === r.height && L.sx === scrollX && L.sy === scrollY) return false;
    lastRect = { left: r.left, top: r.top, width: r.width, height: r.height, sx: scrollX, sy: scrollY };
    if (Math.abs(r.width - bw) > 0.5 || Math.abs(r.height - bh) > 0.5) resize();
    else { place(r); wake(0.1); }
    return true;
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
    snow = new Float32Array(GW * GH); crushed = new Float32Array(GW * GH); slabMask = new Uint8Array(GW * GH); snowBytes = new Uint8Array(GW * GH * 2);
    blockMask = new Uint8Array(GW * GH);
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
  let lost = false;
  const savedTouch = button.style.touchAction;
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
      delay: 0, seed: Math.random() * 0.9 + 0.05, world: null, gone: false, crushT: -1, crush: 0, fade: 1, strain: 0,
      hold: 0, bond: 8 + A * 0.014, free: false, pushT: -9, dust: 0, lim: Math.min(4.5, Math.max(3, 0.2 * Math.sqrt(A / Math.PI))) };
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
    button.style.touchAction = savedTouch;
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
    scanBlockers();
    button.style.touchAction = 'none';
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
    for (let k = 0; k < snow.length; k++) if (slabMask[k]){ snow[k] *= 0.1; crushed[k] *= 0.1; }
    snowDirty = true;
    wake(3);
  }

  // Step two: one chunk is crushed down into the page.
  function crushShard(s, px, py){
    if (s.crushT >= 0 || s.gone) return;
    s.crushT = 0; s.crushAt = [px, py]; s.crushHome = homeOf(s, px, py); s.vx = s.vy = s.va = 0; s.vz = 0; s.z = 0;
    idleT = 0;
    snd('crush', Math.min(1, s.area / 1800));
    wake(1);
  }
  // A pressed-flat layer of chips and powder over the chunk's footprint, squeezed outward from the fist:
  // most of it stays inside the outline, the rest shoots out along a few radial streaks, so what is left
  // is a splat rather than a stencil of the chunk. Nothing is thrown beyond the slab plus a small margin.
  function spawnChipsIn(s){
    const W = s.world, [cx, cy] = s.crushAt;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const [x, y] of W){ x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
    const A = s.area * s.scale * s.scale, far = Math.max(8, Math.sqrt(A));
    const n = Math.round(Math.min(280, Math.max(6, A / 26)));
    const streaks = Array.from({ length: 5 + Math.floor(Math.random() * 4) }, () => rand(0, TAU));
    const EDGE = 12;
    const inBounds = (x, y) => x > M - EDGE && y > M - EDGE && x < M + bw + EDGE && y < M + bh + EDGE && wallDist(x, y) > 6;
    // a point out along a streak: from the fist, past the chunk's outline by up to ~0.7 of its size
    const streakPt = () => {
      const a = streaks[Math.floor(Math.random() * streaks.length)] + rand(-0.09, 0.09);
      const ux = Math.cos(a), uy = Math.sin(a);
      let t = 0;
      while (t < far * 3 && pointInPoly(W, cx + ux * t, cy + uy * t)) t += 1.5;
      const d = t + far * 0.7 * Math.pow(Math.random(), 1.6);
      return [cx + ux * d, cy + uy * d, d];
    };
    let made = 0;
    for (let tries = 0; made < n && tries < n * 8; tries++){
      let x, y, streak = Math.random() < 0.28;
      if (streak){ [x, y] = streakPt(); if (!inBounds(x, y)) continue; }
      else { x = rand(x0, x1); y = rand(y0, y1); if (!pointInPoly(W, x, y)) continue; }
      const dx = x - cx, dy = y - cy, d = Math.hypot(dx, dy) || 1;
      const near = Math.exp(-d / (far * 0.5));
      const v = streak ? 0 : Math.sqrt(2 * MU * rand(0, 1) * (1 + 7 * near));
      const bondMax = rand(5, 16);
      const [hx, hy] = homeOf(s, Math.max(M, Math.min(M + bw, x)), Math.max(M, Math.min(M + bh, y)));
      chips.push({
        x, y, hx: hx + M, hy: hy + M, ang: rand(0, TAU), va: rand(-6, 6) * near,
        size: (Math.random() < 0.1 ? rand(4.5, 7) : 1.1 + 3.4 * Math.pow(Math.random(), 1.8) * (1 - 0.3 * near)) * (streak ? 0.7 : 1), seed: Math.random(),
        frost: Math.random() < 0.3 ? rand(0.55, 0.95) : rand(0, 0.2), fn: facets(), bl: 0,
        stuck: streak, bond: bondMax, bondMax, strain: 0, rest: 0,
        vx: dx / d * v * rand(0.8, 1.2), vy: dy / d * v * rand(0.8, 1.2),
        z: 0, vz: Math.random() < 0.1 * near ? rand(25, 70) : 0, jx: 0, jy: 0, a: 1,
      });
      made++;
    }
    for (let k = Math.round(A / 5); k > 0; k--){
      const x = rand(x0 - 3, x1 + 3), y = rand(y0 - 3, y1 + 3);
      const [qx, qy, dd] = closestOnPoly(W, x, y);
      if (!pointInPoly(W, x, y) && dd > 2.5 * Math.random()) continue;
      deposit(x, y, 0.16 + 0.34 * Math.exp(-Math.hypot(x - cx, y - cy) / (far * 0.5)) * Math.random(), rand(2, 3.5), true);
    }
    for (let k = Math.round(A / 14); k > 0; k--){
      const [x, y, d] = streakPt();
      if (!inBounds(x, y)) continue;
      deposit(x, y, 0.1 + 0.25 * Math.exp(-d / far), rand(1.2, 2.4), true);
    }
    for (let i = 0; i < 6 + n / 6; i++){
      const a = rand(0, TAU), sp = rand(15, 60);
      parts.push({ x: cx, y: cy, z: rand(0, 3), vx: Math.cos(a) * sp, vy: Math.sin(a) * sp, vz: rand(30, 80), size: rand(1.2, 2.4), kind: 1, age: 0, life: rand(0.5, 1), dep: 0.05 });
    }
    parts.push({ x: cx, y: cy, z: 3, vx: 0, vy: 0, vz: 10, size: rand(5, 7), kind: 2, age: 0, life: 0.3, dep: 0 });
    snowDirty = true;
  }

  function loosen(c, vx, vy){
    const v = Math.hypot(vx, vy);
    if (v > SKID_MAX){ vx *= SKID_MAX / v; vy *= SKID_MAX / v; }
    c.stuck = false; c.vx = vx; c.vy = vy; c.va = rand(-9, 9); c.rest = 0; c.strain = 0; c.jx = c.jy = 0;
    c.trav = 0; c.skid0 = Math.min(v, SKID_MAX);
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
    // a knuckle on bare page or powder still chips a few flecks off the frozen layer
    if (state === 'smashed' && hit < 3){
      for (let i = 3 + Math.floor(Math.random() * 3); i > 0; i--){
        const a = rand(0, TAU), sp = rand(40, 120);
        chips.push({ x: x + Math.cos(a) * 2, y: y + Math.sin(a) * 2, hx: x, hy: y, ang: rand(0, TAU), va: rand(-9, 9), size: rand(1.4, 3.2), seed: Math.random(),
          frost: rand(0.4, 1), fn: facets(), bl: 0, stuck: false, bond: 3, bondMax: 8, strain: 0, rest: 0,
          vx: Math.cos(a) * sp, vy: Math.sin(a) * sp, z: 0, vz: rand(15, 45), jx: 0, jy: 0, a: 1 });
      }
      deposit(x, y, 0.25, 3, true);
    }
    for (let i = 0; i < 8; i++){
      const a = rand(0, TAU), sp = rand(10, 45);
      parts.push({ x, y, z: 1, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp, vz: rand(20, 60), size: rand(1, 2.2), kind: 1, age: 0, life: 1, dep: 0.04 });
    }
    snd('tap');
    wake(1);
  }

  // ---- powder grid
  // ice: true for crushed ice, false for snowfall; the grid tracks how much of each cell is which
  function deposit(x, y, amount, r, ice = false){
    const i0 = Math.max(0, Math.floor((x - r) / CS)), i1 = Math.min(GW - 1, Math.ceil((x + r) / CS));
    const j0 = Math.max(0, Math.floor((y - r) / CS)), j1 = Math.min(GH - 1, Math.ceil((y + r) / CS));
    let wsum = 0;
    const ws = [];
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++){
      const d = Math.hypot((i + 0.5) * CS - x, (j + 0.5) * CS - y) / r;
      if (d < 1 && !blockMask[j * GW + i]){ const w = (1 - d * d); ws.push(j * GW + i, w); wsum += w; }
    }
    if (!wsum) return;
    for (let k = 0; k < ws.length; k += 2){ const v = amount * ws[k + 1] / wsum * (CS * CS); snow[ws[k]] += v; if (ice) crushed[ws[k]] += v; }
    snowDirty = true;
  }
  // A finger shoving powder cuts a groove: what lies under it is pushed out of its way, mostly off to the
  // sides and some ahead, and dropped in clumps just past its edge. How much it takes and how wide it
  // bites wobble along the stroke, so the banks come out ragged rather than as ruled lines.
  let plowS = 0;
  function plow(x, y, dx, dy, r, take = 0.85){
    const L = Math.hypot(dx, dy); if (L < 0.01) return 0;
    plowS += L;
    const R = r * (0.75 + 0.5 * wob(plowS * 0.15)), tk = take * (0.55 + 0.6 * wob(plowS * 0.21 + 9));
    const i0 = Math.max(0, Math.floor((x - R) / CS)), i1 = Math.min(GW - 1, Math.ceil((x + R) / CS));
    const j0 = Math.max(0, Math.floor((y - R) / CS)), j1 = Math.min(GH - 1, Math.ceil((y + R) / CS));
    const moves = [];
    let moved = 0;
    const ux = dx / L, uy = dy / L;
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++){
      const k = j * GW + i, h = snow[k];
      if (h < 0.005) continue;
      const cx = (i + 0.5) * CS - x, cy = (j + 0.5) * CS - y, cl = Math.hypot(cx, cy);
      if (cl > R) continue;
      const along = (cx * ux + cy * uy) / (cl || 1);
      if (along < -0.3) continue;
      // out along the radius, bent forward a little; nearer the centre line, more of it goes ahead
      let ox = cx / (cl || 1) + ux * 0.6, oy = cy / (cl || 1) + uy * 0.6;
      const ol = Math.hypot(ox, oy) || 1; ox /= ol; oy /= ol;
      const dist = R + CS * rand(0.2, 2.2);
      const tx = x + ox * dist + rand(-1.5, 1.5), ty = y + oy * dist + rand(-1.5, 1.5);
      const ti = Math.floor(tx / CS), tj = Math.floor(ty / CS);
      const t = h * tk, ti2 = crushed[k] * tk;
      snow[k] -= t; crushed[k] -= ti2; moved += t;
      if (ti >= 0 && tj >= 0 && ti < GW && tj < GH && !blockMask[tj * GW + ti]) moves.push(tj * GW + ti, t, ti2);
    }
    for (let m = 0; m < moves.length; m += 3){ snow[moves[m]] += moves[m + 1]; crushed[moves[m]] += moves[m + 2]; }
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
          if (d > 0){ const m = d * 0.2, mi = crushed[k] * m / snow[k]; snow[k] -= m; snow[n] += m; crushed[k] -= mi; crushed[n] += mi; }
        }
      }
    }
    // at rest, snow that misses the slab melts into the page and the slab holds only a dusting;
    // crushed powder stays frozen where it lies until the regrowth draws it back in
    const rest = state === 'rest';
    const melt = Math.exp(-dt * (rest ? 1.6 : state === 'regrowing' ? 5 : 0.02));
    for (let k = 0; k < snow.length; k++){
      if (snow[k] <= 0) continue;
      if (blockMask[k]){ snow[k] *= 0.8; crushed[k] *= 0.8; }
      else if (!rest || !slabMask[k]){ snow[k] *= melt; crushed[k] *= melt; }
      else if (snow[k] > 1){ const f = 1 + (1 / snow[k] - 1) * Math.min(1, dt * 2); snow[k] *= f; crushed[k] *= f; }
      if (snow[k] < 0.003) snow[k] = crushed[k] = 0;
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
  function nearShard(x, y, r){
    for (const s of shards){ if (!s.gone && s.crushT < 0 && s.world && closestOnPoly(s.world, x, y)[2] < r) return true; }
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
  // Would a press here land on the broken ice (a chunk, a chip, or packed powder)? Another control
  // underneath always wins, so a chunk or chips lying over a link never swallow its click.
  function iceHit(x, y, target){
    if (state === 'rest' || lost) return null;
    const onBtn = target === button || button.contains(target);
    if (!onBtn && target.closest && target.closest('a,button,input,select,textarea,label,summary,[role=button],[tabindex],[contenteditable]')) return null;
    const hit = onShard(x, y);
    if (hit) return { hit };
    // a press that lands just off the ice still grabs it: fingers and fast cursors miss by a few px
    if (onBtn || nearChip(x, y, GRAB) || nearShard(x, y, GRAB)) return { hit: null };
    for (let j = -GRAB; j <= GRAB; j += GRAB / 2) for (let i = -GRAB; i <= GRAB; i += GRAB / 2){
      const px = x + i, py = y + j;
      if (px < 0 || py < 0 || px >= stageW || py >= stageH) continue;
      if (snow[Math.floor(py / CS) * GW + Math.floor(px / CS)] > 0.12) return { hit: null };
    }
    return null;
  }
  function onDocDown(e){
    if (e.button !== 0 || ptr || lost) return;
    checkPlace();
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
    const ih = iceHit(x, y, e.target);
    if (!ih) return;
    e.preventDefault();
    ptr = { id: e.pointerId, x0: x, y0: y, x, y, lx: x, ly: y, vx: 0, vy: 0, speed: 0, dragging: false, onBtn, captured: true, hit: ih.hit };
    idleT = 0;
    wake(1);
  }
  function onDocMove(e){
    if (lost) return;
    onActivity();
    checkPlace();
    const [x, y] = stageXY(e);
    const near = x > -40 && y > -40 && x < stageW + 40 && y < stageH + 40;
    if (near || lamp){ lamp = [x, y]; if (near) wake(0.25); }
    // playing the cursor-light over the debris counts as using it: hold off the refreeze
    if (state === 'smashed' && x >= 0 && y >= 0 && x < stageW && y < stageH) idleT = 0;
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
    if (!p.dragging){
      const s = state === 'smashed' ? onShard(p.x, p.y) : null;
      if (s) crushShard(s, p.x, p.y); else tapAt(p.x, p.y);
    }
    wake(1);
  }
  function onDocClick(e){
    if (e.detail > 0 && performance.now() < swallowUntil){ swallowUntil = 0; e.preventDefault(); e.stopImmediatePropagation(); }
  }
  // A finger on the broken ice scrapes it rather than panning the page; touches that start anywhere
  // else scroll as usual.
  // Which finger is on the ice is settled at touchstart, and every move of that touch is claimed from
  // then on: waiting for the pointer to be captured leaves the first move free to start a pan.
  let touchOwned = false;
  function onTouch(e){
    if (e.type === 'touchend' || e.type === 'touchcancel'){ if (!e.touches.length) touchOwned = false; return; }
    if (!e.cancelable || lost) return;
    if (e.type === 'touchmove'){ if (touchOwned || (ptr && ptr.captured)) e.preventDefault(); return; }
    if (state === 'rest') return;
    const t = e.touches[0]; if (!t) return;
    const r = canvas.getBoundingClientRect();
    if ((ptr && ptr.captured) || iceHit(t.clientX - r.left, t.clientY - r.top, e.target)){ touchOwned = true; e.preventDefault(); }
  }
  function onClick(e){
    if (lost) return;
    if (state !== 'rest'){
      // a real pointer on the broken ice is working the ice, not pressing; keyboard activation still works,
      // and it crushes the chunk nearest the middle so the press is felt
      if (e.detail > 0){ e.preventDefault(); e.stopImmediatePropagation(); return; }
      if (state === 'smashed' && !reduceMotion.matches){
        let best = null, bd = Infinity;
        for (const s of shards){
          if (s.gone || s.crushT >= 0) continue;
          const c = centroid(s.world), d = Math.hypot(c[0] - M - bw / 2, c[1] - M - bh / 2);
          if (d < bd){ bd = d; best = s; }
        }
        idleT = 0;
        if (best){ const c = centroid(best.world); crushShard(best, c[0], c[1]); } else { tapAt(M + bw / 2, M + bh / 2); wake(1); }
      }
      return;
    }
    audio();
    const at = e.detail > 0 && lastDown ? lastDown : [bw / 2, bh / 2];
    lastDown = null;
    if (reduceMotion.matches){ pressTarget = 0; return; }
    smashAt(Math.max(2, Math.min(bw - 2, at[0])), Math.max(2, Math.min(bh - 2, at[1])));
  }
  const onCancel = e => { if (ptr && e.pointerId === ptr.id){ ptr = null; pressTarget = 0; scrapeLevel(0); } };
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
  addEventListener('keydown', onActivity, true);
  addEventListener('wheel', onActivity, { passive: true });
  button.addEventListener('click', onClick, true);

  // ---- step
  let simT = 0, activeUntil = 0;
  let raf = 0, inFrame = false, last = performance.now();
  // one animation loop at most: a wake from inside a frame must not start a second loop
  function wake(sec){ if (lost) return; activeUntil = Math.max(activeUntil, simT + sec); if (!raf && !inFrame){ raf = requestAnimationFrame(frame); audioAwake(self, true); } }

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
          let mx, my;
          if (inside){ mx = sdx; my = sdy; }
          else { const nx = (qx - px) / (dist || 1), ny = (qy - py) / (dist || 1), pen = PUSHER_R - dist; mx = nx * pen * 0.6 + sdx * 0.4; my = ny * pen * 0.6 + sdy * 0.4; }
          const push = Math.hypot(mx, my);
          s.pushT = simT;
          // Static friction: a chunk is frozen to the page and takes the shove as strain until its bond
          // (bigger chunks, stronger bond) lets go. Free, it still only creeps: a fraction of the finger's
          // travel, never more than a pixel or so per sub-step, and never far from where it broke.
          if (!s.free){
            s.hold += push;
            shove += push * 0.3;
            if (s.hold < s.bond) continue;
            s.free = true;
            snd('tink', 60 + s.rad * 3);
            deposit(qx, qy, 0.12, 2, true);
          }
          const A = s.area * s.scale * s.scale;
          let k = 0.22 * Math.min(1, Math.pow(300 / Math.max(A, 1), 0.35));
          let ex = mx * k, ey = my * k;
          const el = Math.hypot(ex, ey);
          if (el > 1.2){ ex *= 1.2 / el; ey *= 1.2 / el; }
          s.off[0] += ex; s.off[1] += ey;
          const c = centroid(W);
          s.ang += ((qx - c[0]) * ey - (qy - c[1]) * ex) / Math.max(s.rad * s.rad * 4, 60);
          clampShard(s);
          s.world = worldOf(s);
          shove += Math.hypot(ex, ey) * 3;
          if (Math.random() < 0.1) deposit(qx, qy, 0.04, 1.5, true);
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
              const kick = rand(0.5, 1.0) * Math.max(0, Math.min(1, wallDist(c.x, c.y) / EDGE_SOFT));
              loosen(c, ptr.vx * kick + nx * rand(10, 40) * kick, ptr.vy * kick + ny * rand(10, 40) * kick);
              snd('loose', ptr.speed);
              deposit(c.x, c.y, 0.05 * c.size, 1.5 + c.size * 0.3, true);
            }
          } else {
            // A fingertip rides up over loose chips instead of bulldozing them: each is shoved a short,
            // random way (mostly off to the side) and then the finger passes over it.
            if (c.passT > simT) continue;
            if (!c.carry) c.carryMax = rand(3, 14);
            c.carry = (c.carry || 0) + sl;
            if (c.carry > c.carryMax){ c.passT = simT + 0.4; c.carry = 0; continue; }
            const give = Math.max(0, Math.min(1, wallDist(c.x, c.y) / EDGE_SOFT));
            c.x += nx * pen * 0.7 * give; c.y += ny * pen * 0.7 * give;
            const along = c.vx * ux + c.vy * uy;
            if (along < ptr.speed * 0.6){
              const kf = rand(0.35, 0.75) * give, ks = rand(20, 70) * give;
              c.vx = ptr.vx * kf + nx * ks; c.vy = ptr.vy * kf + ny * ks;
              const v = Math.hypot(c.vx, c.vy);
              if (v > SKID_MAX){ c.vx *= SKID_MAX / v; c.vy *= SKID_MAX / v; }
              c.trav = 0; c.skid0 = Math.hypot(c.vx, c.vy);
            }
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

  // a chunk stays near its spot in the slab: it can creep, never travel
  function clampShard(s){
    const l = Math.hypot(s.off[0], s.off[1]);
    if (l > s.lim){ s.off[0] *= s.lim / l; s.off[1] *= s.lim / l; }
    s.ang = Math.max(-0.14, Math.min(0.14, s.ang));
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
      // left alone, a shoved chunk refreezes where it is and creeps back toward its place in the slab
      const since = simT - s.pushT;
      if (since > 0.6){
        if (s.free){ s.free = false; s.bond = (8 + s.area * 0.014) * 0.6; }
        s.hold *= Math.exp(-dt * 2);
        if (s.z === 0){
          const kb = 1 - Math.exp(-dt * 0.9);
          s.off[0] -= s.off[0] * kb; s.off[1] -= s.off[1] * kb; s.ang -= s.ang * kb;
          if (Math.abs(s.off[0]) + Math.abs(s.off[1]) > 0.05) moving = true;
        }
      }
      clampShard(s);
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
          if (!m || m[2] < 1.0) continue;
          const [nx, ny, depth] = m;
          const ma = a.area * a.scale * a.scale, mb = b.area * b.scale * b.scale, wa = mb / (ma + mb), wb = ma / (ma + mb);
          const corr = (depth - 0.9) * 0.5;
          a.off[0] -= nx * corr * wa; a.off[1] -= ny * corr * wa;
          b.off[0] += nx * corr * wb; b.off[1] += ny * corr * wb;
          clampShard(a); clampShard(b);
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
        // Coulomb friction: a constant deceleration, so skids stop short instead of coasting. Near the
        // stage edge or another control the page gets "stickier", so chips stall before they reach it.
        const w = wallDist(c.x, c.y), near = Math.max(0, 1 - w / EDGE_SOFT);
        const mu = MU * (1 + 14 * near * near);
        ns = Math.max(0, sp - mu * dt * (c.z > 0 ? 0.1 : 1));
        if (near > 0) ns *= Math.exp(-dt * 25 * near);
        c.vx *= ns / sp; c.vy *= ns / sp;
        c.x += c.vx * dt; c.y += c.vy * dt;
        c.ang += c.va * dt; c.va *= Math.exp(-dt * 5);
        // the trail: grain-sized clumps scattered either side of the path, tapering in and out, and no
        // longer than ~1.5x the skid itself
        const step = ns * dt;
        c.trav = (c.trav || 0) + step;
        const total = (c.skid0 || sp) ** 2 / (2 * MU);
        if (c.z === 0 && ns > 20 && c.trav < total * 1.5 && Math.random() < 0.3){
          const left = ns * ns / (2 * mu), taper = Math.min(1, c.trav / 3) * Math.min(1, left / 4);
          const j = (Math.random() - 0.5) * (1 + c.size * 0.9), inv = 1 / (ns || 1);
          deposit(c.x - c.vy * inv * j, c.y + c.vx * inv * j, step * 0.002 * c.size * taper * rand(0.2, 2.2) / 0.3, rand(0.7, 1.4) + c.size * 0.12, true);
        }
        moving = true;
      }
      if (ns < 4){
        c.vx = c.vy = 0; c.rest += dt;
        // wet ice refreezes onto the page almost at once, weakly at first
        if (c.rest > 0.1){ c.stuck = true; c.bond = rand(1.5, 4); c.bondMax = rand(5, 12); c.strain = 0; }
        else moving = true;
      } else c.rest = 0;
      c.x = Math.max(2, Math.min(stageW - 2, c.x)); c.y = Math.max(2, Math.min(stageH - 2, c.y));
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
      if (phaseT > REGROW_T){ toRest(); for (let k = 0; k < snow.length; k++){ crushed[k] = 0; if (!slabMask[k]) snow[k] = 0; } snowDirty = true; }
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
      if (p.on){
        // a flake that landed on a loose chunk rides with it until the chunk is crushed or regrows
        const s = p.on;
        if (s.gone || s.crushT >= 0 || state !== 'smashed'){ p.dead = true; continue; }
        const c = Math.cos(s.ang), sn = Math.sin(s.ang), k = s.scale * (1 + s.z * 0.004);
        p.x = s.hc[0] + s.off[0] + M + (c * p.lx - sn * p.ly) * k;
        p.y = s.hc[1] + s.off[1] + M + (sn * p.lx + c * p.ly) * k - s.z * 0.6;
        continue;
      }
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
          const sh = p.kind === 0 && state === 'smashed' ? onShard(p.x, p.y) : null;
          if (sh){
            // settles on the chunk: the flake itself stays a moment, and the chunk's rime thickens
            sh.dust = Math.min(0.9, sh.dust + 0.03 + 0.012 * p.size);
            if (parts.length < 900){
              const c = Math.cos(-sh.ang), sn = Math.sin(-sh.ang), k = sh.scale * (1 + sh.z * 0.004);
              const lx = p.x - (sh.hc[0] + sh.off[0] + M), ly = p.y - (sh.hc[1] + sh.off[1] + M);
              parts.push({ x: p.x, y: p.y, z: 0, vx: 0, vy: 0, vz: 0, size: p.size * 0.8, kind: 0, age: 1, life: 1e9, dep: 0, on: sh, lx: (c * lx - sn * ly) / k, ly: (sn * lx + c * ly) / k });
            }
          } else if (p.kind === 0){
            if (onIce) deposit(p.x, p.y, p.dep * p.size * 0.3, 1.5 + p.size * 0.3);
            else if (inGrid) parts.push({ x: p.x, y: p.y, z: 0.01, vx: 0, vy: 0, vz: 0, size: p.size, kind: 0, age: 0, life: 0.6, dep: 0, melt: 1 });
          } else if (state === 'rest' || !onShard(p.x, p.y)) deposit(p.x, p.y, p.dep * 4, 3, true);
        }
      }
    }
    for (let i = parts.length - 1; i >= 0; i--) if (parts[i].dead) parts.splice(i, 1);
    for (const p of parts) if (p.melt){ p.vz = 0; p.z = 0.01; }
  }

  // ---- render
  let vbuf = new Float32Array(VSTRIDE * 3 * 256);
  function buildVerts(list){
    let n = 0;
    const need = list.reduce((a, s) => a + (s.gone ? 0 : s.local.length * 3), 0) * VSTRIDE;
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
        const ch = s.crushHome || s.hc;
        for (const [p, hx, hy, d] of [[c, hcx, hcy, h], [a, la[0] + hcx, la[1] + hcy, 0], [b, lb[0] + hcx, lb[1] + hcy, 0]]){
          B[n++] = p[0]; B[n++] = p[1] + lift; B[n++] = hx; B[n++] = hy;
          B[n++] = d; B[n++] = nx; B[n++] = ny; B[n++] = kind;
          B[n++] = mid; B[n++] = s.z; B[n++] = s.seed; B[n++] = -lift;
          B[n++] = s.crush; B[n++] = s.fade; B[n++] = s.dust || 0; B[n++] = 0;
          B[n++] = ch[0]; B[n++] = ch[1]; B[n++] = s.rad; B[n++] = Math.max(0, s.crushT);
        }
      }
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, shardBuf);
    gl.bufferData(gl.ARRAY_BUFFER, B.subarray(0, n), gl.DYNAMIC_DRAW);
    return n / VSTRIDE;
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
    gl.uniform1f(u.uCraze, craze); gl.uniform1f(u.uGrow, grow); gl.uniform1f(u.uGrowOn, growOn ? 1 : 0); gl.uniform1f(u.uReach, reach); gl.uniform1f(u.uLight, lightK);
    gl.uniform2f(u.uImpact, ...impact); gl.uniform2f(u.uLamp, ...L); gl.uniform2f(u.uEye, ...eye);
    gl.uniform4fv(u.uPlanes, planes); gl.uniform2fv(u.uSeeds, seeds);
    gl.drawArrays(gl.TRIANGLES, 0, nv);
  }
  function drawPowder(L){
    if (snowDirty){
      for (let k = 0; k < snow.length; k++){
        snowBytes[k * 2] = Math.min(255, snow[k] / SNOW_PX * 255 + 0.5);
        snowBytes[k * 2 + 1] = snow[k] > 0 ? Math.min(255, crushed[k] / snow[k] * 255 + 0.5) : 0;
      }
      gl.bindTexture(gl.TEXTURE_2D, snowTex);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG8, GW, GH, 0, gl.RG, gl.UNSIGNED_BYTE, snowBytes);
      snowDirty = false;
    }
    gl.useProgram(P.snow.p);
    bindTex(2, snowTex); gl.uniform1i(P.snow.u.uSnow, 2);
    gl.uniform2f(P.snow.u.uGridPx, GW * CS, GH * CS); gl.uniform2f(P.snow.u.uStageSz, stageW, stageH);
    gl.uniform1f(P.snow.u.uDpr, dpr); gl.uniform1f(P.snow.u.uStageH, stageH); gl.uniform1f(P.snow.u.uTime, simT); gl.uniform1f(P.snow.u.uLight, lightK);
    gl.uniform2f(P.snow.u.uLamp, ...L); gl.uniform2f(P.snow.u.uEye, ...eye);
    bindTex(0, bgTex); gl.uniform1i(P.snow.u.uBgTex, 0);
    gl.uniform1f(P.snow.u.uHasBg, surface ? 1 : 0); gl.uniform3f(P.snow.u.uBgCol, ...bgCol); gl.uniform4f(P.snow.u.uBg, ...bgOff);
    gl.bindVertexArray(fullVao); gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
  // Each chip facet's specular toward the eye from the cursor lamp (the shader's glintOf, per chip), and
  // the few strongest glints in the pile, which get to flare into stars. Blooms ease in and out so a
  // spark handing over to its neighbour doesn't pop.
  const top = [];
  function lightChips(L){
    top.length = 0;
    const tw = simT * 9;
    // on a pale page a glint shows as colour, which is subtler per facet, so more of them are let through
    const sharp = CHIP_SHARP * (1 - 0.5 * lightK), blooms = Math.round(BLOOMS + 4 * lightK);
    for (const c of chips){
      if (!c.g) c.g = new Float32Array(3);
      const lx = L[0] - c.x, ly = L[1] - c.y, ll = Math.hypot(lx, ly, LIGHT_H);
      const vx = eye[0] - c.x, vy = eye[1] - c.y, vl = Math.hypot(vx, vy, EYE_H);
      let hx = lx / ll + vx / vl, hy = ly / ll + vy / vl, hz = LIGHT_H / ll + EYE_H / vl;
      const hl = Math.hypot(hx, hy, hz); hx /= hl; hy /= hl; hz /= hl;
      const ca = Math.cos(c.ang), sa = Math.sin(c.ang), f = c.fn;
      const twk = (0.85 + 0.15 * Math.sin(tw + c.seed * 60)) * (c.z > 0 ? 0.5 : 1);
      let gm = 0;
      for (let k = 0; k < 3; k++){
        const nx = ca * f[k * 3] - sa * f[k * 3 + 1], ny = sa * f[k * 3] + ca * f[k * 3 + 1];
        const d = Math.max(0, nx * hx + ny * hy + f[k * 3 + 2] * hz);
        const g = Math.pow(d, sharp) * twk;
        c.g[k] = g; if (g > gm) gm = g;
      }
      c.gm = gm * c.a;
      // a spark already flaring keeps its place unless another is clearly brighter
      c.rank = c.gm * (c.bl > 0 ? 1.3 : 1);
      if (c.gm > 0.45){
        if (top.length < blooms) top.push(c);
        else {
          let lo = 0;
          for (let i = 1; i < blooms; i++) if (top[i].rank < top[lo].rank) lo = i;
          if (top[lo].rank < c.rank) top[lo] = c;
        }
      }
    }
    for (const c of chips){
      const want = top.includes(c) ? Math.min(1, (c.gm - 0.45) / 0.4 + 0.35) : 0;
      c.bl += (want - c.bl) * (want > c.bl ? 0.6 : 0.45);
      if (c.bl < 0.05) c.bl = 0;
    }
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
        if (cdata.length < chips.length * 12) cdata = new Float32Array(chips.length * 24);
        lightChips(L);
        let m = 0;
        // flaring chips go last, so their star arms lie over the neighbours instead of under them
        for (const flaring of [false, true]) for (const c of chips){
          if (c.a <= 0.001 || (c.bl > 0) !== flaring) continue;
          const shrink = state === 'regrowing' ? 0.4 + 0.6 * c.a : 1;
          const g = c.g;
          const wf = Math.max(0, Math.min(1, (wallDist(c.x, c.y) - 1) / 12));
          if (wf <= 0) continue;
          cdata.set([c.x + c.jx, c.y + c.jy, c.size * shrink, c.a * wf * wf * (3 - 2 * wf), c.ang, c.seed, c.z, c.frost, g[0], g[1], g[2], c.bl], m * 12); m++;
        }
        if (m){
          gl.bindBuffer(gl.ARRAY_BUFFER, chipBuf);
          gl.bufferData(gl.ARRAY_BUFFER, cdata.subarray(0, m * 12), gl.DYNAMIC_DRAW);
          const u = P.chip.u;
          gl.useProgram(P.chip.p);
          gl.uniform2f(u.uStage, stageW, stageH); gl.uniform1f(u.uDpr, dpr);
          bindTex(0, bgTex); gl.uniform1i(u.uBgTex, 0);
          gl.uniform1f(u.uHasBg, surface ? 1 : 0); gl.uniform3f(u.uBgCol, ...bgCol); gl.uniform4f(u.uBg, ...bgOff);
          gl.uniform2f(u.uLamp, ...L); gl.uniform2f(u.uEye, ...eye); gl.uniform1f(u.uTime, simT); gl.uniform1f(u.uLight, lightK);
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
        if (p.on) continue;
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
    checkPlace();
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
    if (manual || lost || gl.isContextLost()) return;
    inFrame = true;
    let busy;
    try { busy = advance(dt); } finally { inFrame = false; }
    if (busy) raf = requestAnimationFrame(frame);
    else { last = performance.now(); audioAwake(self, false); }
  }

  // ---- keep the stage on the button
  const ro = new ResizeObserver(() => checkPlace());
  ro.observe(button, { box: 'border-box' });
  // the page around it reflowing (anything above growing or shrinking) moves the button too
  ro.observe(document.documentElement);
  if (document.body) ro.observe(document.body);
  // Awake, every frame checks the rect anyway. Asleep, a page that mutates all the time (a ticker, an
  // SPA) must not force a layout per mutation, so the check is throttled; ResizeObserver, scroll, resize
  // and pointer movement near the button cover most moves sooner.
  let moTimer = 0;
  const mo = new MutationObserver(() => {
    if (moTimer || raf || inFrame) return;
    moTimer = setTimeout(() => { moTimer = 0; if (!raf) checkPlace(); }, 300);
  });
  mo.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['class', 'style', 'hidden', 'open'] });
  const onFonts = () => checkPlace();
  document.fonts?.addEventListener?.('loadingdone', onFonts);
  const io = new IntersectionObserver(es => { visible = es[es.length - 1].isIntersecting; if (visible){ checkPlace(); wake(0.2); } });
  io.observe(button);
  const onScrollResize = () => { checkPlace(); place(); onActivity(); };
  const onVis = () => { if (!document.hidden){ last = performance.now(); checkPlace(); wake(0.2); } };
  addEventListener('resize', onScrollResize); addEventListener('scroll', onScrollResize, { passive: true });
  document.addEventListener('visibilitychange', onVis);

  // A lost context shows the plain button until the GPU comes back; then the ice is rebuilt in place.
  function onLost(e){
    e.preventDefault();
    lost = true;
    cancelAnimationFrame(raf); raf = 0;
    ptr = null; pressTarget = 0; scrapeLevel(0); audioAwake(self, false);
    showButton();
    canvas.style.visibility = 'hidden';
  }
  function onRestored(){
    try { initGL(); } catch (err) { console.warn('ElementalFX ice: could not rebuild after context loss', err); return; }
    lost = false;
    hideButton();
    canvas.style.visibility = '';
    uploadSurface();
    lastRect = null; placedAt = '';
    resize();
  }
  canvas.addEventListener('webglcontextlost', onLost);
  canvas.addEventListener('webglcontextrestored', onRestored);

  resize();

  return {
    supported: true,
    get state(){ return state; },
    get _frames(){ return frameCount; },
    get _chips(){ return chips.length; },
    get _shards(){ return shards.filter(s => !s.gone).length; },
    _shardAt(i){ const s = shards.filter(s => !s.gone && s.crushT < 0).sort((a, b) => b.area - a.area)[i]; if (!s) return null; const c = centroid(s.world); const r = canvas.getBoundingClientRect(); return [c[0] + r.left, c[1] + r.top]; },
    get _stuck(){ return chips.filter(c => c.stuck).length; },
    get _glints(){ return { lit: chips.filter(c => c.gm > 0.3).length, blooms: chips.filter(c => c.bl > 0).length }; },
    get _snowing(){ return snowing(); },
    get _asleep(){ return !raf && !inFrame; },
    get _lost(){ return lost; },
    get _dust(){ return { dust: +shards.reduce((a, s) => a + (s.gone ? 0 : s.dust), 0).toFixed(2), riding: parts.filter(p => p.on).length }; },
    _offsets(){ return shards.filter(s => !s.gone && s.crushT < 0).map(s => Math.hypot(s.off[0], s.off[1])); },
    // the page repainted what lies under the ice (e.g. after a resize): re-read it
    refresh(){ if (lost) return; uploadSurface(); lastRect = null; checkPlace(); place(); wake(0.2); },
    detach(){
      cancelAnimationFrame(raf); raf = 0; manual = true; ro.disconnect(); io.disconnect(); mo.disconnect(); clearTimeout(moTimer);
      document.fonts?.removeEventListener?.('loadingdone', onFonts);
      document.removeEventListener('touchstart', onTouch, capNP); document.removeEventListener('touchmove', onTouch, capNP);
      document.removeEventListener('touchend', onTouch, true); document.removeEventListener('touchcancel', onTouch, true);
      canvas.removeEventListener('webglcontextlost', onLost); canvas.removeEventListener('webglcontextrestored', onRestored);
      removeEventListener('resize', onScrollResize); removeEventListener('scroll', onScrollResize);
      document.removeEventListener('visibilitychange', onVis);
      document.removeEventListener('pointerdown', onDocDown, true);
      document.removeEventListener('click', onDocClick, true);
      removeEventListener('pointermove', onDocMove); removeEventListener('pointerup', onDocUp, true); removeEventListener('pointercancel', onCancel, true);
      removeEventListener('keydown', onActivity, true); removeEventListener('wheel', onActivity);
      button.removeEventListener('click', onClick, true);
      scrapeLevel(0);
      audioAwake(self, false);
      showButton();
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
