// Glass button: turns an existing <button> into a pane of thick glass standing just off the page.
//
//   <script src="fx/glass.js"></script>
//   const fx = ElementalFX.attachGlass(document.querySelector('#go'), { tint: 'green' });   // fx.detach() restores it
//
// A classic script (not a module) so pages work straight from file:// too.
//
// The <button> stays the real control (focus, keyboard, form submission, click handlers); this only
// draws it, in a transparent canvas laid over the page around it. The cursor is a lamp: its highlight
// slides over the pane and along the polished edges. A click is the button's one click and breaks the
// pane: a spiderweb of radial cracks and rings runs out from the point of impact, and a beat later the
// pane lets go and its shards drop, tumbling and glinting, onto a ledge below, ringing as they land,
// while jagged teeth of glass stay in the frame. The ledge goes in the first free band under the button
// (clear of text and controls); any part of a shard that still lies over page content is drawn as a faint
// ghost. A shard on the ledge can be flicked (a tap) or pushed around (a drag); neither ever clicks
// anything. A moment after the first shard lands they rise and fly back into the frame, a glint runs
// along every seam and the cracks heal. While it lies broken, pointer clicks on the empty frame are
// swallowed; Enter or Space brings it back early. The label's ink is chosen for contrast against the
// tinted pane, and a disabled button shows as dull, frosted glass. If WebGL2 is unavailable, the button
// is left untouched.
//
// Options:
//   tint: 'clear' (default) | 'green' | 'smoked' | 'frosted' | 'stained' | 'auto' (glass that passes the
//         button's own background colour, keeping its text colour when that reads; else the page's
//         colour) | { color, density, frost, streaks }
//   color: CSS colour of 'stained' glass (default a deep cobalt).
//   surface: { image: CanvasImageSource, rect: () => DOMRect }   what the page looks like under the pane,
//            refracted through it (bent hard at the bevels, split into colour at the edges). Without it
//            the pane lets the real page show through, tinted, without the bending.
//   background: CSS colour to use instead of the detected page colour.
//   restoreAfter: seconds after the first shard lands before they go home (default 2; touching or
//                 hovering the pile holds them there).
//   margin: CSS px of stage beside the button (default max(90, 1.4 x height)).
//   zIndex: of the stage canvas (default 10).   sound: false to stay silent.
//
// fx.setTint(t, color?) changes the glass live; fx.reset() sends the pieces home now.

(() => {
'use strict';

const TAU = Math.PI * 2;
const rand = (a, b) => a + Math.random() * (b - a);
const clamp = (v, a, b) => v < a ? a : v > b ? b : v;
const ease = v => v * v * (3 - 2 * v);
const smoother = v => v * v * v * (v * (v * 6 - 15) + 10);
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');

// ---------------------------------------------------------------- tuning (CSS px, seconds)
const PZ = 0.46;              // the camera looks slightly down: a point 1px toward the viewer sits PZ px lower
const PAGE_D = 8;             // the pane stands this far in front of the page
const WALL_SOFT = 44;         // pieces are pushed back gently once they come this close to a stage side
const GRAV = 2300;
const H_STEP = 1 / 240, MAX_SUB = 10;
const MU = 0.45;
const LIGHT_H = 130;          // height of the cursor lamp over the pane
const EYE_H = 900;            // viewer distance
const CRACK_T = 0.1;          // the web runs out across the pane in this long
const BEAT = 0.34;            // cracked, still standing, before it lets go
const RISE_T = 0.8, RISE_SPREAD = 0.5, HEAL_T = 0.8;
const DRAG_START = 6, GRAB = 7;
const VSTRIDE = 18;
const MAX_GHOST = 24;

// ---------------------------------------------------------------- shaders
const COMMON = `#version 300 es
precision highp float;
float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx)*.1031); p3 += dot(p3, p3.yzx+33.33); return fract((p3.x+p3.y)*p3.z); }
float vnoise(vec2 p){
  vec2 i = floor(p), f = fract(p); vec2 u = f*f*(3.-2.*f);
  return mix(mix(hash12(i), hash12(i+vec2(1,0)), u.x), mix(hash12(i+vec2(0,1)), hash12(i+vec2(1,1)), u.x), u.y);
}
float fbm(vec2 p){ return 0.5*vnoise(p) + 0.3*vnoise(p*2.03+7.1) + 0.2*vnoise(p*4.1+3.7); }
vec3 qrot(vec4 q, vec3 v){ return v + 2.*cross(q.xyz, cross(q.xyz, v) + q.w*v); }
uniform vec2 uSize; uniform float uRadius;
float sdRR(vec2 h){ vec2 q = abs(h - uSize*.5) - (uSize*.5 - uRadius); return length(max(q, 0.)) + min(max(q.x, q.y), 0.) - uRadius; }
uniform vec2 uStage;
// nothing is drawn right up to the stage edge, so the canvas rectangle never shows
float stageFade(vec2 s){ return smoothstep(2., 26., min(min(s.x, s.y), min(uStage.x - s.x, uStage.y - s.y))); }
// page content (text lines, controls) that loose shards must not hide: the part of a shard over one
// turns to a faint ghost, the rest of it stays solid
uniform vec4 uGhost[${MAX_GHOST}]; uniform int uNGhost;
float ghostAt(vec2 s){
  float g = 0.;
  for (int i = 0; i < ${MAX_GHOST}; i++){
    if (i >= uNGhost) break;
    vec4 r = uGhost[i];
    vec2 q = max(r.xy - s, s - r.zw);
    g = max(g, smoothstep(2.5, -1.5, max(q.x, q.y)));
  }
  return g;
}
`;

const VS_SHARD = `#version 300 es
in vec2 aPos;      // stage CSS px
in vec2 aHome;     // pane-local CSS px of the unbroken pane
in vec4 aEdge;     // distance to this fan triangle's edge line, its outward normal (home frame), kind (0 rim, 1 crack)
in vec4 aQ;        // shard orientation
in vec4 aInfo;     // loose (0 docked .. 1 free), seed, alpha, fracture faces shown (0..1)
in vec2 aShadow;   // where this point's shadow falls, relative to it
uniform vec2 uStage; uniform float uShadow;
out vec2 vHome; out vec4 vEdge; out vec4 vQ; out vec4 vInfo; out vec2 vStage; out float vLift;
void main(){
  vec2 p = aPos;
  vLift = length(aShadow);
  if (uShadow > .5) p += aShadow;
  vStage = p; vHome = aHome; vEdge = aEdge; vQ = aQ; vInfo = aInfo;
  gl_Position = vec4(p.x / uStage.x * 2. - 1., 1. - p.y / uStage.y * 2., 0., 1.);
}`;

const FS_SHADOW = COMMON + `
in vec2 vHome; in vec4 vEdge; in vec4 vQ; in vec4 vInfo; in vec2 vStage; in float vLift;
uniform float uShade;      // how much light the glass holds back (smoked glass throws a real shadow)
uniform vec3 uCaustic;     // colour of the light the bevels focus into the shadow
out vec4 o;
void main(){
  float e = -sdRR(vHome);
  float soft = 2.5 + vLift * 0.25;
  float a = smoothstep(0., soft, e);
  if (vEdge.w > .5) a *= mix(1., smoothstep(0., soft * 0.6, vEdge.x), vInfo.w);
  // the bevels bend light inward, so the shadow of a glass edge has a bright seam just inside it
  float c = exp(-pow((e - 5.5 - vLift * 0.1) / 2.2, 2.)) * smoothstep(0., 2., e);
  float k = vInfo.z / (1. + vLift * 0.05) * mix(1., stageFade(vStage), vInfo.x) * (1. - 0.9 * ghostAt(vStage) * vInfo.x);
  o = vec4(uCaustic * c * 0.35 * k, a * uShade * k);
}`;

const FS_GLASS = COMMON + `
in vec2 vHome; in vec4 vEdge; in vec4 vQ; in vec4 vInfo; in vec2 vStage; in float vLift;
uniform sampler2D uBgTex; uniform sampler2D uText;
uniform float uHasBg; uniform vec3 uBgCol;
uniform vec4 uBg;            // stage -> surface: offset xy, size zw (CSS px)
uniform vec2 uLamp; uniform float uLampK;
uniform vec2 uEye; uniform float uLight;
uniform vec3 uTint; uniform float uDensity, uFrost, uStreaks;
uniform vec3 uEdgeCol;
uniform float uBevel, uPress; uniform vec2 uPressAt;
uniform vec3 uInk; uniform float uInkDark;   // label colour, chosen on the CPU for contrast with the pane
uniform float uDisabled;
out vec4 o;
vec3 bgAt(vec2 sp){
  if (uHasBg < .5) return uBgCol;
  return texture(uBgTex, (sp + uBg.xy) / uBg.zw).rgb;
}
void main(){
  vec2 h = vHome;
  float sd = sdRR(h), e = -sd;
  vec2 gR = vec2(sdRR(h + vec2(.5, 0.)) - sdRR(h - vec2(.5, 0.)), sdRR(h + vec2(0., .5)) - sdRR(h - vec2(0., .5)));
  gR /= max(length(gR), 1e-4);
  float x = clamp(1. - e / uBevel, 0., 1.);
  float loose = vInfo.x, faces = vInfo.w;
  bool cut = vEdge.w > .5;
  float xc = cut ? clamp(1. - vEdge.x / 2.6, 0., 1.) * faces : 0.;
  // drawn glass is never quite flat: a slow ripple bends what is seen through it. A piece back in the
  // frame takes the whole pane's ripple, so the mended pane is the unbroken one pixel for pixel.
  float sk = vInfo.y * loose * 7.;
  vec2 wav = vec2(vnoise(h * vec2(.03, .018) + sk), vnoise(h * vec2(.018, .03) + 5.3 + sk)) - .5;
  vec2 pt = (h - uPressAt) / max(uSize.y, 1.);
  vec2 press = pt * exp(-dot(pt, pt) * 1.5) * uPress * 0.1;
  vec3 nl = normalize(vec3(gR * x * x * 1.8 + vEdge.yz * xc * xc * 1.4 + wav * 0.05 + press, 1.));
  vec3 nW = qrot(vQ, nl), nF = qrot(vQ, vec3(0., 0., 1.));
  vec3 V = normalize(vec3((uEye - vStage) / ${EYE_H.toFixed(1)}, 1.) + vec3(0., -${PZ.toFixed(3)}, 0.));
  // glass looks the same from either side: shade the face turned toward the viewer
  if (dot(nF, V) < 0.){ nW = qrot(vQ, vec3(nl.xy, -nl.z)); nF = -nF; }
  float cosT = max(dot(nW, V), 0.);
  float F = 0.04 + 0.96 * pow(1. - cosT, 5.);

  // what is seen through it: shifted by a tilted plate, bent hard at the bevels, split into colour there
  vec2 off = nW.xy * (3. + 18. * x + 12. * xc) + (nF.xy - vec2(0., -${PZ.toFixed(3)} * 0.)) * 5. + wav * 3.;
  float disp = 0.04 + (x + xc) * 0.2;
  float path = 1. + x * 3.2 + xc * 3. + (1. / max(dot(nF, V), 0.25) - 1.);
  float streak = uStreaks * (fbm(h * vec2(.011, .06) + vInfo.y) - .5);
  vec3 T = pow(uTint, vec3(uDensity * path * (1. + streak * 1.1)));
  vec3 bg;
  if (uHasBg > .5){
    bg = vec3(bgAt(vStage - off * (1. + disp)).r, bgAt(vStage - off).g, bgAt(vStage - off * (1. - disp)).b);
    if (uFrost > 0.){
      vec3 acc = vec3(0.);
      float ang = hash12(floor(vStage * 2.)) * 6.2831;
      for (int i = 0; i < 8; i++){
        float a = ang + float(i) * 2.3999;
        acc += bgAt(vStage - off + vec2(cos(a), sin(a)) * (1.5 + float(i) * 0.8));
      }
      bg = mix(bg, acc / 8., uFrost);
    }
  } else bg = uBgCol;
  float lum = dot(bg, vec3(.2126, .7152, .0722));

  // S + K * page: what the glass adds, and how much of the real page still shows through. With a
  // surface image the page under it is known and refracted, so K is 0.
  vec3 S; float K;
  if (uHasBg > .5){ S = bg * T; K = 0.; }
  else { float m = min(T.r, min(T.g, T.b)); S = uBgCol * (T - m); K = m; }
  #define MIXTO(c, w) { float w_ = (w); S = mix(S, (c), w_); K *= 1. - w_; }

  // ground glass scatters: a milky veil, a little brighter than the page
  vec3 milk = mix(vec3(0.82, 0.86, 0.85), bg * T * 1.08 + 0.06, 0.55);
  MIXTO(milk, uFrost * (0.3 + 0.1 * vnoise(h * 1.7)) + uDisabled * 0.3);
  S = mix(S, vec3(dot(S, vec3(.2126, .7152, .0722))), uDisabled * 0.75);

  // looking into the thickness of the plate at the bevels: the long path through it goes deep green
  float band = smoothstep(0.3, 0.95, x) * 0.7 + smoothstep(0.2, 1., xc) * 0.6;
  MIXTO(uEdgeCol * (0.35 + 0.65 * lum) * (0.8 + 0.4 * wav.x), band);

  // reflections of a dim room with a bright window up and to the left: two diagonal streaks on the face
  vec3 R = reflect(-V, nW), R0 = reflect(-V, vec3(0., 0., 1.));
  // the streaks sit on the pane where it faces the viewer; a tilted shard sees the window wherever its
  // own reflection points, so only some of them catch it
  float face = smoothstep(0.93, 0.995, nF.z);
  float u = face * dot(h - uSize * vec2(0.28, 0.5), vec2(0.82, 0.57)) / max(uSize.y, 24.) + dot((R - R0).xy, vec2(0.82, 0.57)) * 2.4
          + (1. - face) * (vInfo.y - 0.5) * 3. + (uEye.x - vStage.x) * 0.0004;
  float win = smoothstep(0.2, 0.12, abs(u)) + 0.75 * smoothstep(0.055, 0.02, abs(u - 0.38));
  float room = mix(0.1, 0.55, smoothstep(-0.4, 0.6, -R.y));
  vec3 env = vec3(room) + vec3(1., 1., 0.97) * win * 5.5 * smoothstep(-0.9, 0.1, -R.y + 0.3);
  S = S * (1. - F) + env * F; K *= 1. - F;

  // the label, etched into the face: grainy strokes with a lit and a shaded wall. A soft halo round the
  // strokes pushes the glass away from the ink, so the words hold up over a busy or mid-tone backdrop.
  vec2 tuv = h / uSize;
  vec4 tx = texture(uText, tuv);
  float t0 = tx.r, tB = tx.g;
  float tS = texture(uText, (h + vec2(0.6, 0.8)) / uSize).r;
  float tL = texture(uText, (h - vec2(0.6, 0.8)) / uSize).r;
  vec3 haloC = uInkDark > .5 ? mix(S, vec3(0.97, 1., 0.98), 0.5) : S * 0.45;
  MIXTO(haloC, clamp(tB * 1.6 - t0, 0., 1.) * 0.45);
  MIXTO(uInk * (0.94 + 0.06 * hash12(floor(h * 2.5))), t0 * (0.94 - uDisabled * 0.3));
  vec3 wallD = uInkDark > .5 ? uInk * 0.4 : vec3(0.02, 0.06, 0.05);
  MIXTO(wallD, clamp(tS - t0, 0., 1.) * 0.5);
  S += vec3(0.7, 0.85, 0.8) * clamp(tL - t0, 0., 1.) * (uInkDark > .5 ? 0.15 : 0.35);

  // light: the cursor lamp (its highlight follows the pointer across the face and along the edges)
  vec3 Ll = normalize(vec3(uLamp - vStage, ${LIGHT_H.toFixed(1)}));
  vec3 Hl = normalize(Ll + V);
  float nh = max(dot(nW, Hl), 0.);
  float spec = (pow(nh, 60.) * 0.35 + pow(nh, 500.) * 1.3) * uLampK;
  // a fixed sun that the flat pane never faces, so only tumbling shards and bevels flash in it
  vec3 Hs = normalize(normalize(vec3(-0.5, -0.8, 0.5)) + V);
  float sun = pow(max(dot(nW, Hs), 0.), 900.) * 3.;
  S += vec3(1.) * (spec + sun) * (1. - 0.7 * uDisabled);
  // grains in the etched lettering sparkle under the lamp
  S += vec3(1.) * t0 * step(0.94, hash12(floor(h * 1.8))) * pow(nh, 30.) * 0.6 * uLampK;

  // polished rim: a crisp Fresnel line, bright on the side facing the lamp, dark on the far side, with
  // a thin spectral fringe inside it where the bevel splits the light
  vec3 gW = qrot(vQ, vec3(gR, 0.));
  vec2 toL = normalize(uLamp - vStage + vec2(0., 1e-3));
  float lit = clamp(0.5 + 0.8 * dot(gW.xy, toL), 0., 1.);
  float line = smoothstep(1.4, 0.3, e);
  MIXTO(mix(uEdgeCol * 0.25, vec3(1.), lit), line * 0.9);
  float ridge = exp(-pow((e - uBevel) / 0.7, 2.));
  S += vec3(1.) * ridge * (0.12 + 0.35 * lit * uLampK);
  vec3 spectral = 0.5 + 0.5 * cos(6.2831 * (e * 0.3 + vec3(0., 0.33, 0.67)));
  S += (spectral - 0.5) * 0.22 * smoothstep(0.1, 0.5, x) * smoothstep(1., 0.6, x) * (0.4 + 0.6 * lit);

  // a loose shard's fractured faces: a bright conchoidal edge that catches everything
  if (cut && faces > 0.){
    float fl = smoothstep(1.3, 0.2, vEdge.x) * faces;
    float fsp = 0.45 + 0.55 * vnoise(h * 0.4 + vInfo.y * 20.);
    MIXTO(mix(vec3(0.75, 0.92, 0.86) * fsp, uEdgeCol * 0.4, uLight * 0.4), fl * 0.85);
    S += spectral * fl * 0.1;
  }

  // only loose pieces fade toward the stage edge: the pane in its frame is always drawn whole, even when
  // the button sits flush against the edge of the page
  float alpha = vInfo.z * mix(1., stageFade(vStage), loose) * (1. - 0.82 * ghostAt(vStage) * loose);
  o = vec4(clamp(S, 0., 1.5) * alpha, (1. - K) * alpha);
}`;

// Crack lines over the pieces still in the frame. Each segment carries its distance along the web from
// the point of impact, so the cracks can run out from it and heal back toward it.
const VS_CRACK = `#version 300 es
in vec4 aAB;        // segment ends, pane-local px
in vec2 aCorner;    // along (0|1), side (-1|1)
in vec4 aD;         // web distance at each end, seed, visibility
uniform vec2 uStage; uniform vec2 uOrigin;
out vec2 vP; out vec4 vAB; out vec4 vD;
void main(){
  vec2 a = aAB.xy, b = aAB.zw, d = b - a;
  float L = length(d);
  vec2 t = L > 1e-4 ? d / L : vec2(1., 0.), n = vec2(-t.y, t.x);
  float W = 6.;
  vec2 p = mix(a - t * W, b + t * W, aCorner.x) + n * aCorner.y * W;
  vP = p; vAB = aAB; vD = aD;
  vec2 s = p + uOrigin;
  gl_Position = vec4(s.x / uStage.x * 2. - 1., 1. - s.y / uStage.y * 2., 0., 1.);
}`;

const FS_CRACK = COMMON + `
in vec2 vP; in vec4 vAB; in vec4 vD;
uniform vec2 uOrigin; uniform vec2 uLamp; uniform float uLampK; uniform vec2 uEye;
uniform float uCrackR, uHealF, uHealOn, uAlpha, uLight;
out vec4 o;
void main(){
  vec2 a = vAB.xy, b = vAB.zw, ab = b - a;
  float L2 = dot(ab, ab);
  float t = L2 > 1e-6 ? clamp(dot(vP - a, ab) / L2, 0., 1.) : 0.;
  float dist = length(vP - (a + ab * t));
  float d = mix(vD.x, vD.y, t);
  float seed = vD.z, vis = vD.w;
  if (sdRR(vP) > -0.3) discard;
  float shown = smoothstep(-0.5, 2., uCrackR - d);
  float healed = uHealOn > .5 ? smoothstep(-1., 2., d - uHealF) : 0.;
  float glint = uHealOn > .5 ? exp(-pow((d - uHealF) / 3.5, 2.)) : 0.;
  float tip = exp(-pow((uCrackR - d) / 5., 2.)) * step(d, uCrackR + 4.) * step(uCrackR, 1e4);
  vec2 sp = vP + uOrigin;
  if (L2 < 1e-6){
    // the crushed spot under the blow: a white star of tiny fractures
    float r = dist;
    float star = smoothstep(4.5, 1., r + (vnoise(vec2(atan(vP.y - a.y, vP.x - a.x) * 3., seed * 9.)) - .5) * 3.);
    float ga = star * shown * (1. - healed) * vis * uAlpha;
    vec3 c = mix(vec3(0.95, 1., 0.98), vec3(0.55, 0.62, 0.6), uLight * 0.5);
    float g2 = glint * smoothstep(8., 0., r) * uAlpha;
    o = vec4(c * ga + vec3(1.) * g2 * 1.4, ga + g2 * 0.3);
    return;
  }
  // the crack face is a plane through the glass, turning a little along its length: where it lines up
  // with the lamp it flashes silver, elsewhere it shows dark
  vec2 tn = normalize(vec2(-ab.y, ab.x));
  float tw = (vnoise(vec2(d * 0.16, seed * 17.)) - .5) * 2.6;
  vec3 nC = normalize(vec3(tn * cos(tw), sin(tw)));
  vec3 V = normalize(vec3((uEye - sp) / 900., 1.));
  vec3 Ll = normalize(vec3(uLamp - sp, 130.));
  float sp1 = pow(abs(dot(nC, normalize(Ll + V))), 14.) * (0.35 + 0.65 * uLampK);
  float sp2 = pow(abs(dot(nC, normalize(normalize(vec3(-0.5, -0.8, 0.5)) + V))), 10.);
  float catchL = clamp(sp1 + sp2 * 0.6, 0., 1.);
  float w = 0.25 + 0.35 * vnoise(vec2(d * 0.35, seed * 5.)) * vnoise(vec2(d * 0.07, seed * 3.));
  float core = smoothstep(w + 0.7, w - 0.3, dist);
  float halo = exp(-dist * dist / 3.5) * 0.3;
  vec3 bright = vec3(0.92, 1., 0.97), dark = vec3(0.3, 0.38, 0.36);
  vec3 col = mix(bright * (0.5 + 0.7 * catchL), mix(dark, bright, catchL), uLight);
  float ca = (core * mix(0.9, 0.75, uLight) + halo * (1. - uLight * 0.6)) * shown * (1. - healed) * vis * uAlpha;
  float add = (tip * 1.3 + glint * 1.7) * exp(-dist * dist / 1.8) * vis * uAlpha;
  o = vec4(col * ca + vec3(1., 1., 0.96) * add, ca + add * 0.25);
}`;

const VS_PART = `#version 300 es
in vec4 aP;         // stage x, y, size, alpha
in vec2 aK;         // kind (0 dust, 1 star), random
uniform vec2 uStage; uniform float uDpr;
out float vA; out vec2 vK;
void main(){
  vA = aP.w; vK = aK;
  gl_PointSize = aP.z * uDpr;
  gl_Position = vec4(aP.x / uStage.x * 2. - 1., 1. - aP.y / uStage.y * 2., 0., 1.);
}`;
const FS_PART = `#version 300 es
precision highp float;
in float vA; in vec2 vK;
out vec4 o;
void main(){
  vec2 c = gl_PointCoord - 0.5;
  float r = length(c);
  if (r > 0.5) discard;
  float a = smoothstep(0.5, 0.15, r) * vA;
  if (vK.x > 0.5){
    float cr = max(smoothstep(0.06, 0., abs(c.x)), smoothstep(0.06, 0., abs(c.y))) * smoothstep(0.5, 0., r);
    a = max(smoothstep(0.2, 0., r) * vA, cr * vA);
  }
  o = vec4(vec3(1., 1., 0.97) * a, a * 0.6);
}`;

// ---------------------------------------------------------------- audio (shared)
// Glass rings: most of this is modal pings, exponentially decaying sinusoids at inharmonic ratios, the
// way a struck plate or a sliver of it sounds. The fracture itself is not a ring but a tear: a few tens of
// milliseconds of band-limited crunch at the moment the pane fails, under the first pings.
const glassAudio = { muted: false };
let actx = null;
function audio(){
  if (glassAudio.muted) return null;
  if (!actx){ try { actx = new (window.AudioContext || window.webkitAudioContext)(); } catch { return null; } }
  if (actx.state === 'suspended') actx.resume().catch(() => {});
  return actx;
}
const SR = 48000;
function modal(d, sr, at, f, tau, amp, att = 0.0004){
  const start = Math.round(at * sr);
  // partials above ~15kHz only add a harsh fizz on laptop speakers
  if (start >= d.length || f >= Math.min(sr * 0.46, 15000) || f <= 20) return;
  const n = Math.min(d.length - start, Math.ceil(tau * sr * 7));
  const w = TAU * f / sr, c = Math.cos(w), s = Math.sin(w), r = Math.exp(-1 / (tau * sr));
  const na = Math.max(1, Math.round(att * sr));
  const ph = Math.random() * TAU;
  let re = Math.cos(ph) * amp, im = Math.sin(ph) * amp;
  for (let i = 0; i < n; i++){
    d[start + i] += im * (i < na ? i / na : 1);
    const nr = (re * c - im * s) * r; im = (re * s + im * c) * r; re = nr;
  }
}
function click(d, sr, at, ms, amp){
  const start = Math.round(at * sr), n = Math.min(d.length - start, Math.round(ms * sr / 1000));
  let prev = 0;
  for (let i = 0; i < n; i++){
    const x = Math.random() * 2 - 1;
    d[start + i] += (x - prev) * 0.5 * amp * Math.exp(-5 * i / n);
    prev = x;
  }
}
// Band-passed noise with a fast exponential decay and a grainy, crackling texture (the noise is cubed,
// so it is mostly quiet with sharp spikes, like many tiny fractures rather than a hiss).
function crunch(d, sr, at, ms, amp, f1 = 2000, f2 = 8000){
  const start = Math.round(at * sr);
  if (start >= d.length) return;
  const n = Math.min(d.length - start, Math.round(ms * sr / 1000));
  const tau = ms / 1000 / 3.2;
  const f0 = Math.sqrt(f1 * f2), Q = f0 / (f2 - f1);
  const w0 = TAU * f0 / sr, al = Math.sin(w0) / (2 * Q), cw = Math.cos(w0), a0 = 1 + al;
  const b0 = al / a0, b2 = -al / a0, a1 = -2 * cw / a0, a2 = (1 - al) / a0;
  // two passes of the same band-pass, for steeper skirts
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0, u1 = 0, u2 = 0, z1 = 0, z2 = 0;
  const na = Math.max(1, Math.round(0.0006 * sr));
  for (let i = 0; i < n; i++){
    const r = Math.random() * 2 - 1, x = r * r * r * 2.2;
    const y = b0 * x + b2 * x2 - a1 * y1 - a2 * y2; x2 = x1; x1 = x; y2 = y1; y1 = y;
    const z = b0 * y + b2 * u2 - a1 * z1 - a2 * z2; u2 = u1; u1 = y; z2 = z1; z1 = z;
    const env = Math.exp(-i / (tau * sr)) * (i < na ? i / na : 1);
    d[start + i] += z * amp * env * 4;
  }
}
// One sliver of glass struck: a few plate modes, the higher ones dying faster. hiCap limits how long the
// partials above 4 kHz ring, so a cloud of slivers is a patter of ticks rather than sustained tones.
const SLIVER = [1, 1.51, 2.18, 2.74, 3.93, 4.61];
function sliver(d, sr, at, f0, amp, tauScale = 1, hiCap = Infinity){
  for (let k = 0; k < SLIVER.length; k++){
    const f = f0 * SLIVER[k] * rand(0.95, 1.05);
    let tau = clamp(0.2 * Math.pow(2500 / f, 0.75), 0.012, 0.35) * tauScale * rand(0.8, 1.2);
    if (f > 4000) tau = Math.min(tau, hiCap);
    modal(d, sr, at, f, tau, amp * [1, 0.7, 0.5, 0.36, 0.24, 0.16][k] * rand(0.7, 1.1));
  }
}
function normalize(d, peak = 0.9){
  let m = 0; for (let i = 0; i < d.length; i++) m = Math.max(m, Math.abs(d[i]));
  if (m > 0) for (let i = 0; i < d.length; i++) d[i] *= peak / m;
  return d;
}
const logRand = (a, b) => a * Math.pow(b / a, Math.random());
const SYN = {
  // the blow: a sharp snap and crunch, the pane's knock (high enough, with an overtone body, to survive
  // laptop speakers), and the web racing out as a patter of tiny ticks
  crack(sr = SR){
    const d = new Float32Array(Math.round(sr * 0.6));
    click(d, sr, 0, 1.2, 0.9);
    crunch(d, sr, 0, 14, 0.55, 1800, 7500);
    const f0 = rand(620, 860);
    [1, 1.59, 2, 2.14, 2.65, 3.6].forEach((r, k) => modal(d, sr, 0.0005, f0 * r * rand(0.98, 1.02), 0.045 / (1 + k * 0.35), [0.5, 0.26, 0.3, 0.18, 0.12, 0.08][k]));
    for (let i = 0; i < 18; i++){
      const t = 0.002 + Math.pow(Math.random(), 1.6) * 0.1;
      sliver(d, sr, t, logRand(3500, 8500), rand(0.05, 0.16) * (1 - t * 5), 0.25, 0.03);
    }
    for (let i = 0; i < 4; i++) crunch(d, sr, rand(0.01, 0.1), rand(3, 7), rand(0.08, 0.18), 3000, 9000);
    for (let i = 0; i < 3; i++) sliver(d, sr, rand(0.01, 0.12), logRand(2200, 4200), rand(0.06, 0.1), 0.7);
    return normalize(d, 0.9);
  },
  // the pane letting go: the tearing crash of the fracture, a few more crunches as pieces break off each
  // other, a body of mid pings, then slivers ticking as they break free and strike, thinning over a second
  shatter(sr = SR){
    const d = new Float32Array(Math.round(sr * 1.4));
    click(d, sr, 0, 2.5, 0.7);
    crunch(d, sr, 0, 34, 1.0, 2000, 8000);
    crunch(d, sr, 0.001, 22, 0.45, 700, 2600);
    for (let i = 0; i < 5; i++) crunch(d, sr, rand(0.015, 0.15), rand(8, 18), rand(0.25, 0.55) * (1 - i * 0.1), 2200, 9000);
    const f0 = rand(620, 820);
    [1, 1.62, 2.31, 3.05, 3.97].forEach((r, k) => modal(d, sr, 0.001, f0 * r * rand(0.97, 1.03), 0.06 / (1 + k * 0.3), 0.3 / (1 + k * 0.5)));
    for (let i = 0; i < 24; i++) sliver(d, sr, Math.pow(Math.random(), 2) * 0.09, logRand(900, 2600), rand(0.04, 0.1), 0.3, 0.04);
    let t = 0.002;
    for (let i = 0; i < 95; i++){
      t += -Math.log(1 - Math.random()) / (320 * Math.exp(-t / 0.18) + 18);
      if (t > 1.2) break;
      const loud = Math.exp(-t / 0.35) * (0.05 + 0.3 * Math.pow(Math.random(), 3));
      sliver(d, sr, t, logRand(2200, 9000), loud, rand(0.5, 1.1), 0.04);
      if (t < 0.4 && Math.random() < 0.25) crunch(d, sr, t, rand(2, 5), loud * 0.8, 3000, 9000);
    }
    return normalize(d, 0.92);
  },
  // a shard landing: pitch by size (bigger is lower)
  ping(sr = SR, f0 = 4000){
    const d = new Float32Array(Math.round(sr * 0.45));
    click(d, sr, 0, 0.3, 0.1);
    sliver(d, sr, 0.0003, f0, 0.5, 1);
    if (Math.random() < 0.5) sliver(d, sr, rand(0.015, 0.05), f0 * rand(1.1, 1.6), 0.18, 0.7);
    return normalize(d, 0.9);
  },
  // the pane whole again: a clear bell-like ring with a slow shimmer (paired partials beating)
  chime(sr = SR){
    const d = new Float32Array(Math.round(sr * 2.2));
    const f = rand(1250, 1420);
    [[1, 1.3, 0.5], [2.32, 0.6, 0.28], [4.25, 0.3, 0.16], [6.63, 0.16, 0.08]].forEach(([r, tau, a]) => {
      modal(d, sr, 0, f * r, tau, a, 0.002);
      modal(d, sr, 0, f * r + rand(0.6, 1.4), tau, a * 0.8, 0.002);
    });
    for (let i = 0; i < 6; i++) sliver(d, sr, 0.02 + i * 0.035, logRand(4200, 9000), 0.04, 0.5);
    return normalize(d, 0.6);
  },
};
function toBuffer(d){
  const b = new AudioBuffer({ length: d.length, numberOfChannels: 1, sampleRate: SR });
  b.copyToChannel(d, 0);
  return b;
}
// Rendering a shatter takes tens of ms, so a few variants are made ahead, off the click path, and a
// used one is replaced in the background.
const bank = { crack: [], shatter: [], chime: [], ping: [] };
let bankStarted = false;
function fillBank(){
  if (bankStarted) return; bankStarted = true;
  const jobs = [];
  for (let i = 0; i < 2; i++) jobs.push(() => bank.crack.push(toBuffer(SYN.crack())), () => bank.shatter.push(toBuffer(SYN.shatter())));
  jobs.push(() => bank.chime.push(toBuffer(SYN.chime())));
  for (let i = 0; i < 18; i++){ const f = 1700 * Math.pow(8000 / 1700, i / 17); jobs.push(() => bank.ping.push({ f, buf: toBuffer(SYN.ping(SR, f)) })); }
  const run = () => {
    const t0 = performance.now();
    while (jobs.length && performance.now() - t0 < 8) jobs.shift()();
    if (jobs.length) setTimeout(run, 30);
  };
  setTimeout(run, 50);
}
function takeBuf(kind){
  const list = bank[kind];
  if (!list.length) return toBuffer(SYN[kind]());
  const b = list.length > 1 ? list.shift() : list[0];
  if (list.length < 2) setTimeout(() => list.push(toBuffer(SYN[kind]())), 400);
  return b;
}
function play(buf, gain, rate = 1, delay = 0){
  const a = audio(); if (!a || !buf) return;
  const src = a.createBufferSource(); src.buffer = buf; src.playbackRate.value = rate;
  const g = a.createGain(); g.gain.value = gain;
  src.connect(g).connect(a.destination); src.start(a.currentTime + delay);
}
let lastPing = 0, pingBurst = 0;
function pingAt(f, gain, delay = 0){
  const a = actx; if (!a || glassAudio.muted) return;
  const now = a.currentTime;
  // many shards landing at once would stack into a hiss; a few per frame is what the ear resolves
  if (now - lastPing > 0.03){ pingBurst = 0; lastPing = now; }
  if (++pingBurst > 3) return;
  if (!bank.ping.length){ play(toBuffer(SYN.ping(SR, f)), gain, 1, delay); return; }
  let best = bank.ping[0];
  for (const p of bank.ping) if (Math.abs(Math.log(p.f / f)) < Math.abs(Math.log(best.f / f))) best = p;
  play(best.buf, gain, clamp(f / best.f, 0.85, 1.18) * rand(0.98, 1.02), delay);
}
const sounds = {
  crack(){ play(takeBuf('crack'), 0.5); },
  shatter(){ play(takeBuf('shatter'), 0.55); },
  land(size, speed){ pingAt(clamp(9000 / Math.pow(size, 0.42), 1700, 8500) * rand(0.9, 1.1), clamp(speed / 2600, 0.02, 0.22)); },
  flick(size){ pingAt(clamp(9000 / Math.pow(size, 0.42), 1700, 8500), 0.12); },
  dock(i, n){ pingAt(2200 * Math.pow(3, i / Math.max(1, n - 1)) * rand(0.97, 1.03), 0.04); },
  chime(){ play(takeBuf('chime'), 0.22); },
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
  }, 2500);
}
glassAudio._synth = (name, sr = SR, arg) => SYN[name](sr, arg);

// ---------------------------------------------------------------- geometry
function parseColor(css){
  if (!css) return null;
  const m = /rgba?\(([^)]+)\)/.exec(css);
  if (m){
    const p = m[1].split(/[\s,/]+/).filter(Boolean).map(parseFloat);
    return [p[0] / 255, p[1] / 255, p[2] / 255, p.length > 3 ? p[3] : 1];
  }
  // hex or named colours go through the browser
  const c = document.createElement('canvas').getContext('2d');
  c.fillStyle = '#000'; c.fillStyle = css;
  const v = c.fillStyle;
  if (/^#[0-9a-f]{6}$/i.test(v)) return [parseInt(v.slice(1, 3), 16) / 255, parseInt(v.slice(3, 5), 16) / 255, parseInt(v.slice(5, 7), 16) / 255, 1];
  return parseColor(/rgba?\(/.test(v) ? v : null);
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
function roundRect(w, h, r){
  const pts = [], seg = 6;
  const cs = [[w - r, r, -Math.PI / 2], [w - r, h - r, 0], [r, h - r, Math.PI / 2], [r, r, Math.PI]];
  for (const [cx, cy, a0] of cs) for (let i = 0; i <= seg; i++){
    const a = a0 + i / seg * Math.PI / 2;
    pts.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]);
  }
  return pts;
}
function area(P){ let s = 0; for (let i = 0; i < P.length; i++){ const a = P[i], b = P[(i + 1) % P.length]; s += a[0] * b[1] - b[0] * a[1]; } return s / 2; }
function centroid(P){
  let cx = 0, cy = 0, A = 0;
  for (let i = 0; i < P.length; i++){ const a = P[i], b = P[(i + 1) % P.length], c = a[0] * b[1] - b[0] * a[1]; A += c; cx += (a[0] + b[0]) * c; cy += (a[1] + b[1]) * c; }
  if (Math.abs(A) < 1e-9){ let x = 0, y = 0; for (const p of P){ x += p[0]; y += p[1]; } return [x / P.length, y / P.length]; }
  return [cx / (3 * A), cy / (3 * A)];
}
// Sutherland-Hodgman against the half-plane dot(p - o, n) <= 0. F[i] flags the edge P[i] -> P[i+1];
// edges made by the cut get newFlag.
function clip(P, F, o, n, newFlag){
  const outP = [], outF = [];
  const side = p => (p[0] - o[0]) * n[0] + (p[1] - o[1]) * n[1];
  for (let i = 0; i < P.length; i++){
    const a = P[i], b = P[(i + 1) % P.length], f = F[i];
    const sa = side(a), sb = side(b), ina = sa <= 0, inb = sb <= 0;
    const I = () => { const t = sa / (sa - sb); return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]; };
    if (ina && inb){ outP.push(a); outF.push(f); }
    else if (ina && !inb){ outP.push(a); outF.push(f); outP.push(I()); outF.push(newFlag); }
    else if (!ina && inb){ outP.push(I()); outF.push(f); }
  }
  return [outP, outF];
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
    const a = P[i], b = P[(i + 1) % P.length];
    const ex = b[0] - a[0], ey = b[1] - a[1];
    const t = clamp(((x - a[0]) * ex + (y - a[1]) * ey) / (ex * ex + ey * ey || 1), 0, 1);
    best = Math.min(best, Math.hypot(x - a[0] - ex * t, y - a[1] - ey * t));
  }
  return best;
}
// quaternions [x, y, z, w]
const qrot = (q, v) => {
  const [x, y, z, w] = q;
  const tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
  return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)];
};
function qnorm(q){ const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1; q[0] /= l; q[1] /= l; q[2] /= l; q[3] /= l; return q; }
function qint(q, w, h){
  const [x, y, z, s] = q, a = w[0] * h * 0.5, b = w[1] * h * 0.5, c = w[2] * h * 0.5;
  q[0] = x + a * s + b * z - c * y;
  q[1] = y + b * s + c * x - a * z;
  q[2] = z + c * s + a * y - b * x;
  q[3] = s - a * x - b * y - c * z;
  qnorm(q);
}
function slerp(a, b, t){
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const bb = d < 0 ? b.map(v => -v) : b.slice(); d = Math.abs(d);
  if (d > 0.9995) return qnorm(a.map((v, i) => v + (bb[i] - v) * t));
  const th = Math.acos(d), s = Math.sin(th), ka = Math.sin((1 - t) * th) / s, kb = Math.sin(t * th) / s;
  return a.map((v, i) => v * ka + bb[i] * kb);
}
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

const TINTS = {
  clear:   { tint: [0.978, 0.992, 0.984], density: 1, edge: [0.34, 0.62, 0.52], frost: 0, streaks: 0 },
  green:   { tint: [0.86, 0.955, 0.89], density: 1, edge: [0.12, 0.42, 0.28], frost: 0, streaks: 0.2 },
  smoked:  { tint: [0.64, 0.62, 0.6], density: 1, edge: [0.13, 0.12, 0.11], frost: 0, streaks: 0 },
  frosted: { tint: [0.97, 0.988, 0.98], density: 1, edge: [0.5, 0.72, 0.64], frost: 1, streaks: 0 },
};
function coloredTint(rgb, density, streaks){
  const m = Math.max(rgb[0], rgb[1], rgb[2], 1e-3);
  // transmission per unit path: the colour's own hue, kept bright enough to see through
  const t = rgb.map(v => clamp(0.35 + 0.65 * v / m, 0.08, 1));
  return { tint: t, density, edge: rgb.map(v => v * 0.45), frost: 0, streaks };
}

// ---------------------------------------------------------------- component
function attachGlass(button, opts = {}){
  const canvas = document.createElement('canvas');
  canvas.setAttribute('aria-hidden', 'true');
  Object.assign(canvas.style, { position: 'absolute', pointerEvents: 'none', zIndex: opts.zIndex ?? 10, left: '0px', top: '0px' });
  const gl = canvas.getContext('webgl2', { alpha: true, premultipliedAlpha: true, antialias: true, depth: false, stencil: false });
  if (!gl) return { detach(){}, supported: false, setTint(){}, reset(){} };
  const sound = opts.sound !== false;
  const snd = (name, ...a) => { if (sound && !glassAudio.muted && actx) sounds[name](...a); };
  if (sound) fillBank();
  const self = {};
  const RESTORE = opts.restoreAfter ?? 2;

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
  const mkTex = () => {
    const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));
    return t;
  };
  const vao = (buf, layout, stride) => {
    const v = gl.createVertexArray();
    gl.bindVertexArray(v); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    let off = 0;
    layout.forEach((size, l) => { gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, size, gl.FLOAT, false, stride * 4, off * 4); off += size; });
    return v;
  };
  let P, shardBuf, shardVao, crackBuf, crackVao, partBuf, partVao, textTex, bgTex;
  // everything GPU-side; run again after a lost context comes back
  function initGL(){
    const sAttrs = ['aPos', 'aHome', 'aEdge', 'aQ', 'aInfo', 'aShadow'];
    P = {
      shadow: program(VS_SHARD, FS_SHADOW, sAttrs), glass: program(VS_SHARD, FS_GLASS, sAttrs),
      crack: program(VS_CRACK, FS_CRACK, ['aAB', 'aCorner', 'aD']), part: program(VS_PART, FS_PART, ['aP', 'aK']),
    };
    shardBuf = gl.createBuffer(); shardVao = vao(shardBuf, [2, 2, 4, 4, 4, 2], VSTRIDE);
    crackBuf = gl.createBuffer(); crackVao = vao(crackBuf, [4, 2, 4], 10);
    partBuf = gl.createBuffer(); partVao = vao(partBuf, [4, 2], 6);
    gl.bindVertexArray(null);
    textTex = mkTex(); bgTex = mkTex();
  }
  initGL();
  const bindTex = (unit, t) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, t); };

  // ---- the button: invisible but still there
  const saved = button.getAttribute('style');
  const cs0 = getComputedStyle(button);
  const btnBg = parseColor(cs0.backgroundColor), btnInk = parseColor(cs0.color);
  // the canvas lies over the button, so a focus ring drawn flush against it would be hidden under the
  // glass rim: push it out past the pane
  const ringOffset = (parseFloat(cs0.outlineOffset) || 0) < 5 ? '6px' : null;
  const hideButton = () => {
    Object.assign(button.style, { background: 'transparent', borderColor: 'transparent', color: 'transparent', boxShadow: 'none', textShadow: 'none' });
    if (ringOffset) button.style.outlineOffset = ringOffset;
  };
  const showButton = () => { if (saved === null) button.removeAttribute('style'); else button.setAttribute('style', saved); };
  hideButton();
  document.body.appendChild(canvas);
  const bgCol = opts.background ? (parseColor(opts.background) || [1, 1, 1]).slice(0, 3) : pageColor(button.parentElement);
  let surface = opts.surface || null;
  let lightK = 0, bgMean = bgCol;
  function uploadSurface(){
    let rgb = bgCol;
    if (surface && surface.image){
      try {
        gl.bindTexture(gl.TEXTURE_2D, bgTex);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, surface.image);
        const c = document.createElement('canvas'); c.width = c.height = 8;
        const x = c.getContext('2d'); x.drawImage(surface.image, 0, 0, 8, 8);
        const d = x.getImageData(0, 0, 8, 8).data; const m = [0, 0, 0];
        for (let i = 0; i < d.length; i += 4) for (let k = 0; k < 3; k++) m[k] += d[i + k] / 255 / 64;
        rgb = m;
      } catch (err) {
        // a cross-origin image can't be read: fall back to letting the page show through
        console.warn('ElementalFX glass: surface image unusable', err);
        surface = null;
      }
    }
    const lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    lightK = ease(clamp((lum - 0.4) / 0.35, 0, 1));
    bgMean = rgb;
    setTint(tintSpec);
    dirty = true;
  }

  // ---- tint
  let tint = TINTS.clear, tintSpec = opts.tint ?? 'clear';
  function setTint(t, color){
    if (color) opts.color = color;
    tintSpec = t;
    if (t && typeof t === 'object'){
      const rgb = (parseColor(t.color) || [0.2, 0.35, 0.8]).slice(0, 3);
      tint = Object.assign(coloredTint(rgb, t.density ?? 1.4, t.streaks ?? 0.6), t.frost != null ? { frost: t.frost } : {});
    } else if (t === 'stained'){
      tint = coloredTint((parseColor(opts.color || '#1f4fd1') || [0.12, 0.31, 0.82]).slice(0, 3), 1.7, 0.9);
    } else if (t === 'auto'){
      if (btnBg && btnBg[3] > 0.2){
        // glass that passes the button's own colour: over this page it looks like the button did
        const c = btnBg.slice(0, 3);
        tint = { tint: c.map((v, i) => clamp(v / Math.max(bgMean[i], 0.05), 0.04, 1)), density: 1, edge: c.map(v => v * 0.45), frost: 0, streaks: 0.25 };
      } else {
        const c = bgCol, mx = Math.max(...c), mn = Math.min(...c);
        tint = mx - mn > 0.12 ? coloredTint(c, 1.1, 0.25) : TINTS.clear;
      }
    } else tint = TINTS[t] || TINTS.clear;
    computeInk();
    dirty = true; wake(0.05);
  }
  // The label's ink is picked against what the pane actually looks like over this page (tint, frost and
  // backdrop together), so it keeps at least 4.5:1 whatever glass is chosen. With 'auto' the button's
  // own text colour is kept when it passes.
  let ink = { rgb: [0.95, 0.98, 0.965], dark: false };
  const lin = v => v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
  const relLum = c => 0.2126 * lin(clamp(c[0], 0, 1)) + 0.7152 * lin(clamp(c[1], 0, 1)) + 0.0722 * lin(clamp(c[2], 0, 1));
  const contrast = (a, b) => { const x = relLum(a), y = relLum(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); };
  function paneColor(){
    let c = bgMean.map((v, i) => v * Math.pow(tint.tint[i], tint.density * 1.1));
    if (tint.frost){
      const milk = [0.82, 0.86, 0.85].map((m, i) => m * 0.45 + (c[i] * 1.08 + 0.06) * 0.55);
      c = c.map((v, i) => v + (milk[i] - v) * tint.frost * 0.35);
    }
    return c.map(v => v * 0.96 + 0.3 * 0.04);
  }
  function computeInk(){
    const pane = paneColor();
    const light = [0.95, 0.98, 0.965], dark = [0.06, 0.09, 0.085];
    const cands = [];
    const own = btnInk;
    if (tintSpec === 'auto' && own && own[3] > 0.5) cands.push(own.slice(0, 3));
    cands.push(relLum(pane) > 0.18 ? dark : light);
    let pick = cands.find(c => contrast(c, pane) >= 4.5);
    if (!pick) pick = contrast(light, pane) >= contrast(dark, pane) ? light : dark;
    ink = { rgb: pick, dark: relLum(pick) < relLum(pane), ratio: contrast(pick, pane) };
  }

  // ---- layout: button-local "home" coords; stage = home + (ox, oy)
  let bw = 0, bh = 0, ox = 0, oy = 0, stageW = 0, stageH = 0, dpr = 1, radius = 0, bevel = 5;
  let fixed = false, bounds = { w: 0, h: 0 }, cbOrigin = { x: 0, y: 0 };
  let panePoly = [], dirty = true;
  let eye = [0, 0];
  function measureBounds(){
    fixed = inFixedContext(button);
    canvas.style.position = fixed ? 'fixed' : 'absolute';
    const de = document.documentElement;
    const disp = canvas.style.display;
    canvas.style.display = 'none';
    bounds = fixed ? { w: de.clientWidth, h: de.clientHeight } : { w: de.clientWidth, h: Math.max(de.scrollHeight, document.body ? document.body.scrollHeight : 0) };
    canvas.style.display = disp;
    canvas.style.left = '0px'; canvas.style.top = '0px';
    const r = canvas.getBoundingClientRect();
    cbOrigin = { x: r.left + (fixed ? 0 : scrollX), y: r.top + (fixed ? 0 : scrollY) };
  }
  function buildText(){
    const c = document.createElement('canvas');
    const s = dpr * 2;
    c.width = Math.max(1, Math.round(bw * s)); c.height = Math.max(1, Math.round(bh * s));
    const x = c.getContext('2d');
    x.fillStyle = '#000'; x.fillRect(0, 0, c.width, c.height);
    const cs = getComputedStyle(button);
    const fsize = parseFloat(cs.fontSize) || 16;
    x.font = `${cs.fontStyle} ${cs.fontWeight} ${fsize * s}px ${cs.fontFamily}`;
    if ('letterSpacing' in x && cs.letterSpacing !== 'normal') x.letterSpacing = `${(parseFloat(cs.letterSpacing) || 0) * s}px`;
    x.textAlign = 'center'; x.textBaseline = 'alphabetic';
    let cx = bw / 2, cy = bh / 2;
    const range = document.createRange();
    range.selectNodeContents(button);
    const rr = range.getBoundingClientRect(), br = button.getBoundingClientRect();
    if (rr.width > 0 && rr.height > 0){ cx = rr.left + rr.width / 2 - br.left; cy = rr.top + rr.height / 2 - br.top; }
    const text = (button.innerText || button.textContent || '').replace(/\s+/g, ' ').trim();
    const m = x.measureText(text);
    const asc = m.fontBoundingBoxAscent ?? fsize * s * 0.8, desc = m.fontBoundingBoxDescent ?? fsize * s * 0.2;
    x.filter = `blur(${0.3 * s}px)`;
    x.fillStyle = '#f00';
    x.fillText(text, cx * s, cy * s + (asc - desc) / 2);
    x.globalCompositeOperation = 'lighter';
    x.filter = `blur(${2.2 * s}px)`;
    x.fillStyle = '#0f0';
    x.fillText(text, cx * s, cy * s + (asc - desc) / 2);
    x.globalCompositeOperation = 'source-over';
    gl.bindTexture(gl.TEXTURE_2D, textTex);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, c);
  }
  function layout(r = button.getBoundingClientRect()){
    const bx = r.left + (fixed ? 0 : scrollX), by = r.top + (fixed ? 0 : scrollY);
    const sizeChanged = Math.abs(r.width - bw) > 0.5 || Math.abs(r.height - bh) > 0.5;
    if (r.width < 2 || r.height < 2){ canvas.style.visibility = 'hidden'; return; }
    canvas.style.visibility = lost ? 'hidden' : '';
    const h = r.height;
    const mx = opts.margin ?? Math.round(Math.max(90, 1.4 * h));
    const mt = Math.round(Math.max(36, 0.6 * h)), mb = Math.round(Math.max(96, 1.6 * h));
    // the stage stays inside the document, so it can never add scrolling
    const x0 = Math.floor(Math.min(bx, Math.max(0, bx - mx))), y0 = Math.floor(Math.min(by, Math.max(0, by - mt)));
    const x1 = Math.ceil(Math.max(bx + r.width, Math.min(bounds.w, bx + r.width + mx))), y1 = Math.ceil(Math.max(by + h, Math.min(bounds.h, by + h + mb)));
    const sw = x1 - x0, sh = y1 - y0;
    ox = bx - x0; oy = by - y0;
    Object.assign(canvas.style, { left: `${x0 - cbOrigin.x}px`, top: `${y0 - cbOrigin.y}px` });
    if (sw !== stageW || sh !== stageH || sizeChanged){
      stageW = sw; stageH = sh;
      dpr = Math.min(devicePixelRatio || 1, 2);
      const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE) || 4096;
      dpr = Math.min(dpr, maxTex / stageW, maxTex / stageH);
      canvas.width = Math.max(1, Math.round(stageW * dpr)); canvas.height = Math.max(1, Math.round(stageH * dpr));
      Object.assign(canvas.style, { width: `${stageW}px`, height: `${stageH}px` });
    }
    if (sizeChanged){
      bw = r.width; bh = r.height;
      radius = Math.min(parseFloat(getComputedStyle(button).borderTopLeftRadius) || 0, bw / 2, bh / 2);
      bevel = clamp(Math.min(bw, bh) * 0.11, 3, 7);
      panePoly = roundRect(bw, bh, Math.max(radius, 1.5));
      buildText();
      toRest();
    }
    const vr = fixed ? { l: x0, t: y0 } : { l: x0 - scrollX, t: y0 - scrollY };
    eye = [innerWidth / 2 - vr.l, innerHeight / 2 - vr.t];
    if (surface && surface.rect){
      const s = surface.rect();
      bgOff = [vr.l - s.left, vr.t - s.top, Math.max(1, s.width), Math.max(1, s.height)];
    }
    dirty = true;
  }
  let bgOff = [0, 0, 1, 1];
  let lastRect = null;
  function checkPlace(){
    if (lost) return false;
    const r = button.getBoundingClientRect();
    const L = lastRect;
    if (L && L.left === r.left && L.top === r.top && L.width === r.width && L.height === r.height && L.sx === scrollX && L.sy === scrollY) return false;
    lastRect = { left: r.left, top: r.top, width: r.width, height: r.height, sx: scrollX, sy: scrollY };
    layout(r);
    wake(0.05);
    return true;
  }
  const slabSD = (x, y) => {
    const qx = Math.abs(x - bw / 2) - (bw / 2 - radius), qy = Math.abs(y - bh / 2) - (bh / 2 - radius);
    return Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0) - radius;
  };

  // ---- state
  let state = 'rest';            // rest | cracked | falling | broken | rising | healing
  let lost = false;
  let phaseT = 0, idleT = 0, crackR = 1e5, healF = 0, crackA = 1, rmFade = false;
  let impact = [0, 0], webMax = 1;
  let shards = [], segs = [], parts = [];
  let press = 0, pressTarget = 0, pressAt = [0, 0];
  const isDisabled = () => button.matches(':disabled') || button.getAttribute('aria-disabled') === 'true';
  let disT = isDisabled() ? 1 : 0, disK = disT;
  let lamp = null, lampK = 0.4, lampKT = 0.4;
  let floorY = 0, zMax = 40, xMin = 0, xMax = 0, landed = false, openK = 0;

  function mkShard(poly, flags){
    const hc = centroid(poly);
    const local = poly.map(p => [p[0] - hc[0], p[1] - hc[1]]);
    const A = Math.max(Math.abs(area(poly)), 0.5);
    let J = 0;
    for (const [x, y] of local) J += x * x + y * y;
    const rg2 = Math.max(J / local.length * 0.5, 4);
    return { poly, flags, hc, local, area: A, m: A, I: A * rg2, p: [hc[0], hc[1], 0], q: [0, 0, 0, 1], v: [0, 0, 0], w: [0, 0, 0],
      seed: Math.random(), mode: 'home', release: 0, sleepT: 0, asleep: false, lastHit: -1, loose: 0, fly: null, alpha: 1, pinned: false, partner: -1 };
  }
  // keepParts: the last glints of a heal play out over the whole pane instead of vanishing with the pieces
  function toRest(keepParts = false){
    state = 'rest'; crackR = 1e5; crackA = 1; rmFade = false; segs = []; blocks = []; content = []; ghostRects = [];
    blockEls = []; mediaEls = []; textNodes = [];
    if (!keepParts) parts = [];
    shards = [mkShard(panePoly, panePoly.map(() => 0))];
    shards[0].seed = 0;
    setCursor(null);
    dirty = true;
  }

  // The spiderweb: straight-ish radial cracks from the blow, joined by rings whose spacing grows
  // outward. Rings are sometimes missing between two rays (making long slivers), quads are sometimes
  // split along a diagonal (sharp triangles), and the outer slivers sometimes fork.
  function buildWeb(ix, iy){
    const corners = [[0, 0], [bw, 0], [0, bh], [bw, bh]];
    const maxR = Math.max(...corners.map(([x, y]) => Math.hypot(x - ix, y - iy)));
    const FAR = maxR * 3 + 50;
    const nR = clamp(Math.round(9 + bw / 90 + rand(0, 3)), 9, 16);
    const base = Math.random() * TAU, ang = [];
    for (let i = 0; i < nR; i++) ang.push(base + (i + rand(-0.32, 0.32)) * TAU / nR);
    const rings = [];
    for (let r = rand(5, 8); r < maxR; r *= rand(1.5, 1.9)) rings.push(r);
    const K = rings.length;
    // ray points: [i][0] is the impact, [i][1..K] on the rings, [i][K+1] far outside the pane
    const R = [], D = [];
    for (let i = 0; i < nR; i++){
      const pts = [[ix, iy]], ds = [0];
      let a = ang[i];
      for (let k = 0; k < K; k++){
        a += rand(-0.07, 0.07);
        const rr = rings[k] * rand(0.88, 1.12);
        const p = [ix + Math.cos(a) * rr, iy + Math.sin(a) * rr];
        ds.push(ds[k] + Math.hypot(p[0] - pts[k][0], p[1] - pts[k][1]));
        pts.push(p);
      }
      const p = [ix + Math.cos(a) * FAR, iy + Math.sin(a) * FAR];
      ds.push(ds[K] + Math.hypot(p[0] - pts[K][0], p[1] - pts[K][1]));
      pts.push(p);
      R.push(pts); D.push(ds);
    }
    const cells = [], rawSegs = [];
    const cellOf = [];
    const seg = (a, b, d0, d1, c1, c2, kind = 1) => rawSegs.push({ a, b, d0, d1, c1, c2, kind });
    for (let i = 0; i < nR; i++){
      const j = (i + 1) % nR, A = R[i], B = R[j];
      const ids = new Array(K + 1);
      let k = 0;
      while (k <= K){
        // merge this band with the next when the ring between them is missing
        let k1 = k;
        while (k1 < K - 1 && k1 >= 1 && Math.random() < 0.22) k1++;
        const poly = [];
        for (let m = k; m <= k1 + 1; m++) poly.push(A[m]);
        for (let m = k1 + 1; m >= k; m--) poly.push(B[m]);
        if (k === 0){ poly.pop(); }
        const id = cells.length;
        if (k1 === k && k >= 1 && k < K && Math.random() < 0.45){
          // a diagonal splits the quad into two slivers
          const flip = Math.random() < 0.5;
          const [p0, p1, p2, p3] = [A[k], A[k + 1], B[k + 1], B[k]];
          if (flip){ cells.push([p0, p1, p2], [p0, p2, p3]); seg(p0, p2, D[i][k], D[i][k] + Math.hypot(p2[0] - p0[0], p2[1] - p0[1]), id, id + 1); }
          else { cells.push([p0, p1, p3], [p1, p2, p3]); seg(p3, p1, D[j][k], D[j][k] + Math.hypot(p1[0] - p3[0], p1[1] - p3[1]), id, id + 1); }
          // which half touches ray A / ray B, and the inner / outer ring
          ids[k] = flip ? { A: id, B: id + 1, inner: id + 1, outer: id } : { A: id, B: id + 1, inner: id, outer: id + 1 };
        } else if (k === K && Math.random() < 0.7){
          // the outer sliver forks
          const t = rand(0.35, 0.65), Q = [A[K][0] + (B[K][0] - A[K][0]) * t, A[K][1] + (B[K][1] - A[K][1]) * t];
          const ma = Math.atan2(Q[1] - iy, Q[0] - ix) + rand(-0.12, 0.12);
          const Fm = [Q[0] + Math.cos(ma) * FAR, Q[1] + Math.sin(ma) * FAR];
          cells.push([A[K], A[K + 1], Fm, Q], [Q, Fm, B[K + 1], B[K]]);
          const dq = Math.hypot(Q[0] - ix, Q[1] - iy) + 3;
          seg(Q, Fm, dq, dq + FAR, id, id + 1);
          ids[k] = { A: id, B: id + 1, inner: id, outer: id, fork: Q };
        } else {
          cells.push(poly);
          for (let m = k; m <= k1; m++) ids[m] = { A: id, B: id, inner: id, outer: id };
        }
        k = k1 + 1;
      }
      cellOf.push(ids);
    }
    // ray segments between neighbouring sectors, and ring segments inside each sector
    for (let i = 0; i < nR; i++){
      const prev = (i + nR - 1) % nR;
      for (let k = 0; k <= K; k++){
        seg(R[i][k], R[i][k + 1], D[i][k], D[i][k + 1], cellOf[prev][k].B, cellOf[i][k].A);
      }
    }
    for (let i = 0; i < nR; i++){
      const j = (i + 1) % nR, ids = cellOf[i];
      for (let k = 1; k <= K; k++){
        const lo = ids[k - 1], hi = ids[k];
        if (lo.outer === hi.inner) continue;
        const a = R[i][k], b = R[j][k];
        const d0 = Math.min(D[i][k], D[j][k]);
        const la = Math.hypot(b[0] - a[0], b[1] - a[1]);
        const c1 = lo.outer, c2 = hi.inner;
        if (hi.fork){
          const Q = hi.fork;
          seg(a, Q, D[i][k], D[i][k] + Math.hypot(Q[0] - a[0], Q[1] - a[1]), c1, hi.A);
          seg(b, Q, D[j][k], D[j][k] + Math.hypot(Q[0] - b[0], Q[1] - b[1]), c1, hi.B);
        } else if (D[i][k] <= D[j][k]) seg(a, b, d0, d0 + la, c1, c2);
        else seg(b, a, d0, d0 + la, c1, c2);
      }
    }
    // clip every cell to the pane; the pane's outline becomes rim edges
    const n = panePoly.length;
    const ctr = [bw / 2, bh / 2];
    const polys = [];
    cells.forEach((poly, ci) => {
      let cp = poly, cf = poly.map(() => 1);
      for (let e = 0; e < n && cp.length >= 3; e++){
        const a = panePoly[e], b = panePoly[(e + 1) % n];
        let nx = b[1] - a[1], ny = a[0] - b[0];
        if ((ctr[0] - a[0]) * nx + (ctr[1] - a[1]) * ny > 0){ nx = -nx; ny = -ny; }
        [cp, cf] = clip(cp, cf, a, [nx, ny], 0);
      }
      if (cp.length >= 3 && Math.abs(area(cp)) > 0.25) polys.push({ P: cp, F: cf, ci });
    });
    // A broken pane keeps jagged teeth in its frame: each piece along the rim loses a thin strip there,
    // cut on a slant so neighbouring strips make a saw edge. The strips stay put when the rest falls.
    const toothCuts = [];
    for (const c of polys.slice()){
      if (Math.random() < 0.15) continue;
      const { P, F } = c, m = P.length;
      let i0 = -1, i1 = -1, chains = 0;
      for (let i = 0; i < m; i++){
        if (F[i] === 0 && F[(i + m - 1) % m] !== 0){ i0 = i; chains++; }
        if (F[i] === 0 && F[(i + 1) % m] !== 0) i1 = (i + 1) % m;
      }
      if (chains !== 1 || i0 < 0 || i1 < 0) continue;
      const Pa = P[i0], Pb = P[i1];
      if (Math.hypot(Pb[0] - Pa[0], Pb[1] - Pa[1]) < 5) continue;
      const inward = p => { const gx = slabSD(p[0] + 0.5, p[1]) - slabSD(p[0] - 0.5, p[1]), gy = slabSD(p[0], p[1] + 0.5) - slabSD(p[0], p[1] - 0.5), l = Math.hypot(gx, gy) || 1; return [-gx / l, -gy / l]; };
      const da = Math.random() < 0.2 ? rand(0.8, 2) : rand(3, 8.5), db = Math.random() < 0.2 ? rand(0.8, 2) : rand(3, 8.5);
      const na = inward(Pa), nb = inward(Pb);
      const A = [Pa[0] + na[0] * da, Pa[1] + na[1] * da], B = [Pb[0] + nb[0] * db, Pb[1] + nb[1] * db];
      let nx = B[1] - A[1], ny = A[0] - B[0];
      const l = Math.hypot(nx, ny); if (l < 1e-3) continue;
      nx /= l; ny /= l;
      // point the cut's normal toward the inside of the pane (away from the rim)
      const mid = [(Pa[0] + Pb[0]) / 2, (Pa[1] + Pb[1]) / 2];
      if ((mid[0] - A[0]) * nx + (mid[1] - A[1]) * ny > 0){ nx = -nx; ny = -ny; }
      const [tP, tF] = clip(P, F, A, [nx, ny], 2);
      const [iP, iF] = clip(P, F, A, [-nx, -ny], 2);
      if (tP.length < 3 || iP.length < 3) continue;
      const tA = Math.abs(area(tP)), cA = Math.abs(area(P));
      let depth = 0; for (const p of tP) depth = Math.max(depth, -slabSD(p[0], p[1]));
      if (tA < 1.5 || depth > 13) continue;
      // a small piece that is mostly rim stays in the frame whole
      if (tA > cA * 0.5){ if (cA < 220) c.whole = true; continue; }
      c.P = iP; c.F = iF;
      const tooth = { P: tP, F: tF, ci: -1, partnerOf: c };
      polys.push(tooth);
      const k = iF.indexOf(2);
      if (k >= 0) toothCuts.push({ a: iP[k], b: iP[(k + 1) % iP.length], tooth, inner: c });
    }
    for (const c of polys) c.F = c.F.map(f => f === 2 ? 1 : f);
    // Neighbouring pieces must share every vertex on a common edge (no T-junctions), or the mended
    // pane shows hairline gaps along some seams that the unbroken pane does not have.
    const allPts = [];
    for (const c of polys) for (const p of c.P) allPts.push(p);
    for (const c of polys){
      const P2 = [], F2 = [], m = c.P.length;
      for (let i = 0; i < m; i++){
        const a = c.P[i], b = c.P[(i + 1) % m];
        P2.push(a); F2.push(c.F[i]);
        const ex = b[0] - a[0], ey = b[1] - a[1], L2 = ex * ex + ey * ey;
        if (L2 < 1e-6) continue;
        const L = Math.sqrt(L2), ins = [];
        for (const q of allPts){
          const t = ((q[0] - a[0]) * ex + (q[1] - a[1]) * ey) / L2;
          if (t <= 1e-4 || t >= 1 - 1e-4) continue;
          if (Math.abs((q[0] - a[0]) * ey - (q[1] - a[1]) * ex) / L > 0.02) continue;
          ins.push([t, q]);
        }
        ins.sort((u, v) => u[0] - v[0]);
        let lt = -1;
        for (const [t, q] of ins){ if (t - lt < 1e-5) continue; lt = t; P2.push(q); F2.push(c.F[i]); }
      }
      c.P = P2; c.F = F2;
    }
    const map = new Map(), out = [];
    polys.forEach((c, i) => { c.idx = i; out.push(mkShard(c.P, c.F)); if (c.ci >= 0) map.set(c.ci, i); });
    polys.forEach((c, i) => {
      if (c.partnerOf){ out[i].pinned = true; out[i].partner = c.partnerOf.idx; }
      else if (c.whole){ out[i].pinned = true; out[i].partner = -1; }
    });
    // segments clipped to the inside of the pane
    const inside = p => slabSD(p[0], p[1]) < -0.2;
    const cut = (a, b) => {
      let lo = 0, hi = 1;
      const at = t => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
      const ia = inside(a), ib = inside(b);
      if (ia && ib) return [0, 1];
      const bis = (x0, x1) => { for (let it = 0; it < 24; it++){ const m = (x0 + x1) / 2; if (inside(at(m)) === inside(at(x0))) x0 = m; else x1 = m; } return (x0 + x1) / 2; };
      if (ia && !ib) return [0, bis(0, 1)];
      if (!ia && ib) return [bis(0, 1), 1];
      // both ends outside: a chord through the pane or no crossing at all
      let best = -1;
      for (let s = 1; s < 64; s++) if (inside(at(s / 64))){ best = s / 64; break; }
      if (best < 0) return null;
      lo = bis(0, best); hi = bis(best, 1);
      return [lo, hi];
    };
    segs = [];
    for (const s of rawSegs){
      const iv = cut(s.a, s.b);
      if (!iv) continue;
      const [t0, t1] = iv;
      const A = [s.a[0] + (s.b[0] - s.a[0]) * t0, s.a[1] + (s.b[1] - s.a[1]) * t0];
      const B = [s.a[0] + (s.b[0] - s.a[0]) * t1, s.a[1] + (s.b[1] - s.a[1]) * t1];
      if (Math.hypot(B[0] - A[0], B[1] - A[1]) < 0.4) continue;
      segs.push({ a: A, b: B, d0: s.d0 + (s.d1 - s.d0) * t0, d1: s.d0 + (s.d1 - s.d0) * t1, seed: Math.random(), s1: map.get(s.c1) ?? -1, s2: map.get(s.c2) ?? -1 });
    }
    for (const t of toothCuts){
      const da = Math.hypot(t.a[0] - ix, t.a[1] - iy), db = Math.hypot(t.b[0] - ix, t.b[1] - iy);
      const [a, b] = da <= db ? [t.a, t.b] : [t.b, t.a];
      const d0 = Math.min(da, db);
      segs.push({ a, b, d0, d1: d0 + Math.hypot(b[0] - a[0], b[1] - a[1]), seed: Math.random(), s1: t.tooth.idx, s2: t.inner.idx });
    }
    segs.push({ a: [ix, iy], b: [ix, iy], d0: 0, d1: 0, seed: Math.random(), s1: -1, s2: -1, crush: true });
    webMax = Math.max(1, ...segs.map(s => s.d1));
    return out;
  }

  // Page content inside the stage: other controls (a link beside or under the button), lines of text,
  // images. The ledge is placed in free space where it can be; any part of a loose shard that still
  // ends up over content turns to a ghost so the page stays readable, and presses on a control always
  // reach it.
  let blockEls = [], mediaEls = [], textNodes = [], blocks = [], content = [], ghostRects = [];
  const MEDIA = 'img,svg,video,canvas,iframe,picture,object';
  function scanBlockers(){
    const r = canvas.getBoundingClientRect(), found = new Set(), media = new Set(), texts = new Set(), seen = new Set();
    for (let y = 4; y < stageH; y += 9) for (let x = 4; x < stageW; x += 9){
      const px = r.left + x, py = r.top + y;
      if (px < 0 || py < 0 || px >= innerWidth || py >= innerHeight) continue;
      const el = document.elementFromPoint(px, py);
      if (!el || seen.has(el)) continue;
      seen.add(el);
      if (el === button || button.contains(el)) continue;
      const hit = el.closest && el.closest(INTERACTIVE);
      if (hit && !hit.contains(button)){ found.add(hit); continue; }
      if (el.matches && el.matches(MEDIA)){ media.add(el); continue; }
      for (const nd of el.childNodes) if (nd.nodeType === 3 && /\S/.test(nd.nodeValue)) texts.add(nd);
    }
    blockEls = [...found]; mediaEls = [...media]; textNodes = [...texts].slice(0, 40);
    refreshBlockers();
  }
  const rng = document.createRange();
  function refreshBlockers(){
    const r = canvas.getBoundingClientRect();
    blocks = []; content = [];
    const add = (list, b, pad) => {
      if (!b.width || !b.height) return;
      const q = [b.left - r.left - pad, b.top - r.top - pad, b.right - r.left + pad, b.bottom - r.top + pad];
      if (q[2] > 0 && q[3] > 0 && q[0] < stageW && q[1] < stageH) list.push(q);
    };
    for (const el of blockEls) if (el.isConnected) add(blocks, el.getBoundingClientRect(), 4);
    for (const el of mediaEls) if (el.isConnected) add(content, el.getBoundingClientRect(), 2);
    for (const t of textNodes){
      if (!t.isConnected) continue;
      rng.selectNodeContents(t);
      for (const b of rng.getClientRects()) add(content, b, 2);
    }
    ghostRects = blocks.concat(content);
    if (ghostRects.length > MAX_GHOST){
      // keep what lies nearest the ledge, where the pieces end up
      const fy = floorY + oy;
      ghostRects.sort((a, b) => Math.abs((a[1] + a[3]) / 2 - fy) - Math.abs((b[1] + b[3]) / 2 - fy));
      ghostRects.length = MAX_GHOST;
    }
  }
  // Where the ledge goes: the first band below the pane, across where the pile spreads, that is free of
  // page content and deep enough for a pile. Failing that, just under the frame (and the ghosting keeps
  // whatever lies under the pile readable).
  function chooseFloor(){
    const sx0 = Math.max(-ox + 24, -50), sx1 = Math.min(stageW - ox - 24, bw + 50);
    const obst = [];
    for (const [x0, y0, x1, y1] of ghostRects){
      if (x1 - ox < sx0 || x0 - ox > sx1) continue;
      obst.push([y0 - oy, y1 - oy]);
    }
    obst.sort((a, b) => a[0] - b[0]);
    const top = bh + 2, bottom = stageH - oy - 22;
    const gaps = [];
    let y = top;
    for (const [a, b] of obst){
      if (y >= bottom) break;
      if (b <= y) continue;
      if (a > y) gaps.push([y, Math.min(a, bottom)]);
      y = Math.max(y, b);
    }
    if (y < bottom) gaps.push([y, bottom]);
    const MIN = 20;
    const g = gaps.find(([a, b]) => b - a >= MIN && a - top < 150);
    if (g){
      floorY = g[0] + (g[0] === top ? 8 : 4);
      zMax = clamp((g[1] - floorY - 5) / PZ, 8, 110);
    } else {
      floorY = bh + 6;
      zMax = clamp((Math.min(bottom, obst.length ? Math.max(obst[0][0], floorY + 12) : bottom) - floorY - 4) / PZ, 8, 110);
    }
    xMin = -ox + 6; xMax = stageW - ox - 6;
  }

  function smashAt(ix, iy){
    impact = [ix, iy];
    floorY = bh + 8;
    scanBlockers();
    shards = buildWeb(ix, iy);
    // the crushed spot is shared by the smallest pieces round it
    const near = shards.filter(s => Math.hypot(s.hc[0] - ix, s.hc[1] - iy) < 14).map(s => shards.indexOf(s));
    const crushSeg = segs[segs.length - 1];
    crushSeg.s1 = near[0] ?? -1; crushSeg.s2 = near[1] ?? crushSeg.s1;
    state = 'cracked'; phaseT = 0; crackR = 0; crackA = 1; idleT = 0; landed = false; rmFade = reduceMotion.matches;
    chooseFloor();
    // pieces around the blow are already loose: they drop out right away; the rest go a beat later,
    // near the blow first
    for (const s of shards){
      if (s.pinned){ s.release = Infinity; continue; }
      const dist = Math.hypot(s.hc[0] - ix, s.hc[1] - iy);
      s.release = s.area < 30 && dist < 18 ? rand(0.03, 0.1) : BEAT + CRACK_T * 0.6 + dist / webMax * 0.08 + rand(0, 0.07);
    }
    if (!rmFade) for (let i = 0; i < 26; i++) dust(ix + rand(-5, 5), iy + rand(-5, 5), rand(0, 0.12));
    snd('crack');
    wake(1);
  }
  function dust(hx, hy, delay = 0){
    parts.push({ x: hx, y: hy, z: rand(0, 10), vx: rand(-90, 90), vy: rand(-140, 20), vz: rand(-10, 60), size: rand(1.2, 2.6), age: -delay, life: rand(0.8, 1.5), kind: 0, tw: Math.random() * TAU, landed: false });
  }
  function star(hx, hy, delay = 0, size = 9){
    parts.push({ x: hx, y: hy, z: 0, vx: 0, vy: 0, vz: 0, size, age: -delay, life: 0.55, kind: 1, tw: 0, still: true });
  }

  function release(s){
    s.mode = 'free'; s.asleep = false; s.sleepT = 0;
    const dx = s.hc[0] - impact[0], dy = s.hc[1] - impact[1], dl = Math.hypot(dx, dy) || 1;
    const near = Math.exp(-dl / 60);
    const out = rand(20, 70) + 140 * near;
    // throw the pieces toward whichever side has room, less hard toward a cramped one
    const roomL = ox, roomR = stageW - ox - bw;
    const bias = (roomR - roomL) / Math.max(1, roomR + roomL);
    const room = dx < 0 ? roomL : roomR;
    const vx = dx / dl * out * clamp(room / 110, 0.25, 1) + bias * rand(40, 110);
    s.v = [vx, dy / dl * out * 0.5 - rand(0, 50), rand(30, 260) - 50 * near];
    // it comes to rest a little tilted, propped on the debris under it
    s.rest = qnorm([rand(-0.14, 0.14), 0, rand(-0.14, 0.14), 1]);
    s.w = [-rand(1.5, 8), rand(-4, 4), rand(-5, 5) * (1 + near)];
  }
  function flyHome(s, delay){
    s.mode = 'fly'; s.asleep = false;
    s.fly = { t: -delay, p0: s.p.slice(), q0: s.q.slice(), lift: rand(24, 46) + Math.max(0, s.p[1] - s.hc[1]) * 0.35, dur: RISE_T * rand(0.9, 1.1) };
  }
  function startRestore(){
    if (state === 'rest' || state === 'rising' || state === 'healing') return;
    if (state === 'cracked'){ state = 'healing'; phaseT = 0; healF = webMax + 8; snd('chime'); wake(1); return; }
    state = 'rising'; phaseT = 0;
    const order = shards.map((s, i) => [Math.hypot(s.hc[0] - impact[0], s.hc[1] - impact[1]), i]).filter(([, i]) => shards[i].mode === 'free').sort((a, b) => b[0] - a[0]);
    order.forEach(([, i], k) => flyHome(shards[i], k / Math.max(1, order.length - 1) * RISE_SPREAD + rand(0, 0.04)));
    dockCount = 0;
    grab = null; setCursor(null);
    wake(1);
  }
  let dockCount = 0;

  // ---- physics
  const PLANES = () => [
    [[0, -1, 0], -floorY, 1],          // the ledge
    [[0, 0, 1], -PAGE_D, 2],           // the page behind
    [[0, 0, -1], -zMax, 0],
    [[1, 0, 0], xMin, 0],
    [[-1, 0, 0], -xMax, 0],
  ];
  function stepBody(s, h, planes){
    s.v[1] += GRAV * h;
    const dr = Math.exp(-0.4 * h), dw = Math.exp(-0.6 * h);
    for (let k = 0; k < 3; k++){ s.v[k] *= dr; s.w[k] *= dw; s.p[k] += s.v[k] * h; }
    qint(s.q, s.w, h);
    let touching = false;
    const rel = s.local.map(([x, y]) => qrot(s.q, [x, y, 0]));
    // The sides of the stage and the front of the ledge are not walls: a piece that comes near one is
    // eased back and slowed, so the pile ends raggedly inside the stage instead of stacking up flat
    // against an invisible plane.
    let lo = Infinity, hi = -Infinity, zf = -Infinity;
    for (const r of rel){ lo = Math.min(lo, r[0]); hi = Math.max(hi, r[0]); zf = Math.max(zf, r[2]); }
    const penL = xMin + WALL_SOFT - (s.p[0] + lo), penR = s.p[0] + hi - (xMax - WALL_SOFT), penZ = s.p[2] + zf - (zMax - Math.min(14, zMax * 0.4));
    if (penL > 0){ s.v[0] += penL * 70 * h; if (s.v[0] < 0) s.v[0] *= Math.exp(-9 * h); }
    if (penR > 0){ s.v[0] -= penR * 70 * h; if (s.v[0] > 0) s.v[0] *= Math.exp(-9 * h); }
    if (penZ > 0){ s.v[2] -= penZ * 120 * h; if (s.v[2] > 0) s.v[2] *= Math.exp(-9 * h); }
    for (const [n, dd, kind] of planes){
      let maxPen = 0;
      for (const r of rel){
        const P = [s.p[0] + r[0], s.p[1] + r[1], s.p[2] + r[2]];
        const dist = dot3(n, P) - dd;
        if (dist >= 0) continue;
        if (kind === 1){ touching = true; landed = true; }
        maxPen = Math.max(maxPen, -dist);
        const wr = cross(s.w, r);
        const vc = [s.v[0] + wr[0], s.v[1] + wr[1], s.v[2] + wr[2]];
        const vn = dot3(vc, n);
        if (vn >= 0) continue;
        const rn = cross(r, n);
        const kn = 1 / s.m + dot3(rn, rn) / s.I;
        const e = vn < -260 ? 0.28 : 0;
        const jn = -(1 + e) * vn / kn;
        for (let k = 0; k < 3; k++){ s.v[k] += n[k] * jn / s.m; s.w[k] += rn[k] * jn / s.I; }
        if (kind && vn < -90 && simT - s.lastHit > 0.07){ s.lastHit = simT; snd('land', s.area, -vn * (kind === 2 ? 0.5 : 1)); }
        // Coulomb friction at the contact
        const wr2 = cross(s.w, r);
        const vc2 = [s.v[0] + wr2[0], s.v[1] + wr2[1], s.v[2] + wr2[2]];
        const vn2 = dot3(vc2, n);
        const vt = [vc2[0] - n[0] * vn2, vc2[1] - n[1] * vn2, vc2[2] - n[2] * vn2];
        const lt = Math.hypot(...vt);
        if (lt > 1e-5){
          const t = vt.map(v => v / lt), rt = cross(r, t);
          const kt = 1 / s.m + dot3(rt, rt) / s.I;
          const jt = -Math.min(lt / kt, MU * jn);
          for (let k = 0; k < 3; k++){ s.v[k] += t[k] * jt / s.m; s.w[k] += rt[k] * jt / s.I; }
        }
      }
      if (maxPen > 0) for (let k = 0; k < 3; k++) s.p[k] += n[k] * maxPen * 0.8;
    }
    if (touching){
      // a sliver balanced on its edge topples flat
      const nF = qrot(s.q, [0, 0, 1]);
      const rn = qrot(s.rest || [0, 0, 0, 1], [0, 1, 0]);
      const sg = nF[1] >= 0 ? 1 : -1;
      const tgt = [rn[0] * sg, rn[1] * sg, rn[2] * sg];
      const ax = cross(nF, tgt);
      const sp = Math.hypot(...s.v);
      const k = sp < 120 ? 90 : 20;
      for (let i = 0; i < 3; i++){ s.w[i] += ax[i] * k * h; s.w[i] *= Math.exp(-3 * h); }
      s.v[0] *= Math.exp(-2.5 * h); s.v[2] *= Math.exp(-2.5 * h);
      if (sp < 14 && Math.hypot(...s.w) < 1.2 && Math.abs(nF[1]) > 0.85){
        s.sleepT += h;
        if (s.sleepT > 0.12){ s.asleep = true; s.v = [0, 0, 0]; s.w = [0, 0, 0]; }
      } else s.sleepT = 0;
    } else s.sleepT = 0;
  }
  let acc = 0;
  function stepShards(dt){
    let moving = false;
    if (state === 'falling' || state === 'broken' || state === 'cracked'){
      acc = Math.min(acc + dt, H_STEP * MAX_SUB);
      const planes = PLANES();
      while (acc >= H_STEP){
        acc -= H_STEP;
        for (const s of shards) if (s.mode === 'free' && !s.asleep) stepBody(s, H_STEP, planes);
      }
      for (const s of shards) if (s.mode === 'free' && !s.asleep) moving = true;
    }
    if (state === 'rising'){
      let all = true;
      for (const s of shards){
        if (s.mode !== 'fly') continue;
        const f = s.fly;
        f.t += dt;
        const t = clamp(f.t / f.dur, 0, 1);
        if (t <= 0){ all = false; continue; }
        const u = smoother(t);
        // up off the ledge first, then over into the frame, easing in so it lands with no velocity
        const home = [s.hc[0], s.hc[1], 0];
        const c1 = [f.p0[0], f.p0[1] - f.lift, f.p0[2] * 0.5 + 14];
        const c2 = [home[0], home[1] + 6, 26];
        const b = (a, c, d, e) => { const m = 1 - u; return m * m * m * a + 3 * m * m * u * c + 3 * m * u * u * d + u * u * u * e; };
        s.p = [0, 1, 2].map(k => b(f.p0[k], c1[k], c2[k], home[k]));
        s.q = slerp(f.q0, [0, 0, 0, 1], smoother(clamp(t * 1.1, 0, 1)));
        if (t >= 1){
          s.p = home; s.q = [0, 0, 0, 1]; s.mode = 'home'; s.v = [0, 0, 0]; s.w = [0, 0, 0];
          snd('dock', dockCount++, shards.length);
        } else all = false;
        moving = true;
      }
      if (all && shards.every(s => s.mode === 'home')){
        state = 'healing'; phaseT = 0; healF = webMax + 8;
        snd('chime');
      }
    }
    if ((blockEls.length || mediaEls.length || textNodes.length) && state !== 'rest') refreshBlockers();
    // how far each piece is from its place in the frame: its seams only show once it is back
    let lsum = 0, ln = 0;
    for (const s of shards){
      const d = Math.hypot(s.p[0] - s.hc[0], s.p[1] - s.hc[1], s.p[2]);
      const ang = 2 * Math.acos(clamp(Math.abs(s.q[3]), 0, 1));
      s.loose = clamp(Math.max(d / 2.5, ang * 25), 0, 1);
      if (!s.pinned){ lsum += s.loose; ln++; }
    }
    // how open the frame is, for pieces that stayed in it whole
    openK = ln ? clamp(lsum / ln * 1.5, 0, 1) : 0;
    return moving;
  }
  function stepPhases(dt){
    phaseT += dt;
    if (state === 'cracked'){
      crackR = Math.min(webMax + 10, phaseT / CRACK_T * webMax);
      if (rmFade){
        crackR = webMax + 10;
        crackA = phaseT < 0.6 ? 1 : Math.max(0, 1 - (phaseT - 0.6) / 0.5);
        if (phaseT > 1.1) toRest();
        return;
      }
      for (const s of shards) if (s.mode === 'home' && phaseT >= s.release) release(s);
      if (phaseT >= BEAT + CRACK_T){ state = 'falling'; phaseT = 0; snd('shatter'); for (let i = 0; i < 40; i++){ const g = segs[Math.floor(Math.random() * segs.length)]; const t = Math.random(); dust(g.a[0] + (g.b[0] - g.a[0]) * t, g.a[1] + (g.b[1] - g.a[1]) * t, rand(0, 0.15)); } }
    } else if (state === 'falling'){
      for (const s of shards) if (s.mode === 'home' && phaseT + BEAT + CRACK_T >= s.release) release(s);
      if (landed && !grab) idleT += dt;
      if (shards.every(s => s.pinned || (s.mode !== 'home' && s.asleep)) || phaseT > 1.6){ state = 'broken'; phaseT = 0; }
      else if (idleT >= RESTORE) startRestore();
    } else if (state === 'broken'){
      if (!grab) idleT += dt;
      if (idleT >= RESTORE) startRestore();
    } else if (state === 'healing'){
      healF = webMax + 8 - phaseT / HEAL_T * (webMax + 16);
      if (healF < 4 && !healStar){ healStar = true; star(impact[0], impact[1], 0, 14); star(impact[0] + rand(-8, 8), impact[1] + rand(-5, 5), 0.08, 8); }
      if (phaseT >= HEAL_T + 0.25){ healStar = false; toRest(true); }
    }
  }
  let healStar = false;
  function stepParts(dt){
    for (const p of parts){
      p.age += dt;
      if (p.age < 0 || p.still) continue;
      if (!p.landed){
        p.vy += GRAV * 0.8 * dt;
        p.x += p.vx * dt; p.y += p.vy * dt; p.z = clamp(p.z + p.vz * dt, -PAGE_D, zMax);
        if (p.y >= floorY){ p.y = floorY; if (p.vy > 200){ p.vy *= -0.25; p.vx *= 0.5; } else { p.landed = true; } }
      }
    }
    parts = parts.filter(p => p.age < p.life);
    return parts.length > 0;
  }

  // ---- interaction
  const root = document.documentElement;
  const baseCursor = root.style.cursor;
  let cursor = null;
  function setCursor(c){ if (c === cursor) return; cursor = c; root.style.cursor = c || baseCursor; }
  let grab = null, lastDown = null, swallowUntil = 0;
  const toStage = (cx, cy) => { const r = canvas.getBoundingClientRect(); return [cx - r.left, cy - r.top]; };
  const INTERACTIVE = 'a,button,input,select,textarea,label,summary,[role=button],[tabindex],[contenteditable]';
  function screenOf(s){
    return s.local.map(([x, y]) => { const r = qrot(s.q, [x, y, 0]); return [s.p[0] + r[0] + ox, s.p[1] + r[1] + (s.p[2] + r[2]) * PZ + oy]; });
  }
  function shardAt(x, y, tol = GRAB){
    if (state !== 'falling' && state !== 'broken') return null;
    // the piece drawn on top (nearest the viewer) wins
    let best = null, bd = tol, top = null;
    for (const s of shards){
      if (s.mode !== 'free') continue;
      const P = screenOf(s);
      if (pointInPoly(P, x, y)){ if (!top || s.p[2] > top.p[2]) top = s; continue; }
      if (top) continue;
      const d = distToPoly(P, x, y);
      if (d < bd){ bd = d; best = s; }
    }
    return top || best;
  }
  function onDocDown(e){
    if (e.button !== 0 || lost || grab) return;
    checkPlace();
    const [x, y] = toStage(e.clientX, e.clientY);
    if (x < 0 || y < 0 || x > stageW || y > stageH) return;
    const onBtn = e.target === button || button.contains(e.target);
    if (sound) audio();
    if (state === 'rest'){
      if (!onBtn) return;
      lastDown = [x - ox, y - oy]; pressAt = lastDown.slice(); pressTarget = 1;
      wake(0.3);
      return;
    }
    if (!onBtn && e.target.closest && e.target.closest(INTERACTIVE)) return;
    const s = shardAt(x, y);
    if (!s) return;
    e.preventDefault();
    grab = { id: e.pointerId, x0: x, y0: y, x, y, lx: x, ly: y, vx: 0, vy: 0, s, dragging: false };
    swallowUntil = performance.now() + 1e9;
    idleT = 0;
    setCursor('grabbing');
    wake(0.5);
  }
  function onDocMove(e){
    if (lost) return;
    checkPlace();
    const [x, y] = toStage(e.clientX, e.clientY);
    const near = x > -40 && y > -40 && x < stageW + 40 && y < stageH + 40;
    if (near){ lamp = [x, y]; lampKT = 1; dirty = true; wake(0.1); }
    else if (lampKT !== 0.4){ lampKT = 0.4; wake(0.5); }
    if (state === 'broken' && x >= 0 && y >= 0 && x < stageW && y < stageH) idleT = Math.min(idleT, RESTORE * 0.5);
    if (grab && e.pointerId === grab.id){
      grab.x = x; grab.y = y;
      if (!grab.dragging && Math.hypot(x - grab.x0, y - grab.y0) > DRAG_START) grab.dragging = true;
      wake(0.3);
      return;
    }
    if (state === 'falling' || state === 'broken') setCursor(near && shardAt(x, y) ? 'pointer' : null);
  }
  function onDocUp(e){
    if (!grab || e.pointerId !== grab.id){ pressTarget = 0; return; }
    const g = grab; grab = null;
    swallowUntil = performance.now() + 500;
    setCursor(null);
    idleT = 0;
    if (!g.dragging && (state === 'falling' || state === 'broken')) flick(g.s);
    wake(0.5);
  }
  function flick(s){
    s.mode = 'free'; s.asleep = false; s.sleepT = 0;
    s.v = [rand(-90, 90), -rand(380, 560), rand(-50, 50)];
    s.w = [rand(-14, 14), rand(-10, 10), rand(-12, 12)];
    snd('flick', s.area);
    // a flick jostles whatever it lies against
    const c0 = s.p;
    for (const o of shards){
      if (o === s || o.mode !== 'free' || Math.hypot(o.p[0] - c0[0], o.p[2] - c0[2]) > 24) continue;
      o.asleep = false; o.sleepT = 0; o.v[1] -= rand(60, 140); o.v[0] += rand(-40, 40); o.w[0] += rand(-3, 3);
    }
  }
  // a drag sweeps through the pile, shoving the shards it passes over
  function dragStep(dt){
    const g = grab;
    const dx = g.x - g.lx, dy = g.y - g.ly, L = Math.hypot(dx, dy);
    g.lx = g.x; g.ly = g.y;
    if (L < 0.01) return;
    const vx = dx / Math.max(dt, 1e-3), vy = dy / Math.max(dt, 1e-3);
    const n = Math.ceil(L / 4);
    const hit = new Set();
    for (let i = 1; i <= n; i++){
      const px = g.x - dx + dx * i / n, py = g.y - dy + dy * i / n;
      for (const s of shards){
        if (s.mode !== 'free' || hit.has(s)) continue;
        const P = screenOf(s);
        if (pointInPoly(P, px, py) || distToPoly(P, px, py) < 5) hit.add(s);
      }
    }
    for (const s of hit){
      s.asleep = false; s.sleepT = 0;
      s.v[0] += (clamp(vx, -900, 900) - s.v[0]) * 0.35;
      s.v[2] += (clamp(vy / PZ, -900, 900) - s.v[2]) * 0.2;
      s.v[1] -= 50;
      s.w[1] += rand(-3, 3);
      if (simT - s.lastHit > 0.09){ s.lastHit = simT; snd('land', s.area, 500); }
    }
  }
  function onDocClick(e){
    if (e.detail > 0 && performance.now() < swallowUntil){ swallowUntil = 0; e.preventDefault(); e.stopImmediatePropagation(); }
  }
  let touchOwned = false;
  function onTouch(e){
    if (e.type === 'touchend' || e.type === 'touchcancel'){ if (!e.touches.length) touchOwned = false; return; }
    if (!e.cancelable || lost) return;
    if (e.type === 'touchmove'){ if (touchOwned || grab) e.preventDefault(); return; }
    if (state !== 'falling' && state !== 'broken') return;
    const t = e.touches[0]; if (!t) return;
    const [x, y] = toStage(t.clientX, t.clientY);
    const el = document.elementFromPoint(t.clientX, t.clientY);
    if (el && el !== button && !button.contains(el) && el.closest && el.closest(INTERACTIVE)) return;
    if (shardAt(x, y)){ touchOwned = true; e.preventDefault(); }
  }
  function onClick(e){
    if (lost) return;
    const now = performance.now();
    if (e.detail > 0 && now < swallowUntil){ swallowUntil = 0; e.preventDefault(); e.stopImmediatePropagation(); return; }
    if (state !== 'rest'){
      // the pane isn't there while it lies in pieces: activating it again only brings it back
      e.preventDefault(); e.stopImmediatePropagation();
      if (state === 'broken' || (state === 'falling' && e.detail === 0)) startRestore();
      return;
    }
    if (sound) audio();
    const at = e.detail > 0 && lastDown ? lastDown : [bw * rand(0.4, 0.6), bh * rand(0.4, 0.6)];
    lastDown = null; pressTarget = 0;
    smashAt(clamp(at[0], 3, bw - 3), clamp(at[1], 3, bh - 3));
  }
  const onCancel = e => { if (grab && e.pointerId === grab.id){ grab = null; setCursor(null); } pressTarget = 0; };
  const onLeave = e => { if (!e.relatedTarget){ lampKT = 0.4; wake(0.5); } };
  const capNP = { capture: true, passive: false };
  document.addEventListener('touchstart', onTouch, capNP);
  document.addEventListener('touchmove', onTouch, capNP);
  document.addEventListener('touchend', onTouch, true);
  document.addEventListener('touchcancel', onTouch, true);
  document.addEventListener('pointerdown', onDocDown, true);
  document.addEventListener('click', onDocClick, true);
  document.addEventListener('pointerout', onLeave, true);
  addEventListener('pointermove', onDocMove, { passive: true });
  addEventListener('pointerup', onDocUp, true);
  addEventListener('pointercancel', onCancel, true);
  button.addEventListener('click', onClick, true);

  // ---- loop
  let simT = 0, activeUntil = 0;
  let raf = 0, inFrame = false, last = performance.now();
  // one animation loop at most: a wake from inside a frame must not start a second loop
  function wake(sec){ if (lost || manual) { activeUntil = Math.max(activeUntil, simT + sec); return; } activeUntil = Math.max(activeUntil, simT + sec); if (!raf && !inFrame){ last = performance.now(); raf = requestAnimationFrame(frame); audioAwake(self, true); } }

  // ---- render
  let sdata = new Float32Array(0), cdata = new Float32Array(0), pdata = new Float32Array(0);
  function buildShardVerts(){
    const list = shards.slice().sort((a, b) => a.p[2] - b.p[2]);
    let nv = 0;
    for (const s of list) nv += s.local.length * 3;
    if (sdata.length < nv * VSTRIDE) sdata = new Float32Array(nv * VSTRIDE * 2);
    let o = 0;
    let faces = 0;
    const put = (sx, sy, hx, hy, ed, nx, ny, kind, s, shx, shy) => {
      sdata.set([sx, sy, hx, hy, ed, nx, ny, kind, s.q[0], s.q[1], s.q[2], s.q[3], s.loose, s.seed, s.alpha, faces, shx, shy], o); o += VSTRIDE;
    };
    for (const s of list){
      // a tooth left in the frame shows its broken face while the piece it came off is gone
      faces = s.pinned ? (s.partner >= 0 ? shards[s.partner].loose : openK) : s.loose;
      const W = s.local.map(([x, y]) => { const r = qrot(s.q, [x, y, 0]); return [s.p[0] + r[0], s.p[1] + r[1], s.p[2] + r[2]]; });
      // shadow: onto the page behind, or straight down onto the ledge, whichever is nearer. A piece in
      // the frame always shadows the page, exactly as the whole pane does.
      const onPage = s.mode === 'home';
      const shOf = (X, Y, Z) => {
        const toPage = Z + PAGE_D, toFloor = onPage ? 1e9 : floorY - Y;
        return toPage <= toFloor ? [toPage * 0.35, toPage * 0.55] : [toFloor * 0.3, toFloor];
      };
      const sh = W.map(([X, Y, Z]) => shOf(X, Y, Z));
      const S = W.map(([X, Y, Z]) => [X + ox, Y + Z * PZ + oy]);
      const n = S.length;
      // the fan's hub is the shard's own centroid (hc in the home frame), carried exactly by its pose, so
      // every triangle maps the home frame affinely and a docked piece renders like the whole pane
      const cx = s.p[0] + ox, cy = s.p[1] + s.p[2] * PZ + oy;
      const [csx, csy] = shOf(s.p[0], s.p[1], s.p[2]);
      for (let i = 0; i < n; i++){
        const j = (i + 1) % n;
        const a = s.poly[i], b = s.poly[j];
        let nx = b[1] - a[1], ny = a[0] - b[0];
        const l = Math.hypot(nx, ny) || 1; nx /= l; ny /= l;
        if ((s.hc[0] - a[0]) * nx + (s.hc[1] - a[1]) * ny > 0){ nx = -nx; ny = -ny; }
        const kind = s.flags[i];
        const dc = kind ? Math.abs((s.hc[0] - a[0]) * nx + (s.hc[1] - a[1]) * ny) : 99;
        put(cx, cy, s.hc[0], s.hc[1], dc, nx, ny, kind, s, csx, csy);
        put(S[i][0], S[i][1], a[0], a[1], kind ? 0 : 99, nx, ny, kind, s, sh[i][0], sh[i][1]);
        put(S[j][0], S[j][1], b[0], b[1], kind ? 0 : 99, nx, ny, kind, s, sh[j][0], sh[j][1]);
      }
    }
    return o / VSTRIDE;
  }
  function buildCrackVerts(){
    if (cdata.length < segs.length * 60) cdata = new Float32Array(segs.length * 120);
    let o = 0;
    for (const g of segs){
      const l1 = g.s1 >= 0 ? shards[g.s1].loose : 0, l2 = g.s2 >= 0 ? shards[g.s2].loose : 0;
      const vis = 1 - Math.max(l1, l2);
      if (vis <= 0.001) continue;
      for (const [al, sd] of [[0, -1], [1, -1], [1, 1], [0, -1], [1, 1], [0, 1]]){
        cdata.set([g.a[0], g.a[1], g.b[0], g.b[1], al, sd, g.d0, g.d1, g.seed, vis], o); o += 10;
      }
    }
    return o / 10;
  }
  const ghostBuf = new Float32Array(MAX_GHOST * 4);
  function ghostUniforms(u){
    const n = state === 'rest' ? 0 : Math.min(ghostRects.length, MAX_GHOST);
    for (let i = 0; i < n; i++) ghostBuf.set(ghostRects[i], i * 4);
    if (u.uGhost) gl.uniform4fv(u.uGhost, ghostBuf);
    if (u.uNGhost) gl.uniform1i(u.uNGhost, n);
  }
  function common(u){
    gl.uniform2f(u.uStage, stageW, stageH);
    if (u.uSize) gl.uniform2f(u.uSize, bw, bh);
    if (u.uRadius) gl.uniform1f(u.uRadius, radius);
  }
  function render(){
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    const L = lamp || [ox + bw * 0.15, oy - 110];
    const Lk = lamp ? lampK : 0.4;
    const nv = buildShardVerts();
    gl.bindBuffer(gl.ARRAY_BUFFER, shardBuf);
    gl.bufferData(gl.ARRAY_BUFFER, sdata.subarray(0, nv * VSTRIDE), gl.DYNAMIC_DRAW);
    gl.bindVertexArray(shardVao);
    // shadows
    {
      const u = P.shadow.u;
      gl.useProgram(P.shadow.p); common(u);
      gl.uniform1f(u.uShadow, 1);
      const tAvg = (tint.tint[0] + tint.tint[1] + tint.tint[2]) / 3;
      gl.uniform1f(u.uShade, (0.1 + (1 - Math.pow(tAvg, tint.density * 4)) * 0.6 + tint.frost * 0.06) * (1 - lightK * 0.25));
      ghostUniforms(u);
      gl.uniform3f(u.uCaustic, ...tint.tint.map(v => Math.pow(v, tint.density * 3)));
      gl.drawArrays(gl.TRIANGLES, 0, nv);
    }
    {
      const u = P.glass.u;
      gl.useProgram(P.glass.p); common(u);
      gl.uniform1f(u.uShadow, 0);
      bindTex(0, bgTex); gl.uniform1i(u.uBgTex, 0);
      bindTex(1, textTex); gl.uniform1i(u.uText, 1);
      gl.uniform1f(u.uHasBg, surface ? 1 : 0); gl.uniform3f(u.uBgCol, ...bgCol); gl.uniform4f(u.uBg, ...bgOff);
      gl.uniform2f(u.uLamp, ...L); gl.uniform1f(u.uLampK, Lk);
      gl.uniform2f(u.uEye, ...eye); gl.uniform1f(u.uLight, lightK);
      gl.uniform3f(u.uTint, ...tint.tint); gl.uniform1f(u.uDensity, tint.density); gl.uniform1f(u.uFrost, tint.frost); gl.uniform1f(u.uStreaks, tint.streaks);
      gl.uniform3f(u.uEdgeCol, ...tint.edge);
      gl.uniform1f(u.uBevel, bevel); gl.uniform1f(u.uPress, press); gl.uniform2f(u.uPressAt, ...pressAt);
      gl.uniform3f(u.uInk, ...ink.rgb); gl.uniform1f(u.uInkDark, ink.dark ? 1 : 0); gl.uniform1f(u.uDisabled, disK);
      ghostUniforms(u);
      gl.drawArrays(gl.TRIANGLES, 0, nv);
    }
    if (state !== 'rest' && segs.length){
      const nc = buildCrackVerts();
      if (nc){
        gl.bindBuffer(gl.ARRAY_BUFFER, crackBuf);
        gl.bufferData(gl.ARRAY_BUFFER, cdata.subarray(0, nc * 10), gl.DYNAMIC_DRAW);
        const u = P.crack.u;
        gl.useProgram(P.crack.p); common(u);
        gl.uniform2f(u.uOrigin, ox, oy);
        gl.uniform2f(u.uLamp, ...L); gl.uniform1f(u.uLampK, Lk); gl.uniform2f(u.uEye, ...eye);
        gl.uniform1f(u.uCrackR, crackR); gl.uniform1f(u.uHealF, healF); gl.uniform1f(u.uHealOn, state === 'healing' ? 1 : 0);
        gl.uniform1f(u.uAlpha, crackA); gl.uniform1f(u.uLight, lightK);
        gl.bindVertexArray(crackVao); gl.drawArrays(gl.TRIANGLES, 0, nc);
      }
    }
    if (parts.length){
      if (pdata.length < parts.length * 6) pdata = new Float32Array(parts.length * 12);
      let m = 0;
      for (const p of parts){
        if (p.age < 0) continue;
        const life = 1 - p.age / p.life;
        let a;
        if (p.kind === 1) a = Math.sin(Math.PI * clamp(p.age / p.life, 0, 1));
        else a = Math.min(1, life * 2.5) * (0.35 + 0.65 * Math.pow(0.5 + 0.5 * Math.sin(p.tw + p.age * 40), 3));
        pdata.set([p.x + ox, p.y + p.z * PZ + oy, p.size * (p.kind === 1 ? 1 : 1), a, p.kind, 0], m * 6); m++;
      }
      if (m){
        gl.bindBuffer(gl.ARRAY_BUFFER, partBuf);
        gl.bufferData(gl.ARRAY_BUFFER, pdata.subarray(0, m * 6), gl.DYNAMIC_DRAW);
        gl.useProgram(P.part.p);
        gl.uniform2f(P.part.u.uStage, stageW, stageH); gl.uniform1f(P.part.u.uDpr, dpr);
        gl.bindVertexArray(partVao); gl.drawArrays(gl.POINTS, 0, m);
      }
    }
    gl.bindVertexArray(null);
    gl.disable(gl.BLEND);
  }

  let manual = false, frameCount = 0;
  function advance(dt){
    frameCount++;
    checkPlace();
    simT += dt;
    press += (pressTarget - press) * Math.min(1, dt * 18);
    disK += (disT - disK) * Math.min(1, dt * 10);
    if (Math.abs(disK - disT) < 0.004) disK = disT;
    lampK += (lampKT - lampK) * Math.min(1, dt * 6);
    if (state !== 'rest') stepPhases(dt);
    if (grab && grab.dragging) dragStep(dt);
    const moving = stepShards(dt);
    const partsLive = stepParts(dt);
    const settling = Math.abs(press - pressTarget) > 0.005 || Math.abs(lampK - lampKT) > 0.005 || disK !== disT;
    const busy = simT < activeUntil || moving || state !== 'rest' || !!grab || settling || partsLive;
    if (busy || dirty){ render(); dirty = false; }
    return busy;
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
  const ro = new ResizeObserver(() => { measureBounds(); lastRect = null; checkPlace(); });
  ro.observe(button, { box: 'border-box' });
  ro.observe(document.documentElement);
  if (document.body) ro.observe(document.body);
  let moTimer = 0;
  const syncDisabled = () => { const d = isDisabled() ? 1 : 0; if (d !== disT){ disT = d; wake(0.05); } };
  const mo = new MutationObserver(() => {
    syncDisabled();
    if (moTimer || raf || inFrame) return;
    moTimer = setTimeout(() => { moTimer = 0; if (!raf) checkPlace(); }, 300);
  });
  mo.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['class', 'style', 'hidden', 'open', 'disabled', 'aria-disabled'] });
  const onFonts = () => { if (lost) return; buildText(); dirty = true; lastRect = null; checkPlace(); wake(0.05); };
  document.fonts?.addEventListener?.('loadingdone', onFonts);
  const onScroll = () => { checkPlace(); };
  const onResize = () => { measureBounds(); lastRect = null; checkPlace(); };
  const onVis = () => { if (!document.hidden){ last = performance.now(); checkPlace(); wake(0.05); } };
  addEventListener('resize', onResize);
  document.addEventListener('scroll', onScroll, { capture: true, passive: true });
  document.addEventListener('visibilitychange', onVis);

  // A lost context shows the plain button until the GPU comes back; then the glass is rebuilt.
  function onLost(e){
    e.preventDefault();
    lost = true;
    cancelAnimationFrame(raf); raf = 0;
    grab = null; pressTarget = 0; setCursor(null); audioAwake(self, false);
    showButton();
    canvas.style.visibility = 'hidden';
  }
  function onRestored(){
    try { initGL(); } catch (err) { console.warn('ElementalFX glass: could not rebuild after context loss', err); return; }
    lost = false;
    hideButton();
    canvas.style.visibility = '';
    uploadSurface();
    bw = bh = 0; lastRect = null;
    measureBounds(); checkPlace();
    wake(0.1);
  }
  canvas.addEventListener('webglcontextlost', onLost);
  canvas.addEventListener('webglcontextrestored', onRestored);

  setTint(opts.tint ?? 'clear');
  uploadSurface();
  measureBounds();
  checkPlace();
  wake(0.1);

  return {
    supported: true,
    get state(){ return state; },
    get _frames(){ return frameCount; },
    get _shards(){ return shards.length; },
    get _asleep(){ return !raf && !inFrame; },
    get _lost(){ return lost; },
    get _segs(){ return segs.length; },
    get _pinned(){ return shards.filter(s => s.pinned).length; },
    get _ink(){ return ink; },
    get _floor(){ return { floorY, zMax, bh, ghost: ghostRects.length }; },
    _shardAt(i){ const s = shards.filter(s => s.mode === 'free')[i]; if (!s) return null; const P = screenOf(s); const r = canvas.getBoundingClientRect(); let x = 0, y = 0; for (const p of P){ x += p[0]; y += p[1]; } return [x / P.length + r.left, y / P.length + r.top]; },
    _extent(){ let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9; for (const s of shards) for (const [x, y] of screenOf(s)){ x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); } return { x0, y0, x1, y1, stageW, stageH }; },
    setTint,
    reset(){ startRestore(); },
    // the page repainted what lies under the glass: re-read it
    refresh(){ if (lost) return; uploadSurface(); lastRect = null; checkPlace(); wake(0.1); },
    detach(){
      cancelAnimationFrame(raf); raf = 0; manual = true; ro.disconnect(); mo.disconnect(); clearTimeout(moTimer);
      document.fonts?.removeEventListener?.('loadingdone', onFonts);
      document.removeEventListener('touchstart', onTouch, capNP); document.removeEventListener('touchmove', onTouch, capNP);
      document.removeEventListener('touchend', onTouch, true); document.removeEventListener('touchcancel', onTouch, true);
      canvas.removeEventListener('webglcontextlost', onLost); canvas.removeEventListener('webglcontextrestored', onRestored);
      removeEventListener('resize', onResize); document.removeEventListener('scroll', onScroll, { capture: true });
      document.removeEventListener('visibilitychange', onVis);
      document.removeEventListener('pointerdown', onDocDown, true);
      document.removeEventListener('click', onDocClick, true);
      document.removeEventListener('pointerout', onLeave, true);
      removeEventListener('pointermove', onDocMove); removeEventListener('pointerup', onDocUp, true); removeEventListener('pointercancel', onCancel, true);
      button.removeEventListener('click', onClick, true);
      setCursor(null);
      audioAwake(self, false);
      showButton();
      canvas.remove();
      gl.getExtension('WEBGL_lose_context')?.loseContext();
    },
    _manual(on){ manual = on; if (!on){ activeUntil = simT + 0.1; if (!raf && !inFrame){ last = performance.now(); raf = requestAnimationFrame(frame); } } },
    _advance(sec, fps = 60){ for (let i = 0, n = Math.round(sec * fps); i < n; i++) advance(1 / fps); },
    _lamp(x, y){ lamp = [x, y]; lampK = lampKT = 1; dirty = true; if (manual) render(); else wake(0.05); },
    _render(){ render(); },
  };
}

window.ElementalFX = Object.assign(window.ElementalFX || {}, { attachGlass, glassAudio });
})();
