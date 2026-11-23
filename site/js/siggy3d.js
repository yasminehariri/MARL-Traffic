// Siggy, the 3D robot mascot. He walks over the page in a transparent, click-through canvas.
// Three.js comes from the local demo/vendor/three files through the import map in index.html.
//
//   import { startSiggy } from './js/siggy3d.js';
//   startSiggy({ messages: { problem: '...', ... }, hello: 'Hi!', helloDelay: 900 });
//
// There is no 3D file of Siggy in the repo, so he is built from primitives to match the Siggy
// character sheet (front, back, sides) and assets/siggy.png.

import * as THREE from 'three';

const ORANGE = 0xf2711c, ORANGE_DARK = 0xd95f12, CHEEK = 0xf58a2e, INK = 0x1d1f22, CREAM = 0xf4e4c9;

/* ---------- model ---------- */
// The design follows the Siggy character sheet: a soft orange body with a black face panel
// (cream eyes, orange cheeks, cream smile), a black chest panel with red / yellow / green buttons,
// chunky arms, two stubby legs, and "MARL" / "2026" panels pressed into his back.

/** Rounded rectangle centred on the origin, with separate top and bottom corner radii. */
function roundRect(w, h, rTop, rBot = rTop) {
  const s = new THREE.Shape(), x = -w / 2, y = -h / 2;
  s.moveTo(x + rBot, y); s.lineTo(x + w - rBot, y); s.quadraticCurveTo(x + w, y, x + w, y + rBot);
  s.lineTo(x + w, y + h - rTop); s.quadraticCurveTo(x + w, y + h, x + w - rTop, y + h);
  s.lineTo(x + rTop, y + h); s.quadraticCurveTo(x, y + h, x, y + h - rTop);
  s.lineTo(x, y + rBot); s.quadraticCurveTo(x, y, x + rBot, y);
  return s;
}
/** A raised rounded frame (outer outline minus inner outline) with a softly bevelled edge. */
function frameGeo(ow, oh, or, iw, ih, ir, depth = 0.025) {
  const s = roundRect(ow, oh, or); s.holes.push(roundRect(iw, ih, ir));
  return new THREE.ExtrudeGeometry(s, { depth, bevelEnabled: true, bevelThickness: 0.018, bevelSize: 0.018, bevelSegments: 3, curveSegments: 18 });
}
const plateGeo = (w, h, r, depth = 0.02) =>
  new THREE.ExtrudeGeometry(roundRect(w, h, r), { depth, bevelEnabled: true, bevelThickness: 0.012, bevelSize: 0.012, bevelSegments: 3, curveSegments: 18 });

/** Engraved lettering for the back panels, drawn into a small transparent texture. */
function engraving(text, w, h, color) {
  const c = document.createElement('canvas'); c.width = 256; c.height = Math.round(256 * h / w);
  const g = c.getContext('2d');
  g.fillStyle = color; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.font = `800 ${Math.round(c.height * 0.62)}px system-ui, "Segoe UI", Arial, sans-serif`;
  g.fillText(text, c.width / 2, c.height * 0.54);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 4;
  return new THREE.Mesh(new THREE.PlaneGeometry(w, h),
    new THREE.MeshStandardMaterial({ map: tex, transparent: true, roughness: 0.8, depthWrite: false }));
}

// Model units: feet at y=0, body top at about 1.9. Front faces +z.
export function buildSiggy() {
  const mat = (c, rough = 0.72) => new THREE.MeshStandardMaterial({ color: c, roughness: rough, metalness: 0 });
  const body = mat(ORANGE), ink = mat(INK, 0.85), cream = mat(CREAM, 0.6), cheek = mat(CHEEK, 0.7), dark = mat(ORANGE_DARK);
  const root = new THREE.Group();          // position + facing
  const bob = new THREE.Group();           // walk bob / hop
  root.add(bob);

  const LEG_H = 0.24, W = 1.3, H = 1.66, D = 1.0, BEV = 0.17, BEVT = 0.3;
  const FRONT = D / 2, BACK = -D / 2;

  // torso: one tall soft block, domed on top, rounded on every edge (head and body are one shape)
  const prof = roundRect(W - 2 * BEV, H - 2 * BEV, 0.4, 0.12);
  const torsoGeo = new THREE.ExtrudeGeometry(prof, { depth: D - 2 * BEVT, bevelEnabled: true, bevelThickness: BEVT, bevelSize: BEV, bevelSegments: 10, curveSegments: 28 });
  torsoGeo.translate(0, 0, -(D - 2 * BEVT) / 2);
  torsoGeo.computeVertexNormals();
  const torso = new THREE.Mesh(torsoGeo, body);
  torso.position.y = LEG_H + H / 2;
  bob.add(torso);

  // face: a black panel set into the front, framed by a thin raised orange lip
  const faceY = LEG_H + 1.2;
  const lip = new THREE.Mesh(frameGeo(0.92, 0.57, 0.21, 0.85, 0.5, 0.18), body);
  lip.position.set(0, faceY, FRONT - 0.02);
  const face = new THREE.Mesh(plateGeo(0.85, 0.5, 0.18), ink);
  face.position.set(0, faceY, FRONT - 0.02);
  bob.add(lip, face);
  const FZ = FRONT + 0.012;                // the face panel's surface

  // eyes: cream discs with a black pupil and a white catch-light (grouped so they can blink)
  const eyes = [-0.21, 0.21].map(x => {
    const g = new THREE.Group();
    const white = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.03, 28), cream);
    white.rotation.x = Math.PI / 2;
    const pupil = new THREE.Mesh(new THREE.CylinderGeometry(0.058, 0.058, 0.03, 24), ink);
    pupil.rotation.x = Math.PI / 2; pupil.position.set(0.012, -0.005, 0.012);
    const glint = new THREE.Mesh(new THREE.CircleGeometry(0.018, 12), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    glint.position.set(0.03, 0.025, 0.03);
    g.add(white, pupil, glint);
    g.position.set(x, faceY + 0.07, FZ + 0.01);
    bob.add(g);
    return g;
  });
  // smile: a cream arc; cheeks: two orange dots
  const smile = new THREE.Mesh(new THREE.TorusGeometry(0.13, 0.026, 10, 32, Math.PI * 0.8), cream);
  smile.rotation.z = Math.PI + Math.PI * 0.1;
  smile.position.set(0, faceY - 0.02, FZ + 0.012);
  bob.add(smile);
  [-0.33, 0.33].forEach(x => {
    const c = new THREE.Mesh(new THREE.CylinderGeometry(0.058, 0.058, 0.02, 24), cheek);
    c.rotation.x = Math.PI / 2; c.position.set(x, faceY - 0.09, FZ + 0.006);
    bob.add(c);
  });

  // chest: a black traffic-light panel with three round buttons (they glow and cycle when idle)
  const chestY = LEG_H + 0.56;
  const chest = new THREE.Mesh(plateGeo(0.3, 0.6, 0.08), ink);
  chest.position.set(0, chestY, FRONT - 0.012);
  bob.add(chest);
  const lamps = [0xe52a2a, 0xf7cc12, 0x1fae3c].map((c, i) => {
    const m = new THREE.MeshStandardMaterial({ color: c, emissive: c, emissiveIntensity: 0.35, roughness: 0.35 });
    const l = new THREE.Mesh(new THREE.SphereGeometry(0.075, 20, 14), m);
    l.scale.z = 0.5;
    l.position.set(0, chestY + 0.17 - i * 0.17, FRONT + 0.02);
    bob.add(l);
    return m;
  });

  // back: "MARL" and "2026" panels pressed into the plastic
  const backPanels = [[0.72, 0.25, LEG_H + 1.22, 'MARL'], [0.72, 0.68, LEG_H + 0.76, '2026']];   // kept on the flat part of the back
  backPanels.forEach(([w, h, y, text], i) => {
    const f = new THREE.Mesh(frameGeo(w, h, 0.09, w - 0.05, h - 0.05, 0.07, 0.012), dark);
    f.rotation.y = Math.PI; f.position.set(0, y, BACK + 0.024);   // sits almost flush, like a pressed-in panel
    const t = engraving(text, w * 0.8, 0.2, '#c4520c');
    t.rotation.y = Math.PI; t.position.set(0, i ? y + h / 2 - 0.2 : y, BACK - 0.004);
    bob.add(f, t);
  });

  // chunky rounded arms hanging at the sides, pivoting at the shoulder
  const armGeo = new THREE.CapsuleGeometry(0.15, 0.4, 8, 16);
  const arms = [-1, 1].map(side => {
    const pivot = new THREE.Group();
    pivot.position.set(side * 0.7, LEG_H + 1.0, 0.02);
    const a = new THREE.Mesh(armGeo, body);
    a.position.y = -0.3;
    pivot.add(a);
    pivot.rotation.z = side * 0.14;
    bob.add(pivot);
    return pivot;
  });

  // two short stubby legs, pivoting at the hip
  const legGeo = new THREE.CapsuleGeometry(0.16, 0.08, 8, 16);
  const legs = [-1, 1].map(side => {
    const pivot = new THREE.Group();
    pivot.position.set(side * 0.27, LEG_H + 0.08, 0);
    const l = new THREE.Mesh(legGeo, body);
    l.position.y = -0.12;
    l.scale.z = 1.15;
    pivot.add(l);
    root.add(pivot);
    return pivot;
  });

  // soft contact shadow (stays on the ground while he bobs)
  const shadow = new THREE.Mesh(new THREE.CircleGeometry(0.72, 32),
    new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.16, depthWrite: false }));
  shadow.scale.y = 0.18;
  shadow.position.set(0, 0.02, -0.8);
  root.add(shadow);

  return { root, bob, eyes, lamps, arms, legs, shadow, height: 1.95 };
}

/* ---------- helpers ---------- */
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;
const damp = (a, b, rate, dt) => lerp(a, b, 1 - Math.exp(-rate * dt));
const rand = (a, b) => a + Math.random() * (b - a);

/** Visible right edge of a heading's text (headings are full-width blocks). */
function textRight(el) {
  const r = document.createRange(); r.selectNodeContents(el);
  const rects = r.getClientRects();
  let right = 0; for (const x of rects) right = Math.max(right, x.right);
  return right || el.getBoundingClientRect().left;
}

export function startSiggy({ messages = {}, hello = '', helloDelay = 900 } = {}) {
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* renderer + scene */
  const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: 'low-power' });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const cv = renderer.domElement;
  cv.className = 'siggy-canvas';
  cv.setAttribute('aria-hidden', 'true');
  document.body.append(cv);

  const bubble = document.createElement('div');
  bubble.className = 'siggy-bubble';
  bubble.setAttribute('aria-hidden', 'true');
  document.body.append(bubble);

  const scene = new THREE.Scene();
  // Orthographic camera in CSS pixels: x right, y up, origin at the bottom-left of the window.
  const camera = new THREE.OrthographicCamera(0, 1, 1, 0, -3000, 3000);
  camera.position.z = 1000;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a6a55, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 2.1);
  sun.position.set(-0.6, 1, 1.2);
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0xffe2cc, 0.6);
  fill.position.set(1, 0.2, 0.6);
  scene.add(fill);

  const S = buildSiggy();
  scene.add(S.root);
  S.root.rotation.x = 0.1;               // a slight top-down tilt so he reads as 3D

  let W = 0, H = 0, size = 100, phone = false;
  function resize() {
    W = innerWidth; H = innerHeight; phone = W < 700;
    size = phone ? 62 : clamp(W * 0.075, 84, 112);
    renderer.setSize(W, H, false);
    camera.left = 0; camera.right = W; camera.top = H; camera.bottom = 0;
    camera.updateProjectionMatrix();
    S.root.scale.setScalar(size / S.height);
    if (reduce) placeStill();
  }

  /* page anchors */
  const sections = [...document.querySelectorAll('main section[id]')];
  const street = document.querySelector('.street');
  const ambroad = document.querySelector('.ambroad');
  const amb = document.querySelector('.amb');
  const nav = document.querySelector('header.nav');
  let current = 'top';

  /** Where Siggy should be right now: a ground line (viewport y) and a horizontal range. */
  function surface() {
    const navH = nav ? nav.getBoundingClientRect().bottom : 0;
    const half = size * 0.36, minY = navH + size + 8, maxY = H - 6;
    const roadOn = (el, laneFromBottom) => {
      if (!el) return null;
      const r = el.getBoundingClientRect(), y = r.bottom - laneFromBottom;
      if (y < minY || y > maxY) return null;
      return { key: 'road:' + current, y, lo: r.left + half + 6, hi: r.right - half - 6, stop: null };
    };
    if (current === 'top') { const s = roadOn(street, 18); if (s) return s; }
    if (current === 'ambulance') { const s = roadOn(ambroad, 30); if (s) { s.road = true; return s; } }
    const sec = document.getElementById(current), h = sec && sec.querySelector('h2');
    if (h && !phone) {
      const r = h.getBoundingClientRect(), tr = textRight(h);
      const lo = tr + 34 + half, hi = W - 18 - half;
      if (r.bottom > minY && r.bottom < maxY && hi - lo > 150) return { key: 'h:' + current, y: r.bottom, lo, hi, stop: lo + 8 };
    }
    // otherwise stand at the bottom, in the right-hand margin when there is one
    const wrap = document.querySelector('main.wrap');
    const margin = wrap ? W - wrap.getBoundingClientRect().right : 0;
    const lo = margin > size + 40 ? W - margin + half + 12 : W - (phone ? 120 : 190);
    return { key: 'bottom', y: maxY, lo: Math.min(lo, W - 18 - half - 20), hi: W - 14 - half, stop: W - 14 - half - (phone ? 10 : 30) };
  }

  /* state */
  const st = {
    x: -60, y: H, dir: 1, speed: 0, yaw: 0.95, phase: 0,
    mode: 'walk',            // walk | goto | wave | idle | dodge
    t: 0, target: null, surfKey: '', hop: null,
    nextIdle: rand(5, 9), nextBlink: rand(2, 4), blink: 0,
    bubbleUntil: 0, lampCycle: false, dodgeY: 0,
  };
  const mouse = { x: -1, y: -1 };
  addEventListener('pointermove', e => { mouse.x = e.clientX; mouse.y = e.clientY; }, { passive: true });

  function say(text, secs) {
    if (!text) return;
    bubble.textContent = text;
    bubble.classList.add('show');
    st.bubbleUntil = performance.now() / 1000 + secs;
  }

  /* section changes (same trigger band as the nav highlight) */
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting || e.target.id === current) return;
    current = e.target.id;
    if (reduce) { say(messages[current] || hello, 1e9); placeStill(); return; }
    st.mode = 'goto'; st.target = null; st.t = 0;
  }), { rootMargin: '-45% 0px -50% 0px' });
  sections.forEach(s => io.observe(s));

  /* reduced motion: stand still in the corner, facing the reader */
  function placeStill() {
    const half = size * 0.36;
    S.root.position.set(W - half - 18, 8, 0);
    S.root.rotation.y = 0;
    S.legs.forEach(l => (l.rotation.x = 0));
    S.arms.forEach(a => (a.rotation.x = 0));
    positionBubble(W - half - 18, H - 8);
    renderer.render(scene, camera);
  }

  function positionBubble(cx, footY) {
    const bw = bubble.offsetWidth, bh = bubble.offsetHeight;
    let left, top;
    if (phone) {
      // on phones, talk from beside him (whichever side has room), so the bubble doesn't cover buttons or text above
      const right = cx + size * 0.45, leftSide = cx - size * 0.45 - bw;
      left = right + bw <= W - 8 ? right : Math.max(8, leftSide);
      top = clamp(footY - size * 0.85 - bh / 2, 66, H - bh - 8);
    } else {
      left = clamp(cx - bw + size * 0.25, 8, W - bw - 8);
      top = clamp(footY - size - bh - 6, 66, H - bh - 8);
    }
    bubble.style.transform = `translate(${left}px, ${top}px)`;
  }

  /* ---------- animation loop ---------- */
  let last = performance.now(), raf = 0, running = false;
  function frame(now) {
    raf = requestAnimationFrame(frame);
    const dt = Math.min((now - last) / 1000, 0.05); last = now;
    const time = now / 1000;
    const surf = surface();
    const walkV = phone ? 60 : 92;

    // a new ground line: hop over to it instead of snapping
    if (surf.key !== st.surfKey) {
      if (st.surfKey) st.hop = { fromY: st.y, t: 0 };
      st.surfKey = surf.key;
      if (st.mode === 'walk' || st.mode === 'idle') st.mode = 'walk';
    }
    let groundY = surf.y;
    if (st.mode === 'dodge') groundY = surf.y - st.dodgeY;

    let wantV = 0, wantYaw = st.dir * 0.95, waving = false, look = false;
    st.t += dt;

    // ambulance: run out of the road when it comes close
    if (surf.road && amb && st.mode !== 'dodge') {
      const a = amb.getBoundingClientRect();
      if (a.right > st.x - size * 2.4 && a.left < st.x + size * 0.5) {
        st.mode = 'dodge'; st.t = 0; st.dodgeY = 0; st.dir = 1;
      }
    }

    switch (st.mode) {
      case 'goto': {
        const tx = surf.stop ?? clamp(st.x, surf.lo, surf.hi);
        const dx = tx - st.x;
        if (Math.abs(dx) < 4) {
          st.mode = 'wave'; st.t = 0;
          say(messages[current] || hello, 4.2);
        } else { st.dir = Math.sign(dx); wantV = (Math.abs(dx) > 260 ? 1.9 : 1.15) * walkV; }
        break;
      }
      case 'wave':
        wantYaw = 0; waving = true;
        if (st.t > 2.4) { st.mode = 'walk'; st.t = 0; st.nextIdle = rand(5, 9); }
        break;
      case 'idle':
        wantYaw = 0; look = true; st.lampCycle = true;
        if (st.t > st.idleFor) { st.mode = 'walk'; st.t = 0; st.lampCycle = false; st.nextIdle = rand(6, 11); }
        break;
      case 'dodge': {
        // run ahead and step up off the road, wait for the ambulance to pass, then hop back down
        const a = amb.getBoundingClientRect();
        const passed = a.left > st.x + size * 0.6 || a.right < -50 || st.t > 6;
        st.dodgeY = damp(st.dodgeY, passed ? 0 : size * 0.62, 7, dt);   // just clear of the ambulance roof
        wantV = passed ? 0 : walkV * 2.8; st.dir = 1; waving = !passed;
        if (st.x > surf.hi) wantV = 0;
        if (passed && st.dodgeY < 2) { st.mode = 'walk'; st.t = 0; }
        break;
      }
      default: { // walk: patrol the range, turning at the ends (the screen edge on most surfaces)
        const outside = st.x < surf.lo - 2 || st.x > surf.hi + 2;
        if (st.x <= surf.lo + 1) st.dir = 1;
        if (st.x >= surf.hi - 1) st.dir = -1;
        wantV = outside ? walkV * 2 : walkV;   // hurry back if the ground moved away from him
        st.nextIdle -= dt;
        if (!outside && st.nextIdle <= 0) { st.mode = 'idle'; st.t = 0; st.idleFor = rand(2.5, 4.5); }
      }
    }

    st.speed = damp(st.speed, wantV, 8, dt);
    const wasInside = st.x >= surf.lo && st.x <= surf.hi;
    st.x += st.dir * st.speed * dt;
    if (wasInside && st.mode === 'walk') st.x = clamp(st.x, surf.lo, surf.hi);
    st.x = clamp(st.x, -size, W + size);

    // vertical: follow the ground exactly (it scrolls with the page); hop when it changes
    let y = groundY, arc = 0;
    if (st.hop) {
      st.hop.t += dt / 0.45;
      const k = Math.min(st.hop.t, 1), e = k * k * (3 - 2 * k);
      y = lerp(st.hop.fromY, groundY, e); arc = Math.sin(Math.PI * k) * size * 0.35;
      if (k >= 1) st.hop = null;
    }
    st.y = y;

    // facing: turn toward travel direction, or toward the reader / mouse when stopped
    if (look && mouse.x >= 0) wantYaw = clamp((mouse.x - st.x) / W * 2.2, -0.7, 0.7);
    st.yaw = damp(st.yaw, wantYaw, 9, dt);

    // walk cycle
    const moving = st.speed / walkV;
    st.phase += dt * (5.5 + 4 * Math.min(moving, 2.5)) * Math.min(moving * 1.4, 1);
    const swing = Math.sin(st.phase) * Math.min(moving, 1.3) * 0.6;
    S.legs[0].rotation.x = swing; S.legs[1].rotation.x = -swing;
    S.arms[0].rotation.x = -swing * 1.1;
    S.arms[1].rotation.x = swing * 1.1;
    S.arms[1].rotation.z = 0.12;
    if (waving) { // right arm up, waving
      S.arms[1].rotation.x = 0;
      S.arms[1].rotation.z = damp(S.arms[1].rotation.z, 2.5 + Math.sin(time * 12) * 0.35, 20, dt);
      if (st.mode === 'dodge') S.arms[0].rotation.z = -2.4 - Math.sin(time * 12) * 0.3;
    } else S.arms[0].rotation.z = -0.12;
    const bobAmt = Math.abs(Math.sin(st.phase)) * Math.min(moving, 1.3) * 0.07;
    S.bob.position.y = bobAmt + (st.mode === 'idle' ? Math.sin(time * 2.2) * 0.015 : 0);
    S.bob.rotation.z = Math.sin(st.phase) * Math.min(moving, 1) * 0.05;
    S.bob.rotation.x = look && mouse.y >= 0 ? clamp((mouse.y - (H - st.y - size)) / H, -0.25, 0.25) * 0.5 : 0;

    // blink
    st.nextBlink -= dt;
    if (st.nextBlink <= 0) { st.blink = 0.14; st.nextBlink = rand(2.2, 5); }
    st.blink = Math.max(0, st.blink - dt);
    const eyeY = st.blink > 0 ? 0.12 : 1;
    const lookX = look && mouse.x >= 0 ? clamp((mouse.x - st.x) / W, -0.5, 0.5) * 0.06 : 0;
    S.eyes.forEach((e, i) => { e.scale.y = eyeY; e.position.x = (i ? 0.21 : -0.21) + lookX; });

    // chest light: all on while walking; cycles red -> yellow -> green while idling
    const active = Math.floor(time / 0.6) % 3;
    S.lamps.forEach((m, i) => { m.emissiveIntensity = st.lampCycle ? (i === active ? 1.6 : 0.06) : 0.35; });

    // stepping off the road reads as stepping back: shrink a little while dodging
    S.root.scale.setScalar(size / S.height * (1 - 0.2 * Math.min(st.dodgeY / (size * 0.62), 1)));
    S.root.position.set(st.x, H - st.y + arc, 0);
    S.root.rotation.y = st.yaw;
    S.shadow.material.opacity = 0.16 * (1 - Math.min(arc / (size * 0.35), 1) * 0.6);

    if (bubble.classList.contains('show')) {
      if (time > st.bubbleUntil) bubble.classList.remove('show');
      positionBubble(st.x, st.y - arc);
    }
    renderer.render(scene, camera);
  }

  function start() { if (running || reduce) return; running = true; last = performance.now(); raf = requestAnimationFrame(frame); }
  function stop() { running = false; cancelAnimationFrame(raf); }
  document.addEventListener('visibilitychange', () => (document.hidden ? stop() : start()));
  addEventListener('resize', resize);

  resize();
  // begin on the hero road, walking in from the left, and say hello
  st.y = surface().y; st.surfKey = surface().key; st.x = surface().lo;
  if (reduce) { say(hello, 1e9); placeStill(); } else setTimeout(() => say(hello, 4), helloDelay);
  if (!document.hidden) start();
  return { stop };
}
