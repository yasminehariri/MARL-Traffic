// Data access for the results website. Every number on the site must come through this module, which
// reads only the generated files in site/data/ (made by scripts/export_site_data.py; format in
// site/data/README.md). Never type a result into a page by hand.
//
//   import { loadData, one, pick, fmt } from './js/data.js';
//   const D = await loadData();
//   const e1 = one(D.summary, { experiment: 'normal', controller: 'E1', mode: 'greedy', scenario: 'peak' });
//   el.textContent = fmt.delay(e1.true_delay_mean);           // "92.1"
//
// Filters default to the main model (train_seed 0) and the 4x4 grid unless you pass those keys.

const FILES = ['summary', 'comparisons', 'training', 'messages', 'meta', 'episodes', 'ambulance'];

export async function loadData(base = new URL('../data/', import.meta.url).href) {
  const got = await Promise.all(FILES.map(async f => {
    const res = await fetch(base + f + '.json');
    if (!res.ok) throw new Error(`site/data/${f}.json could not be loaded (${res.status}). Run scripts/export_site_data.py.`);
    return res.json();
  }));
  const D = Object.fromEntries(FILES.map((f, i) => [f, got[i]]));
  const { columns, rows } = D.episodes;                      // compact table -> list of objects
  D.episodes = rows.map(r => Object.fromEntries(columns.map((c, i) => [c, r[i]])));
  return D;
}

const DEFAULTS = { grid: '4x4', train_seed: 0 };
const matches = (row, filter) => Object.entries({ ...DEFAULTS, ...filter })
  .every(([k, v]) => !(k in row) || (Array.isArray(v) ? v.includes(row[k]) : row[k] === v));

/** All rows matching the filter (a value can be a list of allowed values). */
export const pick = (rows, filter = {}) => rows.filter(r => matches(r, filter));

/** Exactly one matching row; throws if there are none or several, so a wrong filter is never silent. */
export function one(rows, filter) {
  const m = pick(rows, filter);
  if (m.length !== 1) throw new Error(`expected 1 row for ${JSON.stringify(filter)}, found ${m.length}`);
  return m[0];
}

/** Mode filter for one learned-controller mode plus the controllers that have no mode (fixed, tuned, random). */
export const withBaselines = mode => [mode, 'none'];

export const fmt = {
  delay: x => (x == null ? '–' : x.toFixed(1)),                       // seconds per vehicle
  pct: x => (x == null ? '–' : `${Math.round(Math.abs(x))}%`),        // use with your own "lower"/"higher"
  p: p => (p == null ? '–' : p < 0.001 ? 'p < 0.001' : `p = ${p.toFixed(3)}`),
  signif: p => p != null && p < 0.05,
};

/** Display name of a controller id ("E1" -> "E1: learned communication"). */
export const nameOf = (D, id) => D.meta.controllers[id] || id;

// Chart colours: one fixed colour per controller (the reference palette, checked for colour-blind
// separation). Same entity, same colour, on every chart. Light and dark variants of the same hues.
export const COLORS = {
  light: { E1: '#2a78d6', B1: '#eb6834', B2: '#1baf7a', tuned: '#eda100', fixed: '#e87ba4', random: '#6250d6', E1_from2x2: '#008300' },
  dark:  { E1: '#3987e5', B1: '#d95926', B2: '#199e70', tuned: '#c98500', fixed: '#d55181', random: '#9085e9', E1_from2x2: '#0ca30c' },
};
export const colorOf = id => COLORS[matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'][id] || '#888';
/** Order controllers are shown in, everywhere. */
export const ORDER = ['E1', 'B1', 'B2', 'tuned', 'fixed', 'random'];
