import { polygon as turfPolygon, area as turfArea, booleanPointInPolygon, distance as turfDistance } from '@turf/turf';

// Simulated annealing for QUBO (custom, no ml-sa)
async function simulatedAnnealingQUBO(Q, n, candidates, budget, maxIter = 10000, T0 = 20, alpha = 0.999) {
  // Start with no grids
  let x = Array(n).fill(0);
  let bestX = x.slice(), bestE = Infinity;
  let T = T0;
  // Use the same costPenalty as in handler
  const costPenalty = 10000 / budget;
  function energy(x) {
    let e = 0, cost = 0;
    for (let i = 0; i < n; i++) {
      if (x[i]) cost += candidates[i].cost;
      for (let j = 0; j < n; j++) e += Q[i][j] * x[i] * x[j];
    }
    if (cost > budget) e += costPenalty * (cost - budget) ** 2;
    return e;
  }
  for (let iter = 0; iter < maxIter; iter++) {
    // Flip a random bit
    let i = Math.floor(Math.random() * n);
    let x2 = x.slice(); x2[i] = 1 - x2[i];
    let e1 = energy(x), e2 = energy(x2);
    if (e2 < e1 || Math.random() < Math.exp((e1 - e2) / T)) {
      x = x2;
      if (e2 < bestE) { bestE = e2; bestX = x.slice(); }
    }
    T *= alpha;
    if (T < 1e-6) break;
  }
  return bestX;
}

// NAR (greedy) optimization: pick as many candidates as possible with best energy/cost ratio under budget
function narGreedyBudget(Q, candidates, budget) {
  // Sort by energy/cost ratio, pick until budget is reached
  const idxs = candidates.map((c, i) => i)
    .sort((a, b) => (candidates[b].energy / candidates[b].cost) - (candidates[a].energy / candidates[a].cost));
  let total = 0;
  let selected = [];
  for (let i of idxs) {
    if (total + candidates[i].cost <= budget) {
      selected.push(i);
      total += candidates[i].cost;
    }
  }
  return candidates.map((_, i) => selected.includes(i) ? 1 : 0);
}

// Gurobi QUBO solver (budget constraint)
async function gurobiQUBOBudget(Q, candidates, budget) {
  try {
    const { spawnSync } = await import('child_process');
    const fs = await import('fs');
    const path = await import('path');
    const tmpDir = '/tmp';
    const quboPath = path.join(tmpDir, `qubo_gurobi_${Date.now()}.json`);
    const costsPath = path.join(tmpDir, `costs_gurobi_${Date.now()}.json`);
    fs.writeFileSync(quboPath, JSON.stringify({ Q }));
    fs.writeFileSync(costsPath, JSON.stringify(candidates.map(c => c.cost)));
    const pyPath = path.join(process.cwd(), 'gurobi_optimize.py');
    const result = spawnSync('python3', [pyPath, '--qubo_file', quboPath, '--budget', String(budget), '--costs_file', costsPath], { encoding: 'utf-8' });
    fs.unlinkSync(quboPath);
    fs.unlinkSync(costsPath);
    if (result.status !== 0) throw new Error(result.stderr || 'Gurobi failed');
    const out = result.stdout.trim().split('\n').pop();
    const parsed = JSON.parse(out);
    const x = candidates.map((_, i) => parsed.selected_indices.includes(i) ? 1 : 0);
    return x;
  } catch (e) {
    throw new Error('Gurobi optimization failed: ' + e.message);
  }
}

// Simulated annealing QUBO solver using Python simanneal
async function saPythonQUBO(Q, candidates, budget) {
  const { spawnSync } = await import('child_process');
  const fs = await import('fs');
  const path = await import('path');
  const tmpDir = '/tmp';
  const quboPath = path.join(tmpDir, `qubo_sa_${Date.now()}.json`);
  const costsPath = path.join(tmpDir, `costs_sa_${Date.now()}.json`);
  fs.writeFileSync(quboPath, JSON.stringify({ Q }));
  fs.writeFileSync(costsPath, JSON.stringify(candidates.map(c => c.cost)));
  const pyPath = path.join(process.cwd(), 'sa_optimize.py');
  const result = spawnSync('python3', [pyPath, '--qubo_file', quboPath, '--budget', String(budget), '--costs_file', costsPath], { encoding: 'utf-8' });
  fs.unlinkSync(quboPath);
  fs.unlinkSync(costsPath);
  if (result.status !== 0) throw new Error(result.stderr || 'Simulated annealing failed');
  const out = result.stdout.trim().split('\n').pop();
  const parsed = JSON.parse(out);
  const x = candidates.map((_, i) => parsed.selected_indices.includes(i) ? 1 : 0);
  return x;
}

export default async function handler(req, res) {
  const { polygon, budget, gridArea, cellSize, country, popCenter, quboOnly, algo } = req.body;

  if (!polygon || polygon.length !== 4 || !budget || budget <= 0 || gridArea <= 0 || !cellSize || !country || !popCenter) {
    return res.status(400).json({ error: 'Invalid input' });
  }

  // Close the polygon
  const coords = [...polygon.map(p => [p.lng, p.lat]), [polygon[0].lng, polygon[0].lat]];
  const turfPoly = turfPolygon([coords]);
  const totalArea = turfArea(turfPoly); // m^2

  // Generate candidate grid centers (regular mesh inside polygon)
  const nCandidates = Math.max(20, Math.floor(budget / cellSize.price) * 5);
  let candidates = [];
  const [minLng, minLat, maxLng, maxLat] = [
    Math.min(...coords.map(c=>c[0])), Math.min(...coords.map(c=>c[1])),
    Math.max(...coords.map(c=>c[0])), Math.max(...coords.map(c=>c[1]))
  ];
  const steps = Math.ceil(Math.sqrt(nCandidates));
  for (let i = 0; i < steps; i++) {
    for (let j = 0; j < steps; j++) {
      const lat = minLat + (maxLat - minLat) * (i + 0.5) / steps;
      const lng = minLng + (maxLng - minLng) * (j + 0.5) / steps;
      if (booleanPointInPolygon([lng, lat], turfPoly)) {
        // Calculate distance to population center (in km)
        const dist = turfDistance([lng, lat], [popCenter.lng, popCenter.lat], { units: 'kilometers' });
        candidates.push({ lat, lng, cost: cellSize.price, energy: cellSize.energy, dist });
      }
    }
  }
  if (candidates.length === 0) {
    return res.status(200).json({ grids: [] });
  }

  // QUBO: minimize total cost, maximize population served, with penalties for budget, grid count, and minimum population coverage
  // Q(x) = sum_i C_i x_i - alpha * sum_i P_i x_i + theta (sum_i C_i x_i - budget)^2 + mu (sum_i x_i - K)^2 + lambda (M - sum_i P_i x_i)^2
  const alpha = 1;      // trade-off weight for population coverage
  const theta = 1000;   // penalty for budget overflow
  const mu = 1000;      // penalty for grid count
  const lambda = 1000;  // penalty for minimum coverage

  // User must provide K (max grids) and M (min population to cover) in request, else set defaults
  const K = req.body.K ?? Math.floor(budget / cellSize.price); // default: max possible
  const M = req.body.M ?? 0; // default: no minimum

  // Extract C_i and P_i from candidates
  const C = candidates.map(c => c.cost);
  const P = candidates.map(c => c.pop ?? 1); // fallback: 1 if no pop field

  const n = candidates.length;
  const Q = Array.from({ length: n }, (_, i) => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    // Linear terms
    Q[i][i] += C[i];
    Q[i][i] += theta * (C[i] ** 2);
    Q[i][i] += mu;
    Q[i][i] += lambda * (P[i] ** 2);
    Q[i][i] -= alpha * P[i];
    Q[i][i] += -2 * theta * budget * C[i];
    Q[i][i] += -2 * mu * K;
    Q[i][i] += -2 * lambda * M * P[i];
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      // Quadratic terms
      Q[i][j] += 2 * theta * C[i] * C[j];
      Q[i][j] += 2 * mu;
      Q[i][j] += 2 * lambda * P[i] * P[j];
    }
  }
  // Add constant terms to Q[0][0] (does not affect optimization, but for completeness)
  Q[0][0] += theta * (budget ** 2) + mu * (K ** 2) + lambda * (M ** 2);

  // For quantum: return QUBO and candidates (limit size)
  if (quboOnly) {
    const maxCandidates = 1000;
    if (candidates.length > maxCandidates) {
      const step = Math.ceil(candidates.length / maxCandidates);
      const reducedCandidates = candidates.filter((_, i) => i % step === 0).slice(0, maxCandidates);
      const reducedQ = reducedCandidates.map((_, i) => reducedCandidates.map((_, j) => Q[i * step]?.[j * step] ?? 0));
      return res.status(200).json({ Q: reducedQ, candidates: reducedCandidates, budget });
    }
    return res.status(200).json({ Q, candidates, budget });
  }

  // Simulated annealing and greedy: maximize energy under budget
  let x;
  try {
    if (algo === 'nar') {
      x = narGreedyBudget(Q, candidates, budget);
    } else if (algo === 'gurobi') {
      x = await gurobiQUBOBudget(Q, candidates, budget);
    } else {
      x = await simulatedAnnealingQUBO(Q, candidates.length, candidates, budget);
    }
  } catch (e) {
    console.error('Optimization error:', e);
    return res.status(500).json({ error: e.message || e.toString() });
  }
  const grids = candidates.filter((c, i) => x[i]).map(c => ({
    lat: c.lat,
    lng: c.lng,
    cost: c.cost,
    energy: c.energy,
    dist: c.dist
  }));
  console.log('Returned grids:', grids.length);
  res.status(200).json({ grids });
}