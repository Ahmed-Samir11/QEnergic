import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import os from 'os';

export const config = {
  api: {
    bodyParser: {
      sizeLimit: '2mb',
    },
    responseLimit: false,
  },
};

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });
  const { Q, k, candidates } = req.body;
  if (!Q || !k || !candidates) return res.status(400).json({ error: 'Missing Q, k, or candidates' });

  const isDev = process.env.NODE_ENV !== 'production';

  const redact = (text) => {
    if (!text) return text;
    // Best-effort redaction for common secret/identifier patterns.
    return String(text)
      .replace(/crn:v1:[^\s"']+/gi, '[REDACTED_CRN]')
      .replace(/\b(QISKIT_IBM_TOKEN|IBM_QUANTUM_INSTANCE_CRN)\b\s*[:=]\s*[^\s"']+/gi, '$1=[REDACTED]');
  };

  const buildErrorResponse = (message, details) => {
    if (!isDev) return { error: message };
    return { error: message, ...details };
  };

  // Write QUBO to a temp file
  const quboPath = path.join(os.tmpdir(), `qubo_${Date.now()}.json`);
  fs.writeFileSync(quboPath, JSON.stringify(Q));

  // Call quantum_optimize.py
  const pyPath = path.join(process.cwd(), 'quantum_optimize.py');
  // NOTE: `k` is currently validated by the API contract, but the Python solver
  // selects sites based on the QUBO penalties rather than an explicit `--k`.
  const args = ['--qubo_file', quboPath];
  const start = Date.now();
  const pythonCmd = process.env.PYTHON || 'python';
  const py = spawn(pythonCmd, [pyPath, ...args]);
  let out = '', err = '';
  let responded = false;
  const timeout = setTimeout(() => {
    if (!responded) {
      responded = true;
      py.kill('SIGKILL');
      try { fs.unlinkSync(quboPath); } catch (e) {}
      console.error('Quantum optimization timed out.\nSTDOUT:', out, '\nSTDERR:', err);
      res.status(500).json(buildErrorResponse('Quantum optimization timed out', {
        stdout: redact(out),
        stderr: redact(err),
      }));
    }
  }, 60000); // 60s timeout
  py.stdout.on('data', d => { out += d.toString(); });
  py.stderr.on('data', d => { err += d.toString(); });
  py.on('close', code => {
    if (responded) return;
    responded = true;
    clearTimeout(timeout);
    try { fs.unlinkSync(quboPath); } catch (e) {}
    if (code !== 0) {
      console.error('Quantum optimization failed.\nSTDOUT:', out, '\nSTDERR:', err);
      res.status(500).json(buildErrorResponse('Quantum optimization failed', {
        details: redact(err),
        stdout: redact(out),
        code,
      }));
      return;
    }
    try {
      // Defensive: find last valid JSON in output
      const lines = out.trim().split('\n');
      let result = null;
      for (let i = lines.length - 1; i >= 0; i--) {
        try {
          result = JSON.parse(lines[i]);
          break;
        } catch (e) {}
      }
      if (!result) throw new Error('No valid JSON output from quantum_optimize.py');
      result.selected_grids = (result.selected_indices || []).map(i => candidates[i]);
      result.time_ms = Date.now() - start;
      res.status(200).json({ grids: result.selected_grids || [], time_ms: result.time_ms, raw: result });
    } catch (e) {
      console.error('Failed to parse quantum output.\nSTDOUT:', out, '\nSTDERR:', err, '\nException:', e);
      res.status(500).json(buildErrorResponse('Failed to parse quantum output', {
        details: redact(out + err),
        exception: e.toString(),
      }));
    }
  });
}
