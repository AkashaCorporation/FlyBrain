//! Fixed-step dense LIF integration with active outgoing CSR propagation.
//! No RNG, threads, Python dependencies, or implicit model substitution.
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub dt_ms: f64,
    pub delay_steps: usize,
    pub ea: f32,
    pub ed: f32,
    pub k: f32,
    pub rest: f32,
    pub reset: f32,
    pub threshold: f32,
    pub refractory: f32,
    pub reference_order: bool,
    pub max_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Input {
    pub step: usize,
    pub neuron: usize,
    pub dv: f32,
    pub dg: f32,
}

#[derive(Debug)]
pub struct Summary {
    pub completed_steps: usize,
    pub completed: bool,
    pub stop_reason: &'static str,
    pub spikes: u64,
    pub active_edges: u64,
    pub population_counts: Vec<u64>,
    pub recorded: Vec<(usize, usize)>,
    pub recording_truncated: bool,
}

pub struct Core {
    cfg: Config,
    pub v: Vec<f32>,
    pub g: Vec<f32>,
    pub t_last: Vec<f32>,
    pub tau_ref: Vec<f32>,
    pub last_spike: Vec<u8>,
    pub spike_count: Vec<u64>,
    pub step_index: usize,
    pub disable_recurrence: bool,
    kill: Vec<bool>,
    indptr: Vec<usize>,
    edge_ids: Vec<usize>,
    post: Vec<usize>,
    weights: Vec<f32>,
    ring: Vec<Vec<usize>>,
    cursor: usize,
    active: Vec<usize>,
    active_edges: Vec<usize>,
    accumulator: Vec<f64>,
    pub estimated_bytes: usize,
}

impl Core {
    pub fn new(n: usize, edges: &[(usize, usize, f32)], cfg: Config) -> Result<Self, String> {
        if n == 0
            || cfg.delay_steps == 0
            || !cfg.dt_ms.is_finite()
            || cfg.dt_ms <= 0.0
            || ![
                cfg.ea,
                cfg.ed,
                cfg.k,
                cfg.rest,
                cfg.reset,
                cfg.threshold,
                cfg.refractory,
            ]
            .iter()
            .all(|x| x.is_finite())
            || cfg.refractory < 0.0
            || cfg.threshold <= cfg.reset
        {
            return Err("invalid core configuration".into());
        }
        if edges
            .iter()
            .any(|&(a, b, w)| a >= n || b >= n || !w.is_finite())
        {
            return Err("invalid edge endpoint or weight".into());
        }
        // Conservative owned-buffer + caller edge-copy allowance, checked before allocations.
        let estimated = n
            .checked_mul(
                80usize
                    .checked_add(cfg.delay_steps.checked_mul(8).ok_or("size overflow")?)
                    .ok_or("size overflow")?,
            )
            .and_then(|x| edges.len().checked_mul(64).and_then(|e| x.checked_add(e)))
            .and_then(|x| {
                cfg.delay_steps
                    .checked_mul(24)
                    .and_then(|e| x.checked_add(e))
            })
            .ok_or("size overflow")?;
        if estimated > cfg.max_bytes {
            return Err("core memory budget exceeded".into());
        }
        let mut indptr = vec![0; n + 1];
        for &(pre, _, _) in edges {
            indptr[pre + 1] += 1;
        }
        for i in 0..n {
            indptr[i + 1] += indptr[i];
        }
        let mut offsets = indptr.clone();
        let mut edge_ids = vec![0; edges.len()];
        for (id, &(pre, _, _)) in edges.iter().enumerate() {
            edge_ids[offsets[pre]] = id;
            offsets[pre] += 1;
        }
        let mut core = Self {
            cfg,
            v: vec![cfg.rest; n],
            g: vec![0.0; n],
            t_last: vec![-1e7; n],
            tau_ref: vec![cfg.refractory; n],
            last_spike: vec![0; n],
            spike_count: vec![0; n],
            step_index: 0,
            disable_recurrence: false,
            kill: vec![false; n],
            indptr,
            edge_ids,
            post: edges.iter().map(|x| x.1).collect(),
            weights: edges.iter().map(|x| x.2).collect(),
            ring: (0..cfg.delay_steps)
                .map(|_| Vec::with_capacity(n))
                .collect(),
            cursor: 0,
            active: Vec::with_capacity(n),
            active_edges: Vec::with_capacity(edges.len()),
            accumulator: vec![0.0; n],
            estimated_bytes: estimated,
        };
        core.reset();
        Ok(core)
    }

    /// Preserve declared tau_ref, kill mask and recurrence intervention.
    pub fn reset(&mut self) {
        self.v.fill(self.cfg.rest);
        self.g.fill(0.0);
        self.t_last.fill(-1e7);
        self.last_spike.fill(0);
        self.spike_count.fill(0);
        self.step_index = 0;
        self.cursor = 0;
        for slot in &mut self.ring {
            slot.clear();
        }
        self.active.clear();
        self.active_edges.clear();
        self.accumulator.fill(0.0);
    }

    pub fn set_state(
        &mut self,
        v: &[f32],
        g: &[f32],
        last: &[f32],
        tau: &[f32],
    ) -> Result<(), String> {
        let n = self.v.len();
        if [v, g, last, tau]
            .iter()
            .any(|a| a.len() != n || a.iter().any(|x| !x.is_finite()))
            || tau.iter().any(|x| *x < 0.0)
        {
            return Err(
                "state must be finite, correctly sized, with nonnegative refractory".into(),
            );
        }
        self.v.copy_from_slice(v);
        self.g.copy_from_slice(g);
        self.t_last.copy_from_slice(last);
        self.tau_ref.copy_from_slice(tau);
        Ok(())
    }

    pub fn silence(&mut self, indices: &[usize]) -> Result<(), String> {
        if indices.iter().any(|&i| i >= self.v.len()) {
            return Err("silencing index out of range".into());
        }
        self.kill.fill(false);
        for &i in indices {
            self.kill[i] = true;
        }
        Ok(())
    }

    fn step(&mut self, inputs: &[Input]) -> u64 {
        let t = (self.step_index as f64 * self.cfg.dt_ms) as f32;
        self.active.clear();
        for i in 0..self.v.len() {
            self.last_spike[i] = 0;
            if t - self.t_last[i] > self.tau_ref[i] {
                // No fused multiply-add: reference ndarray operations round separately.
                self.v[i] = self.cfg.ea * (self.v[i] - self.cfg.rest)
                    + self.cfg.k * self.g[i]
                    + self.cfg.rest;
                self.g[i] *= self.cfg.ed;
                if self.v[i] > self.cfg.threshold {
                    self.v[i] = self.cfg.reset;
                    self.g[i] = 0.0;
                    self.t_last[i] = t;
                    self.last_spike[i] = 1;
                    self.spike_count[i] += 1;
                    self.active.push(i);
                }
            }
        }
        // External voltage events occur after threshold. Multiple events keep supplied order.
        for event in inputs {
            self.v[event.neuron] += event.dv;
        }
        self.active_edges.clear();
        if !self.disable_recurrence {
            for &pre in &self.ring[self.cursor] {
                if !self.kill[pre] {
                    self.active_edges
                        .extend_from_slice(&self.edge_ids[self.indptr[pre]..self.indptr[pre + 1]]);
                }
            }
            if self.cfg.reference_order {
                self.active_edges.sort_unstable();
            }
            self.accumulator.fill(0.0);
            for &edge in &self.active_edges {
                self.accumulator[self.post[edge]] += self.weights[edge] as f64;
            }
            // Also add +0 to untouched destinations, matching NumPy's full vector addition.
            for (g, &sum) in self.g.iter_mut().zip(&self.accumulator) {
                *g += sum as f32;
            }
        }
        self.ring[self.cursor].clear();
        self.ring[self.cursor].extend_from_slice(&self.active);
        self.cursor = (self.cursor + 1) % self.cfg.delay_steps;
        for event in inputs {
            self.g[event.neuron] += event.dg;
        }
        self.step_index += 1;
        self.active_edges.len() as u64
    }

    pub fn advance(
        &mut self,
        steps: usize,
        inputs: &[Input],
        populations: &[Vec<usize>],
        max_recorded_spikes: usize,
        max_seconds: f64,
    ) -> Result<Summary, String> {
        if !max_seconds.is_finite() || max_seconds <= 0.0 || steps > 10_000_000 {
            return Err("invalid block duration or steps (maximum 10000000)".into());
        }
        if self.step_index.checked_add(steps).is_none() {
            return Err("step overflow".into());
        }
        if inputs.iter().any(|e| {
            e.step >= steps || e.neuron >= self.v.len() || !e.dv.is_finite() || !e.dg.is_finite()
        }) || inputs.windows(2).any(|w| w[0].step > w[1].step)
            || populations.iter().flatten().any(|&i| i >= self.v.len())
        {
            return Err("invalid/sortedness input or population".into());
        }
        if populations.iter().any(|p| {
            let mut ids = p.clone();
            ids.sort_unstable();
            ids.windows(2).any(|x| x[0] == x[1])
        }) {
            return Err("duplicate population neuron".into());
        }
        let transient = max_recorded_spikes
            .checked_mul(16)
            .and_then(|x| inputs.len().checked_mul(32).and_then(|y| x.checked_add(y)))
            .and_then(|x| {
                populations.iter().try_fold(x, |a, p| {
                    p.len().checked_mul(16).and_then(|b| a.checked_add(b))
                })
            })
            .and_then(|x| x.checked_add(populations.len().checked_mul(32)?))
            .ok_or("recording size overflow")?;
        if transient > self.cfg.max_bytes.saturating_sub(self.estimated_bytes) {
            return Err("block memory budget exceeded".into());
        }
        let start = Instant::now();
        let mut summary = Summary {
            completed_steps: 0,
            completed: true,
            stop_reason: "finished",
            spikes: 0,
            active_edges: 0,
            population_counts: vec![0; populations.len()],
            recorded: Vec::with_capacity(max_recorded_spikes),
            recording_truncated: false,
        };
        let mut next = 0;
        for tick in 0..steps {
            if start.elapsed().as_secs_f64() >= max_seconds {
                summary.completed = false;
                summary.stop_reason = "wall_limit";
                break;
            }
            let first = next;
            while next < inputs.len() && inputs[next].step == tick {
                next += 1;
            }
            summary.active_edges += self.step(&inputs[first..next]);
            summary.spikes += self.active.len() as u64;
            for (count, pop) in summary.population_counts.iter_mut().zip(populations) {
                *count += pop.iter().map(|&i| self.last_spike[i] as u64).sum::<u64>();
            }
            if max_recorded_spikes > 0 {
                for &i in &self.active {
                    if summary.recorded.len() < max_recorded_spikes {
                        summary.recorded.push((self.step_index - 1, i));
                    } else {
                        summary.recording_truncated = true;
                    }
                }
            }
            summary.completed_steps += 1;
        }
        Ok(summary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cfg() -> Config {
        Config {
            dt_ms: 0.1,
            delay_steps: 18,
            ea: 1.0,
            ed: 1.0,
            k: 0.0,
            rest: -52.0,
            reset: -52.0,
            threshold: -45.0,
            refractory: 2.2,
            reference_order: true,
            max_bytes: 1_000_000,
        }
    }
    #[test]
    fn threshold_delay_silence_reset() {
        let mut c = Core::new(3, &[(0, 2, 4.0), (1, 2, -1.0)], cfg()).unwrap();
        c.v[0] = -45.0;
        c.v[1] = -44.0;
        let s = c.advance(18, &[], &[vec![0, 1]], 10, 2.0).unwrap();
        assert_eq!(s.recorded, vec![(0, 1)]);
        assert_eq!(c.g[2], 0.0);
        c.advance(1, &[], &[], 0, 2.0).unwrap();
        assert_eq!(c.g[2], -1.0);
        c.reset();
        c.silence(&[1]).unwrap();
        c.v[1] = 0.0;
        let s = c.advance(19, &[], &[], 0, 2.0).unwrap();
        assert_eq!(s.spikes, 1);
        assert_eq!(s.active_edges, 0);
        assert_eq!(c.g[2], 0.0);
    }
    #[test]
    fn mixed_precision_and_recording_cap() {
        let mut c = Core::new(
            4,
            &[(0, 3, 16777216.0), (1, 3, 1.0), (2, 3, -16777216.0)],
            cfg(),
        )
        .unwrap();
        c.v[..3].fill(0.0);
        let s = c.advance(19, &[], &[], 1, 2.0).unwrap();
        assert_eq!(c.g[3], 1.0);
        assert_eq!(s.active_edges, 3);
        assert_eq!(s.recorded.len(), 1);
        assert!(s.recording_truncated);
    }
    #[test]
    fn input_after_threshold_and_budget_refusal() {
        let mut c = Core::new(1, &[], cfg()).unwrap();
        let s = c
            .advance(
                1,
                &[Input {
                    step: 0,
                    neuron: 0,
                    dv: 68.75,
                    dg: 0.0,
                }],
                &[],
                5,
                2.0,
            )
            .unwrap();
        assert_eq!(s.spikes, 0);
        assert_eq!(c.advance(1, &[], &[], 5, 2.0).unwrap().spikes, 1);
        assert!(c.advance(1, &[], &[], usize::MAX, 2.0).is_err());
        assert!(Core::new(1000000, &[], cfg()).is_err());
    }

    #[test]
    fn original_edge_order_is_not_csr_order() {
        let edges = [(2, 3, 1e30f32), (0, 3, -1e30f32), (1, 3, 1.0)];
        let mut reference = Core::new(4, &edges, cfg()).unwrap();
        let mut fast_cfg = cfg();
        fast_cfg.reference_order = false;
        let mut csr = Core::new(4, &edges, fast_cfg).unwrap();
        reference.v[..3].fill(0.0);
        csr.v[..3].fill(0.0);
        reference.advance(19, &[], &[], 0, 2.0).unwrap();
        csr.advance(19, &[], &[], 0, 2.0).unwrap();
        assert_eq!(reference.g[3], 1.0);
        assert_eq!(csr.g[3], 0.0);
    }

    #[test]
    fn interrupted_block_reports_progress_without_reset() {
        let mut c = Core::new(4, &[], cfg()).unwrap();
        let s = c.advance(100, &[], &[], 0, 1e-30).unwrap();
        assert!(!s.completed);
        assert_eq!(s.stop_reason, "wall_limit");
        assert_eq!(s.completed_steps, 0);
        assert_eq!(c.step_index, 0);
    }
}
