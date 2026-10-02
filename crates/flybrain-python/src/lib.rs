use flybrain_core::{Config, Core, Input};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

type StateSnapshot = (
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<u8>,
    Vec<u64>,
    usize,
);

#[pyclass]
struct NativeCore {
    inner: Core,
}

#[pymethods]
impl NativeCore {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n: usize,
        edges: Vec<(usize, usize, f32)>,
        dt_ms: f64,
        delay_steps: usize,
        coefficients: (f32, f32, f32),
        rest: f32,
        reset: f32,
        threshold: f32,
        refractory: f32,
        reference_order: bool,
        max_bytes: usize,
    ) -> PyResult<Self> {
        let cfg = Config {
            dt_ms,
            delay_steps,
            ea: coefficients.0,
            ed: coefficients.1,
            k: coefficients.2,
            rest,
            reset,
            threshold,
            refractory,
            reference_order,
            max_bytes,
        };
        Ok(Self {
            inner: Core::new(n, &edges, cfg).map_err(PyValueError::new_err)?,
        })
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn set_state(
        &mut self,
        v: Vec<f32>,
        g: Vec<f32>,
        last: Vec<f32>,
        tau: Vec<f32>,
    ) -> PyResult<()> {
        self.inner
            .set_state(&v, &g, &last, &tau)
            .map_err(PyValueError::new_err)
    }
    fn silence(&mut self, indices: Vec<usize>) -> PyResult<()> {
        self.inner.silence(&indices).map_err(PyValueError::new_err)
    }
    fn set_recurrence_disabled(&mut self, disabled: bool) {
        self.inner.disable_recurrence = disabled;
    }
    fn state(&self) -> StateSnapshot {
        let c = &self.inner;
        (
            c.v.clone(),
            c.g.clone(),
            c.t_last.clone(),
            c.tau_ref.clone(),
            c.last_spike.clone(),
            c.spike_count.clone(),
            c.step_index,
        )
    }
    #[getter]
    fn estimated_bytes(&self) -> usize {
        self.inner.estimated_bytes
    }

    fn advance<'py>(
        &mut self,
        py: Python<'py>,
        steps: usize,
        inputs: Vec<(usize, usize, f32, f32)>,
        populations: Vec<Vec<usize>>,
        max_recorded_spikes: usize,
        max_seconds: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let events: Vec<Input> = inputs
            .into_iter()
            .map(|(step, neuron, dv, dg)| Input {
                step,
                neuron,
                dv,
                dg,
            })
            .collect();
        let s = py
            .detach(|| {
                self.inner.advance(
                    steps,
                    &events,
                    &populations,
                    max_recorded_spikes,
                    max_seconds,
                )
            })
            .map_err(PyValueError::new_err)?;
        let d = PyDict::new(py);
        d.set_item("completed", s.completed)?;
        d.set_item("stop_reason", s.stop_reason)?;
        d.set_item("completed_steps", s.completed_steps)?;
        d.set_item("spikes", s.spikes)?;
        d.set_item("active_edges", s.active_edges)?;
        d.set_item("population_counts", s.population_counts)?;
        d.set_item("recorded_spikes", s.recorded)?;
        d.set_item("recording_truncated", s.recording_truncated)?;
        Ok(d)
    }
}

#[pymodule]
fn flybrain_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeCore>()?;
    m.add("CONTRACT", "fixed_step_mixed_f32_f64_v1")?;
    Ok(())
}
