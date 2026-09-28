# ARCHITECTURE

**Status:** complete for v0

FlyBrain is a package, not a collection of scripts. The boundaries below exist so
that the parts can be tested against each other and so that a future milestone (an
environment, a decoder) can attach without reaching into the simulation's internals.

---

## 1. Layering

```
                    ┌──────────────────────────────────────────┐
   user surface     │  cli.py            repl.py               │
                    │  experiments/      scripts/              │
                    └───────────────┬──────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────┐
   orchestration    │  runtime/runner.py   plan_run            │
                    │  runtime/recorder.py                     │
                    │  runtime/backend.py  (hardware + device) │
                    └───────────────┬──────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────┐
   dynamics         │  model/network.py  LIFNetwork            │
                    │  model/lif.py      exact integrator      │
                    │  model/stimulus.py model/populations.py  │
                    └───────────────┬──────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────┐
   data             │  data/loader.py   Connectome             │
                    │  data/validate.py data/provenance.py     │
                    │  data/schema.py   data/synthetic.py      │
                    └───────────────┬──────────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────────┐
   analysis         │  analysis/{firing_rates,population_      │
                    │  activity,connectivity,plots}.py         │
                    └──────────────────────────────────────────┘
```

Dependencies point one way only: `data` imports nothing from `model`; `model`
imports `data` for types only; `runtime` imports both; `cli`/`repl` import
everything. There are no cycles, and this is what keeps the model testable without
a dataset (`data/synthetic.py`).

---

## 2. The five operations

The objective asks for clean APIs around `apply_stimulus`, `step`, `observe`,
`silence`, `reset`, so a later milestone can attach a decoder or an environment.
Those five methods on `model/network.py::LIFNetwork` (aliased `Brain`) are the entire
boundary. Nothing below them knows about files, time, or experiments; nothing above
them reaches into their arrays except through these methods plus the read-only
`rates_hz()` / `spike_count` accessors.

| operation | signature | semantics |
|---|---|---|
| `apply_stimulus` | `(*stimuli) -> self` | **replaces** the declared stimulus set; restores the refractory period of neurons no longer targeted |
| `step` | `() -> spike_vector` | advances one `dt`; returns the spike vector for this step |
| `observe` | `(*, include_v, include_g, populations)` | read-only snapshot; heavy arrays are opt-in |
| `silence` | `(target_or_flywire_ids) -> ids` | removes outgoing influence; accepts a population name, a `Population`, or IDs |
| `reset` | `(seed=None) -> self` | zeroes state, **keeps** declared stimuli and silencing |

`reset` deliberately keeps interventions: they are part of the experiment, not the
transient state. `clear_stimuli()` and `clear_silencing()` are the explicit way to
drop them.

`Brain` is an alias of `LIFNetwork`, because the objective's long-term sketch names
the object `brain`.

---

## 3. Data flow

```
third_party/<upstream clone>            (read-only reference, never imported)
        │  stage_dataset()  copies + SHA-256
        ▼
data/raw/<dataset_id>/                  immutable byte-level snapshot
        │  build_connectome()  parquet -> int32/int16 arrays, sorted by `pre`
        ▼
data/processed/<dataset_id>/connectome.npz     derived cache, hash-linked to the raw files
        │  load_connectome()  attaches the DatasetCard
        ▼
Connectome (in memory: flywire_ids int64, pre/post int32, signed_count int16)
        │  subset(seeds, hops, max_neurons) — optional
        ▼
LIFNetwork  ──csr()──▶  (indptr, indices) for analysis
```

Cache invalidation is explicit: `data/processed/<id>/cache.json` records the
`source_digest` of the raw files it was built from, and `load_connectome` compares it
against the `DatasetCard.sha256`. A stale cache is ignored, never trusted.

The dataset's two files are ~90 MB (v630) and ~104 MB (v783). Nothing is downloaded:
`stage_dataset` copies from a local clone, and no code path performs network I/O.

---

## 4. Backends

`runtime/backend.py` defines a deliberately small abstraction. Almost the whole
kernel is written in `backend.xp` (NumPy or `jax.numpy`); only three operations
genuinely differ and only those are abstracted:

| primitive | why it cannot be shared |
|---|---|
| `scatter_add` | NumPy's `bincount` vs JAX's `.at[].add` |
| `add_at` | in-place `np.add.at` vs functional `.at[].add` |
| `set_row` | mutable ring buffer vs immutable `ring.at[i].set(row)` |

Everything else uses `xp.where` and fancy indexing, which both satisfy. `asarray`
**preserves** the input dtype when none is given — a default that coerced index
arrays to float would break the first fancy-index lookup, which is exactly the bug
that was found and fixed during development.

### Hardware presence vs. usability

These are reported separately and must never be conflated:

* `HardwareInfo.nvidia_gpus` — what `nvidia-smi` reports physically present;
* `Backend.is_gpu` — whether the running process can actually use a GPU.

On the machine FlyBrain v0 was developed on (GTX 1650, driver 591.86, CUDA 13.1,
Windows 11), `jax.default_backend()` returns `'cpu'` and
`jax.devices()` returns `[CpuDevice(id=0)]`, because CUDA-enabled `jaxlib` wheels are
published for Linux only. The banner therefore prints:

```
Backend         numpy  [CPU]
GPU usable      no - JAX reports backend='cpu' devices=['cpu:cpu#0']
GPU             NVIDIA GeForce GTX 1650
VRAM            4096 MB total, 2791 MB free
```

It does not print "CUDA". `requirements/gpu.txt` documents the WSL2 route to a real
GPU build rather than pretending the native Windows path has one.

### Why `auto` prefers NumPy

Measured on the development machine over a 4 000-neuron / 126 267-edge subset,
1 000 steps:

| backend | ms per step |
|---|---|
| numpy | **1.78** |
| jax | 3.15 |

Both produce **bit-identical** spike counts for the same seed, so the choice is
purely one of speed. On a machine where JAX can reach a real GPU the answer would
change, and it would be decided by the same measurement, not by assuming a GPU is
fast. `tests/test_runtime.py::test_backends_produce_bit_identical_spike_counts`
guards the correctness half of that claim.

---

## 5. Feasibility gate and the refusal contract

`whole-brain` mode is gated before any large allocation:

1. `estimate_requirements(n_neurons, n_edges, delay_steps)` returns a **breakdown**
   by component, not a single number, so a refusal can name the dominant term;
2. `check_feasibility` compares `peak_gb` against 60 % of *available* RAM (not total
   — a machine with 16 GB total and 3 GB free will die on a run sized against 16 GB);
3. if it does not fit, `ResourceLimitError` is raised with the estimate attached, the
   CLI prints it and exits with code **3**, and **nothing is simulated**;
4. `--allow-fallback` produces a subset plan instead, with `requested_mode` and
   `effective_mode` recorded separately, `mode_changed=True`, and a note beginning
   "WHOLE-BRAIN RUN WAS REQUESTED AND WAS NOT DELIVERED." The word "subset" appears
   in the run id, the banner, the plan notes, `summary.json`, and the console output.

A separate runtime gate (`--max-minutes`, default 60) refuses a run whose *measured*
per-step cost implies a longer wall time than allowed. The cost is measured by
actually stepping the assembled network three times, so the estimate reflects this
machine, this backend, and this stimulus set.

---

## 6. Recording

Four independent channels, each opt-in:

| channel | default | cost control |
|---|---|---|
| `spikes` | **off** | row cap (`max_spike_rows`, default 20 M) with an explicit `truncated` flag; optional neuron filter |
| `population` | **on** | coarse interval (default 10 ms), fixed population panel |
| `voltage` | off | explicit neuron list; interval |
| `summary` | always | scalars only |

The default is population rates only, because a whole-brain spike dump is unbounded
output. Truncation is never silent: `Recorder.summary()` reports
`spike_recording_truncated` and the row count actually written.

---

## 7. Determinism

* one `np.random.Generator(PCG64)` per stimulus, seeded from
  `sha256(base_seed, stimulus_spec_hash)`, so the event stream for a stimulus does not
  depend on how many other stimuli were declared or in what order
  (asserted in `tests/test_model_network.py`);
* the RNG is drawn on **every** step and the stimulus window is applied as a mask
  afterwards, so changing `start_ms` changes *which* events are used, not *which*
  events are generated;
* `spike_digest_sha256` in `summary.json` hashes the ordered
  `(t_ms, flywire_id)` pairs; two identical runs must agree, and the CLI test asserts
  it rather than trusting it;
* every run records the git revision, dataset SHA-256, configuration, seed, backend,
  dtype, `dt`, Python version, dependency versions and hardware.

---

## 8. Project layout

```
FlyBrain/
├── README.md
├── pyproject.toml                 package + `flybrain` console script
├── requirements/{base,gpu}.txt
├── scripts/render_doc_tables.py   regenerates doc tables from code
├── third_party/                   cloned upstream repos (never imported, gitignored)
├── data/{raw,processed,metadata}/ dataset tree; only metadata/ is committed
├── flybrain/
│   ├── config.py  errors.py  cli.py  repl.py  __main__.py
│   ├── data/      schema, loader, validate, provenance, synthetic
│   ├── model/     lif, network, populations (+ populations.json), stimulus, docgen
│   ├── runtime/   backend, runner, recorder
│   ├── analysis/  firing_rates, population_activity, connectivity, plots
│   └── experiments/  smoke_test, baseline_activity, sugar_stimulation, silencing_test
├── experiments/                   thin runnable wrappers over flybrain.experiments
├── tests/
├── outputs/                       runs/, experiments/, dataset_report.json
└── docs/
```

Deviations from the layout in the objective: `analysis/plots.py` was added (the
objective listed three analysis modules but four plot types, and plotting is a
distinct concern), `data/synthetic.py` was added (so the smoke test and the unit
tests do not need the 90 MB dataset), `model/docgen.py` and `scripts/` were added (so
the parameter table cannot drift from the code), and `experiments/` at the root holds
thin wrappers while the implementations live in `flybrain/experiments/` so they are
importable and testable.
