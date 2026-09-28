# FlyBrain v0

An executable, reproducible simulation substrate for the *Drosophila* whole-brain
connectome.

FlyBrain loads the FlyWire-derived connectivity published with the Shiu et al. (2024)
Drosophila brain model, runs a leaky integrate-and-fire network over it, accepts
controlled sensory stimulation, supports causal silencing, records the resulting
activity, and keeps a full provenance record. It is a substrate for experiments — not
a mind.

```
Stimulate the 21 labelled sugar-sensing gustatory receptor neurons at 100 Hz for 1 s
        ↓
253 neurons fire in an 8 000-neuron working set; 21 are the ones we drove
        ↓
MN9 motor neurons fire at 97.5 Hz, and 211 monosynaptic edges run from the sugar GRNs
to the neurons that responded
        ↓
Every number traceable to a dataset SHA-256, a git revision, and a seed
```

---

## What v0 is, and is not

**Is:** a validated connectome loader, a verified LIF implementation with the
published constants, sparse whole-brain execution, subset execution for CPU,
controlled stimulation, silencing, selective recording, plots, a command console, four
packaged experiments, and a test suite that checks the physics rather than the code's
own output.

**Is not:** a language model, an agent, a claims about consciousness, or a claim that
simulated activity equals biological activity. Explicitly out of scope for v0: LLM,
JEV, natural-language interface, reinforcement learning, decoder, robotic body. See
[`docs/SCIENTIFIC_BOUNDARIES.md`](docs/SCIENTIFIC_BOUNDARIES.md).

---

## Install

```bash
git clone <this repo> FlyBrain && cd FlyBrain

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate

pip install -e .
pip install -e ".[plot]"        # optional: matplotlib, for the figures
```

Requirements: Python ≥ 3.10, NumPy, pandas, pyarrow. Nothing else is needed for
subset mode, and nothing downloads anything at run time.

### Get the data

The dataset is FlyWire-derived, already preprocessed and published by the upstream
authors. FlyBrain does not download raw electron-microscopy volumes and never needs
to.

```bash
git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model.git \
    third_party/Drosophila_brain_model

flybrain datasets --stage all      # copies both versions into data/raw and validates
```

Two versions ship in that repository: `flywire_630` (127 400 neurons — the version the
paper used, and FlyBrain's primary) and `flywire_783` (138 639 neurons — registered and
validated, not primary). Details and hashes:
[`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md).

---

## Quick start

```bash
flybrain info                                     # hardware, backend, paths, datasets
flybrain datasets                                 # staged datasets and their SHA-256
flybrain populations --search sugar               # curated neuron populations
flybrain neuron 720575940660219265                # inspect one neuron by FlyWire ID

flybrain run --mode subset --duration-ms 100 --stimulus sugar_grn
flybrain run --mode subset --stimulus sugar_grn --duration-ms 1000 --record population
flybrain run --mode subset --stimulus sugar_grn --duration-ms 1000 --silence mn9

flybrain experiments sugar_stimulation --mode subset --duration-ms 1000
flybrain interactive
```

Python API:

```python
from flybrain import config
from flybrain.data import load_connectome
from flybrain.model import LIFNetwork, Stimulus, PopulationRegistry
from flybrain.runtime import plan_run, run_experiment
from flybrain.config import RunConfig

conn = load_connectome("flywire_630", config.RAW_DIR, config.PROCESSED_DIR, config.METADATA_DIR)

stim = Stimulus.population("sugar_grn", rate_hz=100.0)      # or Stimulus([...], 150, 100, 500)
cfg = RunConfig(mode="subset", duration_ms=1000.0, subset_hops=2, subset_max_neurons=8000)

plan = plan_run(conn, cfg, [stim])                           # decides subset vs whole-brain
result = run_experiment(plan, cfg, [stim])

print(result.summary["total_spikes"], result.summary["population_rates_hz"]["mn9"])
```

The network itself exposes exactly five operations, which are the boundary a future
milestone would attach to:

```python
brain.apply_stimulus(stim)   # set the stimulus set
brain.step()                 # advance one dt
obs = brain.observe()        # read state
brain.silence("mn9")         # remove a population's outgoing influence
brain.reset(seed=0)          # zero state, keep interventions
```

---

## The two execution modes

### `subset`

A bounded k-hop in-and-out neighbourhood around the stimulus neurons (plus any
`--subset-seed`), relabelled but keeping FlyWire IDs. Fast enough to iterate on:
milliseconds per simulated second on CPU.

```bash
flybrain run --mode subset --duration-ms 1000 --stimulus sugar_grn --subset-max-neurons 8000
```

### `whole-brain`

All 127 400 neurons and all 14 687 178 edges, measured on the development machine
(Windows 11, i7-7700K, 16 GB RAM, no usable GPU):

| | value |
|---|---|
| measured step cost | **196 ms/step** |
| throughput | 5.1 steps/s |
| 1000 ms of simulated time | **32.7 min** |
| estimated peak memory | **0.366 GB** |
| a dense connectivity matrix would need | **64.9 GB** |

It runs; it is not fast. See [`docs/STATUS.md`](docs/STATUS.md) for the full
measurement and for the identified bottleneck.

```bash
flybrain run --mode whole-brain --duration-ms 1000 --stimulus sugar_grn --max-minutes 120
```

**If whole-brain cannot run safely, FlyBrain refuses rather than quietly shrinking.**
It estimates the memory requirement by component, compares it against 60 % of
*available* (not total) RAM, prints the estimate, exits with code **3**, and simulates
nothing — not even a run directory:

```
$ flybrain run --mode whole-brain --duration-ms 60000 --max-minutes 1
REFUSED: estimated runtime 2548.4 min exceeds the configured limit of 1.0 min
         (254.8 ms/step measured, 600000 steps x 1 trial)
```

Pass `--allow-fallback` to get a subset instead — every artefact then records
`requested_mode: whole-brain`, `effective_mode: subset`, `mode_changed: true`, and a
note beginning "WHOLE-BRAIN RUN WAS REQUESTED AND WAS NOT DELIVERED."

---

## Where results go

```
outputs/runs/<run-id>/
├── config.json        full configuration, stimuli, model parameters
├── environment.json   hardware, backend, Python, dependency versions, git revision
├── dataset.json       dataset card with per-file SHA-256, subset description
├── summary.json       measured activity, timing, population rates, spike digest
├── spikes.parquet     only when --record spikes was requested
├── population_activity.parquet
├── voltage_samples.parquet   only when --record voltage was requested
├── run.log
└── plots/             population rates, raster, top populations
```

Spike tables are **not** recorded by default: a whole-brain spike dump is unbounded
output. The default is population rates. When recording is capped, the summary says so
rather than silently truncating.

---

## Experiments

| command | what it establishes |
|---|---|
| `flybrain experiments smoke_test` | initialises, stimulates, propagates, is deterministic — on a synthetic graph, so it needs no dataset |
| `flybrain experiments baseline_activity` | with no stimulus the model is exactly quiescent, so later activity is attributable to the declared stimulus |
| `flybrain experiments sugar_stimulation` | the known sensory population drives propagation into non-sensory downstream populations |
| `flybrain experiments silencing_test` | stimulate A, silence B, and measure the change in C against an un-silenced baseline |

Each writes a JSON report containing its hypothesis, its measured numbers, and each
check with the evidence behind it.

---

## Tests

```bash
pytest -q                    # full suite
pytest -q -m "not slow"      # skip the long ones
```

Dataset-dependent tests skip cleanly when the data is not staged, so the suite runs on
a fresh checkout. The tests check physics and structure, not the code's own output:
the integrator is verified against its exact closed-form solution and its fixed point;
the threshold boundary is verified at `v == v_threshold`; refractoriness is verified to
block a clamped suprathreshold neuron; propagation is verified to arrive at *exactly*
the configured delay; and the NumPy and JAX backends are verified to produce
bit-identical spike counts.

---

## Documentation

| document | contents |
|---|---|
| [`docs/UPSTREAM_AUDIT.md`](docs/UPSTREAM_AUDIT.md) | what the three upstream projects contain, which parameters are theirs, and which are modelling assumptions |
| [`docs/MODEL_ASSUMPTIONS.md`](docs/MODEL_ASSUMPTIONS.md) | every parameter with source, reason and confidence; every deliberate deviation |
| [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md) | dataset identity, SHA-256, measured content, validation performed |
| [`docs/SCIENTIFIC_BOUNDARIES.md`](docs/SCIENTIFIC_BOUNDARIES.md) | what may and may not be concluded from the output |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | layering, backends, the refusal contract, determinism |
| [`docs/STATUS.md`](docs/STATUS.md) | DONE / PARTIAL / BLOCKED / NOT STARTED for every v0 requirement, with evidence |

---

## Limits worth knowing before you trust a number

* `w_syn = 0.275 mV` is labelled a **free parameter** upstream. Absolute firing rates
  inherit that uncertainty.
* Sign is a **per-neuron** property in this dataset and is a *predicted*
  neurotransmitter identity, not a measurement.
* The dataset ships **no cell-type annotation**; neuron types come only from 114
  curated population lists.
* A single presynaptic spike needs ≈162 synapses on one target to cross threshold, so
  downstream responses require convergent input.
* JAX has no CUDA wheels for native Windows, so a GPU on a Windows host is not usable
  from this process. FlyBrain reports this instead of printing "CUDA"
  ([`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §4).

---

## License

Apache-2.0. The dataset and the model derive from
[philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)
(Apache-2.0); cite Shiu et al., *Nature* **634**, 210–219 (2024),
[doi:10.1038/s41586-024-07763-9](https://doi.org/10.1038/s41586-024-07763-9).
