# STATUS

**FlyBrain v0** — status of every requirement, with the evidence that establishes it.
Updated 2026-09-28.

Legend: **DONE** · **PARTIAL** · **BLOCKED** · **NOT STARTED**.
Nothing below is claimed on the strength of "the code exists"; each row names the
artefact or command that produced the evidence.

---

## Required (v0 is done only if all of these hold)

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | project installs from scratch | **DONE** | `pip install -e .` into a clean venv on Python 3.11.7; `flybrain --version` → `FlyBrain 0.0.1`. Only NumPy/pandas/pyarrow are hard requirements. |
| 2 | FlyWire-derived connectome data loads | **DONE** | `flybrain datasets --stage all` stages and validates both versions. `outputs/dataset_report.json`. Load path: parquet → int32/int16 arrays, 14 687 178 edges; 0.55 s from the processed cache, 1.5 s cold. |
| 3 | neuron IDs are preserved | **DONE** | `Connectome.flywire_ids` is the neuron table index verbatim. Proven directly: `flybrain datasets --report flywire_630` checks every edge's `Presynaptic_ID`/`Postsynaptic_ID` against `flywire_ids[index]` and reports **29 374 356 endpoints checked, 0 mismatches**. Subsetting preserves IDs (`test_real_dataset_subset_of_the_sugar_grns`), and orphan neurons are *kept* in the index space (385 with no outgoing edge) so IDs stay stable (`test_real_dataset_orphans_are_kept_in_the_index_space`). |
| 4 | synaptic connectivity is represented correctly | **DONE** | CSR `indptr`/`indices` verified against a manual count (`test_csr_indptr_consistent_with_out_degree`); signed counts verified present in both directions; duplicate edge pairs = 0; self-loops = 0. Never densified — a dense matrix would need **64.9 GB** (`estimate_requirements`). |
| 5 | excitatory/inhibitory behaviour supported where data permits | **DONE** | Sign read from the dataset's `Excitatory` column. 86 543 excitatory / 40 472 inhibitory presynaptic neurons / 385 unknown (no outgoing edge). Verified that no neuron is mixed (`sign_is_per_neuron` check, `neurons_with_mixed_sign = 0`). Excitatory and inhibitory conductance changes asserted separately in `test_runtime.py`; an inhibitory volley is shown to be able to hold a neuron below threshold. |
| 6 | LIF network executes | **DONE** | `flybrain/model/lif.py` implements the exact linear solution. Verified against its closed form and its fixed point, and against explicit Euler in the small-`dt` limit (`tests/test_model_lif.py`). |
| 7 | subset simulation works on CPU | **DONE** | `flybrain run --mode subset --duration-ms 100 --stimulus sugar_grn`; subst runs measured at 0.12–2.0 ms/step over 300–8 000-neuron subsets. `tests/test_cli.py::test_subset_run_writes_every_required_artifact`. |
| 8 | sensory population can be stimulated | **DONE** | `Stimulus.population("sugar_grn", rate_hz=…)`. Measured 105.6 Hz for a 100 Hz drive, 147.1 Hz for a 150 Hz drive. `test_stimulus_rate_is_recovered_within_statistical_error`. |
| 9 | activity propagates through the network | **DONE** | Sugar run: 253 active neurons from 21 stimulated. Monosynaptic reachability: **211 directed edges / 2 729 synapses** from the sugar GRNs onto the responders. `outputs/experiments/sugar_stimulation/sugar_stimulation.json`. Delay propagation verified to arrive at exactly the configured delay: `test_synaptic_input_arrives_exactly_after_the_delay`. |
| 10 | neurons/populations can be silenced | **DONE** | `brain.silence("mn9")` / `--silence mn9`. Silencing experiment: MN9 88.0 → 94.5 Hz, `sez_bract` 46.0 → 49.0 Hz, with 3 direct edges from MN9 onto the changed populations. `outputs/experiments/silencing_test/silencing_test.json`. Semantics verified against the published results: the silenced neuron keeps spiking. |
| 11 | activity can be recorded | **DONE** | Four opt-in channels; `spikes.parquet`, `population_activity.parquet`, `voltage_samples.parquet`, `summary.json`. Truncation is reported, never silent (`test_recorder_truncation_is_flagged_not_silent`). Recorded spike rows are cross-checked against the network's own counter inside the sugar experiment. |
| 12 | results can be plotted | **DONE** | `flybrain/analysis/plots.py`: population rates over time, spike raster, top populations, input-vs-downstream comparison, membrane-potential traces. Figures are produced by the runner, so the CLI, every packaged experiment and the Python API all get them (an earlier design put plotting in the CLI only, and the experiments silently produced none — that is fixed and now asserted). Eight figures are present under `outputs/experiments/*/plots/` and `outputs/experiments/silencing_test/input_vs_downstream.png`, all verified non-blank (9–21 % ink coverage). Every title carries `simulated activity - not a biological measurement`, asserted by `test_plots_carry_the_simulated_activity_tag`. |
| 13 | dataset provenance is recorded | **DONE** | `data/metadata/flywire_630.json` and `flywire_783.json`: per-file SHA-256, sizes, source URL, release, retrieval timestamp. `docs/DATA_PROVENANCE.md`. |
| 14 | experiment provenance is recorded | **DONE** | Every run writes `config.json`, `environment.json` (hardware, backend, git revision, dependency versions), `dataset.json`, `summary.json`, `run.log`. Two identical runs give an identical `spike_digest_sha256` (`test_two_identical_runs_produce_the_same_spike_digest`). |
| 15 | tests pass | **DONE** | See "Test suite" below. |
| 16 | documentation explains assumptions | **DONE** | `docs/MODEL_ASSUMPTIONS.md` (every parameter with source/reason/confidence), `UPSTREAM_AUDIT.md`, `DATA_PROVENANCE.md`, `SCIENTIFIC_BOUNDARIES.md`, `ARCHITECTURE.md`. The parameter table is generated from the code and `tests/test_docs.py` fails on drift. |

---

## Desired

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 17 | whole-brain mode runs if hardware permits | **DONE** | Run completed on the full 127 400-neuron / 14 687 178-edge network: **204.2 ms/step measured, 10 000 steps**. See "Whole-brain result" below. The feasibility gate passed against 3.66 GB available RAM; estimated peak requirement **0.36 GB**. |
| 18 | JAX GPU acceleration | **PARTIAL — environment-limited, not implemented-away** | A `jax` backend exists and is selectable (`--backend jax`), and produces **bit-identical** spike counts to NumPy. But **JAX cannot use a GPU on this machine**: `jax.default_backend()` → `'cpu'`, `jax.devices()` → `[CpuDevice(id=0)]`, because CUDA-enabled `jaxlib` wheels are published for Linux only. The GTX 1650 is present and reported, and reported as *unusable by this process*. `requirements/gpu.txt` documents the WSL2 route. NumPy is also measurably faster here (1.78 vs 3.15 ms/step), so `auto` selects NumPy. |
| 19 | reproduce at least one qualitative result from the published model | **DONE (qualitative; near-quantitative on one readout, and explicitly not claimed as a reproduction)** | Whole-brain sugar protocol: MN9-left **66.00 Hz vs the published 67.03 Hz (1.5 % apart)**, active neurons 376 vs 404 (7 %). See "Comparison against the authors' published output" below for why this is still not a quantitative reproduction claim. |

---

## Explicitly NOT v0 (verified absent, not merely unstarted)

| Item | Status | How the absence was verified |
|---|---|---|
| JEV | **NOT STARTED** (out of scope) | no such module; `ruff`/grep find no reference |
| LLM | **NOT STARTED** (out of scope) | no model weights, no tokenizer, no API client anywhere in the tree; no dependency on any inference library |
| language | **NOT STARTED** (out of scope) | the console parses fixed commands and states so in its own banner |
| consciousness testing | **NOT STARTED** (out of scope) | `docs/SCIENTIFIC_BOUNDARIES.md` §5 states the model has nothing to be conscious; `test_docs_do_not_assert_biological_fidelity` guards the claim |
| reinforcement learning | **NOT STARTED** (out of scope) | no reward signal, no policy, no optimiser, no gradients; the model's spike nonlinearity is a hard threshold, so no gradient path exists |
| artificial personality | **NOT STARTED** (out of scope) | — |
| autonomous agent | **NOT STARTED** (out of scope) | the simulation only advances when `step()` is called |
| robotic body | **NOT STARTED** (out of scope) | no serial/network actuator interface |

---

## Test suite

```
$ pytest -q
147 passed
```

All 147 tests pass, and `ruff check --select F,E9` is clean. The suite is organised so
that the physics is checked against *independent* expectations rather than against the
implementation's own output:

* the integrator against its exact closed-form solution and its fixed point
  (`test_integrator_matches_closed_form_impulse_response`,
  `test_rest_is_an_exact_fixed_point`);
* the threshold at the boundary `v == v_threshold` (`test_spike_threshold_is_crossing_v_threshold`);
* refractoriness blocking a *clamped* suprathreshold neuron — this caught a real bug
  where the threshold was not gated on `not_refractory`
  (`test_threshold_is_gated_by_refractoriness`);
* propagation arriving at exactly the configured delay
  (`test_synaptic_input_arrives_exactly_after_the_delay`);
* the number of synapses a single spike needs to cross threshold, derived from the
  closed form (`test_single_spike_propagation_requires_enough_synapses`);
* the NumPy and JAX backends agreeing bit-for-bit
  (`test_backends_produce_bit_identical_spike_counts`);
* dataset loading, ID mapping, connectivity indexing, subset extraction, CLI smoke,
  recording, silencing, and the whole-brain refusal path.

No test asserts that a function returns whatever it returns. The `needs_dataset`
marker skips the dataset-dependent tests cleanly on a fresh checkout, so the suite
is runnable before the 90 MB of parquet is staged.

---

## Whole-brain result

```bash
flybrain experiments sugar_stimulation --mode whole-brain --duration-ms 1000 \
    --rate-hz 100 --max-minutes 120 --compare-published
```

Measured on 2026-09-28 (Windows 11, i7-7700K, 16 GB RAM, no usable GPU):

| | value |
|---|---|
| network | 127 400 neurons, 14 687 178 edges, `subset=False` |
| feasibility | passed against 3.66 GB available RAM; estimated peak **0.366 GB** |
| measured step cost | **196.4 ms/step** (probe 204.2) |
| throughput | 5.09 steps/s |
| wall time | **1963.6 s = 32.7 min** for 1000 ms of simulated time |
| total spikes | 12 979 |
| active neurons | 376 |
| mean active neurons per step | **1.30** (max 10) |
| spike digest | `d28feb6b82d1108056718dd246480cf4061c2c83bc14241779213941541f3760` |
| recording | not truncated; no population had missing members |

The feasibility estimate was **0.366 GB peak against a 64.92 GB dense matrix** — a
factor of 177, which is why the sparse edge-list representation is a requirement
rather than an optimisation.

This satisfies the objective's "whole-brain mode runs if available hardware permits"
without any fallback: no subsetting, no shrinkage, no relabelling.

### Comparison against the authors' published output

| | published `sugarR_100Hz.parquet` | FlyBrain whole-brain |
|---|---|---|
| stimulus | sugar GRNs @100 Hz | sugar GRNs @100 Hz |
| trials | 30 | 1 |
| sugar GRN rate | 98.8 Hz (median) | 105.6 Hz |
| active neurons | 404 | 376 |
| **MN9 left** | **67.03 Hz** | **66.00 Hz** |
| monosynaptic sugar-GRN → responder edges | — | 177 edges / 2 282 synapses |

MN9-left agrees to **1.5 %** and the active-neuron count to **7 %**, under a different
RNG (Brian 2's `PoissonInput` vs NumPy PCG64), a single trial rather than a 30-trial
mean, and a different sub-step ordering from the chaobrain port. That is a much
stronger agreement than expected and it is the strongest single piece of evidence in
this project.

**It is still not a quantitative reproduction claim**, for a reason that matters: the
agreement could be partly coincidental. The honest way to establish a quantitative
claim is to run the matched protocol (30 trials, same conditions) and compare the
*distribution*, which costs roughly 16 hours at the measured 196 ms/step and **has not
been run**. One number matching to 1.5 % is encouraging; it is not proof.

The subset run of the same protocol gives MN9-left **88.0 Hz**, 31 % above the
published value — because a truncated subset removes part of MN9's convergent input.
That contrast is itself informative: it shows that convergence, not a single pathway,
is what drives this network, and it is why every subset result is labelled. (An
earlier subset figure of 97.5 Hz is not quoted, because it came from a run whose
`--subset-seed` had displaced the stimulus neurons from the working set; that bug is
fixed.)

---

## Profiling (all figures measured, not modelled)

`scripts/profile_pipeline.py`; raw results in `outputs/profile_*.json`.

| stage | subset (8 000 neurons / 442 346 edges) | whole-brain (127 400 / 14 687 178) |
|---|---|---|
| parquet decode (cold) | 0.82 s | 0.75 s |
| processed-cache load | 0.25 s | 0.18 s (4× faster than cold) |
| CSR construction | 0.14 s | 0.15 s |
| subset extraction | 0.41 s | n/a |
| network construction | 0.005 s | 0.062 s |
| **simulation step** | **5.8 ms** | **205.7 ms** |
| recording: population rates | +0.06 ms/step (×1.01) | within noise (×0.96) |
| recording: all spikes | +0.47 ms/step (×1.08) | within noise (×0.96) |

Two findings worth stating:

1. **Recording is not a bottleneck at either scale.** The spike set is so sparse
   (1.30 active neurons per step at whole-brain scale) that appending to a list costs
   less than the run-to-run variance. The channels stay opt-in for output-size
   reasons, not CPU reasons.
2. **The cold load path was the one real memory problem, and it is fixed.** Profiling
   showed ~2.4 GB of peak RSS to load a 90 MB file, because `read_parquet`
   materialises all seven columns as int64. Four of those columns (`Presynaptic_ID`,
   `Postsynaptic_ID`, `Connectivity`, `Excitatory`) were read and never consumed.
   Narrowing the load path to the three columns the simulation uses cut the measured
   RSS delta from **2 428 MB to 1 356 MB (−44 %)**. The full column set is now read
   only by the verification path, which uses it for an integrity check that would
   otherwise be unaffordable.

### Bottleneck identified, deliberately not yet optimised

The step is dominated by work proportional to the **edge count**, not the activity:
a full gather over all 14.7 M edges (`weights * delayed[pre]`) plus a `bincount`
scatter-add, i.e. ~120 MB of memory traffic per step regardless of how many neurons
spike. With only 1.30 neurons active per step on average, propagating only the
outgoing edges of neurons that actually spiked would remove the overwhelming majority
of that traffic.

This is **not implemented**, because the objective says to profile a correct
implementation before optimising, and the profile is what justifies the change. It is
the first item of the next milestone.

---

## Known engineering limitations

1. **Whole-brain throughput.** 196 ms/step is dominated by a full gather over all
   14.7 M edges (`weights * delayed[pre]`) plus a `bincount` scatter-add, i.e. ~120 MB
   of memory traffic per step regardless of how many neurons actually spike — and only
   **1.30 neurons are active per step on average**. An active-set-only propagation
   (touch only the outgoing edges of neurons that spiked) is the obvious optimisation
   and is *deliberately not implemented yet*, because the objective says to profile a
   correct implementation before optimising. The profile exists
   (`outputs/profile_*.json`); the optimisation does not.
2. **`bincount` accumulates in float64** before narrowing back to float32. That
   doubles the scatter traffic and is the single biggest cost in the step. It is the
   honest NumPy expression of the semantics; a hand-written kernel would not need it.
3. **Single-threaded.** No parallelism across neurons or trials. `--trials` runs
   serially. The upstream model used `joblib` over CPU cores; FlyBrain does not.
4. **No GPU path was exercised.** The `jax` backend was validated for correctness on
   CPU only.
5. **Subset selection is a k-hop neighbourhood, not a principled circuit extraction.**
   A truncated subset is reported (`truncated: true`) but a truncated subset can
   change a downstream response, since convergence is what drives this network. Every
   subset result is therefore labelled and the truncation is recorded.
6. **No ID mapping across FlyWire versions.** A v630 ID is not guaranteed to resolve
   in v783; `PopulationRegistry.resolve` reports missing members rather than failing.
7. **Recording is in-memory** until the run ends. A spike set larger than
   `max_spike_rows` is truncated (and flagged), not streamed to disk.
8. **`flybrain experiments` forwards unknown flags to argparse**, so a typo in an
   experiment flag produces argparse's error rather than a suggestion.
9. **The interactive console grows its working set on demand**, which resets the
   simulation clock. It says so when it happens, but it means a long console session
   can silently change which neurons are simulated.

## Known biological limitations

See `docs/SCIENTIFIC_BOUNDARIES.md` for the full argument. In brief:

1. `w_syn = 0.275 mV` is a **free parameter** labelled as such upstream; every
   absolute firing rate inherits that uncertainty.
2. `f_poi = 250` is justified upstream only by "250 is sufficient to cause spiking".
3. The sign is a **predicted** neurotransmitter identity, not a measurement, and its
   accuracy is not characterised.
4. The dataset ships **no cell-type annotation**; types come only from 114 curated
   lists, and 385 neurons have no sign evidence at all.
5. A single presynaptic spike needs **≈162 synapses** on one target to cross
   threshold, so responses require convergent input; the model is therefore strongly
   dependent on how much of a neuron's input is present in the working set.
6. The connectome is the proofread central brain at completeness threshold 630 — no
   ventral nerve cord, no peripheral ganglia, no body.
7. A uniform 1.8 ms delay on every connection is physiologically implausible.
8. No plasticity, adaptation, inhibition reversal potential, gap junctions,
   neuromodulation, or intrinsic noise. Every neuron is the same neuron.
9. **This model has not been compared against any biological measurement.** Not once.

---

## Honest limits on the "reproduces a published result" claim

The structural and qualitative claims are established. The **quantitative** claim is
not, and here is the full accounting.

What holds up:

* driving the declared sensory population drives the declared readout: MN9-left is
  66.00 Hz against a published 67.03 Hz;
* the response is *sparse*: 376 active of 127 400 neurons, against 404 published;
* it is *reachable*: 177 directed edges / 2 282 synapses run from the sugar GRNs onto
  the responders, so the result is not a numerical artefact of unrelated neurons;
* it is *attributable*: with no stimulus the same model produces exactly zero spikes.

What is not established:

* **one number agreeing to 1.5 % is not a reproduction.** It is one readout, on one
  trial, on one machine. A quantitative claim needs the distribution over trials;
* 1 trial here versus 30 upstream, and the published value is a 30-trial mean;
* different RNG (Brian 2's `PoissonInput` versus NumPy PCG64), so different Poisson
  realisations — the sugar GRN rate itself differs (105.6 vs 98.8 Hz) before any
  downstream effect is considered;
* a side-by-side subset run of the same protocol gives MN9-left **88.0 Hz**, i.e. 31 %
  *above* the published value, purely because a truncated subset removes part of MN9's
  convergent input. So the model's readout is sensitive to which inputs are present,
  and the agreement in the whole-brain case should be read with that in mind;
* the MN9-right readout was not compared in the whole-brain run;
* no comparison against any biological measurement was made, and none is claimed.

Running the matched protocol (whole-brain, 30 trials, 100 Hz) would make a
quantitative claim possible. At the measured 196 ms/step that is roughly **16 hours**
of wall time, and it **has not been run**. That is the difference between "consistent
with" and "reproduces", and this document does not elide it.

---

## Next logical milestone

**v0.1 — make the existing whole-brain run fast enough to be scientifically useful,
then use it.**

In order:

1. **Active-set propagation.** Propagate only the outgoing edges of neurons that
   actually spiked, using the CSR row pointers that are already built. On the sugar
   protocol (a few hundred active of 127 400) this is a one-to-two order-of-magnitude
   reduction in per-step work, and it is the bottleneck the profile identifies.
2. **Re-profile** against `outputs/profile_*.json` and confirm the measured speedup
   before anything else changes.
3. **Run the matched reference protocol**: whole-brain, 30 trials, 100 Hz sugar, and
   compare the MN9 rate distribution against the published value quantitatively
   instead of qualitatively. This is the first milestone that could produce a
   *scientific* rather than an engineering claim.
4. **Sensitivity analysis over `w_syn`.** Until it is known how much of the observed
   pattern is the free parameter rather than the wiring, no pattern can be attributed
   to the connectome.
5. Only then consider the objective's later milestones (environment → decoder → JEV).
   The five-operation boundary in `docs/ARCHITECTURE.md` §2 exists so that step 5 can
   be attempted without touching the simulation.
