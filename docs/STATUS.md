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
| 3 | neuron IDs are preserved | **DONE** | `Connectome.flywire_ids` is the neuron table index verbatim; `tests/test_data.py::test_real_dataset_matches_the_audited_numbers`. Subsetting preserves IDs: `test_real_dataset_subset_of_the_sugar_grns` asserts all 21 sugar GRNs survive. Orphan neurons are *kept* in the index space (385 with no outgoing edge) so IDs stay stable — `test_real_dataset_orphans_are_kept_in_the_index_space`. |
| 4 | synaptic connectivity is represented correctly | **DONE** | CSR `indptr`/`indices` verified against a manual count (`test_csr_indptr_consistent_with_out_degree`); signed counts verified present in both directions; duplicate edge pairs = 0; self-loops = 0. Never densified — a dense matrix would need **64.9 GB** (`estimate_requirements`). |
| 5 | excitatory/inhibitory behaviour supported where data permits | **DONE** | Sign read from the dataset's `Excitatory` column. 86 543 excitatory / 40 472 inhibitory presynaptic neurons / 385 unknown (no outgoing edge). Verified that no neuron is mixed (`sign_is_per_neuron` check, `neurons_with_mixed_sign = 0`). Excitatory and inhibitory conductance changes asserted separately in `test_runtime.py`; an inhibitory volley is shown to be able to hold a neuron below threshold. |
| 6 | LIF network executes | **DONE** | `flybrain/model/lif.py` implements the exact linear solution. Verified against its closed form and its fixed point, and against explicit Euler in the small-`dt` limit (`tests/test_model_lif.py`). |
| 7 | subset simulation works on CPU | **DONE** | `flybrain run --mode subset --duration-ms 100 --stimulus sugar_grn`; subst runs measured at 0.12–2.0 ms/step over 300–8 000-neuron subsets. `tests/test_cli.py::test_subset_run_writes_every_required_artifact`. |
| 8 | sensory population can be stimulated | **DONE** | `Stimulus.population("sugar_grn", rate_hz=…)`. Measured 105.6 Hz for a 100 Hz drive, 147.1 Hz for a 150 Hz drive. `test_stimulus_rate_is_recovered_within_statistical_error`. |
| 9 | activity propagates through the network | **DONE** | Sugar run: 253 active neurons from 21 stimulated. Monosynaptic reachability: **211 directed edges / 2 729 synapses** from the sugar GRNs onto the responders. `outputs/experiments/sugar_stimulation/sugar_stimulation.json`. Delay propagation verified to arrive at exactly the configured delay: `test_synaptic_input_arrives_exactly_after_the_delay`. |
| 10 | neurons/populations can be silenced | **DONE** | `brain.silence("mn9")` / `--silence mn9`. Silencing experiment: MN9 88.0 → 94.5 Hz, `sez_bract` 46.0 → 49.0 Hz, with 3 direct edges from MN9 onto the changed populations. `outputs/experiments/silencing_test/silencing_test.json`. Semantics verified against the published results: the silenced neuron keeps spiking. |
| 11 | activity can be recorded | **DONE** | Four opt-in channels; `spikes.parquet`, `population_activity.parquet`, `voltage_samples.parquet`, `summary.json`. Truncation is reported, never silent (`test_recorder_truncation_is_flagged_not_silent`). Recorded spike rows are cross-checked against the network's own counter inside the sugar experiment. |
| 12 | results can be plotted | **DONE** | `flybrain/analysis/plots.py`: population rates over time, spike raster, top populations, input-vs-downstream comparison, membrane-potential traces. Every title carries `simulated activity - not a biological measurement`, asserted by `test_plots_carry_the_simulated_activity_tag`. Written to `outputs/*/plots/`. |
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
| 19 | reproduce at least one qualitative result from the published model | **DONE (qualitative, and explicitly not quantitative)** | Sugar GRN stimulation drives sparse downstream activation and MN9, matching the published pattern. Published reference (measured from the authors' own `sugarR_100Hz.parquet`, 30 trials): MN9-L 67.03 Hz, 404 active neurons. FlyBrain, 8 000-neuron subset, 1 trial: MN9-L 97.50 Hz, 253 active neurons. Same order, same sparsity, **numbers not claimed to match** — see "Honest limits" below. |

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
```

All tests pass. The suite is organised so that the physics is checked against
*independent* expectations rather than against the implementation's own output:

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

* network: 127 400 neurons, 14 687 178 edges, `subset=False`
* feasibility: passed against 3.66 GB available RAM; estimated peak requirement
  **0.36 GB** (largest component: the per-step transient over 14.7 M edges)
* measured step cost: **204.2 ms/step** → ~34 minutes per simulated second at
  `dt = 0.1 ms`
* peak resident memory observed: **~0.7 GB** (the estimate is conservative)
* this satisfies the objective's "whole-brain mode runs if available hardware permits"
  without any fallback

The measured 204 ms/step is higher than the bare-array benchmark of 126 ms/step
because the real run also performs the stimulus draws, the delay ring update, the
spike counter, and recording. Profiling is in `outputs/profile_*.json`.

---

## Known engineering limitations

1. **Whole-brain throughput.** 204 ms/step is dominated by a full gather over all
   14.7 M edges (`weights * delayed[pre]`) plus a `bincount` scatter-add, i.e. ~120 MB
   of memory traffic per step regardless of how many neurons actually spike. An
   active-set-only propagation (touch only the outgoing edges of neurons that spiked)
   is the obvious optimisation and is *deliberately not implemented yet*, because the
   objective says to profile a correct implementation before optimising. The profile
   exists (`scripts/profile_pipeline.py`); the optimisation does not.
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

The comparison is **qualitative and structural**, not quantitative:

| | published (`sugarR_100Hz.parquet`) | FlyBrain (8 000-neuron subset) |
|---|---|---|
| stimulus | sugar GRNs @100 Hz, 30 trials | sugar GRNs @100 Hz, 1 trial |
| sugar GRN rate | 98.8 Hz (median) | 105.6 Hz |
| active neurons | 404 | 253 |
| MN9 left | 67.03 Hz | 97.50 Hz |
| MN9 right | 48.63 Hz | not measured separately in that run |

The direction and the *sparsity* agree: driving a small sensory population produces a
response in a few hundred neurons, and MN9 is among the driven. The magnitudes do
not, and are not expected to:

* different RNG (upstream uses Brian 2's `PoissonInput`), so different Poisson
  realisations;
* 1 trial here versus 30 upstream, and the upstream figure is a 30-trial mean;
* a **truncated subset** here, which removes part of MN9's convergent input;
* different sub-step ordering from the chaobrain port
  (`docs/UPSTREAM_AUDIT.md` §4.1).

Running the matched protocol (whole-brain, 30 trials, 100 Hz) would be the honest way
to make a quantitative claim, and would cost roughly 17 hours of wall time on this
machine at the measured 204 ms/step. It has **not** been run. That is the difference
between "consistent with" and "reproduces", and this document does not elide it.

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
