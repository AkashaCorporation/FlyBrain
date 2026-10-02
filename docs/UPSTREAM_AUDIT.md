# UPSTREAM AUDIT — MelanoGraph v0

**Status:** COMPLETE
**Date:** 2026-09-28
**Auditor:** MelanoGraph v0 implementation pass

This document records what was inspected in the upstream projects *before* implementing
MelanoGraph. Every claim below was verified by reading or executing the upstream artefacts;
where a number is given it was measured, not assumed. Uncertainties are flagged explicitly.

---

## 1. Projects inspected

| # | Repository | Commit head inspected | Local path | Role |
|---|---|---|---|---|
| 1 | [seung-lab/FlyConnectome](https://github.com/seung-lab/FlyConnectome) | `main` @ 2024-04-09 (`depth=1`) | `third_party/FlyConnectome` | FlyWire data-access documentation |
| 2 | [chaobrain/drosophila_whole_brain_snn_simulation](https://github.com/chaobrain/drosophila_whole_brain_snn_simulation) | `main` @ 2025-12-02 (`depth=1`) | `third_party/drosophila_whole_brain_snn_simulation` | JAX/brainstate reimplementation + vendored dataset |
| 3 | [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) | `main` @ 2024-09-14 (`depth=1`) | `third_party/Drosophila_brain_model` | **Original** model (Brian 2) accompanying the Nature paper |

Publication: **Shiu, Sterne, Spiller, Franconville et al.,
"A Drosophila computational brain model reveals sensorimotor processing",
Nature 634, 210–219 (2024), DOI [10.1038/s41586-024-07763-9](https://doi.org/10.1038/s41586-024-07763-9)**.
The preprint version is referenced by repo #3's README with the title
"A leaky integrate-and-fire computational model based on the connectome of the entire adult
Drosophila brain reveals insights into sensorimotor processing"
(bioRxiv [10.1101/2023.05.02.539144](https://doi.org/10.1101/2023.05.02.539144)).

> Note: `gh repo clone` failed for all three repositories because the GitHub CLI is configured
> for the SSH protocol and this machine has no working SSH key for github.com. Cloning over
> HTTPS succeeded. This is an environment note, not a project finding.

---

## 2. Repository 1 — FlyConnectome (documentation only)

**Contents:** four Jupyter notebooks and a README. **No runnable library, no data.**

Files: `CAVE tutorial.ipynb`, `Neuropils and Point Lookups.ipynb`, `Renderings with MeshParty.ipynb`,
`Segmentation and Mesh Access.ipynb`, `README.md`.

**Findings (verified by reading README and the CAVE tutorial):**

- The EM and segmentation volumes "are too large for conventional downloads" and require
  `cloudvolume` subvolume reads. → **MelanoGraph must never attempt a raw-volume download.**
- Programmatic connectome access is via **CAVE** (`caveclient`), datastack name
  **`flywire_fafb_public`**, hosted at `prod.flywire-daf.com`.
- CAVE data is **versioned by materialization timestamp**; the client queries the latest
  materialization unless a specific `materialization_version` is given.
- Bulk download without a live client is available from **Codex**
  (`https://codex.flywire.ai/api/download`).
- Relevant CAVE tables mentioned: `proofread_neurons`, `nuclei_v1`, plus a filtered synapse view.
  The annotated synapse view carries per-neuron neurotransmitter evidence columns
  `gaba, ach, glut, oct, ser, da` and a `valid_nt` column.
- CAVE's `root_id` is the FlyWire segment/neuron ID.

**Consequence for v0:** FlyConnectome is a *reference for where data comes from*, not a
dependency. No CAVE client is installed and **no live network access is required to run MelanoGraph**
(objective §2).

---

## 3. Repository 3 — Drosophila_brain_model (the authoritative original)

This is the model the paper was produced with. It is **written against Brian 2**, not JAX.

### 3.1 Files

| File | Size | Notes |
|---|---|---|
| `model.py` | 12 KB | The model: parameters, Brian 2 network construction, experiment driver |
| `utils.py` | 2.2 KB | `load_exps`, `get_rate` — spike-rate analysis |
| `example.ipynb` | 10.4 KB | Tutorial that produced `results/example/*.parquet` |
| `figures.ipynb` | 32.5 KB | Code for the paper figures (the full protocol matrix) |
| `2023_03_23_completeness_630_final.csv` | 3110 KB | Dataset v630 neuron table |
| `2023_03_23_connectivity_630_final.parquet` | 84 600 KB | Dataset v630 connectivity |
| `Completeness_783.csv` | 3385 KB | Dataset v783 neuron table |
| `Connectivity_783.parquet` | 98 442 KB | Dataset v783 connectivity |
| `results/example/*.parquet` | 5 files | **Published spike output = ground truth for comparison** |
| `environment.yml`, `environment_full.yml` | | conda specs (Brian 2, joblib, pandas) |
| `sez_neurons.pickle` | 5 KB | SEZ cell-type → FlyWire ID lists (figure 2 input) |

### 3.2 Neuron dynamics — verbatim from `model.py`

```python
'eqs': '''
      dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
      dg/dt = -g / tau               : volt (unless refractory)
      rfc                            : second
      ''',
'eq_th' : 'v > v_th',
'eq_rst': 'v = v_rst; w = 0; g = 0 * mV',
```

- `NeuronGroup(..., method='linear', refractory='rfc')`
- `Synapses(neu, neu, 'w : volt', on_pre='g += w', delay=t_dly)`
- `syn.w = df_con['Excitatory x Connectivity'] * w_syn`

Note `eq_rst` also assigns `w`, which is **not a variable in this model** — a harmless leftover
from a Brian 2 template. The effective reset is `v = v_rst; g = 0 mV`.

### 3.3 Numerical parameters — verbatim, with the sourcing comments the upstream file carries

| Parameter | Value | Upstream-sourced basis (comment in `model.py`) |
|---|---|---|
| `t_run` | 1000 ms | trial duration |
| `n_run` | 30 | number of stochastic trials |
| `v_0` | −52 mV | Kakaria & de Bivort 2017, [10.3389/fnbeh.2017.00008](https://doi.org/10.3389/fnbeh.2017.00008) |
| `v_rst` | −52 mV | same as resting |
| `v_th` | −45 mV | — |
| `t_mbr` | 20 ms | inline: "capacitance × resistance = .002 µF × 10. MΩ" |
| `tau` (synaptic) | 5 ms | Jürgensen et al., [10.1088/2634-4386/ac3ba6](https://doi.org/10.1088/2634-4386/ac3ba6) |
| `t_rfc` (refractory) | 2.2 ms | Lazar et al., [10.7554/eLife.62362](https://doi.org/10.7554/eLife.62362) |
| `t_dly` (synaptic delay) | 1.8 ms | Paul et al. 2015, [10.3389/fncel.2015.00029](https://doi.org/10.3389/fncel.2015.00029) |
| `w_syn` | 0.275 mV | **explicitly labelled `# Free parameter`** |
| `r_poi` | 150 Hz | default Poisson stimulation rate |
| `r_poi2` | 0 Hz | second stimulation class |
| `f_poi` | 250 | "scaling factor for Poisson synapse; 250 is sufficient to cause spiking" |

**Verified:** the six parameter values quoted in the task prompt
(`tau_m=20 ms, tau_syn=5 ms, refractory=2.2 ms, V_rest=−52 mV, V_threshold=−45 mV,
synaptic delay≈1.8 ms`) **match the upstream source exactly**. The prompt was not treated as
authority; `model.py` was.

### 3.4 Stimulation semantics

`poi()` creates, **per stimulated neuron**, a `PoissonInput(target=neu[i], target_var='v', N=1,
rate=r_poi, weight=w_syn*f_poi)`, and sets that neuron's refractory period to `0 ms`.

Weight = `0.275 mV × 250 = 68.75 mV`, added directly to `v` (not to `g`).
Since `v_th − v_0 = 7 mV ≪ 68.75 mV`, every Poisson event drives the neuron over threshold;
stimulated neurons therefore emit an (approximately) rate-`r_poi` Poisson spike train.
`r_poi` is a **rate**, not a per-step probability — the correct per-step event probability is
`rate × dt` (0.015 at 150 Hz, dt = 0.1 ms).

`target_var='v'` is important: stimulation is a *membrane-potential* perturbation, whereas
network input is a *conductance* (`g`) perturbation. These are not interchangeable.

### 3.5 Silencing semantics — an upstream inconsistency worth recording

`silence()` docstring says it "sets all synaptic connections to and from those neurons to zero",
but the implementation is:

```python
for i in slnc:
    syn.w[' {} == i'.format(i)] = 0*mV
```

In Brian 2, `syn.w['<cond>']` masks on the **presynaptic** index. Therefore only **outgoing**
connections are zeroed. **The code, not the docstring, is authoritative** — and the published
results confirm it: in `sugarR-720575940622695448.parquet` the silenced neuron
`720575940622695448` is still the **most active neuron in the file** (3 439 spikes over 30 trials
≈ 114.6 Hz). A silenced neuron keeps spiking; it simply stops influencing others.

### 3.6 The `Excitatory x Connectivity` column is precomputed sign × count

The dataset already carries the excitatory/inhibitory decision per connection, so **MelanoGraph does
not need to derive neurotransmitter identity**, and does not need a CAVE query, to obtain
sign. See §5.

---

## 4. Repository 2 — drosophila_whole_brain_snn_simulation (JAX reimplementation)

**This is a reimplementation of repo 3 in `brainstate`/`brainevent`/`brainunit` (JAX).**

| File | Size | Notes |
|---|---|---|
| `drosophila_whole_brain.py` | 16.8 KB | `Population`, `Network`, `run_one_exp`, plotting helper |
| `brainstate_figures.ipynb` | 722 KB | Figure reproductions; contains the same input neuron lists |
| `2023_03_23_completeness_630_final.csv` | 2986 KB | **same content as repo 3's copy** |
| `2023_03_23_connectivity_630_final.parquet` | 84 600 KB | **byte-identical to repo 3's copy** |
| `sez_neurons.pickle` | 5 KB | identical purpose |

**Dataset identity check (SHA-256, measured):**

```
630 connectivity parquet  : 94db8c650533bc36ffa3223f2e62325d5648b8d6bd31c3a4e1c804628c7557b3
   -> identical in repo 2 and repo 3  (one source of truth)
630 completeness csv, repo2: e6b71e17671a9bdb05f55e4bc6774640a1418cb7a05125e0fc994ad40f9bfdfb
630 completeness csv, repo3: 479ed718404e7b1d898ee9a353410f481e6ed7694c30f579a7c2699f1c708a5d
   -> different bytes, identical content (127400 rows, same min/max index, same values);
      the difference is line-ending encoding, not data
783 connectivity parquet  : efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347
783 completeness csv      : 52b0ac6094cd32c546f8d4c341e094376f48f4e791f8db9b166de5dff8199ea4
```

### 4.1 Differences from the original model (these matter)

| Aspect | Original (Brian 2, paper) | chaobrain (JAX) |
|---|---|---|
| Integration | `method='linear'` — **exact** solution of the coupled linear system over `dt` | `exp_euler_step` per equation, `g` explicit |
| Sub-step order | integrate → threshold → reset → deliver synaptic events | integrate `v` (using old `g`) → integrate `g` → `g += x` → threshold → reset |
| Spike nonlinearity | hard `v > v_th` | `ReluGrad` surrogate (forward value identical to a step function) |
| Silencing | zero outgoing synaptic weights | multiply delayed presynaptic spike vector by a boolean mask |
| Stimulus | `PoissonInput(target_var='v', N=1)` | `brainstate.nn.poisson_input(..., target=self.pop.v, num_input=1)` |
| Refractory on stimulated neurons | `0 ms` | `0 ms` (comment says "0.5 ms" — the comment is wrong) |
| Weights | `Excitatory × Connectivity × w_syn` | identical |

The `ReluGrad` surrogate changes only the backward pass; for forward simulation it is equivalent
to a hard threshold. Silencing via the presynaptic mask is algebraically identical to zeroing the
outgoing weights. The **integration scheme and the sub-step ordering are genuinely different**,
so bitwise agreement between the two upstream implementations should not be expected.

A second upstream inconsistency: the chaobrain `Network.__init__` sets
`self.pop.tau_ref[exc['indices']] = 0. * u.ms` with the inline comment "set refractory period to
0.5 ms". The code sets **0 ms**, matching the original. The comment is wrong.

### 4.2 Independent confirmation of the parameter set

Repo 2's README independently lists the same six constants, and its `Population` class carries the
same literature citations as repo 3's `default_params`. The two upstream implementations agree on
parameters, so these are not a single-source artefact.

---

## 5. Dataset format (the decisive finding for the data strategy)

Both connectivity files have this exact schema (7 columns + a pandas index column):

```
Presynaptic_ID          : int64     # FlyWire root ID of the presynaptic neuron
Postsynaptic_ID         : int64     # FlyWire root ID of the postsynaptic neuron
Presynaptic_Index       : int64     # integer index into the completeness CSV row order
Postsynaptic_Index      : int64     # ditto
Connectivity            : int64     # number of synapses for this ordered neuron pair, >= 1
Excitatory              : int64     # strictly +1 or -1
Excitatory x Connectivity : int64   # signed synapse count = Excitatory * Connectivity
```

The completeness CSV is `index = FlyWire root ID`, single column `Completed = True`.

**Measured properties of dataset v630 (127 400 neurons):**

| Quantity | Measured value |
|---|---|
| neurons | 127 400 |
| connectivity rows (directed neuron-pair edges) | 14 687 178 |
| total chemical synapses (`Connectivity.sum()`) | 52 793 639 |
| mean synapses per connected pair | 3.5945 |
| excitatory edges (`Excitatory == +1`) | 8 800 532 |
| inhibitory edges (`Excitatory == -1`) | 5 886 646 |
| duplicate `(pre, post)` pairs | **0** |
| self-loops (`pre == post`) | **0** |
| neurons with no outgoing edge | 385 (index range is still 0…127 399) |
| neurons with no incoming edge | 607 |
| neurons with neither | **0** |
| distinct `Presynaptic_Index` values | 127 015 |
| distinct `Postsynaptic_Index` values | 126 793 |
| neurons with ≥1 excitatory outgoing edge | 86 543 |
| neurons with ≥1 inhibitory outgoing edge | 40 472 |
| neurons with **both** excitatory and inhibitory outgoing edges | **0** |
| NaN / zero / negative `Connectivity` | 0 / 0 / 0 |
| `Excitatory` distinct values | exactly `{-1, +1}` |

Two consequences that shape MelanoGraph's design:

1. **Sign is a property of the presynaptic neuron, not of the individual connection.**
   No neuron has both excitatory and inhibitory outgoing edges. This is a Dale's-principle
   encoding: 86 543 purely excitatory and 40 472 purely inhibitory presynaptic neurons, which sum
   to exactly the 127 015 neurons that have any outgoing edge. MelanoGraph therefore reports
   "excitatory neurons" as *presynaptic neurons whose outgoing edges are excitatory*, and must
   record the remaining **385 neurons as `unknown_sign`** (no outgoing edge ⇒ no sign evidence).
2. **The "~50 million synaptic connections" headline figure is the synapse count, not the edge
   count.** The graph has 14.69 M directed edges. A CSR built from the edge list needs
   ~3 × 4 bytes × 14.69 M ≈ 176 MB, **not** the multi-GB that a 50 M-edge CSR would need.

**Dataset v783 additionally available locally:** 138 639 neurons, 15 091 983 edges,
60 326 81 inhibitory / 9 059 302 excitatory edges (measured). It is a plain file in repo 3 —
**no importer needs to be written**; v0 registers it but uses v630 as the primary dataset because
v630 is the version the paper used and the version for which published reference output exists.

**No live network access is required after the files are present.** The objective's pipeline

```
external FlyWire data -> import/preprocess -> versioned local dataset -> MelanoGraph simulation
```

collapses to "stage the two upstream files + verify hashes + derive a versioned processed CSR",
because the upstream project already ships the FlyWire-derived, preprocessed dataset.

---

## 6. Published reference output (ground truth for qualitative comparison)

`third_party/Drosophila_brain_model/results/example/` contains **real published spike output**.
Protocol read from `example.ipynb`: 30 trials × 1000 ms, network built fresh per trial,
default dt (Brian 2 default = 0.1 ms), `neu_exc = neu_sugar` (21 labellar sugar GRNs, right
hemisphere), and for the silencing files `neu_slnc = [i]` for one neuron at a time.

Measured from the files:

| File | Stimulus | trials | spikes | active neurons | MN9-L rate | MN9-R rate | median sugar-GRN rate |
|---|---|---|---|---|---|---|---|
| `sugarR.parquet` | sugar GRN | 30 | 511 566 | 448 | **93.27 Hz** | 61.80 Hz | 197.3 Hz |
| `sugarR_100Hz.parquet` | sugar GRN @100 Hz | 30 | 289 073 | 404 | **67.03 Hz** | 48.63 Hz | 98.8 Hz |
| `sugarR-720575940617937543.parquet` | sugar @100 Hz, silence `…937543` | 30 | 277 853 | 414 | 63.27 Hz | 45.67 Hz | 99.0 Hz |
| `sugarR-720575940621754367.parquet` | sugar @100 Hz, silence `…754367` | 30 | 287 004 | 402 | 66.90 Hz | 49.70 Hz | 99.5 Hz |
| `sugarR-720575940622695448.parquet` | sugar @100 Hz, silence `…695448` | 30 | 300 963 | 413 | 71.17 Hz | 51.00 Hz | 98.6 Hz |

MN9 IDs: `720575940660219265` (left), `720575940645521262` (right).

**Calibration note.** `example.ipynb`'s markdown states "By default, the neurons are excited at
200 Hz", but `model.py`'s current `default_params['r_poi'] = 150 Hz`. The measured median sugar-GRN
rate in `sugarR.parquet` is **197.3 Hz**, i.e. the committed `sugarR.parquet` was produced at
**200 Hz**, before the default was changed to 150 Hz. The file, not the markdown, is authoritative.

**Second upstream inconsistency.** `example.ipynb` cell #18 comments that the three most active
neurons in `sugarR_100Hz` "are all sugar-sensing neurons". Measured: `720575940617937543` and
`720575940621754367` **are** members of `neu_sugar`; `720575940622695448` **is not**. The
`neu_sugar` list is identical in repo 2 and repo 3's `figures.ipynb`, so this is a stale comment
or a changed input list, not a data discrepancy.

**This table is the acceptance target for MelanoGraph's sugar experiment.** The objective only asks
for *qualitative* agreement ("known sensory neurons → stimulation → activity propagates → known
downstream populations respond"), so MelanoGraph's target is: sparse downstream activation
(order 10² active neurons), sugar GRNs firing at ≈ `r_poi`, and MN9 measurably driven.

---

## 7. What is directly from FlyWire vs. what is modelling assumption

| Element | Provenance |
|---|---|
| Neuron set, FlyWire root IDs, index order | **FlyWire** (via the materialized completeness table) |
| Directed connectivity, synapse counts per pair | **FlyWire** (proofread connectome, synapse-level) |
| Excitatory/inhibitory sign | **FlyWire-derived annotation** (predicted neurotransmitter identity), precomputed upstream into `Excitatory` |
| LIF form `dv/dt = (v_0 − v + g)/t_mbr`, `dg/dt = −g/tau` | **Modelling assumption** (two-variable LIF with a conductance-like `g` in mV) |
| `t_mbr`, `tau`, `t_rfc`, `t_dly`, `v_0`, `v_th` | **Modelling assumption** with literature justification (citations in §3.3) |
| `w_syn = 0.275 mV` | **Free parameter**, explicitly labelled as such; "modulated by exponential decay" |
| `f_poi = 250` | **Free parameter**, justified only by "250 is sufficient to cause spiking" |
| Uniform `w_syn` for every neuron | **Strong modelling assumption** — no per-neuron or per-cell-type weight heterogeneity |
| Synapse count → weight linearly | **Modelling assumption**; `Excitatory × Connectivity × w_syn` |
| Uniform 1.8 ms delay for every synapse | **Modelling assumption** (biologically implausible, stated as such upstream) |
| `v_rst = v_0` | **Modelling assumption** |
| No adaptation, no inhibition reversal potential, no gap junctions | **Known omissions** |
| Stimulation = Poisson events adding fixed mV to `v` | **Modelling assumption** (a surrogate for optogenetic activation) |
| Silencing = zero outgoing synaptic weights | **Modelling assumption** (a surrogate for optogenetic silencing) |

---

## 8. Design decisions taken from this audit

1. **Dataset:** use v630 (paper version, has published ground truth). Register v783 as a second
   dataset without promoting it.
2. **Data acquisition:** stage the upstream files; verify SHA-256; derive a processed CSR.
   No CAVE client, no live download required, no raw EM volume.
3. **Do not depend on Brian 2 or brainstate/brainunit/brainevent.** MelanoGraph implements the
   verified equations directly in NumPy (and optionally JAX). Rationale: (a) the objective forbids
   rewriting a validated simulator *unnecessarily* but explicitly requires our own clean
   `model/lif.py`/`model/network.py` and a project that "installs from scratch"; (b) `brainevent`
   compiles custom CUDA kernels and `brainstate` pins a moving JAX API — both are avoidable
   dependency risk on this Windows machine; (c) the model is ~40 lines of arithmetic, so a
   self-contained implementation is *more* auditable than an indirection through a framework.
   The equations, constants, weight rule, delay, stimulation and silencing semantics are copied
   from the original, and the deviations are documented in `MODEL_ASSUMPTIONS.md`.
4. **Integration scheme:** implement the **exact** exponential solution of the coupled linear
   system, matching Brian 2's `method='linear'` (the scheme that produced the paper). Do **not**
   copy the chaobrain sequential `exp_euler` variant.
5. **Sub-step ordering:** integrate → threshold → reset → deliver delayed synaptic events →
   deliver stimulus events. This matches Brian 2 ordering and the `when='synapses'` default of
   `PoissonInput`. The chaobrain ordering differs; documented as a known divergence.
6. **Silencing:** outgoing-only, matching upstream behaviour (and confirmed by the published
   results). The docstring-level claim of "and from" is treated as upstream's error.
7. **Sign:** read `Excitatory` directly; report `unknown_sign` = neurons with no outgoing edge.
8. **Sparse-only:** never materialize a dense 127 400² matrix (16.2 × 10⁹ entries ≈ 65 GB float32).
   Build CSR from the edge list; this is also an explicit objective requirement.
9. **No GPU claim without evidence:** see `docs/ARCHITECTURE.md`; JAX on native Windows is
   CPU-only, which was verified experimentally and is not worked around by pretending otherwise.

---

## 9. Open uncertainties (preserved, not resolved)

- `w_syn = 0.275 mV` is a free parameter. Its value is not derived from measurement, and the
  upstream comment "modulated by exponential decay" is unexplained. MelanoGraph treats it as
  configurable and records it in every run.
- Whether the paper's `method='linear'` and a hard threshold can drift from Brian 2's internal
  ordering in a way that changes spike identity at the single-timestep level is **not verified
  here**; bitwise agreement is not claimed.
- The upstream connectivity table's synapse counts are for the *proofread* subset of the
  connectome (version 630, "completeness" threshold 630). Non-proofread tissue is excluded.
  MelanoGraph inherits this limitation.
- `Excitatory` is a predicted neurotransmitter identity, not a measurement. Its accuracy is not
  characterised in the upstream repository and is inherited as-is.
- The 1.8 ms uniform delay is physiologically implausible for a whole brain but is the upstream
  choice and is retained for reproducibility.
