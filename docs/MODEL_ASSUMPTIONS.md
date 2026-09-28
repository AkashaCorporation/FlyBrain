# MODEL_ASSUMPTIONS

**Status:** complete for v0
**Scope:** every number and every modelling choice that determines FlyBrain's
simulated activity, with its source and a confidence label.

FlyBrain does not claim to be a faithful digital copy of a fly. It implements one
specific, published, deliberately simplified model and says exactly which parts are
data and which are assumptions. The task prompt that commissioned this work listed
six parameter values; those values were treated as *claims to verify*, not as
authority, and all six were confirmed against the upstream source
(see [`UPSTREAM_AUDIT.md`](UPSTREAM_AUDIT.md) §3.3).

The table below is generated from `flybrain/model/lif.py::PARAMETER_PROVENANCE`, and
`tests/test_docs.py` fails if this document and the code disagree. Editing one
without the other is therefore impossible to do silently: run
`python scripts/render_doc_tables.py` to regenerate.

---

## 1. Neuron dynamics

Taken verbatim from `third_party/Drosophila_brain_model/model.py`
(`default_params['eqs']`), which is the implementation that produced
Shiu et al., *Nature* **634**, 210–219 (2024):

```
dv/dt = (v_rest - v + g) / tau_membrane     [mV], unless refractory
dg/dt = -g / tau_synapse                    [mV], unless refractory
threshold : v > v_threshold
reset     : v := v_reset ; g := 0
```

`g` is a conductance-like variable carried in mV, as upstream defines it. It is not
a current and it is not in nS. The reason this matters is that stimulation is
delivered to `v` while network input is delivered to `g`, and the two are not
interchangeable.

### Integration scheme

Upstream runs Brian 2 with `method='linear'`. Both ODEs are linear with constant
coefficients, so `linear` means "solve the system exactly over the step". FlyBrain
implements the same exact solution rather than an Euler approximation:

With `x = (v, g)`, the system is `dx/dt = A x + b` for

```
A = [[-1/tau_membrane,  1/tau_membrane],      b = [v_rest/tau_membrane, 0]
     [               0, -1/tau_synapse ]]
```

whose unique fixed point is `x* = -A^-1 b = (v_rest, 0)`. Hence
`x(t+dt) - x* = expm(A dt)(x(t) - x*)`, and for this upper-triangular `A` with
distinct eigenvalues `a = -1/tau_membrane`, `d = -1/tau_synapse`:

```
v(t+dt) = e^{a dt} (v - v_rest) + K g + v_rest
g(t+dt) = e^{d dt} g
K       = c (e^{a dt} - e^{d dt}) / (a - d),      c = 1/tau_membrane
```

Two properties pin this down and are asserted in the tests: `(v_rest, 0)` is an
*exact* fixed point (a wrong `K` or a sign slip breaks it immediately), and the
small-`dt` limit reproduces `dv/dt = (v_rest - v + g)/tau_membrane`.

**A note on a common slip.** For a conductance impulse `g(t) = g0 e^{-t/tau_syn}`,
the membrane excursion is

```
v(t) - v_rest = g0 * tau_syn/(tau_membrane - tau_syn) * (e^{-t/tau_membrane} - e^{-t/tau_syn})
```

The coefficient is `tau_syn/(tau_membrane - tau_syn)`, **not**
`tau_membrane/(tau_membrane - tau_syn)`. The wrong form overstates the excursion by
`tau_membrane/tau_syn` (= 4 for the default constants). This is not pedantry: it
moves the number of synapses a single presynaptic spike needs to cross threshold
from ~41 to ~162, which changes whether a network looks connected or not.

### Sub-step ordering

Brian 2 runs its objects in the order state-updater → thresholder → resetter →
synapses, and `PoissonInput` defaults to `when='synapses'`. FlyBrain therefore runs,
within each `dt`:

1. integrate exactly, skipping refractory neurons;
2. threshold: `v > v_threshold` **and** `not_refractory`;
3. reset: `v := v_reset`, `g := 0` where a spike occurred;
4. stimulus events: `v += k * w_stim` for Poisson events drawn this step;
5. synaptic events: `g += W @ spikes(t - delay)`.

Steps 4 and 5 touch different variables, so their relative order is immaterial. One
consequence is worth stating: a stimulus event delivered at step `k` can first cause
a spike at step `k+1`. The upstream model has the same one-step latency, so the
emitted *rate* is unaffected.

**The threshold is gated on `not_refractory`.** This is not cosmetic. Brian 2's
Thresholder ANDs its condition with `not_refractory`, so a neuron held above
threshold still cannot fire inside its refractory window. Without the gate, a
suprathreshold neuron fires every single step and the refractory period silently
stops existing. The chaobrain JAX port does not gate it
(`docs/UPSTREAM_AUDIT.md` §4.1), so this is a point where FlyBrain follows Brian 2
and diverges from that port.

### Refractoriness

The condition is `(t - t_last_spike) <= refractory` (Brian 2 spells the same thing
`not_refractory = (t - t_last_spike) > refractory`). While refractory, **both** `v`
and `g` are frozen, because upstream marks both equations `(unless refractory)`.

Incoming synaptic events are still applied to `g` during refractoriness, because
`on_pre` code is not gated by the refractory flag in Brian 2. At the default
`dt = 0.1 ms`, a neuron that spikes at step `j` is frozen for steps `j+1 .. j+22`
(22 steps = 2.2 ms) and resumes integrating at step `j+23`.

### Reset

`v_reset = v_rest = -52 mV`. There is no hyperpolarising after-potential. Upstream's
reset string also assigns `w = 0`, but `w` is not a variable in this model — a
harmless leftover from a Brian 2 template. FlyBrain does not invent a `w`.

---

## 2. Synapses

| aspect | value | provenance |
|---|---|---|
| weight per edge | `signed_synapse_count × w_syn` | upstream: `syn.w = df_con['Excitatory x Connectivity'] * w_syn` |
| sign | from the dataset column `Excitatory` ∈ {−1, +1} | FlyWire-derived annotation, precomputed upstream |
| delay | uniform `1.8 ms` on every edge | upstream `t_dly`, cited to Paul et al. 2015 |
| plasticity | none | upstream has none |
| gap junctions | none | upstream has none |
| conduction-velocity variation | none | uniform delay; physiologically implausible, stated as such upstream |

The dataset's **sign is a property of the presynaptic neuron**, not of the
individual connection: no neuron in the published data has both excitatory and
inhibitory outgoing edges (`outputs/dataset_report.json`, `neurons_with_mixed_sign
= 0`). This is a Dale's-principle encoding and it is inherited, not verified
biologically.

**Convergence, not single connections, drives the network.** The median connection
in the dataset carries 3.59 synapses, i.e. ≈0.99 mV of conductance, which produces a
peak membrane excursion of ≈0.16 mV against a 7 mV rest-to-threshold gap. A *single*
presynaptic spike needs ≈162 synapses onto the same target to cross threshold on its
own. Downstream neurons therefore respond to convergent input only, which is
consistent with the published model activating only a few hundred neurons when the
sugar GRNs are driven.

---

## 3. Stimulation

Upstream (`model.py::poi`) creates, per stimulated neuron, a
`PoissonInput(target=neu[i], target_var='v', N=1, rate=r_poi, weight=w_syn*f_poi)`
and sets that neuron's refractory period to `0 ms`.

Consequences FlyBrain reproduces:

* events are **Poisson**, with per-step event count `Poisson(rate × dt)` per target
  neuron (0.015 at 150 Hz, dt = 0.1 ms) — `r_poi` is a rate, not a per-step
  probability;
* events are added to the **membrane potential `v`**, not to `g`;
* the amplitude is `w_syn × f_poi = 0.275 mV × 250 = 68.75 mV`, which always crosses
  the 7 mV rest-to-threshold gap in a single event, so a stimulated neuron emits an
  approximately rate-`r_poi` Poisson spike train;
* stimulated neurons have **no refractory period**, so they may fire on adjacent
  steps (verified in `tests/test_model_network.py`).

`f_poi = 250` is a free parameter justified upstream only by the comment "250 is
sufficient to cause spiking". FlyBrain gives it no independent justification.

---

## 4. Silencing

Modelled as removing the **outgoing** influence of the silenced neurons, by zeroing
the synaptic weight of every connection *from* them. Upstream's docstring claims
"to and from", but its code masks on the presynaptic index, and the published
results confirm the code: in `sugarR-720575940622695448.parquet` the silenced neuron
is the *most active neuron in the file*, so it clearly keeps spiking
(`UPSTREAM_AUDIT.md` §3.5).

Consequences, both confirmed:

* a silenced neuron still receives input and may still spike;
* it stops influencing other neurons, which is the causal content of the
  intervention.

FlyBrain implements this by multiplying the delayed presynaptic spike vector by a
per-neuron kill mask, which is algebraically identical to zeroing the outgoing
weights and O(N) rather than O(E).

---

## 5. Integration timestep

`dt = 0.1 ms`, which is Brian 2's `defaultclock.dt`; the upstream model never
overrides it. `dt` is required to divide the 1.8 ms synaptic delay exactly, so the
implemented delay equals the specified delay. Valid choices for the default delay
include 0.1, 0.2, 0.6 and 0.9 ms; `LIFParams.validate` rejects anything else rather
than silently rounding.

---

## 6. Parameters

Every value below is checked against the code by `tests/test_docs.py`.
`confidence` means:

* **cited** — the value comes from a cited measurement or a documented derivation;
* **free** — upstream explicitly labels it a free parameter, or gives only a
  plausibility argument;
* **derived** — arithmetic on other quantities.

<!-- BEGIN GENERATED PARAMETER TABLE -->
| FlyBrain field | value | unit | source | reason | confidence |
|---|---|---|---|---|---|
| `v_rest_mv` | -52.0 | mV | upstream model.py default_params; comment cites Kakaria & de Bivort 2017, https://doi.org/10.3389/fnbeh.2017.00008 | resting potential of the model neuron | cited |
| `v_reset_mv` | -52.0 | mV | upstream model.py default_params (`eq_rst`) | post-spike reset; upstream sets it equal to rest (no hyperpolarising reset) | free |
| `v_threshold_mv` | -45.0 | mV | upstream model.py default_params (`eq_th`) | spike threshold; 7 mV above rest | free |
| `tau_membrane_ms` | 20.0 | ms | upstream model.py default_params; inline comment 'capacitance * resistance = .002 * uF * 10. * Mohm' | membrane time constant of the model neuron | cited |
| `tau_synapse_ms` | 5.0 | ms | upstream model.py default_params; comment cites Jurgensen et al., https://doi.org/10.1088/2634-4386/ac3ba6 | post-synaptic conductance decay constant | cited |
| `refractory_ms` | 2.2 | ms | upstream model.py default_params; comment cites Lazar et al., https://doi.org/10.7554/eLife.62362 | absolute refractory period | cited |
| `synaptic_delay_ms` | 1.8 | ms | upstream model.py default_params; comment cites Paul et al. 2015, https://doi.org/10.3389/fncel.2015.00029 | uniform axonal + synaptic transmission delay applied to every connection | cited |
| `weight_per_synapse_mv` | 0.275 | mV | upstream model.py default_params, inline comment literally '# Free parameter' | per-synapse weight; modulated by exponential decay. NOT derived from measurement - it is the single calibrated knob that sets network excitability | free |
| `stimulus_rate_hz` | 150.0 | Hz | upstream model.py default_params; the published tutorial runs used 200 Hz (see docs/UPSTREAM_AUDIT.md section 6) | default rate of the Poisson stimulation applied to target neurons | free |
| `stimulus_weight_scale` | 250.0 | dimensionless | upstream model.py default_params, inline comment '250 is sufficient to cause spiking' | Poisson events are delivered to v, not g, so they need a larger amplitude than a single synapse. 0.275 mV * 250 = 68.75 mV, which always crosses the 7 mV rest-to-threshold gap in one event | free |
| `dt_ms` | 0.1 | ms | Brian 2's default clock; the upstream model never overrides it | integration timestep used for the published results | cited |
<!-- END GENERATED PARAMETER TABLE -->

### Upstream key mapping

Because the two projects name the same quantities differently, the mapping is
recorded so it can never be inferred wrongly:

| upstream key (`model.py`) | FlyBrain field | value |
|---|---|---|
| `v_0` | `v_rest_mv` | -52.0 |
| `v_rst` | `v_reset_mv` | -52.0 |
| `v_th` | `v_threshold_mv` | -45.0 |
| `t_mbr` | `tau_membrane_ms` | 20.0 |
| `tau` | `tau_synapse_ms` | 5.0 |
| `t_rfc` | `refractory_ms` | 2.2 |
| `t_dly` | `synaptic_delay_ms` | 1.8 |
| `w_syn` | `weight_per_synapse_mv` | 0.275 |
| `r_poi` | `stimulus_rate_hz` | 150.0 |
| `f_poi` | `stimulus_weight_scale` | 250.0 |
| `(brian2 defaultclock.dt)` | `dt_ms` | 0.1 |

### Which parameters are not derived from measurement

`w_syn` and `f_poi` are the two numbers that set how excitable the network is, and
neither is derived from a measurement. `w_syn` is labelled a free parameter in the
source, and `f_poi` is justified only by "250 is sufficient to cause spiking". Every
absolute firing rate FlyBrain reports inherits that uncertainty. This is the single
most important caveat in this document.

---

## 7. Deliberate deviations from the two upstream implementations

| aspect | original (Brian 2) | chaobrain (JAX) | FlyBrain | why |
|---|---|---|---|---|
| integration | exact linear | sequential exponential Euler | **exact linear** | matches the implementation that produced the paper |
| sub-step order | integrate → threshold → reset → synapses | `v` integrated with old `g`, then `g`, then input | **Brian 2 order** | documented in §1 |
| refractory gates the threshold | yes | no | **yes** | §1; a clamped neuron must not fire through its refractory window |
| spike nonlinearity | hard `v > v_th` | `ReluGrad` surrogate | **hard** | forward value is identical; the surrogate only exists for gradients, and FlyBrain v0 has no gradients |
| silencing | zero outgoing weights | presynaptic kill mask | **kill mask** | algebraically identical, cheaper |
| stimulation | `PoissonInput` on `v` | `poisson_input` on `v` | **Poisson draws on `v`** | same semantics |
| stimulated refractory | 0 ms | 0 ms (comment says 0.5 ms) | **0 ms** | code, not comment |
| RNG | Brian 2's `PoissonInput` | brainstate's RNG | **NumPy PCG64, per-stimulus seed** | reproducible and order-independent |

FlyBrain is therefore expected to agree with the published *qualitative* behaviour
and **not** to reproduce the published numbers bit-for-bit. The differences that
matter most are the RNG (different Poisson realisations) and the trial count
(published runs used 30 trials; FlyBrain defaults to 1). See
[`STATUS.md`](STATUS.md) for the measured comparison.

---

## 8. What this model does **not** include

No synaptic plasticity, no adaptation or spike-frequency adaptation, no inhibition
reversal potential, no gap junctions, no conduction-velocity variation, no
neuromodulation, no intrinsic noise, no glia, no developmental or hormonal state, no
temperature dependence, no per-cell-type parameters, no cell-type annotation at all.
Every neuron in the model is the same neuron.

See [`SCIENTIFIC_BOUNDARIES.md`](SCIENTIFIC_BOUNDARIES.md) for what may and may not be
concluded from its output.
