# Input contracts: legacy_v0 and v1

The default `LIFNetwork` contract remains `legacy_v0`. Select
`contract_version="v1"` explicitly for corrected target transitions and continuous
sensory inputs. `describe()` reports the selection. No historical golden file or
dataset is replaced.

`apply_stimulus` remains a scheduled, strong voltage intervention in both modes.
It rebuilds samplers, including their RNGs. Targets have zero refractory period
for the entire declaration, including outside the activation window. In v1 every
replacement starts from the default refractory vector and then zeros only the
current targets. Legacy v0 intentionally retains the historical sticky zeros on
nonempty replacements; clear restores defaults and reset reapplies only current
declarations. This preserves replay access to the original behavior.

The historical `Stimulus.spec_hash` includes IDs, rate, start/end and label. Its
32-bit stream seed therefore depends on all these fields. Drawing before masking
does not make a changed configuration's stream independent of its window. Only the
incorrect documentation has changed in that sampler.

## Continuous sensory input v1

```python
brain = LIFNetwork(connectome, backend=NumpyBackend(), seed=47,
                   contract_version="v1")
brain.declare_input_channel(
    "sensor-0", neuron_ids, experiment_id="pilot-1", agent_id="individual-a",
    rate_hz=150, gain_mv=0.275,
)
brain.set_input_rates("sensor-0", 220)
```

This is an explicitly modeled conductance input, applied to `g` after recurrence.
It does not alter refractory periods. Positive and negative finite gains are
allowed. It is not a validated naturalistic sensor. Target IDs are sorted and
deduplicated at declaration; rate vectors follow that canonical order. Scalars
broadcast to all targets. Targets are fixed; duplicate channel declarations fail.

The stream seed is the first 128 bits (big endian) of SHA-256 over the UTF-8 JSON
array `["flybrain.input.v1", base_seed, experiment_id, agent_id, channel_id]`, with
compact separators. Labels, windows, rates and gains do not enter identity.
Give independent individuals distinct `agent_id` values; the API cannot infer
whether two intentionally identical identities describe a pair or a replay.

Each channel owns a NumPy PCG64 generator and draws Poisson counts every neural
step before masking its window. Updating rates does not draw, replace the sampler
or reset the generator. NumPy Poisson uses a rate-dependent number of random
draws; different rate histories are **not** guaranteed common subsequent draws.
A zero mean can consume no randomness. Cross-language equality of Poisson streams
is not promised. Differential kernel tests should use prerecorded events.

`reset_input_rng(channel_id)` resets only that stream using the current network
seed; omitting the ID resets all sensory streams. `reset(seed)` resets physiology
and all RNGs while preserving channel definitions/current rates, scheduled
stimuli and silencing. Channels execute in sorted ID order, making accumulation
independent of declaration order. Separate networks never share mutable channels.

Tests: `tests/test_input_contracts.py` covers transitions, reset, legacy identity,
window/label independence, setter atomicity and individual/declaration ordering.
