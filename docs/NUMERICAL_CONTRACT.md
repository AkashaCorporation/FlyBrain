# Numerical contract for the first cycle

Reference source: commit `30ad237735698c2097dfeaf2f8411b75a33e8d49`,
NumPy backend, float32 state. Historical archive and dataset/source hashes are
in `outputs/first_cycle/initial_manifest.json` and `v0-30ad237.zip`.

At step `k`, `t_ms = k * dt_ms`:

1. Freeze both `v` and `g` where `(t_ms - t_last) <= tau_ref`. Otherwise use the
   exact linear coefficients computed in float64 then stored as float32. Preserve
   the expression's separate float32 subtract/multiply/add operations.
2. Spike only if integrated `v > v_threshold` and not refractory (strict threshold).
3. Reset spiking `v` to reset, `g` to zero; store spike time in the state's dtype.
4. Apply each scheduled stimulus to `v` after threshold, in sampler order.
5. Read the old ring slot, write current spikes, advance cursor modulo delay.
   Recurrence uses the old spikes, with outgoing kill mask applied at delivery.
   A spike at k delivers conductance at k+delay_steps, after that step's threshold.
6. In v1, apply sensory conductance channels in sorted channel-ID order.
7. Increment counts/step and retain current spikes.

The existing `lif.py` overview lists recurrence before stimulus; the actual
`network.py` schedule lists stimulus first. For the legacy independent `v` and `g`
updates the distinction has no state effect. The list above specifies executable
order, including the new sensory stage.

`silence` removes outgoing synaptic influence only. It does not clamp voltage,
erase spikes, suppress incoming input or block a direct readout. Reset retains
declared silencing/stimuli. Recurrence can be explicitly disabled.

`dt` must divide 1.8 ms into at least one integral delay step (validation tolerance
1e-9 in the ratio). 0.5 and 1 ms are rejected. Defaults are unchanged. Floating
point refractory boundaries are part of the executable reference; moving time to
integer ticks requires separate validation.

## Precision and equivalence

For NumPy recurrence each edge contribution is float32. `bincount` reduces those
contributions in float64, in edge-array order, then narrows the sum to float32
before adding it to `g`. A float32 accumulator is a different algorithm. The
cancellation fixture `[2**24, 1, -2**24]` produces one in the reference and zero
under naive float32 running addition.

E0 means exact state/event bytes under stated versions, inputs and scope. E1 means
numeric tolerances specified before comparison, with the first divergent step and
field reported. E2 concerns distributions and experimental effects; it cannot
repair a failing kernel contract. Hashes have no tolerance. No new whole-brain
spike digest or complete reproduction of the paper is claimed by synthetic tests.
