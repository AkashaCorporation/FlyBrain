"""Hand-built small connectomes.

Used by the smoke test and by the unit tests, for two reasons:

1. The dynamical claims (threshold, refractory, delay, propagation, sign,
   silencing) need to be checked against *known* graphs where the correct answer
   can be computed by hand. A 127 400-neuron dataset cannot do that.
2. The smoke test must run without the dataset present at all, so a broken data
   path cannot mask a broken simulation kernel.

The graphs produced here satisfy every :class:`~flybrain.data.loader.Connectome`
invariant, so they exercise exactly the same code path as real data.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from . import schema
from .loader import Connectome

# A small, human-readable ID base far from the real FlyWire range, so a synthetic
# ID can never be mistaken for a real neuron in a log or a report.
SYNTHETIC_ID_BASE = 1_000_000


def make_synthetic_connectome(
    edges: Iterable[Tuple[int, int, int]],
    n_neurons: int,
    *,
    dataset_id: str = "synthetic",
    flywire_ids: Optional[Sequence[int]] = None,
    synapse_count_multiplier: int = 1,
) -> Connectome:
    """Build a connectome from an explicit edge list.

    Parameters
    ----------
    edges:
        ``(presynaptic_index, postsynaptic_index, signed_synapse_count)`` triples.
        The signed count is positive for excitatory and negative for inhibitory,
        matching the semantics of the dataset column ``Excitatory x Connectivity``.
    n_neurons:
        Size of the index space; neurons may be isolated.
    flywire_ids:
        Optional explicit ID array. Defaults to ``SYNTHETIC_ID_BASE + i``.
    synapse_count_multiplier:
        Convenience for making the graph strong enough to propagate in a handful of
        steps. Multiplies every signed count; the *sign* is preserved.
    """
    e = [(int(a), int(b), int(w) * synapse_count_multiplier) for a, b, w in edges]
    if any(a < 0 or a >= n_neurons or b < 0 or b >= n_neurons for a, b, _ in e):
        raise ValueError("synthetic edge endpoint outside [0, n_neurons)")
    if any(w == 0 for _, _, w in e):
        raise ValueError("synthetic edges must have a non-zero signed synapse count")

    if flywire_ids is None:
        ids = np.arange(SYNTHETIC_ID_BASE, SYNTHETIC_ID_BASE + n_neurons, dtype=np.int64)
    else:
        ids = np.asarray(flywire_ids, dtype=np.int64)
        if ids.size != n_neurons:
            raise ValueError("flywire_ids must have exactly n_neurons entries")

    if e:
        pre = np.fromiter((a for a, _, _ in e), dtype=schema.DTYPE_NEURON_INDEX, count=len(e))
        post = np.fromiter((b for _, b, _ in e), dtype=schema.DTYPE_NEURON_INDEX, count=len(e))
        w = np.fromiter((c for _, _, c in e), dtype=np.int64, count=len(e))
        order = np.argsort(pre, kind="stable")
        pre, post, w = pre[order], post[order], w[order]
    else:
        pre = np.empty(0, dtype=schema.DTYPE_NEURON_INDEX)
        post = np.empty(0, dtype=schema.DTYPE_NEURON_INDEX)
        w = np.empty(0, dtype=np.int64)

    info = np.iinfo(schema.DTYPE_SIGNED_COUNT)
    if w.size and (w.min() < info.min or w.max() > info.max):
        raise ValueError("signed synapse count overflows the int16 dtype contract")

    return Connectome(
        dataset_id=dataset_id,
        flywire_ids=ids,
        pre=pre,
        post=post,
        signed_count=w.astype(schema.DTYPE_SIGNED_COUNT),
    )


def chain_connectome(
    n_neurons: int = 4,
    *,
    synapses: int = 200,
    sign: int = 1,
) -> Connectome:
    """A feed-forward chain ``0 -> 1 -> 2 -> ...`` with uniform signed weights.

    The default weight is *derived*, not guessed. One presynaptic spike delivers
    ``g0 = synapses * w_syn`` of conductance, which then decays with ``tau_syn``
    while driving ``v`` with ``tau_m``::

        v(t) - v_rest = g0 * ts/(tm - ts) * (e^{-t/tm} - e^{-t/ts})

    That bracket peaks at ``t* = tm*ts/(tm-ts) * ln(tm/ts)`` = 9.242 ms for the
    verified constants, where it equals 0.4725, so the peak excursion is
    ``g0 * 0.1575``. Crossing the 7 mV rest-to-threshold gap from a *single*
    presynaptic spike therefore needs ``g0 >= 44.4 mV``, i.e. **162 synapses**.

    That number is worth stating plainly, because it is a real property of this
    model rather than a quirk of the test graph: the dataset's median connection
    carries 3.6 synapses (≈1 mV of conductance, ≈0.16 mV of excursion), so no
    single FlyWire connection can drive a neuron on its own. Downstream neurons
    respond only to *convergent* input, which is consistent with the published
    model activating only a few hundred neurons when the sugar GRNs are driven.

    The default of 200 synapses (``g0 = 55 mV``, excursion 8.7 mV) gives
    one-spike-per-hop propagation real margin, which is what makes this graph a
    useful delay/propagation fixture. Use 100 synapses to get a graph that is
    strongly driven yet provably cannot propagate a single spike.
    """
    edges = [(i, i + 1, sign * synapses) for i in range(max(n_neurons - 1, 0))]
    return make_synthetic_connectome(edges, n_neurons)
