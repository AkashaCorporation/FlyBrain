"""Explicit column contract for the FlyWire-derived dataset files.

The dataset is the one shipped with the Shiu et al. (2024) model
(``2023_03_23_*`` for version 630, ``*_783`` for version 783). Both versions have
an identical schema, verified in ``docs/UPSTREAM_AUDIT.md`` section 5.

We never guess column names. Everything the loader needs is named here, and a
mismatch is a hard error rather than a silent None.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------------------
# Neuron (completeness) table
# --------------------------------------------------------------------------------------

#: The neuron table is a CSV whose *index* carries the FlyWire root ID and which has a
#: single boolean column. The index has no header name, so pandas gives it ``None``.
NEURON_TABLE_INDEX_COL = None
NEURON_COMPLETED_COL = "Completed"

# --------------------------------------------------------------------------------------
# Connectivity table
# --------------------------------------------------------------------------------------

COL_PRE_ID = "Presynaptic_ID"
COL_POST_ID = "Postsynaptic_ID"
COL_PRE_INDEX = "Presynaptic_Index"
COL_POST_INDEX = "Postsynaptic_Index"
COL_SYNAPSE_COUNT = "Connectivity"
COL_EXCITATORY = "Excitatory"
COL_SIGNED_COUNT = "Excitatory x Connectivity"

#: The only columns the simulation consumes: the two endpoint indices and the signed
#: synapse count. Reading just these is not a micro-optimisation - it is the difference
#: between ~0.5 GB and ~2.4 GB of peak RSS on the cold path, because ``read_parquet``
#: materialises every requested column as int64 before anything is narrowed.
CONNECTIVITY_COLUMNS_MINIMAL = [
    COL_PRE_INDEX,
    COL_POST_INDEX,
    COL_SIGNED_COUNT,
]

#: Every column in the table. Used by the thorough verification path, which checks
#: that the FlyWire ID columns agree with the index columns - an integrity check worth
#: its memory cost when run deliberately, and waste on every simulate.
CONNECTIVITY_COLUMNS_FULL = [
    COL_PRE_ID,
    COL_POST_ID,
    COL_PRE_INDEX,
    COL_POST_INDEX,
    COL_SYNAPSE_COUNT,
    COL_EXCITATORY,
    COL_SIGNED_COUNT,
]

# --------------------------------------------------------------------------------------
# In-memory dtypes
# --------------------------------------------------------------------------------------
# int32 for indices: neuron count is <= 138_639, far below 2**31.
# int16 for the signed synapse count: measured range is [-2405, +1897] for both
#   dataset versions, well inside int16. The loader *verifies* this rather than trusting
#   it, and raises if a future dataset would overflow.
# float32 for synaptic values: the model is specified in mV with 3 decimal places.

DTYPE_NEURON_INDEX = np.int32
DTYPE_SIGNED_COUNT = np.int16
DTYPE_WEIGHT = np.float32

#: Every value in the ``Excitatory`` column must be one of these. The published data
#: contains exactly {-1, +1}; a 0 would mean "unknown sign per edge", which the
#: upstream model does not define, so it is rejected.
VALID_SIGN_VALUES = (-1, 1)

#: Sanity bounds used by the validator. Not magic numbers: they are the measured
#: values for the two shipped dataset versions, with headroom, and they exist to catch
#: a corrupted or substituted file.
EXPECTED_NEURONS = {"flywire_630": 127_400, "flywire_783": 138_639}
EXPECTED_EDGES = {"flywire_630": 14_687_178, "flywire_783": 15_091_983}
