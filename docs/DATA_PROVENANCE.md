# DATA_PROVENANCE

**Status:** complete for v0

Nothing in MelanoGraph is downloaded at run time. The two dataset files are staged from
a local clone of the upstream repository, hashed, validated, and only then simulated
on. This document records exactly what those files are, where they came from, and what
their numbers are.

---

## 1. Pipeline

```
external FlyWire data
        ↓   (already preprocessed and published by the upstream authors —
            MelanoGraph performs no EM processing and no CAVE query)
data/raw/<dataset_id>/               staged copy, byte-identical to upstream
        ↓   SHA-256 per file, recorded in a DatasetCard
data/metadata/<dataset_id>.json      committed provenance record
        ↓   parquet -> compact arrays, sorted by presynaptic index
data/processed/<dataset_id>/connectome.npz   derived cache, hash-linked to the raw files
        ↓
MelanoGraph simulation
```

`data/raw/` and `data/processed/` are gitignored (they are ~90 MB and ~90 MB
respectively). `data/metadata/*.json` **is** committed: it is small, and it is the
provenance record. That asymmetry is deliberate — a fresh checkout can verify the
dataset's identity before spending the disk to stage it.

---

## 2. Datasets

### `flywire_630` — the primary v0 dataset

| field | value |
|---|---|
| source | https://github.com/philshiu/Drosophila_brain_model |
| release | FlyWire 2023_03_23 materialization, completeness threshold 630 |
| neuron table | `2023_03_23_completeness_630_final.csv` — 3 057 611 bytes |
| neuron table SHA-256 | `e6b71e17671a9bdb05f55e4bc6774640a1418cb7a05125e0fc994ad40f9bfdfb` |
| connectivity table | `2023_03_23_connectivity_630_final.parquet` — 86 630 944 bytes |
| connectivity SHA-256 | `94db8c650533bc36ffa3223f2e62325d5648b8d6bd31c3a4e1c804628c7557b3` |
| combined card digest | `67de1c152ff8d47b8460f1c980052abfe3ed40c6e98870ff8af073a9e12dcf4a` |
| neurons | 127 400 |
| directed edges | 14 687 178 |
| chemical synapses | 52 793 639 |

All byte counts are read from `data/metadata/flywire_630.json`, which is the record the
staging step wrote after hashing the files — not transcribed by hand.

Staged from `third_party/drosophila_whole_brain_snn_simulation`. The connectivity
table is **byte-identical** to the copy vendored by
`chaobrain/drosophila_whole_brain_snn_simulation`
(`UPSTREAM_AUDIT.md` §4), so there is one source of truth for the connectivity. The
neuron table has a different SHA-256 in the two repositories but identical *content*
(127 400 rows, same index range, same values); the bytes differ because of line-ending
encoding, which the audit verified rather than assumed.

This is the version the Nature paper used, and the only version for which the authors'
published spike output exists, so it is the only version on which a reference
comparison is possible.

### `flywire_783` — registered, validated, not primary

| field | value |
|---|---|
| neuron table | `Completeness_783.csv` — 3 465 987 bytes |
| neuron table SHA-256 | `52b0ac6094cd32c546f8d4c341e094376f48f4e791f8db9b166de5dff8199ea4` |
| connectivity table | `Connectivity_783.parquet` — 100 804 642 bytes |
| connectivity SHA-256 | `efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347` |
| combined card digest | `4a621918d7ee52a7e915e98f83814356115c0fcf04392ac32d285893521e1097` |
| neurons | 138 639 |
| directed edges | 15 091 983 |
| chemical synapses | 54 492 922 |
| excitatory / inhibitory presynaptic neurons | 96 672 / 41 333 (634 with no outgoing edge) |

**No importer had to be written for this.** The upstream repository ships it
alongside v630, so MelanoGraph's data strategy reduces to "stage the two files, verify
their hashes, derive a cache" — Option A in the objective. v783 is registered and
validated but is not promoted to primary, because no published reference output
exists for it.

---

## 3. Measured content of `flywire_630`

Full machine-readable report: [`outputs/dataset_report.json`](../outputs/dataset_report.json).

| quantity | measured |
|---|---|
| neurons | 127 400 |
| directed edges | 14 687 178 |
| chemical synapses (`Connectivity.sum()`) | 52 793 639 |
| mean synapses per connected pair | 3.5945 |
| excitatory edges (`Excitatory == +1`) | 8 800 532 |
| inhibitory edges (`Excitatory == −1`) | 5 886 646 |
| excitatory presynaptic neurons | 86 543 |
| inhibitory presynaptic neurons | 40 472 |
| neurons with **no** outgoing edge (sign unknown) | 385 |
| neurons with no incoming edge | 607 |
| neurons with neither | **0** |
| neurons with both excitatory and inhibitory outgoing edges | **0** |
| duplicate `(pre, post)` pairs | **0** |
| self-connections | **0** |
| invalid edges (out-of-range endpoint, NaN, or zero magnitude) | **0** |
| NaN or non-positive `Connectivity` values | **0** |
| `Excitatory` distinct values | exactly `{−1, +1}` |
| max out-degree / in-degree | 9 615 / 10 196 |
| mean out-degree | 115.28 |
| signed synapse count range | −2 358 … +1 801 |
| CSR bytes | ~119.5 MB |

Two facts from this table shape the whole design:

1. **Sign is per presynaptic neuron.** No neuron has mixed-sign outgoing edges, so a
   per-neuron excitatory/inhibitory label is well defined — and the 385 neurons with
   no outgoing edge have no sign evidence at all and are reported as `unknown_sign`
   rather than being silently assigned a default.
2. **The "~50 million synaptic connections" headline is the synapse count, not the
   edge count.** The graph has 14.69 M directed edges, so the resident CSR needs
   well under 1 GB. A dense 127 400² matrix would need **64.9 GB** at float32, which
   is why `estimate_requirements` reports that number explicitly: the sparse
   representation is a requirement, not an optimisation.

---

## 4. Validation performed before any simulation

`flybrain/data/validate.py` runs before the network is built, and
`DatasetValidationError` aborts the run on any hard failure. Checks:

**Neuron table** — at least one neuron; FlyWire IDs unique; IDs positive; index
strictly increasing (the row order *is* the index space, so this is load-bearing);
count matching the declared release.

**Edge table** — column lengths consistent; at least one edge; source indices in
`[0, n)`; target indices in `[0, n)`; endpoints resolve to exactly one neuron; no NaN
weights; no zero-magnitude weights; signed count within the declared dtype; no
duplicate `(pre, post)` pairs; self-connections counted; orphan counts by direction.

**Sign** — `Excitatory` values are exactly `{−1, +1}`; sign is consistent with the
signed count; the per-neuron sign is well defined (no mixed neurons); unknown-sign
neurons counted.

**Honest gaps** — the `neuron_types_available` check reports `warn` unconditionally,
because the dataset ships **no cell-type column**: the neuron table has a single
`Completed` column. MelanoGraph does not substitute a guess. Neuron types exist only
through the curated populations in §5, and every output that could be read as a
type annotation says so.

The report distinguishes `fail` (the dataset violates an invariant that makes
simulation meaningless — abort) from `warn` (a true property the operator must know
about — proceed, but record it). On `flywire_630`: **3 warnings, 0 failures**. The
three warnings are `orphan_counts`, `sign_coverage` and `neuron_types_available` —
each one a true property of the data, not a defect in the loading.

### The ID ↔ index check (verification path only)

`flybrain datasets --verify` and `--report` additionally run
`check_id_index_consistency`, which proves that index `k` really does name the neuron
whose FlyWire root ID is `Presynaptic_ID[k]`, for every edge. Measured:

```
ID<->index consistency: [pass] 29374356 endpoint(s) checked; 0 mismatch(es)
```

This is the check that makes "neuron IDs are preserved" a verified statement rather
than an assertion, and it is the reason the full seven-column read still exists in the
codebase.

It runs only on the verification path, because reading all seven columns is expensive.
`scripts/measure_load_memory.py` measures the same decode step both ways in one
process, back to back, and is reproducible:

```
all 7 columns      : RSS delta    2318.0 MB (peak 2318.0 MB)
3 required columns : RSS delta     501.2 MB (peak  280.9 MB)

reduction: 78.4% less RSS on the decode step
reduction: 87.9% less peak resident memory
```

Raw output: `outputs/load_memory_flywire_630.json`. So the expensive read is now paid
deliberately once on the verification path, instead of on every run.

(That script also prints a `build_connectome` line whose RSS delta is negative: it is
measured immediately after the previous arrays were released, so the delta reflects
the release as much as the allocation. The two decode figures above are the meaningful
pair — same operation, same process, different column count.)

---

## 5. Curated populations

`flybrain/model/populations.json` holds 114 named neuron groups (sugar/bitter/water/
IR94e gustatory receptor neurons, Johnston's organ neurons, the MN9 motor neurons,
and 106 SEZ cell types). It is **generated by parsing the upstream notebooks with
`ast`** rather than by retyping 20-digit IDs, and it records the SHA-256 of every
file it was parsed from. The generator lives at the history of this repository's
`_scratch/extract_populations.py` step and the resulting file is committed, so the
lists can be re-derived and checked.

This is the *only* neuron-type information available to MelanoGraph. It is inherited
from the upstream authors' curation and is not independently verified.

---

## 6. Reproducing the dataset layer

```bash
# clone the upstream reference (read-only; MelanoGraph never imports it)
git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model.git \
    third_party/Drosophila_brain_model

# copy both versions into data/raw, hash them, and validate their content
flybrain datasets --stage all

# re-validate content and rewrite outputs/dataset_report.json
flybrain datasets --report flywire_630

# list the registry and verify what is staged
flybrain datasets --verify
```

`stage_dataset` refuses to guess: it raises if the expected filenames are not where
it was told to look, and `load_connectome` raises if a staged file's size disagrees
with its dataset card. A truncated or substituted file is a hard error, never a
warning — see `tests/test_data.py::test_loader_refuses_a_truncated_file`.

---

## 7. Known limitations inherited with the data

* The connectivity is for the **proofread** subset of the connectome at completeness
  threshold 630. Non-proofread tissue is excluded, so the model's brain is not the
  whole animal's nervous system — no ventral nerve cord, no peripheral ganglia.
* `Excitatory` is a **predicted** neurotransmitter identity, not a measurement. Its
  accuracy is not characterised in the upstream repository and MelanoGraph inherits it
  unexamined.
* Synapse counts come from automated synapse detection with upstream filtering; false
  positives and false negatives are both possible and are not quantified here.
* The uniform 1.8 ms delay applied to every connection is physiologically
  implausible; it is upstream's modelling choice, retained for reproducibility.
* FlyWire IDs are materialization-specific. A v630 ID is not guaranteed to resolve in
  a later FlyWire release, and MelanoGraph does not implement ID mapping across versions.
