"""Documentation generation.

The parameter table in ``docs/MODEL_ASSUMPTIONS.md`` is *generated* from
:data:`flybrain.model.lif.PARAMETER_PROVENANCE`, and ``tests/test_docs.py`` asserts
that the file on disk matches a fresh render. Documentation that drifts from the
code is a scientific-accuracy bug, not a formatting one: a reader who trusts the
table would be misled about the model's constants.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .lif import PARAMETER_PROVENANCE

BEGIN = "<!-- BEGIN GENERATED PARAMETER TABLE -->"
END = "<!-- END GENERATED PARAMETER TABLE -->"

COLUMNS = ("MelanoGraph field", "value", "unit", "source", "reason", "confidence")


def render_parameter_table() -> str:
    """Render the parameters as a Markdown table, in a stable order.

    Ordering follows the dataclass definition (membrane, synapse, stimulation)
    rather than alphabetical, because that is the order a reader wants.
    """
    order = [
        "v_rest_mv", "v_reset_mv", "v_threshold_mv",
        "tau_membrane_ms", "tau_synapse_ms", "refractory_ms", "synaptic_delay_ms",
        "weight_per_synapse_mv", "stimulus_rate_hz", "stimulus_weight_scale",
        "dt_ms",
    ]
    missing = set(PARAMETER_PROVENANCE) - set(order)
    extra = set(order) - set(PARAMETER_PROVENANCE)
    if missing or extra:
        raise RuntimeError(
            f"PARAMETER_PROVENANCE and the render order disagree; "
            f"missing from order: {sorted(missing)}, not in provenance: {sorted(extra)}"
        )

    def esc(s: str) -> str:
        return str(s).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "|".join("---" for _ in COLUMNS) + "|",
    ]
    for field in order:
        info = PARAMETER_PROVENANCE[field]
        lines.append(
            "| `{field}` | {value} | {unit} | {source} | {reason} | {confidence} |".format(
                field=field,
                value=info["value"],
                unit=esc(info["unit"]),
                source=esc(info["source"]),
                reason=esc(info["reason"]),
                confidence=info["confidence"],
            )
        )
    return "\n".join(lines)


def render_mapped_section() -> str:
    """Upstream key -> MelanoGraph field, so the mapping is never guessed."""
    lines = [
        "| upstream key (``model.py``) | MelanoGraph field | value |",
        "|---|---|---|",
    ]
    for field, info in PARAMETER_PROVENANCE.items():
        lines.append(f"| `{info['upstream_key']}` | `{field}` | {info['value']} |")
    return "\n".join(lines)


def replace_generated_section(text: str, rendered: str) -> str:
    """Replace everything between the markers (exclusive) with ``rendered``."""
    if BEGIN not in text or END not in text:
        raise RuntimeError(
            f"the target document is missing the {BEGIN} / {END} markers"
        )
    head, rest = text.split(BEGIN, 1)
    _old, tail = rest.split(END, 1)
    return f"{head}{BEGIN}\n{rendered}\n{END}{tail}"


def update_document(path: str | Path, *, check: bool = False) -> bool:
    """Rewrite the generated section. Returns True when the file was already current.

    With ``check=True`` nothing is written; the return value reports whether the
    file matches, which is what the test uses.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    updated = replace_generated_section(text, render_parameter_table())
    if check:
        return updated == text
    if updated != text:
        path.write_text(updated, encoding="utf-8")
    return updated == text


def render_all_tables() -> List[str]:
    return [render_parameter_table(), render_mapped_section()]
