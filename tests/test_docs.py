"""Documentation consistency.

The parameter table in ``docs/MODEL_ASSUMPTIONS.md`` is generated from the code, and
these tests fail if the two disagree. This is not formatting pedantry: a reader who
trusts a stale table would be misled about the model's constants, which is a
scientific-accuracy bug.
"""

from __future__ import annotations

import re

import pytest

from flybrain import config as fb_config
from flybrain.model.docgen import (
    BEGIN,
    END,
    render_parameter_table,
    replace_generated_section,
    update_document,
)
from flybrain.model.lif import LITERATURE, PARAMETER_PROVENANCE, LIFParams

DOCS = fb_config.DOCS_DIR
REQUIRED_DOCS = [
    "UPSTREAM_AUDIT.md",
    "MODEL_ASSUMPTIONS.md",
    "DATA_PROVENANCE.md",
    "SCIENTIFIC_BOUNDARIES.md",
    "ARCHITECTURE.md",
    "STATUS.md",
]


# --------------------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", REQUIRED_DOCS)
def test_required_document_exists_and_is_not_a_stub(name):
    p = DOCS / name
    assert p.is_file(), f"missing required document docs/{name}"
    text = p.read_text(encoding="utf-8")
    assert len(text) > 1500, f"docs/{name} looks like a stub ({len(text)} chars)"
    assert text.lstrip().startswith("#"), f"docs/{name} must start with a title"


def test_readme_exists_and_links_the_docs():
    readme = (fb_config.PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    for name in REQUIRED_DOCS:
        assert f"docs/{name}" in readme, f"README does not link docs/{name}"


# --------------------------------------------------------------------------------------
# the generated parameter table
# --------------------------------------------------------------------------------------


def test_model_assumptions_table_matches_the_code():
    ok = update_document(DOCS / "MODEL_ASSUMPTIONS.md", check=True)
    assert ok, (
        "docs/MODEL_ASSUMPTIONS.md is out of date with flybrain/model/lif.py. "
        "Run `python scripts/render_doc_tables.py`."
    )


def test_generated_section_markers_are_present_and_paired():
    text = (DOCS / "MODEL_ASSUMPTIONS.md").read_text(encoding="utf-8")
    assert text.count(BEGIN) == 1
    assert text.count(END) == 1
    assert text.index(BEGIN) < text.index(END)


def test_render_is_deterministic():
    assert render_parameter_table() == render_parameter_table()


def test_replace_generated_section_is_idempotent():
    text = (DOCS / "MODEL_ASSUMPTIONS.md").read_text(encoding="utf-8")
    once = replace_generated_section(text, render_parameter_table())
    twice = replace_generated_section(once, render_parameter_table())
    assert once == twice


# --------------------------------------------------------------------------------------
# values: docs <-> provenance <-> dataclass
# --------------------------------------------------------------------------------------


def test_every_provenance_entry_has_the_required_fields():
    for field, info in PARAMETER_PROVENANCE.items():
        for key in ("value", "unit", "source", "reason", "confidence", "upstream_key"):
            assert key in info, f"{field} is missing {key!r}"
        assert info["confidence"] in ("cited", "free", "derived"), (
            f"{field} has an unknown confidence {info['confidence']!r}"
        )
        assert info["source"], f"{field} has no source"
        assert info["reason"], f"{field} has no reason"


def test_provenance_values_match_the_instantiated_dataclass():
    """The table is only trustworthy if it equals what the code actually uses."""
    params = LIFParams()
    for field, info in PARAMETER_PROVENANCE.items():
        if field == "dt_ms":
            assert info["value"] == fb_config.DEFAULT_DT_MS
            continue
        assert hasattr(params, field), f"{field} is not a LIFParams field"
        actual = getattr(params, field)
        assert actual == info["value"], (
            f"{field}: LIFParams says {actual}, provenance says {info['value']}"
        )


def test_every_lif_parameter_is_documented():
    """An undocumented parameter is a parameter nobody can audit."""
    documented = set(PARAMETER_PROVENANCE)
    for field in LIFParams.__dataclass_fields__:
        assert field in documented, f"LIFParams.{field} has no provenance entry"


def test_mapped_table_values_match_provenance():
    """The upstream-key table in the doc must not drift either."""
    text = (DOCS / "MODEL_ASSUMPTIONS.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\|\s*`([A-Za-z0-9_]+)`\s*\|\s*`([a-z0-9_]+)`\s*\|\s*([-\d.]+)\s*\|",
                      text, re.MULTILINE)
    assert rows, "could not parse the upstream-key mapping table"
    for upstream_key, field, value in rows:
        assert field in PARAMETER_PROVENANCE, f"unknown field {field} in the mapping table"
        entry = PARAMETER_PROVENANCE[field]
        assert upstream_key == entry["upstream_key"], (
            f"{field}: doc says upstream key {upstream_key!r}, code says {entry['upstream_key']!r}"
        )
        assert float(value) == float(entry["value"]), (
            f"{field}: mapping table says {value}, provenance says {entry['value']}"
        )


def test_literature_references_have_dois_or_are_the_paper():
    for key, ref in LITERATURE.items():
        assert "doi" in ref.lower() or "Nature" in ref, (
            f"literature entry {key!r} has no DOI: {ref!r}"
        )


# --------------------------------------------------------------------------------------
# the boundaries the docs promise are actually enforced in code
# --------------------------------------------------------------------------------------


def _call_arguments(text: str, func: str) -> list[str]:
    """Return the argument source text of every ``func(...)`` call, paren-matched.

    A line-based scan is wrong here because the tag is often on a continuation line
    of a multi-line call, so a naive per-line check reports false positives - which
    is exactly what it did before this helper existed.
    """
    out = []
    needle = func + "("
    start = 0
    while True:
        i = text.find(needle, start)
        if i < 0:
            break
        j = i + len(needle)
        depth = 1
        while j < len(text) and depth:
            ch = text[j]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            j += 1
        out.append(text[i:j])
        start = j
    return out


def test_plots_carry_the_simulated_activity_tag():
    """Every figure title must say the activity is simulated.

    ``SIMULATED_TAG`` is a module constant precisely so it cannot be dropped from
    one figure while being kept in another.
    """
    from flybrain.analysis.plots import SIMULATED_TAG

    assert "not a biological measurement" in SIMULATED_TAG
    src = (fb_config.PROJECT_ROOT / "flybrain" / "analysis" / "plots.py").read_text(encoding="utf-8")

    titles = _call_arguments(src, "set_title")
    assert len(titles) >= 4, f"expected at least 4 set_title calls, found {len(titles)}"
    for call in titles:
        assert "SIMULATED_TAG" in call, f"plot title without the simulated-activity tag: {call!r}"

    sup = _call_arguments(src, "suptitle")
    for call in sup:
        assert "SIMULATED_TAG" in call, f"suptitle without the simulated-activity tag: {call!r}"


def test_every_plot_function_labels_its_axes():
    """An unlabelled axis is a readability defect the objective explicitly calls out."""
    src = (fb_config.PROJECT_ROOT / "flybrain" / "analysis" / "plots.py").read_text(encoding="utf-8")
    for fn in ("plot_population_rates", "plot_raster", "plot_top_populations",
               "plot_input_vs_downstream", "plot_voltage_trace"):
        assert fn in src
    assert src.count("set_xlabel") >= 4
    assert src.count("set_ylabel") >= 4


def _paragraphs_with_context(text: str):
    """Yield ``(heading, paragraph)`` pairs, tracking the enclosing Markdown heading."""
    heading = ""
    block: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            if block:
                yield heading, "\n".join(block)
                block = []
            heading = line.lstrip("#").strip()
            continue
        if not line.strip():
            if block:
                yield heading, "\n".join(block)
                block = []
            continue
        block.append(line)
    if block:
        yield heading, "\n".join(block)


NEGATIONS = (" not ", "n't ", "never", "cannot", "does not", "is not", "no ", "without",
             "rather than", "out of scope")


def test_docs_do_not_assert_biological_fidelity():
    """Guard against a doc edit that quietly overstates the result.

    A claim is allowed only when its own paragraph or its enclosing heading negates
    it - which is how a "what may NOT be claimed" section legitimately states the
    thing it forbids.
    """
    banned = [
        "is conscious",
        "faithful digital copy",
        "reproduces the fly",
        "replicates the animal",
        "matches biological activity",
        "equals biological activity",
    ]
    targets = [(fb_config.PROJECT_ROOT / "README.md").resolve()]
    targets += [(DOCS / n).resolve() for n in REQUIRED_DOCS]

    for p in targets:
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        for heading, para in _paragraphs_with_context(text):
            low = para.lower()
            context = (heading + " " + para).lower()
            for phrase in banned:
                if phrase in low and not any(neg in context for neg in NEGATIONS):
                    pytest.fail(
                        f"{p.name} appears to assert {phrase!r} without negation:\n"
                        f"  heading: {heading!r}\n  {para.strip()[:200]}"
                    )


def test_boundary_docs_state_the_key_disclaimers():
    """Positive checks: the disclaimers must actually be present, not merely not-banned."""
    text = (DOCS / "SCIENTIFIC_BOUNDARIES.md").read_text(encoding="utf-8").lower()
    for required in (
        "connectome",
        "lif model",
        "simulated activity",
        "biological activity",
        "may **not** claim",
        "free parameter",
        "w_syn",
    ):
        assert required.lower() in text, f"SCIENTIFIC_BOUNDARIES.md is missing {required!r}"

    ma = (DOCS / "MODEL_ASSUMPTIONS.md").read_text(encoding="utf-8").lower()
    for required in ("free parameter", "derived from a measurement", "confidence", "w_syn"):
        assert required in ma, f"MODEL_ASSUMPTIONS.md is missing {required!r}"
