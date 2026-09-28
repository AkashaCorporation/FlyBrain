"""The interactive console.

Tested through ``Console.dispatch`` with a scripted command list, which is exactly
how a user drives it (and is why the console is a class with a dispatch table rather
than a ball of ``input()`` calls). The commands are the ones the objective's example
session uses.
"""

from __future__ import annotations

import pytest

from flybrain.model.populations import PopulationRegistry

from conftest import needs_dataset

pytestmark = needs_dataset


@pytest.fixture(scope="module")
def console():
    from flybrain.repl import Console

    lines: list[str] = []
    c = Console(mode="subset", duration_ms=50.0, seed=0, echo=lines.append,
                subset_hops=1, subset_max_neurons=600)
    c.echoed = lines  # type: ignore[attr-defined]
    return c


def test_console_opens_on_a_labelled_subset(console):
    banner = console.banner()
    assert "flywire_630" in banner
    assert "subset" in banner
    # it must never let a subset be mistaken for a whole brain
    assert "neurons:" in banner
    assert console.connectome.is_subset
    assert console.plan.effective_mode == "subset"


def test_console_states_it_is_not_a_language_interface(console):
    # normalise the banner's line wrapping before matching phrases
    banner = " ".join(console.banner().lower().split())
    assert "not a language interface" in banner
    assert "no language" in banner
    assert "no learning" in banner


def test_unknown_command_is_reported_not_executed(console):
    before = console.steps_run
    assert console.dispatch("definitely_not_a_command") is True
    assert console.steps_run == before
    assert any("unknown command" in line for line in console.echoed)


def test_help_lists_the_five_operations(console):
    console.echoed.clear()
    console.dispatch("help")
    text = "\n".join(console.echoed)
    for token in ("stimulate", "run", "silence", "reset", "neuron", "top-active"):
        assert token in text


def test_stimulate_runs_the_simulation_and_reports(console):
    console.echoed.clear()
    console.dispatch("reset")
    console.dispatch("stimulate sugar_grn 150 50")
    text = "\n".join(console.echoed)
    assert "stimulating sugar_grn" in text
    assert "Spikes:" in text
    assert console.steps_run == 500, "50 ms at dt=0.1 ms is 500 steps"
    assert console.net.n == console.connectome.n_neurons


def test_every_help_command_actually_dispatches(console):
    """Each command named in the help text must resolve to a handler.

    Hyphenated command names cannot map directly onto method names, and the failure
    mode is silent: the console reports "unknown command" for a command it advertises.
    """
    import re

    console.echoed.clear()
    console.dispatch("help")
    text = "\n".join(console.echoed)
    names = re.findall(r"^\s{2}([a-z][a-z-]+)", text, re.MULTILINE)
    assert names, "could not parse any command names from the help text"
    for name in names:
        assert getattr(console, f"cmd_{name.replace('-', '_')}", None) is not None, (
            f"help advertises {name!r} but there is no handler for it"
        )


def test_top_active_is_empty_before_any_run(console):
    console.dispatch("clear")
    console.echoed.clear()
    console.dispatch("top-active 5")
    assert any("nothing has run yet" in line for line in console.echoed)


def test_silence_accepts_a_population_name(console):
    console.dispatch("reset")
    console.echoed.clear()
    console.dispatch("silence mn9")
    assert len(console.net.silenced_ids) == 2
    assert any("outgoing influence only" in line for line in console.echoed)


def test_reset_keeps_interventions(console):
    console.dispatch("clear")
    console.dispatch("stimulate sugar_grn 150 20")
    console.dispatch("silence mn9")
    silenced_before = console.net.silenced_ids
    console.dispatch("reset")
    assert console.steps_run == 0
    assert console.net.silenced_ids == silenced_before
    assert len(console.net._samplers) == 1


def test_clear_drops_interventions(console):
    console.dispatch("reset")
    console.dispatch("clear")
    assert console.net.silenced_ids == ()
    assert len(console.net._samplers) == 0
    assert console.steps_run == 0


def test_neuron_command_reports_provenance_limits(console):
    console.echoed.clear()
    console.dispatch("neuron 720575940660219265")
    text = "\n".join(console.echoed)
    assert "720575940660219265" in text
    assert "unavailable from dataset" in text, "must not invent a cell type"
    assert "neurotransmitter:" in text


def test_seed_command_resets_deterministically(console):
    console.dispatch("clear")
    console.dispatch("stimulate sugar_grn 150 30")
    first = console.net.spike_count.copy()
    console.dispatch("seed 4")
    assert console.steps_run == 0
    console.dispatch("stimulate sugar_grn 150 30")
    second = console.net.spike_count
    # seed changed, so equality is not required; the point is that it re-ran cleanly
    assert second.shape == first.shape


def test_save_writes_a_state_file(console, tmp_path):
    console.dispatch("clear")
    console.dispatch("stimulate sugar_grn 150 20")
    console.echoed.clear()
    console.dispatch(f"save {tmp_path}")
    import json

    p = tmp_path / "console_state.json"
    assert p.is_file()
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["dataset"]["dataset_id"] == "flywire_630"
    assert payload["steps_run"] == 200
    assert payload["stimuli"][0]["label"] == "sugar_grn"
    assert payload["top_neurons"], "the saved state must include the active neurons"


def test_growing_the_working_set_for_an_outside_population(console):
    """Stimulating a population outside the subset grows the set and says so.

    The alternative - silently ignoring the target, or refusing - would both be worse:
    the first would measure nothing, the second would make the console unusable.
    """
    console.dispatch("clear")
    before = console.connectome.n_neurons
    console.echoed.clear()
    console.dispatch("stimulate jon_all 100 20")
    text = "\n".join(console.echoed)
    assert console.connectome.n_neurons >= before
    # every Jonston's-organ neuron that is in the dataset must now be stimulable
    reg = PopulationRegistry.builtin()
    present, _ = reg.resolve("jon_all", console.full_connectome.flywire_ids)
    missing = [int(x) for x in present if int(x) not in console.connectome.id_to_index]
    assert not missing, f"{len(missing)} stimulus neurons are outside the working set"
    assert "stimulating jon_all" in text


def test_scripted_session_via_run_repl(monkeypatch):
    """The piped-stdin path, including the UTF-8 BOM that PowerShell prepends."""
    import io

    from flybrain.repl import run_repl

    script = "\ufeffstatus\nrun 20\nquit\n"
    out = io.StringIO()
    rc = run_repl(
        mode="subset", duration_ms=20.0, seed=0,
        stdin=io.StringIO(script), stdout=out,
    )
    assert rc == 0
    text = out.getvalue()
    assert "unknown command" not in text, "the BOM must not break the first command"
    assert "Spikes:" in text
