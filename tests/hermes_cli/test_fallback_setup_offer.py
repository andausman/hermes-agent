"""Tests for the post-model-setup fallback offer.

Covers ``_maybe_offer_fallback_setup`` in hermes_cli/main.py and its wiring
into the three interactive entry points of ``select_provider_and_model``:

- ``hermes model`` (cmd_model)
- setup wizard (setup_model_provider in setup.py)
- first-run setup (_offer_first_run_setup in cli_agent_setup_mixin.py)

Contract under test:
- Offer fires only when a primary exists, no fallback chain is configured,
  and the model config actually changed (picker not cancelled).
- Declining the offer leaves config untouched.
- Accepting routes into ``hermes fallback add`` logic (chain appended).
- Offer failures never propagate (setup must not break).
- Non-interactive contexts never see a prompt.
"""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# Shared fixture — isolate HERMES_HOME so save_config writes to tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Ensure prompt_yes_no's non-interactive guard doesn't silently skip.
    monkeypatch.delenv("HERMES_NONINTERACTIVE", raising=False)
    return tmp_path


@pytest.fixture()
def fake_tty(monkeypatch):
    """Simulate an interactive terminal for _maybe_offer_fallback_setup."""
    import sys

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    return True


def _write_config(home: Path, data: dict) -> None:
    config_path = home / ".hermes" / "config.yaml"
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _read_config(home: Path) -> dict:
    config_path = home / ".hermes" / "config.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


PRIMARY = {"provider": "anthropic", "default": "claude-sonnet-4-6"}


# ---------------------------------------------------------------------------
# _maybe_offer_fallback_setup — gating conditions
# ---------------------------------------------------------------------------

class TestOfferGating:
    def test_offer_even_when_model_config_unchanged(self, isolated_home, fake_tty):
        """'Leave unchanged' still gets the offer — declining the primary says
        nothing about wanting a backup. The offer only goes quiet once a
        chain exists or no primary is configured."""
        _write_config(isolated_home, {"model": dict(PRIMARY)})

        with patch("hermes_cli.setup.prompt_yes_no", return_value=False) as p:
            from hermes_cli.main import _maybe_offer_fallback_setup
            _maybe_offer_fallback_setup()

        assert p.call_count == 1

    def test_no_offer_when_no_primary(self, isolated_home):
        """Fresh install with picker cancelled → nothing to back up."""
        _write_config(isolated_home, {})

        def fail_prompt(*a, **kw):
            raise AssertionError("prompt_yes_no must not be called without a primary")

        with patch("hermes_cli.setup.prompt_yes_no", fail_prompt):
            from hermes_cli.main import _maybe_offer_fallback_setup
            _maybe_offer_fallback_setup()

    def test_no_offer_when_chain_already_configured(self, isolated_home):
        """Don't nag on every model switch once a chain exists."""
        _write_config(isolated_home, {
            "model": dict(PRIMARY),
            "fallback_providers": [
                {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
            ],
        })

        def fail_prompt(*a, **kw):
            raise AssertionError("prompt_yes_no must not be called when chain exists")

        with patch("hermes_cli.setup.prompt_yes_no", fail_prompt):
            from hermes_cli.main import _maybe_offer_fallback_setup
            _maybe_offer_fallback_setup()

    def test_offer_prompted_when_primary_changed(self, isolated_home, fake_tty, capsys):
        """Happy path: primary changed + no chain → user is asked."""
        _write_config(isolated_home, {"model": dict(PRIMARY)})

        with patch("hermes_cli.setup.prompt_yes_no", return_value=False) as p:
            from hermes_cli.main import _maybe_offer_fallback_setup
            _maybe_offer_fallback_setup()

        assert p.call_count == 1
        # The question text mentions fallback so the user knows what they're
        # agreeing to.
        question = p.call_args[0][0] if p.call_args[0] else p.call_args[1].get(
            "question", ""
        )
        assert "fallback" in question.lower()


# ---------------------------------------------------------------------------
# _maybe_offer_fallback_setup — accept / decline behavior
# ---------------------------------------------------------------------------

class TestOfferOutcome:
    def test_decline_leaves_config_untouched(self, isolated_home, fake_tty):
        _write_config(isolated_home, {"model": dict(PRIMARY)})

        with patch("hermes_cli.setup.prompt_yes_no", return_value=False):
            from hermes_cli.main import _maybe_offer_fallback_setup
            _maybe_offer_fallback_setup()

        cfg = _read_config(isolated_home)
        assert cfg["model"] == PRIMARY
        assert "fallback_providers" not in cfg

    def test_accept_appends_to_chain_via_fallback_add(self, isolated_home, fake_tty, capsys):
        """Accepting routes into cmd_fallback_add with the canonical picker."""
        _write_config(isolated_home, {"model": dict(PRIMARY)})

        def fake_picker(args=None):
            # Simulate the user picking a fallback in the canonical picker:
            # writes the selection to config["model"], like the real flow.
            from hermes_cli.config import load_config, save_config
            cfg = load_config()
            cfg["model"] = {
                "provider": "openrouter",
                "default": "anthropic/claude-sonnet-4.6",
            }
            save_config(cfg)

        with patch("hermes_cli.setup.prompt_yes_no", return_value=True), \
                patch("hermes_cli.main._require_tty"), \
                patch("hermes_cli.main.select_provider_and_model", side_effect=fake_picker):
            from hermes_cli.main import _maybe_offer_fallback_setup
            _maybe_offer_fallback_setup()

        cfg = _read_config(isolated_home)
        # Primary restored, fallback appended — the cmd_fallback_add contract.
        assert cfg["model"] == PRIMARY
        assert cfg["fallback_providers"] == [
            {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
        ]
        assert "Added fallback" in capsys.readouterr().out

    def test_offer_failure_never_propagates(self, isolated_home, fake_tty):
        """A broken fallback-add path must not break the setup flow."""
        _write_config(isolated_home, {"model": dict(PRIMARY)})

        def broken_add(args=None):
            raise RuntimeError("simulated fallback-add crash")

        with patch("hermes_cli.setup.prompt_yes_no", return_value=True), \
                patch("hermes_cli.fallback_cmd.cmd_fallback_add", broken_add):
            from hermes_cli.main import _maybe_offer_fallback_setup
            # Must not raise.
            _maybe_offer_fallback_setup()


# ---------------------------------------------------------------------------
# Entry-point wiring — hermes model
# ---------------------------------------------------------------------------

class TestCmdModelWiring:
    def test_offer_fires_after_model_change(self, isolated_home, fake_tty, monkeypatch, capsys):
        _write_config(isolated_home, {"model": dict(PRIMARY)})
        monkeypatch.setattr("hermes_cli.main._require_tty", lambda *a: None)

        def fake_picker(args=None):
            from hermes_cli.config import load_config, save_config
            cfg = load_config()
            cfg["model"] = {"provider": "openrouter", "default": "x/y"}
            save_config(cfg)

        prompted = []
        monkeypatch.setattr(
            "hermes_cli.setup.prompt_yes_no",
            lambda *a, **kw: prompted.append(1) or False,
        )
        monkeypatch.setattr(
            "hermes_cli.main.select_provider_and_model", fake_picker
        )

        from hermes_cli.main import cmd_model
        cmd_model(types.SimpleNamespace(refresh=False))

        assert prompted, "fallback offer must fire after a real model change"

    def test_offer_after_cancel_when_no_chain(self, isolated_home, fake_tty, monkeypatch):
        """'Leave unchanged' + no chain yet → still offered (user may want a
        backup even though they kept their primary)."""
        _write_config(isolated_home, {"model": dict(PRIMARY)})
        monkeypatch.setattr("hermes_cli.main._require_tty", lambda *a: None)
        monkeypatch.setattr("hermes_cli.main.select_provider_and_model", lambda args=None: None)

        prompted = []
        monkeypatch.setattr(
            "hermes_cli.setup.prompt_yes_no",
            lambda *a, **kw: prompted.append(1) or False,
        )

        from hermes_cli.main import cmd_model
        cmd_model(types.SimpleNamespace(refresh=False))

        assert prompted, "'Leave unchanged' with no chain must still offer"

    def test_no_repeat_nag_once_chain_exists(self, isolated_home, fake_tty, monkeypatch):
        """Chain configured → later `hermes model` runs stay silent."""
        _write_config(isolated_home, {
            "model": dict(PRIMARY),
            "fallback_providers": [
                {"provider": "nous", "model": "Hermes-4-405B"},
            ],
        })
        monkeypatch.setattr("hermes_cli.main._require_tty", lambda *a: None)
        monkeypatch.setattr("hermes_cli.main.select_provider_and_model", lambda args=None: None)

        prompted = []
        monkeypatch.setattr(
            "hermes_cli.setup.prompt_yes_no",
            lambda *a, **kw: prompted.append(1) or False,
        )

        from hermes_cli.main import cmd_model
        cmd_model(types.SimpleNamespace(refresh=False))

        assert not prompted, "must not nag once a chain exists"


# ---------------------------------------------------------------------------
# Entry-point wiring — setup wizard
# ---------------------------------------------------------------------------

class TestSetupWizardWiring:
    def test_offer_fires_after_successful_setup(self, isolated_home, fake_tty, monkeypatch, capsys):
        _write_config(isolated_home, {"model": dict(PRIMARY)})

        def fake_picker():
            from hermes_cli.config import load_config, save_config
            cfg = load_config()
            cfg["model"] = {"provider": "openrouter", "default": "x/y"}
            save_config(cfg)

        prompted = []
        monkeypatch.setattr(
            "hermes_cli.setup.prompt_yes_no",
            lambda *a, **kw: prompted.append(1) or False,
        )
        monkeypatch.setattr(
            "hermes_cli.main.select_provider_and_model", fake_picker
        )

        from hermes_cli.setup import setup_model_provider
        config = {}
        setup_model_provider(config, quick=True)

        assert prompted, "wizard must offer fallback after successful setup"
        # The wizard re-syncs its config dict from disk afterwards — the
        # fallback offer must not corrupt that sync.
        assert config.get("model", {}).get("provider") == "openrouter"

    def test_no_offer_when_picker_raises(self, isolated_home, fake_tty, monkeypatch):
        """Wizard's except-branch path (cancel/error) skips the offer — the
        else-branch contract. A clean return still offers (covered by the
        success test above)."""
        _write_config(isolated_home, {"model": dict(PRIMARY)})

        def broken_picker():
            raise KeyboardInterrupt()

        prompted = []
        monkeypatch.setattr(
            "hermes_cli.setup.prompt_yes_no",
            lambda *a, **kw: prompted.append(1) or False,
        )
        monkeypatch.setattr(
            "hermes_cli.main.select_provider_and_model", broken_picker
        )

        from hermes_cli.setup import setup_model_provider
        config = {}
        setup_model_provider(config, quick=True)  # swallows KeyboardInterrupt

        assert not prompted, "error/cancel branch must skip the offer"


# ---------------------------------------------------------------------------
# Entry-point wiring — first-run setup
# ---------------------------------------------------------------------------

class TestFirstRunWiring:
    def test_offer_fires_on_first_run_success(self, isolated_home, fake_tty, monkeypatch):
        _write_config(isolated_home, {})

        def fake_picker():
            from hermes_cli.config import load_config, save_config
            cfg = load_config()
            cfg["model"] = dict(PRIMARY)
            save_config(cfg)

        prompted = []
        monkeypatch.setattr(
            "hermes_cli.setup.prompt_yes_no",
            lambda *a, **kw: prompted.append(1) or False,
        )
        monkeypatch.setattr(
            "hermes_cli.main.select_provider_and_model", fake_picker
        )
        # Answer the "Set up a provider now?" gate with yes.
        monkeypatch.setattr("builtins.input", lambda *a: "y")
        # _offer_first_run_setup imports _cprint from cli — keep it quiet-safe.
        monkeypatch.setattr("builtins.print", lambda *a, **kw: None)

        import cli as cli_module

        shell = cli_module.HermesCLI.__new__(cli_module.HermesCLI)
        # The method ends with a live credential probe; stub it so the test
        # stays hermetic (no real provider resolution).
        shell.requested_provider = None
        shell._explicit_api_key = None
        shell._explicit_base_url = None
        shell._runtime_credentials_ready = lambda: True
        shell.agent = None
        shell._active_agent_route_signature = None
        assert shell._offer_first_run_setup() is True
        assert prompted, "first-run must offer fallback after configuring a primary"

    def test_no_offer_on_first_run_decline(self, isolated_home, monkeypatch):
        _write_config(isolated_home, {})
        monkeypatch.setattr("builtins.input", lambda *a: "n")

        prompted = []
        monkeypatch.setattr(
            "hermes_cli.setup.prompt_yes_no",
            lambda *a, **kw: prompted.append(1) or False,
        )

        from cli import HermesCLI
        shell = HermesCLI.__new__(HermesCLI)
        assert shell._offer_first_run_setup() is False
        assert not prompted


# ---------------------------------------------------------------------------
# Non-interactive safety
# ---------------------------------------------------------------------------

class TestNonInteractive:
    def test_noninteractive_flag_defaults_to_decline(self, isolated_home, monkeypatch):
        """HERMES_NONINTERACTIVE=1 → prompt_yes_no returns default (False)."""
        import sys

        _write_config(isolated_home, {"model": dict(PRIMARY)})
        monkeypatch.setenv("HERMES_NONINTERACTIVE", "1")
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

        from hermes_cli.main import _maybe_offer_fallback_setup
        # Must not block, must not add anything.
        _maybe_offer_fallback_setup()

        cfg = _read_config(isolated_home)
        assert "fallback_providers" not in cfg

    def test_non_tty_skips_silently(self, isolated_home, monkeypatch):
        """Desktop-spawned / piped stdin → no prompt, no error, no change."""
        import sys

        _write_config(isolated_home, {"model": dict(PRIMARY)})
        monkeypatch.delenv("HERMES_NONINTERACTIVE", raising=False)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

        def fail_prompt(*a, **kw):
            raise AssertionError("prompt_yes_no must not be called without a TTY")

        with patch("hermes_cli.setup.prompt_yes_no", fail_prompt):
            from hermes_cli.main import _maybe_offer_fallback_setup
            # Must return cleanly even though everything else would allow it.
            _maybe_offer_fallback_setup()
