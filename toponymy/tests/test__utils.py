"""Tests for _utils.py functions."""

import pytest
import warnings
from toponymy._utils import resolve_api_key, handle_verbose_params


class TestResolveApiKey:
    """Tests for resolve_api_key function."""

    def test_explicit_key_takes_precedence(self):
        """Explicitly provided API key should take precedence over env vars."""
        result = resolve_api_key("explicit-key", env_new="TEST_KEY")
        assert result == "explicit-key"

    def test_explicit_key_ignores_env(self, monkeypatch):
        """Explicit key should be used even when env var is set."""
        monkeypatch.setenv("TEST_KEY", "env-key")
        result = resolve_api_key("explicit-key", env_new="TEST_KEY")
        assert result == "explicit-key"

    def test_empty_string_falls_back_to_env(self, monkeypatch):
        """Empty string should be treated as None and fall back to env var."""
        monkeypatch.setenv("TEST_KEY", "env-key")
        result = resolve_api_key("", env_new="TEST_KEY")
        assert result == "env-key"

    def test_whitespace_string_falls_back_to_env(self, monkeypatch):
        """Whitespace-only string should be treated as None."""
        monkeypatch.setenv("TEST_KEY", "env-key")
        result = resolve_api_key("   ", env_new="TEST_KEY")
        assert result == "env-key"

    def test_whitespace_with_trailing_spaces_normalized(self):
        """API key with trailing/leading whitespace should be stripped."""
        result = resolve_api_key("  my-key  ", env_new="TEST_KEY")
        assert result == "my-key"

    def test_env_var_used_when_no_explicit_key(self, monkeypatch):
        """Environment variable should be used when no explicit key."""
        monkeypatch.setenv("TEST_KEY", "env-key")
        result = resolve_api_key(None, env_new="TEST_KEY")
        assert result == "env-key"

    def test_env_var_whitespace_normalized(self, monkeypatch):
        """Environment variable with whitespace should be stripped."""
        monkeypatch.setenv("TEST_KEY", "  env-key  ")
        result = resolve_api_key(None, env_new="TEST_KEY")
        assert result == "env-key"

    def test_empty_env_var_falls_back_to_legacy(self, monkeypatch):
        """Empty env var should fall back to legacy."""
        monkeypatch.setenv("NEW_KEY", "")
        monkeypatch.setenv("OLD_KEY", "legacy-key")

        with pytest.warns(FutureWarning):
            result = resolve_api_key(None, env_new="NEW_KEY", env_legacy="OLD_KEY")

        assert result == "legacy-key"

    def test_whitespace_env_var_falls_back_to_legacy(self, monkeypatch):
        """Whitespace-only env var should fall back to legacy."""
        monkeypatch.setenv("NEW_KEY", "   ")
        monkeypatch.setenv("OLD_KEY", "legacy-key")

        with pytest.warns(FutureWarning):
            result = resolve_api_key(None, env_new="NEW_KEY", env_legacy="OLD_KEY")

        assert result == "legacy-key"

    def test_legacy_env_var_with_warning(self, monkeypatch):
        """Legacy env var should work with deprecation warning."""
        monkeypatch.delenv("NEW_KEY", raising=False)
        monkeypatch.setenv("OLD_KEY", "legacy-key")

        with pytest.warns(FutureWarning, match="OLD_KEY.*deprecated.*NEW_KEY"):
            result = resolve_api_key(None, env_new="NEW_KEY", env_legacy="OLD_KEY")

        assert result == "legacy-key"

    def test_legacy_env_var_normalized(self, monkeypatch):
        """Legacy env var with whitespace should be stripped."""
        monkeypatch.delenv("NEW_KEY", raising=False)
        monkeypatch.setenv("OLD_KEY", "  legacy-key  ")

        with pytest.warns(FutureWarning):
            result = resolve_api_key(None, env_new="NEW_KEY", env_legacy="OLD_KEY")

        assert result == "legacy-key"

    def test_empty_legacy_env_var_treated_as_missing(self, monkeypatch):
        """Empty legacy env var should be treated as not found."""
        monkeypatch.delenv("NEW_KEY", raising=False)
        monkeypatch.setenv("OLD_KEY", "")

        with pytest.raises(ValueError, match="No.*API key provided"):
            resolve_api_key(None, env_new="NEW_KEY", env_legacy="OLD_KEY")

    def test_required_raises_on_missing_key(self, monkeypatch):
        """Should raise ValueError when required=True and no key found."""
        monkeypatch.delenv("MISSING_KEY", raising=False)

        with pytest.raises(ValueError, match="No.*API key provided.*MISSING_KEY"):
            resolve_api_key(None, env_new="MISSING_KEY", required=True)

    def test_not_required_returns_none(self, monkeypatch):
        """Should return None when required=False and no key found."""
        monkeypatch.delenv("MISSING_KEY", raising=False)
        result = resolve_api_key(None, env_new="MISSING_KEY", required=False)
        assert result is None

    def test_empty_string_raises_when_required_no_env(self, monkeypatch):
        """Empty string should raise error when required=True and no env var."""
        monkeypatch.delenv("MISSING_KEY", raising=False)

        with pytest.raises(ValueError, match="No.*API key provided"):
            resolve_api_key("", env_new="MISSING_KEY", required=True)

    def test_whitespace_string_raises_when_required_no_env(self, monkeypatch):
        """Whitespace string should raise error when required=True and no env var."""
        monkeypatch.delenv("MISSING_KEY", raising=False)

        with pytest.raises(ValueError, match="No.*API key provided"):
            resolve_api_key("   ", env_new="MISSING_KEY", required=True)

    def test_error_message_format(self, monkeypatch):
        """Error message should have readable provider name."""
        monkeypatch.delenv("COHERE_API_KEY", raising=False)

        with pytest.raises(
            ValueError, match="No Cohere API key provided.*COHERE_API_KEY"
        ):
            resolve_api_key(None, env_new="COHERE_API_KEY", required=True)

    def test_no_env_new_with_explicit_key(self):
        """Should work with explicit key even if env_new is None."""
        result = resolve_api_key("my-key", env_new=None)
        assert result == "my-key"

    def test_no_env_new_required_raises(self):
        """Should raise with clear error when env_new is None and required."""
        with pytest.raises(ValueError, match="No API API key provided"):
            resolve_api_key(None, env_new=None, required=True)


class TestHandleVerboseParams:
    """Tests for handle_verbose_params function."""

    def test_verbose_true_returns_both_true(self):
        """verbose=True should return (True, True)."""
        show_pb, verbose = handle_verbose_params(verbose=True)
        assert show_pb is True
        assert verbose is True

    def test_verbose_false_returns_both_false(self):
        """verbose=False should return (False, False)."""
        show_pb, verbose = handle_verbose_params(verbose=False)
        assert show_pb is False
        assert verbose is False

    def test_verbose_takes_precedence_over_legacy(self):
        """verbose parameter should take precedence over legacy parameters."""
        show_pb, verbose = handle_verbose_params(
            verbose=False, verbose_legacy=True, show_progress_bar=True
        )
        assert show_pb is False
        assert verbose is False

    def test_show_progress_bar_with_deprecation_warning(self):
        """show_progress_bar should work with deprecation warning."""
        with pytest.warns(DeprecationWarning, match="show_progress_bar.*deprecated"):
            show_pb, verbose = handle_verbose_params(show_progress_bar=True)

        assert show_pb is True

    def test_verbose_legacy_with_deprecation_warning(self):
        """verbose_legacy should work with deprecation warning."""
        with pytest.warns(DeprecationWarning, match="verbose.*deprecated"):
            show_pb, verbose = handle_verbose_params(verbose_legacy=True)

        assert verbose is True
        assert (
            show_pb is True
        )  # When verbose_legacy=True, progress bar should also be True

    def test_show_progress_bars_with_deprecation_warning(self):
        """show_progress_bars should work with deprecation warning."""
        with pytest.warns(DeprecationWarning, match="show_progress_bars.*deprecated"):
            show_pb, verbose = handle_verbose_params(show_progress_bars=False)

        assert show_pb is False

    def test_show_progress_bars_takes_precedence_over_show_progress_bar(self):
        """show_progress_bars should take precedence over show_progress_bar."""
        with pytest.warns(DeprecationWarning):
            show_pb, verbose = handle_verbose_params(
                show_progress_bars=False, show_progress_bar=True
            )

        assert show_pb is False

    def test_multiple_legacy_params_in_warning(self):
        """Warning should list all legacy parameters used."""
        with pytest.warns(
            DeprecationWarning, match="verbose.*show_progress_bar.*deprecated"
        ):
            handle_verbose_params(verbose_legacy=True, show_progress_bar=True)

    def test_default_verbose_true(self):
        """Default should return (True, True) when no params provided."""
        show_pb, verbose = handle_verbose_params()
        assert show_pb is True
        assert verbose is True

    def test_default_verbose_false(self):
        """Should respect default_verbose=False."""
        show_pb, verbose = handle_verbose_params(default_verbose=False)
        assert show_pb is False
        assert verbose is False

    def test_verbose_legacy_true_sets_progress_bar(self):
        """When verbose_legacy=True, progress bar should also be True."""
        with pytest.warns(DeprecationWarning):
            show_pb, verbose = handle_verbose_params(verbose_legacy=True)

        assert verbose is True
        assert show_pb is True

    def test_verbose_legacy_false_uses_default_for_progress_bar(self):
        """When verbose_legacy=False, progress bar should use default."""
        with pytest.warns(DeprecationWarning):
            show_pb, verbose = handle_verbose_params(
                verbose_legacy=False, default_verbose=True
            )

        assert verbose is False
        assert show_pb is True  # Uses default

    def test_only_show_progress_bar_set(self):
        """When only show_progress_bar is set, verbose should use default."""
        with pytest.warns(DeprecationWarning):
            show_pb, verbose = handle_verbose_params(
                show_progress_bar=False, default_verbose=True
            )

        assert show_pb is False
        assert verbose is True  # Uses default

    def test_no_warning_when_verbose_used(self):
        """No warning should be issued when using the new verbose parameter."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            handle_verbose_params(verbose=True)  # Should not raise

    def test_no_warning_when_no_params(self):
        """No warning should be issued when using defaults."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            handle_verbose_params()  # Should not raise
