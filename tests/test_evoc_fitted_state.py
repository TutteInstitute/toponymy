"""Fitted data ownership and failure contracts across the EVoC file handoff."""

from copy import copy, deepcopy
from pathlib import Path
import pickle
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest

from toponymy import _evoc
from toponymy.clustering import EVoCClusterer


class NativeEVoCDouble:
    """Pickleable native-shaped object with observable lifecycle/property calls."""

    fit_calls = []
    tree_reads = []

    def __init__(self, **options):
        self.__dict__.update(options)

    def fit(self, vectors):
        type(self).fit_calls.append("fit")
        count = len(vectors)
        self.cluster_layers_ = [np.array([0, 0, 1, 1, -1, -1], dtype=np.int64)]
        assert len(self.cluster_layers_[0]) == count
        self.membership_strength_layers_ = [
            np.array([0.9, 0.8, 0.7, 0.6, 0.0, 0.0], dtype=np.float64)
        ]
        self.persistence_scores_ = [3.5]
        self.nn_inds_ = np.tile(np.array([0, 1], dtype=np.int64), (count, 1))
        self.nn_dists_ = np.tile(np.array([0.0, 0.25]), (count, 1))
        self.duplicates_ = {(0, 1)}
        self.labels_ = self.cluster_layers_[0]
        self.membership_strengths_ = self.membership_strength_layers_[0]
        return self

    def fit_predict(self, vectors, y=None, **fit_params):
        type(self).fit_calls.append("fit_predict")
        return self.fit(vectors).labels_

    @property
    def cluster_tree_(self):
        type(self).tree_reads.append("native tree")
        return {(1, 0): [(0, 0), (0, 1)]}


@pytest.fixture
def fake_evoc_handoff(monkeypatch, tmp_path):
    """Use real file handoff code without launching a process or native kernel."""
    monkeypatch.setattr(NativeEVoCDouble, "fit_calls", [])
    monkeypatch.setattr(NativeEVoCDouble, "tree_reads", [])
    monkeypatch.setitem(sys.modules, "evoc", SimpleNamespace(EVoC=NativeEVoCDouble))
    monkeypatch.setattr(
        _evoc,
        "TemporaryDirectory",
        lambda **kwargs: TemporaryDirectory(dir=tmp_path, **kwargs),
    )
    directories = []

    class Child:
        def __init__(self, command, **kwargs):
            self.directory = Path(command[-1])
            directories.append(self.directory)

        def wait(self):
            # Execute the actual child handoff function with a fake EVoC. This
            # accommodates state conversion in either child or parent without
            # replacing fit_isolated with the contract we intend to test.
            _evoc._fit_child(self.directory)
            return 0

    monkeypatch.setattr(_evoc.subprocess, "Popen", Child)
    return directories


def _vectors():
    return np.arange(24, dtype=np.float32).reshape(6, 4)[:, ::2]


def _adapter(isolated=True):
    return EVoCClusterer(
        random_state=31, n_neighbors=8, n_epochs=10, isolated=isolated
    ).fit(_vectors())


@pytest.mark.parametrize("isolated", [True, False])
def test_public_evoc_state_preserves_native_data(fake_evoc_handoff, isolated):
    adapter = _adapter(isolated)
    state = adapter.evoc_

    # Retain all documented native learned data and constructor values. No
    # assumption is made about the replacement state container's class name.
    assert len(state.cluster_layers_) == 1
    np.testing.assert_array_equal(state.labels_, [0, 0, 1, 1, -1, -1])
    np.testing.assert_array_equal(state.cluster_layers_[0], state.labels_)
    np.testing.assert_array_equal(
        state.membership_strengths_, [0.9, 0.8, 0.7, 0.6, 0.0, 0.0]
    )
    np.testing.assert_array_equal(
        state.membership_strength_layers_[0], state.membership_strengths_
    )
    assert state.persistence_scores_ == [3.5]
    np.testing.assert_array_equal(state.nn_inds_, np.tile([0, 1], (6, 1)))
    np.testing.assert_array_equal(state.nn_dists_, np.tile([0.0, 0.25], (6, 1)))
    assert state.duplicates_ == {(0, 1)}
    for name, value in adapter.get_params(deep=False).items():
        if name not in {"isolated", "verbose"}:
            assert getattr(state, name) == value

    np.testing.assert_array_equal(adapter.cluster_layers_[0].labels, state.labels_)
    assert not np.shares_memory(adapter.cluster_layers_[0].labels, state.labels_)
    assert not isinstance(state, NativeEVoCDouble)
    assert not any(
        isinstance(value, NativeEVoCDouble) for value in vars(state).values()
    ), "The fitted-state container must not expose a wrapped native estimator"
    assert all(not directory.exists() for directory in fake_evoc_handoff)


@pytest.mark.parametrize("isolated", [True, False])
def test_public_native_tree_is_materialized_data(fake_evoc_handoff, isolated):
    state = _adapter(isolated).evoc_
    reads_after_fit = list(NativeEVoCDouble.tree_reads)
    assert state.cluster_tree_ == {(1, 0): [(0, 0), (0, 1)]}
    assert state.cluster_tree_ == {(1, 0): [(0, 0), (0, 1)]}
    assert NativeEVoCDouble.tree_reads == reads_after_fit


def _unchanged(value):
    return value


def _pickle_roundtrip(value):
    return pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


@pytest.mark.parametrize("isolated", [True, False])
@pytest.mark.parametrize("method", ["fit", "fit_predict"])
@pytest.mark.parametrize(
    "roundtrip",
    [_unchanged, copy, deepcopy, _pickle_roundtrip],
    ids=["original", "copy", "deepcopy", "pickle"],
)
def test_public_state_cannot_reenter_native_fitting(
    fake_evoc_handoff, isolated, method, roundtrip
):
    adapter = roundtrip(_adapter(isolated))
    state = adapter.evoc_
    fit_calls_before = list(NativeEVoCDouble.fit_calls)
    labels_before = state.labels_.copy()

    # Both an absent lifecycle and an explicit unsupported-operation error
    # satisfy the data-only contract. Native fitting must never be entered.
    with pytest.raises((AttributeError, RuntimeError)):
        getattr(state, method)(_vectors())

    assert NativeEVoCDouble.fit_calls == fit_calls_before
    np.testing.assert_array_equal(state.labels_, labels_before)


@pytest.mark.parametrize("isolated", [True, False])
def test_adapter_refit_keeps_previous_public_state(fake_evoc_handoff, isolated):
    adapter = _adapter(isolated)
    previous = adapter.evoc_
    previous_labels = previous.labels_.copy()
    previous_neighbors = previous.nn_inds_.copy()

    assert adapter.fit(_vectors()) is adapter
    assert adapter.evoc_ is not previous
    np.testing.assert_array_equal(previous.labels_, previous_labels)
    np.testing.assert_array_equal(previous.nn_inds_, previous_neighbors)
    layers, tree = adapter.fit_predict(_vectors())
    assert layers is adapter.cluster_layers_
    assert tree is adapter.cluster_tree_


def test_failed_child_stop_preserves_initiating_interrupt(monkeypatch, tmp_path):
    interruption = KeyboardInterrupt("stop this fit")
    stop_failure = OSError("owned child termination failed")
    stopped = []
    created = []

    class Child:
        def __init__(self, command, **kwargs):
            created.append(self)

        def wait(self):
            raise interruption

    def cannot_stop(child):
        stopped.append(child)
        raise stop_failure

    monkeypatch.setattr(_evoc.subprocess, "Popen", Child)
    monkeypatch.setattr(_evoc, "_stop_child", cannot_stop)
    monkeypatch.setattr(
        _evoc,
        "TemporaryDirectory",
        lambda **kwargs: TemporaryDirectory(dir=tmp_path, **kwargs),
    )

    with pytest.raises(KeyboardInterrupt, match="stop this fit") as caught:
        _evoc.fit_isolated(_vectors(), {})

    assert caught.value is interruption
    assert len(created) == 1
    assert stopped == created
    assert (
        caught.value.__cause__ is stop_failure
        or caught.value.__context__ is stop_failure
    ), "Preserve failed termination as inspectable exception evidence"


@pytest.mark.parametrize("interrupted", [False, True])
def test_failed_file_cleanup_preserves_primary_failure(
    monkeypatch, tmp_path, interrupted
):
    primary = KeyboardInterrupt("stop fit") if interrupted else ValueError("bad fit")
    cleanup = PermissionError("mapped file still held")

    class Directory:
        name = str(tmp_path)

        def cleanup(self):
            raise cleanup

    def cannot_launch(*args, **kwargs):
        raise primary

    monkeypatch.setattr(_evoc, "TemporaryDirectory", lambda **kwargs: Directory())
    monkeypatch.setattr(_evoc.subprocess, "Popen", cannot_launch)
    with pytest.raises(type(primary)) as caught:
        _evoc.fit_isolated(_vectors(), {})
    assert caught.value is primary
    assert caught.value.__cause__ is cleanup


@pytest.mark.parametrize("interrupted", [False, True])
def test_file_cleanup_retains_child_or_termination_cause(
    monkeypatch, tmp_path, interrupted
):
    primary = KeyboardInterrupt("stop fit")
    stop_error = OSError("could not terminate child")
    file_error = PermissionError("could not remove mapped file")

    class Directory:
        name = str(tmp_path)

        def cleanup(self):
            raise file_error

    class Child:
        def __init__(self, command, **kwargs):
            kwargs["stdout"].write(b"native fit failure")

        def wait(self):
            if interrupted:
                raise primary
            return 7

    def cannot_stop(process):
        raise stop_error

    monkeypatch.setattr(_evoc, "TemporaryDirectory", lambda **kwargs: Directory())
    monkeypatch.setattr(_evoc.subprocess, "Popen", Child)
    monkeypatch.setattr(_evoc, "_stop_child", cannot_stop)
    with pytest.raises(KeyboardInterrupt if interrupted else RuntimeError) as caught:
        _evoc.fit_isolated(_vectors(), {})
    assert caught.value.__cause__ is file_error
    if interrupted:
        assert caught.value is primary
        assert file_error.__cause__ is stop_error
    else:
        assert isinstance(file_error.__cause__, _evoc.subprocess.CalledProcessError)
        assert file_error.__cause__.returncode == 7
