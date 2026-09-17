"""Subprocess ownership, failure handling, and real clustering coexistence."""

from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.base import clone

from toponymy import _evoc
from toponymy.clustering import EVoCClusterer, PLSCANClusterer, validate_cluster_tree


def test_evoc_default_isolation_preserves_fitted_data(monkeypatch):
    calls = []
    fitted = SimpleNamespace(
        cluster_layers_=[np.array([4, 4, 9, 9, -1, -1])],
        membership_strength_layers_=[np.ones(6)],
        persistence_scores_=[3.5],
    )

    class ExternalEVoC:
        def __init__(self, **options):
            raise AssertionError("Isolated fits must not instantiate in the parent")

    def fit_isolated(vectors, options):
        calls.append((vectors, options))
        return fitted

    monkeypatch.setitem(sys.modules, "evoc", SimpleNamespace(EVoC=ExternalEVoC))
    monkeypatch.setattr(_evoc, "fit_isolated", fit_isolated)
    vectors = np.arange(12.0).reshape(6, 2)
    adapter = EVoCClusterer(random_state=31).fit(vectors)
    assert adapter.isolated is True
    assert adapter.evoc_ is fitted
    assert calls[0][0] is vectors
    assert calls[0][1]["random_state"] == 31
    assert "verbose" not in calls[0][1]
    assert "isolated" not in calls[0][1]
    np.testing.assert_array_equal(
        adapter.cluster_layers_[0].labels, fitted.cluster_layers_[0]
    )
    assert clone(EVoCClusterer(isolated=False)).isolated is False


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_evoc_isolation_flag_validated_before_empty_input(value):
    with pytest.raises(ValueError, match="isolated must be a boolean"):
        EVoCClusterer(isolated=value).fit(np.empty((0, 2)))


def test_private_handoff_roundtrips_model_and_cleans_up(monkeypatch):
    launched = []
    source = np.arange(24, dtype=np.float32).reshape(6, 4)[:, ::2]
    fitted = SimpleNamespace(cluster_layers_=[np.arange(6)], details="retained")

    class Child:
        def __init__(self, command, **kwargs):
            work = Path(command[-1])
            launched.append((command, kwargs, work))
            np.testing.assert_array_equal(np.load(work / "vectors.npy"), source)
            with (work / "options.pkl").open("rb") as stream:
                assert pickle.load(stream) == {"random_state": 31}
            with (work / "model.pkl").open("wb") as stream:
                pickle.dump(fitted, stream)

        def wait(self):
            return 0

    monkeypatch.setattr(_evoc.subprocess, "Popen", Child)
    result = _evoc.fit_isolated(source, {"random_state": 31})
    assert result.details == "retained"
    np.testing.assert_array_equal(result.cluster_layers_[0], np.arange(6))
    command, kwargs, work = launched[0]
    assert command[0] == sys.executable
    assert command[1] == "-c"
    assert "runpy.run_path" in command[2]
    assert Path(command[3]) == Path(_evoc.__file__).resolve()
    assert kwargs["cwd"] == work
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["stderr"] == subprocess.STDOUT
    assert "env" not in kwargs
    assert "shell" not in kwargs
    assert not work.exists()
    np.testing.assert_array_equal(
        source, np.arange(24, dtype=np.float32).reshape(6, 4)[:, ::2]
    )


def test_child_failure_propagates_traceback_and_cleans_up(monkeypatch):
    directories = []

    class Child:
        def __init__(self, command, **kwargs):
            directories.append(Path(command[-1]))
            kwargs["stdout"].write(b"ValueError: external constructor rejected input\n")

        def wait(self):
            return 7

    monkeypatch.setattr(_evoc.subprocess, "Popen", Child)
    with pytest.raises(RuntimeError, match="external constructor rejected") as caught:
        _evoc.fit_isolated(np.ones((6, 2)), {})
    assert isinstance(caught.value.__cause__, subprocess.CalledProcessError)
    assert caught.value.__cause__.returncode == 7
    assert not directories[0].exists()


@pytest.mark.parametrize("terminate_times_out", [False, True])
def test_interrupt_reaps_posix_child_before_temporary_cleanup(
    monkeypatch, terminate_times_out
):
    monkeypatch.setattr(
        _evoc, "sys", SimpleNamespace(platform="linux", executable=sys.executable)
    )
    instances = []

    class Child:
        def __init__(self, command, **kwargs):
            self.work = Path(command[-1])
            self.wait_count = 0
            self.terminated = False
            self.killed = False
            instances.append(self)

        def wait(self, timeout=None):
            assert self.work.exists()
            self.wait_count += 1
            if self.wait_count == 1:
                raise KeyboardInterrupt
            if self.wait_count == 2 and terminate_times_out:
                raise subprocess.TimeoutExpired("child", timeout)
            return 0

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    monkeypatch.setattr(_evoc.subprocess, "Popen", Child)
    with pytest.raises(KeyboardInterrupt):
        _evoc.fit_isolated(np.ones((6, 2)), {})
    child = instances[0]
    assert child.terminated
    assert child.killed == terminate_times_out
    assert not child.work.exists()


def test_launch_failure_cleans_up_and_preserves_exception(monkeypatch):
    directories = []

    def cannot_launch(command, **kwargs):
        directories.append(Path(command[-1]))
        raise OSError("cannot start the configured interpreter")

    monkeypatch.setattr(_evoc.subprocess, "Popen", cannot_launch)
    with pytest.raises(OSError, match="configured interpreter"):
        _evoc.fit_isolated(np.ones((6, 2)), {})
    assert not directories[0].exists()


@pytest.mark.parametrize("child_exited", [False, True])
def test_windows_tree_stop_reports_failure_unless_child_already_exited(
    monkeypatch, child_exited
):
    polls = iter([None, 0 if child_exited else None])
    waits = []
    process = SimpleNamespace(
        pid=12345, poll=lambda: next(polls), wait=lambda timeout: waits.append(timeout)
    )

    def stop(command, **kwargs):
        assert command == ["taskkill", "/PID", "12345", "/T", "/F"]
        assert kwargs["timeout"] == 10
        assert kwargs["stdout"] == subprocess.DEVNULL
        assert kwargs["stderr"] == subprocess.PIPE
        assert kwargs["creationflags"] == 0x08000000
        return subprocess.CompletedProcess(command, 128, stderr=b"termination failed")

    monkeypatch.setattr(_evoc, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(
        _evoc,
        "subprocess",
        SimpleNamespace(
            run=stop,
            DEVNULL=subprocess.DEVNULL,
            PIPE=subprocess.PIPE,
            CREATE_NO_WINDOW=0x08000000,
        ),
    )
    if child_exited:
        _evoc._stop_child(process)
        assert waits == [5]
    else:
        with pytest.raises(subprocess.CalledProcessError) as caught:
            _evoc._stop_child(process)
        assert caught.value.returncode == 128
        assert caught.value.stderr == b"termination failed"
        assert waits == []


def test_failure_output_is_bounded(tmp_path):
    log = tmp_path / "child.log"
    log.write_bytes(b"a" * 70000 + b"\nfinal error")
    output = _evoc._failure_output(log)
    assert output.startswith("[earlier child output omitted]")
    assert output.endswith("final error")
    assert len(output) < 65600


class MappedExternalEVoC:
    """Pickleable fake proving the child boundary supplies writable mmap data."""

    def __init__(self, **options):
        self.options = options

    def fit(self, vectors):
        assert isinstance(vectors, np.memmap)
        assert vectors.flags.writeable
        vectors[0, 0] = 99
        self.cluster_layers_ = [np.zeros(len(vectors), dtype=np.int64)]
        self.cluster_tree_ = {(1, 0): [(0, 0)]}
        return self


def test_child_uses_copy_on_write_mapped_input(monkeypatch, tmp_path):
    source = np.ones((6, 2), dtype=np.float32)
    np.save(tmp_path / "vectors.npy", source, allow_pickle=False)
    with (tmp_path / "options.pkl").open("wb") as stream:
        pickle.dump({"random_state": 31}, stream)
    monkeypatch.setitem(sys.modules, "evoc", SimpleNamespace(EVoC=MappedExternalEVoC))
    _evoc._fit_child(tmp_path)
    np.testing.assert_array_equal(np.load(tmp_path / "vectors.npy"), source)
    with (tmp_path / "model.pkl").open("rb") as stream:
        model = pickle.load(stream)
    assert isinstance(model, SimpleNamespace)
    assert not hasattr(model, "fit")
    assert model.cluster_tree_ == {(1, 0): [(0, 0)]}
    assert model.options == {"random_state": 31}


def _two_blobs():
    rng = np.random.default_rng(1729)
    return np.vstack(
        [rng.normal(-4, 0.1, (48, 8)), rng.normal(4, 0.1, (48, 8))]
    ).astype(np.float32)


@pytest.mark.real_clustering
@pytest.mark.parametrize("evoc_first", [False, True])
def test_real_adapters_coexist_in_both_fit_orders(evoc_first):
    from evoc import EVoC

    vectors = _two_blobs()
    original = vectors.copy()
    evoc = EVoCClusterer(
        random_state=31, n_neighbors=8, n_epochs=10, approx_n_clusters=2
    )
    plscan = PLSCANClusterer(reproducible=True)
    adapters = (evoc, plscan) if evoc_first else (plscan, evoc)
    for adapter in adapters:
        adapter.fit(vectors)
        assert adapter.cluster_layers_
        assert all(len(layer.labels) == len(vectors) for layer in adapter)
        validate_cluster_tree(adapter.cluster_tree_, adapter.cluster_layers_)
    assert not isinstance(evoc.evoc_, EVoC)
    assert isinstance(evoc.evoc_.cluster_tree_, dict)
    assert not hasattr(evoc.evoc_, "fit")
    assert not hasattr(evoc.evoc_, "fit_predict")
    assert hasattr(evoc.evoc_, "membership_strength_layers_")
    assert hasattr(evoc.evoc_, "nn_inds_")
    np.testing.assert_array_equal(vectors, original)


@pytest.mark.real_clustering
def test_real_evoc_preserves_natural_default_layers_after_plscan():
    rng = np.random.default_rng(1729)
    axes = np.eye(8, dtype=np.float32)
    groups = []
    for major in range(4):
        for minor in range(4):
            center = 3 * axes[major] + 0.9 * axes[4 + minor]
            groups.append(rng.normal(center, (0.025, 0.05, 0.09, 0.15)[minor], (32, 8)))
    vectors = np.vstack(groups).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    PLSCANClusterer(reproducible=True).fit(_two_blobs())
    adapter = EVoCClusterer(random_state=31).fit(vectors)
    native = adapter.evoc_
    assert native.approx_n_clusters is None
    assert native.base_n_clusters is None
    assert native.max_layers == 10
    assert native.n_epochs == 50
    assert len(adapter.cluster_layers_) == len(native.cluster_layers_) >= 1
    for layer, native_labels in zip(adapter.cluster_layers_, native.cluster_layers_):
        np.testing.assert_array_equal(layer.labels, native_labels)
    validate_cluster_tree(adapter.cluster_tree_, adapter.cluster_layers_)
