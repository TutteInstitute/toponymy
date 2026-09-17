"""Fit external EVoC in a fresh interpreter with a private file handoff.

The child runs this file through a bootstrap, not with ``-m``, so it does not
import Toponymy or fast_hdbscan before EVoC runs. The bootstrap also keeps this
package directory off the child's import path, where types.py could shadow
the standard library. Pickle files are created and consumed only within one
fit's private temporary directory, not through a public model loading interface.
"""

from pathlib import Path
import pickle
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

_CHILD_BOOTSTRAP = (
    "import runpy\n"
    "import sys\n"
    "sys.argv = sys.argv[1:]\n"
    "runpy.run_path(sys.argv[0], run_name='__main__')\n"
)


def _stop_child(process):
    """Stop the fit before removing its mapped input and inherited log."""
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        # A venv launcher can exit before its interpreter releases these files.
        stopped = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if stopped.returncode and process.poll() is None:
            stopped.check_returncode()
        process.wait(timeout=5)
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _failure_output(path):
    # Bound the exception size even if an external library writes a large log.
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - 65536))
        output = stream.read().decode("utf-8", errors="replace")
    return ("[earlier child output omitted]\n" if size > 65536 else "") + output


def fit_isolated(vectors, options):
    """Return fitted EVoC data, or propagate a child failure.

    A fresh interpreter avoids the process-global Numba namedtuple fingerprint
    collision between EVoC and fast_hdbscan. The caller's environment and thread
    settings are inherited unchanged. No kernels or library globals are patched.
    """
    directory = TemporaryDirectory(prefix="toponymy-evoc-")
    failure = None
    try:
        work = Path(directory.name)
        np.save(work / "vectors.npy", vectors, allow_pickle=False)
        with (work / "options.pkl").open("wb") as stream:
            pickle.dump(options, stream, protocol=pickle.HIGHEST_PROTOCOL)
        command = [
            sys.executable,
            "-c",
            _CHILD_BOOTSTRAP,
            str(Path(__file__).resolve()),
            str(work),
        ]
        with (work / "child.log").open("wb") as log:
            process = subprocess.Popen(
                command,
                cwd=work,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                returncode = process.wait()
            except BaseException as error:
                try:
                    _stop_child(process)
                except Exception as cleanup_error:
                    if cleanup_error.__context__ is error:
                        cleanup_error.__context__ = None
                    raise error from cleanup_error
                raise
        if returncode:
            output = _failure_output(work / "child.log")
            failure = subprocess.CalledProcessError(returncode, command, output=output)
            raise RuntimeError(
                f"EVoC fit failed in its isolated process (exit {returncode}).\n{output}"
            ) from failure
        with (work / "model.pkl").open("rb") as stream:
            return pickle.load(stream)
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            directory.cleanup()
        except OSError as cleanup_error:
            if failure is not None:
                # Keep the child/termination failure when file removal also fails.
                previous = failure.__cause__
                if previous is None and not failure.__suppress_context__:
                    previous = failure.__context__
                cleanup_error.__cause__ = previous
                if cleanup_error.__context__ is failure:
                    cleanup_error.__context__ = None
                raise failure from cleanup_error
            raise


def _fitted_state(estimator):
    # Evaluate the native lazy property while still inside the fit boundary.
    tree = estimator.cluster_tree_
    state = SimpleNamespace(**vars(estimator))
    state.cluster_tree_ = tree
    return state


def _fit_child(directory):
    from evoc import EVoC

    work = Path(directory)
    with (work / "options.pkl").open("rb") as stream:
        options = pickle.load(stream)
    # Copy-on-write mapping supplies writable arrays to Numba without pipe copies
    # or allowing a library write to alter the saved input.
    vectors = np.load(work / "vectors.npy", mmap_mode="c", allow_pickle=False)
    estimator = EVoC(**options).fit(vectors)
    with (work / "model.pkl").open("wb") as stream:
        pickle.dump(_fitted_state(estimator), stream, protocol=pickle.HIGHEST_PROTOCOL)
    # Process exit closes the mapping before the parent removes the directory.


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("This private helper expects one fit directory")
    _fit_child(sys.argv[1])
