.. _installation:

Installation
============

Toponymy requires Python 3.10 or newer. To install a released package:

.. code-block:: shell

   pip install toponymy

To install a source checkout, run ``pip install .`` from its root. Use
``pip install '.[dev]'`` for the development dependencies. The source-tree
documentation describes v0.6; check your installed package version when using
an older release.

PLSCAN is supplied by the required ``fast-hdbscan`` dependency. EVoC is optional:

.. code-block:: shell

   pip install 'toponymy[evoc]'

For the optional contrastive TreeSHAP keyphrase extractor:

.. code-block:: shell

   pip install 'toponymy[treeshap]'

Naming wrappers may require a provider SDK or a separately configured local
model. Embedding providers are optional for the default exemplar-only pipeline
when you already have semantic vectors. See :doc:`llm_wrappers` and
:doc:`embedding_wrappers` for those boundaries.

The :doc:`basic_usage` example uses local arrays and deterministic responses.
It does not download models or call a service.

Windows paths
-------------

Long environment paths, including paths expanded by Windows app containers,
can exceed the filename limits used by numerical dependencies when Numba writes
compiled caches. This can raise ``FileNotFoundError`` during clustering or
feature extraction even when installation and imports succeed. Use a shorter
environment location, or set a short writable cache directory before starting
Python or a notebook kernel. For example, in PowerShell:

.. code-block:: powershell

   $env:NUMBA_CACHE_DIR = 'C:\work\tp-cache'
   New-Item -ItemType Directory -Force -Path $env:NUMBA_CACHE_DIR
   python your_script.py

This setting applies to processes started from that shell. It does not change
machine-wide path settings. Lance dataset paths and temporary directories can
also exceed native filename limits; keep those paths short separately, since
``NUMBA_CACHE_DIR`` only controls the numerical cache.

EVoC on Linux
-------------

EVoC's nested parallel work needs a thread-safe Numba runtime. With EVoC 0.3.1
and Numba 0.67, a Linux environment without loadable OpenMP or TBB libraries can
fall back to ``workqueue`` and abort during fitting. Set
``NUMBA_THREADING_LAYER=threadsafe`` before starting Python to require a suitable
backend; an unavailable backend then produces a configuration error.

For an activated virtual environment using TBB, its native libraries may also
need to be included in the process's library search path:

.. code-block:: shell

   python -m pip install tbb
   LD_LIBRARY_PATH="$VIRTUAL_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
     NUMBA_THREADING_LAYER=threadsafe python your_script.py

Installing TBB alone does not ensure the native loader can find it. Apply the
same environment when launching a notebook kernel. These settings affect the
launched process and its children; they do not change system configuration.
An environment with a working OpenMP runtime may already satisfy the requirement.
See `Numba's threading-layer documentation
<https://numba.readthedocs.io/en/stable/user/threading-layer.html>`_ for backend
selection and diagnostics. This runtime requirement is separate from Toponymy's
EVoC subprocess isolation for numerical type-cache compatibility.
