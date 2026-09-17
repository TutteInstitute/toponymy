
# Contributing

Contributions of all kinds are welcome. In particular pull requests are appreciated. 
The authors will endeavour to help walk you through any issues in the pull request
discussion, so please feel free to open a pull request even if you are new to such things.

## Issues

The easiest contribution to make is to [file an issue](https://github.com/TutteInstitute/toponymy/issues/new).
First, search the [existing issues](https://github.com/TutteInstitute/toponymy/issues)
for similar reports. It also helps to provide clear instructions for reproducing
the problem. If you have resolved an issue yourself, please share the resolution
on the issue or in the documentation so others can benefit from your work.

## Documentation

Contributing to documentation is the easiest way to get started. Providing simple
clear or helpful documentation for new users is critical. Anything that *you* as 
a new user found hard to understand, or difficult to work out, are excellent places
to begin. Contributions to more detailed and descriptive error messages is
especially appreciated. To contribute to the documentation please 
[fork the project](https://github.com/TutteInstitute/toponymy/fork)
into your own repository, make changes there, and then submit a pull request.

### Building the Documentation Locally

To build the docs locally, install the documentation tools requirements:

```bash
pip install -r doc/requirements.txt
```

Then run:

```bash
sphinx-build -b html doc doc/_build
```

This will build the documentation in HTML format. You will be able to find the output
in the `doc/_build` folder.

## Code

Code contributions are always welcome, from simple bug fixes, to new features. To
contribute code please 
[fork the project](https://github.com/TutteInstitute/toponymy/fork)
into your own repository, make changes there, and then submit a pull request. If
you are fixing a known issue please add the issue number to the PR message. If you
are fixing a new issue feel free to file an issue and then reference it in the PR.
You can [browse open issues](https://github.com/TutteInstitute/toponymy/issues).

### Code formatting

This project uses [black](https://github.com/python/black) for code formatting (version pinned in `pyproject.toml`). 
All code contributions must be formatted with black before submitting a pull request.

**Using pip:**

```bash
pip install -e '.[dev]'
black toponymy/ doc/
```

**Using uv:**

```bash
uv sync --extra dev
uv run black toponymy/ doc/
```

The CI system will automatically check that code is properly formatted. 
If the check fails, you'll see which files need formatting.

### Running the Tests

Toponymy uses `pytest`. The consolidated tests live under `tests/`.

Install the project dependencies from the repo root:
```shell
pip install --upgrade uv
uv sync --extra dev
```
Run all tests:
```shell
uv run pytest tests -v
```
Run a specific test file:
```shell
uv run pytest tests/test_pipeline_contracts.py -v
```
Run tests with coverage:
```shell
uv run pytest tests --show-capture=no -v \
  --junitxml=junit/test-results.xml \
  --cov=toponymy/ --cov-report=xml --cov-report=html
```

The default suite uses local inputs and fake provider transports. Tests requiring
cached embedding models are retained under `tests/upstream/` and skipped unless
`--run-local-models` is supplied; no model is downloaded by those fixtures.
Real clustering tests use small local arrays. Property checks use a bounded quick
profile by default; set `TOPONYMY_PROPERTY_PROFILE=extended` for 3,000 clustering
examples and `TOPONYMY_EXTENDED_TESTS=1` for 2,000 parser examples. See the test
modules for their exact budgets.

