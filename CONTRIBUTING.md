# Contributing

Report reproducible defects or propose changes through GitHub issues and pull requests.
Include the source commit, operating system, Python version, a minimal permitted input,
configuration and observed/expected result. Remove private images and paths first.

Install with `python -m pip install -e ".[test]"`, run `python -m pytest -q`, and
run the locked synthetic benchmark described in `docs/joss/validation.md`.
Numerical changes must retain displacement/strain conventions, explicit invalid states,
fixed-reference semantics and manifest verification. Add a regression for the original
failure; do not relax a locked tolerance merely to make an algorithm pass.

Delun Gong maintains the project and reviews changes. There is no guaranteed response
time. Contributions are made under the MIT license. Credit substantive contributions
in the change history and assess software-paper authorship using JOSS criteria.
