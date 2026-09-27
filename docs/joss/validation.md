# Reproduce the candidate checks

Run with the Python versions supported by the project. Create an isolated environment
first. Use a new output directory for each run; never overwrite research outputs.

```bash
python -m pip install ".[test]"
python -m pytest -q
python -m ezdic_cli benchmark --output outputs/joss-benchmark
python -m pip install build PyYAML cffconvert
python -m build
cffconvert --validate -i CITATION.cff
python scripts/check_joss_artifacts.py
```

Install the generated wheel in a second empty environment. Leave the source directory,
then run the documented numerical example with that interpreter. Check `pip check`.
Record the Git commit, platform, Python/dependency versions, commands, exit status,
output paths and CI URLs. GUI construction tests are not a complete interactive-user
acceptance test. The source and installed distributions must agree.

`check_joss_artifacts.py` checks paper sections, approximate prose word count, author
identity, bibliography keys, figure paths and citation metadata. It reports unresolved
submission checks without failing technical CI. Use `--submission` to return nonzero
while those records are incomplete. It does not validate the truth of author statements,
actual public history, current policy, experimental correctness or editorial eligibility.

The draft-PDF workflow compiles the manuscript with Open Journals tooling. Download
its artifact for the selected commit and inspect every page, including references,
formulas, figure captions and author details. Rebuild after manuscript changes. The
paper date is a preparation date until the author selects the actual submission date.

Final submission still requires a human check of substantive public iteration over
more than six months, the actual research-use mapping, author review of all assisted
work, current policy and related-publication disclosure. Record the selected SHA and
all evidence in the local submission package; do not replace failures with old CI.
