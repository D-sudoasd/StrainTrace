"""Check submission artifacts, not editorial eligibility. Requires PyYAML."""
from pathlib import Path
import argparse
import json
import re
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["Summary", "Statement of need", "State of the field", "Software design",
            "Research impact statement", "AI usage disclosure", "Acknowledgements", "References"]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", action="store_true", help="also fail on outstanding author/history records")
    args = parser.parse_args()
    errors = []
    record = json.loads((ROOT / "docs/joss/submission-record.json").read_text(encoding="utf-8"))
    paper = (ROOT / "paper/paper.md").read_text(encoding="utf-8")
    _, header, body = paper.split("---", 2)
    meta = yaml.safe_load(header)
    author = record["author"]
    expected = {"name": author["name"], "orcid": author["orcid"], "email": author["email"]}
    if len(meta.get("authors", [])) != 1:
        errors.append("Expected the confirmed sole author")
    else:
        for key, value in expected.items():
            if str(meta["authors"][0].get(key)) != value:
                errors.append("Author mismatch: " + key)
    if not any(a.get("name") == author["affiliation"] for a in meta.get("affiliations", [])):
        errors.append("Affiliation does not match the confirmed record")
    for heading in REQUIRED:
        if "# " + heading + "\n" not in body:
            errors.append("Missing section: " + heading)
    bib = (ROOT / "paper/paper.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\s*\{\s*([^,]+),", bib))
    cited = set(re.findall(r"(?<![\w.])@([A-Za-z][\w:.-]*)", body))
    for key in sorted(cited - keys):
        errors.append("Unresolved citation: " + key)
    # Whitespace count of prose before References, excluding fenced code and headings.
    prose = body.split("# References", 1)[0]
    prose = re.sub(r"```.*?```", "", prose, flags=re.S)
    prose = re.sub(r"^#+.*$", "", prose, flags=re.M)
    words = len(prose.split())
    if not 750 <= words <= 1750:
        errors.append("Draft prose word count outside 750-1750: " + str(words))
    for target in re.findall(r"!\[[^]]*\]\(([^)]+)\)", body):
        if not target.startswith("http") and not (ROOT / "paper" / target).is_file():
            errors.append("Missing figure: " + target)
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    if citation.get("license") != "MIT":
        errors.append("Current citation license must be MIT")
    if citation.get("repository-code") != record["repository"]:
        errors.append("Citation repository mismatch")
    pending = []
    if not record["history"]["public_since_verified"]:
        pending.append("Verify public-availability start and more than six months of substantive development")
    use = record["research_use"]
    for field in ["historical_version", "specific_operations_and_outputs", "processing_record_location"]:
        if not use.get(field):
            pending.append("Research-use record: " + field)
    if not record["declarations"]["human_review_of_this_revision_confirmed"]:
        pending.append("Author review of this revision and complete AI disclosure")
    if not record.get("submission_date"):
        pending.append("Set actual submission date and recompile paper")
    print(json.dumps({"artifact_errors": errors, "draft_prose_words": words,
                      "pending_submission_checks": pending,
                      "note": "Mechanical artifact checks do not establish JOSS eligibility or scientific validity."}, indent=2))
    return 1 if errors or (args.submission and pending) else 0

if __name__ == "__main__":
    sys.exit(main())
