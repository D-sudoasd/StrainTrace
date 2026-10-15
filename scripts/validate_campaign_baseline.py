"""Read-only external golden-data import/export acceptance (no raw image reads)."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from straintrace_campaign import workflow

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))

def rows(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):h.update(block)
    return h.hexdigest()

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--work-dir",type=Path,required=True)
    args=parser.parse_args(argv)
    source=args.source.resolve();work=args.work_dir.resolve()
    if source==work or source in work.parents:
        parser.error("work-dir must be outside the read-only source")
    if (source/"CURRENT.json").exists():
        cur=read_json(source/"CURRENT.json")
        source=source/(cur.get("version_relative_path") or cur.get("version_path"))
    original_catalog=read_json(source/"catalog.json")
    source_pins={str(p.relative_to(source)):digest(p) for p in source.rglob("*") if p.is_file()}
    work.mkdir(parents=True,exist_ok=True)
    project_dir=work/"project"
    workflow.init_project(project_dir,"External accepted baseline",{})
    imported=workflow.import_keystone(project_dir,source)
    if len(imported)!=len(original_catalog):raise AssertionError("acquisition count changed")
    result=workflow.export(project_dir,work/"delivery",timeout_s=180)
    version=Path(result["version_path"])
    exported_catalog=read_json(version/"catalog.json")
    by_id={r["acquisition_id"]:r for r in exported_catalog}
    if set(by_id)!={r["acquisition_id"] for r in original_catalog}:raise AssertionError("acquisition identity changed")
    counts={"acquisitions":len(by_id),"frames":0,"finite_stress":0,"finite_strain":0,"finite_pairs":0}
    source_rows_exact=True
    for original in original_catalog:
        after=by_id[original["acquisition_id"]]
        a=rows(source/original["frames_jsonl"]);b=rows(version/after["frames_jsonl"])
        if a!=b:raise AssertionError("accepted frame records changed: "+original["acquisition_id"])
        counts["frames"]+=len(a)
        for row in a:
            stress=row.get("stress_MPa");strain=row.get("engineering_strain")
            counts["finite_stress"]+=stress is not None
            counts["finite_strain"]+=strain is not None
            counts["finite_pairs"]+=stress is not None and strain is not None
    if any(digest(source/rel)!=pin for rel,pin in source_pins.items()):raise AssertionError("source package was modified")
    report=workflow.verify(work/"delivery")
    if not report["valid"]:raise AssertionError(report)
    receipt={"passed":True,"status":"PASS",**counts,"all_frame_columns_exact":source_rows_exact,
             "source_files_unchanged":True,"raw_images_opened":False,"accepted_result_status":"IMPORTED_ACCEPTED",
             "source_manifest_sha256":hashlib.sha256(json.dumps(source_pins,sort_keys=True).encode()).hexdigest(),
             "verification":report}
    (work/"BASELINE_VALIDATION.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")
    print(json.dumps({k:v for k,v in receipt.items() if k!="verification"},ensure_ascii=False))
    return 0
if __name__=="__main__":raise SystemExit(main())
