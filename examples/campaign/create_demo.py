"""Generate a synthetic specimen and optionally exercise the real campaign CLI.

Only the generated synthetic specimen is automatically approved by this fixture.
This script never accepts an existing experimental image folder.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,subprocess,sys
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def create(output: Path):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory; existing files are protected")
    images=output/"images"/"synthetic_specimen"/"DIC"
    images.mkdir(parents=True)
    rng=np.random.default_rng(413)
    reference=np.full((200,512),10,np.uint8)
    texture=rng.integers(30,225,(160,472),dtype=np.uint8)
    # Several-pixel, band-limited speckles avoid the aliasing bias of single-
    # pixel noise; this is a synthetic regression, not an uncertainty estimate.
    reference[20:180,20:492]=cv2.GaussianBlur(texture,(7,7),1.2)
    metadata=[]
    for i in range(6):
        image=cv2.warpAffine(reference,np.array([[1+0.002*i,0,-0.002*i*256],[0,1,0]],np.float64),(512,200),flags=cv2.INTER_CUBIC,borderValue=10)
        filename=f"dic_{i+1:06d}.tif"
        if not cv2.imwrite(str(images/filename),image):raise OSError(filename)
        metadata.append({"filename":filename,"frame":i+1,"native_frame":i+1,"timestamp_unix_s":1700000000.0+i,
                         "time_s":float(i),"timestamp_status":"SYNTHETIC_RECORDED_CLOCK","timestamp_source":"generated fixture clock"})
    with (images/"frame_metadata.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(metadata[0]));writer.writeheader();writer.writerows(metadata)
    with (output/"force.csv").open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=["timestamp_unix_s","force_N"]);writer.writeheader()
        for t in np.arange(0,6.25,0.25):writer.writerow({"timestamp_unix_s":1700000000.0+float(t),"force_N":1000+100*float(t)})
    (output/"SYNTHETIC_FIXTURE.json").write_text(json.dumps({"kind":"generated_synthetic_only","free_region":[20,20,472,160],"expected_engineering_strain":[0,.002,.004,.006,.008,.010]},indent=2),encoding="utf8")
    return images

def run_cli(output: Path, executable=None):
    command=[str(executable)] if executable else [sys.executable,"-B","-m","ezdic_cli"]
    def invoke(*args):
        result=subprocess.run(command+["campaign",*[str(a) for a in args]],cwd=ROOT,capture_output=True,text=True,encoding="utf8",timeout=180)
        if result.returncode:raise RuntimeError(result.stdout+"\n"+result.stderr)
        return json.loads(result.stdout.strip().splitlines()[-1])
    project=output/"project"
    invoke("init",project,"--name","Synthetic campaign","--root","images="+str((output/"images").resolve()))
    scan=invoke("scan",project,"images")
    if len(scan["acquisitions"])!=1:raise AssertionError("fixture must discover exactly one acquisition")
    acquisition=scan["acquisitions"][0]["acquisition_id"]
    # This synthetic nominal geometry is known by construction. Real geometry
    # must be recorded and confirmed through the workbench or project API.
    config=json.loads((project/"project.json").read_text(encoding="utf8"))
    config["acquisitions"][0]["dimensions"]={"width_mm":5.0,"thickness_mm":2.0,"area_mm2":10.0,"status":"confirmed","source":"synthetic nominal geometry"}
    (project/"project.json").write_text(json.dumps(config,ensure_ascii=False,allow_nan=False),encoding="utf8")
    invoke("propose-rois",project,acquisition,"--free-region",20,20,472,160)
    invoke("approve-rois",project,acquisition,"--reviewer","synthetic-fixture","--specimen-confirmed","true")
    first=invoke("run",project)
    first_state=json.loads((project/"project.json").read_text(encoding="utf8"))
    tracked=project/first_state["acquisitions"][0]["result"]["rows_path"]
    before=(tracked.stat().st_mtime_ns,hashlib.sha256(tracked.read_bytes()).hexdigest())
    resumed=invoke("run",project)
    if before!=(tracked.stat().st_mtime_ns,hashlib.sha256(tracked.read_bytes()).hexdigest()):
        raise AssertionError("unchanged analysis was rewritten instead of reused")
    aligned=invoke("align",project,acquisition,output/"force.csv","--max-gap-s",1)
    exported=invoke("export",project,output/"delivery","--publish-to",output/"published","--timeout-s",180)
    verified=invoke("verify",output/"delivery")
    if not verified["report"]["valid"]:raise AssertionError(verified)
    published=invoke("verify",output/"published")
    if not published["report"]["valid"]:raise AssertionError(published)
    version=Path(exported["receipt"]["version_path"])
    catalog=json.loads((version/"catalog.json").read_text(encoding="utf8"))
    rows=[json.loads(line) for line in (version/catalog[0]["frames_jsonl"]).read_text(encoding="utf8").splitlines()]
    if len(rows)!=6:raise AssertionError("synthetic frame identity/count changed")
    for i,row in enumerate(rows):
        if row.get("valid_gauge_count",0)<3 or row.get("engineering_strain") is None:
            raise AssertionError("synthetic extensometer mean is unsupported")
        if abs(row["engineering_strain"]-0.002*i)>5e-4:
            raise AssertionError("synthetic strain regression exceeded tolerance")
        if abs(row["stress_MPa"]-(100+10*i))>1e-8:
            raise AssertionError("synthetic clock/force/area mapping changed")
    report={"status":"PASS","tracking_cache_reused":True,"synthetic_numeric_gate":True,"first_run":first,"resumed_run":resumed,"alignment":aligned,"export":exported,"verification":verified}
    (output/"DEMO_VALIDATION.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf8")
    return report

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--run",action="store_true",help="also exercise the real CLI on these generated images")
    parser.add_argument("--cli",type=Path,help="optional frozen CLI executable")
    args=parser.parse_args(argv);output=args.output.resolve()
    create(output)
    if args.run:run_cli(output,args.cli)
    print(json.dumps({"output":str(output),"synthetic_only":True,"workflow_exercised":args.run}))
    return 0
if __name__=="__main__":raise SystemExit(main())
