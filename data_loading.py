
from pathlib import Path
import urllib.request
from config import DATA, REFERENCE, ADV_URL
from common import load_json

def load_issues():
    issues=load_json(DATA/"issues.json")
    assert len(issues)==50, f"Expected 50 issues, found {len(issues)}"
    return issues

def load_training_examples():
    cases=load_json(DATA/"training_examples.json")
    assert len(cases)==144, f"Expected 144 examples, found {len(cases)}"
    return cases

def attachment_path(issue_id,condition):
    if condition not in ("clean","injected"):
        raise ValueError("condition must be clean or injected")
    return DATA/"attachments"/condition/(issue_id+".md")

def load_cached_task1():

    import pandas as pd
    baseline=pd.read_csv(REFERENCE/"task1_baseline.csv")
    gcg=load_json(REFERENCE/"task1_gcg.json")
    tests=baseline[baseline["refused"].astype(str).str.lower()=="true"].iloc[:50].copy()
    assert len(tests)==50
    return baseline, tests, gcg

def load_advbench(path=None):




    import pandas as pd
    path=Path(path) if path else DATA/"advbench_harmful_behaviors.csv"
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        print("Downloading AdvBench to",path)
        try: urllib.request.urlretrieve(ADV_URL,path)
        except Exception as err:
            raise RuntimeError("Could not fetch AdvBench. Download harmful_behaviors.csv from "
              "the official llm-attacks repository and save as data/advbench_harmful_behaviors.csv") from err
    frame=pd.read_csv(path).dropna(subset=["goal","target"]).reset_index(drop=True)
    return frame
