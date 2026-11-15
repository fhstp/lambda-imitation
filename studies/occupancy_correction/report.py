"""Read-only scientific analysis of complete fresh cohorts; no survivor pooling.

Writes new report files only. Never edits the manuscript or historical results.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from .records import read, sha, utc, write
from .study import CORRECTIONS, DEFAULT_STATE, MEASURES


def interval(values):
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=1 or len(values)<2 or not np.isfinite(values).all():
        raise ValueError("a scientific interval requires every finite planned seed")
    mean=float(values.mean())
    half=float(t.ppf(0.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
    return {"mean":mean,"ci95":[mean-half,mean+half],"n_seeds":len(values)}


def collect(state,study,environment,kind):
    expected_seeds=study["learning"][environment]["seeds_per_condition"]
    curves={}
    references={}
    for measure in MEASURES:
        for correction in CORRECTIONS:
            selected=sorted([j for j in study["jobs"] if (j["environment"],j["kind"],j["measure"],j["correction"])==
                (environment,kind,measure,correction)],key=lambda j:j["block"])
            seeds=[];arrays=[];grid=None;provenance=[]
            for job in selected:
                directory=state/"jobs"/job["name"]
                status=read(directory/"status.json")
                if status["status"]!="complete":raise RuntimeError("incomplete/failed planned block: "+job["name"])
                if read(directory/"training-exit.json")["exit_code"]!=0:raise RuntimeError("nonzero learner exit")
                rows=[json.loads(line) for line in (directory/"metrics.jsonl").read_text().splitlines()]
                if kind=="learning":
                    budget=study["learning"][environment]
                    expected=budget["prefill"]+budget["steps_per_round"]*np.arange(0 if budget["eval_initial"] else 1,budget["rounds"]+1)
                    coordinates=np.array([r["env_interactions"] for r in rows])
                    key="eval/greedy/return"
                else:
                    budget=study["frozen"]
                    expected=np.arange(0,budget["updates"]+1,budget["eval_every_updates"])
                    coordinates=np.array([r["updates"] for r in rows])
                    key="fixed_mu_heldout/discrepancy_rms"
                    summary=read(directory/"summary.json")
                    identity=(summary["frozen_reference_sha256"],summary["dataset_sha256"],summary["fixed_encoder_actor_task_hash"])
                    old=references.setdefault(job["block"],identity)
                    if old!=identity:raise RuntimeError("fixed fits do not share the same reference and data")
                np.testing.assert_array_equal(coordinates,expected)
                if grid is not None:np.testing.assert_array_equal(grid,coordinates)
                grid=coordinates
                array=np.array([[r[f"{key}/seed_{i}"] for i in range(len(job["seeds"]))] for r in rows],np.float64)
                if not np.isfinite(array).all():raise FloatingPointError("nonfinite seed results cannot be omitted")
                arrays.append(array);seeds.extend(job["seeds"])
                provenance.append({"job":job["name"],"metrics_sha256":sha(directory/"metrics.jsonl"),
                    "config_sha256":sha(directory/"config.json"),"summary_sha256":sha(directory/"summary.json")})
            if len(seeds)!=expected_seeds or len(set(seeds))!=expected_seeds:raise RuntimeError("wrong independent seed count")
            values=np.concatenate(arrays,axis=1)
            curves[(measure,correction)]={"coordinates":grid,"values":values,"seeds":seeds,"provenance":provenance}
    return curves


def summarize_curves(curves,endpoint):
    summaries={};contrasts={}
    for condition,curve in curves.items():
        locations=np.flatnonzero(curve["coordinates"]==endpoint)
        if len(locations)!=1:raise RuntimeError("predeclared endpoint missing")
        values=curve["values"][locations[0]]
        summaries["/".join(condition)]={**interval(values),"seed_values":values.tolist(),
            "seeds":curve["seeds"],"endpoint":int(endpoint),"provenance":curve["provenance"]}
    for measure in MEASURES:
        for comparator in ("uncorrected","is"):
            a,b=curves[(measure,"retrace")],curves[(measure,comparator)]
            if a["seeds"]!=b["seeds"]:raise RuntimeError("unmatched paired contrast")
            av=summaries[measure+"/retrace"]["seed_values"]
            bv=summaries[measure+"/"+comparator]["seed_values"]
            contrasts[measure+"/retrace-minus-"+comparator]=interval(np.array(av)-np.array(bv))
    for correction in CORRECTIONS:
        a,b=curves[("discounted",correction)],curves[("uniform",correction)]
        if a["seeds"]!=b["seeds"]:raise RuntimeError("unmatched sampling contrast")
        av=summaries["discounted/"+correction]["seed_values"]
        bv=summaries["uniform/"+correction]["seed_values"]
        contrasts[correction+"/discounted-minus-uniform"]=interval(np.array(av)-np.array(bv))
    return {"conditions":summaries,"exploratory_paired_contrasts":contrasts,
            "multiple_testing_adjustment":False,"peak_selection":False}


def plot(curves,path,*,limit,ylabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors={"uncorrected":"#d27c26","is":"#437ab8","retrace":"#26906d"}
    fig,ax=plt.subplots(figsize=(7,4))
    for (measure,correction),curve in curves.items():
        keep=curve["coordinates"]<=limit
        x,y=curve["coordinates"][keep],curve["values"][keep]
        mean=y.mean(1)
        half=t.ppf(.975,y.shape[1]-1)*y.std(1,ddof=1)/np.sqrt(y.shape[1])
        ax.plot(x,mean,color=colors[correction],linestyle="-" if measure=="discounted" else "--",
                label=measure+" / "+correction)
        ax.fill_between(x,mean-half,mean+half,color=colors[correction],alpha=.08)
    ax.set_xlabel("Training interactions (prefill included)" if "learning" in path.name else "Auxiliary fitting updates")
    ax.set_ylabel(ylabel);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(path);plt.close(fig)


def generate(state=DEFAULT_STATE):
    state=Path(state);study=read(state/"study.json")
    directory=state/"report";directory.mkdir(exist_ok=True)
    incomplete=[j["name"] for j in study["jobs"] if not (state/"jobs"/j["name"]/"status.json").exists() or
                read(state/"jobs"/j["name"]/"status.json")["status"]!="complete"]
    if incomplete:
        write(directory/"incomplete.json",{"utc":utc(),"incomplete_or_failed_jobs":incomplete,
            "inferential_summary_produced":False,"survivor_only_reporting_allowed":False})
        raise RuntimeError("study incomplete; refusing survivor-only inferential summaries")
    report={"utc":utc(),"study_sha256":sha(state/"study.json"),"admission_sha256":sha(state/"admission.json"),
        "historical_results_pooled":False,"all_predeclared_seeds_retained":True,"environments":{},
        "scope":{"learning":"adaptive empirical replay; not an exact stationary-policy theorem test",
                 "fixed_fits":"fixed-mu discounted held-out neural disagreement, not exact fixed-point error"}}
    for environment in ("battleship","minesweeper"):
        learning=collect(state,study,environment,"learning")
        frozen=collect(state,study,environment,"fit")
        endpoint=study["learning"][environment]["primary_endpoint"]
        updates=study["frozen"]["updates"]
        report["environments"][environment]={"learning":summarize_curves(learning,endpoint),
                                             "fixed_fits":summarize_curves(frozen,updates)}
        plot(learning,directory/(environment+"-learning.pdf"),limit=endpoint,ylabel="Mean greedy return")
        plot(frozen,directory/(environment+"-fixed-fits.pdf"),limit=updates,
             ylabel="Fixed-mu discounted held-out discrepancy RMS")
    write(directory/"summary.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state",type=Path,default=DEFAULT_STATE)
    args=parser.parse_args();generate(args.state)
    print(args.state/"report/summary.json")


if __name__=="__main__":main()
