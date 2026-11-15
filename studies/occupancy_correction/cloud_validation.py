"""Administrative W&B checkpoint-byte roundtrip; does not launch a learner."""

import base64
import hashlib
from pathlib import Path

import numpy as np
import wandb

from .records import load_owned_pickle, read, save_pickle, sha, utc, write
from .study import DEFAULT_STATE, ENTITY, prepare


def main():
    state=DEFAULT_STATE
    study=prepare(state)
    directory=state/"validation/cloud-canary"
    directory.mkdir(parents=True,exist_ok=True)
    record_path=state/"validation/cloud-roundtrip.json"
    api=wandb.Api(timeout=120)
    existing_projects={p.name for p in api.projects(ENTITY,per_page=100)}
    for project in {j["project"] for j in study["jobs"]}:
        if project not in existing_projects:
            api.create_project(project,ENTITY)
    if record_path.exists():
        record=read(record_path)
        remote=api.artifact(record["artifact"])
        assert remote.state=="COMMITTED" and remote.metadata["sha256"]==record["sha256"]
        print("Existing cloud canary remains COMMITTED",flush=True)
        return
    source=directory/"checkpoint-canary.pkl.gz"
    if not source.exists():
        record=save_pickle(source,{"validation_only":True,"values":np.arange(8192,dtype=np.float32),
                                  "study_sha256":sha(state/"study.json")})
    run=wandb.init(entity=ENTITY,project=study["validation_project"],id="oc-preflight-20261113-v1",
                   resume="allow",name="Occupancy checkpoint byte canary",job_type="validation",
                   config={"production_training":False,"training_resume_allowed":False})
    artifact=wandb.Artifact("oc-checkpoint-canary-20261113-v1",type="validation",
                            metadata={"sha256":sha(source)})
    artifact.add_file(str(source),name=source.name,policy="immutable",skip_cache=True)
    uploaded=run.log_artifact(artifact);uploaded.wait()
    remote=api.artifact(uploaded.qualified_name)
    entry=remote.manifest.entries[source.name]
    with source.open("rb") as stream:md5=base64.b64encode(hashlib.file_digest(stream,"md5").digest()).decode()
    assert remote.state=="COMMITTED" and entry.size==source.stat().st_size and entry.digest==md5
    restored=Path(remote.get_entry(source.name).download(root=str(directory/"restored"),skip_cache=True))
    assert sha(restored)==sha(source)==remote.metadata["sha256"]
    payload=load_owned_pickle(restored,sha(source))
    np.testing.assert_array_equal(payload["values"],np.arange(8192,dtype=np.float32))
    write(record_path,{"passed":True,"artifact":remote.qualified_name,"sha256":sha(source),"md5":md5,
        "bytes":source.stat().st_size,"utc":utc(),"production_training":False},exclusive=True)
    run.finish()
    print("Checkpoint-byte upload/download and new project access verified",flush=True)


if __name__=="__main__":main()
