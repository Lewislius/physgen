"""Read only allocation termination metadata; never export auth or task configs."""
import json
from pathlib import Path
from determined.common.api import authentication, bindings

session=authentication.login_with_cache('http://10.130.130.5:4096')
allocations=[
 '66355.630865ea-302b-4e53-a013-b9aea685e58b.1',
 '66357.a78b50ac-8e45-4e5d-9dc2-d61147aadab0.1',
]
result=[]
for identity in allocations:
 try:
  value=bindings.get_GetAllocation(session,allocationId=identity).allocation.to_json()
  result.append(value)
 except Exception as error:
  result.append(dict(allocationId=identity,error=str(error)))
path=Path(__file__).with_name('allocation_exit_evidence.json')
path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
me=bindings.get_GetMe(session).user
experiments=bindings.get_GetExperiments(session,userIds=[me.id],name='v522',
 experimentIdFilter_gte=66329,limit=80,sortBy=bindings.v1GetExperimentsRequestSortBy.ID,
 orderBy=bindings.v1OrderBy.DESC).experiments
latest=[]
for experiment in experiments:
 value=experiment.to_json()
 latest.append({key:value.get(key) for key in ['id','name','state','trialIds','startTime','endTime']})
Path(__file__).with_name('recent_experiments.json').write_text(json.dumps(latest,indent=2)+'\n')
print(json.dumps(latest,indent=2))
