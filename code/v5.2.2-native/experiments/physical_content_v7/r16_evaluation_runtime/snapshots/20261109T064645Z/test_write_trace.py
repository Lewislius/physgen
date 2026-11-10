from pathlib import Path
import importlib.util,json,sys,tempfile
sys.path.insert(0,str(Path(__file__).resolve().parent))
from runtime import activate
from trace_physical_write import install_trace
SOURCE=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/native_joint_read_r16a/snapshots/20261109T040426Z')
activate(SOURCE,'amp-48g')
spec=importlib.util.spec_from_file_location('tiny_trace_fixture',SOURCE/'test_request_prior.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
import torch,guidance
case=fixture.JointMemoryTest();case.setUp()
try:
    model=case.model().eval().requires_grad_(False);case.activate(model)
    with torch.no_grad():
        c,g=model.enhancement(case.evidence)
        args=(model,case.noise,case.sample['first'],case.text,torch.zeros_like(case.text),case.sigma,c,c,case.sample['times'],g,g)
        before=guidance.guided_velocity(*args,composition=guidance.LEGACY)
        logfile=case.path/'trace.jsonl';traced=install_trace(model,logfile,sites={'1'})
        after=traced(*args,composition=guidance.LEGACY)
        torch.testing.assert_close(before,after,atol=0,rtol=0)
        rows=[json.loads(x) for x in logfile.read_text().splitlines()]
        assert len(rows)==1 and rows[0]['step']==1 and rows[0]['layer']==2
        assert rows[0]['physical_output_finite']==1
        assert rows[0]['physical_to_native_rms']>=0 and rows[0]['request_prior_write_ratio']>=0
        print(json.dumps(dict(trace_forward_exactly_unchanged=True,rows=len(rows),row=rows[0])))
finally:case.doCleanups()
