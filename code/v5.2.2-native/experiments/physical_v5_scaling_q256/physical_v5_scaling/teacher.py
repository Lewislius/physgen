"""Original V5 input and semantic targets, with independent cache writes."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import torch
from physical_training.teacher import Teacher as OriginalTeacher


class Teacher(OriginalTeacher):
    def __init__(self,cfg,device='cuda',log=None):
        super().__init__(cfg,device=device,log=log)
        self.readonly_roots=[Path(p)/'r1' for p in cfg['paths']['supervision_cache_readonly']]

    @contextmanager
    def cache(self,kind,name):
        folder=self.root/kind
        folder.mkdir(parents=True,exist_ok=True)
        path=folder/(name+'.pt')
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            candidates=[path,*[p/kind/(name+'.pt') for p in self.readonly_roots]]
            source=next((p for p in candidates if p.is_file()),None)
            value=torch.load(source,map_location='cpu',weights_only=True) if source else None
            yield path,value
