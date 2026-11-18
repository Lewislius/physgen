"""A finished helper suite may be reused only for the exact model and mode."""
from pathlib import Path
import json,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from completed_suite import completion


class CompletionTests(unittest.TestCase):
    def test_exact_provenance_partial_scores_and_missing_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ids=['VP002','VP004'];source='/frozen/source';checkpoint='/own400.pt'
            (root/'manifest.json').write_text(json.dumps(dict(cases=[dict(id=x) for x in ids],
                checkpoint=checkpoint,training_source=source,checkpoint_step=400,mode='no_c',generation_seed=42)))
            (root/'summary.json').write_text(json.dumps(dict(status='completed',completed_cases=2)))
            for case in ids:
                video=root/'cases'/case/'planned.mp4';video.parent.mkdir(parents=True)
                video.write_bytes(b'fixture')
                video.with_suffix('.json').write_text(json.dumps(dict(checkpoint=checkpoint,step=400,
                    mode='no_c',case=case,seed=42,inference_source=source)))
            self.assertEqual(completion(root,checkpoint,source,ids,'no_c'),(True,False))
            (root/'scores.json').write_text(json.dumps(dict(cases=[dict(id=x,sa=3,pc=4) for x in ids])))
            self.assertEqual(completion(root,checkpoint,source,ids,'no_c'),(True,True))
            with self.assertRaises(ValueError):completion(root,'/different.pt',source,ids,'no_c')
            with self.assertRaises(ValueError):completion(root,checkpoint,source,ids,'real')
            video.unlink()
            self.assertEqual(completion(root,checkpoint,source,ids,'no_c'),(False,False))


if __name__=='__main__':unittest.main(verbosity=2)
