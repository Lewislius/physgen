"""Publish one directly reviewed caption and its matching input-only reasoning.

Preserve prior artifacts. Reuse the image-only branch; it does not depend on the
caption. Run only before local encoding of this selected400 sample.
"""
from pathlib import Path
import argparse
import copy
from datetime import datetime, timezone
import json
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from annotate import CACHE, Client, input_reasoning, save, words, caption_flags


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--correction',type=Path,required=True)
    args=p.parse_args();output=args.output.resolve();folder=args.correction.resolve()
    review=json.loads((folder/'review.json').read_text());sample=review['sample']
    selection=json.loads((output/'selection.json').read_text())
    if selection['scope']!='actual400_training_only' or sample not in selection['ids']:
        raise ValueError('Only the selected400 training inputs may be corrected')
    if (folder/'published.json').exists():
        print((folder/'published.json').read_text());return
    if (output/'encoded/content/input'/f'{sample}_text.pt').exists():
        raise ValueError('This helper must run before local feature encoding')
    caption_path=output/'records'/sample/'caption.json'
    reasoning_path=output/'reasoning'/f'{sample}_text.json'
    for source,name in ((caption_path,'before.caption.json'),(reasoning_path,'before.text.json')):
        if not (folder/name).exists():shutil.copyfile(source,folder/name)
    before=json.loads((folder/'before.caption.json').read_text())
    text=(folder/'corrected_caption.txt').read_text().strip()
    after=copy.deepcopy(before)
    after.update(caption=text,new_words=words(text),length_ratio=words(text)/before['original_words'],
        flags=caption_flags(text,before['original_wisa_caption']),
        review_status='automatic_video_review_followed_by_direct_visual_correction',
        post_review_correction=dict(evidence=review['evidence'],record=str(folder/'review.json'),
            original_automatic_caption=str(folder/'before.caption.json'),
            reason=review['reason'],editor='Codex direct chronological-frame review'))
    save(folder/'after.caption.json',after)
    if not (folder/'after.text.json').exists():
        from PIL import Image
        rows=json.loads((CACHE/'base/manifest.json').read_text())['records']
        row=next(r for r in rows if r['id']==sample)
        first=CACHE/'base/first'/f'{sample}.png'
        with Image.open(first) as image:rgb=image.convert('RGB')
        # The API receives the corrected literal request and initial image only,
        # never these review notes or subsequent video frames.
        value=input_reasoning(rgb,text,row['pts'][80]-row['pts'][0],Client())
        value.update(sample=sample,revision=selection['revision'],arm='text',first_image=str(first))
        save(folder/'after.text.json',value)
    reasoning=json.loads((folder/'after.text.json').read_text())
    if reasoning['caption']!=text:raise ValueError('Replacement reasoning caption mismatch')
    current=json.loads(caption_path.read_text())
    if current['caption'] not in (before['caption'],text):
        raise ValueError('Another correction changed this caption')
    save(caption_path,after);save(reasoning_path,reasoning)
    result=dict(sample=sample,utc=datetime.now(timezone.utc).isoformat(),published=True,
        positive_reasoning_regenerated=True,negative_reasoning_unchanged=True,
        prior_artifacts_preserved=str(folder),local_features_not_yet_encoded=True,
        old_training_cache_changed=False,current_r11a_changed=False,
        corrected_words=words(text),style_length_ratio=after['length_ratio'])
    save(folder/'published.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
