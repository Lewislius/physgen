"""Fetch official, evaluation-only VBench assets without changing any environment."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parent / 'assets'
ASSETS = {
    'clip/ViT-B-32.pt': 'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt',
    'clip/ViT-L-14.pt': 'https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt',
    'aesthetic/sa_0_4_vit_l_14_linear.pth': 'https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth',
    'dino/dino_vitbase16_pretrain.pth': 'https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth',
    'dino/vision_transformer.py': 'https://raw.githubusercontent.com/facebookresearch/dino/main/vision_transformer.py',
    'dino/utils.py': 'https://raw.githubusercontent.com/facebookresearch/dino/main/utils.py',
}


def fetch(item):
    name, url = item
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        tmp = path.with_suffix(path.suffix + '.partial')
        with urllib.request.urlopen(url, timeout=60) as response, tmp.open('wb') as out:
            shutil.copyfileobj(response, out, length=8 * 1024 * 1024)
        os.replace(tmp, path)
    record = dict(path=str(path), source=url, bytes=path.stat().st_size)
    print(json.dumps(record), flush=True)
    return record


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(fetch, ASSETS.items()))
    (ROOT / 'sources.json').write_text(json.dumps(records, indent=2))
