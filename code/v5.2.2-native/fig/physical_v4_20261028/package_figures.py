"""Package the five unchanged raster diagrams into a review PDF and file index."""
from pathlib import Path
import json

from PIL import Image


ROOT = Path(__file__).resolve().parent
PAGES = [
    ("01_overview_C", "总览与 C 的形成"),
    ("02_grounding_read", "首帧桥接与当前状态读取"),
    ("03_interaction_time", "物理交互与真实时间"),
    ("04_write_native", "共享关联写回与原生 Wan 合流"),
    ("05_training", "两项损失与渐进训练"),
]


def main():
    images, records = [], []
    for name, title in PAGES:
        path = ROOT / f"{name}.png"
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
            records.append({
                "name": name, "title": title, "png": str(path),
                "width": image.width, "height": image.height,
                "prompt": str(ROOT / f"{name}.prompt.txt"),
                "correction_prompt": str(ROOT / f"{name}.edit.txt"),
            })
    pdf_path = ROOT / "Physical_Block_V4_pipeline_5pages.pdf"
    images[0].save(
        pdf_path, "PDF", save_all=True, append_images=images[1:],
        resolution=144, quality=100, subsampling=0,
        title="Physical Block V4 - dimension flows",
        subject="Five architecture design diagrams; new model not yet implemented",
    )
    for image in images:
        image.close()
    manifest = {
        "status": "design_proposal_not_implemented",
        "method": "built-in image_gen; reference-style generation and targeted edits",
        "reference": str(ROOT.parent / "v522_native_v12_03_C_injection_dimensions_20261026.png"),
        "style_prompt": str(ROOT / "common_style.txt"),
        "dimension_notes": str(ROOT / "DIMENSION_NOTES.txt"),
        "pdf": str(pdf_path), "pages": records,
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"pdf": str(pdf_path), "pages": len(records),
                      "dimensions": [(r["width"], r["height"]) for r in records]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
