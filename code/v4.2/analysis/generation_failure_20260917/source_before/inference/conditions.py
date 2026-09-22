"""Small cache keys for deployable conditions; never read model files to hash them."""
from pathlib import Path

from physgen_v42.runtime import ROOT, digest


def file_stamp(path):
    stat = Path(path).stat()
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def model_stamp(path):
    path = Path(path).resolve()
    return dict(path=str(path), **file_stamp(path))


def encoder_stamps(paths):
    wan = Path(paths["wan_checkpoint"])
    return dict(text=model_stamp(wan / "models_t5_umt5-xxl-enc-bf16.pth"),
                vae=model_stamp(wan / "Wan2.2_VAE.pth"),
                teacher=model_stamp(paths["teacher_checkpoint"]))


def text_cache(plan, prompt):
    return ROOT / "cache/inference_conditions/v1/text" / (digest(dict(
        prompt=prompt, encoder=plan["encoders"]["text"])) + ".pt")


def image_key(case):
    geometry = {k: v for k, v in case["geometry"].items() if k != "frames"}
    return dict(image=case.get("image"), source=case.get("image_file"), geometry=geometry)


def image_cache(plan, case, kind):
    encoder = plan["encoders"]["vae" if kind == "image" else "teacher"]
    return ROOT / "cache/inference_conditions/v1" / kind / (digest(dict(
        input=image_key(case), encoder=encoder)) + ".pt")


def condition_key(plan, case):
    return digest(dict(case=case, text=plan["encoders"]["text"], vae=plan["encoders"]["vae"],
                       negative=plan["negative"]))


def teacher_identity(plan, case, source, phase):
    if phase == "anchors":
        return dict(cache=str(image_cache(plan, case, "anchor")), phase=phase)
    return dict(input_file=file_stamp(source), encoder=plan["encoders"]["teacher"], phase=phase)
