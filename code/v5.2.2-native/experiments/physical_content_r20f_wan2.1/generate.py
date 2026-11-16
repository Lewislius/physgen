"""Fixed-codec video writer for the Wan2.1 runner.

The formal Wan2.1 protocol uses its native 16 fps presentation rate.  Wan2.2
candidates remain on their 24 fps protocol; this writer is only used by this
Wan2.1 experiment.
"""
from pathlib import Path


def write_video(frames, path, fps=16.0):
    import imageio.v2 as imageio
    path = Path(path)
    temporary = path.with_suffix('.partial.mp4')
    try:
        with imageio.get_writer(
                str(temporary), fps=float(fps), codec='libx264',
                pixelformat='yuv420p', macro_block_size=None,
                ffmpeg_params=['-crf', '18', '-profile:v', 'baseline']) as stream:
            for frame in frames:
                stream.append_data(frame)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
