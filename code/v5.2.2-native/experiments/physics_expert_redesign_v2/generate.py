"""Small video writer used by the inference entry point."""
from pathlib import Path


def write_video(frames, path):
    import imageio.v2 as imageio
    path = Path(path)
    temporary = path.with_suffix('.partial.mp4')
    try:
        with imageio.get_writer(str(temporary), fps=24., codec='libx264',
                                pixelformat='yuv420p', macro_block_size=None,
                                ffmpeg_params=['-crf', '18', '-profile:v', 'baseline']) as stream:
            for frame in frames:
                stream.append_data(frame)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
