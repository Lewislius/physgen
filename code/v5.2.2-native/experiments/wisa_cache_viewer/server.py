"""Local website for WISA clips, original captions, and both Qwen revisions."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse
import argparse
import hashlib
import json
import mimetypes
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import threading

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
QWEN38 = (BASE / 'experiments/physical_content_v7/data_revision_qwen38/runs/'
          '20261103_dataset')
MANIFEST = QWEN38 / 'encoded/base/manifest.json'
REASONING = QWEN38 / 'reasoning'
DATA_CACHE = Path('/home/liuzhirui/Project/physGen/data_cache/wisa_4000')
FIRST = DATA_CACHE / 'base/first'
OLD_ROOTS = (
    BASE / 'experiments/physical_scaling_q256/supervision_cache/r1/input',
    BASE / 'experiments/physical_variants/supervision_cache/r1/input',
    BASE / 'experiments/physical_v5_scaling_q256/supervision_cache/r1/input',
)
CATALOG = ROOT / 'data/catalog.json'
MEDIA = ROOT / 'media/videos'
STATIC = ROOT / 'static'
_ID = re.compile(r'^[A-Za-z0-9_]+$')
_DETAILS = {}
_DETAIL_LOCK = threading.Lock()
_VIDEO_LOCKS = {}
_VIDEO_LOCK = threading.Lock()
_ROWS = None


def rows():
    global _ROWS
    if _ROWS is None:
        _ROWS = {
            row['id']: row
            for row in json.loads(MANIFEST.read_text())['records']
        }
    return _ROWS


def old_path(sample, arm):
    for root in OLD_ROOTS:
        path = root / f'{sample}_{arm}.pt'
        if path.is_file():
            return path
    return None


def old_evidence(sample, arm):
    path = old_path(sample, arm)
    if path is None:
        return dict(available=False, arm=arm, source=None)
    import torch
    value = torch.load(path, map_location='cpu', weights_only=True)
    return dict(
        available=True,
        arm=arm,
        relations=list(value.get('relations', [])),
        analysis=value.get('analysis'),
        caption=value.get('caption'),
        prompt=value.get('prompt'),
        model=value.get('model'),
        analysis_complete=value.get('analysis_complete'),
        scope=value.get('scope'),
        source=str(path),
    )


def new_evidence(sample, arm):
    path = REASONING / f'{sample}_{arm}.json'
    if not path.is_file():
        return dict(available=False, arm=arm, source=None)
    value = json.loads(path.read_text())
    trace = value.get('trace') or {}
    relations = list(value.get('relations', []))
    return dict(
        available=True,
        arm=arm,
        relations=relations,
        analysis='\n'.join(relations),
        caption=value.get('caption'),
        prompt=trace.get('prompt'),
        model=trace.get('model') or value.get('producer'),
        producer=value.get('producer'),
        raw=trace.get('raw'),
        thinking=trace.get('thinking'),
        explicit_times=trace.get('explicit_times'),
        usage=trace.get('usage'),
        seconds=trace.get('seconds'),
        request_id=trace.get('request_id'),
        finish_reason=trace.get('finish_reason'),
        scope=value.get('scope'),
        revision=value.get('revision'),
        source=str(path),
    )


def clip_meta(row):
    pts = [float(value) for value in row['pts']]
    diffs = [b - a for a, b in zip(pts, pts[1:]) if b > a]
    fps = 1 / statistics.median(diffs) if diffs else 24.0
    return dict(
        start_frame=int(row['indices'][0]),
        end_frame=int(row['indices'][-1]),
        frames=len(row['indices']),
        fps=round(fps, 6),
        duration=round(pts[-1] - pts[0], 6) if len(pts) > 1 else 0.0,
        pts_start=pts[0],
        pts_end=pts[-1],
    )


def sample_detail(sample):
    if not _ID.match(sample):
        raise KeyError(sample)
    row = rows().get(sample)
    if row is None:
        raise KeyError(sample)
    with _DETAIL_LOCK:
        cached = _DETAILS.get(sample)
    if cached is not None:
        return cached
    first = FIRST / f'{sample}.png'
    if not first.is_file():
        first = QWEN38 / 'encoded/base/first' / f'{sample}.png'
    value = dict(
        id=sample,
        split=row['split'],
        manifest_caption=row.get('caption'),
        original_caption=row.get('original_caption'),
        clip=clip_meta(row),
        geometry=row['geometry'],
        source_video=row['video'],
        first_image=f'/media/first/{sample}.png' if first.is_file() else None,
        video=f'/media/video/{sample}.mp4',
        old=dict(
            text=old_evidence(sample, 'text'),
            null=old_evidence(sample, 'null'),
        ),
        new=dict(
            text=new_evidence(sample, 'text'),
            null=new_evidence(sample, 'null'),
        ),
        qwen38=(sample in json.loads((QWEN38 / 'selection.json').read_text())['ids']),
        source={'manifest': str(MANIFEST), 'qwen38_reasoning': str(REASONING)},
    )
    with _DETAIL_LOCK:
        _DETAILS[sample] = value
    return value


def make_video(sample):
    row = rows()[sample]
    MEDIA.mkdir(parents=True, exist_ok=True)
    target = MEDIA / f'{sample}.mp4'
    if target.is_file() and target.stat().st_size > 1024:
        return target
    with _VIDEO_LOCK:
        lock = _VIDEO_LOCKS.setdefault(sample, threading.Lock())
    with lock:
        if target.is_file() and target.stat().st_size > 1024:
            return target
        meta = clip_meta(row)
        geometry = row['geometry']
        source = Path(row['video'])
        if not source.is_file():
            raise FileNotFoundError(str(source))
        filters = (
            f"select='between(n\\,{meta['start_frame']}\\,{meta['end_frame']})',"
            f"scale={geometry['resize_w']}:{geometry['resize_h']}:flags=bilinear,"
            f"crop={geometry['w']}:{geometry['h']}:{geometry['crop_left']}:"
            f"{geometry['crop_top']},setpts=N/({meta['fps']}*TB)"
        )
        handle, temporary = tempfile.mkstemp(
            prefix=f'.{sample}.', suffix='.mp4', dir=MEDIA)
        os.close(handle)
        command = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-i', str(source), '-vf', filters, '-frames:v', str(meta['frames']),
            '-r', str(meta['fps']), '-an', '-c:v', 'libx264',
            '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', temporary,
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
            if result.returncode:
                raise RuntimeError(result.stderr[-2000:])
            if not Path(temporary).is_file() or Path(temporary).stat().st_size < 1024:
                raise RuntimeError('ffmpeg did not produce a video')
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return target


def send_json(handler, value, status=200):
    body = json.dumps(value, ensure_ascii=False).encode()
    handler.send_response(status)
    handler.send_header('Content-Type', 'application/json; charset=utf-8')
    handler.send_header('Content-Length', str(len(body)))
    handler.send_header('Cache-Control', 'no-store')
    handler.end_headers()
    handler.wfile.write(body)


def send_error_json(handler, status, message):
    send_json(handler, dict(error=message), status=status)


def send_file(handler, path, content_type=None, head=False):
    path = Path(path)
    if not path.is_file():
        send_error_json(handler, 404, 'file not found')
        return
    size = path.stat().st_size
    start, end, status = 0, size - 1, 200
    header = handler.headers.get('Range')
    if header and header.startswith('bytes='):
        try:
            left, right = header[6:].split('-', 1)
            if left:
                start = int(left)
            if right:
                end = int(right)
            start = max(0, start)
            end = min(size - 1, end)
            status = 206
            if start > end:
                handler.send_response(416)
                handler.send_header('Content-Range', f'bytes */{size}')
                handler.end_headers()
                return
        except Exception:
            status = 200
    length = end - start + 1
    handler.send_response(status)
    handler.send_header('Content-Type', content_type or mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
    handler.send_header('Accept-Ranges', 'bytes')
    handler.send_header('Content-Length', str(length))
    if status == 206:
        handler.send_header('Content-Range', f'bytes {start}-{end}/{size}')
    handler.send_header('Cache-Control', 'public, max-age=3600')
    handler.end_headers()
    if head:
        return
    with path.open('rb') as stream:
        stream.seek(start)
        remaining = length
        while remaining:
            block = stream.read(min(1 << 20, remaining))
            if not block:
                break
            handler.wfile.write(block)
            remaining -= len(block)


class Handler(BaseHTTPRequestHandler):
    server_version = 'WISAViewer/1.0'

    def log_message(self, fmt, *args):
        print(f'{self.address_string()} {fmt % args}', flush=True)

    def do_HEAD(self):
        self.route(head=True)

    def do_GET(self):
        self.route(head=False)

    def route(self, head=False):
        path = unquote(urlparse(self.path).path)
        try:
            if path == '/health':
                send_json(self, dict(ok=True, catalog=CATALOG.is_file()))
                return
            if path == '/api/catalog':
                send_json(self, json.loads(CATALOG.read_text()))
                return
            if path.startswith('/api/sample/'):
                sample = path.rsplit('/', 1)[-1]
                send_json(self, sample_detail(sample))
                return
            if path.startswith('/media/first/'):
                sample = path.rsplit('/', 1)[-1].removesuffix('.png')
                if not _ID.match(sample):
                    send_error_json(self, 400, 'bad sample')
                    return
                first = FIRST / f'{sample}.png'
                if not first.is_file():
                    first = QWEN38 / 'encoded/base/first' / f'{sample}.png'
                send_file(self, first, 'image/png', head=head)
                return
            if path.startswith('/media/video/'):
                sample = path.rsplit('/', 1)[-1].removesuffix('.mp4')
                if not _ID.match(sample) or sample not in rows():
                    send_error_json(self, 400, 'bad sample')
                    return
                send_file(self, make_video(sample), 'video/mp4', head=head)
                return
            if path == '/':
                send_file(self, STATIC / 'index.html', 'text/html; charset=utf-8', head=head)
                return
            if path.startswith('/static/'):
                relative = Path(path[len('/static/'):])
                if '..' in relative.parts:
                    send_error_json(self, 400, 'bad path')
                    return
                send_file(self, STATIC / relative, head=head)
                return
            send_error_json(self, 404, 'not found')
        except KeyError:
            send_error_json(self, 404, 'sample not found')
        except FileNotFoundError as error:
            send_error_json(self, 404, str(error))
        except Exception as error:
            self.log_message('request failed: %r', error)
            send_error_json(self, 500, repr(error))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--rebuild-catalog', action='store_true')
    args = parser.parse_args(argv)
    if args.rebuild_catalog or not CATALOG.is_file():
        import build_catalog
        build_catalog.main()
    if not CATALOG.is_file():
        raise FileNotFoundError(CATALOG)
    if not shutil.which('ffmpeg'):
        raise RuntimeError('ffmpeg is required for on-demand cropped clips')
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'WISA viewer: http://{args.host}:{args.port}/', flush=True)
    print(f'catalog: {CATALOG}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
