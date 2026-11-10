"""Submit and monitor four single-slot jobs; retry only rejected GPU allocations."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference.no_verifier_jobs import JOBS, MASTER, RECORD, DOCUMENT, DEFAULT_JOBS
from tools.gpu_policy import is_rtx_pro_6000

TERMINAL = {'COMPLETED', 'ERROR', 'CANCELED', 'DELETED'}


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def det(*arguments):
    env = dict(os.environ)
    env['NO_PROXY'] = ','.join(filter(None, [env.get('NO_PROXY', ''), '10.130.130.5,localhost,127.0.0.1']))
    result = subprocess.run(['det', '-m', MASTER, *map(str, arguments)], env=env,
                            capture_output=True, text=True, timeout=45)
    if result.returncode:
        raise RuntimeError(f'det {arguments[0]} {arguments[1]} failed (exit {result.returncode})')
    return result.stdout


def gpu_policy_event(log):
    for line in log.splitlines():
        position = line.find('{"kind": "gpu_policy"')
        if position >= 0:
            return json.loads(line[position:])
    return None


def free_rtx_slots():
    text = det('slot', 'list', '--csv')
    # The old CLI can emit a version warning before CSV.
    text = text[text.index('Agent ID,'):]
    return sum(row['Resource Pool'] == 'blk-96g' and row['Allocation ID'] == 'FREE' and
               row['Enabled'] == 'True' and row['Draining'] == 'False' and is_rtx_pro_6000(row['Device'])
               for row in csv.DictReader(io.StringIO(text)))


def read_summary(job, attempt):
    path = Path(job['output']) / 'summary.json'
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    if str(summary.get('experiment_id')) != str(attempt['experiment_id']):
        return None
    job.update(completed_cases=summary['completed_cases'], generation_status=summary['status'],
               gpu=summary.get('device'), summary=str(path))
    if summary.get('cases'):
        done = [case for case in summary['cases'] if case['status'] == 'completed']
        if done:
            job['first_verified_video'] = done[0]['output']
            job['first_verified_encoding'] = done[0]['encoded_video']
    if 'error' in summary:
        job['error'] = summary['error']
    return summary


def refresh(job):
    attempt = job['attempts'][-1]
    payload = json.loads(det('experiment', 'describe', attempt['experiment_id'], '--json'))[0]['experiment']
    # Never persist the full Determined config: it can contain registry credentials.
    state = payload['state'].removeprefix('STATE_')
    attempt.update(state=state, trial_ids=payload['trialIds'], ended_at=payload.get('endTime'))
    summary = read_summary(job, attempt)
    if 'gpu_policy' not in attempt and payload['trialIds']:
        event = gpu_policy_event(det('trial', 'logs', payload['trialIds'][0], '--head', '80'))
        if event is None and state in TERMINAL:
            event = gpu_policy_event(det('trial', 'logs', payload['trialIds'][0], '--tail', '80'))
        if event is None and summary and summary.get('device'):
            event = dict(kind='gpu_policy', actual=summary['device'],
                         accepted=is_rtx_pro_6000(summary['device']), source='worker summary')
        if event is not None:
            attempt['gpu_policy'] = event
            job['gpu'] = event['actual']
    rejected = attempt.get('gpu_policy', {}).get('accepted') is False
    if state in TERMINAL:
        if rejected and state == 'ERROR':
            job['status'] = 'waiting_rtx'
        elif state == 'COMPLETED' and summary and summary['status'] == 'completed' and summary['completed_cases'] == job['total_cases']:
            job['status'] = 'completed'
        else:
            job['status'] = 'failed'
            job.setdefault('error', f'Determined {state}; inspect trial logs before retrying')
    else:
        job['status'] = 'running' if summary else 'scheduled'


def submit(job, record):
    output = det('experiment', 'create', job['yaml'])
    match = re.search(r'Created experiment\s+(\d+)', output, re.I)
    if not match:
        raise RuntimeError('Submission response has no experiment ID; inspect Determined before resubmitting')
    experiment = int(match.group(1))
    job['attempts'].append(dict(experiment_id=experiment, state='SUBMITTED', submitted_at=now()))
    job.update(status='scheduled', experiment_id=experiment)
    job.pop('gpu', None)
    save_json(RECORD, record)
    print(f'{now()} {job["key"]}: submitted experiment {experiment}', flush=True)


def write_document(record):
    labels = {'pending': '待提交', 'scheduled': '已提交，等待启动', 'running': '生成中',
              'waiting_rtx': '已拒绝非 RTX 卡，等待 RTX 空闲后自动重提', 'completed': '全部完成',
              'deferred': '按用户更新暂缓，已撤回本次提交', 'failed': '失败，未自动掩盖错误'}
    q_video = record['jobs'].get('q_videophy40', {})
    arrangement = (
        '**最新提交安排：Q VideoPhy2-40 已获用户授权继续提交，复用训练 66443 的最终权重和原有 40 例输入。'
        '此前优先安排的 H/Q demo21 与 H VideoPhy2-40 的状态见下表。**'
        if q_video and q_video.get('status') != 'deferred' else
        '**最新提交安排：先执行 3 个任务，优先 H demo21、Q demo21，第三个为 H VideoPhy2-40。'
        'Q VideoPhy2-40 已撤回，本轮不会自动重提；对应代码、SH 和 YAML 保留。**')
    lines = ['# 无 Verifier 的 H/Q 最终权重推理记录', '', f'更新时间（UTC）：{record["updated_at"]}', '',
        'H 来自训练实验 **66438 / v29**；Q 来自 **66443 / v30**。两者均使用最终 **224 步** checkpoint，'
        '训练预算为 64 步 interface + 160 步 joint。', '',
        arrangement, '',
        '推理使用训练代码中的 `PhysGenNative`、`restore`、`sample` 和 `export_video`，精确恢复 message、'
        'C 投影、alignment 与 Wan LoRA。H/Q 均执行两轮 message refinement；Q 另保留原有 R0 初始 grounding 和 128 个槽位。'
        '不构造 verifier，不启用评分、校准、反馈反传或 JEPA，不重新校准已保存的 alignment。', '',
        '生成设置：832×480、81 帧、24 FPS、50 步 shifted Euler、shift=5、CFG=5、condition_strength=1。'
        'demo21 使用训练 seed=42；VideoPhy2 使用原有每例 seed 和持久化 FP32 初始噪声，H/Q 共用同一组输入与噪声。', '',
        '测试清单：demo21 = reference + P01–P20；VideoPhy2 = 既有固定 VP001–VP040，共 40 例。'
        'VideoPhy2 是本项目的共享首帧 I2V 测试，不是官方完整 T2V 榜单。这里只提交视频生成；独立 SA/PC 打分未在本次自动运行。', '',
        '- demo 输入：`../v5.2/inference_outputs/1d-v52-lora-demo21_20261010/`（相对于项目目录）。',
        '- VideoPhy2 输入：`../v5.2/inference_outputs/videophy2_lora_run64543_joint400_60_seed42/`。',
        '- 固定 40 例清单：`analysis/videophy40_20261026/case_ids.json`。',
        '- 只读取 `first/text/null_text/times/valid`、首帧与 caption 派生的 requirements，以及保存的随机噪声；不读取历史未来视频或 JEPA/task tokens。', '',
        '资源池固定 **blk-96g**，每个实验 **slots_per_trial: 1**。在读取 checkpoint、输入和模型前检查实际 GPU，'
        '只接受 RTX PRO 6000 / RTX 6000 Blackwell；H100 或其他 GPU 立即退出。监控器只对该型号拒绝重提，'
        '并在 RTX 忙碌时等待空闲，保留全部尝试 ID。未修改共享调度、GPU slot 或其他任务。', '',
        '运行前检查最终权重版本/步数、输入尺寸和有限数值；每个生成结果检查实际推理轮数、视频可完整解码、尺寸/帧数/FPS。'
        '没有新增哈希或 SHA256 检查。通过这些检查表示生成链路可运行，不代表物理效果已经评测达标。', '',
        '提交前已完成 4 组真实权重与输入预检（21/40/21/40 例），以及 11 项推理回归测试；'
        '覆盖 H100 在 checkpoint 打开前退出、旧固定 judge 权重拒绝、两种模式的共享噪声精确重放、'
        '输入隔离、两轮推理和只重提型号不符的失败。记录见 `analysis/no_verifier_inference_20261027/validation.json`。', '',
        '## 推理代码与提交入口', '',
        '- 专用入口：[infer_no_verifier_benchmarks.py](../inference/infer_no_verifier_benchmarks.py)。',
        '- 共用生成实现：[infer_message_demo21.py](../inference/infer_message_demo21.py)。',
        '- 权重与输出映射：[no_verifier_jobs.py](../inference/no_verifier_jobs.py)。',
        '- 自动提交/重提/记录：[submit_no_verifier_benchmarks.py](../inference/submit_no_verifier_benchmarks.py)。',
        '- 默认先提交 3 组：`bash inference/submit_no_verifier_first3_blk96g.sh`；单组可运行对应 SH 的 `--check` 或 `--submit`。',
        '- 单独补交 Q 的 40 例：`bash inference/infer_q_videophy40_no_verifier_blk96g.sh --submit`。',
        '- 相同输出目录重启会核对 manifest、复用已验证视频并继续；不覆盖已完成的历史版本。', '',
        '目录名中的 `66438` / `66443` 是训练 checkpoint 的来源实验；实际推理 Experiment 和 Trial 在下表及每段视频旁 JSON 中单独记录。', '',
        '## 实验与结果', '',
        '| 任务 | 训练实验 | 当前推理实验 / trial | 已验证视频 | 状态 | 实际 GPU |',
        '|---|---:|---|---:|---|---|']
    for job in record['jobs'].values():
        attempt = job.get('attempts', [])[-1] if job.get('attempts') else {}
        eid = attempt.get('experiment_id')
        link = f'[{eid}]({MASTER}/det/experiments/{eid})' if eid else '—'
        trials = ','.join(map(str, attempt.get('trial_ids', []))) or '—'
        lines.append(f'| {job["variant"]} {job["suite"]} | {job["training_experiment_id"]} | {link} / {trials} | '
                     f'{job.get("completed_cases", 0)}/{job["total_cases"]} | {labels.get(job["status"], job["status"])} | {job.get("gpu", "待分配")} |')
    for job in record['jobs'].values():
        lines += ['', f'### {job["variant"]} / {job["suite"]}', '',
                  f'- SH：[{Path(job["sh"]).name}](../inference/{Path(job["sh"]).name})。',
                  f'- YAML：[{Path(job["yaml"]).name}](../inference/{Path(job["yaml"]).name})。',
                  f'- 最终训练权重：`{job["checkpoint"]}`。',
                  f'- 输出目录：`{job["output"]}`。',
                  '- 视频位置：`reference/i2v/seed42.mp4`、`P01/i2v/seed42.mp4` … `P20/i2v/seed42.mp4`。'
                  if job['suite'] == 'demo21' else '- 视频位置：`cases/VP001/planned.mp4` … `cases/VP040/planned.mp4`。',
                  '- 目录内 `checkpoint.pt` 是最终权重的硬链接，`manifest.json` 记录输入/配置，`summary.json` 记录生成进度；每段视频有同名 JSON 元数据。']
        if job.get('first_verified_video'):
            lines.append(f'- 已通过完整解码检查的首条视频：`{job["first_verified_video"]}`。')
        if job.get('error'):
            lines.append(f'- 错误：`{job["error"]}`。')
        for attempt in job.get('attempts', []):
            gpu = attempt.get('gpu_policy', {})
            lines.append(f'- 尝试 {attempt["experiment_id"]}：{attempt["state"]}；'
                         f'GPU={gpu.get("actual", "待确认")}；GPU 放行={gpu.get("accepted", "待确认")}。')
    lines += ['', '## 机器记录与监控', '',
              f'- 完整尝试与进度：[jobs.json](../{RECORD.relative_to(ROOT)})。',
              '- 监控日志：`analysis/no_verifier_inference_20261027/controller.log`。',
              '- 本文由监控器每 30 秒更新；最终是否完成以本次实验状态及各目录 `summary.json` 为准。', '']
    DOCUMENT.parent.mkdir(parents=True, exist_ok=True)
    temp = DOCUMENT.with_suffix('.md.tmp')
    temp.write_text('\n'.join(lines))
    temp.replace(DOCUMENT)


def run(selected, watch):
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    with (RECORD.parent / 'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = json.loads(RECORD.read_text()) if RECORD.exists() else dict(created_at=now(), jobs={})
        for key in selected:
            record['jobs'].setdefault(key, dict(JOBS[key], status='pending', attempts=[]))
            if record['jobs'][key]['status'] == 'deferred':
                record['jobs'][key]['status'] = 'pending'
                record['jobs'][key]['resubmit_authorized_at'] = now()
        record['requested_jobs'] = selected
        while True:
            if (RECORD.parent / 'controller.stop').exists():
                print('Stopped by controller.stop; submitted experiments are unchanged.', flush=True)
                return
            for key in selected:
                job = record['jobs'][key]
                if job['status'] in ('completed', 'failed'):
                    continue
                try:
                    if job['attempts'] and job['status'] != 'pending':
                        refresh(job)
                    if job['status'] == 'pending' or (job['status'] == 'waiting_rtx' and free_rtx_slots() > 0):
                        try:
                            submit(job, record)
                        except Exception as exc:
                            # Do not repeat an ambiguous create response and risk duplicates.
                            job.update(status='failed', error=f'Submission needs inspection: {exc}')
                    record.pop('last_query_error', None)
                except Exception as exc:
                    record['last_query_error'] = f'{now()} {key}: {type(exc).__name__}: {exc}'
                    print(record['last_query_error'], flush=True)
            record['updated_at'] = now()
            save_json(RECORD, record)
            write_document(record)
            if not watch or all(record['jobs'][key]['status'] in ('completed', 'failed') for key in selected):
                break
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', nargs='+', choices=JOBS, default=DEFAULT_JOBS)
    parser.add_argument('--background', action='store_true')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    if args.background:
        RECORD.parent.mkdir(parents=True, exist_ok=True)
        log = RECORD.parent / 'controller.log'
        with log.open('a') as stream:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--watch', '--jobs', *args.jobs],
                                     stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True, cwd=ROOT)
        (RECORD.parent / 'controller.pid').write_text(str(child.pid) + '\n')
        print(json.dumps(dict(pid=child.pid, log=str(log), record=str(RECORD), document=str(DOCUMENT))))
    else:
        run(args.jobs, args.watch)


if __name__ == '__main__':
    main()
