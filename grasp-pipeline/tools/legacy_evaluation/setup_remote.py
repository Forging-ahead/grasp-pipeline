"""Create a new evaluation from the already frozen v10; never copy past runs."""
import collections, datetime, hashlib, json, shutil, subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
original = Path('/mnt/public/changwei/Graspgen/libero-policy-broad-20261111/versions/v10')
destination = root / 'versions/v10'
assert not destination.exists(), 'Refuse to overwrite an evaluation snapshot'
manifest = json.loads((original / 'FROZEN_SOURCE.json').read_text())
assert len(manifest) == 39
for name, sha in manifest.items():
    assert hashlib.sha256((original / name).read_bytes()).hexdigest() == sha, name
destination.mkdir(parents=True)
shutil.copy2(original / 'run_task.py', destination / 'run_task.py')
shutil.copy2(original / 'FROZEN_SOURCE.json', destination / 'FROZEN_SOURCE.json')
shutil.copytree(original / 'bridge', destination / 'bridge',
                ignore=shutil.ignore_patterns('runs', '__pycache__', '.pytest_cache', '*.pyc'))
shutil.copy2(root / 'ALL_TASKS.json', destination / 'ALL_TASKS.json')
for name, sha in manifest.items():
    assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == sha, name
rows = json.loads((root / 'ALL_TASKS.json').read_text())
taxonomy = json.loads((root / 'TASK_TAXONOMY.json').read_text())
assert len(rows) == len(taxonomy) == len({x['key'] for x in rows}) == 130
assert {x['key'] for x in rows} == {x['key'] for x in taxonomy}
assert len({x['category'] for x in taxonomy}) == 16
assert all(x['config'].get('init_state_id', 0) == 0 and x['config'].get('seed', 0) == 0 for x in rows)
external = {}
project = Path('/mnt/public/changwei/Graspgen')
for relative in ('LIBERO/libero/libero', 'graspgen-curobo-collect/third_party/GraspGen', 'setup/deps/curobo/src',
                 'graspgen-curobo-collect/models/checkpoints', 'setup/planner-python-wrapper'):
    base = project / relative
    if not base.exists():
        continue
    files = [base] if base.is_file() else sorted(p for p in base.rglob('*') if p.is_file()
        and '__pycache__' not in p.parts and '.git' not in p.parts
        and (p.suffix in ('.py', '.yaml', '.yml', '.urdf', '.xml', '.pth', '.pt', '.ckpt', '.cu', '.cuh', '.cpp', '.h')
             or 'init_files' in p.parts))
    for p in files:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(block)
        external[str(p)] = h.hexdigest()
(root / 'EXTERNAL_SOURCE_BEFORE.json').write_text(json.dumps(external, indent=2))
protocol = dict(
    evaluation_id=root.name, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_version='v10', source_original=str(original), source_snapshot=str(destination),
    source_manifest_sha256=hashlib.sha256((destination/'FROZEN_SOURCE.json').read_bytes()).hexdigest(),
    driver_sha256=manifest['run_task.py'], frozen_python_files=len(manifest),
    catalog_sha256=hashlib.sha256((root/'ALL_TASKS.json').read_bytes()).hexdigest(),
    taxonomy_sha256=hashlib.sha256((root/'TASK_TAXONOMY.json').read_bytes()).hexdigest(),
    trials=130, trials_per_task=1, init_state_id=0, seed=0, task_order_index=0,
    max_control_steps=5000, control_hz=20,
    wall_budget_counts=dict(collections.Counter(x['wall_budget_s'] for x in rows)),
    wall_budget_rule='Same previous per-task assignment: 96 original single tasks 1200s; 34 additional/composite/other tasks 1800s. Includes model loading, planning, rendering and control.',
    watchdog_grace_s=60, concurrency='Two simulation+GraspGen+cuRobo workers per explicitly available GPU; GPU_SLOTS.json records scheduling.',
    driver_overrides='Unmodified v10 run_task.py, explicit --init 0; all source/config hashes recorded.',
    success_definition='pipeline_success AND official_goal_at_end AND steps > 0; full original BDDL goals unchanged; no historical outcomes imported.',
    language_order_review='The two stack_transport tasks require separate order/whole-stack-carry review; final predicate alone does not certify that sequence.',
    video='One uninterrupted simulator rollout per task, including failures. 20fps and two camera views; H264 report copy adds 2s final-frame hold. Planning wait is not simulated video time.',
    external_fingerprints=len(external),
    gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True),
    server_ssh_port=40596, server_hostname=subprocess.check_output(['hostname'],text=True).strip(),
    runtime='Exact Python3.11.10 environment copied from the original server into this evaluation runtime; original /opt/venv/openpi Python3.10 remains untouched.',
    original_server_partial_trials_excluded=True,
)
(root/'PROTOCOL.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
print(json.dumps(dict(snapshot=str(destination),files=len(manifest),tasks=len(rows),external_files=len(external)),ensure_ascii=False))
