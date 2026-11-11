"""Start one owned queue only after environment and GPU checks have passed."""
import hashlib,json,os,subprocess,time
from pathlib import Path

r=Path(__file__).resolve().parent
assert json.loads((r/'RUNTIME_VERSIONS.json').read_text())['all_compared_versions_and_module_hashes_match']
inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
cards={int(line.split(',')[0]):int(line.split(',')[1]) for line in inventory.strip().splitlines()}
assert set(cards)=={0,1,2,3} and all(memory<256 for memory in cards.values()),cards
assert not (r/'PROGRESS.json').exists(),'Refuse a second launch of the same evaluation'
fd=os.open(r/'LAUNCH.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
protocol=json.loads((r/'PROTOCOL.json').read_text())
protocol.update(settle_steps=10,input_mode='rgbd',gpu_worker_slots={'0':2,'1':2,'2':2,'3':2},
    runtime_tmp_directory=str(r/'runtime_tmp'),
    runtime_versions_match_reference=True,
    harness_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in r.glob('*.py')})
(r/'PROTOCOL.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
with (r/'queue.log').open('x') as log:
    p=subprocess.Popen([str(r/'runtime/openpi/bin/python'),'-u',str(r/'run_queue.py')],stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
launch=dict(pid=p.pid,launched_unix=time.time(),server_port=40596,server_hostname=subprocess.check_output(['hostname'],text=True).strip(),
    gpu_inventory=cards,evaluation_id=r.name,source_version='v10',tasks=130,workers=8)
(r/'LAUNCH.json').write_text(json.dumps(launch,indent=2));print(json.dumps(launch))
