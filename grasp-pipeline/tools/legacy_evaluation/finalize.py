"""Wait for this evaluation only, then audit and package its 130 new trials."""
import hashlib,json,subprocess,time,zipfile
from pathlib import Path

r=Path(__file__).resolve().parent;python=str(r/'runtime/openpi/bin/python');last=None
while True:
    try:state=json.loads((r/'PROGRESS.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):time.sleep(1);continue
    if state['finished']!=last:
        print(json.dumps(dict(stage='waiting_for_all130',finished=state['finished'],running=len(state['running']))),flush=True)
        last=state['finished']
    if state['complete']:
        assert state['finished']==state['total']==130 and not state['running'] and state['pending']==0
        break
    time.sleep(5)
print('All130 completed; starting report and full video decoding',flush=True)
subprocess.run([python,str(r/'package.py')],check=True)
subprocess.run([python,str(r/'validate_delivery.py'),str(r/'delivery')],check=True)
delivery=r/'delivery'
files={str(p.relative_to(delivery)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
       for p in sorted(delivery.rglob('*')) if p.is_file()}
(delivery/'DELIVERY_MANIFEST.json').write_text(json.dumps(dict(files=files,manifest_excludes_itself=True),ensure_ascii=False,indent=2))
archive=r/'libero-final130-v10-results.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(delivery.rglob('*')):
        if p.is_file():z.write(p,str(p.relative_to(delivery)))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
page=delivery/'LIBERO130任务最终v10复测报告_含全部视频.html'
receipt=dict(evaluation_id=r.name,complete=True,finished_unix=time.time(),
    summary=json.loads((delivery/'SUMMARY.json').read_text()),archive_path=str(archive),
    archive_bytes=archive.stat().st_size,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    html_path=str(page),html_bytes=page.stat().st_size,html_sha256=hashlib.sha256(page.read_bytes()).hexdigest(),
    validated_locally_on_server=True)
(r/'FINAL_COMPLETE.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
print(json.dumps(receipt,ensure_ascii=False),flush=True)
