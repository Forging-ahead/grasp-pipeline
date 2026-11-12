#!/usr/bin/env python3
"""Finish video delivery only after every frozen subset task has returned."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('evaluation',type=Path);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--source-archive',type=Path,required=True)
    a=p.parse_args();r=a.evaluation.resolve();tools=Path(__file__).parent
    while not (r/'SUBSET_COMPLETE.json').exists():time.sleep(20)
    protocol=json.loads((r/'PROTOCOL.json').read_text())
    environment=dict(os.environ,PYTHONPATH=str(Path(protocol['source'])/'tools/libero_evaluation'))
    try:
        subprocess.run([sys.executable,str(tools/'package_subset.py'),str(r),'--baseline',str(a.baseline),
            '--source-archive',str(a.source_archive)],env=environment,check=True)
        subprocess.run([sys.executable,str(tools/'validate_subset.py'),str(r/'delivery')],check=True)
        archive=r/'LIBERO_envscreen_contact_39_20261113_results.zip'
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for path in sorted((r/'delivery').rglob('*')):
                if path.is_file():z.write(path,str(path.relative_to(r/'delivery')))
        report=next((r/'delivery').glob('*.html'))
        checks={str(path.relative_to(r)):dict(sha256=sha(path),bytes=path.stat().st_size)
            for path in [report,archive,r/'delivery/SUMMARY.json',r/'delivery/RESULTS.json',r/'delivery/DELIVERY_QA.json',r/'delivery/FILES_SHA256.json']}
        (r/'DELIVERY_CHECKSUMS.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
        qa=json.loads((r/'delivery/DELIVERY_QA.json').read_text())
        (r/'DELIVERY_COMPLETE.json').write_text(json.dumps(dict(passed=True,qa=qa),indent=2)+'\n')
        print('DELIVERY_COMPLETE',flush=True)
    except BaseException as exc:
        (r/'DELIVERY_FAILED.json').write_text(json.dumps(dict(error=repr(exc)),indent=2)+'\n')
        raise


if __name__=='__main__':main()
