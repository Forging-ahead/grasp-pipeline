#!/usr/bin/env python3
"""Wait for every new trial, then package and independently audit this run."""
import argparse,json,subprocess,sys,time,zipfile
from pathlib import Path
from watch import snapshot

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();root=a.root.resolve();tools=Path(__file__).parent
    while True:
        state=snapshot(root);print(json.dumps(state),flush=True)
        if state['complete']:break
        time.sleep(45)
    subprocess.run([sys.executable,str(tools/'package_results.py'),str(root)],check=True)
    subprocess.run([sys.executable,str(tools/'validate_delivery.py'),str(root/'delivery')],check=True)
    archive=root.parent/'LIBERO130_libero_20261113_results.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for path in sorted((root/'delivery').rglob('*')):
            if path.is_file():z.write(path,path.relative_to(root/'delivery'))
    (root/'DELIVERY_COMPLETE.json').write_text(json.dumps({'complete':True,'archive':str(archive),'local_qa':json.loads((root/'delivery/LOCAL_QA.json').read_text())},indent=2))
    print('DELIVERY_COMPLETE',flush=True)
