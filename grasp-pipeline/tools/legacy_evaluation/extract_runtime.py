"""Materialize the environment on a filesystem that rejects hardlinks."""
import json, os, shutil, tarfile, time
from pathlib import Path

r=Path(__file__).resolve().parent;destination=r/'runtime';start=time.time()
counts=dict(files=0,reused=0,materialized_links=0)
with tarfile.open(r/'runtime-original.tar.gz','r|gz') as tar:
    for m in tar:
        if not (m.name=='openpi/lib64' or m.name.startswith('openpi/lib64/')):continue
        assert not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
        out=destination/m.name
        if m.isdir():out.mkdir(parents=True,exist_ok=True)
        elif m.isreg():
            if out.is_file() and out.stat().st_size==m.size:counts['reused']+=1;continue
            out.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(m) as source,out.open('wb') as target:shutil.copyfileobj(source,target,8*1024*1024)
            os.chmod(out,m.mode);counts['files']+=1
        elif m.islnk():
            target=destination/m.linkname
            assert target.is_file(),m.linkname
            if not out.exists():shutil.copy2(target,out);counts['materialized_links']+=1
        else:raise RuntimeError('Unexpected member type: '+m.name)
    # lib was a symlink to lib64 in the reference environment; preserve the
    # relationship instead of recreating every file as a forbidden hardlink.
lib=destination/'openpi/lib'
assert not lib.exists(),lib
lib.symlink_to('lib64',target_is_directory=True)
bin_dir=destination/'openpi/bin';bin_dir.mkdir(exist_ok=True)
(bin_dir/'python').symlink_to('/opt/venv/.python/cpython-3.11.10-linux-x86_64-gnu/bin/python3.11')
for name in ['python3','python3.11']:(bin_dir/name).symlink_to('python')
(destination/'openpi/pyvenv.cfg').write_text('home = /opt/venv/.python/cpython-3.11.10-linux-x86_64-gnu/bin\ninclude-system-site-packages = false\nversion = 3.11.10\n')
counts['wall_seconds']=time.time()-start
(r/'RUNTIME_EXTRACTION.json').write_text(json.dumps(counts,indent=2));print(json.dumps(counts),flush=True)
