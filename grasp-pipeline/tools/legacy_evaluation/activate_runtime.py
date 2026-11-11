"""Add missing CPython3.11 aliases only; do not overwrite the host Python3.10 env."""
import json, os
from pathlib import Path

root=Path(__file__).resolve().parent
runtime=root/'runtime'
aliases={
    Path('/opt/venv/.python/cpython-3.11.10-linux-x86_64-gnu'): runtime/'.python/cpython-3.11.10-linux-x86_64-gnu',
    Path('/opt/venv/openpi/lib/python3.11'): runtime/'openpi/lib/python3.11',
}
for alias,target in aliases.items():
    assert target.is_dir(),target
    alias.parent.mkdir(parents=True,exist_ok=True)
    if alias.is_symlink():
        assert alias.resolve()==target.resolve(),alias
    else:
        assert not alias.exists(), 'Refuse to overwrite: '+str(alias)
        alias.symlink_to(target,target_is_directory=True)
(root/'RUNTIME_ALIASES.json').write_text(json.dumps({str(k):str(v) for k,v in aliases.items()},indent=2))
print('Added two previously absent Python3.11 aliases; host Python3.10 untouched')
