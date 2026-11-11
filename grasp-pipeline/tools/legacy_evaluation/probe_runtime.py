"""Compare the actual packages used by sim/model/planner, not just environment names."""
import hashlib, json, subprocess, sys
from pathlib import Path

root=Path(__file__).resolve().parent
reference='--reference' in sys.argv
sim='/opt/venv/openpi/bin/python' if reference else str(root/'runtime/openpi/bin/python')
roles={
    'simulation':[sim],
    'graspgen':['/mnt/public/changwei/Graspgen/setup/grasp-model/bin/python'],
    'curobo':['/mnt/public/changwei/Graspgen/setup/planner-python-wrapper'],
}
script='''
import hashlib,importlib,json,sys
from pathlib import Path
names=NAMES
info={"python_version":sys.version,"executable":sys.executable,"modules":{}}
for name in names:
 m=importlib.import_module(name);p=Path(m.__file__)
 info["modules"][name]={"version":str(getattr(m,"__version__","unspecified")),"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
print(json.dumps(info))
'''
data={}
for role,command in roles.items():
    names=['numpy','scipy','torch']
    names+=['mujoco','robosuite','cv2','PIL','imageio_ffmpeg'] if role=='simulation' else ['curobo'] if role=='curobo' else ['omegaconf']
    result=subprocess.run(command+['-c',script.replace('NAMES',repr(names))],capture_output=True,text=True,check=True)
    data[role]=json.loads(result.stdout.splitlines()[-1])
if reference:
    (root/'ORIGINAL_RUNTIME.json').write_text(json.dumps(data,indent=2))
else:
    old=json.loads((root/'ORIGINAL_RUNTIME.json').read_text())
    for role,values in data.items():
        assert values['python_version']==old[role]['python_version'],role
        for name,m in values['modules'].items():
            assert m['version']==old[role]['modules'][name]['version'],(role,name,'version')
            assert m['sha256']==old[role]['modules'][name]['sha256'],(role,name,'source')
    data=dict(all_compared_versions_and_module_hashes_match=True,original=old,four_gpu_server=data)
    (root/'RUNTIME_VERSIONS.json').write_text(json.dumps(data,indent=2))
print(json.dumps(dict(reference=reference,roles=list(roles),matched=not reference)))
