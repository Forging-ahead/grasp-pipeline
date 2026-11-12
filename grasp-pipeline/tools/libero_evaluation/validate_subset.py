#!/usr/bin/env python3
"""Independent standard-library audit of a portable subset delivery."""
import argparse
import ast
import base64
import collections
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import struct
import tarfile
import xml.etree.ElementTree as ET
import zipfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path):
    def bad(value):raise ValueError('Non-finite JSON: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=bad)


def same(left,right):
    if isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:same(left[key],right[key])
    elif isinstance(left,list):
        assert len(left)==len(right)
        for a,b in zip(left,right):same(a,b)
    elif isinstance(left,(int,float)):
        assert math.isfinite(left) and math.isfinite(right) and abs(left-right)<=1e-10
    else:assert left==right


def array_shape(blob):
    assert blob[:6]==b'\x93NUMPY'
    version=blob[6]
    length=struct.unpack('<H' if version==1 else '<I',blob[8:10 if version==1 else 12])[0]
    start=10 if version==1 else 12
    header=ast.literal_eval(blob[start:start+length].decode('utf-8' if version==3 else 'latin1'))
    dtype=header['descr']
    match=re.fullmatch(r'([<>=|])([fiubUS])(\d+)',dtype)
    assert match,dtype
    kind,width=match[2],int(match[3]);item_size=width*(4 if kind=='U' else 1)
    shape=tuple(header['shape']);payload=blob[start+length:]
    assert len(payload)==math.prod(shape)*item_size
    if kind=='f':
        assert width in (4,8)
        endian='>' if match[1]=='>' else '<'
        assert all(math.isfinite(value[0]) for value in struct.iter_unpack(endian+('d' if width==8 else 'f'),payload))
    return shape


class Media(HTMLParser):
    def __init__(self):
        super().__init__();self.videos=[];self.cases=0
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='details' and attrs.get('class')=='case':self.cases+=1
        if tag=='video':
            src=attrs['src'];assert src.startswith('data:video/mp4;base64,')
            raw=base64.b64decode(src.split(',',1)[1],validate=True)
            assert raw[4:8]==b'ftyp'
            self.videos.append(hashlib.sha256(raw).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('delivery',type=Path);a=p.parse_args()
    r=a.delivery.resolve();rows=read(r/'RESULTS.json');s=read(r/'SUMMARY.json');protocol=read(r/'PROTOCOL.json')
    manifest=read(r/'SOURCE_IDENTITY.json');catalog=read(r/'ALL_TASKS.json');before=read(r/'BASELINE_RESULTS.json')
    assert len(rows)==len(catalog)==len(before)==s['tasks']==protocol['trials']
    keys={x['key'] for x in rows};assert len(keys)==len(rows) and keys=={x['key'] for x in catalog}=={x['key'] for x in before}
    old={x['key']:x for x in before};cfgs=read(r/'FROZEN_CONFIGS.json')
    assert manifest['source_digest']==protocol['source_digest']==s['source_digest']
    encoded=json.dumps(manifest['files'],sort_keys=True,separators=(',',':')).encode()
    assert hashlib.sha256(encoded).hexdigest()==manifest['source_digest']
    archive_info=read(r/'FROZEN_SOURCE.json')
    assert archive_info['source_commit']==protocol['revision']==s['source_commit']
    assert sha(r/'frozen-source.tar.gz')==archive_info['archive_sha256']
    with tarfile.open(r/'frozen-source.tar.gz') as source:
        assert source.pax_headers.get('comment')==protocol['revision']
        for name,expected in manifest['files'].items():
            assert hashlib.sha256(source.extractfile('grasp-pipeline/'+name).read()).hexdigest()==expected,name
    suites=list(ET.parse(r/'REGRESSION_FROZEN.xml').getroot().iter('testsuite'))
    assert sum(int(x.get('tests',0)) for x in suites)==146
    assert all(not int(x.get('failures',0)) and not int(x.get('errors',0)) and not int(x.get('skipped',0)) for x in suites)
    source_files={n:h for n,h in manifest['files'].items() if n.startswith('src/graspgen_libero_bridge/') and n.endswith('.py')}
    videos=[]
    for row in rows:
        key=row['key'];case=r/'cases'/key;result=read(case/'RESULT.json');queue=read(case/'QUEUE_RECORD.json')
        assert result==row['result'] and queue==row['queue']
        assert result['key']==key and result['initial_state_id']==queue['initial_state_id']==0
        assert result['source_sha256']==source_files and result['driver_sha256']==manifest['files']['apps/libero/run_task.py']
        assert queue['source_digest']==manifest['source_digest'] and queue['source_unchanged'] and queue['result_identity_valid']
        assert result['original_goals_unchanged'] and result['original_goals']==old[key]['result']['original_goals']
        assert sha(case/'task.bddl')==row['original_bddl_sha256']==old[key]['bddl_sha256']
        assert row['initial_state_file_sha256']==old[key]['initial_state_file_sha256']
        state=read(case/'states.json');initial=read(case/'INITIAL_STATE_BEFORE.json')
        for name in ['q_arm','q_fingers','qvel_arm','qvel_fingers','world_T_eef','world_T_base','object_poses']:same(state[0][name],initial[name])
        config=read(case/'environment/config.json');old_config=read(case/'INITIAL_CONFIG_BEFORE.json')
        ignore={'simulation_gpu','model_gpu','planner_gpu'}
        assert {k:v for k,v in config.items() if k not in ignore}=={k:v for k,v in old_config.items() if k not in ignore}
        expected=dict(cfgs[key])
        for name in ignore:expected[name]=queue['gpu']
        assert all(config.get(k)==v for k,v in expected.items())
        assert config['init_state_id']==config['seed']==0 and config['max_steps']==5000
        n=result['steps'];assert len(state)==n+1
        assert all(abs(b['sim_time']-aa['sim_time']-.05)<=1e-9 for aa,b in zip(state,state[1:]))
        with zipfile.ZipFile(case/'trajectory.npz') as trace:
            for name,shape in [('q_arm',(n+1,7)),('q_fingers',(n+1,2)),('world_T_eef',(n+1,4,4)),('actions',(n,7)),('phases',(n,))]:
                assert array_shape(trace.read(name+'.npy'))==shape,(key,name)
        passed=bool(not queue['watchdog_reached'] and result.get('pipeline_success') and result.get('official_goal_at_end') and n>0)
        assert row['after_success']==queue['pipeline_success']==passed
        assert row['before_success']==old[key]['full_success']
        assert row['continuous_states'] and row['identical_initial_state'] and row['identical_numeric_config']
        assert row['decoded_frames']==n+41
        for prefix in ['before','after']:
            video=r/row[prefix+'_video'];expected=row[prefix+'_video_sha256']
            assert sha(video)==expected
            videos.append(expected)
        assert row['before_video_sha256']==old[key]['video_sha256']
    assert s['after_success']==sum(x['after_success'] for x in rows)
    assert s['before_success']==sum(x['before_success'] for x in rows)
    assert s['improved']==sum(x['after_success'] and not x['before_success'] for x in rows)
    assert s['regressed']==sum(not x['after_success'] and x['before_success'] for x in rows)
    assert s['failed']==len(rows)-s['after_success']
    assert s['failure_counts']==dict(collections.Counter(x['failure']['label'] for x in rows if not x['after_success']))
    files=read(r/'FILES_SHA256.json')
    actual={str(f.relative_to(r)) for f in r.rglob('*') if f.is_file()}
    assert set(files)==actual-{'FILES_SHA256.json','DELIVERY_QA.json'}
    for name,expected in files.items():assert sha(r/name)==expected,name
    report=next(r.glob('*.html'));media=Media();media.feed(report.read_text(encoding='utf-8'))
    assert media.cases==len(rows) and collections.Counter(media.videos)==collections.Counter(videos)
    result=dict(passed=True,tasks=len(rows),before_success=s['before_success'],after_success=s['after_success'],
        embedded_videos=len(media.videos),embedded_videos_match_files=True,source_commit=s['source_commit'],
        same_frozen_source=True,recorded_initial_states_match=True,numeric_configs_match=True,
        original_goals_match=True,continuous_episodes=True,artifact_hashes_checked=len(files))
    (r/'DELIVERY_QA.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
