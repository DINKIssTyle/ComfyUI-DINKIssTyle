"""Prepare the CC0 MakeHuman data; no MakeHuman program code is used.

python tools/build_human_asset.py /path/to/source-folder
Source files: base.obj, rig.json (default.mhskel), weights.json (default_weights.mhw), LICENSE.ASSETS.md.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(sys.argv[1])
OUT=ROOT/'web/assets/human';OUT.mkdir(parents=True,exist_ok=True)
rig=json.loads((SOURCE/'rig.json').read_text());weights=json.loads((SOURCE/'weights.json').read_text())
assert rig['license']==weights['license']=='CC0'
vertices=[];faces=[];group=''
for line in (SOURCE/'base.obj').read_text().splitlines():
    if line.startswith('v '):vertices.append(list(map(float,line.split()[1:4])))
    elif line.startswith('g '):group=line[2:]
    elif line.startswith('f ') and group=='body':faces.append([int(t.split('/')[0])-1 for t in line.split()[1:]])

def joint(name):
    indices=rig['joints'][name]
    return [sum(vertices[i][axis] for i in indices)/len(indices) for axis in range(3)]

source_bones={name:{'head':joint(b['head']),'tail':joint(b['tail'])} for name,b in rig['bones'].items()}
names=list(json.loads((ROOT/'web/default_scene.json').read_text())['joints'])

def mapped(name):
    side=name.rsplit('.',1)[-1] if '.' in name else None
    prefix=(side+'_') if side in ('L','R') else ''
    if name.startswith('finger'):
        finger,segment=name.split('.')[0][6:].split('-')
        return prefix+['thumb','index','middle','ring','pinky'][int(finger)-1]+'_'+str(int(segment)-1)
    if name.startswith('upperarm'):return prefix+'shoulder'
    if name.startswith('lowerarm'):return prefix+'elbow'
    if name.startswith(('wrist','metacarpal')):return prefix+'wrist'
    if name.startswith(('clavicle','shoulder')):return prefix+'clavicle'
    if name.startswith('upperleg'):return prefix+'hip'
    if name.startswith('lowerleg'):return prefix+'knee'
    if name.startswith('foot'):return prefix+'ankle'
    if name.startswith('toe'):return prefix+'toe'
    if name.startswith('neck'):return 'neck'
    if name in ('root','spine05') or name.startswith('pelvis'):return 'pelvis'
    if name in ('spine03','spine04'):return 'waist'
    if name in ('spine01','spine02') or name.startswith('breast'):return 'chest'
    return 'head'

influences=[{} for _ in vertices]
for name,rows in weights['weights'].items():
    target=mapped(name)
    for index,w in rows:influences[index][target]=influences[index].get(target,0)+w

used=sorted({i for f in faces for i in f});remap={index:i for i,index in enumerate(used)}
skin_indices=[];skin_weights=[]
for index in used:
    items=sorted(influences[index].items(),key=lambda row:-row[1])[:4]
    if not items:raise ValueError(f'Missing authored skin weights for vertex {index}')
    total=sum(w for _,w in items)
    for j in range(4):
        skin_indices.append(names.index(items[j][0]) if j<len(items) else 0)
        skin_weights.append(round(items[j][1]/total,7) if j<len(items) else 0)

anchors={
    'pelvis':{'head':source_bones['spine05']['head'],'tail':source_bones['spine03']['head']},
    'waist':{'head':source_bones['spine03']['head'],'tail':source_bones['spine02']['head']},
    'chest':{'head':source_bones['spine02']['head'],'tail':source_bones['neck01']['head']},
    'neck':{'head':source_bones['neck01']['head'],'tail':source_bones['head']['head']},
    'head':{'head':[0,(8.4913+6.1555)/2,.45],'tail':[0,8.4913,.45]},
}
for side in ('L','R'):
    def segment(start,end):return {'head':source_bones[start+'.'+side]['head'],'tail':source_bones[end+'.'+side]['tail']}
    anchors[side+'_clavicle']=segment('clavicle','shoulder01')
    anchors[side+'_shoulder']=segment('upperarm01','upperarm02')
    anchors[side+'_elbow']=segment('lowerarm01','lowerarm02')
    start=source_bones['wrist.'+side]['head']
    tip=[sum(source_bones[f'finger{i}-3.{side}']['tail'][a] for i in (2,3,4,5))/4 for a in range(3)]
    width=[source_bones['finger5-1.'+side]['head'][a]-source_bones['finger2-1.'+side]['head'][a] for a in range(3)]
    anchors[side+'_wrist']={'head':start,'tail':tip,'across':width}
    anchors[side+'_hip']=segment('upperleg01','upperleg02')
    anchors[side+'_knee']=segment('lowerleg01','lowerleg02')
    anchors[side+'_ankle']=segment('foot','foot')
    toe=[sum(source_bones[f'toe{i}-1.{side}']['head'][a] for i in (1,2,3,4,5))/5 for a in range(3)]
    anchors[side+'_toe']={'head':toe,'tail':[toe[0],toe[1],toe[2]+.5]}
    for finger,name in enumerate(('thumb','index','middle','ring','pinky'),1):
        for segment_index in range(3):anchors[f'{side}_{name}_{segment_index}']={**source_bones[f'finger{finger}-{segment_index+1}.{side}'],'across':width}

data={'version':1,'source':'MakeHuman official base mesh and authored default skin weights','license':'CC0-1.0','joint_names':names,
    'positions':[n for i in used for n in vertices[i]],'quads':[[remap[i] for i in f] for f in faces],
    'skin_indices':skin_indices,'skin_weights':skin_weights,'anchors':anchors}
(OUT/'base.json').write_text(json.dumps(data,separators=(',',':'))+'\n')
(OUT/'LICENSE.CC0.md').write_bytes((SOURCE/'LICENSE.ASSETS.md').read_bytes())
provenance={'repository':'https://github.com/makehumancommunity/makehuman','sources':{name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in ('base.obj','rig.json','weights.json')},'body_vertices':len(used),'body_quads':len(faces)}
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(f'Prepared {len(used)} authored body vertices, {len(faces)} quads, 52 mapped controls.')
