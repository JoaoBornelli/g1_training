import csv, numpy as np, mujoco
XML='/home/joaobornelli/Documents/g1_training/.venv/lib/python3.12/site-packages/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml'
m=mujoco.MjModel.from_xml_path(XML); d=mujoco.MjData(m)
jn=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(m.njnt)]
HN=[jn[j] for j in range(1,m.njnt)]
lo=np.array([m.jnt_range[j,0] for j in range(1,m.njnt)]); hi=np.array([m.jnt_range[j,1] for j in range(1,m.njnt)])
centro=(hi+lo)/2; meia=(hi-lo)/2
SID={n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,n) for n in ('left_foot','right_foot','left_palm','right_palm')}
M_ROBO=float(m.body_mass.sum())
P='/home/joaobornelli/Documents/g1_training/carregar_15200_055.csv'
r=list(csv.DictReader(open(P)))
qs=[c for c in r[0] if c.startswith('q_')]; nomes=[c[2:] for c in qs]
Q=np.array([[float(x[c]) for c in qs] for x in r])[:,[nomes.index(n) for n in HN]]
F=np.abs(Q-centro)/meia
fases=[x['fase'] for x in r]; elos=[x['elo'] for x in r]
t=[float(x['t']) for x in r]
from collections import Counter
print('n=',len(r),' t_max=',t[-1])
print('elos :',dict(Counter(elos)))
print('fases:',dict(Counter(fases)))
prev=None
for i,x in enumerate(r):
    k=(x['elo'],x['fase'])
    if k!=prev: print(f'   t={t[i]:6.2f}  {x["elo"]:10} {x["fase"]}'); prev=k
ORD=[f for f in ('espera','pegar','carregar','andar','botar') if f in set(fases)]
def geo(q):
    bz=0.76
    for _ in range(50):
        d.qpos[:3]=[0,0,bz]; d.qpos[3:7]=[1,0,0,0]; d.qpos[7:]=q
        mujoco.mj_kinematics(m,d); mujoco.mj_comPos(m,d)
        bz-=min(d.site_xpos[SID['left_foot']][2],d.site_xpos[SID['right_foot']][2])
    d.qpos[:3]=[0,0,bz]; d.qpos[3:7]=[1,0,0,0]; d.qpos[7:]=q
    mujoco.mj_kinematics(m,d); mujoco.mj_comPos(m,d)
    pm=0.5*(d.site_xpos[SID['left_palm']]+d.site_xpos[SID['right_palm']])
    com=d.subtree_com[1]
    mid=0.5*(d.site_xpos[SID['left_foot']]+d.site_xpos[SID['right_foot']])
    return bz, pm[2], pm[0]-d.xpos[1][0], com[0]-mid[0]
ik=[HN.index('left_knee_joint'),HN.index('right_knee_joint')]
ih=[HN.index('left_hip_pitch_joint'),HN.index('right_hip_pitch_joint')]
iw=HN.index('waist_pitch_joint')
print()
print('GEOMETRIA por fase (FK, pes planos, pelve vertical)')
print(f"  {'fase':10} {'n':>4} {'pelve z':>8} {'palma z':>8} {'palma x':>8} {'CoM-pes x':>10} {'joelho':>8} {'hip_p':>8} {'waist':>7}")
for f in ORD:
    mm=[i for i in range(len(r)) if fases[i]==f]
    if len(mm)<5: continue
    v=np.array([geo(Q[i]) for i in mm[::max(1,len(mm)//30)]])
    print(f'  {f:10} {len(mm):4d} {np.median(v[:,0]):8.3f} {np.median(v[:,1]):8.3f} {np.median(v[:,2]):8.3f} '
          f'{np.median(v[:,3]):10.3f} {np.median(Q[mm][:,ik].mean(1)):+8.3f} {np.median(Q[mm][:,ih].mean(1)):+8.3f} {np.median(Q[mm,iw]):+7.3f}')
print()
print('LIMITE DE JUNTA por fase')
print(f"  {'fase':10} {'max':>7}  {'junta do pico':26} {'>0,90':>7} {'>1,00':>7}")
for f in ORD:
    mm=[i for i in range(len(r)) if fases[i]==f]
    if len(mm)<5: continue
    sub=F[mm]; j=int(np.unravel_index(sub.argmax(),sub.shape)[1])
    print(f'  {f:10} {sub.max():7.3f}  {HN[j]:26} {100*(sub.max(1)>0.90).mean():6.1f}% {100*(sub.max(1)>1.00).mean():6.1f}%')
print()
print('p50 do |frac| nas juntas que importam')
alvos=['left_ankle_pitch_joint','right_ankle_pitch_joint','left_ankle_roll_joint',
       'right_ankle_roll_joint','left_knee_joint','left_hip_pitch_joint','waist_pitch_joint']
print(f"  {'junta':26} " + "".join(f'{f:>11}' for f in ORD))
for n in alvos:
    j=HN.index(n); linha=''
    for f in ORD:
        mm=[i for i in range(len(r)) if fases[i]==f]
        linha += f'{np.median(F[mm,j]):11.3f}' if len(mm)>=5 else f'{"-":>11}'
    print(f'  {n:26} {linha}')
