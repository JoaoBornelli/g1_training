import csv, sys, numpy as np, mujoco
XML='/home/joaobornelli/Documents/g1_training/.venv/lib/python3.12/site-packages/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml'
m=mujoco.MjModel.from_xml_path(XML); d=mujoco.MjData(m)
SID={n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,n) for n in ('left_foot','right_foot')}
jn=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(m.njnt)]
HN=[jn[j] for j in range(1,m.njnt)]
def carrega(p):
    r=list(csv.DictReader(open(p)))
    qs=[c for c in r[0] if c.startswith('q_')]; nomes=[c[2:] for c in qs]
    Q=np.array([[float(x[c]) for c in qs] for x in r])
    return r, Q[:,[nomes.index(n) for n in HN]]
def pelve(q):
    bz=0.76
    for _ in range(50):
        d.qpos[:3]=[0,0,bz]; d.qpos[3:7]=[1,0,0,0]; d.qpos[7:]=q; mujoco.mj_kinematics(m,d)
        bz-=min(d.site_xpos[SID['left_foot']][2],d.site_xpos[SID['right_foot']][2])
    return bz
ik=[HN.index('left_knee_joint'),HN.index('right_knee_joint')]
ih=[HN.index('left_hip_pitch_joint'),HN.index('right_hip_pitch_joint')]
iw=HN.index('waist_pitch_joint')
med=lambda v: float(np.median(v))
for rot,p in [('13999','/home/joaobornelli/Downloads/carrega085.csv')]+[
     (f'15200 laje {s}', f'/home/joaobornelli/Documents/g1_training/botar_15200_{n}.csv')
     for s,n in (('0,05','ckp'),('0,25','025'),('0,55','055'))]+sys.argv[1:] and []:
    pass
CASOS=[('13999','/home/joaobornelli/Downloads/carrega085.csv'),
       ('15200 laje 0,05','/home/joaobornelli/Documents/g1_training/botar_15200_ckp.csv'),
       ('15200 laje 0,25','/home/joaobornelli/Documents/g1_training/botar_15200_025.csv'),
       ('15200 laje 0,55','/home/joaobornelli/Documents/g1_training/botar_15200_055.csv')]
CASOS+= [(a,b) for a,b in (x.split('=',1) for x in sys.argv[1:])]
print(f"{'caso':18} {'fase':8} {'n':>4} {'pelve z':>8} {'knee':>7} {'hip_p':>7} {'waist':>7}")
for rot,p in CASOS:
    try: r,Q=carrega(p)
    except Exception as e: print(f'{rot}: {e}'); continue
    fases=[x['fase'] for x in r]
    for fase in ('espera','pegar','carregar','botar','andar'):
        mm=[i for i in range(len(r)) if fases[i]==fase]
        if len(mm)<5: continue
        pz=[pelve(Q[i]) for i in mm[::max(1,len(mm)//30)]]
        print(f'{rot:18} {fase:8} {len(mm):4d} {med(pz):8.3f} {med(Q[mm][:,ik].mean(1)):+7.3f} '
              f'{med(Q[mm][:,ih].mean(1)):+7.3f} {med(Q[mm,iw]):+7.3f}')
