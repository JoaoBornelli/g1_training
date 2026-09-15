import csv, numpy as np, mujoco
XML='/home/joaobornelli/Documents/g1_training/.venv/lib/python3.12/site-packages/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml'
m=mujoco.MjModel.from_xml_path(XML); d=mujoco.MjData(m)
SID={n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,n) for n in ('left_foot','right_foot')}
jn=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(m.njnt)]
HN=[jn[j] for j in range(1,m.njnt)]
lo=np.array([m.jnt_range[j,0] for j in range(1,m.njnt)]); hi=np.array([m.jnt_range[j,1] for j in range(1,m.njnt)])
centro=(hi+lo)/2; meia=(hi-lo)/2
BID_T=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'torso_link')
def carrega(p):
    r=list(csv.DictReader(open(p)))
    qs=[c for c in r[0] if c.startswith('q_')]; nomes=[c[2:] for c in qs]
    Q=np.array([[float(x[c]) for c in qs] for x in r])
    return r, Q[:,[nomes.index(n) for n in HN]]
def pelve(q):
    bz=0.76
    for _ in range(60):
        d.qpos[:3]=[0,0,bz]; d.qpos[3:7]=[1,0,0,0]; d.qpos[7:]=q; mujoco.mj_kinematics(m,d)
        bz-=min(d.site_xpos[SID['left_foot']][2],d.site_xpos[SID['right_foot']][2])
    d.qpos[:3]=[0,0,bz]; d.qpos[3:7]=[1,0,0,0]; d.qpos[7:]=q; mujoco.mj_kinematics(m,d)
    return bz, np.degrees(np.arccos(np.clip(d.xmat[BID_T].reshape(3,3)[2,2],-1,1)))
BRACO=[i for i,n in enumerate(HN) if any(s in n for s in ('shoulder','elbow','wrist'))]
PERNA=[i for i,n in enumerate(HN) if any(s in n for s in ('hip','knee','ankle'))]
iw=HN.index('waist_pitch_joint'); ik=[HN.index('left_knee_joint'),HN.index('right_knee_joint')]
iel=[HN.index('left_elbow_joint'),HN.index('right_elbow_joint')]
isp=[HN.index('left_shoulder_pitch_joint'),HN.index('right_shoulder_pitch_joint')]
med=lambda v: float(np.median(v))
CASOS=[('13999 laje ?','/home/joaobornelli/Downloads/carrega085.csv'),
       ('15200 laje 0,05','/home/joaobornelli/Documents/g1_training/botar_15200_ckp.csv'),
       ('15200 laje 0,25','/home/joaobornelli/Documents/g1_training/botar_15200_025.csv'),
       ('15200 laje 0,55','/home/joaobornelli/Documents/g1_training/botar_15200_055.csv')]
print(f"{'caso':17} {'n':>4} {'waist':>7} {'knee':>7} {'pelve':>7} {'tronco':>7} {'cotov E':>8} {'cotov D':>8} {'omb E':>7} {'omb D':>7} {'fracao':>7}")
linhas={}
for rot,p in CASOS:
    r,Q=carrega(p); fases=[x['fase'] for x in r]
    mb=[i for i in range(len(r)) if fases[i]=='botar']
    if not mb: print(f'{rot}: sem fase botar'); continue
    pz=[];tr=[]
    for i in mb[::max(1,len(mb)//25)]:
        a,b=pelve(Q[i]); pz.append(a); tr.append(b)
    frac=(np.abs(Q[mb]-centro)/meia).max()
    print(f'{rot:17} {len(mb):4d} {med(Q[mb,iw]):+7.3f} {med(Q[mb][:,ik].mean(1)):+7.3f} {med(pz):7.3f} {med(tr):7.1f} '
          f'{med(Q[mb,iel[0]]):+8.3f} {med(Q[mb,iel[1]]):+8.3f} {med(Q[mb,isp[0]]):+7.3f} {med(Q[mb,isp[1]]):+7.3f} {frac:7.3f}')
    linhas[rot]=(r,Q,fases,mb)
print()
print('Movimento no BOTAR — RMS de dq/dt, rad/s')
print(f"{'caso':17} {'perna':>8} {'braco':>8} {'cintura':>8}")
for rot,p in CASOS:
    if rot not in linhas: continue
    r,Q,fases,mb=linhas[rot]
    mm=[i for i in mb if i>0]
    dq=(Q[mm]-Q[[i-1 for i in mm]])/0.02
    icin=[i for i,n in enumerate(HN) if 'waist' in n]
    print(f'{rot:17} {np.sqrt((dq[:,PERNA]**2).mean()):8.3f} {np.sqrt((dq[:,BRACO]**2).mean()):8.3f} {np.sqrt((dq[:,icin]**2).mean()):8.3f}')
