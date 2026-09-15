import csv, numpy as np, mujoco
XML='/home/joaobornelli/Documents/g1_training/.venv/lib/python3.12/site-packages/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml'
m=mujoco.MjModel.from_xml_path(XML); d=mujoco.MjData(m)
SID={n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,n) for n in ('left_foot','right_foot')}
jn=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(m.njnt)]
HN=[jn[j] for j in range(1,m.njnt)]
BID_T=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'torso_link')

def carrega(p):
    r=list(csv.DictReader(open(p)))
    qs=[c for c in r[0] if c.startswith('q_')]
    nomes=[c[2:] for c in qs]
    Q=np.array([[float(x[c]) for c in qs] for x in r])
    # reordena para a ordem do modelo
    idx=[nomes.index(n) for n in HN]
    return r, Q[:,idx]

def pelve_e_tronco(q):
    bz=0.76
    for _ in range(60):
        d.qpos[:3]=[0,0,bz]; d.qpos[3:7]=[1,0,0,0]; d.qpos[7:]=q
        mujoco.mj_kinematics(m,d)
        bz-=min(d.site_xpos[SID['left_foot']][2], d.site_xpos[SID['right_foot']][2])
    d.qpos[:3]=[0,0,bz]; d.qpos[3:7]=[1,0,0,0]; d.qpos[7:]=q; mujoco.mj_kinematics(m,d)
    tr=np.degrees(np.arccos(np.clip(d.xmat[BID_T].reshape(3,3)[2,2],-1,1)))
    return bz, tr

GRUPO={'perna':['hip','knee','ankle'],'cintura':['waist'],
       'ombro/cotovelo':['shoulder','elbow'],'punho':['wrist']}
def grupos():
    g={}
    for k,subs in GRUPO.items():
        g[k]=[i for i,n in enumerate(HN) if any(s in n for s in subs)]
    return g
G=grupos()

for rot,path in (('13999 (carrega085)','/home/joaobornelli/Downloads/carrega085.csv'),
                 ('15200 (novo)','/home/joaobornelli/Documents/g1_training/botar_15200_ckp.csv')):
    r,Q=carrega(path)
    elos=[x['elo'] for x in r]; fases=[x['fase'] for x in r]
    print('='*78); print(rot)
    # --- ITEM 3: cintura, joelho, pelve no BOTAR ---
    mb=[i for i in range(len(r)) if elos[i]=='BOTAR' and fases[i]=='botar']
    if not mb: mb=[i for i in range(len(r)) if elos[i]=='BOTAR']
    iw=HN.index('waist_pitch_joint'); ik=[HN.index('left_knee_joint'),HN.index('right_knee_joint')]
    iel=[HN.index('left_elbow_joint'),HN.index('right_elbow_joint')]
    isp=[HN.index('left_shoulder_pitch_joint'),HN.index('right_shoulder_pitch_joint')]
    pz=[];tr=[]
    for i in mb[::max(1,len(mb)//25)]:
        a,b=pelve_e_tronco(Q[i]); pz.append(a); tr.append(b)
    med=lambda v: float(np.median(v))
    print(f'  ITEM 3, fase `botar` ({len(mb)} passos):')
    print(f'     waist_pitch  p50={med(Q[mb,iw]):+.3f}  max={Q[mb,iw].max():+.3f}   (limite +-0,520)')
    print(f'     knee         p50={med(Q[mb][:,ik].mean(1)):+.3f}  max={Q[mb][:,ik].mean(1).max():+.3f}')
    print(f'     pelve z      p50={med(pz):.3f}   (FK, pes planos)')
    print(f'     tronco       p50={med(tr):5.1f} deg')
    print(f'     cotovelo     p50 E={med(Q[mb,iel[0]]):+.3f}  D={med(Q[mb,iel[1]]):+.3f}   (default 0,600)')
    print(f'     ombro_pitch  p50 E={med(Q[mb,isp[0]]):+.3f}  D={med(Q[mb,isp[1]]):+.3f}')
    # cotovelo no PEGAR, para a comparacao "mantem a flexao do PEGAR"
    mp=[i for i in range(len(r)) if fases[i]=='pegar']
    if mp:
        print(f'     cotovelo no PEGAR  p50 E={med(Q[mp,iel[0]]):+.3f}  D={med(Q[mp,iel[1]]):+.3f}'
              f'   -> delta BOTAR-PEGAR  E={med(Q[mb,iel[0]])-med(Q[mp,iel[0]]):+.3f}  D={med(Q[mb,iel[1]])-med(Q[mp,iel[1]]):+.3f}')
    # --- ITEM 4 (proxy): onde o movimento acontece, por grupo e por elo ---
    print(f'  ITEM 4 (proxy por |dq|, NAO e a acao): RMS de dq/dt por grupo, rad/s')
    print(f"     {'fase':10} " + " ".join(f'{k:>15}' for k in G))
    for fase in ('pegar','botar','andar'):
        mm=[i for i in range(1,len(r)) if fases[i]==fase]
        if len(mm)<3: continue
        dq=(Q[mm]-Q[[i-1 for i in mm]])/0.02
        linha=" ".join(f'{np.sqrt((dq[:,ix]**2).mean()):15.3f}' for ix in G.values())
        print(f'     {fase:10} {linha}')
    # fracao do dq2 total que e braco
    for fase in ('pegar','botar'):
        mm=[i for i in range(1,len(r)) if fases[i]==fase]
        if len(mm)<3: continue
        dq2=((Q[mm]-Q[[i-1 for i in mm]])/0.02)**2
        tot=dq2.sum()
        braco=dq2[:,G['ombro/cotovelo']+G['punho']].sum()
        print(f'     fatia do braco em Sigma dq^2, fase {fase}: {100*braco/tot:.1f}%')
