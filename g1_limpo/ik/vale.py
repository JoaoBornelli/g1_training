import csv, numpy as np, mujoco
XML='/home/joaobornelli/Documents/g1_training/.venv/lib/python3.12/site-packages/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml'
m=mujoco.MjModel.from_xml_path(XML); d=mujoco.MjData(m)
jn=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(m.njnt)]
HN=[jn[j] for j in range(1,m.njnt)]
SID={n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,n) for n in ('left_foot','right_foot','left_palm','right_palm')}
def carrega(p):
    r=list(csv.DictReader(open(p)))
    qs=[c for c in r[0] if c.startswith('q_')]; nomes=[c[2:] for c in qs]
    Q=np.array([[float(x[c]) for c in qs] for x in r])
    return r, Q[:,[nomes.index(n) for n in HN]]
def geo(q):
    bz=0.76
    for _ in range(45):
        d.qpos[:3]=[0,0,bz]; d.qpos[3:7]=[1,0,0,0]; d.qpos[7:]=q; mujoco.mj_kinematics(m,d)
        bz-=min(d.site_xpos[SID['left_foot']][2],d.site_xpos[SID['right_foot']][2])
    d.qpos[:3]=[0,0,bz]; d.qpos[3:7]=[1,0,0,0]; d.qpos[7:]=q; mujoco.mj_kinematics(m,d)
    pm=0.5*(d.site_xpos[SID['left_palm']]+d.site_xpos[SID['right_palm']])
    return bz, pm[2], pm[0]-d.xpos[1][0]
RAMPA=lambda z: float(np.clip((z-0.45)/(0.75-0.45),0,1))
CASOS=[('carregar_055 (com caixa)','/home/joaobornelli/Documents/g1_training/carregar_15200_055.csv'),
       ('botar_ckp (andar vazio)','/home/joaobornelli/Documents/g1_training/botar_15200_ckp.csv'),
       ('botar_025 (andar vazio)','/home/joaobornelli/Documents/g1_training/botar_15200_025.csv')]
print('PELVE ao longo do tempo — a marcha oscila ou o agachamento e ESTAVEL?')
print(f"  {'arquivo':26} {'fase':10} {'n':>4} {'pelve p50':>10} {'p10':>7} {'p90':>7} {'desvio':>7} {'rampa p50':>10}")
for rot,p in CASOS:
    try: r,Q=carrega(p)
    except Exception as e: print(rot,e); continue
    fases=[x['fase'] for x in r]
    for f in ('espera','pegar','carregar','andar'):
        mm=[i for i in range(len(r)) if fases[i]==f]
        if len(mm)<20: continue
        pz=np.array([geo(Q[i])[0] for i in mm[::max(1,len(mm)//60)]])
        print(f'  {rot:26} {f:10} {len(mm):4d} {np.median(pz):10.3f} {np.percentile(pz,10):7.3f} '
              f'{np.percentile(pz,90):7.3f} {pz.std():7.3f} {RAMPA(float(np.median(pz))):10.3f}')
print()
print('CAIXA: desvio da palma (proxy do `precise_pos`, sigma fixo 0,18 m)')
r,Q=carrega('/home/joaobornelli/Documents/g1_training/carregar_15200_055.csv')
fases=[x['fase'] for x in r]
for f in ('espera','carregar'):
    mm=[i for i in range(len(r)) if fases[i]==f]
    v=np.array([geo(Q[i]) for i in mm[::max(1,len(mm)//60)]])
    # desvio em relacao a mediana da propria fase, em x_base e z
    dx=v[:,2]-np.median(v[:,2]); dz=v[:,1]-np.median(v[:,1])
    dist=np.sqrt(dx**2+dz**2)
    perda=1-np.exp(-(dist/0.18)**2)
    print(f'  {f:10} desvio p50={np.median(dist):.4f} m  p90={np.percentile(dist,90):.4f} m'
          f'   perda de precise_pos p90 = {np.percentile(perda,90)*100:.1f}%')
