import csv, numpy as np, mujoco
XML='/home/joaobornelli/Documents/g1_training/.venv/lib/python3.12/site-packages/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml'
m=mujoco.MjModel.from_xml_path(XML)
jn=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(m.njnt)]
HN=[jn[j] for j in range(1,m.njnt)]
lo=np.array([m.jnt_range[j,0] for j in range(1,m.njnt)]); hi=np.array([m.jnt_range[j,1] for j in range(1,m.njnt)])
centro=(hi+lo)/2; meia=(hi-lo)/2
def carrega(p):
    r=list(csv.DictReader(open(p)))
    qs=[c for c in r[0] if c.startswith('q_')]; nomes=[c[2:] for c in qs]
    Q=np.array([[float(x[c]) for c in qs] for x in r])
    return r, Q[:,[nomes.index(n) for n in HN]]
for rot,p in (('13999','/home/joaobornelli/Downloads/carrega085.csv'),
              ('15200 laje 0,05','/home/joaobornelli/Documents/g1_training/botar_15200_ckp.csv'),
              ('15200 laje 0,25','/home/joaobornelli/Documents/g1_training/botar_15200_025.csv'),
              ('15200 laje 0,55','/home/joaobornelli/Documents/g1_training/botar_15200_055.csv')):
    r,Q=carrega(p); fases=[x['fase'] for x in r]
    F=np.abs(Q-centro)/meia
    print('='*70); print(rot)
    print(f"  {'fase':9} {'max':>6}  {'junta do pico':26} {'>1,00':>7} {'>0,90':>7}")
    for fase in ('espera','pegar','botar','andar'):
        mm=[i for i in range(len(r)) if fases[i]==fase]
        if not mm: continue
        sub=F[mm]; mx=sub.max(); j=int(np.unravel_index(sub.argmax(),sub.shape)[1])
        f100=100*(sub.max(1)>1.00).mean(); f90=100*(sub.max(1)>0.90).mean()
        print(f'  {fase:9} {mx:6.3f}  {HN[j]:26} {f100:6.1f}% {f90:6.1f}%')
    # top 5 juntas por p99 no arquivo inteiro
    p99=np.percentile(F,99,axis=0)
    ordem=np.argsort(-p99)[:5]
    print('  top 5 juntas por p99 (arquivo inteiro):')
    for j in ordem:
        print(f'     {HN[j]:26} p99={p99[j]:.3f}  p50={np.median(F[:,j]):.3f}  tempo>1,00={100*(F[:,j]>1).mean():5.1f}%')
