"""Item 1: o alvo do BOTAR (x_base 0,30 a 0,40) exige extensao de cotovelo?
Cinematica direta pura: pelve na origem, sem inclinacao, ombro no default."""
import numpy as np, mujoco
XML='/home/joaobornelli/Documents/g1_training/.venv/lib/python3.12/site-packages/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml'
m=mujoco.MjModel.from_xml_path(XML); d=mujoco.MjData(m)
jn=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(m.njnt)]
hn=[jn[j] for j in range(1,m.njnt)]
SID={n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,n) for n in ('left_palm','right_palm')}
KEY={'_hip_pitch_joint':-0.312,'_knee_joint':0.669,'_ankle_pitch_joint':-0.363,'_elbow_joint':0.6,
     'left_shoulder_roll_joint':0.2,'left_shoulder_pitch_joint':0.2,
     'right_shoulder_roll_joint':-0.2,'right_shoulder_pitch_joint':0.2}
q0=np.zeros(len(hn))
for i,n in enumerate(hn):
    for k,v in KEY.items():
        if n==k or n.endswith(k): q0[i]=v
iE=[hn.index('left_elbow_joint'),hn.index('right_elbow_joint')]
iSP=[hn.index('left_shoulder_pitch_joint'),hn.index('right_shoulder_pitch_joint')]
lim_e=m.jnt_range[1+iE[0]]; lim_sp=m.jnt_range[1+iSP[0]]
print(f'faixa do cotovelo  {lim_e}   ombro_pitch {lim_sp}')
def palma_x(qh):
    d.qpos[:3]=[0,0,0.76]; d.qpos[3:7]=[1,0,0,0]; d.qpos[7:]=qh
    mujoco.mj_kinematics(m,d)
    pm=0.5*(d.site_xpos[SID['left_palm']]+d.site_xpos[SID['right_palm']])
    return pm[0]-d.xpos[1][0], pm[2]
print()
print('Alcance horizontal da palma a partir da PELVE, tronco reto, ombro variando:')
print(f"{'ombro_pitch':>12} {'cotovelo':>9} {'x_base':>8} {'z':>7}")
for sp in (0.2,-0.3,-0.8,-1.3):
    for el in (1.28,0.9,0.6,0.3,0.0):
        q=q0.copy()
        for i in iSP: q[i]=sp
        for i in iE: q[i]=el
        x,z=palma_x(q)
        print(f'{sp:12.2f} {el:9.2f} {x:8.3f} {z:7.3f}')
print()
print('Maximo alcance em x com o cotovelo TOTALMENTE flexionado (1,28 = default do mjlab KEYFRAME antigo):')
melhor={}
for el in (1.28,0.9,0.6,0.3,0.0):
    mx=-9
    for sp in np.linspace(float(lim_sp[0]),float(lim_sp[1]),60):
        q=q0.copy()
        for i in iSP: q[i]=sp
        for i in iE: q[i]=el
        x,_=palma_x(q)
        mx=max(mx,x)
    melhor[el]=mx
    print(f'  cotovelo {el:5.2f} rad  ->  x_base maximo = {mx:6.3f} m')
print()
print('ALVO DO BOTAR: x_base 0,30 a 0,40 (comando.py:1617-1643)')
for el,mx in melhor.items():
    v='ALCANCA 0,40' if mx>=0.40 else ('alcanca 0,30' if mx>=0.30 else 'NAO ALCANCA')
    print(f'  cotovelo {el:5.2f}: {v}')
