import numpy as np, mujoco
XML='/home/joaobornelli/Documents/g1_training/.venv/lib/python3.12/site-packages/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml'
m=mujoco.MjModel.from_xml_path(XML); d=mujoco.MjData(m)
SID={n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,n) for n in ('left_foot','right_foot','left_palm','right_palm')}
jn=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(m.njnt)]
hn=[jn[j] for j in range(1,m.njnt)]
lo=np.array([m.jnt_range[j,0] for j in range(1,m.njnt)]); hi=np.array([m.jnt_range[j,1] for j in range(1,m.njnt)])
centro=(hi+lo)/2; meia=(hi-lo)/2
KEY={'_hip_pitch_joint':-0.312,'_knee_joint':0.669,'_ankle_pitch_joint':-0.363,'_elbow_joint':0.6,
     'left_shoulder_roll_joint':0.2,'left_shoulder_pitch_joint':0.2,
     'right_shoulder_roll_joint':-0.2,'right_shoulder_pitch_joint':0.2}
q0=np.zeros(len(hn))
for i,n in enumerate(hn):
    for k,v in KEY.items():
        if n==k or n.endswith(k): q0[i]=v
M_ROBO=float(m.body_mass.sum()); M_CAIXA=5.0; MEIA=0.13
def set_q(qh,bz,brp):
    d.qpos[:3]=[0.,0.,bz]
    q=np.zeros(4); mujoco.mju_euler2Quat(q,np.array([brp[0],brp[1],0.]),b'xyz'); d.qpos[3:7]=q
    d.qpos[7:]=qh; mujoco.mj_kinematics(m,d); mujoco.mj_comPos(m,d)
def res(ax,az):
    e=[]
    for f in ('left_foot','right_foot'):
        s=SID[f]; e.append(d.site_xpos[s][2])
        R=d.site_xmat[s].reshape(3,3); e+=[R[0,2],R[1,2]]
    pelv=d.xpos[1]; alvo=np.array([pelv[0]+ax,0.,az])
    e+=list(d.site_xpos[SID['left_palm']]-(alvo+np.array([0., MEIA,0.])))
    e+=list(d.site_xpos[SID['right_palm']]-(alvo+np.array([0.,-MEIA,0.])))
    com=(M_ROBO*d.subtree_com[1]+M_CAIXA*alvo)/(M_ROBO+M_CAIXA)
    lf=d.site_xpos[SID['left_foot']]; rf=d.site_xpos[SID['right_foot']]; mid=0.5*(lf+rf)
    e+=[com[0]-mid[0],com[1]-mid[1], lf[1]+rf[1], max(0.,(lf[1]-rf[1])-0.40)]
    return np.array(e)
def resolve(ax,az,seed,iters=240,margem=0.25):
    rng=np.random.default_rng(seed)
    qh=np.clip(q0+(rng.random(len(q0))-0.5)*(0.7 if seed else 0.),lo+1e-3,hi-1e-3)
    st=np.concatenate([qh,[0.72],[0.,0.]])
    for _ in range(iters):
        set_q(st[:29],st[29],st[30:32]); f0=res(ax,az)
        J=np.zeros((len(f0),32))
        for k in range(32):
            s=st.copy(); s[k]+=1e-5; set_q(s[:29],s[29],s[30:32]); J[:,k]=(res(ax,az)-f0)/1e-5
        set_q(st[:29],st[29],st[30:32])
        st=st+np.clip(J.T@np.linalg.solve(J@J.T+2e-2*np.eye(len(f0)),-f0),-0.15,0.15)
        fo=margem*(hi-lo)/2
        st[:29]=np.clip(st[:29],lo+fo,hi-fo); st[29]=np.clip(st[29],0.25,1.05)
    set_q(st[:29],st[29],st[30:32]); f=res(ax,az)
    return np.abs(f[6:12]).max(), st
iE=[hn.index('left_elbow_joint'),hn.index('right_elbow_joint')]
iSP=[hn.index('left_shoulder_pitch_joint'),hn.index('right_shoulder_pitch_joint')]
iW=hn.index('waist_pitch_joint')
print('COTOVELO que o IK escolhe no alvo do BOTAR (default do keyframe = 0,60):')
print(f"{'x_base':>7} {'z':>6} {'err':>8} {'cotov E':>8} {'cotov D':>8} {'ombro E':>8} {'waist':>7} {'pelve':>6}")
for ax in (0.30,0.35,0.40):
    for az in (0.68,0.18):
        best=None
        for s in range(10):
            e,st=resolve(ax,az,s)
            if best is None or e<best[0]: best=(e,st)
        e,st=best
        print(f'{ax:7.2f} {az:6.2f} {e:8.4f} {st[iE[0]]:8.3f} {st[iE[1]]:8.3f} {st[iSP[0]]:8.3f} {st[iW]:+7.3f} {st[29]:6.3f}')
