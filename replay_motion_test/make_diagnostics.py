import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import make_plots as M

ayush=M.load_all(M.FILES); nalin=M.load_all(M.NALIN_FILES)
OUT=os.path.join(M.PLOTS,"hand_diagnostics"); os.makedirs(OUT,exist_ok=True)

def og(run): on,off=M.motion_bounds(run); return run["t_rel"]-on, off-on
def rs(ra,rb,key,n=1400):
    ta,da=og(ra); tb,db=og(rb); g=np.linspace(0,min(da,db),n)
    A=np.stack([np.interp(g,ta,ra[key][:,i]) for i in range(ra[key].shape[1])],1)
    B=np.stack([np.interp(g,tb,rb[key][:,i]) for i in range(rb[key].shape[1])],1)
    return g,A,B

# ---- 1) Thumb detail: command(action) vs achieved(act_pos) both hands, + PWM ----
cond,mode="balls","position"
ra,rb=ayush[(cond,mode)],nalin[(cond,mode)]
act=list(ra["actuator_order"])
g,cA,cB=rs(ra,rb,"action"); _,pA,pB=rs(ra,rb,"act_pos"); _,eA,eB=rs(ra,rb,"command")
thumbs=["rh_THJ1","rh_THJ2","rh_THJ4","rh_THJ5"]
fig,axes=plt.subplots(2,4,figsize=(17,7))
for j,name in enumerate(thumbs):
    i=act.index(name)
    ax=axes[0][j]
    ax.plot(g,cA[:,i],color="k",ls="--",lw=1.2,label="command (both ~equal)")
    ax.plot(g,pA[:,i],color="tab:blue",lw=1.3,label="ayush achieved")
    ax.plot(g,pB[:,i],color="tab:green",lw=1.3,label="nalin achieved")
    ax.set_title(f"{name}  achieved vs command   (ROM a/n={pA[:,i].max()-pA[:,i].min():.2f}/{pB[:,i].max()-pB[:,i].min():.2f})",fontsize=9)
    ax.grid(alpha=.3); ax.tick_params(labelsize=7)
    if j==0: ax.legend(fontsize=7)
    ax2=axes[1][j]
    ax2.plot(g,np.abs(eA[:,i]),color="tab:blue",lw=1.0,label="ayush |PWM|")
    ax2.plot(g,np.abs(eB[:,i]),color="tab:green",lw=1.0,label="nalin |PWM|")
    ax2.set_title(f"{name}  |command PWM|   (mean a/n={np.abs(eA[:,i]).mean():.0f}/{np.abs(eB[:,i]).mean():.0f})",fontsize=9)
    ax2.grid(alpha=.3); ax2.tick_params(labelsize=7)
    if j==0: ax2.legend(fontsize=7)
fig.suptitle(f"THUMB diagnostic — {cond}/{mode}: ayush's THJ1 under-travels while its motor pushes harder (binding)",fontsize=12)
fig.text(0.5,0.005,"x: time since motion onset (s)   top y: rad   bottom y: |PWM| (0-600)",ha="center",fontsize=9)
fig.tight_layout(rect=[0,0.02,1,0.97]); fig.savefig(os.path.join(OUT,"thumb_detail_balls_position.png"),dpi=140); plt.close(fig)

# ---- 2) Per-joint ROM ratio (ayush/nalin), joint space, both ball modes ----
fig,axes=plt.subplots(1,2,figsize=(15,5))
for m,mode in enumerate(["position","trajectory"]):
    ra,rb=ayush[("balls",mode)],nalin[("balls",mode)]
    jo=list(ra["joint_order"])
    _,pA,pB=rs(ra,rb,"gt_pos")
    romA=pA.max(0)-pA.min(0); romB=pB.max(0)-pB.min(0)
    ratio=romA/np.clip(romB,1e-6,None)
    ax=axes[m]; xs=np.arange(len(jo))
    colors=["crimson" if (r<0.8 or r>1.25) else "steelblue" for r in ratio]
    ax.bar(xs,ratio,color=colors)
    ax.axhline(1.0,color="k",lw=.8); ax.axhline(0.8,color="crimson",ls=":",lw=.8); ax.axhline(1.25,color="crimson",ls=":",lw=.8)
    ax.set_xticks(xs); ax.set_xticklabels([j.replace("rh_","") for j in jo],rotation=90,fontsize=7)
    ax.set_title(f"balls/{mode}: achieved ROM ratio ayush/nalin (red = >25% off)",fontsize=10)
    ax.set_ylabel("ROM_ayush / ROM_nalin")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"rom_ratio_balls.png"),dpi=140); plt.close(fig)
print("wrote diagnostics to",OUT)
print(os.listdir(OUT))
