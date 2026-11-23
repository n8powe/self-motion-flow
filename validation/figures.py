"""Create the descriptor figures from the released frames and audit tables."""
import argparse
from pathlib import Path
from zipfile import ZipFile
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image

HERE=Path(__file__).resolve().parent
STYLE={"train":("Training","#246787","-"),"validation":("Validation","#bf772e","--"),
       "test":("Original test","#8b406f","-."),"expanded_test":("Expanded test","#348465",":")}
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
                     "font.size":9,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})


def save(fig, path):
    fig.savefig(path.with_suffix(".png"),dpi=240,bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"),bbox_inches="tight")
    plt.close(fig)


def example(archive,out):
    fig,axes=plt.subplots(2,3,figsize=(8.3,5),layout="constrained")
    with ZipFile(archive) as z:
        root="self-motion-flow-v1/DATA/rendered/deep100k_test_00005/"
        for ax,c in zip(axes[0],["absent","static","moving"]):
            ax.imshow(Image.open(io.BytesIO(z.read(root+c+"/0000.png"))))
            ax.set_title({"absent":"Target absent","static":"Target stationary","moving":"Target moving"}[c])
        def read(c):
            with np.load(io.BytesIO(z.read(root+c+"/0000_reference.npz"))) as v:
                return {k:v[k] for k in ["flow","depth","valid","occupancy"]}
        s,m=read("static"),read("moving")
    a,b,c=axes[1]
    im=a.imshow(s["depth"],cmap="viridis")
    fig.colorbar(im,ax=a,orientation="horizontal",shrink=.78,pad=.01,label="Axial depth (m)")
    y,x=np.mgrid[4:128:9,4:128:9]
    ok=s["valid"][y,x]
    u,v=s["flow"][y[ok],x[ok]].T
    b.quiver(x[ok],y[ok],u,v,angles="xy",scale_units="xy",scale=.035,color="#20627a",
             width=.006,pivot="middle")
    b.set(xlim=(0,127),ylim=(127,0),aspect="equal",facecolor="#ecf2f4")
    b.set_xlabel("Arrows enlarged 28.6×")
    delta=np.linalg.norm(m["flow"]-s["flow"],axis=-1)
    im=c.imshow(np.ma.masked_where(~(s["valid"] & m["valid"]),delta),cmap="magma",vmin=0)
    fig.colorbar(im,ax=c,orientation="horizontal",shrink=.78,pad=.01,label="Difference (px/interval)")
    for ax,title in zip(axes[1],["Stationary scene depth","Stationary scene flow","Moving minus stationary flow"]):
        ax.set_title(title)
    for ax in axes.flat:
        ax.set(xticks=[],yticks=[])
        for spine in ax.spines.values(): spine.set_visible(False)
    save(fig,out/"figure1_matched_scene")


def coverage(k,p,out):
    fig,axes=plt.subplots(2,2,figsize=(7.9,5.9))
    for ax,(col,title,xlabel,scale) in zip(axes.flat,[
        ("camera_m_s","a   Observer translation","Speed (m/s)","log"),
        ("rotation_deg_s","b   Camera rotation","Angular speed (degrees/s)","symlog"),
        ("depth_p50_m","c   Scene depth","Median axial depth of static frame (m)","linear"),
        ("moving_target_m_s","d   Independent target translation","Speed when target moves (m/s)","log")]):
        data=p[p.condition=="static"] if col=="depth_p50_m" else k
        for split,(name,color,ls) in STYLE.items():
            x=np.sort(data.loc[data.split==split,col].dropna().to_numpy())
            if len(x):
                ax.step(x,100*np.arange(1,len(x)+1)/len(x),where="post",color=color,ls=ls,lw=1.8)
        ax.set(xlabel=xlabel,ylabel="Scenes at or below (%)",ylim=(0,102),yticks=[0,25,50,75,100])
        ax.set_title(title,loc="left")
        if scale=="symlog": ax.set_xscale("symlog",linthresh=1); ax.set_xticks([0,1,10,100,600],labels=["0","1","10","100","600"])
        else: ax.set_xscale(scale)
    legend=[Line2D([],[],color=c,ls=ls,label=n,lw=2) for n,c,ls in STYLE.values()]
    fig.legend(handles=legend,loc="upper center",ncol=4,frameon=False)
    fig.tight_layout(rect=(0,0,1,.93))
    save(fig,out/"figure2_coverage")


def geometry(p,out):
    fig,axes=plt.subplots(1,2,figsize=(8,3.6),layout="constrained")
    static=p[p.condition=="static"]
    for j,(split,(name,color,_)) in enumerate(STYLE.items()):
        x=np.sort(static.loc[static.split==split,"mean_epe_px"])
        axes[0].plot(np.linspace(j-.23,j+.23,len(x)),x,".",color=color,ms=2.7)
        axes[0].plot([j-.26,j+.26],[np.median(x)]*2,c="black",lw=1.3)
    axes[0].set(xticks=range(4),xticklabels=["Train","Val.","Original\ntest","Expanded\ntest"],
                ylabel="Mean endpoint error (native pixels)",yscale="log")
    axes[0].set_title("a   Static scenes",loc="left")
    for split in ["test","expanded_test"]:
        a=p[(p.split==split)&(p.condition=="moving")]
        axes[1].scatter(a.moving_corrected_px,a.rigid_only_target_px,s=10,
                        color=STYLE[split][1],alpha=.65,label=STYLE[split][0])
    axes[1].set(xscale="log",yscale="log",xlabel="Including target translation (px)",
                ylabel="Omitting target translation (px)")
    axes[1].set_title("b   Valid moving-target pixels",loc="left")
    axes[1].legend(frameon=False,fontsize=8,loc="lower left")
    save(fig,out/"figure3_projection")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("--results",type=Path,default=HERE/"results")
    parser.add_argument("--out",type=Path,default=HERE/"figures")
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    k=pd.read_csv(args.results/"scene_kinematics.csv")
    p=pd.read_csv(args.results/"projection_and_depth.csv")
    example(args.archive,args.out)
    coverage(k,p,args.out)
    geometry(p,args.out)
