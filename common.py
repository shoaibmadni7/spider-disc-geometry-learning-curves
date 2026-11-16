import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.metrics import roc_auc_score
FEATS=['disc_height_ratio','ap_ratio','canal_ratio','volume_ratio','slip_pct']
TARGETS=['narrowing','bulging','pfirrmann_severe','modic_any']
U='../data/'   # folder with overview.csv and radiological_gradings.csv
def grades():
    g=pd.read_csv(U+'radiological_gradings.csv'); g.columns=[c.strip() for c in g.columns]
    g=g.rename(columns={'IVD label':'level','Pfirrman grade':'pfirrmann','Disc narrowing':'narrowing','Disc bulging':'bulging','Modic':'modic'})
    g['pfirrmann_severe']=(g.pfirrmann>=4).astype(int); g['modic_any']=(g.modic>0).astype(int)
    g=g.rename(columns={'Patient':'patient'})
    return g
def scanners():
    ov=pd.read_csv(U+'overview.csv'); ov['patient']=ov['new_file_name'].str.split('_').str[0].astype(int)
    return ov.groupby('patient')['ManufacturerModelName'].agg(lambda x:x.mode().iloc[0]).rename('scanner').reset_index()
def build(raw, series=None, levels=(1,2,3,4,5), plaus=True, exps=(1,0,2)):
    r=raw if series is None else raw[raw.series.isin(series)]
    agg=r.groupby(['patient','level'])[['disc_h','body_h','ap_ext','vol','canal_area','slip_pct','angle']].mean().reset_index()
    d=agg.merge(grades(),on=['patient','level']).merge(scanners(),on='patient',how='left')
    d['disc_height_ratio']=d.disc_h/d.body_h
    d['flag']=~(d.body_h.between(15,40)&d.disc_h.between(2,18)&d.disc_height_ratio.between(0.05,0.7))
    d=d[d.level.isin(levels)]
    if plaus: d=d[~d.flag]
    d=d.reset_index(drop=True)
    return scale(d,exps)
def scale(d,exps=(1,0,2)):
    d=d.copy()
    d['ap_ratio']=d.ap_ext/d.body_h**exps[0]; d['canal_ratio']=d.canal_area/d.body_h**exps[1]; d['volume_ratio']=d.vol/d.body_h**exps[2]
    return d
def zlevel(Xtr,Ltr,Xte,Lte,minn=20):
    Ztr=np.empty_like(Xtr,dtype=float); Zte=np.empty_like(Xte,dtype=float)
    gm=np.nanmean(Xtr,0); gs=np.nanstd(Xtr,0); gs[~(gs>0)]=1
    for l in np.unique(np.concatenate([Ltr,Lte])):
        a=Ltr==l; b=Lte==l
        if a.sum()>=minn:
            m=np.nanmean(Xtr[a],0); s=np.nanstd(Xtr[a],0); s[~(s>0)]=1
            m=np.where(np.isnan(m),gm,m); s=np.where(np.isnan(s),gs,s)
        else: m,s=gm,gs
        Ztr[a]=(Xtr[a]-m)/s; Zte[b]=(Xte[b]-m)/s
    return Ztr,Zte
def clf(): return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=1.0,max_iter=5000,class_weight='balanced',random_state=0))
def fit_predict(d,tr,te,t,feats=FEATS,minn=20,nanzero=False):
    X=d[feats].values.astype(float); L=d.level.values
    Ztr,Zte=zlevel(X[tr],L[tr],X[te],L[te],minn)
    if nanzero: Ztr=np.nan_to_num(Ztr); Zte=np.nan_to_num(Zte)
    m=clf().fit(Ztr,d[t].values[tr]); return m.predict_proba(Zte)[:,1]
def oof(d,t,rep,feats=FEATS):
    y=d[t].values; p=np.zeros(len(d))
    for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=rep).split(d,y,d.patient): p[te]=fit_predict(d,tr,te,t,feats)
    return p
