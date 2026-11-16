import os,json,warnings,sys,time
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, brier_score_loss, roc_curve
D='original_outputs/'
z=np.load('original_outputs/spider_crops_tight_index.npz',allow_pickle=True); ri=z['row_index']; seq=z['sequence']
prim=pd.read_csv(D+'spider_primary_cohort.csv').loc[ri].reset_index(drop=True)
Eemb=np.load(D+'spider_dinov2_tight.npy')
ov=pd.read_csv('../data/overview.csv'); ov['Patient']=ov['new_file_name'].str.split('_').str[0].astype(int)
scan=ov.groupby('Patient').agg(scanner=('ManufacturerModelName',lambda s:s.mode().iat[0])).reset_index()
prim=prim.merge(scan,on='Patient',how='left')
INV=['disc_height_ratio','ap_ratio','canal_ratio','volume_ratio','slip_pct']
TARGETS={'disc narrowing':'narrowing','disc bulging':'bulging','Pfirrmann >= 4':'pfirrmann_severe','Modic (any)':'modic_any'}
ARMS=['geometry','dinov2','fusion','cnn']
pats=prim.Patient.values; uniq=prim.Patient.unique(); N=len(prim)
pidx={u:np.flatnonzero(pats==u) for u in uniq}
def boots(seed,B,sub=None):
    rng=np.random.default_rng(seed); uu=uniq if sub is None else np.unique(pats[sub])
    pi=pidx if sub is None else {u:np.flatnonzero(pats[sub]==u) for u in uu}
    return [np.concatenate([pi[u] for u in rng.choice(uu,len(uu))]) for _ in range(B)]
ci=lambda v:[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]
def bh(p):
    p=np.asarray(p); o=np.argsort(p); m=len(p); adj=p[o]*m/(np.arange(m)+1); adj=np.minimum.accumulate(adj[::-1])[::-1].clip(max=1); out=np.empty(m); out[o]=adj; return out
def _zscore(ref,ref_lvl,X,lvl,minn=20):
    out=X.copy(); gm,gs=ref.mean(),ref.std().replace(0,1)
    for l in np.unique(lvl):
        m=ref_lvl==l
        mu=ref[m].mean() if m.sum()>=minn else gm
        sd=ref[m].std().replace(0,1) if m.sum()>=minn else gs
        out.loc[lvl==l]=(X.loc[lvl==l]-mu)/sd
    return out
def geom_fit(df,tr,te,y,minn=20,nanzero=False,feats=INV):
    lv=df['ivd_label'].values; A,Bq=df[feats].iloc[tr],df[feats].iloc[te]
    A2=_zscore(A,lv[tr],A,lv[tr],minn); B2=_zscore(A,lv[tr],Bq,lv[te],minn)
    if nanzero: A2=A2.fillna(0); B2=B2.fillna(0)
    m=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=5000,class_weight='balanced',random_state=0)).fit(A2,y[tr])
    return m.predict_proba(B2)[:,1],m.predict_proba(A2)[:,1]
def dino_fit(tr,te,y):
    m=make_pipeline(StandardScaler(),PCA(n_components=min(128,len(tr)-1),random_state=0),LogisticRegression(max_iter=5000,class_weight='balanced',random_state=0)).fit(Eemb[tr],y[tr])
    return m.predict_proba(Eemb[te])[:,1]
R=json.load(open('orig_results.json')) if os.path.exists('orig_results.json') else {}
def save(): json.dump(R,open('orig_results.json','w'),indent=1)
oof=np.load(D+'nb3_oof_rep.npy',allow_pickle=True).item()
for t,tc in TARGETS.items():
    for r in range(5):
        q=np.load(D+f'nb3_cnn/{tc}_rep{r}.npz'); oof[(t,'cnn',r)]=(q['p'],q['b'])
mets={'auc':lambda y,p,b:roc_auc_score(y,p),'prauc':lambda y,p,b:average_precision_score(y,p),'mcc':lambda y,p,b:matthews_corrcoef(y,b),'brier':lambda y,p,b:brier_score_loss(y,p)}
# ---- A: all-arm metrics, repeat-averaged patient bootstrap; paired differences
if 'A' not in R:
    Bt=boots(0,1000); A={}; Dd={}; ps=[]; keys=[]
    for t,tc in TARGETS.items():
        y=prim[tc].values.astype(int); A[t]={}
        for arm in ARMS:
            A[t][arm]={}
            for k,f in mets.items():
                pt=np.mean([f(y,*oof[(t,arm,r)]) for r in range(5)])
                bs=[np.mean([f(y[b],oof[(t,arm,r)][0][b],oof[(t,arm,r)][1][b]) for r in range(5)]) for b in Bt]
                A[t][arm][k]=dict(mean=float(pt),ci=ci(bs))
            pm=np.mean([oof[(t,arm,r)][0] for r in range(5)],0); q=pd.qcut(pm,10,labels=False,duplicates='drop')
            A[t][arm]['ece']=float(sum(abs(pm[q==i].mean()-y[q==i].mean())*(q==i).sum() for i in np.unique(q))/N)
        for arm in ARMS[1:]:
            pt=np.mean([roc_auc_score(y,oof[(t,arm,r)][0])-roc_auc_score(y,oof[(t,'geometry',r)][0]) for r in range(5)])
            bs=np.array([np.mean([roc_auc_score(y[b],oof[(t,arm,r)][0][b])-roc_auc_score(y[b],oof[(t,'geometry',r)][0][b]) for r in range(5)]) for b in Bt])
            p=max(2*min((bs<=0).mean(),(bs>=0).mean()),1/len(bs)); Dd[f'{t}|{arm}']=dict(d=float(pt),ci=ci(bs),p=float(p)); ps.append(p); keys.append(f'{t}|{arm}')
    for k,a in zip(keys,bh(ps)): Dd[k]['p_bh']=float(a)
    # T2-only crops
    m2=seq=='t2'; T2={'n':int(m2.sum()),'patients':int(len(np.unique(pats[m2])))}
    for t,tc in TARGETS.items():
        y=prim[tc].values.astype(int); T2[t]={arm:[float(np.mean([roc_auc_score(y[m2],oof[(t,arm,r)][0][m2]) for r in range(5)])),float(np.mean([roc_auc_score(y,oof[(t,arm,r)][0]) for r in range(5)]))] for arm in ARMS}
    R['A']=A; R['D']=Dd; R['T2']=T2; save(); print('A done',json.dumps(Dd)[:1500],T2,flush=True)
# ---- LC: repeat-averaged paired differences per fraction
if 'LC' not in R:
    g=np.load(D+'nb4_learning_curve/oof_geom_dino.npy',allow_pickle=True).item(); Bt=boots(0,1000); LC={}; ps=[]; keys=[]
    FT={0.05:'005',0.1:'010',0.25:'025',0.5:'050',1.0:'100'}
    for tc in ['narrowing','pfirrmann_severe']:
        y=prim[tc].values.astype(int)
        for f,ft in FT.items():
            dr=[dd for dd in (0,1) if os.path.exists(D+f'nb4_learning_curve/cnn_{tc}_f{ft}_d{dd}.npz')]
            cn=[np.load(D+f'nb4_learning_curve/cnn_{tc}_f{ft}_d{dd}.npz')['p'] for dd in dr]; ge=[g[(tc,'geometry',f,dd)] for dd in dr]
            pt=np.mean([roc_auc_score(y,c)-roc_auc_score(y,e) for c,e in zip(cn,ge)])
            bs=np.array([np.mean([roc_auc_score(y[b],c[b])-roc_auc_score(y[b],e[b]) for c,e in zip(cn,ge)]) for b in Bt])
            p=max(2*min((bs<=0).mean(),(bs>=0).mean()),1/len(bs)); k=f'{tc}|{f}'; LC[k]=dict(d=float(pt),ci=ci(bs),p=float(p),draws=len(dr),geom=float(np.mean([roc_auc_score(y,e) for e in ge])),cnn=float(np.mean([roc_auc_score(y,c) for c in cn]))); ps.append(p); keys.append(k)
    for k,a in zip(keys,bh(ps)): LC[k]['p_bh']=float(a)
    R['LC']=LC; save(); print('LC',json.dumps(LC),flush=True)
# ---- E1: fold-nested exponents (original features)
RAW={'ap_ratio':'disc_ap_mm','canal_ratio':'canal_area_mm2','volume_ratio':'disc_volume_mm3'}
def with_exps(df,ks):
    d=df.copy()
    for (f,rw),k in zip(RAW.items(),ks): d[f]=d[rw]/d['body_scale_mm']**k
    return d
def pick(df,idx):
    s=df['body_scale_mm'].values[idx]; ks=[]
    for rw in RAW.values():
        v=df[rw].values[idx]; ok=np.isfinite(v)
        ks.append(int(np.argmin([abs(np.corrcoef(v[ok]/s[ok]**k,s[ok])[0,1]) for k in range(4)])))
    return tuple(ks)
if 'E1' not in R:
    chk=with_exps(prim,(1,0,2)); print('recompute check',[float(np.abs(chk[f]-prim[f]).max()) for f in RAW]); 
    E1={'full_k':list(pick(prim,np.arange(N))),'corr':{rw:[float(np.corrcoef(prim[rw]/prim.body_scale_mm**k,prim.body_scale_mm)[0,1]) for k in range(4)] for rw in RAW.values()},'t':{}}
    Bt=boots(1,1000)
    for t,tc in TARGETS.items():
        y=prim[tc].values.astype(int); PN=[]; rec=[]; PF=[oof[(t,'geometry',r)][0] for r in range(5)]; PR=[]
        for r in range(5):
            p=np.zeros(N); p2=np.zeros(N)
            for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=r).split(prim,y,pats):
                ks=pick(prim,tr); rec.append(ks); p[te]=geom_fit(with_exps(prim,ks),tr,te,y)[0]; p2[te]=geom_fit(prim,tr,te,y)[0]
            PN.append(p); PR.append(p2)
        f=lambda P,b=None:np.mean([roc_auc_score(y if b is None else y[b],p if b is None else p[b]) for p in P])
        E1['t'][t]=dict(saved=float(f(PF)),recomputed=float(f(PR)),nested=float(f(PN)),diff=float(f(PN)-f(PF)),diff_ci=ci([f(PN,b)-f(PF,b) for b in Bt]),sel={str(k):rec.count(k) for k in set(rec)})
        print('E1',t,E1['t'][t],flush=True)
    R['E1']=E1; save()
# ---- level 6 (original features file)
if 'L6' not in R:
    F=pd.read_csv(D+'spider_features.csv'); print(F.groupby('ivd_label').quality_ok.agg(['size','sum']).head(8))
    F6=F[(F.ivd_label<=6)&F.quality_ok].reset_index(drop=True); F5=F6[F6.ivd_label<=5]
    L={'n':len(F6),'n6':int((F6.ivd_label==6).sum()),'flag_all':int((~F[F.ivd_label<=6].quality_ok).sum()),'flag6':int((~F[F.ivd_label==6].quality_ok).sum()),'pos6':{tc:int(F6[F6.ivd_label==6][tc].sum()) for tc in TARGETS.values()},'t':{}}
    for t,tc in TARGETS.items():
        y=F6[tc].values.astype(int); m5=(F6.ivd_label<=5).values; m6=~m5; a=[];b=[];c=[]
        for r in range(5):
            p=np.zeros(len(F6))
            for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=r).split(F6,y,F6.Patient.values): p[te]=geom_fit(F6,tr,te,y)[0]
            a.append(roc_auc_score(y,p)); b.append(roc_auc_score(y[m5],p[m5])); c.append(roc_auc_score(y[m6],p[m6]))
        L['t'][t]=dict(all=float(np.mean(a)),on15=float(np.mean(b)),within6=float(np.mean(c)))
    R['L6']=L; save(); print('L6',L,flush=True)
# ---- E2/E6: geometry learning curves 5 partitions x 5 orderings, fixed and nested exponents
if 'E2' not in R:
    FR=[0.05,0.10,0.25,0.50,1.0]; E2={}; E6={}
    for t,tc in TARGETS.items():
        y=prim[tc].values.astype(int); res={f:[] for f in FR}; resn={f:[] for f in FR}; npos={f:[] for f in FR}; npp={f:[] for f in FR}
        for part in range(5):
            folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=100+part).split(prim,y,pats))
            for o in range(5):
                P={f:np.zeros(N) for f in FR}; Pn={f:np.zeros(N) for f in FR}
                for k,(tr,te) in enumerate(folds):
                    tp=np.unique(pats[tr]); perm=np.random.default_rng(10000*o+100*part+k).permutation(tp)
                    for f in FR:
                        keep=perm[:max(int(round(f*len(tp))),4)]; sub=tr[np.isin(pats[tr],keep)]
                        if len(np.unique(y[sub]))<2: P[f][te]=0.5; Pn[f][te]=0.5; continue
                        P[f][te]=geom_fit(prim,sub,te,y,8,True)[0]
                        if tc in ('narrowing','pfirrmann_severe'): Pn[f][te]=geom_fit(with_exps(prim,pick(prim,sub)),sub,te,y,8,True)[0]
                        npos[f].append(int(y[sub].sum())); npp[f].append(int(pd.Series(y[sub]).groupby(pats[sub]).max().sum()))
                for f in FR:
                    res[f].append(roc_auc_score(y,P[f]))
                    if tc in ('narrowing','pfirrmann_severe'): resn[f].append(roc_auc_score(y,Pn[f]))
        E2[tc]={str(f):dict(mean=float(np.mean(res[f])),sd=float(np.std(res[f])),min=float(np.min(res[f])),max=float(np.max(res[f])),pos=float(np.mean(npos[f])),posp=float(np.mean(npp[f]))) for f in FR}
        if resn[0.05]: E6[tc]={str(f):float(np.mean(resn[f])) for f in FR}
        print('E2',tc,{f:(round(v['mean'],3),round(v['sd'],3),round(v['pos']),round(v['posp'])) for f,v in E2[tc].items()},E6.get(tc),flush=True)
    R['E2']=E2; R['E6']=E6; save()
# ---- E4: LOSO per scanner, all arms with patient-clustered CI
if 'E4' not in R:
    saved=pd.read_csv(D+'nb3_loso_all_arms.csv'); E4={}
    counts=prim.drop_duplicates('Patient').scanner.value_counts(); held=[s for s,n in counts.items() if n>=15]
    for s in held:
        te=np.flatnonzero((prim.scanner==s).values); tr=np.flatnonzero((prim.scanner!=s).values); Bt=boots(3,2000,te); E4[s]={'patients':int(counts[s]),'discs':len(te),'t':{}}
        for t,tc in TARGETS.items():
            y=prim[tc].values.astype(int); yt=y[te]
            P={'geometry':geom_fit(prim,tr,te,y)[0],'dinov2':dino_fit(tr,te,y),'cnn':np.load(D+f'nb3_cnn/loso_{tc}_{s}.npz')['p']}
            o={'pos':int(yt.sum()),'neg':int(len(yt)-yt.sum())}
            for arm,p in P.items():
                sv=float(saved[(saved.target==t)&(saved.scanner==s)&(saved.arm==arm)].auc.iloc[0])
                o[arm]=dict(auc=float(roc_auc_score(yt,p)),saved=sv,ci=ci([roc_auc_score(yt[b],p[b]) for b in Bt if 0<yt[b].sum()<len(b)]))
            o['cnn_minus_geom']=dict(d=o['cnn']['auc']-o['geometry']['auc'],ci=ci([roc_auc_score(yt[b],P['cnn'][b])-roc_auc_score(yt[b],P['geometry'][b]) for b in Bt if 0<yt[b].sum()<len(b)]))
            E4[s]['t'][t]=o
        print('E4',s,json.dumps(E4[s])[:900],flush=True)
    R['E4']=E4; save()
print('ALL DONE')
