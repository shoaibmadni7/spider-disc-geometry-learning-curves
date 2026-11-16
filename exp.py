import pandas as pd,numpy as np,json
from common import *
from sklearn.metrics import average_precision_score, matthews_corrcoef, brier_score_loss, roc_curve
import os,pickle
R=json.load(open('results.json')) if os.path.exists('results.json') else {}
def save(): json.dump(R,open('results.json','w'),indent=1)
raw=pd.read_csv('P_base.csv'); print(raw.series.value_counts().to_dict())
d=build(raw); N=len(d); R['cohort']=dict(n=N,patients=int(d.patient.nunique()),prev={t:float(d[t].mean()) for t in TARGETS})
pats=d.patient.values; upat=np.unique(pats); pidx={p:np.where(pats==p)[0] for p in upat}
def boot_idx(rng,d_=None):
    pp=upat if d_ is None else np.unique(d_.patient.values)
    pi=pidx if d_ is None else {p:np.where(d_.patient.values==p)[0] for p in pp}
    return np.concatenate([pi[p] for p in rng.choice(pp,len(pp))])
def ci(v): return [float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]
# ---------- E0: full metrics, repeat-averaged patient bootstrap
def oof_thr(d,t,rep,feats=FEATS):
    y=d[t].values; p=np.zeros(len(d)); yh=np.zeros(len(d),int)
    for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=rep).split(d,y,d.patient):
        X=d[feats].values.astype(float); L=d.level.values
        Ztr,Zte=zlevel(X[tr],L[tr],X[te],L[te]); m=clf().fit(Ztr,y[tr])
        ptr=m.predict_proba(Ztr)[:,1]; fpr,tpr,th=roc_curve(y[tr],ptr); thr=th[np.argmax(tpr-fpr)]
        p[te]=m.predict_proba(Zte)[:,1]; yh[te]=(p[te]>=thr).astype(int)
    return p,yh
OOF={}
mets={'auc':lambda y,p,h:roc_auc_score(y,p),'prauc':lambda y,p,h:average_precision_score(y,p),'mcc':lambda y,p,h:matthews_corrcoef(y,h),'brier':lambda y,p,h:brier_score_loss(y,p)}
if os.path.exists('oof.pkl'): OOF=pickle.load(open('oof.pkl','rb'))
for t in ([] if 'E0' in R and len(R['E0'])==4 else TARGETS):
    R.setdefault('E0',{})
    y=d[t].values; PP=[oof_thr(d,t,r) for r in range(5)]; OOF[t]=PP
    out={}
    rng=np.random.default_rng(0); B=[boot_idx(rng) for _ in range(500)]
    for k,f in mets.items():
        pt=np.mean([f(y,p,h) for p,h in PP]); sd=np.std([f(y,p,h) for p,h in PP])
        bs=[np.mean([f(y[b],p[b],h[b]) for p,h in PP]) for b in B]
        out[k]=dict(mean=float(pt),sd=float(sd),ci=ci(bs))
    # calibration (10 quantile bins on repeat-averaged probability)
    pm=np.mean([p for p,h in PP],0); q=pd.qcut(pm,10,labels=False,duplicates='drop')
    out['calib']=[[float(pm[q==i].mean()),float(y[q==i].mean()),int((q==i).sum())] for i in np.unique(q)]
    R['E0'][t]=out; print('E0',t,{k:round(v['mean'],3) for k,v in out.items() if k!='calib'},flush=True)
pickle.dump(OOF,open('oof.pkl','wb')); save()
# ---------- E1: fold-nested exponents; level-6 sensitivity
def nested_oof(d,t,rep,rec):
    y=d[t].values; p=np.zeros(len(d))
    for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=rep).split(d,y,d.patient):
        s=d.body_h.values; ks=[]
        for f in ['ap_ext','canal_area','vol']:
            v=d[f].values; ok=np.isfinite(v[tr])
            ks.append(int(np.argmin([abs(np.corrcoef(v[tr][ok]/s[tr][ok]**k,s[tr][ok])[0,1]) for k in range(4)])))
        rec.append(tuple(ks)); dd=scale(d,ks); p[te]=fit_predict(dd,tr,te,t)
    return p
full_k=[int(np.argmin([abs(np.corrcoef((d[f]/d.body_h**k)[d[f].notna()],d.body_h[d[f].notna()])[0,1]) for k in range(4)])) for f in ['ap_ext','canal_area','vol']]
full_r={f:[float(np.corrcoef((d[f]/d.body_h**k)[d[f].notna()],d.body_h[d[f].notna()])[0,1]) for k in range(4)] for f in ['ap_ext','canal_area','vol']}
R['E1']={'full_sample_k':full_k,'corr_by_k':full_r,'targets':{}}
rng=np.random.default_rng(1); B=[boot_idx(rng) for _ in range(1000)]
for t in TARGETS:
    rec=[]; y=d[t].values; PN=[nested_oof(d,t,r,rec) for r in range(5)]; PF=[p for p,h in OOF[t]]
    dd=scale(d,full_k); PK=[oof(dd,t,r) for r in range(5)]
    f=lambda P,b=None: np.mean([roc_auc_score(y if b is None else y[b],p if b is None else p[b]) for p in P])
    diff=[f(PN,b)-f(PF,b) for b in B]
    R['E1']['targets'][t]=dict(fixed=float(f(PF)),nested=float(f(PN)),full_sample_selected=float(f(PK)),diff=float(f(PN)-f(PF)),diff_ci=ci(diff),selected={str(k):rec.count(k) for k in set(rec)})
    print('E1',t,R['E1']['targets'][t],flush=True)
d6=build(raw,levels=(1,2,3,4,5,6)); d6np=build(raw,levels=(1,2,3,4,5,6),plaus=False)
R['E1']['level6']={'n_1to6':len(d6),'n_level6':int((d6.level==6).sum()),'flag_1to6':int(d6np.flag.sum()),'flag_level6':int(d6np[d6np.level==6].flag.sum()),
   'pos_level6':{t:int(d6[d6.level==6][t].sum()) for t in TARGETS},'targets':{}}
for t in TARGETS:
    y=d6[t].values; P6=[oof(d6,t,r) for r in range(5)]; m5=(d6.level<=5).values; m6=(d6.level==6).values
    R['E1']['level6']['targets'][t]=dict(auc_1to6=float(np.mean([roc_auc_score(y,p) for p in P6])),auc_on_1to5_when_trained_1to6=float(np.mean([roc_auc_score(y[m5],p[m5]) for p in P6])),
        auc_within_level6=float(np.mean([roc_auc_score(y[m6],p[m6]) for p in P6])) if 0<y[m6].sum()<m6.sum() else None, auc_1to5_primary=R['E0'][t]['auc']['mean'])
    print('E1 L6',t,R['E1']['level6']['targets'][t],flush=True)
from scipy.stats import pointbiserialr
R['E1']['level6']['r_height_narrowing_by_level']={int(l):[float(x) for x in pointbiserialr(g.narrowing,g.disc_h)] for l,g in d6.groupby('level')}
# ---------- E2: geometry learning curves, 5 partitions x 5 orderings
FR=[0.05,0.10,0.25,0.50,1.0]; R['E2']={}
for t in TARGETS:
    y=d[t].values; res={f:[] for f in FR}; npos={f:[] for f in FR}; nposp={f:[] for f in FR}; npat={f:[] for f in FR}
    for part in range(5):
        folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=100+part).split(d,y,d.patient))
        for o in range(5):
            P={f:np.zeros(N) for f in FR}
            for k,(tr,te) in enumerate(folds):
                tp=np.unique(pats[tr]); perm=np.random.default_rng(10000*o+100*part+k).permutation(tp)
                for f in FR:
                    keep=perm[:max(int(round(f*len(tp))),4)]; sub=tr[np.isin(pats[tr],keep)]
                    if len(np.unique(y[sub]))<2: P[f][te]=0.5; continue
                    P[f][te]=fit_predict(d,sub,te,t,minn=8,nanzero=True)
                    npos[f].append(int(y[sub].sum())); npat[f].append(len(keep)); nposp[f].append(int(pd.Series(y[sub]).groupby(pats[sub]).max().sum()))
            for f in FR: res[f].append(roc_auc_score(y,P[f]))
    R['E2'][t]={str(f):dict(mean=float(np.mean(res[f])),sd=float(np.std(res[f])),min=float(np.min(res[f])),max=float(np.max(res[f])),patients=float(np.mean(npat[f])),pos_discs=float(np.mean(npos[f])),pos_discs_range=[int(np.min(npos[f])),int(np.max(npos[f]))],pos_patients=float(np.mean(nposp[f])),
        by_partition=[float(np.mean(res[f][5*p:5*p+5])) for p in range(5)]) for f in FR}
    print('E2',t,{f:(round(v['mean'],3),round(v['sd'],3),round(v['pos_discs']),round(v['pos_patients'])) for f,v in R['E2'][t].items()},flush=True)
# ---------- E3: mask perturbation
R['E3']={}
raws={k:pd.read_csv(f'P_{k}.csv') for k in ['erode1','dilate1','jitter50']}
for k,rw in raws.items():
    dp_all=build(rw,plaus=False); dp=build(rw)
    m=d.merge(dp_all,on=['patient','level'],suffixes=('','_p'),how='left')
    out={'n_plausible':len(dp),'n_flagged':int(dp_all.flag.sum()),'n_missing':int(m.disc_h_p.isna().sum()),'features':{},'targets':{}}
    for f in FEATS+['disc_h','body_h']:
        ok=m[f].notna()&m[f+'_p'].notna(); a=m[f][ok].values; b=m[f+'_p'][ok].values
        # within-level z-score shift is what the classifier sees
        out['features'][f]=dict(r=float(np.corrcoef(a,b)[0,1]),mean_rel_change_pct=float(100*np.mean(b-a)/np.mean(np.abs(a))),median_abs_rel_diff_pct=float(100*np.median(np.abs(b-a)/np.maximum(np.abs(a),1e-6))))
    mm=m.copy()
    for f in FEATS: mm[f]=mm[f+'_p']
    rng=np.random.default_rng(2); B=[boot_idx(rng) for _ in range(1000)]
    for t in TARGETS:
        y=d[t].values; Pc=[p for p,h in OOF[t]]
        Ppp=[oof(mm,t,r) for r in range(5)]           # trained and tested on perturbed masks
        Pcp=[]                                         # trained on reference masks, tested on perturbed
        for r in range(5):
            p=np.zeros(N)
            for tr,te in StratifiedGroupKFold(5,shuffle=True,random_state=r).split(d,y,d.patient):
                X=d[FEATS].values.astype(float); Xp=mm[FEATS].values.astype(float); L=d.level.values
                Ztr,Zte=zlevel(X[tr],L[tr],Xp[te],L[te]); p[te]=clf().fit(Ztr,y[tr]).predict_proba(Zte)[:,1]
            Pcp.append(p)
        f=lambda P,b=None: np.mean([roc_auc_score(y if b is None else y[b],p if b is None else p[b]) for p in P])
        out['targets'][t]=dict(ref=float(f(Pc)),pert_pert=float(f(Ppp)),d_pp=float(f(Ppp)-f(Pc)),d_pp_ci=ci([f(Ppp,b)-f(Pc,b) for b in B]),ref_pert=float(f(Pcp)),d_cp=float(f(Pcp)-f(Pc)),d_cp_ci=ci([f(Pcp,b)-f(Pc,b) for b in B]))
    # learning-curve end points under perturbation: 10% fraction, 5 orderings, partition 0
    for t in ['narrowing','pfirrmann_severe']:
        y=d[t].values; a=[]; bb=[]
        for o in range(5):
            Pa=np.zeros(N); Pb=np.zeros(N)
            for kk,(tr,te) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=100).split(d,y,d.patient)):
                tp=np.unique(pats[tr]); keep=np.random.default_rng(10000*o+kk).permutation(tp)[:int(round(0.1*len(tp)))]; sub=tr[np.isin(pats[tr],keep)]
                Pa[te]=fit_predict(d,sub,te,t,minn=8,nanzero=True); Pb[te]=fit_predict(mm,sub,te,t,minn=8,nanzero=True)
            a.append(roc_auc_score(y,Pa)); bb.append(roc_auc_score(y,Pb))
        out['targets'][t]['lc10_ref']=float(np.mean(a)); out['targets'][t]['lc10_pert']=float(np.mean(bb))
    R['E3'][k]=out; print('E3',k,json.dumps(out)[:900],flush=True)
# ---------- E4: leave-one-scanner-out with patient-clustered CI
R['E4']={}
sc=d.groupby('scanner').patient.nunique(); held=[s for s in sc.index if sc[s]>=15]
for s in held:
    te=np.where(d.scanner.values==s)[0]; tr=np.where(d.scanner.values!=s)[0]; dt=d.iloc[te].reset_index(drop=True)
    out={'patients':int(sc[s]),'discs':len(te),'targets':{}}
    rng=np.random.default_rng(3); B=[boot_idx(rng,dt) for _ in range(2000)]
    for t in TARGETS:
        y=d[t].values; p=fit_predict(d,tr,te,t); yt=y[te]
        bs=[roc_auc_score(yt[b],p[b]) for b in B if 0<yt[b].sum()<len(b)]
        out['targets'][t]=dict(auc=float(roc_auc_score(yt,p)),ci=ci(bs),pos=int(yt.sum()),neg=int(len(yt)-yt.sum()),pos_patients=int(dt.groupby('patient')[t].max().sum()))
    R['E4'][s]=out; print('E4',s,out,flush=True)
# ---------- E5: sequences
ser=raw.groupby('patient').series.agg(lambda x:'+'.join(sorted(set(x)))); R['E5']={'availability':ser.value_counts().to_dict(),'targets':{}}
for name,sl in {'t2_only':['t2'],'t1_only':['t1']}.items():
    ds=build(raw,series=sl); m=d.merge(ds[['patient','level']+FEATS],on=['patient','level'],suffixes=('','_s'))
    ms=m.copy()
    for f in FEATS: ms[f]=ms[f+'_s']
    R['E5'][name+'_n']=dict(discs=len(m),patients=int(m.patient.nunique()))
    for t in TARGETS:
        y=m[t].values; fa=np.mean([roc_auc_score(y,oof(m,t,r)) for r in range(5)]); fs=np.mean([roc_auc_score(y,oof(ms,t,r)) for r in range(5)])
        R['E5']['targets'].setdefault(t,{})[name]=dict(all_series_same_discs=float(fa),single_sequence=float(fs))
    print('E5',name,R['E5'][name+'_n'],{t:R['E5']['targets'][t][name] for t in TARGETS},flush=True)
