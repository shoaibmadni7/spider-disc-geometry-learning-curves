import numpy as np,pandas as pd,json,glob,os
from sklearn.metrics import roc_auc_score
B='revision_experiments/'; D='original_outputs/'
c=pd.read_csv(B+'cohort_rows.csv'); pats=c.Patient.values; uniq=np.unique(pats); pidx={u:np.flatnonzero(pats==u) for u in uniq}; N=len(c)
def boots(seed,n,sub=None):
    rng=np.random.default_rng(seed)
    if sub is None: return [np.concatenate([pidx[u] for u in rng.choice(uniq,len(uniq))]) for _ in range(n)]
    uu=np.unique(pats[sub]); pi={u:np.flatnonzero(pats[sub]==u) for u in uu}
    return [np.concatenate([pi[u] for u in rng.choice(uu,len(uu))]) for _ in range(n)]
ci=lambda v:[float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]
def bh(p):
    p=np.asarray(p); o=np.argsort(p); m=len(p); a=p[o]*m/(np.arange(m)+1); a=np.minimum.accumulate(a[::-1])[::-1].clip(max=1); out=np.empty(m); out[o]=a; return out
def pv(bs): return float(max(2*min((bs<=0).mean(),(bs>=0).mean()),1/len(bs)))
R={}
FR=[0.05,0.10,0.25,0.50,1.0]; Bt=boots(0,1000)
# ---- learning curves
LC={}
for scheme in ['matched_updates','early_stop','epochs8']:
    ps=[];keys=[]
    for tc in ['narrowing','pfirrmann_severe']:
        y=c[tc].values
        G={p:np.load(B+f'lc/geodino_{tc}_p{p}.npz') for p in range(3)}
        for f in FR:
            runs=[(p,d) for p in range(3) for d in ([0,1,2] if f<1 else [0])]
            cn=[];ge=[];di=[];upd=[];pos=[];posp=[]
            for p,d in runs:
                z=np.load(B+f'lc/cnn_{scheme}_{tc}_p{p}_f{int(f*100):03d}_d{d}.npz'); cn.append(z['p']); info=json.loads(str(z['info']))
                upd+= [i['updates'] for i in info]; pos+=[i['pos_discs'] for i in info]; posp+=[i['pos_patients'] for i in info]
                ge.append(G[p][f'geometry|{f}|{d}']); di.append(G[p][f'dinov2|{f}|{d}'])
            assert not any(np.isnan(x).any() for x in cn+ge)
            a=lambda P:[roc_auc_score(y,x) for x in P]
            bs=np.array([np.mean([roc_auc_score(y[b],x[b])-roc_auc_score(y[b],g[b]) for x,g in zip(cn,ge)]) for b in Bt])
            k=f'{scheme}|{tc}|{f}'
            LC[k]=dict(cnn=float(np.mean(a(cn))),cnn_sd=float(np.std(a(cn))),cnn_min=float(np.min(a(cn))),cnn_max=float(np.max(a(cn))),geom=float(np.mean(a(ge))),geom_sd=float(np.std(a(ge))),dino=float(np.mean(a(di))),
                       d=float(np.mean(a(cn))-np.mean(a(ge))),ci=ci(bs),p=pv(bs),runs=len(runs),updates=float(np.mean(upd)),upd_min=int(np.min(upd)),upd_max=int(np.max(upd)),pos=float(np.mean(pos)),posp=float(np.mean(posp)),
                       by_part=[float(np.mean([roc_auc_score(y,x)-roc_auc_score(y,g) for (pp,dd),x,g in zip(runs,cn,ge) if pp==p_])) for p_ in range(3)])
            ps.append(LC[k]['p']); keys.append(k)
    for k,q in zip(keys,bh(ps)): LC[k]['p_bh']=float(q)
R['LC']=LC
for k,v in LC.items(): print(k, f"cnn {v['cnn']:.3f}±{v['cnn_sd']:.3f} geom {v['geom']:.3f} d {v['d']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] pbh {v['p_bh']:.3f} upd {v['updates']:.0f} pos {v['pos']:.0f}/{v['posp']:.0f} parts {[round(x,3) for x in v['by_part']]}")
# check partition 0 draws 0,1 matched vs original nb4
# ---- LOSO seeds
LO={}
for s in ['Ingenia','Avanto_fit','SymphonyTim','Aera']:
    LO[s]={}
    for tc in ['narrowing','bulging','pfirrmann_severe','modic_any']:
        y=c[tc].values; da=np.load(B+f'loso/descriptor_arms_{tc}_{s}.npz'); te=da['te']; yt=y[te]
        P=[np.load(B+f'loso/cnn_{tc}_{s}_seed{sd}.npz')['p'] for sd in (7,11,13)]
        aucs=[roc_auc_score(yt,p) for p in P]; g=da['geometry']; Bs=boots(3,2000,te)
        bs=np.array([np.mean([roc_auc_score(yt[b],p[b]) for p in P])-roc_auc_score(yt[b],g[b]) for b in Bs if 0<yt[b].sum()<len(b)])
        bc=[np.mean([roc_auc_score(yt[b],p[b]) for p in P]) for b in Bs if 0<yt[b].sum()<len(b)]
        LO[s][tc]=dict(cnn_seeds=[float(x) for x in aucs],cnn_mean=float(np.mean(aucs)),cnn_ci=ci(bc),geom=float(roc_auc_score(yt,g)),dino=float(roc_auc_score(yt,da['dinov2'])),fusion=float(roc_auc_score(yt,da['fusion'])),d=float(np.mean(aucs)-roc_auc_score(yt,g)),d_ci=ci(bs))
    print(s,{tc:(round(v['geom'],3),[round(x,3) for x in v['cnn_seeds']],round(v['d'],3),[round(x,3) for x in v['d_ci']]) for tc,v in LO[s].items()})
R['LOSO']=LO
# ---- sensitivity
oof=np.load(D+'nb3_oof_rep.npy',allow_pickle=True).item(); TT={'narrowing':'disc narrowing','bulging':'disc bulging','pfirrmann_severe':'Pfirrmann >= 4','modic_any':'Modic (any)'}
S={}; ps=[];keys=[]
for arch in ['effb0_aug_es','convnext_t_aug_es']:
    for tc,t in TT.items():
        y=c[tc].values; P=[np.load(B+f'sensitivity/{arch}_{tc}_rep{r}.npz')['p'] for r in range(3)]; G=[oof[(t,'geometry',r)][0] for r in range(3)]; C0=[np.load(D+f'nb3_cnn/{tc}_rep{r}.npz')['p'] for r in range(3)]
        info=[json.loads(str(np.load(B+f'sensitivity/{arch}_{tc}_rep{r}.npz')['info'])) for r in range(3)]
        f=lambda X,b=None:np.mean([roc_auc_score(y if b is None else y[b],x if b is None else x[b]) for x in X])
        bs=np.array([f(P,b)-f(G,b) for b in Bt]); bs2=np.array([f(P,b)-f(C0,b) for b in Bt])
        k=f'{arch}|{tc}'; S[k]=dict(auc=float(f(P)),reps=[float(roc_auc_score(y,x)) for x in P],geom=float(f(G)),cnn0=float(f(C0)),d_geom=float(f(P)-f(G)),ci_geom=ci(bs),p=pv(bs),d_cnn0=float(f(P)-f(C0)),ci_cnn0=ci(bs2),epochs=float(np.mean([i['best_epoch'] for ii in info for i in ii])))
        ps.append(S[k]['p']); keys.append(k)
for k,q in zip(keys,bh(ps)): S[k]['p_bh']=float(q)
for k,v in S.items(): print(k,{a:(round(b,3) if isinstance(b,float) else [round(x,3) for x in b]) for a,b in v.items()})
R['SENS']=S
json.dump(R,open('b8_results.json','w'),indent=1)
