import pandas as pd,numpy as np,json
from common import *
R=json.load(open('results.json'))
raw=pd.read_csv('P_base.csv'); d=build(raw); N=len(d); pats=d.patient.values
FR=[0.05,0.10,0.25,0.50,1.0]; out={}
for t in ['narrowing','pfirrmann_severe']:
    y=d[t].values; res={f:[] for f in FR}
    for part in range(5):
        folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=100+part).split(d,y,d.patient))
        for o in range(5):
            P={f:np.zeros(N) for f in FR}
            for k,(tr,te) in enumerate(folds):
                tp=np.unique(pats[tr]); perm=np.random.default_rng(10000*o+100*part+k).permutation(tp)
                for f in FR:
                    keep=perm[:max(int(round(f*len(tp))),4)]; sub=tr[np.isin(pats[tr],keep)]
                    s=d.body_h.values; ks=[]
                    for ft in ['ap_ext','canal_area','vol']:
                        v=d[ft].values; ok=np.isfinite(v[sub])
                        ks.append(int(np.argmin([abs(np.corrcoef(v[sub][ok]/s[sub][ok]**kk,s[sub][ok])[0,1]) for kk in range(4)])))
                    P[f][te]=fit_predict(scale(d,ks),sub,te,t,minn=8,nanzero=True)
            for f in FR: res[f].append(roc_auc_score(y,P[f]))
    out[t]={str(f):[float(np.mean(res[f])),float(np.std(res[f]))] for f in FR}; print(t,out[t])
R['E6']=out
R['ECE']={t:float(sum(abs(a-b)*n for a,b,n in R['E0'][t]['calib'])/sum(n for a,b,n in R['E0'][t]['calib'])) for t in TARGETS}; print(R['ECE'])
json.dump(R,open('results.json','w'),indent=1)
