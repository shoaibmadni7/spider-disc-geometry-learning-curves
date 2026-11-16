import pandas as pd,glob,sys
import extract as E
from joblib import Parallel, delayed
files=sorted(glob.glob('../data/masks/*.mha'))
for name,(k,a) in {'base':(None,0),'erode1':('erode',1),'dilate1':('dilate',1),'jitter50':('jitter',0.5)}.items():
    res=Parallel(n_jobs=2)(delayed(E.extract)(f,k,a,i) for i,f in enumerate(files))
    df=pd.DataFrame([r for rr in res for r in rr]); df.to_csv(f'P_{name}.csv',index=False); print(name,len(df),flush=True)
