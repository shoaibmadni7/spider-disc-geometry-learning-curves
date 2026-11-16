"""Geometry extractor for the IJIES study (re-implementation from the Methods text).
Orientation-free: uses array axes and voxel spacing only."""
import numpy as np, SimpleITK as sitk, re, sys, os, glob
from scipy import ndimage
CANAL=100

def perturb_mask(M, kind, amt, rng):
    if kind is None: return M
    out=M.copy()
    objs=ndimage.find_objects(M)
    for l,sl in enumerate(objs, start=1):
        if sl is None: continue
        sl=tuple(slice(max(s.start-3,0), min(s.stop+3,M.shape[i])) for i,s in enumerate(sl))
        b=(M[sl]==l); o=out[sl]
        st=ndimage.generate_binary_structure(3,1).copy(); ax=int(np.argmin(M.shape)); ix=[slice(None)]*3
        for k in (0,2):
            ix[ax]=k; st[tuple(ix)]=False
        if kind=='dilate':
            g=ndimage.binary_dilation(b,st,iterations=int(amt))&(o==0); o[g]=l
        elif kind=='erode':
            e=ndimage.binary_erosion(b,st,iterations=int(amt)); o[b&~e]=0
        elif kind=='jitter':
            grow=ndimage.binary_dilation(b,st)&(o==0); shrink=b&~ndimage.binary_erosion(b,st)
            o[grow&(rng.random(b.shape)<amt)]=l
            o[shrink&(rng.random(b.shape)<amt)]=0
    return out

PD=(1,99); PB=(2.5,97.5); PA=(1,99); SLIPH=False
def extract(path, kind=None, amt=0, seed=0, inplane_only=True):
    msk=sitk.ReadImage(path)
    M=sitk.GetArrayFromImage(msk).astype(np.int32)
    sp=np.array(msk.GetSpacing())[::-1].astype(float)   # array-axis order
    if kind is not None:
        M=perturb_mask(M,kind,amt,np.random.default_rng(seed))
    objs=ndimage.find_objects(M)
    def pts(l):
        if l>len(objs) or objs[l-1] is None: return None
        sl=objs[l-1]; idx=np.argwhere(M[sl]==l)
        if not len(idx): return None
        return (idx+np.array([s.start for s in sl]))*sp
    labs=[l for l in range(1,len(objs)+1) if objs[l-1] is not None]
    vert=[l for l in labs if 1<=l<=99]; discs=[l for l in labs if l>=201]
    if not vert or not discs: return []
    VP={l:pts(l) for l in vert}; VC={l:VP[l].mean(0) for l in vert}
    sl_ax=int(np.argmin(M.shape)); inpl=[a for a in range(3) if a!=sl_ax]
    if len(vert)>=3:
        r=[abs(np.corrcoef(vert,[VC[l][a] for l in vert])[0,1]) for a in inpl]
        si=inpl[int(np.argmax(r))]
    else: si=inpl[0]
    ap=[a for a in inpl if a!=si][0]
    sgn_si=np.sign(np.corrcoef(vert,[VC[l][si] for l in vert])[0,1]) if len(vert)>=2 else 1.0
    CP=pts(CANAL)
    allv=np.concatenate(list(VP.values()))
    sgn_ap=1.0
    if CP is not None: sgn_ap=np.sign(CP[:,ap].mean()-allv[:,ap].mean()) or 1.0
    e_si=np.zeros(3); e_si[si]=sgn_si; e_ap=np.zeros(3); e_ap[ap]=sgn_ap
    vox=float(np.prod(sp)); rows=[]
    for dl in discs:
        P=pts(dl)
        if P is None or len(P)<50: continue
        c=P.mean(0)
        _,_,V=np.linalg.svd(P-c,full_matrices=False)
        n=V[2]
        if n@e_si<0: n=-n
        u=e_ap-n*(n@e_ap); nu=np.linalg.norm(u)
        if nu<1e-6: continue
        u/=nu; w=np.cross(n,u)
        t=(P-c)@n; q1,q99=np.percentile(t,PD); h=float(q99-q1)
        ou=(P-c)@u; ow=(P-c)@w
        a1,a99=np.percentile(ou,PA); ap_ext=float(a99-a1)
        offs={l:float((VC[l]-c)@n) for l in vert}
        above=[l for l in vert if offs[l]>0]; below=[l for l in vert if offs[l]<0]
        if not above and not below: continue
        ls=min(above,key=lambda l:offs[l]) if above else None
        li=max(below,key=lambda l:offs[l]) if below else None
        adj=[l for l in (ls,li) if l is not None]
        alo,ahi=np.percentile(ou,[2,98]); llo,lhi=np.percentile(ow,[2,98])
        def fp(Q):
            qu,qw=(Q-c)@u,(Q-c)@w
            return (qu>=alo)&(qu<=ahi)&(qw>=llo)&(qw<=lhi)
        H={};Dp={};ok=True
        for l in adj:
            Q=VP[l]; f=fp(Q)
            if f.sum()<50: ok=False;break
            tn=(Q[f]-c)@n; b1,b99=np.percentile(tn,PB); H[l]=float(b99-b1)
            tu=(Q[f]-c)@u; d1,d99=np.percentile(tu,PB); Dp[l]=float(d99-d1)
        if not ok: continue
        s=float(np.mean([H[l] for l in adj]))
        canal=np.nan
        if CP is not None:
            th=max(h,4.0); sel=np.abs((CP-c)@n)<=th/2
            if sel.sum()>10: canal=sel.sum()*vox/th
        slip_mm=slip_pct=np.nan
        if len(adj)==2:
            post={}
            for l in adj:
                Q=VP[l]; near=(np.abs((Q-c)@n)<=(8.0+h/2 if SLIPH else 8.0))&fp(Q)
                post[l]=np.percentile((Q[near]-c)@u,95) if near.sum()>20 else np.nan
            slip_mm=post[ls]-post[li]
            if Dp[li]>0: slip_pct=100*slip_mm/Dp[li]
        rows.append(dict(level=dl-200,n_adj=len(adj),disc_h=h,body_h=s,ap_ext=ap_ext,vol=len(P)*vox,
            canal_area=canal,slip_mm=slip_mm,slip_pct=slip_pct,
            angle=float(np.degrees(np.arccos(np.clip(abs(n@e_si),0,1)))),n_vox=len(P)))
    fn=os.path.basename(path).replace('.mha','')
    pat=int(re.match(r'(\d+)',fn).group(1)); seq=fn.split('_',1)[1]
    for r in rows: r.update(patient=pat,series=seq)
    return rows

if __name__=='__main__':
    import pandas as pd
    from joblib import Parallel, delayed
    kind=sys.argv[1] if len(sys.argv)>1 and sys.argv[1]!='none' else None
    amt=float(sys.argv[2]) if len(sys.argv)>2 else 0
    out=sys.argv[3] if len(sys.argv)>3 else 'feat_raw.csv'
    files=sorted(glob.glob('../data/masks/*.mha'))
    res=Parallel(n_jobs=-1)(delayed(extract)(f,kind,amt,i) for i,f in enumerate(files))
    df=pd.DataFrame([r for rr in res for r in rr]); df.to_csv(out,index=False); print(len(files),len(df))
