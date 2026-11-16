import re,sys,json,os,time
src=open('exp.py').read(); parts=re.split(r'\n(?=# ---------- E\d)',src)
ns={'__name__':'exp'}; exec(compile(parts[0],'head','exec'),ns)
for p in parts[1:]:
    name=re.match(r'# ---------- (E\d)',p).group(1)
    if name!='E0' and name in ns['R']: print('skip',name,flush=True); continue
    t=time.time(); exec(compile(p,name,'exec'),ns); ns['save'](); print('DONE',name,round(time.time()-t),'s',flush=True)
print('ALL DONE',flush=True)
