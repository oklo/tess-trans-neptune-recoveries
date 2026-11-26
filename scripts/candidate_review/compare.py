"""Compare fixed-center shapes with matched point sources added to nearby residual stamps."""
from pathlib import Path
import csv,json
import numpy as np
base=Path(__file__).parent
def correlation(images,template):
 t=template-template.mean()
 a=images-images.mean(axis=1,keepdims=True)
 return a@t/np.maximum(1e-20,np.linalg.norm(a,axis=1)*np.linalg.norm(t))
rows=[]
for p in sorted((base/'plans').glob('*.json')):
 j=json.loads(p.read_text())
 for r in j['rows']:
  for s in r['sectors']:
   path=base/'measurements'/f"{p.stem}-{r['trace']}-{s}.npz"
   if not path.exists():continue
   with np.load(path) as z:
    v,u=np.indices((25,25))-12;mask=u*u+v*v<=36
    w=z['weight'][:,mask];ok=np.all(w>0,axis=1)
    if not ok[0]:continue
    e=z['response'][mask]/w[0];y=z['numerator'][:,mask][ok]/w[ok]
    # Compare the nominal response of the same filtered point source against
    # backgrounds of roughly comparable weight in the same sector.
    use=np.median(w[ok]/w[0],axis=1);bg=y[1:][(use[1:]>.5)&(use[1:]<2)]
    a=correlation(y[:1],e)[0];b=correlation(bg+float(z['flux'])*e,e)
    rows.append(dict(field=p.stem,trace=r['trace'],sector=s,r=round(a,2),controls=len(b),lower_or_equal=int((b<=a).sum()),snr=round(float(z['flux']/z['sigma']),1)))
with (base/'comparison.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
for key in dict.fromkeys((x['field'],x['trace']) for x in rows):
 print(*key,' '.join(f"{x['sector']}:{x['r']}({x['lower_or_equal']}/{x['controls']})" for x in rows if (x['field'],x['trace'])==key))
