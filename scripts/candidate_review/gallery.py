from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
base=Path(__file__).parent
out=base/'images';out.mkdir(exist_ok=True)
def paint(img,raw=False):
 a=np.nan_to_num(img)
 if raw:
  a=np.arcsinh((a-np.percentile(a,10))/max(1,np.percentile(a,70)-np.percentile(a,10)))
  z=np.clip(a/max(.001,np.percentile(a,99.7)),0,1)
  return Image.fromarray(np.uint8(255*z)).convert('RGB')
 lim=max(1e-12,np.percentile(np.abs(a),98))
 a=np.clip(a/lim,-1,1)
 return Image.fromarray(np.uint8(np.array([10,16,22])+np.maximum(a,0)[:,:,None]*np.array([235,163,82])+np.maximum(-a,0)[:,:,None]*np.array([82,173,220])))
for plan in sorted((base/'plans').glob('*.json')):
 j=json.loads(plan.read_text())
 for row in j['rows']:
  field=j['field'];trace=row['trace']
  dst=out/f'{field}-{trace}.png'
  paths=[base/'measurements'/f'{field}-{trace}-{s}.npz' for s in sorted(map(int,row['sectors']))]
  if not all(p.is_file() for p in paths):continue
  with np.load(paths[0]) as z:context='field_image' in z
  im=Image.new('RGB',(575 if context else 304,36+len(paths)*137),(16,20,24));d=ImageDraw.Draw(im)
  d.text((6,4),f'{field} · trace {trace}');d.text((43,21),'Measured');d.text((179,21),'Point model')
  if context:d.text((314,21),'Star field + path');d.text((448,21),'Scene residual + path')
  for k,p in enumerate(paths):
   with np.load(p) as z:
    sid=p.stem.split('-')[-1];y=36+k*137;d.text((1,y+57),'S'+sid)
    w=z['weight'][0]
    for x,a in [(42,z['numerator'][0]),(177,z['response'])]:
     im.paste(paint(np.divide(a,w,out=np.zeros_like(w),where=w>0)).resize((125,125)),(x,y))
    if context:
     for x,key in [(312,'field_image'),(447,'field_residual')]:
      if key not in z:
       d.text((x,y+50),'Not archived',fill=(175,175,175));continue
      im.paste(paint(z[key],key=='field_image').resize((123,123)),(x,y))
      pts=[(x+(a+.5)*3,y+(b+.5)*3) for a,b in z['path']]
      d.line(pts,fill=(249,170,82),width=1)
  im.save(dst)
