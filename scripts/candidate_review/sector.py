"""Reconstruct the joint-figure ingredients and the point response of its fitted filters."""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from p9search import foreground, runtime
from p9search.science.geometry import SectorGeometry, SkyFrame
from p9search.science.templates import bin_template
from p9search.storage import read_json

out = Path(__file__).resolve().parent
axis = np.arange(-6., 6.01, .5)
y, x = np.meshgrid(axis, axis, indexing="ij")
grid = np.stack([x, y], axis=-1)
offsets = [(0,0)] + [(dx,dy) for dx in (-14,-7,0,7,14) for dy in (-14,-7,0,7,14) if dx or dy]


def response(model, template, region, bounds):
    scene, temporal = model["scene"], model["temporal"]
    u, v = scene.time_weight, scene.pixel_weight
    nt = len(u)
    basis = scene.basis.reshape(*v.shape,-1)
    t = template.reshape(nt,-1).astype(float)
    centered = t - (u@t/u.sum())
    coeff = (centered * v[region[1:]].ravel()) @ basis[region[1:]].reshape(t.shape[1],-1)
    x0,y0,x1,y1 = bounds
    crop = np.s_[y0:y1,x0:x1]
    arr = -(coeff @ basis[crop].reshape(-1,scene.rank).T).reshape(nt,y1-y0,x1-x0)
    sy,sx = region[1:]
    yy=slice(max(y0,sy.start),min(y1,sy.stop));xx=slice(max(x0,sx.start),min(x1,sx.stop))
    arr[:,yy.start-y0:yy.stop-y0,xx.start-x0:xx.stop-x0]+=centered.reshape(template.shape)[:,yy.start-sy.start:yy.stop-sy.start,xx.start-sx.start:xx.stop-sx.start]
    w = temporal.weight[:,crop[0],crop[1]].reshape(nt,-1)
    a = arr.reshape(nt,-1)
    fold=temporal.fold[crop].ravel()
    inv=temporal.inverse_gram[crop].reshape(len(fold),*temporal.inverse_gram.shape[-2:])
    for group,b in enumerate(temporal.bases):
        ids=np.flatnonzero(fold==group)
        rhs=(b.T@(w[:,ids]*a[:,ids])).T
        a[:,ids]-=b@np.einsum('pij,pj->ip',inv[ids],rhs)
    return (a*w).reshape(arr.shape)


def measure(g,ref,row,model):
    obs=model['temporal'];pars=row['parameters']
    xy=g.exact_positions(pars,native=True,kepler=True)
    t,r=bin_template(xy,g.meta,g.cadences)
    score=model['projector'].score(t,r)
    original=row['sectors'][str(g.meta['sector'])]
    assert abs(score['flux']-original['flux'])<.03*original['sigma']
    assert abs(score['sigma']/original['sigma']-1)<.005
    track=g.exact_positions(pars,native=False,kepler=True)
    keep=np.flatnonzero(g.geom['keep']&(model['scene'].time_weight>0))
    transform=g.jacobian@np.linalg.inv(ref.jacobian)
    sample=(grid[None]+np.asarray(offsets)[:,None,None,:])@transform.T
    lo=np.maximum(0,np.floor(track[keep].min(axis=0)+sample[0].min(axis=(0,1))-1).astype(int))
    hi=np.minimum(140,np.ceil(track[keep].max(axis=0)+sample[0].max(axis=(0,1))+2).astype(int))
    prf=response(model,t,r,(*lo,*hi))
    num=np.zeros(sample.shape[:-1]);weight=np.zeros_like(num);expected=np.zeros(grid.shape[:2])
    for i in keep:
        pts=track[i]+sample
        coords=pts[...,::-1].transpose(3,0,1,2)
        for dest,cube in [(num,obs.weighted_residual),(weight,obs.weight)]:
            dest+=map_coordinates(cube[i],coords,order=1,mode='constant',cval=0,prefilter=False)
        expected+=map_coordinates(prf[i],coords[:,0]-lo[::-1,None,None],order=1,mode='constant',cval=0,prefilter=False)
    return dict(numerator=num,weight=weight,response=expected,flux=score['flux'],sigma=score['sigma'],offsets=offsets,parameters=pars)


if __name__=="__main__":
    plan=read_json(sys.argv[1]);sid=int(sys.argv[2]);data=Path(sys.argv[3])
    runtime.configure('mps',threads=2,max_memory_bytes=5*2**30)
    frame=SkyFrame(data/f"s{plan['reference_sector']:04d}")
    ref=SectorGeometry(frame_dir:=data/f"s{plan['reference_sector']:04d}",frame)
    sector=SectorGeometry(data/f's{sid:04d}',frame)
    model=foreground.fit(sector,empirical_stars=sid in plan['emp'])
    for row in plan['rows']:
        path=out/'measurements'/f"{plan['field']}-{row['trace']}-{sid}.npz"
        if str(sid) not in row['sectors'] or path.exists():continue
        result=measure(sector,ref,row,model)
        xy=sector.exact_positions(row['parameters'],native=False,kepler=True)
        keep=np.flatnonzero(sector.geom['keep'])
        at=keep[np.linspace(0,len(keep)-1,48,dtype=int)]
        mid=np.rint(np.mean(xy[keep],axis=0)).astype(int)
        lo=np.clip(mid-20,0,99)
        cut=np.s_[lo[1]:lo[1]+41,lo[0]:lo[0]+41]
        raw=np.load(sector.directory/'raw.npy',mmap_mode='r')
        result.update(field_image=np.nanmedian(raw[(at,*cut)],axis=0),path=xy[at]-lo)
        spatial=model['scene'].residual.reshape(-1,140,140)
        result['field_residual']=np.nanmedian(spatial[(at,*cut)],axis=0)

        tmp=path.with_suffix('.partial.npz')
        np.savez_compressed(tmp,**result);tmp.replace(path)
        print(path.name,flush=True)
    if plan['field']=='N2215131':
        import gc
        from p9search.injections import inject
        row=next(r for r in plan['rows'] if r['trace']==1)
        path=out/'measurements'/f"N2215131-1-{sid}.npz"
        with np.load(path) as a: original=dict(a)
        pars=row['parameters'];xy=sector.exact_positions(pars,native=True,kepler=True)
        flux=2.;weights=model['weights']
        del model
        gc.collect();runtime.release()
        trial=foreground.fit(sector,raw_override=inject(sector,xy,flux),fixed_weights=weights,empirical_stars=sid in plan['emp'])
        g=sector.exact_positions(pars,native=False,kepler=True)
        projected=grid@(sector.jacobian@np.linalg.inv(ref.jacobian)).T
        wsum=np.zeros(grid.shape[:2]);total=np.zeros_like(wsum)
        temporal=trial['temporal']
        for it in np.flatnonzero(sector.geom['keep']&(trial['scene'].time_weight>0)):
            coords=(g[it]+projected)[...,::-1].transpose(2,0,1)
            for dst,cube in [(wsum,temporal.weight),(total,temporal.weighted_residual)]:
                dst+=map_coordinates(cube[it],coords,order=1,mode='constant',cval=0,prefilter=False)
        t,r=bin_template(xy,sector.meta,sector.cadences)
        gain=(trial['projector'].score(t,r)['flux']-original['flux'])/flux
        np.savez_compressed(out/f'point-test-{sid}.npz',numerator=total,weight=wsum,gain=gain,added_flux=flux)
