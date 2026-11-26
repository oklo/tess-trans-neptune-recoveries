from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from p9search.tasks import fetch_bundle
from p9search.storage import DiskBudget,read_json,sha256
from p9search.archive import acquire
import sys, shutil
out=Path(__file__).resolve().parent
def geometry(plan):
    if not plan.get('task'):
        orig=Path(plan['data'])
        data=out/'cache'/plan['field']/'data';data.mkdir(parents=True,exist_ok=True)
        for file in orig.glob('observer*.npz'):shutil.copyfile(file,data/file.name)
        for sector in orig.glob('s[0-9]*'):
            dest=data/sector.name;dest.mkdir(exist_ok=True)
            for name in ('metadata.json','cadences.npz','geometry.npz','wcs.txt','native_wcs.npz','stars.npz'):
                shutil.copyfile(sector/name,dest/name)
        return data
    directory=out/'cache'/plan['field']
    directory.mkdir(exist_ok=True,parents=True)
    return fetch_bundle(read_json(plan['task']),directory,DiskBudget(directory,12*2**30,25*2**30))
def fetch(plan,workers=2):
    data=geometry(plan)
    def load(sid):
        directory=data/f's{sid:04d}'
        if not (directory/'raw.npy').exists():
            acquire(directory,DiskBudget(data,15*2**30,25*2**30),lambda *a,**k:None)
        for k in ('raw.npy','error.npy'):
            if plan.get('hashes'):expected=plan['hashes'][str(sid)][k]
            else:
                receipts=read_json(Path(plan['data'])/directory.name/'ready.json')['sha256']
                expected,=[v for path,v in receipts.items() if path.endswith('/'+directory.name+'/'+k)]
            assert sha256(directory/k)==expected
    sectors=sorted({int(s) for row in plan['rows'] for s in row['sectors'] if not (out/'measurements'/f"{plan['field']}-{row['trace']}-{s}.npz").exists()})
    with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(load,sectors))
    return data
if __name__=='__main__':print(fetch(read_json(out/'plans'/(sys.argv[1]+'.json'))))
