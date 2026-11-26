from pathlib import Path
from datetime import datetime,timezone
import time,sys,fcntl,shutil
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'scripts'))
import publish_travel_reviews as t
from p9search.storage import read_json as read,write_json as save,sha256 as sha
batch=root/'outputs/planet_nine_travel_publication_20261120'
checkout=Path(read(batch/'config.json')['checkout'])
site=checkout/t.publication.SNAPSHOT

lock=(batch/'lock').open('a')
while True:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if (batch/'hold.json').exists(): raise RuntimeError('Operator publication hold')
 current=read(batch/'head.json')
 frozen=any(int(p.parent.name[7:])>current['result_id'] for p in batch.glob('result-*/credit-plan.json'))
 calibration=batch.parent/'planet_nine_calibration_publication'/current['snapshot_sha256']/'plan.json'
 out=batch.parent/'planet_nine_neptune_cut'/current['snapshot_sha256']
 if (out/'publish-plan.json').exists() or (not frozen and not calibration.exists() and sha(site)==current['snapshot_sha256']):break
 fcntl.flock(lock,fcntl.LOCK_UN)
 time.sleep(8)
with lock:
 head=read(batch/'head.json')
 journal=out/'publish-plan.json'
 if not journal.exists():
  snap=read(site)
  estimates=read(batch.parent/'planet_nine_atlas_candidates/orbits.json')
  rows=[]
  for field in snap['fields']:
   if field['state']!='needs_followup':continue
   code=field['code'];tid=field.get('task_id')
   base=batch.parent/'planet_nine_backlog_service'/'visual'/str(tid)
   plan=base/'plans'/f'{code}.json'
   if not plan.exists():plan=batch.parent/'planet_nine_visual_followup_20261123'/'plans'/f'{code}.json'
   if not plan.exists() or code not in estimates:continue
   import hashlib
   digest=hashlib.sha256(plan.read_bytes()+(root/'scripts/atlas_trial_orbits.py').read_bytes()).hexdigest()
   if estimates[code]['digest']!=digest:continue
   source=read(plan)['source']
   if tid:assert source=='coordinated task '+tid
   for trace in field.get('retained_candidate_flags',range(1,field['candidate_count']+1)):
    orbit=estimates[code]['rows'].get(str(trace),{})
    radius=orbit.get('radius')
    if radius is None:continue
    over=radius>17.15/3
    rows.append(dict(field=code,trace=trace,source=source,disposition='anomaly' if over else 'retain',
                     discrepancy=dict(radius_earth=radius,trial_distance_au=orbit['distance'],T=orbit['T'],
                     geometric_albedo=.4,V_minus_T=.5,maximum_mass_earth=17.15,
                     maximum_catalog_radius_earth=17.15/3,plan_digest=digest)))
  if not any(x['disposition']=='anomaly' for x in rows):sys.exit(0)
  t.sync_checkout(checkout,head)
  assert sha(site)==head['snapshot_sha256']
  out.mkdir(exist_ok=True,parents=True)
  save(out/'assessment.json',rows)
  prev=Path(head['ledger'])
  t.verify_tree(prev)
  assert sha(prev/'summary.json')==head['ledger_sha256']
  ledger=read(prev/'summary.json')
  before=snap['physical_followup_flags']
  assert sum(len(e['candidate_flags']) for e in ledger['retained_followup_searches'])==before
  utc=datetime.now(timezone.utc).isoformat()
  removed=0
  for code in sorted({x['field'] for x in rows if x['disposition']=='anomaly'}):
   f,=[f for f in snap['fields'] if f['code']==code]
   src='coordinated task '+f['task_id'] if f.get('task_id') else 'outputs/planet_nine_neighbor_'+{'N16':'sixteen','N38':'thirty_eight'}[code]+'_run_2026_11_04'
   assert all(x['source']==src for x in rows if x['field']==code)
   e,=[e for e in ledger['retained_followup_searches'] if e['source_search']==src]
   entries=[x for x in rows if x['field']==code and x['disposition']!='duplicate']
   ids=e['candidate_flags']
   assert f['state']=='needs_followup' and set(x['trace'] for x in entries)<=set(ids) and len(ids)==f['physical_followup_flags']
   aliases={str(x['trace']):x['discrepancy'] for x in entries if x['disposition']=='anomaly'}
   retained=[i for i in ids if str(i) not in aliases]
   revision=dict(utc=utc,policy='neptune-catalog-mass-ceiling-v1',previous_flags=ids,retained_flags=retained,discrepancy=aliases,sensitivity_credit=0)
   e.setdefault('historical_candidate_flags',ids)
   e.setdefault('candidate_revisions',[]).append(revision)
   e['candidate_flags']=retained
   f.setdefault('historical_flag_count',len(ids))
   f.update(retained_candidate_flags=retained,candidate_count=len(retained),physical_followup_flags=len(retained),above_neptune_radius_at_trial_distance=aliases,result=f"{len(aliases)} traces archived above the Neptune-mass limit at the trial distance and assumed albedo. {len(retained)} remain for follow-up.")
   if not retained:
    f['state']='calibration_pending'
    f['depth_note']='All candidate traces resolved; no open follow-up. Recovery calibration pending.'
   removed+=len(aliases)
  from collections import Counter
  snap['field_counts']=dict(Counter(f['state'] for f in snap['fields']))
  snap.update(physical_followup_flags=before-removed,updated_utc=utc)
  assert sum((f.get('physical_followup_flags') or 0) for f in snap['fields'])==before-removed==sum(len(e['candidate_flags']) for e in ledger['retained_followup_searches'])
  ledger['unresolved_flags_in_other_fields']=before-removed
  assert removed==sum(x["disposition"]=="anomaly" for x in rows)
  assert ledger['cumulative_expected_planets_proxy']==snap['sensitivity_proxy']['value']
  for path in [out/'assessment.json',prev/'summary.json']:
   ledger['sha256'][str(path)]=sha(path)
  new=out/'ledger';shutil.copytree(prev,new,dirs_exist_ok=True)
  save(new/'summary.json',ledger)
  freeze=read(prev/'result_freeze.json');freeze['sha256']['summary.json']=sha(new/'summary.json')
  save(new/'result_freeze.json',freeze);t.verify_tree(new)
  save(out/'snapshot.json',snap);(out/'previous.json').write_bytes(site.read_bytes())
  save(journal,dict(head=head,removed=removed,before=before,after=before-removed,sha=sha(out/'snapshot.json')))
 plan=read(journal);assert plan['head']==head
 expected=(out/'snapshot.json').read_bytes();previous=(out/'previous.json').read_bytes()
 assert t.publication.digest(expected)==plan['sha'] and t.publication.digest(previous)==head['snapshot_sha256']
 assert site.read_bytes() in [previous,expected]
 if site.read_bytes()==previous:site.write_bytes(expected)
 while True:
  try:
   commit=t.reviewed_git_publication(checkout,expected,previous)
   t.publication.publish_site(expected,head['snapshot_sha256'],'neptune-cut-'+plan['sha'][:12],commit)
   break
  except t.publication.PublicationRetry:
   time.sleep(15)
 head.update(ledger=str(out/'ledger'),ledger_sha256=sha(out/'ledger/summary.json'),snapshot_sha256=plan['sha'])
 save(batch/'head.json',head,private=True);save(out/'deployed.json',dict(commit=commit,**plan))
 print('Verified and deployed',plan['before'],plan['after'],commit)
