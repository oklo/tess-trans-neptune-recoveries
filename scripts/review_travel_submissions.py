"""Review fresh public submissions on the traveling Mac, one result at a time."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import argparse

from p9search.diagnostics import BoundedOutput, capture_output

from p9search.provenance import source_hash
from p9search.review import verify_signed
from p9search.storage import write_json
from p9search.tasks import validate_task

ROOT = None
KEY = None
RELEASE = None
INTERPRETERS = {}
APPROVED = []


def configure(batch):
    global ROOT, KEY, RELEASE, INTERPRETERS, APPROVED
    ROOT = batch.resolve()
    KEY = ROOT.parent / 'planet_nine_coordinator_deploy_2026_11_05/private/review.key'
    config = read(ROOT / 'runtime.json')
    RELEASE = config['source_sha256']
    APPROVED = config['approved_sources']
    INTERPRETERS = {k: Path(v) for k, v in config['interpreters'].items()}
    assert source_hash() == RELEASE and KEY.is_file() and not KEY.stat().st_mode & 0o077


REMOTE = '/home3/okloorg'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', 'oklo']
ADMIN = ['env', 'P9_REVIEW_KEY='+REMOTE+'/.planet-nine/review.key', '/usr/local/bin/php',
         REMOTE+'/public_html/planet-nine/admin.php']
INBOX = """<?php
require 'common.php';
$db=database();
$stmt=query($db, "SELECT r.id,r.submitted,r.digest,r.worker_id,r.task_id,t.spec FROM results r
 JOIN tasks t ON t.id=r.task_id JOIN workers w ON w.id=r.worker_id
 WHERE r.status='pending' AND t.state='submitted' AND w.status='approved'
 ORDER BY r.submitted,r.id");
echo json_encode($stmt->fetchAll(PDO::FETCH_ASSOC),JSON_THROW_ON_ERROR);
"""

def remote_php(code):
    return json.loads(subprocess.check_output(SSH+['cd ~/public_html/planet-nine && php'],input=code,text=True,timeout=45))

def read(path):
    return json.loads(path.read_bytes())

def save(path,value):
    write_json(path,value,private=True)

def review(row):
    num=int(row['id'])
    field=row['task_id']
    assert num>0 and re.fullmatch('[0-9a-f]{64}',field)
    task=validate_task(json.loads(row['spec']))
    assert task['task_id']==field
    d=ROOT/'results'/str(num)
    d.mkdir(mode=0o700,parents=True,exist_ok=True)
    receipt=d/'receipt.json'
    if (d/'hold.json').exists():return
    plan_path=d/'plan.json'
    if not receipt.exists():
        if shutil.disk_usage(ROOT).free < 20.25*2**30:
            return
        payload=remote_php("""<?php require 'common.php'; echo json_encode(query(database(),
          "SELECT result FROM results WHERE id=? AND status='pending'",[%d])->fetchColumn(),JSON_THROW_ON_ERROR);"""%num)
        if not isinstance(payload,str):return
        assert hashlib.sha256(payload.encode()).hexdigest()==row['digest']
        report=json.loads(payload)
        release = report.get('source_sha256')
        if release not in APPROVED:
            save(d/'hold.json',{'reason':'Review requires a separately validated scientific release'})
            return
        save(d/'task.json',task)
        submission=dict(result_id=num,submitted=int(row['submitted']),worker_id=row['worker_id'],
                        task_id=field,submission_digest=row['digest'],result_json=payload)
        save(d/'submission.json',submission)
        plan=json.loads(subprocess.check_output(SSH+[shlex.join(ADMIN+['review-plan',field])],text=True,timeout=40))
        decoded=verify_signed(plan,bytes.fromhex(KEY.read_text().strip()))
        assert decoded['submission_digest']==row['digest'] and decoded['worker_id']==row['worker_id']
        save(plan_path,plan)
        device = report['device'] if decoded['mode'] == 'full_replay' else 'cpu'
        if not device:
            save(d/'hold.json',{'reason':'Unsupported replay device','device':report['device']})
            return
        if decoded['mode']=='trusted_local':
            save(d/'hold.json',{'reason':'Participant submissions must use ordinary review'})
            return
        with tempfile.TemporaryDirectory(dir=d,prefix='work-') as temp:
            audit=Path(temp)/'audit'
            interpreter = INTERPRETERS[release] if decoded['mode'] == 'full_replay' else Path(sys.executable)
            cmd=[str(interpreter),'-I','-m','p9search','review',str(d/'task.json'),str(d/'submission.json'),
                 '--plan',str(plan_path),'--directory',str(audit),'--key',str(KEY),
                 '--approved-source',release,'--device',device,'--threads','2',
                 '--max-memory-gb','6','--max-disk-gb','8','--reserve-gb','20','--max-hours','6']
            with BoundedOutput(d/'review.log') as log:
                rc=subprocess.run(cmd,stdout=log.fileno(),stderr=subprocess.STDOUT).returncode
            for name in ['review-launch.json','replay.log','reviewed-result.json']:
                if (audit/name).is_file():shutil.copyfile(audit/name,d/name)
            if rc:
                save(d/'hold.json',{'reason':'Reviewer exited without verdict','exit':rc})
                return
            verified=verify_signed(read(audit/'receipt.json'),bytes.fromhex(KEY.read_text().strip()))
            assert verified['submission_digest']==row['digest'] and verified['plan_signature']==plan['signature']
            save(receipt,read(audit/'receipt.json'))
    # The coordinator rechecks identity, currency, signature, plan and review method.
    stage=REMOTE+f'/.planet-nine/travel-receipt-{num}.json'
    subprocess.run(['scp','-q','-o','BatchMode=yes',str(receipt),'oklo:'+stage],check=True,timeout=60)
    subprocess.run(SSH+[shlex.join(ADMIN+['accept',stage])],check=True,timeout=60)
    save(d/'posted.json',{'id':num,'task_id':field,'posted_epoch':time.time(),
                          'disposition':verify_signed(read(receipt),bytes.fromhex(KEY.read_text().strip()))['disposition']})

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    configure(args.batch)
    with capture_output(ROOT/'service.log'):
        while True:
            try:
                if not (ROOT/'STOP').exists():
                    rows = remote_php(INBOX)
                    save(ROOT/'status.json', {'checked_epoch': time.time(), 'pending': len(rows)})
                    for row in rows:
                        try:
                            review(row)
                        except Exception as error:
                            save(ROOT/'last-error.json', {'epoch': time.time(), 'result_id': row['id'], 'reason': str(error)[:500]})
            except Exception as error:
                save(ROOT/'last-error.json', {'epoch': time.time(), 'reason': str(error)[:500]})
            time.sleep(60)


if __name__ == '__main__':
    main()
