"""Resume an exact atlas publication after a transport failure."""

import base64
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import requests
from p9search import USER_AGENT

SNAPSHOT = "coordinator/coordination/snapshot.json"
URL = "https://oklo.org/planet-nine/coordination/snapshot.json"
PUBLIC = "/home3/okloorg/public_html/planet-nine/coordination/snapshot.json"
PRIVATE = "/home3/okloorg/.planet-nine/deploy-20261105-v2"
SSH = [
    "ssh",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=15",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=2",
    "-o",
    "ControlMaster=no",
    "-o",
    "ControlPath=/private/tmp/p9-publication-%C",
    "oklo",
]


class PublicationRetry(RuntimeError):
    pass


class PublicationConflict(RuntimeError):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise PublicationConflict(message)


def read_public(commit):
    try:
        response = requests.get(
            URL,
            headers={"User-Agent": USER_AGENT},
            params={"verified": commit[:12], "check": time.time_ns()},
            timeout=(10, 30),
            allow_redirects=False,
        )
    except requests.exceptions.SSLError as error:
        raise PublicationConflict("Publication TLS verification failed") from error
    except (requests.ConnectionError, requests.Timeout) as error:
        raise PublicationRetry("Waiting for HTTPS publication confirmation") from error
    try:
        if response.status_code == 429 or response.status_code >= 500:
            raise PublicationRetry("Website temporarily unavailable for confirmation")
        require(response.status_code == 200, "Unexpected publication HTTP status")
        return response.content
    finally:
        response.close()


def confirm_git_publication(package, expected, previous):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=package, timeout=30)

    commit = git("rev-parse", "HEAD").decode().strip()
    require(
        git("rev-parse", "--abbrev-ref", "HEAD").strip() == b"main",
        "Publication branch differs",
    )
    status = git("status", "--porcelain")
    pending = bool(status)
    if pending:
        require(
            status in {f" M {SNAPSHOT}\n".encode(), f"M  {SNAPSHOT}\n".encode()}
            and git("show", f"{commit}:{SNAPSHOT}") == previous
            and (Path(package) / SNAPSHOT).read_bytes() == expected
            and git("show", f":{SNAPSHOT}") in {previous, expected}
            and not git("diff", "--summary")
            and not git("diff", "--cached", "--summary"),
            "Publication checkout changed; reconcile concurrent work",
        )
    else:
        require(
            git("show", f"{commit}:{SNAPSHOT}") == expected,
            "Publication checkout changed; reconcile concurrent work",
        )
    try:
        remote = git("ls-remote", "origin", "refs/heads/main").decode().split()
        require(bool(remote), "Missing GitHub publication branch")
        if pending:
            require(remote[0] == commit, "GitHub contains a different publication")
            try:
                git("add", "--", SNAPSHOT)
                git("commit", "-m", "Resume reviewed field publication")
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
                raise PublicationConflict("Could not record the verified publication") from error
            commit = git("rev-parse", "HEAD").decode().strip()
            require(
                not git("status", "--porcelain")
                and git("show", f"{commit}:{SNAPSHOT}") == expected,
                "Publication checkout changed while recording the snapshot",
            )
        if remote[0] != commit:
            parent = git("rev-parse", f"{commit}^").decode().strip()
            require(
                remote[0] == parent and git("show", f"{parent}:{SNAPSHOT}") == previous,
                "GitHub contains a different publication",
            )
            git("push", "origin", "main")
            remote = git("ls-remote", "origin", "refs/heads/main").decode().split()
            require(
                bool(remote) and remote[0] == commit,
                "GitHub changed during publication",
            )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
        raise PublicationRetry("Waiting to confirm the GitHub publication") from error
    return commit


PHP = r"""<?php
try {
    $p=json_decode(base64_decode('%s'),true,32,JSON_THROW_ON_ERROR);
    $incoming=base64_decode($p['data'],true);
    if($incoming===false || hash('sha256',$incoming)!==$p['expected']) throw new RuntimeException('Incoming snapshot differs');
    umask(0077);
    $lock=fopen($p['lock'],'c');
    if(!$lock || !flock($lock,LOCK_EX)) throw new RuntimeException('Publication lock failed');
    foreach([$p['public'],$p['backup'],$p['temporary'],$p['backup'].'.partial'] as $path) {
        if(is_link($path)) throw new RuntimeException('Publication path is redirected');
    }
    $current=hash_file('sha256',$p['public']);
    if($current===$p['expected']) { echo $current; exit(0); }
    if($current!==$p['previous']) throw new RuntimeException('Concurrent publication');
    function stage($path,$data,$mode) {
        if(file_exists($path)) {
            $partial=file_get_contents($path);
            if($partial===false || !str_starts_with($data,$partial)) throw new RuntimeException('Unrelated staged file');
        }
        $f=fopen($path,'c+b');
        if(!$f || !ftruncate($f,0)) throw new RuntimeException('Stage failed');
        $offset=0;
        while($offset<strlen($data)) {
            $n=fwrite($f,substr($data,$offset));
            if($n===false || $n===0) throw new RuntimeException('Short staged write');
            $offset+=$n;
        }
        if(!fflush($f) || !fsync($f)) throw new RuntimeException('Stage sync failed');
        fclose($f);
        if(!chmod($path,$mode) || hash_file('sha256',$path)!==hash('sha256',$data)) throw new RuntimeException('Stage verification failed');
    }
    if(file_exists($p['backup'])) {
        if(hash_file('sha256',$p['backup'])!==$p['previous']) throw new RuntimeException('Backup differs');
    } else {
        $old=file_get_contents($p['public']);
        if($old===false || hash('sha256',$old)!==$p['previous']) throw new RuntimeException('Previous snapshot changed');
        stage($p['backup'].'.partial',$old,0600);
        if(!rename($p['backup'].'.partial',$p['backup'])) throw new RuntimeException('Backup failed');
    }
    stage($p['temporary'],$incoming,0644);
    if(hash_file('sha256',$p['public'])!==$p['previous']) throw new RuntimeException('Concurrent publication');
    if(!rename($p['temporary'],$p['public'])) throw new RuntimeException('Publish failed');
    echo hash_file('sha256',$p['public']);
} catch(Throwable $e) {
    fwrite(STDERR,$e->getMessage());
    exit(73);
}
"""


def remote_script(data, previous, label, *, public=PUBLIC, private=PRIVATE):
    require(bool(re.fullmatch(r"[a-z0-9-]{1,80}", label)), "Invalid publication label")
    require(
        bool(re.fullmatch(r"[a-f0-9]{64}", previous)), "Invalid previous snapshot hash"
    )
    values = {
        "public": public,
        "backup": f"{private}/{label}-previous-snapshot.json",
        "temporary": f"{public}.{label}.partial",
        "lock": f"{private}/snapshot-publish.lock",
        "data": base64.b64encode(data).decode(),
        "previous": previous,
        "expected": digest(data),
    }
    return PHP % base64.b64encode(json.dumps(values).encode()).decode()


def publish_site(data, previous, label, commit):
    expected = digest(data)
    current = digest(read_public(commit))
    if current == expected:
        return
    require(current == previous, "Public snapshot differs from both recorded versions")
    try:
        actual = subprocess.check_output(
            SSH + ["/usr/local/bin/php"],
            input=remote_script(data, previous, label),
            text=True,
            timeout=90,
        ).strip()
    except subprocess.TimeoutExpired as error:
        raise PublicationRetry("Waiting to resume website publication") from error
    except subprocess.CalledProcessError as error:
        if error.returncode == 255:
            raise PublicationRetry("Waiting to resume website publication") from error
        raise PublicationConflict(
            "Website refused the conditional publication"
        ) from error
    require(actual == expected, "Website publication hash differs")
    current = digest(read_public(commit))
    if current == previous:
        raise PublicationRetry("Waiting for the published snapshot to reach HTTPS")
    require(current == expected, "Public snapshot differs from both recorded versions")
