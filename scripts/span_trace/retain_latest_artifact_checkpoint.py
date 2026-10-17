"""Keep iteration 30 fixed; retain only the newest committed later checkpoint."""
import argparse,json,re,shutil,time
from pathlib import Path

def committed_checkpoint(root,iteration):
    path=root/f'iter_{iteration:07d}'
    if path.is_symlink() or not path.is_dir():return False
    if not all((path/name).is_file() and (path/name).stat().st_size>0 for name in ('common.pt','.metadata','metadata.json')):return False
    try:json.loads((path/'metadata.json').read_text())
    except (ValueError,OSError):return False
    shards=list(path.glob('*.distcp'))
    return len(shards)>=8 and all(p.stat().st_size>0 for p in shards)

def prune(root,protected=30):
    tracker=root/'latest_checkpointed_iteration.txt'
    try:latest=int(tracker.read_text().strip())
    except (OSError,ValueError):return []
    if latest<=protected or not committed_checkpoint(root,protected) or not committed_checkpoint(root,latest):return []
    removed=[]
    for path in sorted(root.glob('iter_*')):
        match=re.fullmatch(r'iter_(\d{7})',path.name)
        if not match or path.is_symlink():continue
        iteration=int(match[1])
        if not protected<iteration<latest:continue
        # A concurrent tracker change or rollback cancels this cleanup pass.
        if tracker.read_text().strip()!=str(latest):break
        shutil.rmtree(path)
        removed.append(path.name)
    return removed

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--once',action='store_true');args=parser.parse_args()
    root=args.root.resolve()
    while True:
        try:
            removed=prune(root)
            record={'time':time.time(),'protected_iteration':30,'removed':removed,'latest':(root/'latest_checkpointed_iteration.txt').read_text().strip()}
            (root/'checkpoint_retention_status.json').write_text(json.dumps(record,indent=2))
            if removed:print(json.dumps(record),flush=True)
            if args.once or (record['latest']=='39' and committed_checkpoint(root,39)):return
        except Exception as exc:print(f'Retention skipped: {exc!r}',flush=True)
        time.sleep(30)
if __name__=='__main__':main()
