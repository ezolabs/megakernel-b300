import os,re,sys,time,statistics as s
# usage: acc.py START END  (HH:MM:SS, same day) -> mean accept len over TP0 decode batches in the copy's serve log
# SERVE_LOG (battery.sh sets it: /workspace/sglang-serve-a.log | -b.log; default the legacy log) and DAY (YYYY-MM-DD,
# default today in UTC: battery.sh takes START/END with `date -u` as window 1 did) select the lines; window 1 hard-coded both.
log=os.environ.get("SERVE_LOG","/workspace/sglang-serve.log"); day=os.environ.get("DAY") or time.strftime("%Y-%m-%d",time.gmtime())
st,en=sys.argv[1],sys.argv[2]; a=[]
for l in open(log,errors="ignore"):
    if "TP0" not in l or "accept len" not in l: continue
    m=re.search(re.escape(day)+r" (\d\d:\d\d:\d\d).*accept len: ([\d.]+)",l)
    if m and st<=m.group(1)<=en: a.append(float(m.group(2)))
print(f"{st}-{en} n={len(a)} mean_accept={s.mean(a):.3f} med={s.median(a):.3f}" if a else "none")
