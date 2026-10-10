#!/bin/bash
# cross_split_homology_audit.sh  —  P0-A1 pre-submission risk gating audit (user PI final ruling 2026-09-19)
# Goal: quantify cross-split homology between train (9,242) and test (1,155) recipient genomes (R3-B2 / Editor "run first" item)
# method: minimap2 asm10 (≤10% divergence), primary-only; per test genome statistics:
#       f_cov95/f_cov90 = fraction of query bases covered by ≥95%/≥90% identity alignments; best_id = highest alignment identity
set -euo pipefail
ROOT=${DEEPFLUHGT_ROOT:-.}
MG=${DEEPFLUHGT_GENOMES:-microbe_genomes}
OUT=$ROOT/docs/evidence/homology_audit
LOG=$ROOT/logs/cross_split_homology_$(date +%Y%m%d_%H%M).log
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
echo "== cross-split homology audit start $(date) =="

# 1. File lists
python3 - << 'PY'
import json, os
ROOT='${DEEPFLUHGT_ROOT:-.}'
MG='${DEEPFLUHGT_GENOMES:-microbe_genomes}'
m=json.load(open(f'{ROOT}/simulated_data/Contig_500k/genome_split_manifest.json'))
tr=m['microbe']['train']['files']; te=m['microbe']['test']['files']
assert len(set(tr)&set(te))==0
with open(f'{ROOT}/docs/evidence/homology_audit/train_list.txt','w') as f:
    f.write('\n'.join(os.path.join(MG,x) for x in tr if not x.endswith('.fai')))
with open(f'{ROOT}/docs/evidence/homology_audit/test_list.txt','w') as f:
    f.write('\n'.join(os.path.join(MG,x) for x in te if not x.endswith('.fai')))
import subprocess
print('train files:',sum(1 for _ in open(f'{ROOT}/docs/evidence/homology_audit/train_list.txt')))
print('test files:',sum(1 for _ in open(f'{ROOT}/docs/evidence/homology_audit/test_list.txt')))
PY

# 2. Merged pools (train ≈20G, test ≈2.5G)
cd "$OUT"
if [ ! -f train_pool.fna ]; then
  echo "== concatenating train pool =="
  xargs -a train_list.txt cat > train_pool.fna
fi
if [ ! -f test_pool.fna ]; then
  echo "== concatenating test pool =="
  xargs -a test_list.txt cat > test_pool.fna
fi
ls -lh train_pool.fna test_pool.fna

# 3. minimap2 asm10, primary-only
echo "== minimap2 mapping =="
minimap2 -t 64 -x asm10 --secondary=no -I 16G train_pool.fna test_pool.fna > cross_split.paf 2> mm2.log
ls -lh cross_split.paf

# 4. statistics
python3 - << 'PY'
import json, statistics as st
rows=[]
q={}
for line in open('cross_split.paf'):
    f=line.split('\t')
    qname,qlen,qstart,qend=f[0],int(f[1]),int(f[2]),int(f[3])
    nmatch,alnlen=int(f[9]),int(f[10])
    ident=nmatch/alnlen if alnlen else 0
    d=q.setdefault(qname,{'qlen':qlen,'cov95':0,'cov90':0,'best':0})
    d['best']=max(d['best'],ident)
    if ident>=0.95: d['cov95']+=qend-qstart
    if ident>=0.90: d['cov90']+=qend-qstart
out=[]
for name,d in q.items():
    L=d['qlen']
    out.append({'genome':name,'len':L,'f_cov95':min(1.0,d['cov95']/L),'f_cov90':min(1.0,d['cov90']/L),'best_identity':round(d['best'],4)})
n=len(out)
f95=sorted(x['f_cov95'] for x in out); f90=sorted(x['f_cov90'] for x in out); bi=sorted(x['best_identity'] for x in out)
summary={
 'n_test_genomes_with_alignments':n,
 'f_cov95_median':round(st.median(f95),4),'f_cov95_p90':round(f95[int(0.9*n)],4),'f_cov95_max':round(max(f95),4),
 'n_genomes_f95_gt_50pct':sum(1 for x in f95 if x>0.5),'n_genomes_f95_gt_10pct':sum(1 for x in f95 if x>0.1),'n_genomes_f95_gt_1pct':sum(1 for x in f95 if x>0.01),
 'best_identity_median':round(st.median(bi),4),'best_identity_p90':round(bi[int(0.9*n)],4),'best_identity_max':round(max(bi),4),
 'n_best_id_ge_95':sum(1 for x in bi if x>=0.95),'n_best_id_ge_90':sum(1 for x in bi if x>=0.90),'n_best_id_ge_85':sum(1 for x in bi if x>=0.85),
}
json.dump({'summary':summary,'per_genome':out},open('cross_split_homology.json','w'),indent=1)
print(json.dumps(summary,indent=1))
PY
echo "== audit done $(date) =="
