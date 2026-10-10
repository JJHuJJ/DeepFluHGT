#!/usr/bin/env python3
"""homology_summary.py — Summarize minimap2 PAF to per-genome coverage.

Canonical implementation extracted from scripts/run/cross_split_homology_audit.sh.
See that script for the full pipeline (minimap2 invocation + this summary).
"""

# === Block 1 from cross_split_homology_audit.sh ===
import json, os
import os
ROOT=os.environ.get("DEEPFLUHGT_ROOT", ".")
MG=os.environ.get("DEEPFLUHGT_GENOMES", "microbe_genomes")
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

# === Block 2 from cross_split_homology_audit.sh ===
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

# Provenance note (R1-TierB 2026-10-04): shipped verbatim from the canonical
# cross-split homology audit chain (scripts/run/cross_split_homology_audit.sh +
# this summary). Paths inside are environment-specific (host archive layout);
# inputs are the split manifest and minimap2 PAF output; claim: 22.3% of test
# genomes with a train near-twin (>=95%-identity coverage >50%).
