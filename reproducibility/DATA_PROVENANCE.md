# Data Provenance (public summary)

## Simulated benchmark (Contig_500k)
- Task: influenza A virus -> microbe HGT detection, contig-level binary classification
- Splits: 400,000 / 50,000 / 50,000 contigs (train/val/test), balanced by construction; construction seed 42
- Genome isolation: training and test share no microbial recipient genome (9,242 training vs 1,155 test genomes)
- Sliding windows (3 kb, 300-bp overlap): 940,028 / 199,438 / 199,707
  - training windows: breakpoint-centered sampling, negatives capped at 2 windows/contig
  - validation/test windows: full stride coverage capped at 10/contig
- Positives: length-stratified viral fragments (20-2,341 bp), substitution-model divergence (JC/K2P), GC amelioration, 3-bp generator-encoded junction features
- Difficult negatives: 10 categories (see manuscript Extended Data Table 4)
- Realism perturbations: sequencing errors, log-normal coverage, batch effects, contamination, assembly artifacts, label noise (2%, one-direction), breakpoint jitter, open-set donors

## External anchors (zero-shot)
| Anchor | n | Construction |
|---|---|---|
| Benbow | 566 (413/153) | literature-curated HGT test set |
| IslandViewer4 | 600 (300/300) | curated GI database; negatives single 3-kb windows |
| ICEberg3 | 414 (207/207) | integrative/conjugative elements; length-matched negatives |
| Philympics | 104 (53/51) | paired prophage events, 25 species |

## Cross-task datasets
Five external datasets (human-versus-worm, two promoter, two 4mC) under one protocol; sources cited in the manuscript reference list.

## Availability
Simulated benchmark generation is reproducible from this code tree; anchor sequences are available from their original sources (cited in the manuscript). Deposited evidence records (per-claim artifacts) accompany the data deposition.
