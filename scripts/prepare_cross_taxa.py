#!/usr/bin/env python3
"""Cross-taxa data preparation: extract taxonomy from FASTA headers → annotate the manifest → report phylum distribution.
Read each microbe genome's FASTA header → organism name → phylum mapping → per-split (train/val/test) distribution.

output:
  audit_output/taxonomy_manifest.json (per-genome phylum)
  audit_output/cross_taxa_summary.json (phylum × split distribution)

usage:
  python scripts/prepare_cross_taxa.py
  --manifest simulated_data/Contig_500k/genome_split_manifest.json
  --genome_dir ./microbe_genomes
  --out audit_output/cross_taxa_summary.json

usage (trainingafterevaluation):
  python scripts/audit_05_cross_taxa.py --taxonomy audit_output/taxonomy_manifest.json --model-predictions ...
"""
import os, sys, json, argparse, re
from pathlib import Path
from collections import Counter, defaultdict

# Common genus/family → phylum mapping (16S-free approximation)
# Covers the most common microbial phyla
PHYLUM_MAP = {
    # Firmicutes (Firmicutes + Firmicutes_A/B (GTDB)
    "firmicutes": [],  # populated at runtime
    "bacillus": "Firmicutes", "staphylococcus": "Firmicutes", "lactobacillus": "Firmicutes",
    "clostridium": "Firmicutes", "streptococcus": "Firmicutes", "enterococcus": "Firmicutes",
    "listeria": "Firmicutes", "lactococcus": "Firmicutes", "weissella": "Firmicutes",
    "leuconostoc": "Firmicutes", "pediococcus": "Firmicutes", "oceanobacillus": "Firmicutes",
    "geobacillus": "Firmicutes", "alkalicoccus": "Firmicutes", "furfurilactobacillus": "Firmicutes",
    # Proteobacteria
    "proteobacteria": [],
    "escherichia": "Proteobacteria", "pseudomonas": "Proteobacteria", "salmonella": "Proteobacteria",
    "shewanella": "Proteobacteria", "serratia": "Proteobacteria", "burkholderia": "Proteobacteria",
    "vibrio": "Proteobacteria", "rhizobium": "Proteobacteria", "cupriavidus": "Proteobacteria",
    "shigella": "Proteobacteria", "klebsiella": "Proteobacteria", "campylobacter": "Proteobacteria",
    "helicobacter": "Proteobacteria", "acinetobacter": "Proteobacteria", "zonitorhabdus": "Proteobacteria",
    "photobacterium": "Proteobacteria", "xenorhabdus": "Proteobacteria", "legionella": "Proteobacteria",
    # Actinobacteria
    "actinobacteria": [],
    "mycobacterium": "Actinobacteria", "streptomyces": "Actinobacteria", "corynebacterium": "Actinobacteria",
    "mycolicibacterium": "Actinobacteria", "nocardia": "Actinobacteria", "bifidobacterium": "Actinobacteria",
    "microbacterium": "Actinobacteria", "propionibacterium": "Actinobacteria", "arthrobacter": "Actinobacteria",
    "frankia": "Actinobacteria", "renibacterium": "Actinobacteria", "cutibacterium": "Actinobacteria",
    "mycobacteroides": "Actinobacteria",
    # Bacteroidetes
    "bacteroidetes": [],
    "bacteroides": "Bacteroidetes", "prevotella": "Bacteroidetes", "flavobacterium": "Bacteroidetes",
    "cytophaga": "Bacteroidetes", "porphyromonas": "Bacteroidetes", "gingivalis": "Bacteroidetes",
    # Cyanobacteria
    "cyanobacteria": [],
    "synechococcus": "Cyanobacteria", "prochlorococcus": "Cyanobacteria", "anabaena": "Cyanobacteria",
    "nostoc": "Cyanobacteria", "microcystis": "Cyanobacteria",
    # Spirochaetes
    "spirochaetes": [],
    "treponema": "Spirochaetes", "borrelia": "Spirochaetes", "leptospira": "Spirochaetes",
    # Tenericutes
    "tenericutes": [],
    "mycoplasma": "Tenericutes", "ureaplasma": "Tenericutes",
    # Fusobacteria
    "fusobacteria": [],
    "fusobacterium": "Fusobacteria",
    # Euryarchaeota (Archaea)
    "euryarchaeota": [],
    "methanococcus": "Euryarchaeota", "methanosarcina": "Euryarchaeota", "halobacterium": "Euryarchaeota",
    "methanocaldococcus": "Euryarchaeota", "methanobrevibacter": "Euryarchaeota",
    "methanosaeta": "Euryarchaeota",
    # Crenarchaeota
    "crenarchaeota": [],
    "sulfolobus": "Crenarchaeota", "thermoproteus": "Crenarchaeota", "pyrobaculum": "Crenarchaeota",
    # Eukaryota (fungi)
    "ascomycota": [],
    "aspergillus": "Ascomycota", "saccharomyces": "Ascomycota", "neurospora": "Ascomycota",
    "pyricularia": "Ascomycota", "cryptococcus": "Basidiomycota", "schizosaccharomyces": "Ascomycota",
    "candida": "Ascomycota", "pichia": "Ascomycota", "yarrowia": "Ascomycota",
    "kluyveromyces": "Ascomycota", "crymecococcus": "Basidiomycota",
    "cryptococcus": "Basidiomycota", "ustilago": "Basidiomycota", "rhodotorula": "Basidiomycota",
    "spizellomyces": "Chytridiomycota",
}


def get_phylum_from_name(organism_name: str) -> str:
    """Extract phylum-level taxonomy from the organism name (approximate, based on a genus → phylum mapping; GCF/ASM accessions cannot be looked up in NCBI taxonomy, so approximate from the genus name in the header."""
    name = organism_name.lower()
    words = name.split()
    if not words:
        return "Unknown"
    genus = words[0]
    # Direct match
    if genus in PHYLUM_MAP:
        result = PHYLUM_MAP[genus]
        if isinstance(result, str):
            return result
    # Partial match
    for key, phylum in PHYLUM_MAP.items():
        if isinstance(phylum, str) and genus in key:
            return phylum
        if isinstance(phylum, list) and genus in key:
            return key.capitalize()
    # fallback: kingdom-level inference
    if any(k in name for k in ["bacterium", "bacillus", "coccus"]):
        return "Bacteria_Unknown"
    if any(k in name for k in ["archaeon", "archaeum"]):
        return "Archaea_Unknown"
    if any(k in name for k in ["fungus", "yeast", "myces", "mycota"]):
        return "Eukarya_Unknown"
    return "Unknown"


def extract_organism_from_fasta(fasta_path: str) -> str:
    """Extract the organism name from a FASTA header."""
    try:
        with open(fasta_path) as f:
            line = f.readline().strip()
            if line.startswith(">"):
                # Format: >ACCESSION Organism name, description
                parts = line[1:].split(None, 1)
                if len(parts) > 1:
                    return parts[1].split(",")[0].strip()
    except Exception:
        pass
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="simulated_data/Contig_500k/genome_split_manifest.json")
    ap.add_argument("--genome_dir", default="./microbe_genomes")
    ap.add_argument("--out", default="audit_output/taxonomy_manifest.json")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    genome_dir = Path(args.genome_dir)

    # Read microbe files from all splits
    taxonomy = {}
    for split in ["train", "val", "test"]:
        files = manifest["microbe"][split].get("files", manifest["microbe"][split].get("strains", []))
        for fname in files:
            fpath = genome_dir / fname
            if not fpath.exists():
                continue
            organism = extract_organism_from_fasta(str(fpath))
            phylum = get_phylum_from_name(organism)
            taxonomy[fname] = {"organism": organism, "phylum": phylum, "split": split}

    # Summarize phylum distribution
    phylum_counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0, "total": 0})
    for fname, info in taxonomy.items():
        ph = info["phylum"]
        phylum_counts[ph]["total"] += 1
        phylum_counts[ph][info["split"]] += 1

    # output
    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"per_genome": taxonomy, "phylum_distribution": dict(phylum_counts)}, f, indent=2, ensure_ascii=False)

    # Print
    print(f"\n{'='*60}")
    print(f"Cross-Taxa Taxonomy Summary")
    print(f"{'='*60}")
    print(f"Total genomes annotated: {len(taxonomy)}")
    print(f"\nPhylum distribution:")
    sorted_phyla = sorted(phylum_counts.items(), key=lambda x: -x[1]["total"])
    print(f"{'Phylum':<25} {'Total':>6} {'Train':>6} {'Val':>5} {'Test':>5}")
    print("-" * 50)
    for ph, counts in sorted_phyla:
        print(f"{ph:<25} {counts['total']:>6} {counts['train']:>6} {counts['val']:>5} {counts['test']:>5}")

    # Cross-taxa holdout feasibility
    test_phyla = {p for p, c in phylum_counts.items() if c["test"] > 0}
    train_phyla = {p for p, c in phylum_counts.items() if c["train"] > 0}
    train_only = train_phyla - test_phyla
    test_only = test_phyla - train_phyla
    print(f"\nCross-taxa holdout feasibility:")
    print(f"  Train phyla: {len(train_phyla)}")
    print(f"  Test phyla: {len(test_phyla)}")
    print(f"  Train-only phyla: {train_only}")
    print(f"  Test-only phyla: {test_only}")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
