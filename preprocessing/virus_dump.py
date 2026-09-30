#!/usr/bin/env python3
# Claude sucks and I did not end up using this because it's useless
# I think I just used the tax names that intact gives from their interaction db
"""
Download NCBI's taxonomy dump and build a CSV mapping
NCBI Taxonomy ID -> scientific name, for virus species under the
"Viruses" node (taxid 10239).

NCBI's virus taxonomy quirk: formal ICTV binomial species names
(e.g. "Betacoronavirus pandemicum") are tagged rank=='species', while
the names people actually use (e.g. "Severe acute respiratory
syndrome coronavirus 2" / SARS-CoV-2, taxid 2697049; "Human
alphaherpesvirus 1" / HSV-1, taxid 10298) are filed one level below
as rank=='no rank' children of that species node. The default
--mode practical accounts for this. See --help for alternatives.

Usage:
    python ncbi_virus_species_map.py [-o output.csv] [--mode practical|strict|all]

Requires:
    pip install requests
"""

import argparse
import csv
import io
import sys
import tarfile
from collections import defaultdict

import requests

TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
VIRUS_ROOT_TAXID = "10239"  # "Viruses" node


def download_taxdump(url=TAXDUMP_URL):
    """Download taxdump.tar.gz into memory and return it as a tarfile object."""
    print(f"Downloading {url} ...", file=sys.stderr)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    buf = io.BytesIO()
    total = 0
    for chunk in resp.iter_content(chunk_size=1024 * 1024):
        buf.write(chunk)
        total += len(chunk)
        print(f"\r  {total / 1e6:.1f} MB downloaded", end="", file=sys.stderr)
    print(file=sys.stderr)

    buf.seek(0)
    return tarfile.open(fileobj=buf, mode="r:gz")


def parse_nodes(tar):
    """
    Parse nodes.dmp.
    Returns:
        children_of: dict taxid -> list of child taxids
        rank_of:     dict taxid -> rank string
        parent_of:   dict taxid -> parent taxid
    """
    children_of = defaultdict(list)
    rank_of = {}
    parent_of = {}

    member = tar.extractfile("nodes.dmp")
    for raw_line in member:
        line = raw_line.decode("utf-8")
        fields = [f.strip() for f in line.split("|")]
        tax_id, parent_id, rank = fields[0], fields[1], fields[2]
        rank_of[tax_id] = rank
        parent_of[tax_id] = parent_id
        children_of[parent_id].append(tax_id)

    return children_of, rank_of, parent_of


def parse_names(tar):
    """Parse names.dmp. Returns dict: taxid -> scientific name."""
    names = {}
    member = tar.extractfile("names.dmp")
    for raw_line in member:
        line = raw_line.decode("utf-8")
        fields = [f.strip() for f in line.split("|")]
        tax_id, name, name_class = fields[0], fields[1], fields[3]
        if name_class == "scientific name":
            names[tax_id] = name

    return names


def collect_descendants(root, children_of):
    """Walk the taxonomy tree from `root`, returning all descendant taxids (incl. root)."""
    seen = {root}
    stack = [root]
    while stack:
        current = stack.pop()
        for child in children_of.get(current, []):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        default="ncbi_virus_species.csv",
        help="Output CSV path (default: %(default)s)",
    )
    parser.add_argument(
        "--mode",
        choices=["practical", "strict", "all"],
        default="practical",
        help=(
            "practical (default): rank=='species' PLUS any 'no rank' taxon "
            "whose parent is rank=='species'. This is the level most people "
            "mean by 'virus species' -- it includes commonly-used names like "
            "'Severe acute respiratory syndrome coronavirus 2' (taxid 2697049) "
            "and 'Human alphaherpesvirus 1' / HSV-1 (taxid 10298), which NCBI "
            "files as 'no rank' children just below the formal ICTV species "
            "node ('Betacoronavirus pandemicum', 'Simplexvirus humanalpha1'). "
            "strict: only taxa NCBI tags rank=='species' (the formal ICTV "
            "binomial names only -- will NOT include HSV-1 or SARS-CoV-2 by "
            "their familiar taxids). "
            "all: every taxon in the Viruses subtree, any rank."
        ),
    )
    args = parser.parse_args()

    tar = download_taxdump()

    print("Parsing nodes.dmp ...", file=sys.stderr)
    children_of, rank_of, parent_of = parse_nodes(tar)

    print("Parsing names.dmp ...", file=sys.stderr)
    names = parse_names(tar)

    print("Walking virus subtree ...", file=sys.stderr)
    virus_taxids = collect_descendants(VIRUS_ROOT_TAXID, children_of)

    if args.mode == "strict":
        virus_taxids = {t for t in virus_taxids if rank_of.get(t) == "species"}
    elif args.mode == "practical":

        def is_practical_species(t):
            if rank_of.get(t) == "species":
                return True
            if rank_of.get(t) == "no rank":
                parent = parent_of.get(t)
                if rank_of.get(parent) == "species":
                    return True
            return False

        virus_taxids = {t for t in virus_taxids if is_practical_species(t)}
    # "all" mode: keep every taxid as-is

    print(f"Writing {len(virus_taxids)} rows to {args.output} ...", file=sys.stderr)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tax_id", "name"])
        for tax_id in sorted(virus_taxids, key=int):
            writer.writerow([tax_id, names.get(tax_id, "")])

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
