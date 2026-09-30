"""
Convert a MITAB 2.5 interaction file into a simplified CSV with columns:
A, B, taxonomy_A, taxonomy_B, entrez_A, entrez_B, name_A, name_B

Usage:
    python process_mitab.py input.txt output.csv
"""

import csv
import re
import sys


def get_uniprot(field):
    """Extract UniProt ID from a field like 'biogrid:112315|uniprot/swiss-prot:P45985|refseq:NP_003001|'"""
    for part in field.split("|"):
        m = re.match(r"uniprot/swiss-prot:(.*)", part)
        if m:
            return m.group(1)
    raise ValueError


def get_taxid(field):
    """Extract numeric taxonomy ID from a field like 'taxid:9606(human)'."""
    m = re.search(r"taxid:(\d+)", field)
    if m:
        return m.group(1)
    raise ValueError


def get_entrez(*fields):
    """Search one or more pipe-separated fields for an Entrez Gene ID."""
    for field in fields:
        for part in field.split("|"):
            m = re.match(r"entrez gene/locuslink:(\S+)", part.strip(), re.IGNORECASE)
            if m:
                return m.group(1)
    raise ValueError


def get_name(field):
    """
    Pick a 'best' name from an alias/identifier field.
    Prefers entries tagged (gene name) or (display_short), otherwise
    falls back to the first entry's value.
    """
    parts = [p.strip() for p in field.split("|") if p.strip() and p.strip() != "-"]

    # Prefer a tagged gene name / display short name
    for part in parts:
        m = re.match(r".*:(.+)\((gene name|display_short)\)", part, re.IGNORECASE)
        if m:
            return m.group(1)

    # Fallback: first entry, value before any "(...)"
    if parts:
        m = re.match(r".*:(.+?)(\(.*\))?$", parts[0])
        if m:
            return m.group(1)

    raise ValueError


def main(infile, outfile, virus_map):
    with open(virus_map, "r") as f:
        virus_map = {line.strip().split(",")[0]: line.strip().split(",")[1] for line in f}

    unmapped = set()
    with open(infile, encoding="utf-8") as fin, open(outfile, "w", newline="", encoding="utf-8") as fout:
        reader = csv.reader(fin, delimiter="\t")
        writer = csv.writer(fout)
        writer.writerow(["A", "B", "taxonomy_A", "taxonomy_B", "entrez_A", "entrez_B", "name_A", "name_B"])

        for row in reader:
            if not row or row[0].startswith("#"):
                continue  # skip empty lines / header comments

            id_a, id_b = row[2], row[3]  # alt IDs for uniprot
            alt_a, alt_b = row[0], row[1]
            tax_a, tax_b = row[9], row[10]

            if not (tax_a == "taxid:9606") ^ (tax_b == "taxid:9606"):
                continue

            if tax_a == "taxid:9606":  # make sure human is always second interactor
                tax_a, tax_b = tax_b, tax_a
                alt_b, alt_a = alt_a, alt_b
                id_b, id_a = id_a, id_b

            name_a = virus_map.get(tax_a.split(":")[1])
            name_b = virus_map.get(tax_b.split(":")[1])

            if not name_a or not name_b:
                unmapped.update({tax_a, tax_b})
                continue

            try:
                writer.writerow(
                    [
                        get_uniprot(id_a),
                        get_uniprot(id_b),
                        get_taxid(tax_a),
                        get_taxid(tax_b),
                        get_entrez(alt_a),
                        get_entrez(alt_b),
                        name_a if name_a is not None else "human",
                        name_b if name_b is not None else "human",
                    ]
                )
            except ValueError:
                continue
    print(len(unmapped))
    for i in sorted(unmapped):
        print(i)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python mitab_to_csv.py <input_mitab_file> <output_csv_file>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], "../data/virus_names.csv")
