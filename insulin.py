"""Human insulin 1A7F preprocessing (SPEC.md section 5.3).

Turn the PDB entry 1A7F (human insulin, solution NMR) into the *experimental*
validation target for the trained network. For both chains we produce:
  - an integer-encoded amino-acid sequence (canonical ordering from `encoding.py`), and
  - a per-position binary alpha-helix-vs-other label taken from the structure's
    deposited HELIX (mmCIF `struct_conf`) annotation.
This is a hard 0/1 ground-truth annotation -- NOT a Forward-Backward posterior --
that later gets compared against the network's predicted P(alpha).

1A7F is a mutant NMR structure ("INSULIN MUTANT B16 GLU, B24 GLY, DES-B30"):
    chain A = 21 aa, matches wild-type human insulin A exactly;
    chain B = 29 aa (des-B30), with B16 Tyr->Glu and B24 Phe->Gly.
The alpha label counts right-handed alpha-helices only (mmCIF helix class 1); the
deposited helices are A2-A7, A13-A19 and B9-B18.

Output is saved via `forward_backward.save_dataset`, so it loads back through
`load_dataset` exactly like the simulated datasets: chain A at index 0, chain B at
index 1, with the binary alpha label stored in the posterior slot and the matching
hidden-state encoding (0=alpha, 1=other) in the state slot.
"""

import os
import urllib.request

import numpy as np
from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from encoding import AA_TO_IDX, IDX_TO_AA
from forward_backward import load_dataset, save_dataset

PDB_ID = "1A7F"
RCSB_CIF_URL = f"https://files.rcsb.org/download/{PDB_ID}.cif"

DATA_DIR = "dataset"
CIF_PATH = os.path.join(DATA_DIR, f"{PDB_ID}.cif")
DATASET_PATH = os.path.join(DATA_DIR, "insulin_1A7F.npz")

CHAIN_IDS = ("A", "B")

# Right-handed alpha-helix in the mmCIF `pdbx_PDB_helix_class` numbering.
ALPHA_HELIX_CLASS = 1

# --- Reference sequences for verification (SPEC.md section 5.4) ---------------
# Wild-type human insulin, canonical one-letter sequences.
WT_INSULIN_A = "GIVEQCCTSICSLYQLENYCN"           # 21 aa
WT_INSULIN_B = "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"  # 30 aa
# 1A7F's actual deposited chain B: B16 Y->E, B24 F->G, des-B30 (C-terminal T removed).
MUTANT_1A7F_B = "FVNQHLCGSHLVEALELVCGERGGFYTPK"  # 29 aa

# Three-letter -> one-letter for the 20 standard amino acids. Defined locally so we
# don't depend on biopython's moved/deprecated `three_to_one` helper.
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def download_structure(dest=CIF_PATH, url=RCSB_CIF_URL):
    """Download the mmCIF for 1A7F to `dest` (skip if it already exists)."""
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    return dest


def parse_chain(model, chain_id):
    """Extract one chain's residues and integer-encoded sequence from a model.

    Keeps only standard amino-acid residues (blank hetflag, known 3-letter code), in
    the order they appear in the chain.

    Returns:
        (residues, sequence):
          residues -- list of (auth_seqnum, one_letter_code) in sequence order;
          sequence -- 1-D int array of canonical amino-acid indices (0..19).
    """
    residues = []
    for res in model[chain_id]:
        if res.id[0] != " ":  # skip HETATM / waters
            continue
        one = THREE_TO_ONE.get(res.resname)
        if one is None:  # skip non-standard residues
            continue
        residues.append((res.id[1], one))  # res.id[1] is the author sequence number

    sequence = np.array([AA_TO_IDX[one] for _, one in residues], dtype=np.int64)
    return residues, sequence


def helix_ranges(cif_dict, chain_id, alpha_classes=(ALPHA_HELIX_CLASS,)):
    """Return [(beg, end), ...] author-residue-number ranges of qualifying helices.

    Reads the deposited mmCIF `_struct_conf` (HELIX) records and keeps only those on
    `chain_id` whose `pdbx_PDB_helix_class` is in `alpha_classes` (default: right-handed
    alpha only). Ranges are inclusive in author numbering.
    """
    if "_struct_conf.beg_auth_asym_id" not in cif_dict:
        return []  # no helix annotation in this entry

    def as_list(key):
        value = cif_dict[key]
        return value if isinstance(value, list) else [value]

    chains = as_list("_struct_conf.beg_auth_asym_id")
    begs = as_list("_struct_conf.beg_auth_seq_id")
    ends = as_list("_struct_conf.end_auth_seq_id")
    classes = as_list("_struct_conf.pdbx_PDB_helix_class")

    wanted = set(alpha_classes)
    return [
        (int(beg), int(end))
        for ch, beg, end, cls in zip(chains, begs, ends, classes)
        if ch == chain_id and int(cls) in wanted
    ]


def alpha_labels(residues, ranges):
    """Binary per-position label: 1 if a residue's author number lies in a helix range."""
    labels = np.zeros(len(residues), dtype=np.int64)
    for i, (seqnum, _) in enumerate(residues):
        if any(beg <= seqnum <= end for beg, end in ranges):
            labels[i] = 1
    return labels


def build_insulin_dataset(cif_path=CIF_PATH):
    """Assemble the insulin dataset for chains A and B from a local mmCIF file.

    Returns four lists, aligned by index (0 = chain A, 1 = chain B):
        sequences  -- integer-encoded amino-acid sequences;
        posteriors -- binary 0/1 alpha ground-truth labels (posterior slot);
        states     -- hidden-state encoding, 0=alpha / 1=other (= 1 - label);
        residues   -- list of (auth_seqnum, one_letter) per chain (kept for verification).
    """
    structure = MMCIFParser(QUIET=True).get_structure(PDB_ID, cif_path)
    model = next(iter(structure))  # first NMR model
    cif_dict = MMCIF2Dict(cif_path)

    sequences, posteriors, states, residues_by_chain = [], [], [], []
    for chain_id in CHAIN_IDS:
        residues, sequence = parse_chain(model, chain_id)
        labels = alpha_labels(residues, helix_ranges(cif_dict, chain_id))

        sequences.append(sequence)
        posteriors.append(labels.astype(np.float64))
        states.append((1 - labels).astype(np.int64))
        residues_by_chain.append(residues)
    return sequences, posteriors, states, residues_by_chain


def _diff_vs_wildtype(observed, wild_type):
    """Human-readable list of how `observed` differs from `wild_type` (1-based positions)."""
    diffs = [
        f"B{i} {w}->{o}"
        for i, (w, o) in enumerate(zip(wild_type, observed), start=1)
        if w != o
    ]
    if len(observed) < len(wild_type):
        removed = wild_type[len(observed):]
        diffs.append(f"des-B{len(wild_type)} ({removed} removed)")
    return diffs


if __name__ == "__main__":
    download_structure()
    sequences, posteriors, states, residues_by_chain = build_insulin_dataset()

    seqA, seqB = sequences
    strA = "".join(IDX_TO_AA[i] for i in seqA)
    strB = "".join(IDX_TO_AA[i] for i in seqB)

    # --- Sequence identity (SPEC.md section 5.4) -----------------------------
    assert len(seqA) == 21, f"chain A length {len(seqA)} != 21"
    assert strA == WT_INSULIN_A, f"chain A {strA!r} != wild-type {WT_INSULIN_A!r}"
    assert len(seqB) == 29, f"chain B length {len(seqB)} != 29"
    assert strB == MUTANT_1A7F_B, f"chain B {strB!r} != expected 1A7F mutant {MUTANT_1A7F_B!r}"

    # --- Label invariants ----------------------------------------------------
    for labels, seq, st in zip(posteriors, sequences, states):
        assert set(np.unique(labels)).issubset({0.0, 1.0}), "labels must be binary 0/1"
        assert len(labels) == len(seq), "label/sequence length mismatch"
        assert np.array_equal(st, 1 - labels.astype(np.int64)), "states must equal 1 - label"

    # --- Label correctness: alpha positions match the deposited HELIX ranges --
    alpha_nums = [
        {num for (num, _), lab in zip(residues, labels) if lab == 1}
        for residues, labels in zip(residues_by_chain, posteriors)
    ]
    expected_A = set(range(2, 8)) | set(range(13, 20))  # A2-A7, A13-A19
    expected_B = set(range(9, 19))                       # B9-B18
    assert alpha_nums[0] == expected_A, f"chain A alpha {sorted(alpha_nums[0])} != {sorted(expected_A)}"
    assert alpha_nums[1] == expected_B, f"chain B alpha {sorted(alpha_nums[1])} != {sorted(expected_B)}"

    # --- Save + round-trip through load_dataset ------------------------------
    save_dataset(DATASET_PATH, sequences, posteriors, states)
    loaded = load_dataset(DATASET_PATH)
    for original, restored in zip((sequences, posteriors, states), loaded):
        for a, b in zip(original, restored):
            assert np.array_equal(a, b), "load_dataset round-trip mismatch"

    # --- Summary -------------------------------------------------------------
    print(f"Parsed {PDB_ID}: chain A = {len(seqA)} aa, chain B = {len(seqB)} aa\n")
    for cid, seq_str, labels in zip(CHAIN_IDS, (strA, strB), posteriors):
        label_str = "".join(str(int(x)) for x in labels)
        print(f"chain {cid}: {seq_str}")
        print(f"  alpha : {label_str}   ({int(labels.sum())}/{len(labels)} alpha)\n")

    print("chain A matches wild-type human insulin A.")
    print("chain B vs wild-type human insulin B: " + "; ".join(_diff_vs_wildtype(strB, WT_INSULIN_B)))
    print(f"\nSaved dataset -> {DATASET_PATH} (index 0 = chain A, index 1 = chain B)")
    print("load_dataset round-trip: OK")
