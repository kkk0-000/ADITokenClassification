"""
Generate token-level labels for ADI enzyme sequences
based on US-align structural alignment against a known template.

Workflow:
  1. Parse US-align output to get residue-residue correspondence
  2. Map template insertion region to target sequence positions
  3. Optionally fill gaps within the mapped region
  4. Add catalytic triad positions
  5. Save labels as .npy files

Usage:
  python generate_adi_labels.py \
      --usalign_dir ./usalign_results \
      --template_insert_start 150 \
      --template_insert_end 250 \
      --template_triad 130,270,278 \
      --fasta ./all_adi_sequences.fasta \
      --output_label_dir ./labels \
      --output_csv ./data/all_adi.csv \
      --mode fill \
      --max_gap 15
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def parse_usalign_output(filepath):
    """
    Parse US-align default output (-outfmt 0 or -outfmt -1).
    Returns the 3-line alignment block:
      line1: structure 1 (target) aligned sequence
      line2: alignment symbols (: . space)
      line3: structure 2 (template) aligned sequence
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the alignment block: it's the last 3 non-empty lines
    # before any trailing blank lines. The pattern is:
    #   target_seq
    #   symbols (: and . characters)
    #   template_seq
    aln_lines = []
    in_alignment = False
    for line in lines:
        stripped = line.rstrip('\n')
        # The alignment symbols line contains only :, ., space
        # The sequence lines contain amino acid letters and -
        if not in_alignment:
            # Look for the alignment indicator line
            # "(\":\" denotes residue pairs of d <  5.0 Angstrom..."
            if '(":" denotes' in stripped or "denotes residue pairs" in stripped:
                in_alignment = True
                continue
        else:
            if stripped.strip():
                aln_lines.append(stripped)

    if len(aln_lines) < 3:
        raise ValueError(f"Could not parse alignment from {filepath}. "
                         f"Found {len(aln_lines)} alignment lines.")

    target_aln = aln_lines[0]
    symbols = aln_lines[1]
    template_aln = aln_lines[2]

    assert len(target_aln) == len(symbols) == len(template_aln), \
        f"Alignment line lengths mismatch in {filepath}"

    return target_aln, symbols, template_aln


def parse_usalign_fasta(filepath):
    """
    Parse US-align FASTA output (-outfmt 1).
    Format:
      >structure1_name  Len=X  TM-score=X  ...
      TARGET_ALIGNED_SEQ
      >structure2_name  Len=X  TM-score=X  ...
      TEMPLATE_ALIGNED_SEQ
    """
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    seqs = []
    for line in lines:
        if not line.startswith('>'):
            seqs.append(line)

    if len(seqs) < 2:
        raise ValueError(f"Could not parse FASTA alignment from {filepath}")

    target_aln = seqs[0]
    template_aln = seqs[1]

    # Generate symbols from gap pattern
    symbols = ''
    for t1, t2 in zip(target_aln, template_aln):
        if t1 != '-' and t2 != '-':
            symbols += ':'
        else:
            symbols += ' '

    return target_aln, symbols, template_aln


def build_residue_mapping(target_aln, template_aln):
    """
    Build residue index mapping from alignment.

    Returns:
      template_to_target: dict mapping template residue index -> target residue index
                          (only for aligned positions, i.e., both are non-gap)
    """
    template_to_target = {}
    target_idx = -1
    template_idx = -1

    for i in range(len(target_aln)):
        t_char = target_aln[i]
        r_char = template_aln[i]

        if t_char != '-':
            target_idx += 1
        if r_char != '-':
            template_idx += 1

        if t_char != '-' and r_char != '-':
            template_to_target[template_idx] = target_idx

    return template_to_target


def map_region_to_target(template_to_target, region_start, region_end, target_seq_len, mode='fill', max_gap=15):
    """
    Map a continuous template region [region_start, region_end) to target positions.

    Args:
      template_to_target: residue mapping dict
      region_start: start position in template (0-indexed, inclusive)
      region_end: end position in template (0-indexed, exclusive)
      target_seq_len: length of target sequence
      mode: 'strict' or 'fill'
      max_gap: only fill gaps smaller than this (only used in 'fill' mode)

    Returns:
      set of target residue positions to label as 1
    """
    # Find all target positions that correspond to the template region
    mapped_positions = []
    for templ_pos in range(region_start, region_end):
        if templ_pos in template_to_target:
            mapped_positions.append(template_to_target[templ_pos])

    if not mapped_positions:
        return set()

    if mode == 'strict':
        return set(mapped_positions)

    elif mode == 'fill':
        # Fill from first to last mapped position
        first = min(mapped_positions)
        last = max(mapped_positions)

        # Check if the total span is reasonable
        span = last - first + 1
        expected = region_end - region_start
        if span > expected * 2:
            print(f"  WARNING: mapped span ({span}) is >2x expected ({expected}). "
                  f"Consider using 'strict' mode.")

        # Fill, but respect max_gap: break into segments if gaps are too large
        if max_gap > 0:
            sorted_pos = sorted(mapped_positions)
            filled = set()
            seg_start = sorted_pos[0]
            prev = sorted_pos[0]

            for p in sorted_pos[1:]:
                if p - prev - 1 > max_gap:
                    # Close current segment, start new one
                    filled.update(range(seg_start, prev + 1))
                    seg_start = p
                prev = p
            filled.update(range(seg_start, prev + 1))
            return filled
        else:
            return set(range(first, last + 1))

    else:
        raise ValueError(f"Unknown mode: {mode}")


def map_point_positions(template_to_target, template_positions, tolerance=2):
    """
    Map individual template positions (e.g., catalytic triad) to target.
    If exact match not found, try nearby positions within tolerance.

    Returns:
      list of target positions (may be shorter than input if mapping fails)
    """
    mapped = []
    for pos in template_positions:
        if pos in template_to_target:
            mapped.append(template_to_target[pos])
        else:
            found = False
            for offset in range(1, tolerance + 1):
                for nearby in [pos - offset, pos + offset]:
                    if nearby in template_to_target:
                        mapped.append(template_to_target[nearby])
                        found = True
                        break
                if found:
                    break
            if not found:
                print(f"  WARNING: template catalytic position {pos} "
                      f"could not be mapped to target")
    return mapped


def generate_label(target_seq_len, insert_positions, triad_positions, triad_expand=3):
    """
    Generate a label array for one target sequence.

    Args:
      target_seq_len: length of the target protein sequence
      insert_positions: set of target positions for ADI-specific insertion
      triad_positions: list of target positions for catalytic triad
      triad_expand: expand catalytic triad by this many residues on each side
                    (to label the active site pocket, not just the single residue)
    """
    label = np.zeros(target_seq_len, dtype=np.int64)

    # Label insertion region
    for pos in insert_positions:
        if 0 <= pos < target_seq_len:
            label[pos] = 1

    # Label catalytic triad (with optional expansion)
    for pos in triad_positions:
        for p in range(pos - triad_expand, pos + triad_expand + 1):
            if 0 <= p < target_seq_len:
                label[p] = 1

    return label


def read_fasta(fasta_path):
    """Simple FASTA reader returning dict of {id: sequence}."""
    sequences = {}
    current_id = None
    current_seq = []
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_id is not None:
        sequences[current_id] = ''.join(current_seq)
    return sequences


def main():
    parser = argparse.ArgumentParser(
        description='Generate ADI token-level labels from US-align results')

    parser.add_argument('--usalign_dir', type=str, required=True,
                        help='Directory containing US-align output files. '
                             'Each file should be named <target_id>.txt or <target_id>.fasta')
    parser.add_argument('--usalign_fmt', type=str, default='default',
                        choices=['default', 'fasta'],
                        help='US-align output format: "default" (-outfmt 0) or "fasta" (-outfmt 1)')
    parser.add_argument('--template_insert_start', type=int, required=True,
                        help='Insertion region start in template (0-indexed, inclusive)')
    parser.add_argument('--template_insert_end', type=int, required=True,
                        help='Insertion region end in template (0-indexed, exclusive)')
    parser.add_argument('--template_triad', type=str, default='',
                        help='Comma-separated catalytic triad positions in template (0-indexed)')
    parser.add_argument('--triad_expand', type=int, default=0,
                        help='Expand catalytic triad labeling by N residues on each side (default: 0)')
    parser.add_argument('--fasta', type=str, required=True,
                        help='FASTA file of all ADI sequences (for getting full sequences)')
    parser.add_argument('--neg_fasta', type=str, default='',
                        help='FASTA file of negative (non-ADI) family sequences')
    parser.add_argument('--output_label_dir', type=str, default='./labels',
                        help='Output directory for .npy label files')
    parser.add_argument('--output_csv', type=str, default='./data/all.csv',
                        help='Output CSV file')
    parser.add_argument('--mode', type=str, default='fill', choices=['strict', 'fill'],
                        help='"strict": only label aligned residues. '
                             '"fill": fill gaps between first and last mapped positions.')
    parser.add_argument('--max_gap', type=int, default=15,
                        help='Maximum gap size to fill in "fill" mode. '
                             'Gaps larger than this break the region into segments.')

    args = parser.parse_args()

    os.makedirs(args.output_label_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv) or '.', exist_ok=True)

    # Parse catalytic triad positions
    triad_positions = []
    if args.template_triad:
        triad_positions = [int(x.strip()) for x in args.template_triad.split(',')]

    # Read all sequences
    sequences = read_fasta(args.fasta)
    print(f"Loaded {len(sequences)} ADI sequences from {args.fasta}")

    neg_sequences = {}
    if args.neg_fasta:
        neg_sequences = read_fasta(args.neg_fasta)
        print(f"Loaded {len(neg_sequences)} negative sequences from {args.neg_fasta}")

    # Process each US-align result
    records = []
    usalign_files = list(Path(args.usalign_dir).glob('*'))

    for aln_file in usalign_files:
        target_id = aln_file.stem
        if target_id not in sequences:
            continue

        target_seq = sequences[target_id]
        target_len = len(target_seq)

        try:
            if args.usalign_fmt == 'fasta':
                target_aln, symbols, template_aln = parse_usalign_fasta(str(aln_file))
            else:
                target_aln, symbols, template_aln = parse_usalign_output(str(aln_file))
        except Exception as e:
            print(f"  SKIP {target_id}: {e}")
            continue

        # Build residue mapping
        mapping = build_residue_mapping(target_aln, template_aln)

        # Map insertion region
        insert_positions = map_region_to_target(
            mapping,
            args.template_insert_start,
            args.template_insert_end,
            target_len,
            mode=args.mode,
            max_gap=args.max_gap
        )

        # Map catalytic triad
        mapped_triad = map_point_positions(mapping, triad_positions)

        # Generate label
        label = generate_label(target_len, insert_positions, mapped_triad,
                               triad_expand=args.triad_expand)

        # Sanity check
        n_labeled = int(label.sum())
        insert_labeled = len(insert_positions)
        print(f"  {target_id}: seq_len={target_len}, "
              f"insert_residues={insert_labeled}, "
              f"triad_mapped={len(mapped_triad)}/{len(triad_positions)}, "
              f"total_labeled={n_labeled}")

        # Save
        np.save(os.path.join(args.output_label_dir, f"{target_id}.npy"), label)
        records.append({"Class": 1, "ProId": target_id, "Sequence": target_seq})

    # Add negative samples
    for seq_id, seq in neg_sequences.items():
        records.append({"Class": 0, "ProId": seq_id, "Sequence": seq})

    # Write CSV
    df = pd.DataFrame(records)
    df.to_csv(args.output_csv, index=False)

    n_pos = sum(1 for r in records if r['Class'] == 1)
    n_neg = sum(1 for r in records if r['Class'] == 0)
    print(f"\nDone. Total: {len(records)} (positive: {n_pos}, negative: {n_neg})")
    print(f"Labels saved to: {args.output_label_dir}")
    print(f"CSV saved to: {args.output_csv}")


if __name__ == '__main__':
    main()
