#!/usr/bin/env python3
"""
Token-level ADI annotation pipeline.

For each query AlphaFold structure:
  1. Read sequence + per-residue pLDDT from PDB B-factor
  2. Use US-align mapping to transfer 1rxx annotations to query positions
  3. Assign labels:
       -1 = low pLDDT (< threshold, masked)
        0 = background
        1 = ADI unique structural region
        2 = functional site (catalytic / substrate binding / regulatory)
        3 = both ADI region + functional site

Output:
  - token_labels_detail.csv  (per-residue详细)
  - token_labels_seq.csv     (per-sequence，适合ML输入)
"""

import os
import re
import glob
import csv
from collections import OrderedDict

#============================================================
# 常量
# ============================================================
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}

LABEL_NAMES = {
    -1: "masked(low_pLDDT)",
    0: "background",
    1: "ADI_structural_region",
    2: "functional_site",
    3: "ADI_region+functional_site",
}

# Unique ADI structural regions (1rxx PDB 残基编号)
ADI_REGIONS = OrderedDict([
    ("alpha_helical_domain",range(72, 158)),
    ("beta_strand_module_I",     range(8, 11)),
    ("beta_strand_module_III_1", range(302, 309)),
    ("beta_strand_module_III_2", range(316, 321)),
])

# 功能位点（允许同一位点多注释，用list 避免 dict key 冲突）
TARGET_SITES_1RXX = [
    # 催化三联体
    (224, "E", "Catalytic Triad"),
    (278, "H", "Catalytic Triad"),
    (406, "C", "Catalytic Triad"),
    # 底物结合
    (163, "F", "Substrate Binding"),
    (166, "D", "Substrate Binding"),
    (185, "R", "Substrate Binding"),
    (280, "D", "Substrate Binding"),
    (401, "R", "Substrate Binding"),
    # 调控/稳定化
    (13,  "E", "Regulatory / Stabilization Network"),
    (165, "R", "Regulatory / Stabilization Network"),
    (188, "E", "Regulatory / Stabilization Network"),
    (224, "E", "Regulatory Network"),
    (227, "D", "Regulatory / Stabilization Network"),
    (278, "H", "Regulatory Network"),
    (280, "D", "Regulatory Network"),
    (401, "R", "Regulatory Network"),
    (405, "H", "Regulatory / Stabilization Network"),
]


# ============================================================
# PDB 解析
# ============================================================
def pdb_chain_to_fasta_with_map(pdb_file, chain_target="A"):
    residues = OrderedDict()
    with open(pdb_file, "r") as f:
        for line in f:
            record = line[:6].strip()
            if record not in {"ATOM", "HETATM"}:
                continue
            chain_id = line[21].strip() or "_"
            if chain_id != chain_target:
                continue
            resname = line[17:20].strip()
            if resname not in AA3_TO_1:
                continue
            resseq = line[22:26].strip()
            icode = line[26].strip()
            resid = (resseq, icode)
            if resid not in residues:
                residues[resid] = {
                    "aa1": AA3_TO_1[resname],
                    "resname": resname,
                    "resseq": resseq,
                    "icode": icode,
                }
    seq = []
    mapping = []
    for i, ((resseq, icode), info) in enumerate(residues.items(), start=1):
        seq.append(info["aa1"])
        mapping.append({
            "fasta_index": i,
            "aa": info["aa1"],
            "pdb_resseq": resseq,
            "pdb_icode": icode,"pdb_resname": info["resname"],
        })
    return ''.join(seq), mapping


def build_ref_index_map(pdb_mapping):
    return {row["fasta_index"]: int(row["pdb_resseq"]) for row in pdb_mapping}


def read_query_pdb_plddt(pdb_file, chain="A"):
    """读取 AlphaFold PDB → (sequence, [pLDDT per residue])"""
    residues = OrderedDict()
    with open(pdb_file, "r") as f:
        for line in f:
            if line[:6].strip() != "ATOM":
                continue
            ch = line[21].strip() or "_"
            if ch != chain:
                continue
            resname = line[17:20].strip()
            if resname not in AA3_TO_1:
                continue
            resseq = line[22:26].strip()
            icode = line[26].strip()
            bfactor = float(line[60:66].strip())
            rid = (resseq, icode)
            if rid not in residues:
                residues[rid] = {
                    "aa": AA3_TO_1[resname],
                    "plddt": bfactor,
                    "resseq": resseq,
                }
    seq = ''.join(r["aa"] for r in residues.values())
    plddts = [r["plddt"] for r in residues.values()]
    return seq, plddts


# ============================================================
# 区域 / 功能位点构建
# ============================================================
def build_region_target_sites(pdb_mapping, regions):
    pdb_to_aa = {int(row["pdb_resseq"]): row["aa"] for row in pdb_mapping}
    target_sites = OrderedDict()
    pos_to_region = {}
    for region_name, pos_range in regions.items():
        for pos in pos_range:
            if pos in pdb_to_aa:
                target_sites[pos] = pdb_to_aa[pos]
                pos_to_region[pos] = region_name
    return target_sites, pos_to_region


def build_functional_pos_set(sites_list):
    """从 TARGET_SITES_1RXX 提取去重的位点集合（用于 label赋值）"""
    return {pos for pos, aa, label in sites_list}


def build_functional_detail_map(sites_list):
    """pos → [func_label1, func_label2, ...]用于详细注释"""
    detail = {}
    for pos, aa, label in sites_list:
        detail.setdefault(pos, []).append(label)
    return detail


# ============================================================
# US-align 解析
# ============================================================
def parse_usalign_aln(aln_file):
    with open(aln_file, 'r') as f:
        lines = f.readlines()

    info = {}
    for line in lines:
        line_s = line.strip()
        if line_s.startswith('Name of Structure_1:'):
            info['name1'] = line_s.split(':', 1)[1].strip()
        elif line_s.startswith('Name of Structure_2:'):
            info['name2'] = line_s.split(':', 1)[1].strip()

        m = re.match(
            r'Aligned length=\s*(\d+),\s*RMSD=\s*([\d.]+),\s*Seq_ID=.*=\s*([\d.]+)',
            line_s,
        )
        if m:
            info['aligned_len'] = int(m.group(1))
            info['rmsd'] = float(m.group(2))
            info['seq_id'] = float(m.group(3))

        m = re.match(
            r'TM-score=\s*([\d.]+)\s*\(normalized by length of Structure_2',
            line_s,
        )
        if m:
            info['tm_score'] = float(m.group(1))
            aln_start = None
    for i, line in enumerate(lines):
        if '(":" denotes' in line:
            aln_start = i + 1
            break
    if aln_start is None:
        return None

    while aln_start < len(lines) and lines[aln_start].strip() == '':
        aln_start += 1
    if aln_start + 2 >= len(lines):
        return None

    seq_line1 = lines[aln_start].rstrip('\n')
    match_line = lines[aln_start + 1].rstrip('\n')
    seq_line2 = lines[aln_start + 2].rstrip('\n')

    max_len = max(len(seq_line1), len(match_line), len(seq_line2))
    info['seq1'] = seq_line1.ljust(max_len)
    info['match'] = match_line.ljust(max_len)
    info['seq2'] = seq_line2.ljust(max_len)
    return info


# ============================================================
# 位置映射
# ============================================================
def build_position_mapping(seq_ref, seq_query, match_line, ref_index_map):
    mapping = {}
    ref_orig_pos = 0
    query_orig_pos = 0

    for align_pos in range(len(seq_ref)):
        r_char = seq_ref[align_pos]
        q_char = seq_query[align_pos]
        m_char = match_line[align_pos]

        if r_char != '-':
            ref_orig_pos += 1
        if q_char != '-':
            query_orig_pos += 1

        if r_char != '-':
            ref_pdb_pos = ref_index_map.get(ref_orig_pos, None)
            if ref_pdb_pos is None:
                continue
            mapping[ref_pdb_pos] = {
                'ref_aa': r_char,
                'ref_orig_pos': ref_orig_pos,
                'query_aa': q_char if q_char != '-' else '-',
                'query_orig_pos': query_orig_pos if q_char != '-' else None,
                'quality': m_char,
            }
    return mapping


def build_query_to_ref_map(position_mapping):
    """反转映射: query_orig_pos (1-based) → ref_pdb_pos"""
    return {
        info['query_orig_pos']: ref_pos
        for ref_pos, info in position_mapping.items()
        if info['query_orig_pos'] is not None
    }


# ============================================================
# Token-level 标注
# ============================================================
# def assign_token_labels(query_len, q2r_map, adi_pos_set,
#                         catalytic_sites, substrate_binding_sites, regulatory_sites,
#                         plddts, plddt_threshold=70.0):
#     """
#     新label定义：
#     0 = background (阴性)
#     1 = ADI结构域
#     2 = 催化三联体
#     3 = 底物结合位点
#     4 = 调控/稳定位点
#     -1 = pLDDT低质量，masked
#     """
#     labels = [0] * query_len
#     for qpos_1 in range(1, query_len + 1):
#         idx = qpos_1 - 1
#         ref_pos = q2r_map.get(qpos_1)
#         if ref_pos is None:
#             continue
#         # 先赋ADI结构域
#         if ref_pos in adi_pos_set:
#             labels[idx] = 1
        
#         # 覆盖赋值功能位点（优先级可调，举例催化 > 底物结合 > 调控）
#         if ref_pos in catalytic_sites:
#             labels[idx] = 2
#         elif ref_pos in substrate_binding_sites:
#             labels[idx] = 3
#         elif ref_pos in regulatory_sites:
#             labels[idx] = 4

#     # pLDDT阈值下，掩码为-1
#     for i in range(query_len):
#         if plddts[i] < plddt_threshold:
#             labels[i] = 0

#     return labels
def assign_token_labels(query_len, q2r_map, adi_pos_set,
                        catalytic_sites, substrate_binding_sites, regulatory_sites,
                        plddts, plddt_threshold=70.0):
    """
    新label定义（多标签）：
      -1 = low pLDDT, masked
       0 = background
       1 = ADI结构domain
       2 = 催化三联体
       3 = 底物结合位点
       4 = 调控/稳定位点

    多标签位点使用 '-' 连接，如 '1-2' 表示同时是ADI和催化位点
    """
    labels = []
    for qpos_1 in range(1, query_len + 1):
        idx = qpos_1 - 1
        ref_pos = q2r_map.get(qpos_1)
        if plddts[idx] < plddt_threshold:
            # 低质量掩码
            labels.append("0")
            continue

        if ref_pos is None:
            labels.append("0")
            continue

        label_set = set()

        # 添加对应标签
        if ref_pos in adi_pos_set:
            label_set.add("1")
        if ref_pos in catalytic_sites:
            label_set.add("2")
        if ref_pos in substrate_binding_sites:
            label_set.add("3")
        if ref_pos in regulatory_sites:
            label_set.add("4")

        if len(label_set) == 0:
            labels.append("0")
        else:
            # 多个标签排序后合并成字符串
            labels.append("-".join(sorted(label_set)))

    return labels

# ============================================================
# Token-level 标注 下面是token-level 二分类问题
# ============================================================
# def assign_token_labels(query_len, q2r_map, adi_pos_set, func_pos_set,
#                         plddts, plddt_threshold=70.0):
#     """
#     为query 序列每个残基赋标签:
#       -1 = pLDDT < threshold (masked)
#        0 = background
#        1 = ADI structural region
#        2 = functional site
#        3 = both (1|2)
#     """
#     labels = [0] * query_len
#     for qpos_1 in range(1, query_len + 1):
#         idx = qpos_1 - 1
#         ref_pos = q2r_map.get(qpos_1)
#         if ref_pos is not None:
#             if ref_pos in adi_pos_set:
#                 labels[idx] |= 1
#             if ref_pos in func_pos_set:
#                 labels[idx] |= 2

#     # pLDDT 掩码
#     for i in range(query_len):
#         if plddts[i] < plddt_threshold:
#             labels[i] = -100

#     return labels


def build_detail_annotation(query_len, q2r_map, pos_to_region, func_detail_map):
    """为每个 query 残基生成详细区域/功能注释字符串"""
    annotations = [""] * query_len
    for qpos_1 in range(1, query_len + 1):
        idx = qpos_1 - 1
        ref_pos = q2r_map.get(qpos_1)
        if ref_pos is None:
            continue
        parts = []
        if ref_pos in pos_to_region:
            parts.append(pos_to_region[ref_pos])
        if ref_pos in func_detail_map:
            parts.extend(func_detail_map[ref_pos])
        annotations[idx] = "|".join(parts)
    return annotations

# ============================================================
# 主流程
# ============================================================
def main():
    # ---------- 路径配置 ----------
    aln_dir       = "/home/nick/data/prokaryo/ADI_predict_prepare_data/new_data/align_data/ref_usalign_1rxx/usalign"
    # query_pdb_dir = "/home/nick/data/prokaryo/ADI_predict_prepare_data/new_data/align_data/ref_usalign_1rxx/query_pdbs"
    query_pdb_dir = '/home/nick/data/prokaryo/ADI_predict_prepare_data/new_data/align_data/ref_r1xx_1lxy_agument_pdb'
    ref_pdb       = "ref_1rxx/1rxx_chainA.pdb"
    output_detail = "/home/nick/data/prokaryo/ADI_predict_prepare_data/new_data/align_data/ref_usalign_1rxx/token_labels_detail_multi_labels.csv"
    output_seq    = "/home/nick/data/prokaryo/ADI_predict_prepare_data/new_data/align_data/ref_usalign_1rxx/token_labels_seq_multi_labels.csv"
    PLDDT_THRESHOLD = 70.0
    QUERY_PDB_CHAIN = "A"
    # PDB 文件后缀，按优先级尝试
    PDB_EXTENSIONS= [".pdb", ".ent", ".cif"]
    catalytic_triads = {224, 278, 406}
    substrate_binding = {163, 166, 185, 280, 401}
    regulatory_sites = {13, 165, 188,224,227,278,280,401, 405}

    # ---------- 1. 参考结构 ----------
    ref_seq, pdb_mapping = pdb_chain_to_fasta_with_map(ref_pdb, "A")
    ref_index_map = build_ref_index_map(pdb_mapping)

    TARGET_SITES, POS_TO_REGION = build_region_target_sites(pdb_mapping, ADI_REGIONS)
    adi_pos_set   = set(TARGET_SITES.keys())
    func_pos_set  = build_functional_pos_set(TARGET_SITES_1RXX)
    func_detail   = build_functional_detail_map(TARGET_SITES_1RXX)

    print(f"Ref: {len(ref_seq)} residues, {len(adi_pos_set)} ADI positions, "
          f"{len(func_pos_set)} functional positions")

    # ---------- 2. 遍历比对文件 ----------
    aln_files = sorted(glob.glob(os.path.join(aln_dir, '*1rxx_chainA.aln')))
    print(f"Found {len(aln_files)} alignment files\n")

    detail_rows = []
    seq_rows= []
    skipped     = 0

    for aln_file in aln_files:
        fname = os.path.basename(aln_file).replace('.aln', '')
        # query名称: '@'前面部分
        query_name = fname.split('@')[0] if '@' in fname else fname

        # 2a. 解析比对
        info = parse_usalign_aln(aln_file)
        if info is None:
            print(f"  ⚠ 比对解析失败: {fname}")
            skipped += 1
            continue

        # 2b. 查找query PDB
        query_pdb_path = None
        for ext in PDB_EXTENSIONS:
            candidate = os.path.join(query_pdb_dir, query_name + ext)
            if os.path.isfile(candidate):
                query_pdb_path = candidate
                break

        if query_pdb_path is None:
            print(f"  ⚠ 未找到 query PDB: {query_name}")
            skipped += 1
            continue

        # 2c. 读取 query 序列 + pLDDT
        query_seq, plddts = read_query_pdb_plddt(query_pdb_path, QUERY_PDB_CHAIN)
        if len(query_seq) == 0:
            print(f"  ⚠ query PDB 序列为空: {query_name}")
            skipped += 1
            continue

        # 2d. 构建映射
        pos_mapping = build_position_mapping(
            info['seq1'], info['seq2'], info['match'], ref_index_map
        )
        q2r_map = build_query_to_ref_map(pos_mapping)

        # 2e. 赋标签
        # def assign_token_labels(query_len, q2r_map, adi_pos_set, plddts, plddt_threshold=70.0):
        labels = assign_token_labels(
            len(query_seq), q2r_map, adi_pos_set,
            catalytic_triads, substrate_binding, regulatory_sites,
            plddts, PLDDT_THRESHOLD
        )

        # labels = assign_token_labels(
        #     len(query_seq), q2r_map, adi_pos_set, func_pos_set,
        #     plddts, PLDDT_THRESHOLD,
        # )
        annotations = build_detail_annotation(
            len(query_seq), q2r_map, POS_TO_REGION, func_detail,
        )

        # 2f. 收集 per-residue 详细数据
        tm= info.get('tm_score', '')
        rms = info.get('rmsd', '')
        sid = info.get('seq_id', '')

        for i in range(len(query_seq)):
            qpos_1 = i + 1
            ref_pos = q2r_map.get(qpos_1, '')
            ref_aa  = ''
            quality = ''
            if ref_pos != '' and ref_pos in pos_mapping:
                ref_aa  = pos_mapping[ref_pos]['ref_aa']
                quality = pos_mapping[ref_pos]['quality']

            # detail_rows.append({
            #     'file':query_name,
            #     'tm_score':    tm,
            #     'rmsd':        rms,
            #     'seq_id':      sid,
            #     'query_pos':   qpos_1,
            #     'query_aa':    query_seq[i],
            #     'plddt':       round(plddts[i], 2),
            #     'ref_pdb_pos': ref_pos,
            #     'ref_aa':      ref_aa,
            #     'quality':     quality,
            #     'label':       labels[i],
            #     'label_name':  LABEL_NAMES.get(labels[i], ''),
            #     'annotation':  annotations[i],
            # })
            detail_rows.append({
                'file': query_name,
                'tm_score':    tm,
                'rmsd':        rms,
                'seq_id':      sid,
                'query_pos':   qpos_1,
                'query_aa':    query_seq[i],
                'plddt':       round(plddts[i], 2),
                'ref_pdb_pos': ref_pos,
                'ref_aa':      ref_aa,
                'quality':     quality,
                'label':       labels[i],        # 这里是字符串，比如 "1", "2", "1-3", "-1"
                'label_name':  LABEL_NAMES.get(int(labels[i].split('-')[0]) if labels[i] != "-1" else -1, ''),
                'annotation':  annotations[i],
            })

        # 2g. 收集 per-sequence 汇总
        label_str  = ' '.join(str(l) for l in labels)
        plddt_str  = ' '.join(f'{p:.1f}' for p in plddts)
        n_total    = len(labels)
        n_masked   = labels.count(-1)
        n_bg       = labels.count(0)
        n_adi      = labels.count(1)
        n_func     = labels.count(2)
        n_both     = labels.count(3)

        seq_rows.append({
            'file':       query_name,
            'tm_score':   tm,
            'rmsd':       rms,
            'seq_id':     sid,
            'seq_len':    n_total,
            'sequence':   query_seq,
            'labels':     label_str,
            'plddts':     plddt_str,
            'n_masked':   n_masked,
            'n_background': n_bg,
            'n_adi':      n_adi,
            'n_func':     n_func,
            'n_both':     n_both,
        })

    # ---------- 3. 写 CSV ----------
    if detail_rows:
        detail_fields = [
            'file', 'tm_score', 'rmsd', 'seq_id',
            'query_pos', 'query_aa', 'plddt',
            'ref_pdb_pos', 'ref_aa', 'quality',
            'label', 'label_name', 'annotation',
        ]
        with open(output_detail, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=detail_fields)
            w.writeheader()
            w.writerows(detail_rows)
        print(f"Per-residue 详细标注 → {output_detail}")

    if seq_rows:
        seq_fields = [
            'file', 'tm_score', 'rmsd', 'seq_id', 'seq_len',
            'sequence', 'labels', 'plddts',
            'n_masked', 'n_background', 'n_adi', 'n_func', 'n_both',
        ]
        with open(output_seq, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=seq_fields)
            w.writeheader()
            w.writerows(seq_rows)
        print(f"Per-sequence 标注 → {output_seq}")

    # ---------- 4. 汇总 ----------
    print(f"\n{'='*50}")
    print(f"处理: {len(aln_files)} 文件, 成功: {len(seq_rows)}, 跳过: {skipped}")
    if seq_rows:
        total_tokens = sum(r['seq_len'] for r in seq_rows)
        total_adi    = sum(r['n_adi'] for r in seq_rows)
        total_func   = sum(r['n_func'] for r in seq_rows)
        total_both   = sum(r['n_both'] for r in seq_rows)
        total_masked = sum(r['n_masked'] for r in seq_rows)
        print(f"总 tokens: {total_tokens}")
        print(f"  label 0(background):  {total_tokens - total_adi - total_func - total_both - total_masked}")
        print(f"  label 1 (ADI region):  {total_adi}")
        print(f"  label 2 (func site):   {total_func}")
        print(f"  label 3 (both):        {total_both}")
        print(f"  label -1 (masked):     {total_masked}")
        print(f"  pLDDT threshold:       {PLDDT_THRESHOLD}")


if __name__ == "__main__":
    main()
