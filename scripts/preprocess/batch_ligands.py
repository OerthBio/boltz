#!/usr/bin/env python3
"""
Batch-generate Boltz YAMLs from a template and a .smi list by remapping ligand atom names.
- Preserves comments and order (ruamel.yaml round-trip).
- Maps template constraint atom names (token2) to new ligands (Boltz-style atom naming).
- Writes one YAML per SMILES/ID to --outdir.
- Enforces exact inline format for token lists:
    token1: [A, 185]
    token2: [C, 'C93']
  via a small text post-processor (version-proof against ruamel quirks).

Usage:
  python boltz_batch_from_template.py \
      --template TEMPLATE.yaml \
      --smi ligands.smi \
      --outdir out_yaml \
      [--chain C]
"""

from __future__ import annotations
import sys
import os
import io
import re
import argparse
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Tuple, Union

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS
from rdkit.Chem.MolStandardize import rdMolStandardize
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

# ----------------------------
# Constants / regex
# ----------------------------
ATOMNAME_RE = re.compile(r"^[A-Z]{1,2}\d{1,3}$")

# ----------------------------
# RDKit: standardize and name atoms exactly as Boltz does
# ----------------------------

def _boltz_like_standardize(smiles: str) -> str:
    """
    Conservative but robust standardization akin to ChEMBL curation.
    Replace with your exact Boltz 'standardize()' if available.
    """
    chooser = rdMolStandardize.LargestFragmentChooser()
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"SMILES failed to parse: {smiles}")
    mol = chooser.choose(mol)
    Chem.SanitizeMol(mol)
    normalizer = rdMolStandardize.Normalizer()
    uncharger = rdMolStandardize.Uncharger()
    reion = rdMolStandardize.Reionizer()
    mol = normalizer.normalize(mol)
    mol = uncharger.uncharge(mol)
    mol = reion.reionize(mol)
    return Chem.MolToSmiles(mol, canonical=True)

STANDARDIZE_FN = _boltz_like_standardize  # point to Boltz's exact function if you have it


def make_boltz_named_mol(smiles: str) -> Tuple[Chem.Mol, Chem.Mol]:
    """
    Returns (mol_with_Hs, mol_no_H) with atom property 'name' set to the Boltz style:
      name = SYMBOL.upper() + (canonical_rank+1)
    """
    std = STANDARDIZE_FN(smiles)
    mol = AllChem.MolFromSmiles(std)
    if mol is None:
        raise ValueError(f"Standardized SMILES failed to parse: {std}")

    mol_H = AllChem.AddHs(mol)
    canonical_order = AllChem.CanonicalRankAtoms(mol_H)
    Chem.AssignStereochemistry(mol_H, force=True, cleanIt=True)

    for atom, can_idx in zip(mol_H.GetAtoms(), canonical_order):
        atom_name = atom.GetSymbol().upper() + str(can_idx + 1)
        if len(atom_name) > 4:
            raise ValueError(f"Atom name too long for Boltz: {atom_name}")
        atom.SetProp("name", atom_name)

    mol_no_H = AllChem.RemoveHs(mol_H, sanitize=False)
    for a in mol_no_H.GetAtoms():
        if not a.HasProp("name"):
            raise RuntimeError("Missing 'name' after RemoveHs.")
    return mol_H, mol_no_H


def _names_on_heavy(m: Chem.Mol) -> List[str]:
    return [a.GetProp("name") for a in m.GetAtoms()]


def _best_substructure_match(t: Chem.Mol, q: Chem.Mol,
                             prefer: Optional[List[int]] = None) -> Optional[Tuple[int, ...]]:
    params = Chem.SubstructMatchParameters()
    params.useChirality = True
    params.useQueryQueryMatches = True
    matches = q.GetSubstructMatches(t, params)
    if not matches:
        return None
    if len(matches) == 1 or not prefer:
        return matches[0]

    def score(match: Tuple[int, ...]) -> int:
        s = 0
        for ti in prefer:
            qi = match[ti]
            at_t = t.GetAtomWithIdx(ti)
            at_q = q.GetAtomWithIdx(qi)
            s += int(at_t.GetAtomicNum() == at_q.GetAtomicNum())
            s += int(at_t.GetDegree() == at_q.GetDegree())
            s += int(at_t.IsInRing() == at_q.IsInRing())
            s += int(at_t.GetFormalCharge() == at_q.GetFormalCharge())
        return s

    return max(matches, key=score)


def _mcs_match(t: Chem.Mol, q: Chem.Mol) -> Optional[Tuple[int, ...]]:
    p = rdFMCS.MCSParameters()
    p.Timeout = 10
    p.AtomCompare = rdFMCS.AtomCompare.CompareElements
    p.BondCompare = rdFMCS.BondCompare.CompareOrder
    p.MatchValences = True
    p.RingMatchesRingOnly = True
    p.CompleteRingsOnly = False
    p.MatchChiralTag = True
    res = rdFMCS.FindMCS([t, q], p)
    if not res or not res.smartsString:
        return None
    core = Chem.MolFromSmarts(res.smartsString)
    if core is None:
        return None
    t_matches = t.GetSubstructMatches(core, useChirality=True)
    q_matches = q.GetSubstructMatches(core, useChirality=True)
    if not t_matches or not q_matches:
        return None
    t_core, q_core = t_matches[0], q_matches[0]
    core_map = {ti: qi for ti, qi in zip(t_core, q_core)}
    return tuple(core_map.get(i, -1) for i in range(t.GetNumAtoms()))


def map_template_atoms_to_new_names(template_smiles: str,
                                    new_smiles: str,
                                    template_atom_names: Iterable[str]) -> Dict[str, Optional[str]]:
    _, t = make_boltz_named_mol(template_smiles)
    _, q = make_boltz_named_mol(new_smiles)
    q_names = _names_on_heavy(q)
    name_to_idx = {a.GetProp("name"): a.GetIdx() for a in t.GetAtoms()}

    prefer = [name_to_idx[n] for n in template_atom_names if n in name_to_idx]
    match = _best_substructure_match(t, q, prefer)
    if match is None:
        match = _mcs_match(t, q)
        if match is None:
            return {n: None for n in template_atom_names}

    out: Dict[str, Optional[str]] = {}
    for n in template_atom_names:
        ti = name_to_idx.get(n, None)
        if ti is None or ti >= len(match):
            out[n] = None
        else:
            qi = match[ti]
            out[n] = q_names[qi] if (qi is not None and qi >= 0) else None
    return out

# ----------------------------
# YAML: load, navigate, write (with final formatting fix)
# ----------------------------

def _yaml_loader() -> YAML:
    y = YAML(typ="rt")  # round-trip: preserve comments & order
    y.preserve_quotes = True
    y.width = 4096
    return y

_yaml = _yaml_loader()

def _read_yaml(path: str) -> CommentedMap:
    with open(path, "r", encoding="utf-8") as f:
        return _yaml.load(f)

def _collapse_token_block_lists(yaml_text: str) -> str:
    """
    Collapse block-style 2-item lists for token1/token2 to flow style.
    Keeps all other content untouched.
    """
    lines = yaml_text.splitlines(keepends=True)
    out = []
    i = 0
    n = len(lines)

    def is_dash_item(line):
        m = re.match(r"^(\s*)-\s+(.*?)(\s*)\n?$", line)
        if not m:
            return None
        indent, item, _ = m.groups()
        return indent, item

    while i < n:
        m_key = re.match(r"^(\s*)(token1|token2):\s*(#.*)?\n?$", lines[i])
        if not m_key:
            out.append(lines[i]); i += 1; continue

        base_indent, key, _cmt = m_key.groups()
        if i + 2 >= n:
            out.append(lines[i]); i += 1; continue

        d1 = is_dash_item(lines[i+1])
        d2 = is_dash_item(lines[i+2])
        if not d1 or not d2:
            out.append(lines[i]); i += 1; continue

        indent1, item1 = d1
        indent2, item2 = d2
        if len(indent1) < len(base_indent) or len(indent2) < len(base_indent):
            out.append(lines[i]); i += 1; continue

        # token2 second element should be quoted if it looks like an atom name
        if key == "token2":
            if not (item2.startswith("'") or item2.startswith('"')):
                if ATOMNAME_RE.match(item2):
                    item2 = f"'{item2}'"

        out.append(f"{base_indent}{key}: [{item1}, {item2}]\n")
        i += 3
    return "".join(out)

_yaml_dump = YAML(typ="rt")
_yaml_dump.preserve_quotes = True
_yaml_dump.width = 4096

def _write_yaml(doc: CommentedMap, path: str) -> None:
    buf = io.StringIO()
    _yaml_dump.dump(doc, buf)
    text = buf.getvalue()
    text = _collapse_token_block_lists(text)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# ----------------------------
# YAML navigation helpers
# ----------------------------

def _first_underscore_prefix(filename: str) -> str:
    base = os.path.basename(filename)
    return base.split("_", 1)[0] if "_" in base else os.path.splitext(base)[0]

def _find_ligand_smiles_node(doc: CommentedMap, chain: str = "C") -> Tuple[List[Union[str,int]], str]:
    """
    Return (node_path, smiles_str) where 'smiles' lives.
    Tries common locations then DFS.
    """
    try_paths = [
        ["chains", chain, "ligand", "smiles"],
        ["ligand", "smiles"],
        ["ligands", chain, "smiles"],
    ]
    for p in try_paths:
        cur = doc; ok = True
        for k in p:
            if isinstance(cur, dict) and k in cur: cur = cur[k]
            else: ok = False; break
        if ok and isinstance(cur, str):
            return p, cur

    # Fallback DFS
    def dfs(path, node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "smiles" and isinstance(v, str):
                    return path + [k], v
                hit = dfs(path + [k], v)
                if hit: return hit
        elif isinstance(node, list):
            for i, v in enumerate(node):
                hit = dfs(path + [i], v)
                if hit: return hit
        return None

    hit = dfs([], doc)
    if hit: return hit
    raise KeyError("Could not locate a ligand 'smiles' field in template YAML.")

def _get_by_path(doc, path):
    cur = doc
    for p in path:
        cur = cur[p]
    return cur

def _set_by_path(doc, path, value):
    cur = doc
    for p in path[:-1]:
        cur = cur[p]
    cur[path[-1]] = value

def _iter_constraints_lists(doc: CommentedMap):
    def dfs(path, node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "constraints" and isinstance(v, list):
                    yield (path + [k], v)
                yield from dfs(path + [k], v)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                yield from dfs(path + [i], v)
    yield from dfs([], doc)

def _iter_token_nodes(doc: CommentedMap, key: str):
    """Yield (path, value) for every <key> under any 'constraints' list."""
    for cl_path, clist in _iter_constraints_lists(doc):
        for idx, item in enumerate(clist):
            def dfs_item(path, node):
                if isinstance(node, dict):
                    for k, v in node.items():
                        if k == key:
                            yield (path + [k], v)
                        else:
                            yield from dfs_item(path + [k], v)
                elif isinstance(node, list):
                    for i, v in enumerate(node):
                        yield from dfs_item(path + [i], v)
            yield from dfs_item(cl_path + [idx], item)

def _iter_token2_nodes(doc: CommentedMap):
    yield from _iter_token_nodes(doc, "token2")

def _infer_template_ligand_chain_from_constraints(doc: CommentedMap) -> Optional[str]:
    chains = set()
    for _, tok in _iter_token2_nodes(doc):
        if isinstance(tok, list) and len(tok) >= 2 and isinstance(tok[0], str) and isinstance(tok[1], str):
            if ATOMNAME_RE.match(tok[1]):
                chains.add(tok[0])
    return next(iter(chains)) if len(chains) == 1 else None

def _collect_template_constraint_atomnames(doc: CommentedMap, template_chain: str) -> List[str]:
    """
    Collect unique atom names from token2 targeting the template ligand chain.
    Accepts token2 as 'C93' or [chain, 'C93'].
    """
    names, seen = [], set()
    for _, tok in _iter_token2_nodes(doc):
        if isinstance(tok, str) and ATOMNAME_RE.match(tok):
            if tok not in seen:
                names.append(tok); seen.add(tok)
        elif isinstance(tok, list) and len(tok) >= 2:
            chain_id, atom = tok[0], tok[1]
            if isinstance(chain_id, str) and isinstance(atom, str) and ATOMNAME_RE.match(atom):
                if chain_id == template_chain and atom not in seen:
                    names.append(atom); seen.add(atom)
    return names

def _rewrite_binder_chain(doc: CommentedMap, old_chain: str, new_chain: str) -> None:
    """Flip 'binder: <old>' → 'binder: <new>' anywhere under constraints/pocket."""
    def dfs(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "binder" and isinstance(v, str) and v == old_chain:
                    node[k] = new_chain
                else:
                    dfs(v)
        elif isinstance(node, list):
            for v in node:
                dfs(v)
    if old_chain != new_chain:
        dfs(doc)

# ----------------------------
# Remapping logic (no styling tricks; formatting handled post-dump)
# ----------------------------

def _remap_constraints_token2(doc: CommentedMap,
                              name_map: Dict[str, Optional[str]],
                              template_chain: str,
                              target_chain: str) -> List[str]:
    """
    For each token2:
      - If scalar 'C93': map to name_map if possible; then set to [target_chain, atom].
      - If list [chain, 'C93'] and chain == template_chain: map and set [target_chain, mapped].
      - Non-ligand token2 are left unchanged.
    The writer collapses block lists to inline flow form after dumping.
    """
    warnings: List[str] = []
    for path, tok in _iter_token2_nodes(doc):
        # Scalar atom name -> normalize to list
        if isinstance(tok, str) and ATOMNAME_RE.match(tok):
            mapped = name_map.get(tok)
            atom = mapped if mapped else tok
            _set_by_path(doc, path, [target_chain, atom])
            if mapped is None and tok in name_map:
                warnings.append(f"Unmapped token2 '{tok}' (kept).")
            continue

        # List form -> [chain, 'atom']
        if isinstance(tok, list) and len(tok) >= 2:
            chain_id, atom = tok[0], tok[1]
            if isinstance(chain_id, str) and isinstance(atom, str) and ATOMNAME_RE.match(atom):
                if chain_id == template_chain:
                    mapped = name_map.get(atom)
                    atom2 = mapped if mapped else atom
                    _set_by_path(doc, path, [target_chain, atom2])
                    if mapped is None and atom in name_map:
                        warnings.append(f"Unmapped token2 '{chain_id},{atom}' (kept).")
            # else: leave untouched
    return warnings

# ----------------------------
# .smi I/O
# ----------------------------

def _read_smi(path: str) -> List[Tuple[str, str]]:
    """
    Read a .smi: lines of "SMILES ID" (whitespace-separated). If ID missing, auto-number.
    """
    out: List[Tuple[str, str]] = []
    auto = 1
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 1:
                smi, ID = parts[0], f"{auto:06d}"; auto += 1
            else:
                smi, ID = parts[0], parts[1]
            out.append((smi, ID))
    if not out:
        raise ValueError("No ligands found in .smi file.")
    return out

# ----------------------------
# Main workflow
# ----------------------------

def _process(template_yaml: str, smi_path: str, outdir: str, target_chain: str = "C") -> None:
    os.makedirs(outdir, exist_ok=True)
    template_doc = _read_yaml(template_yaml)

    tmpl_chain = _infer_template_ligand_chain_from_constraints(template_doc) or target_chain
    try:
        smiles_path, template_smiles = _find_ligand_smiles_node(template_doc, tmpl_chain)
    except KeyError:
        smiles_path, template_smiles = _find_ligand_smiles_node(template_doc, target_chain)

    template_names = _collect_template_constraint_atomnames(template_doc, tmpl_chain)
    prefix = _first_underscore_prefix(template_yaml)
    ligands = _read_smi(smi_path)

    for smi, ID in ligands:
        doc = deepcopy(template_doc)
        _set_by_path(doc, smiles_path, smi)

        if template_names:
            try:
                name_map = map_template_atoms_to_new_names(template_smiles, smi, template_names)
            except Exception as e:
                sys.stderr.write(f"[WARN] Mapping failed for {ID}: {e}\n")
                name_map = {n: None for n in template_names}

            warns = _remap_constraints_token2(doc, name_map, tmpl_chain, target_chain)
            for w in warns:
                sys.stderr.write(f"[WARN] {ID}: {w}\n")

        _rewrite_binder_chain(doc, tmpl_chain, target_chain)

        out_path = os.path.join(outdir, f"{prefix}_{ID}.yaml")
        _write_yaml(doc, out_path)
        print(f"Wrote {out_path}")

# ----------------------------
# CLI
# ----------------------------

def _parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Batch-generate Boltz YAMLs with remapped ligand atom names.")
    ap.add_argument("--template", required=True, help="Template YAML (Boltz format).")
    ap.add_argument("--smi",      required=True, help="Ligands .smi file (SMILES ID per line).")
    ap.add_argument("--outdir",   required=True, help="Output directory.")
    ap.add_argument("--chain",    default="C",   help="Target ligand chain (default: C).")
    return ap.parse_args(argv)

def main(argv: List[str]) -> int:
    args = _parse_args(argv)
    try:
        _process(args.template, args.smi, args.outdir, target_chain=args.chain)
    except Exception as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

