import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Crippen, Descriptors3D
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator
import networkx as nx
import requests
from pathlib import Path
import argparse
import time
import json
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import os
import multiprocessing as mp

# ----------------------------------------------------------------------
# -------------- STAR ATOM HANDLING (robust original style) ------------
# ----------------------------------------------------------------------

def process_star_atoms(mol):
    """
    Replace all wildcard atoms (‘*’ or atomicNum == 0) with **Astatine (At)**.
    Astatine has atomic number 85.
    """
    ATOMIC_NUM_AT = 85  # Astatine

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(ATOMIC_NUM_AT)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            atom.SetAtomicNum(ATOMIC_NUM_AT)

    return mol

# ----------------------------------------------------------------------
# -------------- SINGLE POLYMER PROCESSING -----------------------------
# ----------------------------------------------------------------------

def process_single_polymer(args):
    idx, row_dict, extractor = args
    polymer_data = None
    failed_info = None
    try:
        smiles = row_dict['psmiles']
        source = row_dict['source']

        if pd.isna(smiles) or not isinstance(smiles, str) or len(smiles.strip()) == 0:
            failed_info = {'index': idx, 'smiles': str(smiles), 'error': 'Empty or invalid SMILES'}
            return polymer_data, failed_info

        canonical_smiles = extractor.validate_and_standardize_smiles(smiles)
        if canonical_smiles is None:
            failed_info = {'index': idx, 'smiles': smiles, 'error': 'Invalid SMILES or contains wildcards'}
            return polymer_data, failed_info

        polymer_data = {
            'original_index': idx,
            'psmiles': canonical_smiles,
            'source': source,
            'smiles': canonical_smiles
        }

        try:
            graph_data = extractor.generate_molecular_graph(canonical_smiles)
            polymer_data['graph'] = graph_data
        except Exception:
            polymer_data['graph'] = {}

        try:
            geometry_data = extractor.optimize_3d_geometry(canonical_smiles)
            polymer_data['geometry'] = geometry_data
        except Exception:
            polymer_data['geometry'] = {}

        try:
            fingerprint_data = extractor.calculate_morgan_fingerprints(canonical_smiles)
            polymer_data['fingerprints'] = fingerprint_data
        except Exception:
            polymer_data['fingerprints'] = {}

        return polymer_data, failed_info

    except Exception as e:
        failed_info = {'index': idx, 'smiles': row_dict.get('psmiles', ''), 'error': str(e)}
        return polymer_data, failed_info

# ----------------------------------------------------------------------
# -------------- MAIN MULTIMODAL EXTRACTOR -----------------------------
# ----------------------------------------------------------------------

class AdvancedPolymerMultimodalExtractor:
    def __init__(self, csv_file: str):
        self.csv_file = csv_file

    # ---------- SMILES VALIDATION & STANDARDIZATION ----------
    def validate_and_standardize_smiles(self, smiles: str) -> Optional[str]:
        try:
            if not smiles or pd.isna(smiles):
                return None
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            mol = process_star_atoms(mol)         # CONVERT * to Astatine (At)
            Chem.SanitizeMol(mol)
            mol = process_star_atoms(mol)         # SECOND PASS (robust)
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            if len(canonical_smiles) == 0:
                return None
            return canonical_smiles
        except Exception:
            return None

    # ---------- POLYMER VALIDITY CHECKS ----------
    def _has_invalid_polymer_features(self, mol) -> bool:
        try:
            if mol.GetNumAtoms() > 200:
                return True
            for atom in mol.GetAtoms():
                if atom.GetFormalCharge() > 5 or atom.GetFormalCharge() < -5:
                    return True
            return False
        except:
            return True

    def _is_valid_polymer(self, mol) -> bool:
        num_atoms = mol.GetNumAtoms()
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        return num_atoms > 10 or num_rings > 1

    # ---------- MOLECULAR GRAPH GENERATION ----------
    def generate_molecular_graph(self, smiles: str) -> Dict:
        mol = Chem.MolFromSmiles(smiles)
        mol = process_star_atoms(mol)  # Ensure no stars left
        if mol is None:
            return {}

        mol = Chem.AddHs(mol)          # Explicit hydrogens (unchanged)

        node_features = []
        for atom in mol.GetAtoms():
            node_features.append({
                'atomic_num': atom.GetAtomicNum(),
                'degree': atom.GetDegree(),
                'formal_charge': atom.GetFormalCharge(),
                'hybridization': int(atom.GetHybridization()),
                'is_aromatic': atom.GetIsAromatic(),
                'is_in_ring': atom.IsInRing(),
                'chirality': int(atom.GetChiralTag()),
                'mass': atom.GetMass(),
                'valence': atom.GetTotalValence(),
                'num_radical_electrons': atom.GetNumRadicalElectrons()
            })
        edge_features = []
        edge_indices = []
        for bond in mol.GetBonds():
            start_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()
            edge_features.append({
                'bond_type': int(bond.GetBondType()),
                'is_aromatic': bond.GetIsAromatic(),
                'is_in_ring': bond.IsInRing(),
                'stereo': int(bond.GetStereo()),
                'is_conjugated': bond.GetIsConjugated()
            })
            edge_indices.extend([[start_atom, end_atom], [end_atom, start_atom]])
        graph_features = {
            'num_atoms': mol.GetNumAtoms(),
            'num_bonds': mol.GetNumBonds(),
            'num_rings': rdMolDescriptors.CalcNumRings(mol),
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Crippen.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'num_h_acceptors': rdMolDescriptors.CalcNumHBA(mol),
            'num_h_donors': rdMolDescriptors.CalcNumHBD(mol)
        }
        adj = Chem.GetAdjacencyMatrix(mol).tolist()
        return {
            'node_features': node_features,
            'edge_features': edge_features,
            'edge_indices': edge_indices,
            'graph_features': graph_features,
            'adjacency_matrix': adj
        }

    # ---------- 3-D GEOMETRY ----------
    def optimize_3d_geometry(self, smiles: str, num_conformers: int = 10) -> Dict:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() > 200:
            return {}
        mol = process_star_atoms(mol)
        mol_h = Chem.AddHs(mol)  # explicit hydrogens

        # Collect atomic numbers (matches the order in coordinates)
        atomic_numbers = [atom.GetAtomicNum() for atom in mol_h.GetAtoms()]

        try:
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            conformer_ids = AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conformers, params=params)
        except Exception:
            conformer_ids = []

        best_conformer = None
        best_energy = float('inf')

        for conf_id in conformer_ids:
            try:
                mmff_ok = AllChem.MMFFHasAllMoleculeParams(mol_h)
                if mmff_ok:
                    AllChem.MMFFOptimizeMolecule(mol_h, confId=conf_id)
                    props = AllChem.MMFFGetMoleculeProperties(mol_h)
                    ff = AllChem.MMFFGetMoleculeForceField(mol_h, props, confId=conf_id)
                    energy = ff.CalcEnergy() if ff is not None else None
                else:
                    AllChem.UFFOptimizeMolecule(mol_h, confId=conf_id)
                    ff = AllChem.UFFGetMoleculeForceField(mol_h, confId=conf_id)
                    energy = ff.CalcEnergy() if ff is not None else None

                if energy is not None and energy < best_energy:
                    conf = mol_h.GetConformer(conf_id)
                    coords = [
                        [conf.GetAtomPosition(i).x,
                         conf.GetAtomPosition(i).y,
                         conf.GetAtomPosition(i).z]
                        for i in range(mol_h.GetNumAtoms())
                    ]
                    descriptors_3d = {}
                    try:
                        descriptors_3d = {
                            'asphericity': Descriptors3D.Asphericity(mol_h, confId=conf_id),
                            'eccentricity': Descriptors3D.Eccentricity(mol_h, confId=conf_id),
                            'inertial_shape_factor': Descriptors3D.InertialShapeFactor(mol_h, confId=conf_id),
                            'radius_of_gyration': Descriptors3D.RadiusOfGyration(mol_h, confId=conf_id),
                            'spherocity_index': Descriptors3D.SpherocityIndex(mol_h, confId=conf_id)
                        }
                    except Exception:
                        pass

                    best_conformer = {
                        'conformer_id': conf_id,
                        'coordinates': coords,
                        'atomic_numbers': atomic_numbers,
                        'energy': energy,
                        'descriptors_3d': descriptors_3d
                    }
                    best_energy = energy

            except Exception:
                continue

        if best_conformer is not None:
            return {
                'best_conformer': best_conformer,
                'num_conformers_generated': len(conformer_ids),
                'converted_smiles': Chem.MolToSmiles(mol)
            }

        # Fallback 2-D coordinates
        try:
            rdDepictor.Compute2DCoords(mol)
            coords_2d = mol.GetConformer().GetPositions().tolist()
            # match the atomic_numbers to 2D duplicate (should have same order)
            atomic_numbers_2d = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            return {
                'best_conformer': {
                    'conformer_id': -1,
                    'coordinates': coords_2d,
                    'atomic_numbers': atomic_numbers_2d,
                    'energy': None,
                    'descriptors_3d': {},
                },
                'num_conformers_generated': 0,
                'converted_smiles': Chem.MolToSmiles(mol)
            }
        except Exception:
            return {}

    # ---------- MORGAN FINGERPRINTS ----------
    def calculate_morgan_fingerprints(self, smiles: str, radius: int = 3, n_bits: int = 2048) -> Dict:
        mol = Chem.MolFromSmiles(smiles)
        mol = process_star_atoms(mol)
        if mol is None:
            return {}
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp_bitvect = generator.GetFingerprint(mol)
        fingerprints = {
            f'morgan_r{radius}_bits': list(fp_bitvect.ToBitString()),
            f'morgan_r{radius}_counts': dict(AllChem.GetMorganFingerprint(mol, radius).GetNonzeroElements()),
        }
        # Extended multi-radius support
        for r in range(1, radius):
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=r, fpSize=n_bits)
            bitvect = gen.GetFingerprint(mol)
            fingerprints[f'morgan_r{r}_bits'] = list(bitvect.ToBitString())
            counts = AllChem.GetMorganFingerprint(mol, r).GetNonzeroElements()
            fingerprints[f'morgan_r{r}_counts'] = dict(counts)
        return fingerprints

    # ---------- PARALLEL PROCESSING ----------
    def process_all_polymers_parallel(self, chunk_size: int = 100, num_workers: int = 40):
        chunk_iterator = pd.read_csv(self.csv_file, chunksize=chunk_size, engine='python')

        for chunk in chunk_iterator:
            for col in ['graph', 'geometry', 'fingerprints']:
                if col not in chunk.columns:
                    chunk[col] = None
                chunk[col] = chunk[col].astype(object)

            chunk_to_process = chunk[
                chunk[['graph', 'geometry', 'fingerprints']].isnull().any(axis=1)
            ].copy()
            if len(chunk_to_process) == 0:
                self.save_chunk_to_csv(chunk)
                continue

            rows = list(chunk_to_process.iterrows())
            argslist = [(i, row.to_dict(), self) for i, row in rows]
            with mp.Pool(num_workers) as pool:
                results = pool.map(process_single_polymer, argslist)

            failed_list = []
            for n, (output, fail) in enumerate(results):
                idx = rows[n][0]
                if output:
                    chunk.at[idx, 'graph'] = json.dumps(output['graph'])
                    chunk.at[idx, 'geometry'] = json.dumps(output['geometry'])
                    chunk.at[idx, 'fingerprints'] = json.dumps(output['fingerprints'])
                if fail:
                    failed_list.append(fail)

            self.save_chunk_to_csv(chunk)
            self.save_failed_to_json(failed_list)

        return "Processing Done"

    # ---------- SAVE HELPERS ----------
    def save_chunk_to_csv(self, chunk):
        out_csv = self.csv_file.replace('.csv', '_processed.csv')
        if not os.path.exists(out_csv):
            chunk.to_csv(out_csv, index=False, mode='w')
        else:
            chunk.to_csv(out_csv, index=False, mode='a', header=False)

    def save_failed_to_json(self, failed_list):
        if not failed_list:
            return
        fail_json = self.csv_file.replace('.csv', '_failures.jsonl')
        with open(fail_json, 'a') as f:
            for fail in failed_list:
                json.dump(fail, f)
                f.write('\n')

    # ---------- OPTIONAL RESULT SAVER (stub) ----------
    def save_results(self, output_file: str = 'polymer_multimodal_data.json'):
        pass

    # ---------- OPTIONAL SUMMARY (stub) ----------
    def generate_summary_statistics(self) -> Dict:
        return {}

# ----------------------------------------------------------------------
# -------------- SCRIPT ENTRY POINT ------------------------------------
# ----------------------------------------------------------------------

def main():
    csv_file = "Polymer_Foundational_Model/polymer_structures_unified.csv"
    extractor = AdvancedPolymerMultimodalExtractor(csv_file)
    try:
        extractor.process_all_polymers_parallel(chunk_size=1000, num_workers=24)
    except KeyboardInterrupt:
        return extractor, None
    except Exception as e:
        print(f"CRASH! Error: {e}")
        return extractor, None
    print("\n=== Processing Complete ===")
    return extractor, None

if __name__ == "__main__":
    extractor, results = main()
