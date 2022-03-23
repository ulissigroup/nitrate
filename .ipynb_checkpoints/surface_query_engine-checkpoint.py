# from atomate.vasp.drones import VaspDrone 
from pymatgen.db import QueryEngine
import json, random, string, os, glob
import numpy as np

from pymatgen.core.structure import Structure, Molecule, Composition
from pymatgen.core.surface import Slab
from pymatgen.analysis.surface_analysis import *

from pymongo import MongoClient


def write_metadata_json(structure, calc_type, fname=None, name_of_adsorbate=None, 
                        entry_id=None, database=None, bulk_rid=None, 
                        slab_rid=None, ads_rid=None, additional_data=None, 
                        functional=None, custom_rid=None):
    
    rid = custom_rid if custom_rid else ''.join(random.choice(string.ascii_lowercase + string.ascii_uppercase \
                                                              + string.digits) for _ in range(20))
    metadata = {}
    d = str(structure.as_dict())
    
    bulk = structure if calc_type == 'bulk' else structure.oriented_unit_cell
    bulk_formula = bulk.composition.reduced_formula
    bulk_composition = bulk.composition.as_dict()
    bulk_chemsys = bulk.composition.chemical_system

    if calc_type == 'adsorbate_in_a_box':
        name_of_adsorbate = structure.composition.to_pretty_string() \
        if not name_of_adsorbate else name_of_adsorbate 
        metadata = {'adsorbate': name_of_adsorbate, 'calc_type': calc_type, 
                    'rid': 'ads-%s' %(rid), 'init_pmg_structure': d, 'func': functional}

    if calc_type == 'bulk':
        metadata =  {'entry_id': entry_id, 'database': database, 
                     'rid': 'bulk-%s' %(rid), 'calc_type': calc_type, 
                     'init_pmg_structure': d, 'func': functional}

    if calc_type == 'bare_slab':
         metadata = {'entry_id': entry_id, 'database': database, 
                     'bulk_rid': bulk_rid, 'rid': 'slab-%s' %(rid), 
                     'miller_index': list(structure.miller_index), 
                     'bulk_formula': bulk_formula, 'bulk_composition': bulk_composition, 'bulk_chemsys': bulk_chemsys, 
                     'init_pmg_slab': d, 'calc_type': calc_type, 'func': functional}

    if calc_type == 'adsorbed_slab':
        metadata = {'entry_id': entry_id, 'database': database, 
                    'adsorbate': name_of_adsorbate,
                    'bulk_rid': bulk_rid, 'slab_rid': slab_rid, 
                    'ads_rid': ads_rid, 'rid': 'adslab-%s' %(rid), 
                    'miller_index': list(structure.miller_index), 
                    'bulk_formula': bulk_formula, 'bulk_composition': bulk_composition, 'bulk_chemsys': bulk_chemsys, 
                    'init_pmg_slab': d, 'calc_type': calc_type, 'func': functional}

    if additional_data:
        metadata.update(additional_data)
    if fname:
        with open(os.path.join(fname, 'metadata.json'), 'w') as outfile:
            outfile.write(json.dumps(metadata, indent=True))
        outfile.close()
    else:
        return metadata


class SurfaceQueryEngine(QueryEngine):

    """
    Collections:
        - CatPiss: NO3rr project for water purification
        - IndianPaleAle: Alcohol dehydrogenation project, adsorption of IsoPropyl Alcohol
        
    We should have 4 types of documents, each formatted in a standardized fashion: Adslab, Slab, 
        Adsorbate, and Bulk. All entries should have a random ID and the initial pre-relaxed structure
        - Adsorbate: Document describing the VASP calculation of an adsorbate in a box. All documents 
            must have a distinct ID associated with it. In addition to standard inputs parsed by the 
            TaskDrone, will also have the following metadata:
            {'adsorbate': name_of_adsorbate, 'adsorbate_id', 
            'calc_type': 'adsorbate_in_a_box', 'rid': 'ads-BGg3rg4gerG6'}
        - Bulk: Document describing the VASP calculation of a standard bulk structure. In addition to 
            standard inputs parsed by the TaskDrone, will also have the following metadata:
            {'entry_id': mpid, auid, etc, 'database': 'Materials_Project, AFLOW, OQMD etc', 
            rid: 'bulk-rgt4h65u45gg', 'calc_type': 'bulk'}
        - Slab: Document describing the VASP calculation of a Slab. In addition to 
            standard inputs parsed by the TaskDrone, will also have the following metadata:
            {'entry_id': mpid, auid, etc, 'database': 'Materials_Project, AFLOW, OQMD etc', 
            'bulk_rid': 'bulk-rgt4h65u45gg', 'rid': 'slab-eege4h4herhe4', 'miller_index': (1,1,1), 
            'Slab': pmg_slab_as_dict, 'calc_type': 'bare_slab'}
        - AdSlab: Document describing the VASP calculation of an adsorbed slab. In addition to 
            standard inputs parsed by the TaskDrone, will also have the following metadata:
            {'entry_id': mpid, auid, etc, 'database': 'Materials_Project, AFLOW, OQMD etc', 
            'bulk_rid': 'bulk-rgt4h65u45gg', 'slab_rid': 'slab-eege4h4herhe4', 
            'ads_rid': 'ads-BGg3rg4gerG6', 'rid': 'adslab-reg3g53g3h4h2hj204', 
            'miller_index': (1,1,1), 'Slab': pmg_slab_as_dict, 'calc_type': 'bare_slab'}
    """
    def __init__(self, dbconfig):


        conn = MongoClient(host=dbconfig["host"],
                           port=dbconfig["port"])
        db = conn.get_database(dbconfig["database"])
        db.authenticate(dbconfig["user"],
                        dbconfig["password"])

        surface_properties = db[dbconfig['collection']]

        self.surface_properties = surface_properties
        self.adsorbates_dict = self.get_all_adsorbate_entries_dict()

        super(SurfaceQueryEngine, self).__init__(**dbconfig)
        
    def get_structure_from_doc(self, doc):
        if doc['calc_type'] in ['adsorbate_in_a_box', 'bulk']:
            return Structure.from_dict(doc['init_pmg_structure'])
        else:
            return Slab.from_dict(doc['init_pmg_slab'])
        
    def get_adslab_docs_with_Eads(self, criteria, E_adsorbate=None, ads_rid=None):
        
        # makes sure we only get adsorbed entries
        if 'calc_type' not in criteria:
            criteria['calc_type'] = 'adsorbed_slab'
        criteria['calc_type'] = 'adsorbed_slab'
        
        adslab_entries = [entry for entry in self.surface_properties.find(criteria)]
        adslab_w_Eads = []
        for entry in adslab_entries:
            bare_slab_entry = self.surface_properties.find_one({'rid': entry['slab_rid']})
            if not bare_slab_entry:
                continue
            bare_slab = Structure.from_dict(bare_slab_entry['calcs_reversed'][0]['output']['structure'])
            A = self.get_surface_area(bare_slab)
            adslab = Structure.from_dict(entry['calcs_reversed'][0]['output']['structure'])
            A_ads = self.get_surface_area(adslab)
            n = A_ads/A
            
            E_adslab = entry['calcs_reversed'][0]['output']['energy']
            E_slab = bare_slab_entry['calcs_reversed'][0]['output']['energy']
            if not E_adsorbate:
                ads_entry = self.surface_properties.find_one({'rid': ads_rid}) if ads_rid\
                else self.surface_properties.find_one({'rid': entry['ads_rid']})
                E_adsorbate = ads_entry['energy'] if 'OC20' in ads_entry.values() \
                else ads_entry['calcs_reversed'][0]['output']['energy']

            entry['adsorption_energy'] = E_adslab - n*E_slab - E_adsorbate
            adslab_w_Eads.append(entry)
        return adslab_w_Eads
    
    def get_OC22_docs_with_Eads(self, criteria, E_adsorbate=None,
                                ads_rid=None, include_trajectories=False, 
                                normalize=False):
        
        ads_entries = self.adsorbates_dict
        
        if 'calc_type' not in criteria:
            criteria['calc_type'] = 'adsorbed_slab'
        criteria['calc_type'] = 'adsorbed_slab'
        
        # get all adsorbed slab entries
        entries = [entry for entry in self.surface_properties.find(criteria)]
        
        # now pre-load all bare slab entries
        slab_rids = []
        for entry in entries:
            if entry['slab_rid'] not in slab_rids:
                slab_rids.append(entry['slab_rid'])
        slab_rids_to_entry_dict = {}
        for rid in slab_rids:
            bare = self.surface_properties.find_one({'rid': rid})
            if bare:
                slab_rids_to_entry_dict[rid] = bare
        
        # now get the adsorption energies
        new_entries = []
        for adslab_entry in entries:
            if adslab_entry['slab_rid'] not in slab_rids_to_entry_dict.keys():
                continue
            bare_slab_entry = slab_rids_to_entry_dict[adslab_entry['slab_rid']]
            n = self.get_unit_primitive_area(adslab_entry, bare_slab_entry)
            
            ads = 'O' if 'couple' in adslab_entry['adsorbate'] else adslab_entry['adsorbate']
            Nads = self.Nads_in_slab(Composition(adslab_entry['vasprun']['unit_cell_formula']), 
                                     Composition(bare_slab_entry['vasprun']['unit_cell_formula']), 
                                     Composition(ads))
            Eadslab = adslab_entry['vasprun']['output']['final_energy']
            Eslab = bare_slab_entry['vasprun']['output']['final_energy']
            Eads = ads_entries[adslab_entry['ads_rid']]['energy']
            BE = (Eadslab - n * Eslab) / Nads - Eads
            BE = BE * Nads if not normalize else BE
            adslab_entry['adsorption_energy'] = BE
            adslab_entry['Nads'] = Nads
            new_entries.append(adslab_entry)
            
        return new_entries
            
    def Nads_in_slab(self, asdlab_comp, bare_slab_comp, ads_comp):
        """
        Returns the TOTAL number of adsorbates in the slab on BOTH sides. Does 
            so by comparing the composition of the adsorbed and clean slab and 
            determining the number of adsorbates needed to make the resulting difference
        """
        rxn = Reaction([bare_slab_comp, ads_comp], [asdlab_comp])
        return abs(rxn.coeffs[1])

    def surface_area(self, slab):
        """
        Calculates the surface area of the slab
        """
        m = slab.lattice.matrix
        return np.linalg.norm(np.cross(m[0], m[1]))
            
    def get_unit_primitive_area(self, adslab_entry, bare_slab_entry):
        """
        Returns the surface area of the adsorbed system per
        unit area of the primitive slab system.
        """
        
        A_ads = self.surface_area(Structure.from_dict(adslab_entry['vasprun']['output']['crystal']))
        A_clean = self.surface_area(Structure.from_dict(bare_slab_entry['vasprun']['output']['crystal']))
        n = A_ads / A_clean
        return n
    
    def get_all_adsorbate_entries_dict(self):
        # get the adsorbate reference entries
        ads_entries = {ads_entry['rid']: ads_entry
                       for ads_entry in self.surface_properties.find({"calc_type": 'adsorbate_in_a_box',
                                                                      'func': 'OC20'})}
        
        return ads_entries
                
    def get_slab_entries_OC22(self, criteria, E_adsorbate=None,
                              ads_rid=None, include_trajectories=False):
        
        ads_entries = self.adsorbates_dict
        entries = [entry for entry in self.surface_properties.find(criteria)]
        
        # store clean slabs in a dict for easy reference in building adsorbed slab entry
        clean_dict = {entry['rid']: SlabEntry(Structure.from_dict(entry['vasprun']['output']['crystal']), 
                                  entry['vasprun']['output']['final_energy'], 
                                  entry['miller_index'], entry_id=entry['entry_id'], 
                                  data={k: entry[k] for k in entry.keys() if k != 'vasprun'})
                      for entry in entries if entry['calc_type'] == 'bare_slab'}
        slab_entries = list(clean_dict.values())

        for entry in entries:
            if entry['calc_type'] == 'adsorbed_slab':
                s = Structure.from_dict(entry['vasprun']['output']['crystal'])
                e = entry['vasprun']['output']['final_energy']
                hkl = entry['miller_index']
                entry_id = entry['entry_id']
                data = {k: entry[k] for k in entry.keys() if k != 'vasprun'}
                if entry['slab_rid'] not in clean_dict.keys():
                    clean_entry = self.surface_properties.find_one({'rid': entry['slab_rid']})
                    if not clean_entry:
                        continue
                    clean_dict[entry['slab_rid']] = SlabEntry(Structure.from_dict(clean_entry['vasprun']['output']['crystal']), 
                                                               clean_entry['vasprun']['output']['final_energy'], 
                                                               clean_entry['miller_index'], entry_id=clean_entry['entry_id'],
                                                               data={k: clean_entry[k] 
                                                                     for k in clean_entry.keys() if k != 'vasprun'})

                adslab_entry = SlabEntry(s, e, entry['miller_index'], entry_id=entry['entry_id'], 
                                         adsorbates=[ads_entries[entry['ads_rid']]], 
                                         data={k: entry[k] for k in entry.keys() if k != 'vasprun'}, 
                                         clean_entry=clean_dict[entry['slab_rid']])
                
                if include_trajectories:
                    trajectories = []
                    for traj in entry['vasprun']['output']['ionic_steps']:
                        s = Structure.from_dict(traj['structure'])
                        setattr(s, 'adsorption_energy', (adslab_entry.gibbs_binding_energy(eads=True) \
                                - adslab_entry.energy + traj['e_wo_entrp'])/adslab_entry.Nads_in_slab)
                        trajectories.append(s)
                    adslab_entry.data['trajectory_entries'] = trajectories
                    
                slab_entries.append(adslab_entry)
            
        return slab_entries
            
    def get_surface_area(self, slab):
        """
        Calculates the surface area of the slab
        """
        m = slab.lattice.matrix
        return np.linalg.norm(np.cross(m[0], m[1]))

def json_compatible_dict(dictionary):
    for key, value in dictionary.items():
        if type(value) is dict:
            if type(key).__name__ != 'str':
                
                dictionary[str(key)] = dictionary[key]
                del dictionary[key]
            dictionary = recursive_items(value)
        else:
            if type(key).__name__ != 'str':
                dictionary[str(key)] = dictionary[key]
                del dictionary[key]
    return dictionary

def json_compatible_string(s):
    s = s.replace("'", '"')
    s = s.replace("None", '0')
    s = s.replace("array", '')
    s = s.replace("(", '')
    s = s.replace(")", '')
    s = s.replace("        ", ' ')
    s = s.replace("       ", ' ')
    s = s.replace("      ", ' ')
    s = s.replace("     ", ' ')
    s = s.replace("    ", ' ')
    s = s.replace("   ", ' ')
    s = s.replace("  ", ' ')
    s = s.replace(" ", ' ')
    s = s.replace("[ ", '[')
    s = s.replace(" ]", ']')
    
    return s