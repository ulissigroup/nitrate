# from atomate.vasp.drones import VaspDrone 
from pymatgen.db import QueryEngine
import json, random, string, os, glob
import numpy as np

from pymatgen.core.structure import Structure, Molecule
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
                     'init_pmg_slab': d, 'calc_type': calc_type, 'func': functional}

    if calc_type == 'adsorbed_slab':
        metadata = {'entry_id': entry_id, 'database': database, 
                    'adsorbate': name_of_adsorbate,
                    'bulk_rid': bulk_rid, 'slab_rid': slab_rid, 
                    'ads_rid': ads_rid, 'rid': 'adslab-%s' %(rid), 
                    'miller_index': list(structure.miller_index), 
                    'bulk_formula': structure.oriented_unit_cell.composition.reduced_formula,
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
        for entry in adslab_entries:
            bare_slab_entry = self.surface_properties.find_one({'rid': entry['slab_rid']})
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
        
        return adslab_entries
    
    def get_slab_entries_OC22(self, criteria, E_adsorbate=None, ads_rid=None, include_trajectories=False):
        
        # get the adsorbate reference entries
        ads_entries = {ads_entry['adsorbate']: \
                       ComputedStructureEntry(Molecule.from_dict(json.loads(ads_entry['init_pmg_structure'].replace("'", "\""))),
                                              ads_entry['energy']) 
                       for ads_entry in self.surface_properties.find({"calc_type": 'adsorbate_in_a_box'})}
        
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
                    clean_entry[entry['slab_rid']] = SlabEntry(Structure.from_dict(clean_entry['vasprun']['output']['crystal']), 
                                                               clean_entry['vasprun']['output']['final_energy'], 
                                                               clean_entry['miller_index'], entry_id=clean_entry['entry_id'],
                                                               data={k: clean_entry[k] for k in clean_entry.keys() if k != 'vasprun'})

                adslab_entry = SlabEntry(s, e, entry['miller_index'], entry_id=entry['entry_id'], 
                                         adsorbates=[ads_entries[entry['adsorbate']]], 
                                         data={k: entry[k] for k in entry.keys() if k != 'vasprun'}, 
                                         clean_entry=clean_entry[entry['slab_rid']])
                
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