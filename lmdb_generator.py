from pymatgen import Composition, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.datasets.trajectory_lmdb import TrajectoryLmdbDataset
import sys, os, glob
sys.path.append(os.getcwd())
from other_functions import str_to_hkl


import lmdb
import pickle
import torch

def read_trajectory_extract_features(a2g, adslabs_list):
    tags_dict = {"subsurface": 2, 'surface': 1, 'adsorbate': 0}
    adaptor = AseAtomsAdaptor()
    list_of_atoms = [adaptor.get_atoms(adslab) for adslab in adslabs_list]

    # Converts a list of atoms object representing
    # slabs into Data objects with features inside
    tagged_data_objects = []
    data_objects = a2g.convert_all(list_of_atoms, disable_tqdm=True)
    for i, dat in enumerate(data_objects):
        slab = adslabs_list[i]
        tags = [tags_dict[site.surface_properties] for site in slab]
        dat.tags = torch.LongTensor(tags)
        tagged_data_objects.append(dat)
    return tagged_data_objects


def fid_writer(adslab, suffix=None):
    hkl = adslab.miller_index
    bulk_comp = {}
    for site in adslab:
        if site.surface_properties == 'adsorbate':
            ads = site.species_string
            continue
        el = site.species_string
        if el not in bulk_comp.keys():
            bulk_comp[el] = 0
        bulk_comp[el] += 1
    comp = Composition.from_dict(bulk_comp).reduced_formula
    n = '%s_%s%s%s_%s' % (comp, hkl[0], hkl[1], hkl[2], ads)
    if suffix:
        n += '_' + suffix
    return n


def test_lmdb_builder(adslabs_list, lmdb_path):
    # Write to LMDB
    a2g = AtomsToGraphs(max_neigh=50, radius=6, r_energy=False,
                        r_distances=False, r_fixed=True)
    db = lmdb.open(lmdb_path, map_size=1099511627776 * 2,
                   subdir=False, meminit=False, map_async=True)

    data_objects = read_trajectory_extract_features(a2g, adslabs_list)

    idx_to_sys_dict = {}
    for idx, dat in enumerate(data_objects):

        # add additional attributes
        dat.y_init = None
        del dat.y
        dat.y_relaxed = None
        dat.pos_relaxed = dat.pos
        try:
            suffix = adslabs_list[idx].suffix
        except AttributeError:
            suffix = None
        name = fid_writer(adslabs_list[idx], suffix=suffix)
        dat.sid = adslabs_list[idx].sid
        dat.name = name
        idx_to_sys_dict[idx] = [name, adslabs_list[idx].to_json()]
        # no neighbor edge case check
        if dat.edge_index.shape[1] == 0:
            print("no neighbors")
            continue

        # Write to LMDB
        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(dat, protocol=-1))
        txn.commit()
        db.sync()

    db.close()
    return idx_to_sys_dict


from pymatgen import Molecule
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import time

sys.path.append(os.getcwd())
sys.path.append('/home/jovyan/repos/mo-wulff-workflow/')
from surface import *

O = Molecule(['O'], [[0, 0, 0]])
N = Molecule(['N'], [[0, 0, 0]])
ads_dict = {"*O": O, "*N": N}

def generate_multiple_lmdbs(entries_list, lmdb_dir):

    count = 0
    sid = 0
    all_adslabs = []

    for j, entry in enumerate(entries_list):

        tstart = time.time()
        s = entry.structure
        sg = SpacegroupAnalyzer(s)
        s = sg.get_conventional_standard_structure()
        if sg.get_crystal_system() in ['hexagonal', 'cubic']:
            mmi = 2
        else:
            mmi = 1

        mmi = 1 if len(s) > 10 else mmi
        if len(s) > 24:
            all_slabs = []
            for hkl in get_symmetrically_distinct_miller_indices(s, mmi):
                slabgen = SlabGenerator(s, hkl, 4, 8, lll_reduce=True, center_slab=True, in_unit_planes=True)
                all_slabs.append(slabgen.get_slab())
        else:
            all_slabs = generate_all_slabs(s, mmi, 4, 8, lll_reduce=True, center_slab=True, in_unit_planes=True)

        print(len(s), sg.get_crystal_system(), entry.composition.reduced_formula, mmi, len(all_slabs))

        for i, slab in enumerate(all_slabs):
            if len(slab) > 300:
                continue
            adslabgen = AdsorbateSiteFinder(slab)
            adslabs = adslabgen.generate_adsorption_structures(ads_dict['*O'], min_lw=8)
            for ii, adslab in enumerate(adslabs):

                setattr(adslab, 'suffix', '%s_slab_%s_%s' % (ii, i, entry.entry_id))
                setattr(adslab, 'sid', sid)
                count += 1
                sid += 1
                if len(adslab) < 800:
                    all_adslabs.append(adslab)

                adslab2 = adslab.copy()
                adslab2.replace(-1, 'N')
                setattr(adslab2, 'suffix', '%s_slab_%s_%s' % (ii, i, entry.entry_id))
                setattr(adslab2, 'sid', sid)
                count += 1
                sid += 1
                adslab2.add_site_property('surface_properties',
                                          adslab.site_properties['surface_properties'])
                if len(adslab) < 800:
                    all_adslabs.append(adslab2)

        tend = time.time()
        print(len(all_slabs), len(all_adslabs), tend - tstart, j)

        if count > 4000:
            print('max slab size', max([len(slab) for slab in all_adslabs]))
            test_lmdb_builder(all_adslabs, os.path.join(lmdb_dir, '%s_no3rr_screen.lmdb' % (count)))
            all_adslabs = []
            count = 0


def get_eads_dicts(lmdb_dir, checkpoints_dir, get_struct_dict=False):

    # load predicted adsorption energies
    all_eads_name = {}
    count = 0
    for cpt in glob.glob(os.path.join(checkpoints_dir, '*')):
        checkpoints = np.load(os.path.join(cpt, 'is2re_predictions.npz'))
        count += len(checkpoints.get('ids'))
        for i, idx in enumerate(checkpoints.get('ids')):
            all_eads_name[int(idx)] = checkpoints.get('energy')[i]

    traj_lmdb = TrajectoryLmdbDataset({"src": lmdb_dir})

    ads_dict, struct_dict = {}, {}
    for dat in traj_lmdb:

        sid = dat.sid
        formula, hkl, ads, nads, r, nslab, entry_id = dat.name.split('_')
        hkl = str(str_to_hkl(hkl))
        n = '%s_%s' % (formula, entry_id)
        if n not in ads_dict.keys():
            ads_dict[n] = {}
        if hkl not in ads_dict[n].keys():
            ads_dict[n][hkl] = {'N': [], 'O': []}
        ads_dict[n][hkl][ads].append(all_eads_name[sid])

        if get_struct_dict:
            if n not in struct_dict.keys():
                struct_dict[n] = {}
            if hkl not in struct_dict[n].keys():
                struct_dict[n][hkl] = {'N': [], 'O': []}

            slab = Structure(Lattice(dat.cell), dat.atomic_numbers,
                             dat.pos, coords_are_cartesian=True)
            struct_dict[n][hkl][ads].append(slab.as_dict())

    if get_struct_dict:
        return ads_dict, struct_dict
    else:
        return ads_dict
