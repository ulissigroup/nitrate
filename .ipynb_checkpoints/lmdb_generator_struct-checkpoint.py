from pymatgen import Composition, Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from ocpmodels.preprocessing import AtomsToGraphs 
from ocpmodels.preprocessing.struct_to_graphs import StructToGraphs
from ocpmodels.datasets.trajectory_lmdb import TrajectoryLmdbDataset
from ocpmodels.datasets.single_point_lmdb import SinglePointLmdbDataset
import sys, os, glob
sys.path.append(os.getcwd())
from other_functions import str_to_hkl


import lmdb, pickle, torch, string, random

def read_trajectory_extract_features(a2g, adslabs_list):
    tags_dict = {"subsurface": 2, 'surface': 1, 'adsorbate': 0}

    # Converts a list of atoms object representing
    # slabs into Data objects with features inside
    tagged_data_objects = []
    data_objects = a2g.convert_all(adslabs_list, disable_tqdm=True)
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
    a2g = StructToGraphs(max_neigh=50, radius=6, r_energy=False,
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
        dat.idx = idx
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

def generate_multiple_lmdbs(entries_list, lmdb_dir, max_slabs=10000, set_mmi=None, prefix=None):

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
        mmi = set_mmi if set_mmi else mmi
        if len(s) > 24:
            all_slabs = []
            for hkl in get_symmetrically_distinct_miller_indices(s, mmi):
                slabgen = SlabGenerator(s, hkl, 4, 8, lll_reduce=True, center_slab=True, in_unit_planes=True)
                all_slabs.append(slabgen.get_slab())
        else:
            all_slabs = generate_all_slabs(s, mmi, 4, 8, lll_reduce=True, center_slab=True, in_unit_planes=True)

        print(len(s), sg.get_crystal_system(), entry.composition.reduced_formula, mmi,
              len(all_slabs), max([len(s) for s in all_slabs]))

        for i, slab in enumerate(all_slabs):
            if len(slab) > 300:
                continue
            adslabgen = AdsorbateSiteFinder(slab)
            adslabs = adslabgen.generate_adsorption_structures(ads_dict['*O'], min_lw=8,
                                                               find_args={'symm_reduce': 1e-1,
                                                                          'near_reduce': 1e-1})
                                                            
            for ii, adslab in enumerate(adslabs):
                setattr(adslab, 'suffix', '%s_slab_%s_%s' % (ii, i, entry.entry_id))
                setattr(adslab, 'sid', sid)
                if len(adslab) < 800:
                    all_adslabs.append(adslab)
                    count += 1
                    sid += 1

                adslab2 = adslab.copy()
                adslab2.replace(-1, 'N')
                setattr(adslab2, 'suffix', '%s_slab_%s_%s' % (ii, i, entry.entry_id))
                setattr(adslab2, 'sid', sid)
                adslab2.add_site_property('surface_properties',
                                          adslab.site_properties['surface_properties'])
                if len(adslab) < 800:
                    all_adslabs.append(adslab2)
                    count += 1
                    sid += 1

                if count > max_slabs:
                    rid = ''.join([random.choice(string.ascii_letters
                                                 + string.digits) for n in range(10)])
                    lmdb_name = '%s_no3rr_screen.lmdb' % (rid) if not prefix else '%s_%s_no3rr_screen.lmdb' % (prefix, rid)
                    print('max slab size', max([len(slab) for slab in all_adslabs]), lmdb_name)
                    test_lmdb_builder(all_adslabs, os.path.join(lmdb_dir, lmdb_name))
                    all_adslabs = []
                    count = 0

        tend = time.time()
        print(len(all_slabs), len(all_adslabs), tend - tstart, j)

    rid = ''.join([random.choice(string.ascii_letters
                                 + string.digits) for n in range(10)])
    lmdb_name = '%s_no3rr_screen.lmdb' % (rid) if not prefix else '%s_%s_no3rr_screen.lmdb' % (prefix, rid)
    print('max slab size', max([len(slab) for slab in all_adslabs]), lmdb_name)
    test_lmdb_builder(all_adslabs, os.path.join(lmdb_dir, lmdb_name))


def get_eads_dicts(lmdb_dir, checkpoints_dir, name_tag=None):

    # load predicted adsorption energies
    chpt_to_lmdb_dict = {}
    all_chpts = {chpt.split('/')[-1].split('-')[-1]: np.load(os.path.join(chpt, 'is2re_predictions.npz')) \
                 for chpt in glob.glob(os.path.join(checkpoints_dir, '*'))}
    for lmdb in glob.glob(os.path.join(lmdb_dir, '*')):
        if 'lock' in lmdb:
            continue
        name = lmdb.split('/')[-1].split('_')[0]
        traj = SinglePointLmdbDataset({"src": lmdb})
        chpt_to_lmdb_dict[name] = {'traj': traj, 'chpt': all_chpts[name]}

    dat_dict = {}
    for count in chpt_to_lmdb_dict.keys():
        checkpoints = chpt_to_lmdb_dict[count]['chpt']
        single_traj = chpt_to_lmdb_dict[count]['traj']

        idx_list, eads_list = zip(*sorted(zip(checkpoints.get('ids'), checkpoints.get('energy'))))

        for i, eads in enumerate(eads_list):
            dat = single_traj[i]
            if name_tag and name_tag not in dat.name:
                continue
            formula, hkl, ads, nads, r, nslab, entry_id = dat.name.split('_')
            hkl = str(str_to_hkl(hkl))
            n = '%s_%s' % (formula, entry_id)
            if n not in dat_dict.keys():
                dat_dict[n] = {}
            if hkl not in dat_dict[n].keys():
                dat_dict[n][hkl] = {'N': [], 'O': []}
            dat_dict[n][hkl][ads].append([dat, eads])
            dat_dict[n][hkl][ads].append({'eads': eads, 'idx': i, 'lmdb': count})

    return dat_dict
