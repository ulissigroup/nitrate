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

from pymatgen import Molecule
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import time
from pymatgen.core.surface import *

sys.path.append(os.getcwd())

O = Molecule(['O'], [[0, 0, 0]])
N = Molecule(['N'], [[0, 0, 0]])
ads_dict = {"*O": O, "*N": N}

def generate_multiple_lmdbs(entries_list, lmdb_dir, 
                            in_unit_planes=True, max_slabs=10000, mmi=2, 
                            max_i_only=False, prefix=None, hkl_list=[], find_args=None):

    count = 0
    sid = 0
    all_adslabs = []

    for j, entry in enumerate(entries_list):

        tstart = time.time()
        s = entry.structure
        sg = SpacegroupAnalyzer(s)
        s = sg.get_conventional_standard_structure()
        
        if hkl_list:
            all_slabs = []
            for hkl in hkl_list:
                slabgen = SlabGenerator(s, hkl, 15, 15, lll_reduce=True, center_slab=True)
                all_slabs.extend(slabgen.get_slabs(symmetrize=True))
        else:
            all_slabs = []
            for hkl in get_symmetrically_distinct_miller_indices(s, mmi):
                if max(hkl) != mmi and max_i_only:
                    continue
                slabgen = SlabGenerator(s, hkl, 15, 15, lll_reduce=True, center_slab=True)
                all_slabs.extend(slabgen.get_slabs(symmetrize=True))

        print(len(s), sg.get_crystal_system(), entry.composition.reduced_formula, mmi,
              len(all_slabs), max([len(s) for s in all_slabs]))

        for i, slab in enumerate(all_slabs):
            if len(slab) > 300:
                continue
            adslabgen = AdsorbateSiteFinder(slab)
            
            adslabs = adslabgen.generate_adsorption_structures(ads_dict['*O'], min_lw=8, 
                                                               find_args=find_args)
                                                            
            for ii, adslab in enumerate(adslabs):
                setattr(adslab, 'suffix', '%s_slab_%s_%s' % (ii, i, entry.entry_id))
                setattr(adslab, 'sid', sid)
                if len(adslab) < 300:
                    all_adslabs.append(adslab)
                    count += 1
                    sid += 1

                adslab2 = adslab.copy()
                adslab2.replace(-1, 'N')
                setattr(adslab2, 'suffix', '%s_slab_%s_%s' % (ii, i, entry.entry_id))
                setattr(adslab2, 'sid', sid)
                adslab2.add_site_property('surface_properties',
                                          adslab.site_properties['surface_properties'])
                if len(adslab) < 300:
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
    
    
def generate_multiple_lmdbs_random(entries_list, lmdb_dir, prefix=None, 
                                   in_unit_planes=True, max_slabs=5000):

    count = 0
    sid = 0
    all_adslabs = []

    for j, entry in enumerate(entries_list):

        tstart = time.time()
        s = entry.structure
        sg = SpacegroupAnalyzer(s)
        s = sg.get_conventional_standard_structure()
        if len(s) > 100:
            continue
        hkls = get_symmetrically_distinct_miller_indices(s, 2)
        mmi1 = [hkl for hkl in hkls if max(hkl) == 1]
        mmi2 = [hkl for hkl in hkls if max(hkl) == 2]
        
        all_slabs = []
        n = len(mmi1) if len(mmi1) < 5 else 5
        for hkl in random.sample(mmi1, n):
            slabgen = SlabGenerator(s, hkl, 15, 15, lll_reduce=True, center_slab=True, primitive=False)
            all_slabs.append(slabgen.get_slab())
        n = len(mmi2) if len(mmi2) < 5 else 5
        for hkl in random.sample(mmi2, n):
            slabgen = SlabGenerator(s, hkl, 15, 15, lll_reduce=True, center_slab=True, primitive=False)
            all_slabs.append(slabgen.get_slab())
            
        print(len(s), sg.get_crystal_system(), entry.composition.reduced_formula,
              len(all_slabs), max([len(s) for s in all_slabs]))

        for i, slab in enumerate(all_slabs):
            if len(slab) > 250:
                continue
            adslabgen = AdsorbateSiteFinder(slab)
            adslabs = adslabgen.generate_adsorption_structures(ads_dict['*O'], min_lw=8, 
                                                               find_args={'symm_reduce': 1e-1, 
                                                                          'near_reduce': 1e-1,
                                                                          'positions': ['hollow']})
                                                            
            for ii, adslab in enumerate(adslabs):
                setattr(adslab, 'suffix', '%s_slab_%s_%s' % (ii, i, entry.entry_id))
                setattr(adslab, 'sid', sid)
                if len(adslab) < 250:
                    all_adslabs.append(adslab)
                    count += 1
                    sid += 1

                    adslab2 = adslab.copy()
                    adslab2.replace(-1, 'N')
                    setattr(adslab2, 'suffix', '%s_slab_%s_%s' % (ii, i, entry.entry_id))
                    setattr(adslab2, 'sid', sid)
                    adslab2.add_site_property('surface_properties',
                                              adslab.site_properties['surface_properties'])
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
                    os.system('gzip %s' %(os.path.join(lmdb_dir, lmdb_name)))
                    count = 0

        tend = time.time()
        print(len(all_slabs), len(all_adslabs), tend - tstart, j)
    
    if len(all_adslabs) != 0:
        rid = ''.join([random.choice(string.ascii_letters
                                     + string.digits) for n in range(10)])
        lmdb_name = '%s_no3rr_screen.lmdb' % (rid) if not prefix else '%s_%s_no3rr_screen.lmdb' % (prefix, rid)
        print('max slab size', max([len(slab) for slab in all_adslabs]), lmdb_name)
        test_lmdb_builder(all_adslabs, os.path.join(lmdb_dir, lmdb_name))
#         os.system('gzip %s' %(os.path.join(lmdb_dir, lmdb_name)))



def get_eads_dicts(lmdb_dir, checkpoints_dir, name_tag=None):

    # load predicted adsorption energies
    chpt_to_lmdb_dict = {}
    all_chpts = {chpt.split('/')[-1].split('-')[-1]: np.load(os.path.join(chpt, 'is2re_predictions.npz')) \
                 for chpt in glob.glob(os.path.join(checkpoints_dir, '*'))}
    for lmdb in glob.glob(os.path.join(lmdb_dir, '*')):
        if 'lock' in lmdb:
            continue
        name = lmdb.split('/')[-1].replace('_no3rr_screen.lmdb', '')
        traj = SinglePointLmdbDataset({"src": lmdb})
        
        chpt_to_lmdb_dict[name] = {'traj': traj, 'chpt': all_chpts[name]}

    dat_dict = {}
    for count in chpt_to_lmdb_dict.keys():
        checkpoints = chpt_to_lmdb_dict[count]['chpt']
        single_traj = chpt_to_lmdb_dict[count]['traj']

        idx_list, eads_list = checkpoints.get('ids'), checkpoints.get('energy')

        for i, dat in enumerate(single_traj):
            if name_tag and name_tag not in dat.name:
                continue
            print(dat.name)
            namelist = dat.name.split('_')
            formula = namelist[0]
            hkl = namelist[1]
            ads = namelist[2]
            nads = namelist[3]
            r = namelist[4]
            nslabs = namelist[5]
            entry_id = ''
            for ii, k in enumerate(namelist):
                if ii > 5:
                    if ii < len(namelist)-2:
                        entry_id += k+'-'
                    else:
                        entry_id += k
            hkl = str(str_to_hkl(hkl))
            n = '%s_%s' % (formula, entry_id)
            if n not in dat_dict.keys():
                dat_dict[n] = {}
            if hkl not in dat_dict[n].keys():
                dat_dict[n][hkl] = {'N': [], 'O': []}

            eads = eads_list[list(idx_list).index(str(i))]
            dat_dict[n][hkl][ads].append({'eads': eads, 'idx': i, 'lmdb': count})

    return dat_dict

def get_eads_dicts_single(lmdb_dir, checkpoint_dir, name_tag=None):
    
    checkpoints = np.load(os.path.join(checkpoint_dir, 'is2re_predictions.npz'))
    single_traj = SinglePointLmdbDataset({"src": lmdb_dir})
    checkpoints = dict(zip(checkpoints.get('ids'), checkpoints.get('energy')))
    
    dat_dict = {}
    for i, dat in enumerate(single_traj):
        formula, hkl, ads, nads, r, nslabs, entry_id = dat.name.split('_')
        
        n = '%s_%s' % (formula, entry_id)
        if n not in dat_dict.keys():
            dat_dict[n] = {}
        if hkl not in dat_dict[n].keys():
            dat_dict[n][hkl] = {'N': [], 'O': []}

        sid = str(dat.sid)
        dat_dict[n][hkl][ads].append({'eads': checkpoints[sid], 'sid': sid})

    return dat_dict

def dat2struct(dat):
    return Structure(Lattice(dat.cell), dat.atomic_numbers, dat.pos, coords_are_cartesian=True)
def dat2dict(dat):
    s = dat2struct(dat).as_dict()
    name = dat.name
    sid = dat.sid
    idx = dat.idx
    return {'structure': s, 'name': name, 'sid': sid, 'idx': idx}
def traj2dict(traj):
    traj2list = []
    for dat in traj:
        traj2list.append(dat2dict(dat))
    return traj2list