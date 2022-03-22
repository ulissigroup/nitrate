from pymatgen.core.structure import Composition, Structure, Lattice, StructureError
from pymatgen.io.ase import AseAtomsAdaptor
from ocpmodels.preprocessing import AtomsToGraphs 
from ocpmodels.datasets.trajectory_lmdb import TrajectoryLmdbDataset
from ocpmodels.datasets.single_point_lmdb import SinglePointLmdbDataset
import sys, os, glob
sys.path.append(os.getcwd())
from other_functions import str_to_hkl

import lmdb, pickle, torch, string, random

def rid_generator(r=random):
    return ''.join([r.choice(string.ascii_letters
                             + string.digits) for n in range(10)])


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
    return data_objects


def fid_writer(adslab, suffix=None):
    hkl = adslab.miller_index
    bulk_comp = {}
    ads = ''
    for site in adslab:
        if site.surface_properties == 'adsorbate':
            ads += site.species_string
            continue
        el = site.species_string
        if el not in bulk_comp.keys():
            bulk_comp[el] = 0
        bulk_comp[el] += 1
    if hasattr(adslab, 'oriented_unit_cell'):
        comp = adslab.oriented_unit_cell.composition.reduced_formula
    else:
        comp = Composition.from_dict(bulk_comp).reduced_formula
    n = '%s_%s%s%s_%s' % (comp, hkl[0], hkl[1], hkl[2], ads)
    if suffix:
        n += '_' + suffix
    return n


def test_lmdb_builder(adslabs_list, lmdb_path, s2ef=False):
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
        
        if s2ef:
            dat.fid = torch.LongTensor([idx])
        else:
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
        
    if s2ef:
        txn = db.begin(write=True)
        txn.put(f"length".encode("ascii"), pickle.dumps(len(data_objects), protocol=-1))
        txn.commit()
        db.sync()
    db.close()

from pymatgen.core.structure import Molecule
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import time
from pymatgen.core.surface import *

sys.path.append(os.getcwd())

O = Molecule(['O'], [[0, 0, 0]])
N = Molecule(['N'], [[0, 0, 0]])
ads_dict = {"*O": O, "*N": N}

def get_hkl_list(s, mmi, hkl_list=[]):
    
    if hkl_list == 'all':
        return get_symmetrically_distinct_miller_indices(s, mmi)
    
    elif hkl_list:
        return hkl_list
    else:
        all_hkls = get_symmetrically_distinct_miller_indices(s, mmi)
        hkls = []
#         if len(all_hkls) > 14:
#             mmi1 = [hkl for hkl in all_hkls if max(hkl) == 1]
#             mmibig = [hkl for hkl in all_hkls if max(hkl) > 1]
#             sam1 = len(mmi1) if len(mmi1) < 3 else 3
#             sam2 = len(mmibig) if len(mmibig) < 9 else 9
#             hkls.extend(random.sample(mmi1, sam1))
#             hkls.extend(random.sample(mmibig, sam2))
        if len(all_hkls) > 12:
#             mmi1 = [hkl for hkl in all_hkls if max(hkl) == 1]
            mmibig = [hkl for hkl in all_hkls if max(hkl) > 1]
#             sam1 = len(mmi1) if len(mmi1) < 3 else 3
            sam2 = len(mmibig) if len(mmibig) < 12 else 12
#             hkls.extend(random.sample(mmi1, sam1))
            hkls.extend(random.sample(mmibig, sam2))

        else:
            hkls = all_hkls
        return hkls
    
def get_repeat_from_min_lw(slab, min_lw):
    
    xlength = np.linalg.norm(slab.lattice.matrix[0])
    ylength = np.linalg.norm(slab.lattice.matrix[1])
    xrep = np.ceil(min_lw / xlength)
    yrep = np.ceil(min_lw / ylength)
    rtslab = slab.copy()
    rtslab.make_supercell([[1,1,0], [1,-1,0], [0,0,1]])
    rt_matrix = rtslab.lattice.matrix
    xlength_rt = np.linalg.norm(rt_matrix[0])
    ylength_rt = np.linalg.norm(rt_matrix[1])
    xrep_rt = np.ceil(min_lw / xlength_rt)
    yrep_rt = np.ceil(min_lw / ylength_rt)
    xrep = xrep*np.array([1,0,0]) if xrep*xlength < xrep_rt*xlength_rt else xrep_rt*np.array([1,1,0]) 
    yrep = yrep*np.array([0,1,0]) if yrep*ylength < yrep_rt*ylength_rt else yrep_rt*np.array([1,-1,0]) 
    zrep = [0,0,1]
    
    return [xrep, yrep, zrep]

def adslab_generator_fail_safe(adslabs, list_of_slabs=True):
    
    slab_correct, slab_errors = [], []
    for adslab in adslabs:
        csites = [site.frac_coords[2] for site in adslab if site.surface_properties != 'adsorbate']
        cadsite = [site.frac_coords[2] for site in adslab if site.surface_properties == 'adsorbate'][0]

        if cadsite > min(csites) and cadsite < max(csites):
            slab_errors.append('adsorbate_inside_slab')
            continue
        slab_correct.append(adslab)
        
    if list_of_slabs:
        return slab_correct
    return slab_errors
            

def slab_generator_fail_safe(slabs, min_c_size=10, percentage_slab=0.2, 
                             percentage_fully_occupied_c=0.8, list_of_slabs=True):
    """
    Slab input analyzer double checks the input slabs are generated correctly. Will sort slabs based on:
        1. Whether or not the c lattice parameter is inappropriately thin (less than min_c_size). Note 
            that min_c_size should be the min_vacuum_size+min_slab_size that was used in SlabGenerator
        2. Whether there are actually enough atoms to make a slab representative of the material. By 
            default, the slab should obviously have more atoms than the oriented unit cell used to 
            build it, otherwise something went wrong...
        3. Whether the slab is too thin which is defined by the pecentage of the lattice along c 
            being occupied by a slab layer (percentage_slab).
        4. Whether the slab's hkl plane has been appropriately oriented parallel to the xy plane of the 
            slab lattice. If the atoms are occupying a pecentage of the the lattice along c defined by 
            percentage_fully_occupied_c, then the slab was probably reoriented along the xz or yz plane 
            which really messes with a lot of the analysis and input generation.
    
    If a slab passes all these check, then it should be fine for the most part. Ideally once these issues 
    have been fixed in the generator, we won't need this anymore...
    
    params::
        slabs (pmg Slab): List of Slab structures to check
        min_c_size (float Å): minimum c lattice parameter you expect your slab model to have
        percentage_slab (float): if the slab layer of the lattice only occupies this percentage
            of the model along the c direction, its way too thin
        percentage_fully_occupied_c (float): if the slab layer of the lattice occupies this 
            percentage of the model along the c direction, it probably means the hkl plane was 
            reoriented parallel to xz or yz.
    
    Returns list of slabs (default) or list of errors
    """
    
    slab_correct, slab_errors = [], []
    for slab in slabs:
        # note please change min_c_size to whatever the actual minimum 
        # c-lattice parameter should be when anaylzing the actual set
        if slab.lattice.c < min_c_size: 
            slab_errors.append('bad_c_lattice')
            continue
        elif len(slab.oriented_unit_cell) > len(slab):
            slab_errors.append('not_enough_atoms')
            continue

        if any([site.frac_coords[2] > 0.9 for site in slab]) or \
        any([site.frac_coords[2] < 0.1 for site in slab]):
            slab = center_slab(slab)

        top_atoms, bottom_atoms  = [], []
        
        ccords = [site.coords[2] for site in slab]
        if (max(ccords) - min(ccords))/slab.lattice.c < percentage_slab:
            slab_errors.append('too_thin')
            continue

        ccoords = []
        for site in slab:
            ccoords.append(site.coords[2])
        if (max(ccoords)-min(ccoords))/slab.lattice.c > percentage_fully_occupied_c:
            slab_errors.append('xy_not_hkl')
            continue
        slab_correct.append(slab)
        
    if list_of_slabs:
        if not slab_correct:
            print(slab_errors)
        return slab_correct
    return slab_errors


def generate_multiple_lmdbs(entries_list, lmdb_dir, max_slabs=10000, ssize=15, vsize=15, 
                            mmi=2, symmetrize=False, prefix=None, hkl_list=[], find_args=None,
                            validate_proximity=False, max_slab_size=300, min_lw=7.9):

    count = 0
    sid = 0
    all_adslabs = []

    for j, entry in enumerate(entries_list):

        tstart = time.time()
        s = entry.structure
        sg = SpacegroupAnalyzer(s)
        s = sg.get_conventional_standard_structure()
        primitive = True #if len(s) > 50 else False
        if type(hkl_list).__name__ == 'dict':
            hkls = [hkl_list[entry.entry_id]]
        else:
            hkls = get_hkl_list(s, mmi, hkl_list=hkl_list)
        
        all_slabs = []
        for hkl in hkls:

            slabgen = SlabGenerator(s, hkl, ssize, vsize, lll_reduce=False, 
                                    center_slab=True, primitive=False)
            slabs = slabgen.get_slabs(symmetrize=symmetrize)
            adslabgen = AdsorbateSiteFinder(slabs[0])
            if any([site.c < 0.5 for site in adslabgen.slab if site.surface_properties == 'surface']):
                slabgen = SlabGenerator(s, hkl, ssize, vsize, lll_reduce=False, 
                                        center_slab=True, primitive=False, max_normal_search=1)
                slabs = slabgen.get_slabs(symmetrize=symmetrize)

            all_slabs.extend(slabs) 
            
            if not symmetrize:
                inverted = []
                for slab in slabs:
                    if not slab.is_symmetric():
                        l = slab.lattice
                        l = Lattice.from_parameters(l.a, l.b, l.c, l.alpha, l.beta, l.gamma)
                        spec = [sites.species_string for sites in slab]
                        coords = [sites.coords for sites in slab]
                        slab_copy = Structure(l, spec, coords, coords_are_cartesian=True, 
                                              site_properties=slab.site_properties)
                        invrt = slab_copy.copy()
                        invrt.make_supercell([1,1,-1])
                        invrt_slab = Slab(invrt.lattice, [site.species_string for site in invrt], 
                                          [site.frac_coords for site in invrt], slab.miller_index, 
                                          slab.oriented_unit_cell, 0, None, 
                                          site_properties=invrt.site_properties)
                        
                        inverted.append(invrt_slab)
                all_slabs.extend(inverted)
                
        all_slabs = slab_generator_fail_safe(all_slabs)
        print(len(s), sg.get_crystal_system(), entry.composition.reduced_formula, 
              mmi,len(all_slabs), max([len(s) for s in all_slabs]))
        
        for i, slab in enumerate(all_slabs):
            if len(slab) > max_slab_size:
                continue
                
            latt = get_repeat_from_min_lw(slab, min_lw)
            slab.make_supercell(latt)
            adslabgen = AdsorbateSiteFinder(slab)
            
            adslabs = adslabgen.generate_adsorption_structures(ads_dict['*O'], repeat=[1,1,1], 
                                                               find_args=find_args)
            adslabs = adslab_generator_fail_safe(adslabs)
            if not adslabs:
                print(entry.entry_id, slab.miller_index)
            for ii, adslab in enumerate(adslabs):
                
                if validate_proximity:
                    try:
                        inval_site = adslab[0]
                        species = [site.species for site in adslab]
                        fcoords = [site.frac_coords for site in adslab]
                        species.append('H')
                        fcoords.append(inval_site.frac_coords)
                        isval = Structure(adslab.lattice, species, fcoords, 
                                          validate_proximity=True)
                    except StructureError:
                        continue
                        
                if len(adslab) > max_slab_size:
                    continue
                
                setattr(adslab, 'suffix', '%s_slab_%s_%s' % (ii, i, entry.entry_id))
                setattr(adslab, 'sid', sid)
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
                    lmdb_name = '%s_no3rr_screen.lmdb' % (rid) \
                    if not prefix else '%s_%s_no3rr_screen.lmdb' % (prefix, rid)
                    print('max slab size', max([len(slab) for slab in all_adslabs]), lmdb_name)
                    test_lmdb_builder(all_adslabs, os.path.join(lmdb_dir, lmdb_name))
                    os.system('gzip %s' %(os.path.join(lmdb_dir, lmdb_name)))
                    all_adslabs = []
                    count = 0

        tend = time.time()
        print(len(all_slabs), len(all_adslabs), tend - tstart, j)

    rid = ''.join([random.choice(string.ascii_letters
                                 + string.digits) for n in range(10)])
    lmdb_name = '%s_no3rr_screen.lmdb' % (rid) if not prefix else '%s_%s_no3rr_screen.lmdb' % (prefix, rid)
    if all_adslabs:
        print('max slab size', max([len(slab) for slab in all_adslabs]), lmdb_name)
        test_lmdb_builder(all_adslabs, os.path.join(lmdb_dir, lmdb_name))
        os.system('gzip %s' %(os.path.join(lmdb_dir, lmdb_name)))


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
        try:
            formula, hkl, ads, nads, r, nslabs, entry_id = dat.name.split('_')
            
        except ValueError:
#             AgPd_dun_O_POSCAR_40_O
            formula, hkl, ads, entry_id = dat.name.split('_')
        n = '%s_%s' % (formula, entry_id)
        if n not in dat_dict.keys():
            dat_dict[n] = {}
        if hkl not in dat_dict[n].keys():
            dat_dict[n][hkl] = {'N': [], 'O': []}

        sid = str(dat.sid)
        dat_dict[n][hkl][ads].append({'eads': checkpoints[sid], 'idx': i, 
                                      'sid': sid, 'lmdb': lmdb_dir})

    return dat_dict

def dat2struct(dat):
    return Structure(Lattice(dat.cell), dat.atomic_numbers, 
                     dat.pos, coords_are_cartesian=True, validate_proximity=True)

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
