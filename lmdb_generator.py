from pymatgen import Composition
from ocpmodels.preprocessing import AtomsToGraphs
from pymatgen.io.ase import AseAtomsAdaptor

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
        dat.sid = idx
        try:
            suffix = adslabs_list[idx].suffix
        except AttributeError:
            suffix = None
        idx_to_sys_dict[idx] = [fid_writer(adslabs_list[idx], suffix=suffix), adslabs_list[idx].to_json()]
        # no neighbor edge case check
        if dat.edge_index.shape[1] == 0:
            print("no neighbors")
            continue

        # Write to LMDB
        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(dat, protocol=-1))
        txn.commit()
        db.sync()
        idx += 1

    db.close()
    return idx_to_sys_dict