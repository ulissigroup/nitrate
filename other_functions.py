from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram, PourbaixPlotter
from pymatgen import MPRester
from pymatgen import Composition, Structure, Lattice


def pourbaix_stable(all_entries, mprester=None):
    mprester = mprester if mprester else MPRester()
    entry = all_entries[0]
    comp = entry.composition
    chemsys = list(comp.as_dict().keys())
    entries = mprester.get_pourbaix_entries(chemsys)
    pbd = PourbaixDiagram(entries)
    mpids = []
    for entry in pbd.stable_entries:
        if type(entry.entry_id).__name__ == 'str':
            mpids.append(entry.entry_id)
        else:
            mpids.extend(entry.entry_id)
    stable_entries = []
    for entry in all_entries:
        mpid = entry.data['material_id']
        if mpid in mpids:
            stable_entries.append(entry)
    return stable_entries

def str_to_hkl(s):
    hkl = []
    norm = 1
    for i in s:
        if i in ['(', ')', ',', ' ']:
            continue
        if i == '-':
            norm = -1
            continue
        hkl.append(norm * int(i))
        norm = 1
    return tuple(hkl)

def vegards_law(composition, bulk1, bulk2):
    d = composition.as_dict()
    sum_a, count = 0, 0
    latt_dict = {bulk[0].species_string: bulk.lattice.a for bulk in [bulk1, bulk2]}
    for el in d.keys():
        n = d[el]
        sum_a += n*latt_dict[el]
        count += n
    return (sum_a/count)*composition.get_reduced_composition_and_factor()[1]

def bcc_vegard_bulk(B1, A3):

    species, fcoords = [], []
    comp = ''
    species.extend([B1[0].species_string]*4)
    fcoords.extend([[0.7500, 0.2500, 0.2500], [0.2500, 0.7500, 0.2500],
                    [0.2500, 0.2500, 0.7500], [0.7500, 0.7500, 0.7500]])
    comp+='%s%s' %(B1[0].species_string, 4)
    species.extend([A3[0].species_string]*12)
    fcoords.extend([[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0, 0],
                    [0, 0.5, 0],[0.5, 0.5, 0],[0.75, 0.75, 0.25],
                    [0, 0, 0.5],[0.5, 0, 0.5],[0.75, 0.25, 0.75],
                    [0, 0.5, 0.5],[0.25, 0.75, 0.75],[0.5, 0.5, 0.5]])
    comp+='%s%s' %(A3[0].species_string, 12)

    comp = Composition(comp)
    a = vegards_law(comp, B1, A3)
    l = Lattice.cubic(a)

    return Structure(l, species, fcoords)

def fcc_vegard_bulk(B1, A3):

    species, fcoords = [], []
    comp = ''
    species.extend([B1[0].species_string]*1)
    fcoords.extend([[0.0, 0.0, 0.0]])
    comp+='%s%s' %(B1[0].species_string, 1)
    species.extend([A3[0].species_string]*3)
    fcoords.extend([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    comp+='%s%s' %(A3[0].species_string, 3)

    comp = Composition(comp)
    a = vegards_law(comp, B1, A3)
    l = Lattice.cubic(a)

    return Structure(l, species, fcoords)