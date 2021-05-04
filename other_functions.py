from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram, PourbaixPlotter
from pymatgen import MPRester


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
: