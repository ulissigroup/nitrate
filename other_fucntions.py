from pymatgen.analysis.pourbaix_diagram import PourbaixDiagram, PourbaixPlotter


def pourbaix_stable(entry):
    mpid = entry.data['material_id']
    comp = entry.composition
    chemsys = list(comp.as_dict().keys())
    entries = mprester.get_pourbaix_entries(chemsys)
    pbd = PourbaixDiagram(entries)
    mpids = []
    for entry in pbd.unstable_entries:
        if type(entry.entry_id).__name__ == 'str':
            mpids.append(entry.entry_id)
        else:
            mpids.extend(entry.entry_id)
    if mpid in mpids:
        return False
    else:
        return True