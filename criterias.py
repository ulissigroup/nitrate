import json, glob, copy, sys
from pymatgen.core.structure import *
from matplotlib import pylab as plt
sys.path.append('../')
from matplotlib import cm
import numpy as np
from matplotlib.ticker import FixedLocator, FormatStrFormatter
from matplotlib import patches
from constants import all_tms, all_pairs, costanalyzer


def filter_ehull(valid_mpids, max_ehull=0.08, return_entries=None, 
                 fdir='datasets/materials_dataset/*'):
    
    # Filter out mpids or auids if the energy above hull is above max_ehull

    tier_ehull_dict = {}
    entries_list = []
    for f in glob.glob(fdir):
        if any(el in f.split('/')[-1] for el in ['Hg', 'Cd', 'Tc', 'La', 'Ac']):
            continue
        entries = [d for d in json.load(open(f, 'rb')) if d['entry_id'] in valid_mpids]
        entries = [d for d in entries if d['data']['energy_above_hull'] < max_ehull]
        if return_entries:
            entries_list.extend(entries)
        tier_ehull_dict.update({d['entry_id']: Composition(d['composition']).reduced_formula for d in entries})

    tier_ehull_pairs = []
    for entry_id in tier_ehull_dict.keys():
        comp = Composition(tier_ehull_dict[entry_id]).as_dict()
        p = tuple([c[0] for c in sorted(list(comp.items()), reverse=True, key=lambda c: c[1])])
        if p not in tier_ehull_pairs:
            tier_ehull_pairs.append(p)
    if return_entries:
        return tier_ehull_dict, tier_ehull_pairs, entries_list
    else:
        return tier_ehull_dict, tier_ehull_pairs

def filter_pbx_stable(valid_mpids, gpbx=0.5, phrange=[6,8], vrange=0, 
                      include_pbx_stable=True, fdir='datasets/materials_dataset/*'):
    
    # Filter out mpids or auids if the Pourbaix decomposition energy is above energy above gpbx
    
    tier2_dict = {}
    for f in glob.glob(fdir):
        if any(el in f.split('/')[-1] for el in ['Hg', 'Cd', 'Tc', 'Ac', 'La']):
            continue
        entries = [d for d in json.load(open(f, 'rb')) if d['entry_id'] in valid_mpids]
        if include_pbx_stable:
            entries = [d for d in entries if any([g[0] >= vrange and phrange[0] <= g[1] <= phrange[1] and g[2] < gpbx
                                                  for g in d['data']['G_pbx']]) or d['data']['PBX_stable']]
        else:
            entries = [d for d in entries if any([g[0] >= vrange and phrange[0] <= g[1] <= phrange[1] and g[2] < gpbx
                                                  for g in d['data']['G_pbx']])]
        tier2_dict.update({d['entry_id']: Composition(d['composition']).reduced_formula for d in entries})

    tier2_pairs = []
    for entry_id in tier2_dict.keys():
        comp = Composition(tier2_dict[entry_id]).as_dict()
        p = tuple([c[0] for c in sorted(list(comp.items()), reverse=True, key=lambda c: c[1])])
        if p not in tier2_pairs:
            tier2_pairs.append(p)
    return tier2_dict, tier2_pairs

def filter_active(valid_mpids, check_existing=False, 
                  vrange=None, fdir='datasets/materials_dataset/*'):
    
    # What are all the materials that are within the region of 'high' activity
    
    tier3_dict = {}
    for f in glob.glob(fdir):
        if any(el in f.split('/')[-1] for el in ['Hg', 'Cd', 'Tc', 'Ac', 'La']):
            continue
        entries = [d for d in json.load(open(f, 'rb')) if d['entry_id'] in valid_mpids]
        
        if check_existing:
            existing_formula_dict = {}
            for d in json.load(open(f, 'rb')):
                comp = Composition(d['composition']).reduced_formula
                if 'ML_eads' in d['data'].keys():
                    if comp not in existing_formula_dict.keys():
                        existing_formula_dict[comp] = []
                    existing_formula_dict[comp].append(d)            

        new_entries = copy.deepcopy(entries)
        for entry in entries:
            if 'ML_eads' not in entry['data'].keys():
                if check_existing:
                    comp = Composition(entry['composition'])
                    if comp.reduced_formula in existing_formula_dict.keys():
                        new_entries.extend(existing_formula_dict[comp.reduced_formula])
                    else:
                        p = tuple([c[0] for c in sorted(list(comp.as_dict().items()), 
                                                        reverse=True, key=lambda c: c[1])])
                        for comp in existing_formula_dict.keys():
                            comp = Composition(comp)
                            p2 = tuple([c[0] for c in sorted(list(comp.as_dict().items()),
                                                             reverse=True, key=lambda c: c[1])])
                            if p == p2:
                                new_entries.extend(existing_formula_dict[comp.reduced_formula])
                else:
                    continue
                    
        entries = new_entries
        for entry in entries:
            if 'ML_eads' not in entry['data'].keys():
                continue
            active = False
            for v in entry['data']['activity_selectivity'].keys():
                if vrange and v not in vrange:
                    continue
                for act in entry['data']['activity_selectivity'][v].keys():
                    if any([d[0] < 0.31 for d in entry['data']['activity_selectivity'][v][act].values()]):
                        active = True
            if active:
                tier3_dict[entry['entry_id']] = Composition(entry['composition']).reduced_formula

    tier3_pairs = []
    for entry_id in tier3_dict.keys():
        comp = Composition(tier3_dict[entry_id]).as_dict()
        p = tuple([c[0] for c in sorted(list(comp.items()), reverse=True, key=lambda c: c[1])])
        if p not in tier3_pairs:
            tier3_pairs.append(p)
    return tier3_dict, tier3_pairs

def filter_selectivity(valid_mpids, return_entry_id_to_hatch_dict=False, 
                       exclude_NH3_only=False, fdir='datasets/materials_dataset/*'):
    
    # Filter materials that select for N2 (or NH3)
    
    bool_hatch_dict = {(True, False): '\\\\', (False, True): '////', (True, True): 'xx', (False, False): ''}
    tier4_dict, mpid_entry_dict = {}, {}
    entry_id_to_hatch_dict = {}
    for f in glob.glob(fdir):
        if any(el in f.split('/')[-1] for el in ['Hg', 'Cd', 'Tc', 'Ac', 'La']):
            continue
        entries = [d for d in json.load(open(f, 'rb')) if d['entry_id'] in valid_mpids]

        for entry in entries:
            comp = Composition(entry['composition']).as_dict()
            p = tuple([c[0] for c in sorted(list(comp.items()), reverse=True, key=lambda c: c[1])])
            if 'ML_eads' not in entry['data'].keys():
                continue
            selective = False
            for v in entry['data']['activity_selectivity'].keys():
                for act in entry['data']['activity_selectivity'][v].keys():
                    if exclude_NH3_only:
                        if any([d[1] for d in entry['data']['activity_selectivity'][v][act].values()]):
                            selective = True
                    else:
                        if any([d[1] or d[2] for d in entry['data']['activity_selectivity'][v][act].values()]):
                            selective = True

            if selective:
                entry_id = entry['entry_id']
                mpid_entry_dict[entry_id] = entry
                tier4_dict[entry_id] = Composition(entry['composition']).reduced_formula

    hatch_dict = {}
    tier4_pairs = []
    for entry_id in tier4_dict.keys():
        comp = Composition(tier4_dict[entry_id]).as_dict()
        p = tuple([c[0] for c in sorted(list(comp.items()), reverse=True, key=lambda c: c[1])])
        if p not in tier4_pairs:
            tier4_pairs.append(p)
                
        entry = mpid_entry_dict[entry_id]
        if p not in hatch_dict.keys():
            hatch_dict[p] = []
        N2_NH3 = []
        for v in entry['data']['activity_selectivity'].keys():
            for act in entry['data']['activity_selectivity'][v].keys():
                N2_NH3.extend([bool_hatch_dict[(d[1], d[2])] for d in 
                               entry['data']['activity_selectivity'][v][act].values()
                               if (d[1], d[2]) != (False, False)])
        entry_id_to_hatch_dict[entry_id] = N2_NH3
        
        n_hatch = {N2_NH3.count(h): h  for h in N2_NH3}
        hatch_dict[p].append(n_hatch[max(n_hatch.keys())])
    
    for p in hatch_dict.keys():
        if '\\\\' in hatch_dict[p]:
            hatch_dict[p] = '\\\\'
        elif 'xx' in hatch_dict[p]:
            hatch_dict[p] = 'xx'
        elif '////' in hatch_dict[p]:
            hatch_dict[p] = '////'
        else:
            hatch_dict[p] = ''

    if return_entry_id_to_hatch_dict:
        return tier4_dict, tier4_pairs, hatch_dict, entry_id_to_hatch_dict
    else:
        return tier4_dict, tier4_pairs, hatch_dict

def filter_cost(valid_mpids, fdir='datasets/materials_dataset/*'):
    
    # Filter out materials costing more than $500/kg

    tier5_dict = {}
    for f in glob.glob(fdir):

        if any(el in f.split('/')[-1] for el in ['Hg', 'Cd', 'Tc', 'Ac', 'La']):
            continue
        entries = [d for d in json.load(open(f, 'rb')) if d['entry_id'] in valid_mpids]

        for entry in entries:
            comp = Composition(entry['composition']).reduced_formula
            if costanalyzer.get_cost_per_kg(comp) < 500:
                tier5_dict[entry['entry_id']] = comp

    tier5_pairs = []
    for entry_id in tier5_dict.keys():
        comp = Composition(tier5_dict[entry_id]).as_dict()
        p = tuple([c[0] for c in sorted(list(comp.items()), reverse=True, key=lambda c: c[1])])
        if p not in tier5_pairs:
            tier5_pairs.append(p)
    return tier5_dict, tier5_pairs

def filter_size(valid_mpids, n=35, fdir='datasets/materials_dataset/*'):
    
    # filter out bulks with number of atoms greater than n
    
    tier_size_dict = {}
    for f in glob.glob(fdir):

        if any(el in f.split('/')[-1] for el in ['Hg', 'Cd', 'Tc', 'Ac', 'La']):
            continue
        entries = [d for d in json.load(open(f, 'rb')) if d['entry_id'] in valid_mpids]
        for entry in entries:            
            if len(entry['structure']['sites']) < n:
                tier_size_dict[entry['entry_id']] = Composition(entry['composition']).reduced_formula

    tier_size_pairs = []
    for entry_id in tier_size_dict.keys():
        comp = Composition(tier_size_dict[entry_id]).as_dict()
        p = tuple([c[0] for c in sorted(list(comp.items()), reverse=True, key=lambda c: c[1])])
        if p not in tier_size_pairs:
            tier_size_pairs.append(p)
    return tier_size_dict, tier_size_pairs

def plot_gridmap(tier1_pairs, tier2_pairs, tier3_pairs, tier4_pairs, 
                 tier5_pairs, hatch_dict, hatch_tier_order):
    
    # plots colored gridmap showing which pair of elements have passed all the given tiers.

    tick_range = range(0, len(all_tms))
    ellist = all_tms

    costs = [costanalyzer.get_cost_per_kg(el) for el in ellist]
    costs, ellist = zip(*sorted(zip(costs, ellist)))
    even_ticks, odd_ticks, even_els, odd_els = [], [], [], []
    for i in tick_range:
        if i %2 == 0:
            even_ticks.append(i)
            even_els.append(ellist[i])
        else:
            odd_ticks.append(i)
            odd_els.append(ellist[i])

    comp_grid, all_tof_dists, all_selectivity_hatches = [], [], []
    for el1 in ellist:
        tof_dists, selectivity_hatches, comp_vect = [], [], []
        for el2 in ellist:

            binary = tuple([str(el1), str(el2)])
            sorted_bin = tuple(sorted(binary))

            if binary[0] == binary[1]:
                tof_dists.append(float('nan'))
                selectivity_hatches.append('')
                comp_vect.append([])
                continue

            if binary in tier1_pairs:
                if binary in tier2_pairs:
                    if binary in tier3_pairs:
                        if binary in tier4_pairs:
                            if binary in tier5_pairs:
                                # Cost
                                tof_dists.append(19)
                                selectivity_hatches.append(hatch_dict[tuple(binary)])
                            else:
                                # ehull
                                tof_dists.append(38)
                                if hatch_tier_order <= 4:
                                    selectivity_hatches.append(hatch_dict[tuple(binary)])
                                else:
                                    selectivity_hatches.append('')
                        else:
                            # selectivity
                            tof_dists.append(57)
                            if hatch_tier_order <= 3:
                                selectivity_hatches.append(hatch_dict[tuple(binary)])
                            else:
                                selectivity_hatches.append('')
                    else:
                        # TOF
                        tof_dists.append(76)
                        if hatch_tier_order <= 2:
                            selectivity_hatches.append(hatch_dict[tuple(binary)])
                        else:
                            selectivity_hatches.append('')
                else:
                    # pourbaix
                    tof_dists.append(95)
                    if hatch_tier_order <= 1:
                        selectivity_hatches.append(hatch_dict[tuple(binary)])
                    else:
                        selectivity_hatches.append('')
            else:
                tof_dists.append(float('nan'))
                selectivity_hatches.append('')
        all_selectivity_hatches.append(selectivity_hatches)
        all_tof_dists.append(tof_dists)
        comp_grid.append(comp_vect)

    cmap = cm.jet
    cmap.set_bad('grey')
    fig, ax = plt.subplots()

    ax.xaxis.set_major_locator(FixedLocator(range(0, 26, 2)))
    ax.xaxis.set_minor_locator(FixedLocator(range(1, 26, 2)))
    ax.xaxis.set_minor_formatter(FormatStrFormatter("%d"))
    ax.tick_params(which='major', axis='x', labelsize=20)
    ax.tick_params(which='minor', pad=20, axis='x', labelsize=20)
    ax.set_xticklabels(even_els)
    ax.set_xticklabels(odd_els, minor=True)

    ax.set_yticklabels([])

    axy = ax.twinx()
    axy.yaxis.set_major_locator(FixedLocator(range(0, 26, 2)))
    axy.yaxis.set_minor_locator(FixedLocator(range(1, 26, 2)))
    axy.yaxis.set_minor_formatter(FormatStrFormatter("%d"))
    axy.tick_params(which='major', axis='y', labelsize=20)
    axy.tick_params(which='minor', pad=25, axis='y', labelsize=20)
    axy.set_yticklabels(even_els)
    axy.set_yticklabels(odd_els, minor=True)

    fig.set_size_inches(12, 12)
    im = plt.imshow(all_tof_dists, cmap='jet', interpolation='nearest')

    # selectivity
    ax = plt.gca()
    all_selectivity_hatches = np.array(all_selectivity_hatches).T
    comp_grid = np.array(comp_grid).T
    for x, row in enumerate(all_selectivity_hatches):
        for y, hatch in enumerate(row):
            ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1, hatch=hatch,
                                           fill=False, snap=False, color='r'))

    for x, row in enumerate(all_tof_dists):
        for y, hatch in enumerate(row):
            ax.add_patch(patches.Rectangle((x-0.5, y-0.5), 1, 1, fill=False, snap=False, color='k'))

    plt.clim(0,100)
    plt.plot([-1.5, 29], [-1.5, 29], 'k-', linewidth=3, )
    print(plt.xlim(), plt.ylim())
    plt.xlim(-0.5,25.5)
    plt.ylim(-1.5, 26.5)
    print(plt.xlim(), plt.ylim())
    
    return plt