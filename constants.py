from pymatgen.core.periodic_table import Element
import itertools, os

all_tms = [str(el) for el in Element 
           if el.is_transition_metal and str(el) \
           not in ['Ac', 'La', 'Hg', 'Tc', 'Cd'] and el.number < 81]
all_pairs = [pair for pair in itertools.permutations(all_tms, 2)]

from pymatgen.analysis.cost import CostAnalyzer, CostDBCSV
path = []
for p in os.getcwd().split('/'):
    path.append(p)
    if p == 'nitrate':
        break
path.append('datasets/costdb_elements_2021.csv')
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    costdb = CostDBCSV(os.path.join('/'.join(path)))
    costanalyzer = CostAnalyzer(costdb)

Eads_ref_oc20 = {'*O': -7.646083600000001, '*N': -.17228045E+02/2}
Eads_ref_dft = {'*O': -1.53343974, '*N': -3.11173655, '*H': -1.11587797}
Eads_ref_rpbe = {'*O': -2.0552055, '*N': -3.6207807, '*H': -1.11587797}
Ncorr_db = [-0.03508250142669311, 1.0122001719490594]
Ocorr_db = [0.43857541907825404, 1.0530681561315451]

# Get the lines of best activity under 0, 0.1 and 0.2 V vs RHE in accordance to Liu et al.
activity_params = {0.1: {'good': {'pt1': [-5.45, -6.45], 'pt2': [-4.56, -4.25]},
                         'excellent': {'pt1': [-5.3, -5.75], 'pt2': [-4.85, -4.65]}},
                   0: {'good': {'pt1': [-5.925, -6.497], 'pt2': [-4.756, -3.865]},
                         'excellent': {'pt1': [-5.8998, -6.420], 'pt2': [-5.126, -4.666]}},
                   0.2: {'good': {'pt1': [-5.101, -6.387], 'pt2': [-4.391, -4.778]},
                         'excellent': {'pt1': [-5, -5.95], 'pt2': [-4.691, -5.2]}}}
for V in activity_params.keys():
    for act in activity_params[V].keys():
        pt1 = activity_params[V][act]['pt1'] 
        pt2 = activity_params[V][act]['pt2'] 
        activity_params[V][act]['a'] = (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
        activity_params[V][act]['b'] = -1
        activity_params[V][act]['c'] = pt2[1] - pt2[0]*activity_params[V][act]['a']
