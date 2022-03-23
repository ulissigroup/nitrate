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
path.append('costdb_elements_2021.csv')
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