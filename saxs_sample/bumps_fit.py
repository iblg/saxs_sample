from bumps.names import *
from sasmodels.core import load_model
from sasmodels.bumps_model import Model, Experiment
from sasmodels.data import load_data, set_beam_stop, set_top, empty_data1D
from pathlib import Path
import pandas as pd
from bumps.fitters import fit
from datetime import datetime

def get_hardsphere_model(model):
    model.A_radius.range(5, 100)
    model.background.range(1e-3, 1e+11)
    model.scale.range(1e-3, 1e+11)
    cutoff = 1e-5

    return model

def bumps_fit(fp: Path, model:str, fixed_params: dict, param_limits: dict, q_lims=None, method='dream'):
    # fp = '/Users/ianbillinge/Documents/yiplab/projects/resonant/2023-11-30-BM-12/fluor_subbed_data/saxs/data/chis_trimmed/TOA_LiBr_org_phase_1_13.49_keV.chi'
    # fp = '/Users/ianbillinge/Documents/yiplab/projects/resonant/2023-11-30-BM-12/fluor_subbed_data/saxs/data/chis_trimmed/DIPA_KBr_aq_phase-a_13.472_keV.chi'
    # fp = '/Users/ianbillinge/Documents/yiplab/projects/resonant/2023-11-30-BM-12/fluor_subbed_data/saxs/data/csvs/TOA_LiBr_org_phase_1_13.49_keV.csv'


    # data = load_data(fp)
    # data.dy = 0.01 * data.y  # assign an arbitrary 1% error bar
    df = pd.read_csv(fp)

    if q_lims is None:
        pass
    else:
        df = df.where(df['q'] < q_lims[1]).where(df['q'] > q_lims[0]).dropna()

    data = empty_data1D(df['q'])
    data.y, data.dy = df['I'], df['dI']

    kernel = load_model(model)

    model = Model(kernel,
                  )
    help()
    if model == 'sphere*hardsphere':
        model = get_default_hardsphere_model_params(model)


    M = Experiment(data=data, model=model, cutoff=cutoff)

    problem = FitProblem(M)  # to run: bumps 07_fit_sphere_no_main.py --fit=dream --store=T1
    result = fit(problem, method=method, xtol=1e-6, ftol=1e-8)
    return result, problem

def main():
    fp = Path('/Users/ianbillinge/Documents/'
              'yiplab/projects/saxs_new/T_ambient/2024-06-26-with-salts/processed/kapton_maxs.csv')
    mdl = 'sphere*hardsphere'
    # mdl = 'sphere'
    fixed_params = {'scale':1, 'A_sld': 7.22, 'A_sld_solvent': 9.47}
    param_ranges = {'A_radius': (0.1, 100)}
    result, problem = bumps_fit(fp, model=mdl, fixed_params=fixed_params, param_limits=param_ranges, q_lims=(0.05, 0.5))
    print(result)

    outfile = Path('/Users/ianbillinge/Documents/yiplab/projects/saxs_new/T_ambient/2024-06-26-with-salts/fit_results/test.txt')
    # outfile.touch(outfile)
    with open(outfile, 'w') as outfile:
        outfile.write('Model: {:s}\n'.format(mdl))
        d = datetime.now()
        outfile.write('Fitting performed on: {}\n'.format(d))
        outfile.write(str(problem.summarize()))
        outfile.write('\n')
        # outfile.write(str(result))
    help(problem)

    return

if __name__ == '__main__':
    main()