import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from saxs_sample import Saxs_Sample as Sample
from scipy.optimize import curve_fit

def q4(q, a, b):
    return a * q**-4 + b

def fit_q4(sample: Sample, qrange:tuple):
    data = sample.uni
    data = data.where(data.q < qrange[1]).where(data.q > qrange[0]).dropna()
    popt = curve_fit(q4, data.q, data.I, p0 = np.array([1, 1]))
    return popt
#
def main():
    qbounds = {'waxs':(0.1,0.5), 'maxs':(0.04, 0.5), 'saxs': (0,0), 'esaxs': (0,0)}
    kapton_fp = '/Users/ianbillinge/Documents/yiplab/projects/saxs_amine/2023_expts/2023-02-10-2mpd/from_import/kapton.grad'
    kapton = Sample(kapton_fp, 'kapton', qbounds=qbounds)
    fp = '/Users/ianbillinge/Documents/yiplab/projects/saxs_amine/2023_expts/2023-02-10-2mpd/from_import/mpd03.grad'
    mpd = Sample(fp, 'mpd', background=kapton, qbounds = qbounds)
    fig, ax = plt.subplots()
    ax.plot(mpd.uni['q'], mpd.uni['I'])
    # ax.set_xscale('log')
    ax.set_yscale('log')

    qr = (0.37,0.5)
    q4fit = fit_q4(mpd, qr)
    print(*q4fit)
    x = np.linspace(qr[0], qr[1], 50)
    y = q4(x, *q4fit)
    ax.plot(x, y)

    plt.show()
    return

if __name__ == '__main__':
    main()