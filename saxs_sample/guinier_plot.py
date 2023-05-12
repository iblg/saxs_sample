import matplotlib.pyplot as plt
from saxs_sample import Saxs_Sample
import numpy as np

def guinier_plot(*samples, display=False, savefig=False, legend=True,qbounds=[0,5]):
    fig, ax = plt.subplots()


    ax.set_xlabel(r'$q^{2}$')
    ax.set_ylabel(r'$\ln{I(q)}$')

    for s in samples:
        x = s.uni['q'].where((s.uni['q'] > qbounds[0]) & (s.uni['q'] < qbounds[1]))
        x = x**2
        y = s.uni['I'].where((s.uni['q'] > qbounds[0]) & (s.uni['q'] < qbounds[1]))
        y = np.log(y)
        ax.plot(x, y, label=s.name)

    if legend:
        ax.legend()

    if display:
        plt.show()

    if savefig:
        plt.savefig(savefig)

    return fig, ax


def main():
    name = 'dpa03'
    filename = '/Users/ianbillinge/Documents/yiplab/projects/saxs_amine/2023-05-09-dpa/from_import/' + name + '.grad'
    stf = '/Users/ianbillinge/Documents/yiplab/projects/saxs_amine/2023-05-09-dpa/csvs/' + name

    qbounds = {'waxs': [0.055, 5], 'maxs': [0.0, 5], 'saxs': [0, 5], 'esaxs': [0.0, 0.05]}

    dpa03 = Saxs_Sample(filename, name, qbounds,
                        # background=kapton,
                        thickness=0.2,
                        save_to_file=stf)

    guinier_plot(dpa03, display=True, qbounds=[0,0.02])

    return


if __name__ == '__main__':
    main()
