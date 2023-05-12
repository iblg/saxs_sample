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
