import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np


class Saxs_Sample:
    '''
    Takes in a Grad file. Returns a saxs sample object.

    infile : str
        The file path to the .grad input file.
    name : str
        The name attached to the Saxs_Sample object. This is mostly for the user's own clarity.
    qbounds : dict
        q cutoffs for the waxs, maxs, saxs, and esaxs settings, in inverse A.

        Default is {'waxs': [0, 5], 'maxs': [0, 5], 'saxs': [0, 5], 'esaxs': [0, 5]}, i.e., no q cutoff.
    savefile : str
        The filepath to the csv file. savefile should not need to end in '.csv'. That is applied automatically.
    modelfile : str
        The filepath to the input model file. This should be a csv file. The program was designed to accept sasView-generated csv files.
    thickness : float
        The sample thickness in cm. If thickness is provided, the scattering will be divided by thickness, putting the scattering in absolute scale.
    stitch_params : dict, default None
        If true, the sample will be arbitratily stitched together.
    '''

    def __init__(self,
                 infile,
                 name,
                 qbounds={'waxs': [0, 5], 'maxs': [0, 5], 'saxs': [0, 5], 'esaxs': [0, 5]},
                 abs_int=None,
                 transmittance=None,
                 background=None,
                 save_to_file=None,
                 model_infile=None,
                 thickness=None,
                 stitch_params=None):

        self.infile = infile
        self.name = name
        self.qbounds = qbounds

        self.raw = self.open(self.infile)
        self.cleaned = self.clean()
        self.waxs, self.maxs, self.saxs, self.esaxs = self.get_types()
        self.bck = background
        self.thickness = thickness
        # self.stitch_params = stitch_params

        self.transmittance = transmittance
        self.abs_int = abs_int

        if transmittance is None:
            pass
        else:
            self.waxs, self.maxs, self.saxs, self.esaxs = self.correct_transmittance()

        if abs_int is None:
            pass
        else:
            self.waxs, self.maxs, self.saxs, self.esaxs = self.correct_abs_int()

        if background is None:
            pass
        else: # perform background subtraction
            if type(self.bck) is Saxs_Sample:
                pass
            else:
                raise TypeError('Background passed to sub is not a Saxs_Sample object!')

            self.waxs_s, self.maxs_s, self.saxs_s, self.esaxs_s = self.sub()

        self.uni = self.unify()

        if thickness is not None:
            # print('Correcting for thickness {:1.4f} in sample {}'.format(self.thickness, self.name))
            # print(self.uni.columns)
            self.uni['I'] = self.uni['I'] / self.thickness
            # self.uni = self.uni['dI']/self.thickness

        if model_infile is None:
            pass
        else:
            self.model = self.get_model(model_infile)

        if save_to_file is None:
            pass
        else:
            self.uni.to_csv(save_to_file + '.csv', index=False)

        return

    def correct_transmittance(self):
        try:
            self.waxs['I'] = self.waxs['I'] / self.transmittance
            self.waxs['dI'] = self.waxs['dI'] / self.transmittance
        except KeyError:
            pass

        try:
            self.maxs['I'] = self.maxs['I'] / self.transmittance
            self.maxs['dI'] = self.maxs['dI'] / self.transmittance
        except KeyError:
            pass

        try:
            self.saxs['I'] = self.saxs['I'] / self.transmittance
            self.saxs['dI'] = self.saxs['dI'] / self.transmittance
        except KeyError:
            pass

        try:
            self.esaxs['I'] = self.esaxs['I'] / self.transmittance
            self.esaxs['dI'] = self.esaxs['dI'] / self.transmittance
        except KeyError:
            pass

        return self.waxs, self.maxs, self.saxs, self.esaxs

    def correct_abs_int(self):
        if type(self.abs_int) is dict:
            pass
        else:
            print('abs_int for sample {} is not a dict!'.format(self.name))

        for k in self.abs_int.keys():
            if k in ['waxs', 'maxs', 'saxs', 'esaxs']:
                pass
            else:
                print('abs_int for sample {} contains an unrecognized key.'.format(self.name))
                print('The only allowed keys in abs_int are \'waxs\', \'maxs\', \'saxs\', and \'esaxs\'.')


        try:
            self.waxs['I'] = self.waxs['I'] / self.abs_int['waxs']
            self.waxs['dI'] = self.waxs['dI'] / self.abs_int['waxs']
        except KeyError:
            pass

        try:
            self.maxs['I'] = self.maxs['I'] / self.abs_int['maxs']
            self.maxs['dI'] = self.maxs['dI'] / self.abs_int['maxs']
        except KeyError:
            pass

        try:
            self.saxs['I'] = self.saxs['I'] / self.abs_int['saxs']
            self.saxs['dI'] = self.saxs['dI'] / self.abs_int['saxs']
        except KeyError:
            pass

        try:
            self.esaxs['I'] = self.esaxs['I'] / self.abs_int['esaxs']
            self.esaxs['dI'] = self.esaxs['dI'] / self.abs_int['esaxs']
        except KeyError:
            pass

        return self.waxs, self.maxs, self.saxs, self.esaxs

    def unify(self):
        """This should return a single dataframe per sample which is easier to plot."""
        uni = pd.DataFrame(columns=self.waxs.columns)

        # if self.stitch_params is None:
        #     factors = self.get_multiplicative_factors()
        # else:
        #     factors = self.abs_int
        #     # factors = [1, 1, 1, 1]

        if self.bck == None:
            waxs, maxs, saxs, esaxs = self.waxs, self.maxs, self.saxs, self.esaxs
            # uni = pd.concat([self.waxs * factors[0], self.maxs * factors[1], self.saxs * factors[2], self.esaxs * factors[3]], ignore_index = True)
        else:
            waxs, maxs, saxs, esaxs = self.waxs_s, self.maxs_s, self.saxs_s, self.esaxs_s
        types = [waxs, maxs, saxs, esaxs]

        # for f, t in zip(factors, types):
        #     print(t.columns)
        #     try:
        #         t['I'], t['dI'] = f * t['I'], f * t['dI']
        #     except KeyError as ke:
        #         print(ke)
            # uni = pd.concat([self.waxs_s * factors[0], self.maxs_s * factors[1], self.saxs_s * factors[2], self.esaxs_s * factors[3]], ignore_index = True)

        uni = pd.concat([waxs, maxs, saxs, esaxs], ignore_index=True)
        uni = uni.sort_values(by=['q'])
        uni = uni.dropna()
        return uni

    def get_multiplicative_factors(self):
        """

        """
        if isinstance(self.stitch_params, dict):
            pass
        else:
            print('self.stitch_params must either be None or a dict!')

        if self.bck == None:
            # stitch together maxs etc
            waxs, maxs, saxs, esaxs = self.waxs, self.maxs, self.saxs, self.esaxs
        else:
            waxs, maxs, saxs, esaxs = self.waxs_s, self.maxs_s, self.saxs_s, self.esaxs_s

        factors = [1, 1, 1, 1]
        # Do the MAXS/WAXS stitching
        try:
            waxs_comp = waxs.where(
                (waxs['q'] > self.stitch_params['maxs'][0]) & (waxs['q'] < self.stitch_params['maxs'][1])).dropna()
            maxs_comp = maxs.where(
                (maxs['q'] > self.stitch_params['maxs'][0]) & (maxs['q'] < self.stitch_params['maxs'][1])).dropna()
            print('In maxs/saxs bridging')
            print(waxs_comp)
            print(maxs_comp)
            mca = waxs_comp['I'].mean() / maxs_comp['I'].mean()
            # maxs['I'], maxs['dI'] = mca * maxs['I'], mca * maxs['dI']
            factors[1] = mca
        except KeyError as ke:
            mca = 1
            print(ke)

        # Do the SAXS/MAXS stitching
        try:
            maxs_comp = maxs.where(
                (maxs['q'] > self.stitch_params['saxs'][0]) & (maxs['q'] < self.stitch_params['saxs'][1])).dropna()
            saxs_comp = saxs.where(
                (saxs['q'] > self.stitch_params['saxs'][0]) & (saxs['q'] < self.stitch_params['saxs'][1])).dropna()
            sca = maxs_comp['I'].mean() / saxs_comp['I'].mean()
            # saxs['I'], saxs['dI'] = sca * saxs['I'], sca * saxs['dI']
            factors[2] = sca * mca
        except KeyError as ke:
            sca = 1
            print(ke)

        # Do the ESAXS/MAXS stitching
        try:
            maxs_comp = maxs.where(
                (maxs['q'] > self.stitch_params['esaxs'][0]) & (maxs['q'] < self.stitch_params['esaxs'][1])).dropna()
            esaxs_comp = esaxs.where(
                (esaxs['q'] > self.stitch_params['esaxs'][0]) & (esaxs['q'] < self.stitch_params['esaxs'][1])).dropna()
            eca = maxs_comp['I'].mean() / esaxs_comp['I'].mean()
            # esaxs['I'], esaxs['dI'] = eca * esaxs['I'], eca * esaxs['dI']
            factors[3] = eca * mca * sca
        except KeyError as ke:
            eca = 1
            print(ke)

        return factors

    def open(self, filename):
        with open(filename, 'r') as infile:
            lines = infile.readlines()
        return lines

    def get_model(self, model_infile):
        lines = []
        with open(model_infile, 'r') as openfile:
            lines = openfile.readlines()

        del (lines[0])
        lines = [line.split() for line in lines]
        lines = [[str.split('e') for str in line] for line in lines]
        lines = [[[float(num) for num in pair] for pair in line] for line in lines]
        lines = [[pair[0] * 10 ** pair[1] for pair in line] for line in lines]

        lines = pd.DataFrame(lines, columns=['q', 'I'])

        return lines

    def clean(self):
        lines = self.raw
        del lines[0:4]  # delete the metadata and headers
        del lines[
            -10:]  # delete the footers (note: this should be changed in the future so that it changes to exclude everything below "#Header"
        lines2 = []
        for line in lines:
            line2 = line.replace('\n', '')  # replace all \n with empty strings
            line2 = line2.replace('\"', '')  # replace all quotation marks with empty strings
            line2 = line2.split(',')  # split by commas
            line2 = list(filter(('').__ne__, line2))  # remove all instances of empty strings
            lines2.append(line2)
        lines = lines2
        del lines2
        lines = pd.DataFrame(lines[2:], dtype='float')
        return lines

    def get_types(self):
        '''
        Takes cleaned dataframe object and returns list of indices saying which is WAXS, MAXS, SAXS, ESAXS
        '''
        waxs, maxs, saxs, esaxs = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        type_list = []
        for i in range(self.cleaned.shape[1]):  # for each column,
            if i % 3 == 0:
                type_list.append(self.cleaned.iloc[:, i:i + 3])  # if it is a q column, make a new dataframe

        for i, type in enumerate(type_list):
            type_list[i] = type.dropna(axis=0, how='any')  # remove any rows with nan values
            type_list[i] = type.rename(columns={type.columns[0]: 'q', type.columns[1]: 'I', type.columns[2]: 'dI'})

        for i, type in enumerate(type_list):
            # if
            qmax = type.iloc[-1, 0]
            if qmax > 1.4:  # if waxs:
                # print(self.qbounds['waxs'][0])
                filter1 = type['q'] > self.qbounds['waxs'][0]
                filter2 = type['q'] < self.qbounds['waxs'][1]
                waxs = type.where(filter1 & filter2)

            elif qmax < 1.4 and qmax > 0.5:  # if maxs
                filter1 = type['q'] > self.qbounds['maxs'][0]
                filter2 = type['q'] < self.qbounds['maxs'][1]
                maxs = type.where(filter1 & filter2)

            elif qmax < 0.5 and qmax > 0.25:  # if saxs
                filter1 = type['q'] > self.qbounds['saxs'][0]
                filter2 = type['q'] < self.qbounds['saxs'][1]
                saxs = type.where(filter1 & filter2)

            else:  # if esaxs
                filter1 = type['q'] > self.qbounds['esaxs'][0]
                filter2 = type['q'] < self.qbounds['esaxs'][1]
                esaxs = type.where(filter1 & filter2)

        return waxs, maxs, saxs, esaxs

    def plot(self,
             xscale='log', yscale='log',
             fig=None, ax=None,
             filepath=None, display=True, legend=False, color=False):

        if fig is not None and ax is not None:
            fig = fig
            ax = ax
        else:
            fig, ax = plt.subplots()

        if color:
            cl = color
        else:
            cl = None

        ax.plot(self.uni['q'], self.uni['I'], '.', label=self.name, color=cl, alpha=0.3)
        try:
            ax.plot(self.model['q'], self.model['I'], '-', color='black')
        except AttributeError as ae:
            print(ae)

        if xscale == 'log':
            ax.set_xscale('log')
        if yscale == 'log':
            ax.set_yscale('log')

        ax.set_xlabel('q, Å' + r'$^{-1}$')
        ax.set_ylabel('I, cm' + r'$^{-1}$')

        if legend is True:
            ax.legend()

        if filepath is not None:
            plt.savefig(filepath, bbox_inches='tight')

        if display:
            plt.show()

        return fig, ax

    def sub(self):
        """
        sample = a saxs_sample object. Should contain data
        bck = a saxs_sample object.
        """

        if not type(self.bck) is Saxs_Sample:
            raise TypeError('background passed to sub is not a saxs_sample object!')

        c = 1  # simple subtraction

        try:
            waxs_sub = pd.DataFrame()
            waxs_sub['q'] = self.waxs['q']
            waxs_sub['I'] = self.waxs['I'] - c * self.bck.waxs['I']
            waxs_sub['dI'] = self.waxs['dI'] + c * self.bck.waxs['dI']
        except KeyError as ke:
            # print('No waxs in saxs_sample.sub()')
            print(ke)

        try:
            maxs_sub = pd.DataFrame()
            maxs_sub['q'] = self.maxs['q']
            maxs_sub['I'] = self.maxs['I'] - c * self.bck.maxs['I']
            maxs_sub['dI'] = self.maxs['dI'] + c * self.bck.maxs['dI']
        except KeyError as ke:
            maxs_sub = pd.DataFrame()
            print(ke)
            # print('No maxs in saxs_sample.sub()')

        try:
            saxs_sub = pd.DataFrame()
            saxs_sub['q'] = self.saxs['q']
            saxs_sub['I'] = self.saxs['I'] - c * self.bck.saxs['I']
            saxs_sub['dI'] = self.saxs['dI'] + c * self.bck.saxs['dI']
        except KeyError as ke:
            saxs_sub = pd.DataFrame()
            print(ke)
            # print('No saxs in saxs_sample.sub()')

        try:
            esaxs_sub = pd.DataFrame()
            esaxs_sub['q'] = self.esaxs['q']
            esaxs_sub['I'] = self.esaxs['I'] - c * self.bck.esaxs['I']
            esaxs_sub['dI'] = self.esaxs['dI'] + c * self.bck.esaxs['dI']
        except KeyError as ke:
            print(ke)
            esaxs_sub = pd.DataFrame()
            # print('No esaxs in saxs_sample.sub()')

        # return
        return waxs_sub, maxs_sub, saxs_sub, esaxs_sub

    def complex_sub(self):
        if not type(self.bck) is Saxs_Sample:
            raise TypeError('background passed to sub is not a saxs_sample object!')

        waxs_sub, maxs_sub, saxs_sub, esaxs_sub = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),

        try:
            c = opt.minimize(resid, 100, args=(self.waxs.I, self.bck.waxs.I)).fun
            # print('waxs_sub: {}'.format(c))
            waxs_sub['q'] = self.waxs['q']
            waxs_sub['I'] = self.waxs['I'] - c * self.bck.waxs['I']
            waxs_sub['dI'] = self.waxs['dI'] + c * self.bck.waxs['dI']
        except KeyError as ke:
            waxs_sub = pd.DataFrame()
            # print('No waxs in saxs_sample.sub()')

        try:
            c = opt.minimize(resid, 100, args=(self.maxs['I'], self.bck.maxs['I'])).fun
            # print('maxs_sub: {}'.format(c))
            maxs_sub['q'] = self.maxs['q']
            maxs_sub['I'] = self.maxs['I'] - c * self.bck.maxs['I']
            maxs_sub['dI'] = self.maxs['dI'] + c * self.bck.maxs['dI']
        except KeyError as ke:
            maxs_sub = pd.DataFrame()

            # print('No maxs in saxs_sample.sub()')

        try:
            c = opt.minimize(resid, 100, args=(self.saxs['I'], self.bck.saxs['I'])).fun
            # print('saxs_sub: {}'.format(c))

            saxs_sub['q'] = self.saxs['q']
            saxs_sub['I'] = self.saxs['I'] - c * self.bck.saxs['I']
            saxs_sub['dI'] = self.saxs['dI'] + c * self.bck.saxs['dI']
        except KeyError as ke:
            saxs_sub = pd.DataFrame()
            # print('No saxs in saxs_sample.sub()')

        try:
            c = opt.minimize(resid, 100, args=(self.esaxs['I'], self.bck.esaxs['I'])).fun
            # print('esaxs_sub: {}'.format(c))

            esaxs_sub['q'] = self.esaxs['q']
            esaxs_sub['I'] = self.esaxs['I'] - c * self.bck.esaxs['I']
            esaxs_sub['dI'] = self.esaxs['dI'] + c * self.bck.esaxs['dI']
        except KeyError as ke:
            esaxs_sub = pd.DataFrame()
            # print('No esaxs in saxs_sample.sub()')

        uni = pd.concat([waxs_sub, maxs_sub, saxs_sub, esaxs_sub], ignore_index=True)
        uni = uni.sort_values(by=['q']).dropna()
        return uni


#
def resid(c, sc, bck):
    '''
    c : float
    A scalar

    sc : numpy.array or similar
    The scattering data.

    bck : np.array or similar
    The background scattering data.

    The target function to minimize. Takes sc, some scattering data, bck, some background data, and c, a scalar.
    '''
    return np.sum((sc - c * bck) ** 2)


def get_c(data):
    '''
    Adds a column to the data which is the appropriate scaling factor for the background subtraction.
    '''
    sub_data = {}

    for key, val in data.items():
        c = sub_bck(val[:, 1], data['empty'][:, 1])['x'][0]
        sub = val[:, 1:2] - data['empty'][:, 1:2] * c

        sub_data[key] = np.append(val, sub, axis=1)

    return sub_data


def f(c, samp, bck):
    resid = np.sqrt(np.sum((samp - c * bck) ** 2))
    return resid


def main():
    filename = '/Users/ianbillinge/Documents/yiplab/projects/saxs_amine/2023-05-09-dpa/from_import/dpa04.grad'
    dpa = Saxs_Sample(filename, 'dpa03', abs_int={'waxs': 1, 'maxs': 1, 'saxs': 1, 'esaxs': 1}, transmittance=10**4)
    dpa.plot()

    return


if __name__ == '__main__':
    main()
