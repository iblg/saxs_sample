import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.stats import linregress
import numpy as np
from pathlib import Path

class Saxs_Sample:

    def __init__(self,
                 infile,
                 name,
                 qbounds=None,
                 abs_int=None,
                 background=None,
                 background_scaling_factor=None,
                 background_sub_type='simple',
                 save_to_file=None,
                 model_infile=None,
                 thickness=None,
                 water_intensity_calibrant=None,  # should be a saxs_sample object
                 verbose_flag=False,
                 image_numbers = None,
                 ):
        """

        """
        if qbounds is None:
            qbounds = {'waxs': [0, 5], 'maxs': [0, 5], 'saxs': [0, 5], 'esaxs': [0, 5]}

        self.verbose_flag = verbose_flag
        self.infile = infile
        self.name = name
        self.qbounds = qbounds

        self.raw = self.open(self.infile)

        self.metadata = read_metadata_from_gradfile(self.raw) # get metadata

        self.cleaned = self.clean() # clean the raw file
        self.waxs, self.maxs, self.saxs, self.esaxs = self.get_types() # figure out which data is SAXS, MAXS, etc.

        self.bck = background
        self.bck_scalar = background_scaling_factor
        self.bck_type = background_sub_type
        self.thickness = thickness

        self.water_intensity_calibrant = water_intensity_calibrant
        self.image_numbers = image_numbers
        self.stitch_params = [1,1,1,1]
        # correct for background
        if background is None:
            pass
        else:  # perform background subtraction
            if type(self.bck) is Saxs_Sample:
                pass
            else:
                raise TypeError('Background passed to sub is not a Saxs_Sample object!')

            if self.bck_type == 'simple':
                self.waxs_s, self.maxs_s, self.saxs_s, self.esaxs_s = self.sub()
            elif self.bck_type == 'minimize_range':
                self.waxs_s, self.maxs_s, self.saxs_s, self.esaxs_s = self.complex_sub()
            else:
                print('Background subtraction type not recognized.')

        # if self.water_intensity_calibrant is not None:
        #     print(
        #         'Calibrating sample {} using {} intensity calibrant.'.format(self.name, water_intensity_calibrant.name))
        #     self.waxs, self.maxs, self.saxs, self.esaxs = self.calibrate_intensity()
        # else:  # if no calibrant is passed
        #     if transmission is None:
        #         pass
        #     else:
        #         print('Doing intensity calibration using transmission factor')
        #         self.waxs, self.maxs, self.saxs, self.esaxs = self.correct_transmission()
        #
        #     if abs_int is None:
        #         pass
        #     else:
        #         print('Doing intensity calibration using abs. int.')
        #         self.waxs, self.maxs, self.saxs, self.esaxs = self.correct_abs_int()

        self.uni = self.unify()

        if thickness is not None:
            # print('Correcting for thickness {:1.4f} in sample {}'.format(self.thickness, self.name))
            # print(self.uni.columns)
            self.uni['I'] = self.uni['I'] / self.thickness
            self.uni['dI'] = self.uni['dI'] / self.thickness

        if model_infile is None:
            pass
        else:
            self.model = self.get_model(model_infile)

        if save_to_file is None:
            pass
        else:
            self.uni.to_csv(save_to_file + '.csv', index=False)


        return


    def read_transmission_from_file(self):
        return read_metadata_item_from_gradfile(self, 'sample_transfact', print_flag=self.verbose_flag)

    def read_I0_from_file(self):
        return read_metadata_item_from_gradfile(self, 'saxsconf_Izero', print_flag=self.verbose_flag)

    def read_t_from_file(self):
        return read_metadata_item_from_gradfile(self, 'det_exposure_time', print_flag=self.verbose_flag)

    def get_calibration_factors_from_water(self):
        factors = {
            'waxs': 0,
            'maxs': 0,
            'saxs': 0,
            'esaxs': 0,

        }
        water = self.water_intensity_calibrant
        fig, ax = plt.subplots()

        ax.plot(water.waxs.q, water.waxs.I, 'o')
        plt.show()
        qmin = float(input('Please enter qmin for fitting.'))
        qmax = float(input('Please enter qmax for fitting.'))
        plt.close()

        waxs = water.waxs.where(water.waxs.q < qmax).where(water.waxs.q > qmin).dropna()

        result = linregress(waxs.q, waxs.I)
        print(result)
        fig, ax = plt.subplots()
        ax.plot(waxs.q, waxs.I, 'o')
        ax.plot(water.waxs.q, water.waxs.q * result.slope + result.intercept, '-')
        plt.show()
        factors['waxs'] = result[1]

        return factors

    def calibrate_intensity(self):
        # for each setting, fit I(q) in high q
        factors = self.get_calibration_factors_from_water()
        self.waxs = self.waxs / factors['waxs']
        self.maxs = self.maxs / factors['maxs']
        self.saxs = self.saxs / factors['saxs']
        self.esaxs = self.esaxs / factors['esaxs']
        return self.waxs, self.maxs, self.saxs, self.esaxs

    def correct_transmission(self):
        try:
            self.waxs['I'] = self.waxs['I'] / self.transmission
            self.waxs['dI'] = self.waxs['dI'] / self.transmission
        except KeyError:
            pass

        try:
            self.maxs['I'] = self.maxs['I'] / self.transmission
            self.maxs['dI'] = self.maxs['dI'] / self.transmission
        except KeyError:
            pass

        try:
            self.saxs['I'] = self.saxs['I'] / self.transmission
            self.saxs['dI'] = self.saxs['dI'] / self.transmission
        except KeyError:
            pass

        try:
            self.esaxs['I'] = self.esaxs['I'] / self.transmission
            self.esaxs['dI'] = self.esaxs['dI'] / self.transmission
        except KeyError:
            pass

        return self.waxs, self.maxs, self.saxs, self.esaxs

    def correct_abs_int(self):
        if type(self.abs_int) is dict:
            pass
        else:
            print('abs_int for sample {} is not a dict!'.format(self.name))
        sc_types = ['waxs', 'maxs', 'saxs', 'esaxs']

        for k in self.abs_int.keys():
            if k in sc_types:
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

    def get_model(self, filename):
        df = pd.read_csv(filename, delimiter='e')
        return df

    def clean(self):
        """
        Read a grad file.
        """
        lines = self.raw.copy()
        del lines[0:4]  # delete the metadata and headers
        del lines[
            -10:]  # delete the footers (note: this should be changed in the future so that it changes to exclude everything below "#Header"
        lines2 = []
        # lines = [line.replace('\n', '') for line in lines]
        # lines = [line.replace('\"', '') for line in lines]
        # lines = [line.split(',') for line in lines]
        # lines = [list(filter(('').__ne__), line) for line in lines]
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
             filepath=None, display=True, legend=False, color=None,
             ylabel=r'$I(q)$, a.u.'):
        """

        :param xscale: str, default 'log'
        The scale to use with x-axis. Allowed values are 'linear', 'log',... or Matplotlib.scale.scalebase objects.
        :param yscale: str, default 'log'
        The scale to use with x-axis. Allowed values are 'linear', 'log',... or Matplotlib.scale.scalebase objects.
        :param fig: matplotlib.pyplot.figure, default None
        The figure to plot onto. If None, a new figure will be created.
        :param ax: matplotlib.axes.Axes, default None
        The axes to plot onto. If None, new axes will be created.
        :param filepath:
        :param display: bool, default True
        Whether to display the plot. If True, the plot will be displayed.
        :param legend: bool, default False
        Whether to include a legend in the plot. If False, no legend will be displayed.
        :param color: str, default None
        The color to use for the curve. If None, the default Matplotlib style will dictate the color of the plot.
        :return:
        """
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
        ax.set_ylabel(ylabel)

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

        if self.bck_scalar is None:
            c = 1  # simple subtraction
        else:
            print('Subtracting background using {} as a scale factor for sample {}'.format(self.bck_scalar, self.name))
            c = self.bck_scalar
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

        waxs_sub, maxs_sub, saxs_sub, esaxs_sub = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        c0 = np.array([1])
        # def sub(sctype):
        #     try:
        #         subbed_data = pd.DataFrame()
        #         c = opt.minimize(resid, c0, args=(self.sctype.I, self.bck.waxs.I)).fun
        #         subbed_data['q'] = self.sctype['q']
        #         subbed_data['I'] = self.sctype['I'] - c * self.bck.sctype['I']
        #         subbed_data['dI'] = self.sctype['dI'] + c * self.bck.sctype['dI']
        #         return subbed_data
        #     except KeyError as ke:
        #         return pd.DataFrame()



        try:
            c = opt.minimize(resid, c0, args=(self.waxs.I, self.bck.waxs.I)).fun
            waxs_sub['q'] = self.waxs['q']
            waxs_sub['I'] = self.waxs['I'] - c * self.bck.waxs['I']
            waxs_sub['dI'] = self.waxs['dI'] + c * self.bck.waxs['dI']
        except KeyError as ke:
            waxs_sub = pd.DataFrame()

        try:
            c = opt.minimize(resid, c0, args=(self.maxs['I'], self.bck.maxs['I'])).fun
            maxs_sub['q'] = self.maxs['q']
            maxs_sub['I'] = self.maxs['I'] - c * self.bck.maxs['I']
            maxs_sub['dI'] = self.maxs['dI'] + c * self.bck.maxs['dI']
        except KeyError as ke:
            maxs_sub = pd.DataFrame()


        try:
            c = opt.minimize(resid, c0, args=(self.saxs['I'], self.bck.saxs['I'])).fun

            saxs_sub['q'] = self.saxs['q']
            saxs_sub['I'] = self.saxs['I'] - c * self.bck.saxs['I']
            saxs_sub['dI'] = self.saxs['dI'] + c * self.bck.saxs['dI']
        except KeyError as ke:
            saxs_sub = pd.DataFrame()

        try:
            c = opt.minimize(resid, c0, args=(self.esaxs['I'], self.bck.esaxs['I'])).fun

            esaxs_sub['q'] = self.esaxs['q']
            esaxs_sub['I'] = self.esaxs['I'] - c * self.bck.esaxs['I']
            esaxs_sub['dI'] = self.esaxs['dI'] + c * self.bck.esaxs['dI']
        except KeyError as ke:
            esaxs_sub = pd.DataFrame()

        return waxs_sub, maxs_sub, saxs_sub, esaxs_sub


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


def calibrate_intensity_using_water(water: Saxs_Sample):
    int_factors = {'waxs': 1, 'maxs': 1, 'saxs': 1, 'esaxs': 1}

    types = ['waxs', 'maxs', 'saxs', 'esaxs']
    datatypes = [water.waxs, water.maxs, water.saxs, water.esaxs]
    for type, data in zip(types, datatypes):
        print('Fitting {}'.format(type))

        fig, ax = plt.subplots()
        ax.plot(data.q, data.I, 'o', label='un-normalized data')
        plt.show()
        qmin = float(input('Please enter qmin for fitting.'))
        qmax = float(input('Please enter qmax for fitting.'))

        data_in_range = data.where(data.q < qmax).where(data.q > qmin).dropna()

        result = linregress(data.q, data.I)
        print(result)
        xx = np.linspace(0, data.q.max(), 2)
        ax.plot(xx, xx * result.slope + result.intercept, '-', label='fit')

        I0 = result.intercept
        ax.plot(data.q, data.I / I0, label='normalized data')
        plt.show()
        int_factors['waxs'] = result[1]

    return int_factors


def read_metadata_from_gradfile(lines: list):
    """
    Read metadata from a grad file.
    :param lines: Can be either list, if gradfile has already been read, or str, a filepath pointing to the grad file.
    :return:
    """
    if isinstance(lines, list):
        pass
    elif isinstance(lines, str) or isinstance(lines, Path):
        with open(lines, 'r') as infile:
            lines = infile.readlines()

    lines = [i.split('>   <') for i in lines if '</' in i][0]
    lines = [i for i in lines if '<' in i]
    md_keys = [i.split('>')[0] for i in lines]
    md_vals = [i.split('>')[1].split('<')[0] for i in lines]

    md = {}
    for k, v in zip(md_keys, md_vals):
        try:
            v = float(v)
        except ValueError as ve:
            pass

        md[k] = v

    return md


def read_metadata_item_from_gradfile(sample, item_name: str, print_flag: bool = False):
    lines = sample.raw
    start = '<' + item_name + '>'
    end = '</' + item_name + '>'
    md_item = [l for l in lines if start in l]
    md_item = [i.split(start) for i in md_item][0]
    md_item = [i for i in md_item if end in i]
    md_item = [i.split(end) for i in md_item][0][0]
    md_item = float(md_item)

    if print_flag: print(item_name, md_item)

    return md_item


def main():
    qbounds = {'waxs': [0.0, 5],
               'maxs': [0.0, 5],
               'saxs': [0, 5],
               'esaxs': [0, 5]}
    abs_int = {'waxs': 0.0232, 'maxs': 0.0415, 'saxs': None, 'esaxs': 0.02318}
    # transmission = 1.0464

    fig, ax = plt.subplots()
    kapton = Saxs_Sample(
        '/Users/ianbillinge/Documents/yiplab/projects/saxs_amine/2022_expts/2022-04-29/from_import/kapton.grad',
        'kapton',
        qbounds)
    kapton.plot(ax=ax)

    water_fp = '/Users/ianbillinge/Documents/yiplab/projects/saxs_amine/2022_expts/2022-04-29/from_import/water.grad'
    water = Saxs_Sample(water_fp,
                        'water',
                        qbounds,
                        abs_int=None, background=kapton, thickness=0.2)
    water.plot(ax=ax)
    filename = '/Users/ianbillinge/Documents/yiplab/projects/saxs_amine/2022_expts/2022-04-29/from_import/dipa05.grad'

    # dipa_22 = Saxs_Sample(filename, '22', qbounds,
    #                       transmission=None,
    #                       abs_int=abs_int,
    #                       background=kapton,
    #                       thickness=0.2,
    #                       water_intensity_calibrant=water
    #                       )
    water2 = Saxs_Sample(water_fp, '22', qbounds,
                         abs_int=None,
                         background=kapton,
                         thickness=0.2,
                         water_intensity_calibrant=water
                         )
    water2.plot(ax=ax, )

    # water2.plot(yscale='linear')
    ax.legend()
    return


if __name__ == '__main__':
    main()
