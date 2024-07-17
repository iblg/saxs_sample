import pyFAI, fabio, pyFAI.detectors
import pandas as pd
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator as AI
import numpy as np
import matplotlib.pyplot as plt
import glob
from saxs_sample.read_poni import read_poni
from pathlib import Path
import json
import xarray as xr

from pathlib import Path
def get_poni_params(fp=None):
    """
    :param fp: Filepath to poni file
    :return: detector distance in mm
    """

    params = read_poni(fp)

    return params


def read_metadata(folder, metadata_kwargs):
    """
    :param folder: Folder to search for metadata .txt file
    :return: dict full of metadata
    """
    txt_files = folder.glob('*.txt')

    for file in txt_files:
        with open(file, 'r') as infile:
            md = infile.readlines()

    md = [line.strip()[2:] for line in md]
    md = [line for line in md if '<' not in line]
    md = [line.split(':') for line in md]
    md = [[val.strip() if val != '' else None for val in line] for line in md]
    md = [[val for val in line] for line in md if len(line) >= 2]

    md = {line[0]: line[1] if len(line) == 2 else line[1:] for line in md}

    for key, val in md.items():
        try:
            val = float(val)
        except ValueError as ve:
            pass
        except TypeError as te:
            pass

    if metadata_kwargs is None:
        pass
    else:
        for key, val in metadata_kwargs.items():
            md[key] = val
    return md


def process_tiff_single(
        folder_with_frames,
        tiff_num,
        poni_fp,
        outfile,
        dtr=pyFAI.detectors.Pilatus300k(),
        pilatus_mask=pyFAI.detectors.Pilatus300k().mask,
        beamstop_mask=Path(__file__).parent / 'masks'/'beamstop_mask.csv',
        bad_pixel_mask=Path(__file__).parent / 'masks'/'bad_pixel_mask.csv',
        config='waxs',
        dx_pixel=1.72e-4,
        dy_pixel=1.72e-4,
        beamstop_x=364.21,  # maybe these can be read from a grad file?
        beamstop_y=241.60,
        wvl = 1.5418e-10,
        print_metadata=False,
        plot_total_image=False,
        metadata_kwargs:dict = None,
        img_to_append='total'
):
    metadata = read_metadata(folder_with_frames, metadata_kwargs)

    if print_metadata:
        print(metadata['Meas.Description'])
        print(metadata)

    # beamstop_y = kwargs.get('beamstop_y', 241.60)
    # bmstp_x = 364.21;  # pixel units
    # bmstp_y = 241.60;  # pixel units
    # Pilatus 300k detector
    # dtr = pyFAI.detectors.Pilatus300k()  # PyFAI has the characteristics of our detector built in
    # pilatus_mask = dtr.mask  # Could also get mask from AI class later

    # %% Import beam stop mask
    # Array of ones (indicating valid pixels) and zeros (indicating invalid pixels)
    # I made this in saxsgui--it could probably be improved slightly. I made the mask slightly bigger than necessary
    # to make it more flexible based on small changes to beamstop position

    beamstop_mask = np.genfromtxt(beamstop_mask, delimiter=',', dtype=np.int8)
    beamstop_mask = np.transpose(beamstop_mask)
    beamstop_mask = 1 - beamstop_mask  # 0==valid, 1==invalid

    # %% Import bad pixel mask
    # In any given .tiff file, 11 of the pixels will have values==-2,
    # which means that they are 'bad'. These are these 11 pixels in an array of zeros
    bad_pixel_mask = np.genfromtxt(bad_pixel_mask, delimiter=',',
                                   dtype=np.int8)

    # %% Combine masks (pixels==0 are counted, all others are ignored)
    # Initialize mask
    comb_mask = np.zeros_like(pilatus_mask, dtype=np.int8)

    # Combine the pilatus mask and the custom beamstop mask
    comb_mask[np.logical_or(pilatus_mask, beamstop_mask)] = 1

    # Combine the previous mask and the bad pixel mask
    comb_mask[np.logical_or(comb_mask, bad_pixel_mask)] = 1

    # %% Set up the aximuthal integrator object

    # Experimental detector info
    poni_params = get_poni_params(poni_fp)
    det_distance = poni_params['Distance']
    poni1 = poni_params['Poni1']
    poni2 = poni_params['Poni2']
    # poni1 = poni_params[]
    # det_distance = (180.2378 - 12.5) * 1e-3;  # WAXS conformation, meters
    bmstp_x = 364.21  # pixel units
    bmstp_y = 241.60  # pixel units
    # wvl = 1.5418e-10  # Ang

    # Create object (this al)
    ai = AI(dist=det_distance, detector=dtr, poni1=poni1, poni2=poni2, wavelength=wvl)

    # Choose resolution for integration in q space
    q_res = int(399)  # number of bins

    # frame_rate = 3  # seconds, fixed in the implementation
    # n_frame = int(100)  # Choose the number of frames to average
    # n_frame = int(1)  # Choose the number of frames to average
    # period = frame_rate * n_frame  # seconds

    # %% Load .tiff name indicated by exposure number

    # Empty array of pathnames
    # pnm = [];

    # Empty array of images (will end up being very large)
    img_array = []

    # Get all frames corresponding to that exposure
    pn_list = folder_with_frames.rglob('fr_0{:d}_*.tiff'.format(tiff_num))
    # Open all of the images in that list
    # for i in pn_list:
        # img_array.append(fabio.open(i).data)

    img_array = [fabio.open(i).data for i in pn_list]


    # %% Create average images for the desired time period

    # List of image numbers
    # img_list = np.arange(n_frame - 1, len(img_array), n_frame, dtype=int)

    # Array of averaged images
    img_av = []
    print('image array shape:', len(img_array))
    # Set average image to zero
    av_img = np.zeros_like(img_array[0], dtype=float)
    total_image = av_img.copy()

    # Take average of indicated images
    for i in range(0, len(img_array)):
        av_img += img_array[i] / len(img_array)
        total_image += img_array[i]


    img_av.append(av_img)

    if plot_total_image:
        fig, ax = plt.subplots()
        picture = ax.imshow(total_image, cmap='viridis', vmin=0, vmax=total_image.max())
        fig.colorbar(picture)
        plt.show()
        plt.close()

    # %% Initialize some empty lists

    q = []  # nm-1
    I = []  # cts
    dI = []  # cts

    # %% Integrate each of the averaged images
    # Loop over the number of averaged images
    for i in range(0, len(img_av)):
        # Integrate with Pilatus mask
        res = ai.integrate1d(img_av[i], q_res, mask=comb_mask, error_model="azimuthal")
        # result = ai.integrate1d_legacy(img_av[i], res, error_model = "azimuthal");
        # Store result and filter out the anomalous low-q values
        q.append(res[0])
        I.append(res[1])
        dI.append(res[2])

    # %% Save results in .csv with information about the frame rate

    q = pd.Series(np.array(q[0]), name='q')
    q = q / 10. # correcting from inverse nm to inverse A
    I = pd.Series(np.array(I[0]), name='I')
    dI = pd.Series(np.array(dI[0]), name='dI')

    df = pd.concat([q,I, dI], axis='columns')
    df = df[['q', 'I', 'dI']]
    outfile = Path(outfile).resolve()
    df.to_csv(outfile, index=False)
    jsonfile = outfile.with_suffix('.json')

    with open(jsonfile, 'w') as json_file:
        json.dump(metadata, json_file)

    return

def get_abs_intensity_from_rel_intensity(sample, out_dir = None, verbose=True):
    """

    :param path: pathlib.Path. Should match the stem or sample name, so that the 1d data, written as a csv file, is in the same directory as the metadata, a json file.
    :return:

    """
    if out_dir is None:
        out_dir = sample.resolve().parent
    else:
        pass

    data = pd.read_csv(sample.with_suffix('.csv'))
    with open(sample.with_suffix('.json'), 'r') as infile:
        metadata = json.load(infile)
    # metadata = json.load(str(sample.with_suffix('.json')))

    t = float(metadata['livetime'])
    T = float(metadata['sample_transfact'])
    I0 = float(metadata['saxsconf_Izero'])
    Ieff = float(metadata['saxsconf_Ieff'])

    # check that necessary stuff is there
    if verbose:
        print('Normalizing by measurement time, sample transmission, and I0 for sample {}'.format(sample.stem))
        print('Measurement time: {}'.format(t))
        print('Transmission factor: {}'.format(T))
        print('I0: {}'.format(I0))
        print('Ieff: {}'.format(Ieff))
        print('Further normalization by 10**6 performed (unknown reason why, contact Ian for more details)')


    # data['I_rel'] = data['I']
    data['I'] = data['I'] / (t * T * (I0/10**6) * 10 * Ieff)
    data['dI'] = data['dI'] / (t * T * (I0/10**6) * 10 * Ieff)
    # data['I'] = data['I'] / (t * T * (I0))
    data = data[['q', 'I', 'dI']]

    # for some reason the .grad normalization requires dividing by t * T * I0/10**6
    # but the processing when done by me requires that times ten.
    data.to_csv((out_dir / sample.stem).with_suffix('.csv'), index=False)
    return
