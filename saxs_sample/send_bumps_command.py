import pyautogui as gui
from pyautogui import keyDown, keyUp, press, write
from pathlib import Path

def send_bumps_command(fitting_file: str, storage_dir: str,
                       algo: str = 'dream',
                       overwrite=False,
                       noshow=True, **kwargs):
    keyDown('command')
    press('space')
    keyUp('command')
    write('Terminal')
    press('enter')

    write('bumps {:s} --fit={:s} --store={:s}'.format(fitting_file, algo, storage_dir))
    if overwrite:
        write(' --overwrite')
    if noshow:
        write(' --noshow')

    press('enter')
    return


def main():
    fitting_file = '/Users/ianbillinge/Documents/yiplab/projects/saxs_new/T_ambient/2024-06-26-with-salts/fitting' \
                   '/fit_all.py'
    data_files = '/Users/ianbillinge/Documents/yiplab/projects/saxs_new/T_ambient/2024-06-26-with-salts/processed/'
    storage_dir = '/Users/ianbillinge/Documents/yiplab/projects/saxs_new/T_ambient/2024-06-26-with-salts/fitting' \
                  '/dipa40_maxs'
    p = Path('Users/ianbillinge')
    fitting_files = []
    send_bumps_command(fitting_file, storage_dir)
    return


if __name__ == '__main__':
    main()
