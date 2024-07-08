import pandas as pd
# import matplotlib.pyplot as plt

def read_fit_file(fp):
    df = pd.read_csv(fp, sep='e|\s|\t', engine='python', dtype='float'
                     # names=['q_val', 'q_power', 'I_val', 'I_power']
                     )
    df = df.dropna(axis='columns',how="all")
    df2 = pd.DataFrame()
    df2['q'] = df.iloc[:, 0] * 10**df.iloc[:, 1]
    df2['I'] = df.iloc[:, 2] * 10**df.iloc[:, 3]

    return df2
#
# def main():
#     fp = '/Users/ianbillinge/Documents/yiplab/projects/saxs_amine/2023_expts/2023-02-10-2mpd/fits/mpd01_pts.txt'
#     df = read_fit_file(fp)
#     fig, ax = plt.subplots()
#     ax.plot(df['q'], df['I'])
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     plt.show()
#     return
#
# if __name__ == '__main__':
#     main()