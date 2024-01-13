# !/usr/bin/env python3
# ============================================================================== #
# JODIE Dataset distribution visualize
# Powered by xiaolis@outlook.com 202312
# ============================================================================== #
import ssl, fsspec, numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv
from urllib.request import urlopen
from datetime import timedelta
from pathlib import Path
from PIL import Image

def get_jodie_data(dataset, path='/tmp/'):
    assert (dataset in ['wikipedia', 'reddit', 'mooc', 'lastfm'])
    pt = path + dataset + '.csv'
    if Path(pt).exists(): return pt
    print(f'{dataset} is not in {path} yet, downloading now ...') 
    url = f'http://snap.stanford.edu/jodie/{dataset}.csv'
    data = urlopen(url, context= ssl._create_unverified_context())
    with fsspec.open(pt, 'wb') as f:
        while True:
            chunk = data.read(10485760)
            if not chunk: break
            f.write(chunk)
    print(f'{dataset} has been downloaded in {pt}.'); return pt

def reindex(df, bipartite=True):
    rs = df.copy()
    if bipartite:
        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))
        rs.i = df.i + df.u.max() + 1
    rs.u += 1; rs.i += 1; rs.idx += 1; return rs

def get_graph_data(dataset, bipartite=True, cache_path='./data/', raw_path='/tmp/'):
    dataset = dataset.lower()
    pdf = cache_path + dataset + '.csv'
    pft = cache_path + dataset + '.npy'
    pnt = cache_path + dataset + '_node.npy'
    if Path(pdf).exists() and Path(pft).exists() and Path(pnt).exists():
        print(f'Load processed {dataset} data from {cache_path}.')
        return read_csv(pdf), np.load(pft), np.load(pnt)
    Path(cache_path).mkdir(parents=True, exist_ok=True)
    raw = get_jodie_data(dataset=dataset, path=raw_path)
    uss, its, tms, lbs, eis, fts = [], [], [], [], [], []
    with open(raw) as f:
        _ = next(f) #
        for i, line in enumerate(f):
            e = line.strip().split(','); eis.append(i)
            uss.append(int(e[0])); its.append(int(e[1])) 
            tms.append(float(e[2])); lbs.append(float(e[3]))  
            fts.append(np.array([float(x) for x in e[4:]])) 
    DF = reindex(DataFrame({'u':uss, 'i':its, 'ts':tms, 'label':lbs, 'idx':eis}), bipartite)
    FT = np.vstack([np.zeros(np.array(fts).shape[1])[np.newaxis,:], fts])
    mx = max(DF.u.max(), DF.i.max())
    RF = np.zeros((mx+1, 172))
    DF.to_csv(pdf); np.save(pft, FT); np.save(pnt, RF)
    print(f'Cached files in {cache_path} and loaded all.')
    return DF, FT, RF

# ============================================================================== #
def seconds_to_date(seconds):
    days, secs = divmod(seconds, 86400); hours, secs = divmod(secs, 3600); mins, secs = divmod(secs, 60)
    return int(days)

def df_timestamps_to_days(df):
    t = df.iloc[:, 2]
    df.iloc[:, 2] = df.iloc[:, 2].apply(lambda col: col if isinstance(col, float) else col.apply(seconds_to_date))
    df.iloc[:, 3:] = df.iloc[:, 3:].groupby([t]).sum().reset_index()
    return df

def load_raw_data(dataset, path='/tmp/'):
    assert (dataset in ['wikipedia', 'reddit', 'mooc', 'lastfm'])
    pt = path + dataset + '.csv'
    if not Path(pt).exists(): raise ValueError(f'{pt} not exists')
    return read_csv(pt, skiprows=1, header=None)

def time_range(df, name):
    top = int(df.iloc[:,2].max())
    days, secs = divmod(top, 86400); hours, secs = divmod(secs, 3600); mins, secs = divmod(secs, 60)
    print(name, timedelta(days=int(days), hours=int(hours), minutes=int(mins), seconds=int(secs)))

def plot_features(axs, df, name):
    t = df.iloc[:,2].values.astype(int)
    y = lambda i: df.iloc[:,i].tolist()
    [axs.plot(t,y(f),label=f, linewidth=.5) for f in df.columns[3:]]
    axs.set_title(name); axs.set_xlabel('Timestamps (cardinal)'); axs.set_ylabel('Features(values)')

def plot_std(axs, df, name):
    for column in df.columns[3:]: axs.bar(column-3, df[column].std(), color='blue', alpha=0.7, align='center')
    axs.set_title(f'{name} Standard Deviation'); axs.set_xlabel('Feature Dimensions (index)'); axs.set_ylabel('Standard Deviation')

def plot_all_dataset(pool):
    _, axs = plt.subplots(2, 2, figsize=(9, 9))
    df = [load_raw_data(d) for d in pool]
    plot_std(axs[0][0], df[0], pool[0])
    plot_std(axs[0][1], df[1] ,pool[1])
    plot_std(axs[1][0], df[2] ,pool[2])
    plot_std(axs[1][1], df[3] ,pool[3])
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.savefig('dist_std.pdf', format='pdf')
    plt.clf()
    _, axs = plt.subplots(2, 2, figsize=(9, 9))
    plot_features(axs[0][0], df[0], pool[0])
    plot_features(axs[0][1], df[1] ,pool[1])
    plot_features(axs[1][0], df[2] ,pool[2])
    plot_features(axs[1][1], df[3] ,pool[3])
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.savefig('dist_feats.pdf', format='pdf')

# ============================================================================== #
if __name__ == '__main__':
    # [get_graph_data(x) for x in ['wikipedia', 'reddit', 'mooc', 'lastfm']]
    plot_all_dataset(['wikipedia', 'reddit', 'mooc', 'lastfm'])

# ============================================================================== #
