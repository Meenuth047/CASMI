"""
Offline ranker experiments on dumped candidate tables (work/feats_*.parquet): k-fold CV by molecule.

    python -m casmi.ranker_lab                    # compare designs
    python -m casmi.ranker_lab --save mlp         # fit the chosen design on everything -> work/ranker.json
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch

from casmi.pipeline import FEATURES, add_group_features, RANKER_FEATURES

WORK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'work')


def offline_mrr(df, score):
    """MRR@25 per (set, scenario): rank of the first correct candidate after de-duplicating on the metric key."""
    df = df.assign(_s=score)
    out = {}
    for (st, sc), part in df.groupby(['set', 'scenario']):
        rr = []
        for _, g in part.groupby('molecule_id'):
            g = g.sort_values('_s', ascending=False, kind='stable').drop_duplicates('mkey').head(25)
            hit = np.flatnonzero(g.label.values)
            rr.append(1.0 / (hit[0] + 1) if len(hit) else 0.0)
        out[(st, sc)] = float(np.mean(rr))
    return out


class Ranker:
    def __init__(self, features, hidden=0, l2=1e-3, scenario_weights=None, seed=0):
        self.features, self.hidden, self.l2 = features, hidden, l2
        self.sw = scenario_weights or {'c1': 1.0, 'c2': 1.0, 'c3': 0.5}
        self.seed = seed

    def _net(self, n_in):
        torch.manual_seed(self.seed)
        if self.hidden:
            return torch.nn.Sequential(torch.nn.Linear(n_in, self.hidden), torch.nn.Tanh(), torch.nn.Linear(self.hidden, 1))
        return torch.nn.Linear(n_in, 1, bias=False)

    def fit(self, df):
        x = df[self.features].values.astype(np.float64)
        self.mean, self.scale = x.mean(0), x.std(0) + 1e-9
        x = torch.tensor((x - self.mean) / self.scale, dtype=torch.float32)
        y = torch.tensor(df.label.values, dtype=torch.float32)
        codes, uniq = pd.factorize((df.molecule_id + '|' + df.scenario).values)
        gid, n_groups = torch.tensor(codes), len(uniq)
        has_pos = torch.zeros(n_groups).index_add_(0, gid, y) > 0
        gw = torch.tensor([self.sw.get(u.split('|')[1], 1.0) for u in uniq], dtype=torch.float32)
        self.net = self._net(x.shape[1])
        params = list(self.net.parameters())
        opt = torch.optim.LBFGS(params, lr=0.5, max_iter=300 if not self.hidden else 200, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            s = self.net(x).squeeze(-1)
            m = torch.full((n_groups,), -1e9).scatter_reduce(0, gid, s, reduce='amax')
            e = torch.exp(s - m[gid])
            denom = torch.zeros(n_groups).index_add_(0, gid, e)
            numer = torch.zeros(n_groups).index_add_(0, gid, e * y)
            nll = -(torch.log(numer[has_pos] + 1e-12) - torch.log(denom[has_pos]))
            loss = (nll * gw[has_pos]).sum() / gw[has_pos].sum() + self.l2 * sum((p ** 2).sum() for p in params)
            loss.backward()
            return loss

        opt.step(closure)
        return self

    def predict(self, df):
        x = (df[self.features].values.astype(np.float64) - self.mean) / self.scale
        with torch.no_grad():
            return self.net(torch.tensor(x, dtype=torch.float32)).squeeze(-1).numpy()

    def to_json(self):
        sd = {k: v.numpy().astype(float).tolist() for k, v in self.net.state_dict().items()}
        return dict(features=self.features, mean=self.mean.tolist(), scale=self.scale.tolist(), hidden=self.hidden, weights=sd)


def cross_validate(df, make_ranker, k=5, seed=0):
    mols = df.molecule_id.unique()
    rng = np.random.default_rng(seed)
    fold = dict(zip(mols, rng.integers(0, k, len(mols))))
    f = df.molecule_id.map(fold).values
    score = np.zeros(len(df))
    for i in range(k):
        r = make_ranker().fit(df[f != i])
        score[f == i] = r.predict(df[f == i])
    return offline_mrr(df, score)


def fmt(res):
    keys = sorted(res)
    return ' | '.join(f'{st[4:]}-{sc} {res[(st, sc)]:.3f}' for st, sc in keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='')
    ap.add_argument('--save', default='')
    args = ap.parse_args()
    df = pd.concat([pd.read_parquet(os.path.join(WORK, f'feats_{n}{args.tag}.parquet')) for n in ('val_np', 'val_rand')], ignore_index=True)
    df = df.loc[:, ~df.columns.duplicated()]
    if 'analog_sim' not in df.columns:
        df = pd.concat([add_group_features(g) for _, g in df.groupby(['molecule_id', 'scenario'], sort=False)], ignore_index=True)
    print(f'{len(df):,} candidates | {df.groupby(["molecule_id", "scenario"]).ngroups} lists | positives present in '
          f'{df.groupby(["molecule_id", "scenario"]).label.max().mean():.3f} of lists')

    fallback = 2.0 * df.lib_score + df.fp_cos + 0.05 * df.ll_rel + 0.3 * df.in_lib + 0.15 * df.in_coconut
    print('ceiling (truth anywhere in list):', fmt(offline_mrr(df, df.label.astype(float))))
    print('fallback hand weights          :', fmt(offline_mrr(df, fallback)))
    print('library score only             :', fmt(offline_mrr(df, df.lib_score + 1e-3 * df.fp_cos)))
    print('fingerprint cosine only        :', fmt(offline_mrr(df, df.fp_cos)))
    print('decoder likelihood only        :', fmt(offline_mrr(df, df.ll)))
    designs = {
        'linear / base features': lambda: Ranker(FEATURES),
        'linear / + library-confidence': lambda: Ranker(RANKER_FEATURES),
        'mlp16  / + library-confidence': lambda: Ranker(RANKER_FEATURES, hidden=16, l2=1e-3),
        'mlp32  / + library-confidence': lambda: Ranker(RANKER_FEATURES, hidden=32, l2=3e-3),
    }
    for name, make in designs.items():
        print(f'{name:31s}:', fmt(cross_validate(df, make)))
    if args.save:
        make = designs[[k for k in designs if k.startswith(args.save)][0]]
        r = make().fit(df)
        with open(os.path.join(WORK, 'ranker.json'), 'w') as fh:
            json.dump(r.to_json(), fh)
        print('saved work/ranker.json:', args.save)


if __name__ == '__main__':
    main()
