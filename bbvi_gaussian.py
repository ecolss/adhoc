#!/usr/bin/env python

import numpy as np
from scipy import stats
import torch as th
from util import AdaGrad, RTPlot
import pylab as pl
import time
from itertools import chain

float_t = th.float32

# util
############################################################
def TF(x, requires_grad=0):
    return th.tensor(x, dtype=float_t, requires_grad=bool(requires_grad))

def make_posdef(k):
    v = np.random.rand(k,k)
    v = (v + v.T)/2 + np.diag(np.ones(k))
    return v

def make_data(size=100, viz=False):
    A = stats.multivariate_normal.rvs(mean=[0,0], cov=[[16,5.],[5.,4]], size=size)
    if viz:
        pl.scatter(A[:,0],A[:,1])
        pl.show()
    return A

# fit
############################################################
def bbvi_fit(X, n_iter=1, n_sample=1, lrate=0.1, log_every=100, mccv=True):
    nx,dim = X.shape

    px_c = th.tensor([[1,0],[0,1]], dtype=float_t)
    pz_m = th.zeros(dim, dtype=float_t)  # + X.mean(0)
    pz_c = th.eye(dim, dtype=float_t)*10
    qz_m = th.zeros(dim, dtype=float_t, requires_grad=True)
    qz_tril = th.eye(dim, dtype=float_t, requires_grad=True)*10
    tril_mask = th.ones(dim, dim, dtype=float_t).tril_()

    log_px = th.empty(n_sample, dtype=float_t)
    log_pz = th.empty(n_sample, dtype=float_t)
    log_qz = th.empty(n_sample, dtype=float_t)
    G = th.empty(n_sample, dim, dtype=float_t)
    G_tril = th.empty(n_sample, dim, dim, dtype=float_t)

    upd = AdaGrad(['qz_m', 'qz_tril'], init=1.0, max_norm=1)
    elbo = th.tensor(0, dtype=float_t)
    rtp = RTPlot(plots=[3,3], pause=0.01)
    for i_iter in range(n_iter):
        t = time.time()

        tril_ = qz_tril*tril_mask
        tril_.view(-1)[::dim+1].exp_()
        qz_c = tril_@tril_.t()

        for i in range(n_sample):
            m = normal_sample(qz_m, qz_c)
            log_px[i] = normal_lprob(X-m, px_c)/nx
            log_pz[i] = normal_lprob(m-pz_m, pz_c)
            lqz = normal_lprob(m-qz_m, qz_c)
            G[i] = th.autograd.grad(lqz, qz_m, retain_graph=True)[0]
            G_tril[i] = th.autograd.grad(lqz, qz_tril, retain_graph=True)[0]
            log_qz[i] = lqz
        elbo += (log_px+log_pz-log_qz).mean()
        F = th.einsum('i,ij->ij', (log_px+log_pz-log_qz, G))
        F_tril = th.einsum('i,ijk->ijk', (log_px+log_pz-log_qz, G_tril))
        if mccv:
            def mccv_fn_(F, H):
                F_ = F-F.mean(dim=0)
                H_ = H-H.mean(dim=0)
                var_H = th.mean(H_**2, dim=0) + 1e-5
                cov_FH = th.mean(F_*H_, dim=0)
                H_ = th.mean(F-H*cov_FH/var_H, dim=0)
                return cov_FH/var_H
            G_ = th.mean(F-G*mccv_fn_(F, G), dim=0)
            G_tril_ = th.mean(F_tril-G_tril*mccv_fn_(F_tril, G_tril), dim=0)
        else:
            G_ = F.mean(dim=0)
            G_tril_ = F_tril.mean(dim=0)
        upd.update([('qz_m', qz_m.data, G_.data), ('qz_tril', qz_tril.data, G_tril_.data)], lrate=lrate)
        if (i_iter+1)%log_every == 0:
            tril_ = qz_tril*tril_mask
            tril_.view(-1)[::dim+1].exp_()
            qz_c = tril_@tril_.t()

            t = time.time() - t
            print('i_iter = {}, t = {}, elbo = {}, qz_m = {}, qz_c = {}'.format(
                i_iter, t, elbo/log_every, qz_m, qz_c))
            rtp.update([(i_iter,elbo,1), (i_iter,qz_m[0].item(),1),
                (i_iter,qz_m[1].item(),1), (i_iter,G_.norm(),1),
                (i_iter,qz_c[0,0].item(),1), (i_iter,qz_c[0,1].item(),1),
                (i_iter,qz_c[1,1].item(),1), (i_iter,t,1),
                ])
            elbo = 0

############################################################
class Var:
    def __init__(self, x, fixed=True):
        self.data = x.requires_grad_(not fixed)

    def make(self):
        return self.data

    def params(self):
        return [self.data]

class PosdefVar:
    def __init__(self, x, fixed=True):
        self.dim = x.shape[0]
        self.mask = th.ones(self.dim, self.dim, dtype=float_t).tril_()
        self.tr = x.cholesky().detach().requires_grad_(not fixed)
        self.data = x.detach()

    def make(self):
        if self.tr.requires_grad:
            tr = self.tr * self.mask
            tr.view(-1)[::self.dim+1].exp_()
            self.data = tr @ tr.t()
        return self.data

    def params(self):
        return [self.tr]

class ExpVar:
    def __init__(self, x, b=0, fixed=True):
        self.x = th.tensor(np.log(x-b), dtype=float_t, requires_grad=not fixed)
        self.b = b
        self.data = TF(x)

    def make(self):
        if self.x.requires_grad:
            self.data = self.x.exp() + self.b
        return self.data

    def params(self):
        return [self.x]

class GaussianVar:
    def __init__(self, m, c, fixed=True):
        self.mean = Var(m, fixed)
        self.cov = PosdefVar(c, fixed)

    def make(self):
        self.mean.make()
        self.cov.make()
        return self.mean.data, self.cov.data

    def sample(self, size=1, use_cache=True):
        if use_cache:
            m = self.mean.data
            c = self.cov.data
        else:
            m,c = self.make()
        return TF(stats.multivariate_normal.rvs(m.data.numpy(), c.data.numpy(), size=size))
    
    def lprob(self, x, use_cache=True):
        if use_cache:
            m = self.mean.data
            c = self.cov.data
        else:
            m,c = self.make()
        return self._lprob(x, m, c)

    @classmethod
    def _lprob(cls, x, m, c):
        if x.ndimension() == 1:
            x = x.unsqueeze(0)
        x = x - m
        dim = c.shape[0]
        a = -np.log(2*np.pi)*dim/2 - th.logdet(c)/2
        b = th.einsum('ij,jk,ik->i', (x,c.inverse(),x))/2
        return (a-b).sum()

    def params(self):
        return [self.mean, self.cov]

class WishartVar:
    def __init__(self, df, sm, fixed=True):
        self.dim = sm.shape[0]
        self.df = ExpVar(df, self.dim, fixed)
        self.sm = PosdefVar(sm, fixed)

    def make(self):
        self.df.make()
        self.sm.make()
        return self.df.data, self.sm.data

    def sample(self, size=1, use_cache=True):
        if use_cache:
            df = self.df.data
            sm = self.sm.data
        else:
            df, sm = self.make()
        return TF(stats.wishart.rvs(df.item(), sm.data.numpy(), size=size))

    def lprob(self, x, use_cache=True):
        if use_cache:
            df = self.df.data
            sm = self.sm.data
        else:
            df, sm = self.make()
        return self._lprob(x, self.dim, df, sm)

    @classmethod
    def _lprob(cls, x, dim, df, sm):
        a = th.logdet(x)*(df-dim-1)/2 - th.trace(sm.inverse()@x)/2
        b = np.log(2)*df*dim/2 + th.logdet(sm)*df/2 + th.mvlgamma(df/2, dim)
        return a - b

    def params(self):
        return self.df.params() + self.sm.params()

def mccv_fn(F, H):
    F_ = F-F.mean(dim=0)
    H_ = H-H.mean(dim=0)
    var_H = th.mean(H_**2, dim=0) + 1e-7
    cov_FH = th.mean(F_*H_, dim=0)
    return cov_FH/var_H

def bbvi_fit(X, P, Q, lik_fn, n_sample, n_iter=1000, lrate=0.5, log_every=100, calls=[]):
    n,dim = X.shape

    P = [WishartVar(dim+1, th.eye(dim, dtype=float_t))]
    Q = [WishartVar(dim+1, th.eye(dim, dtype=float_t), fixed=False)]
    params_ = list(chain(*[_.params() for _ in Q]))
    params = {
        'names': ['param_{}'.format(_) for _ in range(len(params_))],
        'vals': params_,
        'grads': [None]*len(params_),
        'size': len(params_)
        }

    lpx = th.empty(n_sample, dtype=float_t)
    lpz = th.empty(n_sample, dtype=float_t)
    lqz = th.empty(n_sample, dtype=float_t)
    G = [th.empty((n_sample,)+_.shape) for _ in params['vals']]

    upd = AdaGrad(params['names'], init=1, max_norm=1, decay_step=20, decay_rate=0.90)
    elbo = 0
    for i_iter in range(n_iter):
        samples = [_.sample(size=n_sample, use_cache=False) for _ in Q]
        for i_sample in range(n_sample):
            one = [_[i_sample] for _ in samples]
            lpx[i_sample] = lik_fn(X, *one)
            lpz[i_sample] = 0
            lqz_ = th.tensor(0., dtype=float_t)
            for i in range(len(P)):
                lpz[i_sample] += P[i].lprob(one[i])
                lqz_ += Q[i].lprob(one[i])
            lqz[i_sample] = lqz_
            g = th.autograd.grad(lqz_, params['vals'], retain_graph=True)  # if parallel, need to retain, but when destroy graph?
            for i in range(params['size']):
                G[i][i_sample] = g[i]

        E = lpx + lpz - lqz
        ind_map = {0:'i', 1:'ij', 2:'ijk'}
        for i in range(params['size']):
            ind = ind_map[params['vals'][i].dim()]
            F = th.einsum('i,{}->{}'.format(ind,ind), (E, G[i]))
            alpha = mccv_fn(F, G[i])
            params['grads'][i] = th.mean(F-alpha*G[i], dim=0)

        elbo += E.mean()
        upd.update(list(zip(params['names'], params['vals'], params['grads'])), lrate=lrate)

        if (i_iter+1)%log_every == 0:
            print('i_iter = {}, elbo = {}'.format(i_iter, elbo/log_every))
            elbo = 0
            for fn in calls:
                fn(X, P, Q, i_iter)

# test
############################################################
def fit_gauss_covariance():
    dim = 2
    X = TF(make_data(500))
    P = [WishartVar(dim+1, th.eye(dim, dtype=float_t))]
    Q = [WishartVar(dim+1, th.eye(dim, dtype=float_t), fixed=False)]
    m = th.zeros(dim, dtype=float_t)
    lik_fn = lambda x,s: GaussianVar._lprob(x, m, s.inverse())

    cov = th.inverse(Q[0].df.data*Q[0].sm.data)
    pl.ion()
    def plot(X, cov, i_iter, save='pics'):
        pl.clf()
        pl.scatter(X[:,0], X[:,1], c='b', marker='+', alpha=0.3)
        x = np.linspace(X[:,0].min(), X[:,0].max(), 100)
        y = np.linspace(X[:,1].min(), X[:,1].max(), 100)
        x_ = np.vstack([_.reshape(1,-1) for _ in np.meshgrid(x, y)]).T
        p = np.exp(stats.multivariate_normal.logpdf(x_, m.data, cov.data)).reshape(x.size,-1)
        cs = pl.contour(x, y, p, levels=[0.001,0.01,0.1])
        pl.clabel(cs, inline=1, fontsize=10)
        pl.pause(0.001)
    plot(X, cov, 0)
    pl.show()

    def compute_diff(X, P, Q, i_iter):
        df_ = P[0].df.data + X.shape[0]
        cov_ = (P[0].sm.data.inverse() + X.t()@X).inverse()
        Q[0].make()
        cov = th.inverse(Q[0].df.data*Q[0].sm.data)
        print('df = {}, E_cov = {}, E_cov_ = {}'.format(
            Q[0].df.data, cov, th.inverse(df_*cov_)))
        plot(X, cov, i_iter)


    bbvi_fit(X, P, Q, lik_fn, 5, n_iter=500, lrate=0.5, log_every=5, calls=[compute_diff])
    pl.waitforbuttonpress()

if __name__ == '__main__':
    np.random.seed(123)
    fit_gauss_covariance()
