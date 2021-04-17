#!/usr/bin/env python3

import numpy as np
from scipy import stats
import pylab as pl
from PIL import Image
from sklearn.cluster import KMeans

pl.rcParams['figure.figsize'] = (8,6)


def make_cov2d():
    temp = np.random.rand(2,2) - np.random.rand()
    temp[0,0] *= 5
    temp[1,1] *= 5

    temp[0,1] *= 0.5
    temp[1,0] *= 0.5
    cov = temp @ temp.T
    return cov

def make_gaussian2d(size, k_clusters=1):
    data = []
    for i in range(k_clusters):
        mean = np.random.rand(2) * 50
        cov = make_cov2d()
        data.append(np.random.multivariate_normal(mean, cov, size=size))
        print(f"cluster {i}: \nmean = {mean}, \ncov = {cov}")
    print("generated data params above ================")
    data = np.concatenate(data, axis=0)
    return data

def gmm_em_train(data, k_clusters, n_iters, eps=1e-5):
    size = data.shape[0]

    # prior of clusters
    priors = np.random.rand(k_clusters)
    priors /= priors.sum()

    # gaussian params of clusters
    # note that, if mus and covs are all the same, then the training will fail,
    # since mus and covs will have the same updates, hence not meaningful.
    mus = data[np.random.randint(0, size, k_clusters)]
    covs = np.repeat(np.expand_dims(np.cov(data, rowvar=False), 0), k_clusters, axis=0)
    print(f"init: \nmus = {mus}, \ncovs = {covs}")

    # latent var per each data point
    qz = np.empty((size, k_clusters))

    prev_likelihood = 0
    for i_iter in range(n_iters):
        # E-step
        for k in range(k_clusters):
            qz[:,k] = stats.multivariate_normal.pdf(data, mus[k], cov=covs[k]) * priors[k]
        qz /= np.sum(qz, axis=1, keepdims=True)

        # M-step
        qz_sum = qz.sum(axis=0)
        priors = np.sum(qz, axis=0) / size
        mus = np.dot(qz.T, data) / qz_sum.reshape(-1,1)
        for k in range(k_clusters):
            temp = data - mus[k]
            covs[k] = np.dot(qz[:,k] * temp.T, temp) / np.sum(qz[:,k])

        # likelihood of dataset
        joint = np.zeros(size)
        for k in range(k_clusters):
            joint += stats.multivariate_normal.pdf(data, mus[k], covs[k]) * priors[k]
        likelihood = np.sum(np.log(joint))
        change = abs(likelihood - prev_likelihood)
        prev_likelihood = likelihood
        print(f"change = {change}, likelihood = {likelihood}")
        if change < eps:
            break

        def plot_contour2d(i):
            x = np.linspace(data.min(), data.max(), 100)
            y = np.linspace(data.min(), data.max(), 100)
            x,y = np.meshgrid(x, y)

            for k in range(k_clusters):
                p = stats.multivariate_normal.pdf(np.vstack((x.reshape(-1),y.reshape(-1))).T, mus[k], covs[k])
                cs = pl.contour(x, y, p.reshape(x.shape), levels=1)
                cs.collections[0].remove()
            pl.scatter(data[:,0], data[:,1], alpha=0.2, marker="+", color="red")
            pl.savefig(f"/tmp/{i}.png")
            pl.clf()

        plot_contour2d(i_iter)

    images = [Image.open(f"/tmp/{i}.png") for i in range(i_iter)]
    images[0].save(fp="/tmp/gmm_em.gif", format="GIF", append_images=images[1:],
        save_all=True, duration=200, loop=0)
    print(f"final results: \nmus = {mus}, \ncovs = {covs}")


############################################################
# test
def test__data():
    data = make_gaussian2d(100, 5)
    pl.scatter(data[:,0], data[:,1])
    pl.show()

def test__train():
    #np.random.seed(123)
    data = make_gaussian2d(500, 7)
    gmm_em_train(data, 5, 1500, eps=1e-7)

    km = KMeans(20)
    km.fit(data)
    labels = km.predict(data)
    pl.scatter(data[:,0], data[:,1], c=labels)
    pl.show()



if __name__ == "__main__":
    test__train()
