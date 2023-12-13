
from scipy.spatial import distance
import os
import math
import argparse
import pandas as pd
import numpy as np
from curvatures import *
import sys

parser = argparse.ArgumentParser(description='GBDT or SVM predictions')
parser.add_argument('--dataset_name', type=str, default='GSE45719')
parser.add_argument('--kernel_type', type=str, default='exp')
parser.add_argument('--fetures_path', type=str, default='features-CCP-UMAP')
parser.add_argument('--ccp_dimension', type=int, default=300)
parser.add_argument('--kappa', type=int, default=2)
args = parser.parse_args()

dataset_name = args.dataset_name
kappa = args.kappa
fetures_path = args.fetures_path
ccp_dimension = args.ccp_dimension

ccp_fetures_path = f'{fetures_path}/{dataset_name}_features/{dataset_name}_ccp_umap_n{ccp_dimension}.txt'
features = np.loadtxt(ccp_fetures_path)

nrow, ncol = np.shape(features)
Dists = distance.cdist(features, features)
Dists_use = []
for id, dists in enumerate(Dists):
    mask = [i for i in range(nrow) if i != id]
    dists_use = dists[mask]
    Dists_use.append(dists_use)
Dists_use_min = np.min(Dists_use, axis=1)
Dists_use_max = np.max(Dists_use, axis=1)
Dists_use_mean = np.mean(Dists_use, axis=1)

mean_Dists_use_min = np.mean(Dists_use_min)
mean_Dists_use_max = np.mean(Dists_use_max)
mean_Dists_use_mean = np.mean(Dists_use_mean)

max_Dists_use_min = np.max(Dists_use_min)
max_Dists_use_max = np.max(Dists_use_max)
max_Dists_use_mean = np.max(Dists_use_mean)

etas = np.arange(1, math.ceil(max_Dists_use_max)+0.5, 0.5).tolist()

for eta in etas:
    print(eta)
    curvatures_K = []
    curvatures_H = []
    indices = np.array([i for i in range(nrow)])
    for ip, point in enumerate(features):
        mask = indices != ip
        density_cloudpoints = features[mask]
        if args.kernel_type == "exp":
            curvature_K, curvature_H = Curvature_Exp(
                eta, [point], density_cloudpoints, kappa)
        curvatures_K.append(curvature_K)
        curvatures_H.append(curvature_H)

    curvature_type = 'gaussian'
    save_path = f'{fetures_path}/{dataset_name}_features/{dataset_name}-DGGL-{curvature_type}-{args.kernel_type}-eta-{eta:.2f}-kappa-{kappa}.npy'
    np.save(save_path, curvatures_K)

    curvature_type = 'mean'
    save_path = f'{fetures_path}/{dataset_name}_features/{dataset_name}-DGGL-{curvature_type}-{args.kernel_type}-eta-{eta:.2f}-kappa-{kappa}.npy'
    np.save(save_path, curvatures_H)
