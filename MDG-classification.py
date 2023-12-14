from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.spatial import distance
import os
import math
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import scipy.stats as stats
from sklearn.svm import SVC
from statistics import mean
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import argparse
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--dataset_name', type=str, default='GSE45719')
parser.add_argument('--fp', type=str, default='mdg')
parser.add_argument('--kernel_type', type=str, default='exp')
parser.add_argument('--icycle', type=int, default=0)
parser.add_argument('--dx', type=float, default=0.5)
parser.add_argument('--kappas', type=str, default='5-10')
parser.add_argument('--mlmethod', type=str, default='gbdt')


args = parser.parse_args()
dataset_name = args.dataset_name
icycle = args.icycle
kappas = args.kappas
mlmethod = args.mlmethod
dx = args.dx


def read_dataset(feature_file, label_file):
    ''' Read data set in *.csv to data frame in Pandas'''
    df_X = np.load(feature_file)
    df_y = pd.read_csv(label_file, header=None, index_col=False)
    X = df_X
    y = df_y.values.ravel()
    return X, y


def adjust_train_test(y_train, y_test, train_index, test_index):
    '''
        Adjust training and testing data to ensure there are at least 5 of each in the train and 3 of each in test data
        5 * the average number of samples of each class is sampled
    '''
    np.random.seed(1)
    unique_labels_temp = np.intersect1d(y_train, y_test)
    unique_labels_temp.sort()
    unique_labels = []
    counter = []
    new_test_index = []
    for l in unique_labels_temp:
        l_train = np.where(l == y_train)[0]
        l_test = np.where(l == y_test)[0]
        if l_train.shape[0] > 5 and l_test.shape[0] > 3:
            unique_labels.append(l)
            # get the index of the y_test that satisfies the condition
            new_test_index.append(l_test)
            counter.append(l_train.shape[0])
    new_test_index = np.concatenate((new_test_index))
    new_test_index.sort()
    new_y_test = y_test[new_test_index]  # new y_test
    new_test_index = test_index[new_test_index]

    new_train_index = []
    avgCount = int(np.ceil(np.mean(counter)))  # sample 5x avgCount
    for l in unique_labels:
        l_train = np.where(l == y_train)[0]
        index = np.random.choice(l_train, 5*avgCount)
        new_train_index.append(index)
    new_train_index = np.concatenate(new_train_index)
    new_train_index.sort()
    new_y_train = y_train[new_train_index]
    new_train_index = train_index[new_train_index]
    return new_y_train, new_y_test, new_train_index, new_test_index


def computeGBDT(X_train, X_test, y_train, y_test):
    i = 2000
    j = 7
    k = 5
    m = 8
    lr = 0.002
    ml_method = "GradientBoostingClassifier"
    clf = globals()["%s" % ml_method](n_estimators=i, max_depth=j,
                                      min_samples_split=k, learning_rate=lr, subsample=0.1*m, max_features='sqrt')
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    predicted_prob = clf.predict_proba(X_test)

    return y_pred, predicted_prob


def balanced_accuarcy(y_true, y_pred):
    ba = balanced_accuracy_score(y_true, y_pred)
    return ba


def compute5foldClassification(X_PSPH, y_true, mlmethod, icycle, ikf):
    kf = KFold(n_splits=5, shuffle=True, random_state=icycle)

    # for i, (train_index, test_index) in enumerate(kf.split(X_PSPH)):
    train_index, test_index = list(kf.split(X_train))[ikf]
    y_train = y_true[train_index]
    y_test = y_true[test_index]
    y_train, y_test, train_index, test_index = adjust_train_test(
        y_train, y_test, train_index, test_index)
    X_PSPH_train = X_PSPH[train_index]
    X_PSPH_test = X_PSPH[test_index]
    if mlmethod == 'gbdt':
        y_pred, y_prob = computeGBDT(
            X_PSPH_train, X_PSPH_test, y_train, y_test)
    ba = balanced_accuarcy(y_true[test_index], y_pred)
    acc = accuracy_score(y_true[test_index], y_pred)
    prec = precision_score(y_true[test_index], y_pred, average='macro')
    recall = recall_score(y_true[test_index], y_pred, average='macro')
    f1 = f1_score(y_true[test_index], y_pred, average='macro')
    roc_auc = roc_auc_score(
        y_true[test_index], y_prob, average='macro', multi_class='ovr')

    return ba, acc, prec, recall, f1, roc_auc, y_pred, y_prob


dataset_path = 'scRNA-seq-data/%s' % (dataset_name)
fetures_path = f'features-CCP-UMAP/{dataset_name}_features/{dataset_name}_ccp_umap_n300.txt'
features = np.loadtxt(fetures_path)

nrow, ncol = np.shape(features)
Dists = distance.cdist(features, features)
Dists_use = []
for id, dists in enumerate(Dists):
    mask = [i for i in range(nrow) if i != id]
    dists_use = dists[mask]
    Dists_use.append(dists_use)

Dists_use_max = np.max(Dists_use, axis=1)
max_Dists_use_max = np.max(Dists_use_max)
etas = np.arange(1, math.ceil(max_Dists_use_max)+dx, dx).tolist()

kappas = kappas.split('-')
kappas = np.array(kappas).astype(int)

X_train = []
for kappa in kappas:
    for eta in etas:
        curvature_type = 'gaussian'
        save_path = f'{fetures_path}/{dataset_name}_features/{dataset_name}-DGGL-{curvature_type}-{args.kernel_type}-eta-{eta:.2f}-kappa-{kappa}.npy'
        X_train.append(np.load(save_path))

        curvature_type = 'mean'
        save_path = f'{fetures_path}/{dataset_name}_features/{dataset_name}-DGGL-{curvature_type}-{args.kernel_type}-eta-{eta:.2f}-kappa-{kappa}.npy'
        X_train.append(np.load(save_path))

X_train = np.concatenate(X_train, axis=1)

y_train = pd.read_csv(f'{dataset_path}/{dataset_name}_labels.csv',
                      header=0, index_col=False).loc['Label'].values.ravel()

for ikf in range(5):
    ba, acc, prec, recall, f1, roc_auc, y_pred, y_prob = compute5foldClassification(
        X_train, y_train, mlmethod, icycle, ikf)

    results_path = 'features-CCP-UMAP/%s_features/results' % (
        dataset_name)
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    filename = f'{results_path}/prediction-c{icycle}-kf5-ikf{ikf}-{args.fp}-{args.kernel_type}-kappas-{args.kappas}-{mlmethod}-dx{dx:.1f}.csv'
    fw = open(filename, 'w')
    print("ba=%.3f acc=%.3f prec=%.3f recall=%.3f f1=%.3f auc=%.3f" %
          (ba, acc, prec, recall, f1, roc_auc), file=fw)
    fw.close()
