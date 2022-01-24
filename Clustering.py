#GMM算法对消费者年度数据集进行聚类分析，确定消费者类别

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


#定义make_ellipses函数，根据GMM算法输出的聚类类别，画出相应的高斯分布区域
def make_ellipses(gmm, ax, k):
    for n in np.arange(k):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.3)
        ax.add_artist(ell)



#start 缺失检查
data = pd.read_excel(r'D:\1machine_study\.idea\聚类\客户年消费数据.xlsx')
print('各字段缺失情况：\n', data.isnull().sum())

# 六种商品年消费额分布图
fig = plt.figure(figsize=(16, 9))
for i, col in enumerate(list(data.columns)[1:]):
    plt.subplot(321+i)
    q95 = np.percentile(data[col], 95)
    sns.distplot(data[data[col] < q95][col])
#plt.show()

features = data[['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen']]

# 剔除极值或异常值
ids = []
for i in list(features.columns):
    q1 = np.percentile(features[i], 25)
    q3 = np.percentile(features[i], 75)
    intervel = 1.6*(q3 - q1)/2
    low = q1 - intervel
    high = q3 + intervel
    ids.extend(list(features[(features[i] <= low) |
                             (features[i] >= high)].index))
ids = list(set(ids))
features = features.drop(ids)

#主成分分析降维

# 计算每一列的平均值
meandata = np.mean(features, axis=0)  
# 均值归一化
features = features - meandata    
# 求协方差矩阵
cov = np.cov(features.transpose())
# 求解特征值和特征向量
eigVals, eigVectors = np.linalg.eig(cov) 
# 选择前两个特征向量
pca_mat = eigVectors[:,:2]
pca_data = np.dot(features , pca_mat)
pca_data = pd.DataFrame(pca_data, columns=['pca1', 'pca2'])

# 两个主成分的散点图
plt.subplot(111)
plt.scatter(pca_data['pca1'], pca_data['pca2'])
plt.xlabel('pca_1')
plt.ylabel('pca_2')
plt.show()

print('前两个主成分包含的信息百分比：{:.2%}'.format(np.sum(eigVals[:2])/np.sum(eigVals)))


score_kmean = []
score_gmm = []
random_state = 87
n_cluster = np.arange(2, 5)
for i, k in zip([0, 2, 4, 6], n_cluster):
    # K-means聚类
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    cluster1 = kmeans.fit_predict(pca_data)
    score_kmean.append(silhouette_score(pca_data, cluster1))

    # gmm聚类
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=random_state)
    cluster2 = gmm.fit(pca_data).predict(pca_data)
    score_gmm.append(silhouette_score(pca_data, cluster2))

    # 聚类效果图
    plt.subplot(421+i)
    plt.scatter(pca_data['pca1'], pca_data['pca2'], c=cluster1)
    if i == 6:
        plt.xlabel('K-means')
    ax=plt.subplot(421+i+1)
    plt.scatter(pca_data['pca1'], pca_data['pca2'], c=cluster2)
    make_ellipses(gmm, ax, k)
    if i == 6:
        plt.xlabel('GMM')
plt.show()

# 聚类类别从2到11，统计两种聚类模型的silhouette_score，分别保存在列表score_kmean 和score_gmm 
score_kmean = []
score_gmm = []
random_state = 87
n_cluster = np.arange(2, 12)
for k in n_cluster:
    # K-means聚类
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    cluster1 = kmeans.fit_predict(pca_data)
    score_kmean.append(silhouette_score(pca_data, cluster1))
    # gmm聚类
    gmm = GaussianMixture(n_components=k, covariance_type='spherical', random_state=random_state)
    cluster2 = gmm.fit(pca_data).predict(pca_data)
    score_gmm.append(silhouette_score(pca_data, cluster2))

# 得分变化对比图
sil_score = pd.DataFrame({'k': np.arange(2, 12),
                          'score_kmean': score_kmean,
                          'score_gmm': score_gmm})

                          
# K-means算法和GMM算法得分对比
plt.figure(figsize=(10, 6))
plt.bar(sil_score['k']-0.15, sil_score['score_kmean'], width=0.3,
        facecolor='blue', label='Kmeans_score')
plt.bar(sil_score['k']+0.15, sil_score['score_gmm'], width=0.3,
        facecolor='green', label='GMM_score')
plt.xticks(np.arange(2, 12))
plt.legend(fontsize=16)
plt.ylabel('silhouette_score', fontsize=16)
plt.xlabel('k')
plt.show()