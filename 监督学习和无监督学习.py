import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 1. 监督学习可视化 - 分类问题
# 生成带标签的分类数据
X_supervised, y_supervised = make_classification(
    n_samples=100, n_features=2, n_informative=2,
    n_redundant=0, n_clusters_per_class=1, random_state=42
)

# 训练一个简单的分类模型
model = LogisticRegression()
model.fit(X_supervised, y_supervised)

# 创建网格以绘制决策边界
x_min, x_max = X_supervised[:, 0].min() - 1, X_supervised[:, 0].max() + 1
y_min, y_max = X_supervised[:, 1].min() - 1, X_supervised[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 预测网格点的类别
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制监督学习结果
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)
plt.scatter(X_supervised[:, 0], X_supervised[:, 1], c=y_supervised, s=50, cmap=plt.cm.coolwarm)
plt.title('监督学习 - 分类问题')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.text(0.05, 0.95, '• 数据带有标签\n• 学习输入到输出的映射\n• 有明确的预测目标',
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. 无监督学习可视化 - 聚类问题
# 生成无标签的聚类数据
X_unsupervised, _ = make_blobs(n_samples=100, n_features=2, centers=3, random_state=42)

# 使用K-means进行聚类
kmeans = KMeans(n_clusters=3, random_state=42)
y_unsupervised = kmeans.fit_predict(X_unsupervised)

# 绘制无监督学习结果
plt.subplot(122)
plt.scatter(X_unsupervised[:, 0], X_unsupervised[:, 1], c=y_unsupervised, s=50, cmap=plt.cm.viridis)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='聚类中心')
plt.title('无监督学习 - 聚类问题')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.text(0.05, 0.95, '• 数据没有标签\n• 发现数据中的隐藏结构\n• 没有明确的预测目标',
         transform=plt.gca().transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.legend()

plt.tight_layout()
plt.show()
