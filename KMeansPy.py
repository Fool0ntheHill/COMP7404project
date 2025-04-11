import numpy as np
from sys import float_info
from Sparsifierpy import SparsifierPy  # 修正拼写错误

class KMeansPy(SparsifierPy):  # 修改为继承 SparsifierPy
    """ 稀疏化 K-Means 聚类。

    参数
    ----------

    n_components : int, 默认: 8
        聚类数量。

    init : {ndarray, 'kmpp', 'random'}, 默认: 'kmpp'
        初始化方法:

        ndarray : 形状 (n_components, P)。初始聚类中心，必须已经转换。

        'kmpp': 从数据中选择初始聚类中心，选择概率与每个数据点到当前初始均值的距离成正比。
        计算成本更高但收敛性更好。如果稀疏器可以访问HDX，则从HDX中抽取，否则从RHDX中抽取。

        'random': 从数据点中均匀随机选择初始聚类中心。如果稀疏器可以访问HDX，
        则从HDX中抽取，否则从RHDX中抽取。

    n_init : int, 默认: 10
        使用新初始化运行k-means的次数。保留最佳结果。

    max_iter : int, 默认: 300
        每次运行的最大迭代次数。

    tol : float, 默认: 1e-4
        收敛的惯性相对容差。

    属性
    ----------

    cluster_centers_ : nd.array, 形状 (n_components, P)
        聚类中心坐标

    labels_ : np.array, 形状 (N,)
        每个点的标签

    inertia_ : float
        样本到其聚类中心的平方距离之和。

    """
   
    def fit(self, X=None, HDX=None, RHDX=None):
        """ 计算k-means聚类并为数据点分配标签。
        至少需要设置一个参数。

        参数
        ----------

        X : nd.array, 形状 (N, P), 可选
            默认为None。密集原始数据。

        HDX : nd.array, 形状 (N, P), 可选
            默认为None。密集转换数据。

        RHDX : nd.array, 形状 (N, Q), 可选
            默认为None。子采样转换数据。
         
        """
        self.fit_sparsifier(X=X, HDX=HDX, RHDX=RHDX)  # 直接调用继承的方法
        best_inertia = float_info.max

        for i in range(self.n_init):
            self._fit_single_trial()
            if self.inertia_ < best_inertia:
                best_inertia = self.inertia_
                cluster_centers_ = self.cluster_centers_
                labels_ = self.labels_
                best_counter = self.counter

            self.cluster_centers_ = cluster_centers_
            self.labels_ = labels_
            self.inertia_ = best_inertia
            self.counter = best_counter
       
        # 设置每个聚类包含的样本数量
        self.n_per_cluster = np.zeros(self.n_components, dtype=int)
        for k in range(self.n_components):
            self.n_per_cluster[k] = sum(self.labels_==k)

    def _initialize_cluster_centers(self, init):
        """ 初始化聚类猜测。"""
        means_init_array = self.means_init_array
        if means_init_array is None:
            # 检查 init 是否为字符串类型
            if isinstance(init, str):
                if init == "kmpp":
                    means, _ = self._pick_K_dense_datapoints_kmpp(self.n_components)
                elif init == "random":
                    means, _ = self._pick_K_dense_datapoints_random(self.n_components)
                else:
                    raise ValueError(f"未知的初始化方法: {init}")
            else:
                # 如果 init 不是字符串，假设它是一个数组
                means = init
        elif means_init_array is not None:
            means = means_init_array[self.means_init_array_counter]
            self.means_init_array_counter += 1
        self.cluster_centers_ = means

    # 核心算法

    def _compute_labels(self):
        """ 计算每个数据点的标签。"""
        d = self.pairwise_distances(Y=self.cluster_centers_)
        labels_ = np.argmin(d, axis=1)  # 使用 argmin 而不是 argsort
        d_squared = d ** 2  # 计算平方欧氏距离
        inertia_ = np.sum(d_squared[np.arange(self.num_samp), labels_])
        return [labels_, inertia_]

    def _compute_cluster_centers(self):
        """ 计算每个聚类的均值。"""
        resp = np.zeros((self.num_samp, self.n_components), dtype=float)
        resp[np.arange(self.num_samp), self.labels_] = 1
        # 移除空簇处理逻辑，与原始kmeans保持一致
        cluster_centers_ = self.weighted_means(resp)
        return cluster_centers_

    def _fit_single_trial(self):
        """ 初始化并运行单次试验。"""
        self._initialize_cluster_centers(self.init)
        self.labels_, self.inertia_ = self._compute_labels()
        current_iter = 0
        inertia_change = 2*self.tol
        while(current_iter < self.max_iter and inertia_change > self.tol):
            self.cluster_centers_ = self._compute_cluster_centers()
            previous_inertia = self.inertia_
            self.labels_, self.inertia_ = self._compute_labels()
            current_iter += 1
            if self.inertia_ > 0:
                inertia_change = np.abs(self.inertia_ - previous_inertia)/self.inertia_
            else:
                inertia_change = 0
        
        # 分配收敛结果
        self.iterations_ = current_iter
        if inertia_change < self.tol:
            self.converged = True
        else:
            self.converged = False

        self.counter = current_iter

    def __init__(self, n_components=8, init='kmpp', tol=1e-4, 
                 n_init=10, n_passes=1, max_iter=300, 
                 means_init_array=None,
                 **kwargs):
        
        # 调用父类初始化方法
        super(KMeansPy, self).__init__(**kwargs)

        self.n_components = n_components
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.n_passes = n_passes
        self.means_init_array = means_init_array
        self.means_init_array_counter = 0