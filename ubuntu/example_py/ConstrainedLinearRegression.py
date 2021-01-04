#coding: utf-8
from sklearn.linear_model._base import LinearModel
from sklearn.base import RegressorMixin
from sklearn.utils import check_X_y
import numpy as np

class ConstrainedLinearRegression(LinearModel, RegressorMixin):
    # coef_min_value: 设定系数最小值
    # coef_max_value: 设定系数最大值
    # normalize_coef: 归一化系数（注. 系数和为一，不是平方和为一）
    def __init__(self, fit_intercept=True, 
                        normalize=False, 
                        copy_X=True, 
                        coef_min_value=-np.inf, 
                        coef_max_value=np.inf, 
                        normalize_coef=True, 
                        tol=1e-15):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.coef_min_value = coef_min_value
        self.coef_max_value = coef_max_value
        self.normalize_coef = normalize_coef
        self.tol = tol
        
    # 可通过数组min_coef和max_coef，对coef中的每个值进行范围限定
    # 由于现在对系数的约束方式，是简单的截断（np.clip)，而不是加正则项，可能造成迭代优化无法收敛，因此限定最大迭代次数max_iterator_step。
    def fit(self, X, y, min_coef=None, max_coef=None, max_iterator_step=100):
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'], y_numeric=True, multi_output=False)
        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=self.normalize, copy=self.copy_X)

        # 设置系数的统一的最小值
        self.min_coef_ = np.repeat(self.coef_min_value, X.shape[1])
        # 设置系数的统一的最大值
        self.max_coef_ = np.repeat(self.coef_max_value, X.shape[1])

        # 使用min_coef/max_coef分别给各个系数单独设置了最小/大值，为None则不设置
        if min_coef is not None:
            self.min_coef_ = min_coef
        if max_coef is not None:
            self.max_coef_ = max_coef
        
        # 定义系数归一化函数
        def normalize_coef_func(coef):
            coef_sum = coef.sum()
            # 避免系数和太小（<1e-7）时可能产生数值溢出
            if np.abs(coef_sum) <= 1e-7:
                print("[WARNING]: sum of coef is too small: %f" % coef_sum)
                coef += 1 / X.shape[1]
            else:
                coef = coef / coef_sum
            return coef

        # 限制迭代次数
        step = 0
        beta = np.zeros(X.shape[1]).astype(float)
        prev_beta = beta + 1
        hessian = np.dot(X.transpose(), X)
        # 迭代优化
        while not (np.abs(prev_beta - beta)<self.tol).all():
            prev_beta = beta.copy()
            for i in range(len(beta)):
                # 用牛顿法解线性方程组 AX=Y（这个公式没仔细看）
                grad = np.dot(np.dot(X,beta) - y, X)
                beta[i] = beta[i] - grad[i] / hessian[i,i]
                # 限制系数的范围
                beta[i] = np.maximum(self.min_coef_[i], beta[i])
                beta[i] = np.minimum(self.max_coef_[i], beta[i])
            # 归一化
            if self.normalize_coef:
                beta = normalize_coef_func(beta) 
            # 限制迭代次数       
            step += 1
            print("iterator step: %d" % step)
            if step >= max_iterator_step:
                break

        self.coef_ = beta
        self._set_intercept(X_offset, y_offset, X_scale)
        return self    


from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

# 载入boston数据集
X, y = load_boston(return_X_y=True)
# 创建模型
model = ConstrainedLinearRegression(coef_min_value=0, coef_max_value=1, normalize_coef=True)
# 拟合模型
model.fit(X, y)

print(model.intercept_)
print(model.coef_)
