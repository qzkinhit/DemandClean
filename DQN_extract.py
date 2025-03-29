import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score
import matplotlib.pyplot as plt
import gym
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
import json
from gym import spaces
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams["axes.unicode_minus"] = False


# =============================================================================
# 1. 数据生成与准备
# =============================================================================
def generate_clean_data(task_type='classification', n_samples=1000, n_features=2):
    """
    生成干净的数据集。
    参数:
      - task_type: 'classification'（分类）或 'regression'（回归）
      - n_samples: 样本数量
      - n_features: 特征数量
    返回:
      - DataFrame格式数据集，包含各特征和 'target' 目标列。
    """
    if task_type == 'classification':
        print("生成分类任务数据集...")
        # X, y = make_classification(n_samples=n_samples,
        #                            n_features=n_features,
        #                            n_informative=2,
        #                            n_redundant=n_features - 2,
        #                            n_clusters_per_class=1,  # 每个类别只有一个聚类中心
        #                            n_classes=2,  # 生成2个类别
        #                            flip_y=0,  # 不引入标签噪声
        #                            class_sep=0.4,  # 增大类别间的距离
        #                            shuffle=False,
        #                            random_state=42)
        X, y = make_classification(n_samples=n_samples,
                                   n_features=n_features,
                                   n_informative=2,
                                   n_redundant=n_features - 2,
                                   n_clusters_per_class=1,  # 每个类别只有一个聚类中心
                                   n_classes=2,  # 生成2个类别
                                   flip_y=0,  # 不引入标签噪声
                                   class_sep=0.4,  # 增大类别间的距离
                                   shuffle=False)
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
    else:
        print("生成回归任务数据集...")
        X, y = make_regression(n_samples=n_samples,
                               n_features=n_features,
                               n_informative=n_features,
                               noise=0,  # 添加一定噪声
                               random_state=42)
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y

    return df


def visualize_data(df, task_type='classification', name='test',return_fig=True):
    if task_type == 'classification':
        # print("分类任务数据预览：")
        # print(df.head())

        # 绘制每个特征的直方图
        # df.hist(bins=30, figsize=(12, 8))
        # plt.suptitle("分类任务各特征分布直方图")
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # plt.show()

        # 绘制特征散点图（以 feature_0 和 feature_1 为例），颜色标记目标类别
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(df['feature_0'], df['feature_1'],
                              c=df['target'], cmap='viridis', alpha=0.7)
        plt.xlabel("feature_0")
        plt.ylabel("feature_1")
        plt.title(task_type+"_feature_0 vs feature_1_" + name)
        plt.colorbar(scatter, label='target')
        plt.tight_layout()
        plt.savefig(task_type+"_feature_0 vs feature_1_" + name + '.png')
        # plt.show()
        if return_fig:
            return plt.gcf()
    else:
        print("回归任务数据预览：")
        print(df.head())

        # 绘制回归任务各特征的直方图
        # df.hist(bins=30, figsize=(12, 8))
        # plt.suptitle("回归任务各特征分布直方图")
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # plt.show()

        # 绘制散点图（以 feature_0 和 feature_1 为例），颜色表示 target 值
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(df['feature_0'], df['feature_1'],
                              c=df['target'], cmap='viridis', alpha=0.7)
        plt.xlabel("feature_0")
        plt.ylabel("feature_1")
        plt.title(task_type+"_feature_0 vs feature_1_" + name)
        plt.colorbar(scatter, label='target')
        plt.tight_layout()
        plt.savefig(task_type+"_feature_0 vs feature_1_" + name + '.png')
        # plt.show()
        if return_fig:
            return plt.gcf()

def visualize_action_counts(results):
    """
    根据 results 中记录的 action_counts 绘制不同错误率下各动作个数的条形图
    """

    error_rates = [r['error_rate'] for r in results]
    no_action_counts = [r['action_counts']['no_action'] for r in results]
    repair_counts = [r['action_counts']['repair'] for r in results]
    delete_counts = [r['action_counts']['delete'] for r in results]

    x = np.arange(len(error_rates))  # 错误率组数
    width = 0.25  # 每个条形宽度

    fig, ax = plt.subplots(figsize=(4,6))
    rects1 = ax.bar(x - width, no_action_counts, width, label='No Action', color='gray')
    rects2 = ax.bar(x, repair_counts, width, label='Repair', color='blue')
    rects3 = ax.bar(x + width, delete_counts, width, label='Delete', color='red')

    ax.set_ylabel('动作次数')
    ax.set_xlabel('错误率')
    ax.set_title('不同错误率下动作个数统计')
    ax.set_xticks(x)
    ax.set_xticklabels(error_rates)
    ax.legend()
    plt.tight_layout()
    plt.savefig('action_counts'+str(error_rates)+'.png')
    # plt.show()
# =============================================================================
# 2. 错误注入器：支持缺失值、异常值和噪声注入
# =============================================================================
class ErrorInjector:
    def __init__(self, df, target_col='target'):
        """
        初始化错误注入器
        参数:
          - df: 干净数据集（DataFrame）
          - target_col: 目标列名称
        属性:
          - self.df: 当前数据（可能被注入错误）
          - self.original_df: 原始干净数据（备份）
          - self.feature_cols: 除目标列外的所有特征名称
          - self.error_locations: 用字典记录每个错误位置的信息，
            键为 (行索引, 列名称)，值为 {'type': 错误类型, 'original_value': 原始值, 'current_value': 当前值}
        """
        self.df = df.copy()
        self.original_df = df.copy()
        self.target_col = target_col
        self.feature_cols = [col for col in df.columns if col != target_col]
        self.error_locations = {}

    def inject_missing_values(self, error_rate=0.1, feature_importance=None):
        """
        注入缺失值错误，并在结束时标注注入了多少单元格错误
        """
        df = self.df.copy()
        total_cells = len(df) * len(self.feature_cols)
        missing_cells = int(total_cells * error_rate)
        if feature_importance is None:
            feature_importance = {col: 1 for col in self.feature_cols}
        weights = np.array([feature_importance[col] for col in self.feature_cols])
        weights = weights / weights.sum()
        injected_count = 0  # 记录实际注入的单元格数量
        for _ in range(missing_cells):
            row_idx = np.random.randint(0, len(df))
            col_idx = np.random.choice(len(self.feature_cols), p=weights)
            col_name = self.feature_cols[col_idx]
            # 检查该单元格是否已经注入过错误
            if (row_idx, col_name) not in self.error_locations:
                self.error_locations[(row_idx, col_name)] = {
                    'type': 'missing',
                    'original_value': df.at[row_idx, col_name],
                    'current_value': np.nan
                }
                df.at[row_idx, col_name] = np.nan
                injected_count += 1
        self.df = df
        print(f"注入缺失值错误: 共注入 {injected_count} 个单元格错误.")
        return df

    def inject_outliers(self, error_rate=0.05, outlier_factor=3.0, feature_importance=None):
        """
        注入异常值错误，并在结束时标注注入了多少单元格错误。
        异常值注入时，从该特征列中随机选取一个与当前值不同的值
        """
        df = self.df.copy()
        total_cells = len(df) * len(self.feature_cols)
        outlier_cells = int(total_cells * error_rate)
        if feature_importance is None:
            feature_importance = {col: 1 for col in self.feature_cols}
        weights = np.array([feature_importance[col] for col in self.feature_cols])
        weights = weights / weights.sum()
        injected_count = 0
        for _ in range(outlier_cells):
            row_idx = np.random.randint(0, len(df))
            col_idx = np.random.choice(len(self.feature_cols), p=weights)
            col_name = self.feature_cols[col_idx]
            if (row_idx, col_name) not in self.error_locations:
                original_value = df.at[row_idx, col_name]
                # 从该列中抽取所有非 NaN 的唯一值
                col_values = df[col_name].dropna().unique()
                # 排除当前原始值
                possible_values = [v for v in col_values if v != original_value]
                if len(possible_values) == 0:
                    # 如果没有其他值，则退回原逻辑（用偏移量）
                    col_std = df[col_name].std()
                    outlier_direction = 1 if np.random.random() > 0.5 else -1
                    outlier_value = original_value + outlier_direction * outlier_factor * col_std
                else:
                    outlier_value = random.choice(possible_values)
                self.error_locations[(row_idx, col_name)] = {
                    'type': 'outlier',
                    'original_value': original_value,
                    'current_value': outlier_value
                }
                df.at[row_idx, col_name] = outlier_value
                injected_count += 1
        self.df = df
        print(f"注入异常值错误: 共注入 {injected_count} 个单元格错误.")
        return df
    def inject_noise(self, error_rate=0.1, noise_level=0.9, feature_importance=None):
        """
        注入噪声错误，并在结束时标注注入了多少单元格错误
        """
        df = self.df.copy()
        total_cells = len(df) * len(self.feature_cols)
        noise_cells = int(total_cells * error_rate)
        if feature_importance is None:
            feature_importance = {col: 1 for col in self.feature_cols}
        weights = np.array([feature_importance[col] for col in self.feature_cols])
        weights = weights / weights.sum()
        injected_count = 0
        for _ in range(noise_cells):
            row_idx = np.random.randint(0, len(df))
            col_idx = np.random.choice(len(self.feature_cols), p=weights)
            col_name = self.feature_cols[col_idx]
            if (row_idx, col_name) not in self.error_locations:
                col_std = df[col_name].std()
                noise = np.random.normal(0, noise_level * col_std)
                original_value = df.at[row_idx, col_name]
                noisy_value = original_value + noise
                self.error_locations[(row_idx, col_name)] = {
                    'type': 'noise',
                    'original_value': original_value,
                    'current_value': noisy_value
                }
                df.at[row_idx, col_name] = noisy_value
                injected_count += 1
        self.df = df
        print(f"注入噪声错误: 共注入 {injected_count} 个单元格错误.")
        return df

    def reset(self):
        """
        重置数据为原始干净数据，并清空错误记录
        """
        self.df = self.original_df.copy()
        self.error_locations = {}
        return self.df

    def get_error_mask(self):
        """
        返回一个布尔型DataFrame，标记出所有被注入错误的位置
        """
        error_mask = pd.DataFrame(False, index=self.df.index, columns=self.df.columns)
        for (row_idx, col_name), _ in self.error_locations.items():
            error_mask.at[row_idx, col_name] = True
        return error_mask


# =============================================================================
# 3. 数据预处理（Gym环境）
# =============================================================================

class DataCleaningEnv(gym.Env):
    def __init__(self, df, ml_model, error_locations, target_col='target', task_type='classification',
                 model_type='random_forest'):
        """
        初始化数据清洗环境。
        参数:
          - df: 注入错误后的数据集
          - ml_model: 用于初步构造环境时的备用模型（不一定用于最终评估）
          - error_locations: 错误注入器记录的错误信息
          - target_col: 目标列名称
          - task_type: 'classification' 或 'regression'
          - model_type: 用于下游模型评估的模型类型，例如:
                分类: 'random_forest'、'svm'、'logistic_regression'
                回归: 'random_forest'、'svm'、'linear_regression'
        """
        super(DataCleaningEnv, self).__init__()
        self.delete_count = 0
        self.repair_count = 0
        self.original_df = df.copy()
        self.df = df.copy()
        self.ml_model = ml_model
        self.error_locations = error_locations
        self.target_col = target_col
        self.task_type = task_type
        self.model_type = model_type
        self.processed_errors = {}
        self.feature_cols = [col for col in df.columns if col != target_col]
        # self.feature_importance = self._calculate_feature_importance()
        # 划分训练集和验证集（基于原始数据）
        X = self.original_df[self.feature_cols]
        y = self.original_df[self.target_col]
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.3, random_state=42)

        # 定义动作空间（0: 不操作，1: 修复，2: 删除）和状态空间（5维归一化向量）
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0]),
                                            high=np.array([1, 1, 1, 1, 1]),
                                            dtype=np.float32)

        self.current_error_idx = 0
        self.error_keys = list(self.error_locations.keys())

        # 计算基线性能（不处理错误时的下游模型性能）
        base_perf = self._evaluate_model(self.df)
        if self.task_type == 'classification':
            self.baseline_primary = base_perf["accuracy"]
        else:
            self.baseline_primary = base_perf["r2_score"]

    def reset(self):
        self.df = self.original_df.copy()
        self.processed_errors = {}
        self.current_error_idx = 0
        self.repair_count = 0  # 新增：修复计数器
        self.delete_count = 0  # 新增：删除计数器
        random.shuffle(self.error_keys)
        return self._get_state()

    def _get_state(self):
        """
        构造当前状态向量，包含：
          1. 错误类型：0 表示缺失，1 表示异常/噪声
          2. 特征重要性：通过随机森林估计，并归一化到 [0,1]
          3. 当前列错误率：该列中仍存在错误的比例
          4. 行位置归一化：基于原始数据集中的相对位置
          5. 列索引归一化：将特征在所有特征中的索引映射到 [0,1]
        当所有错误处理完毕时，返回全零向量
        """
        if self.current_error_idx >= len(self.error_keys):
            return np.zeros(5, dtype=np.float32)
        row_idx, col_name = self.error_keys[self.current_error_idx]
        error_info = self.error_locations[(row_idx, col_name)]
        error_type = 0 if error_info['type'] == 'missing' else 1
        # feature_importance = self.feature_importance[col_name]
        feature_importance = self._calculate_feature_importance()[col_name]
        col_error_count = sum(1 for (r, c) in self.error_locations.keys() if c == col_name and r in self.df.index)
        col_error_rate = col_error_count / len(self.df) if len(self.df) > 0 else 0
        row_pos = row_idx / (len(self.original_df) - 1) if len(self.original_df) > 1 else 0
        col_idx = self.feature_cols.index(col_name)
        norm_col_idx = col_idx / (len(self.feature_cols) - 1) if len(self.feature_cols) > 1 else 0
        return np.array([error_type, feature_importance, col_error_rate, row_pos, norm_col_idx], dtype=np.float32)

    def _calculate_feature_importance(self):
        X = self.original_df[self.feature_cols]
        y = self.original_df[self.target_col]
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        if self.task_type == 'classification':
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            importances = model.feature_importances_
            # print(importances)
        else:
            # if self.model_type in ['svm', 'linear_regression']:
            #     if self.model_type == 'svm':
            #         from sklearn.svm import SVR
            #         # 如果是 SVM 回归，使用线性SVR来近似获取特征重要性
            #         model = SVR(kernel='linear')
            #     else:
            #         model = LinearRegression()
            #     model.fit(X, y)
            #     # 对于线性回归，系数的绝对值可以作为重要性
            #     coefs = np.abs(model.coef_)
            #     coefs = coefs / (coefs.sum() + 1e-10)
            #     importances = coefs
            # else:
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)
            importances = model.feature_importances_

        # 将重要性归一化到 [0,1]
        importances = (importances - importances.min()) / (importances.max() - importances.min() + 1e-10)
        return {col: imp for col, imp in zip(self.feature_cols, importances)}

    def reset_rate(self, df_with_errors, error_locations):
        """
        更新环境中的数据为新的错误数据，并重置内部状态
        参数:
          - df_with_errors: 新的包含错误的数据集
          - error_locations: 新的错误记录字典
        """
        self.df = df_with_errors.copy()
        self.original_df = df_with_errors.copy()
        self.error_locations = error_locations
        self.error_keys = list(self.error_locations.keys())
        self.current_error_idx = 0
        return self.reset()

    def step(self, action):
        # 如果当前错误索引已经超过错误列表的长度，则表示所有错误都已处理完毕，
        # 返回全零状态向量、0奖励、done=True以及空的附加信息字典
        if self.current_error_idx >= len(self.error_keys):
            return np.zeros(5, dtype=np.float32), 0, True, {}

        row_idx, col_name = self.error_keys[self.current_error_idx]
        if row_idx not in self.df.index:
            self.current_error_idx += 1
            return self._get_state(), 0, self.current_error_idx >= len(self.error_keys), {}

        error_info = self.error_locations[(row_idx, col_name)]

        # 根据传入的动作执行对应处理：
        if action == 0:
            # 不操作，不更新计数器
            pass
        elif action == 1:
            # 修复动作
            self.df.at[row_idx, col_name] = error_info['original_value']
            self.repair_count += 1  # 增加修复计数
        elif action == 2:
            # 删除动作
            if row_idx in self.df.index:
                self.df = self.df.drop(row_idx)
            self.delete_count += 1  # 增加删除计数

        self.processed_errors[(row_idx, col_name)] = action
        self.current_error_idx += 1

        try:
            new_perf = self._evaluate_model(self.df)

            # 计算性能差异和奖励
            if self.task_type == 'classification':
                perf_diff = new_perf["accuracy"] - self.baseline_primary
            else:
                perf_diff = new_perf["r2_score"] - self.baseline_primary

            # 定义基本成本，可以根据实际情况调整
            base_repair_cost = 0.01
            base_delete_cost = 0.02

            # 根据累计次数计算成本惩罚（这里是线性增加，也可以使用其他函数）
            penalty = 0.0
            if action == 1:
                penalty = base_repair_cost * self.repair_count
            elif action == 2:
                penalty = base_delete_cost * self.delete_count

            reward = perf_diff - penalty
        except Exception as e:
            # 如果评估模型时出现任何未处理的错误
            print(f"环境step执行错误: {e}")
            # 给予负奖励，惩罚导致错误的动作
            reward = -0.5  # 明显的负奖励
            # 如果是删除动作导致的错误，可以给予更严厉的惩罚
            if action == 2:
                reward = -1.0

        done = self.current_error_idx >= len(self.error_keys)
        return self._get_state(), reward, done, {}

    def _evaluate_model(self, df):
        """
        使用下游模型评估当前数据的性能。
        根据 model_type 参数选择模型：
          - 分类任务: 'random_forest'（默认）、'svm'（SVC）、'logistic_regression'
          - 回归任务: 'random_forest'（默认）、'svm'（SVR）、'linear_regression'
        返回字典：
          - 分类: {"accuracy": ..., "f1_score": ...}
          - 回归: {"mse": ..., "r2_score": ...}
        """
        if len(df) == 0:
            if self.task_type == 'classification':
                return {"accuracy": 0, "f1_score": 0}
            else:
                return {"mse": float('inf'), "r2_score": 0}

        X_train = df[df.index.isin(self.X_train.index)][self.feature_cols]
        y_train = df[df.index.isin(self.X_train.index)][self.target_col]

        # 检查训练数据是否为空
        if len(X_train) == 0 or len(y_train) == 0:
            if self.task_type == 'classification':
                return {"accuracy": 0, "f1_score": 0}
            else:
                return {"mse": float('inf'), "r2_score": 0}

        # 对于分类任务，检查类别数量
        if self.task_type == 'classification':
            unique_classes = np.unique(y_train)
            if len(unique_classes) < 2:
                print(f"警告: 训练数据中只有一个类别: {unique_classes[0]}, 返回零性能")
                return {"accuracy": 0, "f1_score": 0}

        # 继续常规处理
        imputer = SimpleImputer(strategy='mean')
        X_train_imp = imputer.fit_transform(X_train)

        # 选择模型
        try:
            if self.task_type == 'classification':
                if self.model_type == 'svm':
                    model = SVC(probability=True, random_state=42)
                elif self.model_type == 'logistic_regression':
                    model = LogisticRegression(random_state=42, max_iter=1000)
                else:
                    model = RandomForestClassifier(n_estimators=10, random_state=42)
            else:
                if self.model_type == 'svm':
                    model = SVR()
                elif self.model_type == 'linear_regression':
                    model = LinearRegression()
                else:
                    model = RandomForestRegressor(n_estimators=10, random_state=42)

            # 尝试训练模型
            model.fit(X_train_imp, y_train)

            # 评估模型
            X_val_imp = imputer.transform(self.X_val)
            if self.task_type == 'classification':
                y_pred = model.predict(X_val_imp)
                acc = accuracy_score(self.y_val, y_pred)
                f1 = f1_score(self.y_val, y_pred, average='weighted')
                return {"accuracy": acc, "f1_score": f1}
            else:
                y_pred = model.predict(X_val_imp)
                mse = mean_squared_error(self.y_val, y_pred)
                r2 = r2_score(self.y_val, y_pred)
                return {"mse": mse, "r2_score": r2}

        except ValueError as e:
            # 捕获可能的训练错误
            print(f"模型训练错误: {e}")
            if self.task_type == 'classification':
                return {"accuracy": 0, "f1_score": 0}
            else:
                return {"mse": float('inf'), "r2_score": 0}
        except Exception as e:
            # 捕获其他可能的错误
            print(f"意外错误: {e}")
            if self.task_type == 'classification':
                return {"accuracy": 0, "f1_score": 0}
            else:
                return {"mse": float('inf'), "r2_score": 0}


# =============================================================================
# 4. DQN强化学习代理
# =============================================================================
class DQNAgent:
    def __init__(self, state_size, action_size):
        """
        初始化DQN代理
        参数:
          - state_size: 状态向量维度（此处为5）
          - action_size: 动作数（3个动作：不操作、修复、删除）
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        """
        构建神经网络模型用于Q值估计
        模型结构：输入层 -> 两个隐藏层（各24个神经元, ReLU激活） -> 输出层（3个动作对应Q值）
        """
        model = models.Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """
        存储一次经验：(状态, 动作, 奖励, 下一个状态, 是否结束)
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        根据当前状态选择动作（epsilon贪婪策略）
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        """
        从经验回放池中随机采样小批量样本进行训练更新
        """
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1), verbose=0)[0])
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# =============================================================================
# 5. ModelEvaluator：独立评估器，用于评估数据清洗后下游模型性能
# =============================================================================
class ModelEvaluator:
    def __init__(self, X_train, y_train, X_val, y_val, task_type='classification', model_type='random_forest'):
        """
        初始化评估器。
        参数:
          - X_train, y_train: 训练集
          - X_val, y_val: 验证集
          - task_type: 'classification' 或 'regression'
          - model_type: 用于评估的模型类型
        """
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.X_val = X_val.copy()
        self.y_val = y_val.copy()
        self.task_type = task_type
        self.model_type = model_type

    def evaluate_model(self, df):
        """
        在给定数据集 df 上评估模型性能。
        返回:
          - 分类任务: {"accuracy": ..., "f1_score": ...}
          - 回归任务: {"mse": ..., "r2_score": ...}
        """
        try:
            # 提取训练数据
            X_train = df[df.index.isin(self.X_train.index)][self.X_train.columns]
            y_train = self.y_train[df.index.intersection(self.X_train.index)]

            # 检查是否有足够的训练数据
            if X_train.shape[0] == 0 or len(y_train) == 0:
                if self.task_type == 'classification':
                    return {"accuracy": 0, "f1_score": 0}
                else:
                    return {"mse": float('inf'), "r2_score": 0}

            # 分类任务特殊检查：确保数据中有至少两个类别
            if self.task_type == 'classification':
                unique_classes = np.unique(y_train)
                if len(unique_classes) < 2:
                    print(f"警告: 训练数据中只有一个类别: {unique_classes}, 返回零性能")
                    return {"accuracy": 0, "f1_score": 0}

            # 数据预处理
            imputer = SimpleImputer(strategy='mean')
            X_train_imp = imputer.fit_transform(X_train)

            # 选择并训练模型
            if self.task_type == 'classification':
                if self.model_type == 'svm':
                    model = SVC(probability=True, random_state=42)
                elif self.model_type == 'logistic_regression':
                    model = LogisticRegression(random_state=42, max_iter=1000)
                else:
                    model = RandomForestClassifier(n_estimators=10, random_state=42)
            else:
                if self.model_type == 'svm':
                    model = SVR()
                elif self.model_type == 'linear_regression':
                    model = LinearRegression()
                else:
                    model = RandomForestRegressor(n_estimators=10, random_state=42)

            # 训练模型
            model.fit(X_train_imp, y_train)

            # 评估模型性能
            X_val_imp = imputer.transform(self.X_val)
            if self.task_type == 'classification':
                y_pred = model.predict(X_val_imp)
                acc = accuracy_score(self.y_val, y_pred)
                f1 = f1_score(self.y_val, y_pred, average='weighted')
                return {"accuracy": acc, "f1_score": f1}
            else:
                y_pred = model.predict(X_val_imp)
                mse = mean_squared_error(self.y_val, y_pred)
                r2 = r2_score(self.y_val, y_pred)
                return {"mse": mse, "r2_score": r2}

        except ValueError as e:
            # 捕获与类别相关的错误
            error_msg = str(e).lower()
            if "class" in error_msg:
                print(f"警告: 模型训练错误可能与类别有关: {e}")
                if self.task_type == 'classification':
                    return {"accuracy": 0, "f1_score": 0}
                else:
                    return {"mse": float('inf'), "r2_score": 0}
            else:
                # 记录并重新抛出其他错误
                print(f"未处理的ValueError: {e}")
                raise

        except Exception as e:
            # 捕获所有其他异常
            print(f"模型评估过程中发生意外错误: {e}")
            if self.task_type == 'classification':
                return {"accuracy": 0, "f1_score": 0}
            else:
                return {"mse": float('inf'), "r2_score": 0}


# # =============================================================================
# # 5. ModelEvaluator：独立评估器，用于评估数据清洗后下游模型性能
# # =============================================================================
# class ModelEvaluator:
#     def __init__(self, X_train, y_train, X_val, y_val, task_type='classification'):
#         """
#         初始化评估器
#         参数:
#           - X_train, y_train: 训练集
#           - X_val, y_val: 验证集
#           - task_type: 'classification' 或 'regression'
#         """
#         self.X_train = X_train.copy()
#         self.y_train = y_train.copy()
#         self.X_val = X_val.copy()
#         self.y_val = y_val.copy()
#         self.task_type = task_type
#
#     def evaluate_model(self, df):
#         """
#         在给定数据集df上评估模型性能。
#         只使用df中与原始训练集索引相交的数据进行训练。
#         返回:
#           - 分类任务: {"accuracy": ..., "f1_score": ...}
#           - 回归任务: {"mse": ..., "r2_score": ...}
#         """
#         # 筛选出与原始训练集有交集的数据
#         X_train = df[df.index.isin(self.X_train.index)][self.X_train.columns]
#         y_train = self.y_train[df.index.intersection(self.X_train.index)]
#         # 如果训练数据为空，则返回默认性能值
#         if X_train.shape[0] == 0:
#             if self.task_type == 'classification':
#                 return {"accuracy": 0, "f1_score": 0}
#             else:
#                 return {"mse": float('inf'), "r2_score": 0}
#
#         imputer = SimpleImputer(strategy='mean')
#         X_train_imp = imputer.fit_transform(X_train)
#         if self.task_type == 'classification':
#             model = RandomForestClassifier(n_estimators=10, random_state=42)
#         else:
#             model = RandomForestRegressor(n_estimators=10, random_state=42)
#         model.fit(X_train_imp, y_train)
#         X_val_imp = imputer.transform(self.X_val)
#         if self.task_type == 'classification':
#             y_pred = model.predict(X_val_imp)
#             acc = accuracy_score(self.y_val, y_pred)
#             f1 = f1_score(self.y_val, y_pred, average='weighted')
#             return {"accuracy": acc, "f1_score": f1}
#         else:
#             y_pred = model.predict(X_val_imp)
#             mse = mean_squared_error(self.y_val, y_pred)
#             r2 = r2_score(self.y_val, y_pred)
#             return {"mse": mse, "r2_score": r2}

# =============================================================================
# 6. evaluate_strategies：对比四种数据清洗策略的性能
# =============================================================================
def evaluate_strategies(task_type, df_corrupted, error_locations, evaluator, env, rl_agent):
    """
    对比四种数据清洗策略在下游模型上的性能：
      1. Do Nothing：直接使用错误数据评估
      2. Delete All：删除所有包含错误的行后评估
      3. Repair All：将所有错误恢复为原始值后评估
      4. RL Optimal：使用RL代理逐个处理错误后评估
    返回:
      - 字典，包含每种策略对应的性能评估结果
    """
    print("Evaluating strategies...")

    # 辅助函数：评估模型并处理错误
    def safe_evaluate(df, strategy_name):
        try:
            return evaluator.evaluate_model(df)
        except ValueError as e:
            error_msg = str(e).lower()
            # 使用更广泛的检查，查找关于类别数量不足的各种错误消息
            if ("class" in error_msg and "one" in error_msg) or \
                    ("classes" in error_msg) or \
                    ("contains only one class" in error_msg) or \
                    ("at least 2 classes" in error_msg) or \
                    ("got 1 class" in error_msg):
                print(f"警告: {strategy_name} 策略导致数据只剩单一类别，性能设为0")
                if task_type == 'classification':
                    return {'accuracy': 0, 'f1_score': 0, 'precision': 0, 'recall': 0}
                else:
                    return {'r2_score': 0, 'mse': float('inf'), 'mae': float('inf')}
            else:
                # 打印错误信息，然后重新抛出
                print(f"未处理的错误: {e}")
                raise

    # 策略1：Do Nothing
    performance_nothing = safe_evaluate(df_corrupted, "Do Nothing")
    # visualize_data(df_corrupted, task_type, 'Do Nothing')
    print(f"Do Nothing: {performance_nothing}")

    # 策略2：Delete All
    df_delete = df_corrupted.copy()
    rows_to_delete = set([row for (row, _) in error_locations.keys()])
    df_delete = df_delete.drop(list(rows_to_delete))
    performance_delete = safe_evaluate(df_delete, "Delete All")
    # visualize_data(df_delete, task_type, 'Delete All')
    print(f"Delete All: {performance_delete}")

    # 策略3：Repair All
    df_repair = df_corrupted.copy()
    for (row, col), info in error_locations.items():
        df_repair.at[row, col] = info['original_value']
    performance_repair = safe_evaluate(df_repair, "Repair All")
    # visualize_data(df_repair, task_type, 'Repair All')
    print(f"Repair All: {performance_repair}")

    # 策略4：RL Optimal
    env.df = df_corrupted.copy()
    env.current_error_idx = 0
    while env.current_error_idx < len(env.error_keys):
        state = env._get_state()
        if np.all(state == 0):
            break
        action = rl_agent.act(state)
        _, _, done, _ = env.step(action)
        if done:
            break
    performance_rl = safe_evaluate(env.df, "RL Optimal")
    # visualize_data(env.df, task_type, 'Repair RL')
    print(f"RL Optimal: {performance_rl}")

    return {
        'do_nothing': performance_nothing,
        'delete_all': performance_delete,
        'repair_all': performance_repair,
        'RL_optimal': performance_rl
    }


# =============================================================================
# 7. run_experiment：对每个错误率进行RL训练和策略对比评估
# =============================================================================
# 修改 run_experiment 函数：统一初始化 env，然后每个错误率下调用 reset_rate 更新数据
import os


def run_experiment(task_type='classification', n_episodes=100,
                   error_rates=[0.05, 0.1, 0.2, 0.3, 0.4],
                   model_type='random_forest', reload_model=False,
                   model_path="dqn_agent.h5"):
    results = []
    # 生成干净数据
    clean_df = generate_clean_data(task_type=task_type)
    # visualize_data(clean_df, task_type, 'Clean Data_'+task_type)
    # 初始化错误注入器
    injector = ErrorInjector(clean_df)
    # 使用第一个错误率初始化错误数据
    default_rate = error_rates[0]
    df_with_errors = injector.inject_missing_values(error_rate=default_rate / 3)
    df_with_errors = injector.inject_outliers(error_rate=default_rate / 3)
    df_with_errors = injector.inject_noise(error_rate=default_rate / 3, noise_level=0.1)
    # visualize_data(df_with_errors, task_type, str(default_rate)+'_Injected Data_'+task_type)
    # 下游模型，用于环境初始化
    if task_type == 'classification':
        ml_model = RandomForestClassifier(n_estimators=10, random_state=42)
    else:
        ml_model = RandomForestRegressor(n_estimators=10, random_state=42)
    # 初始化环境（统一 env）
    env = DataCleaningEnv(df_with_errors, ml_model, injector.error_locations,
                          task_type=task_type, model_type=model_type)
    # 初始化RL代理
    agent = DQNAgent(state_size=5, action_size=3)
    # 如果设置了加载本地模型且模型文件存在，则加载
    if reload_model and os.path.exists(model_path):
        # agent.model = tf.keras.models.load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
        # agent.model = tf.keras.models.load_model(model_path)
        agent.model = tf.keras.models.load_model(model_path, compile=False)
        agent.model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        print(f"Loaded model from {model_path}.")
    # 下游评估器
    X = clean_df.drop('target', axis=1)
    y = clean_df['target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    evaluator = ModelEvaluator(X_train, y_train, X_val, y_val,
                               task_type=task_type, model_type=model_type)

    # 针对不同错误率进行训练与评估
    for error_rate in tqdm(error_rates, desc="Testing error rates"):
        print(f"\nTesting error rate: {error_rate}")
        # 重置注入器，重新生成新的错误数据
        injector.reset()
        df_with_errors = injector.inject_missing_values(error_rate=error_rate / 3)
        df_with_errors = injector.inject_outliers(error_rate=error_rate / 3)
        df_with_errors = injector.inject_noise(error_rate=error_rate / 3, noise_level=0.1)
        # visualize_data(df_with_errors, task_type, str(default_rate) + '_Injected Data_' + task_type)
        # 更新统一的环境：重新设置数据和错误记录
        env.reset_rate(df_with_errors, injector.error_locations)

        # RL训练阶段：在当前错误率下进行 n_episodes 的训练
        for e in tqdm(range(n_episodes), desc="Training episodes", leave=False):
            state = env.reset()
            for _ in range(len(injector.error_locations)):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break
            agent.replay(batch_size=32)
            if (e + 1) % 10 == 0:
                print(f"Episode: {e + 1}/{n_episodes}")

        # 评估各策略性能
        strat_perf = evaluate_strategies(task_type, df_with_errors, injector.error_locations, evaluator, env, agent)

        # 收集代理在环境中动作统计数据
        actions = []
        feature_importances = []
        error_types = []
        state = env.reset()  # 重新初始化状态
        for _ in range(len(injector.error_locations)):
            action = agent.act(state)
            actions.append(action)
            etype, f_imp, _, _, _ = state
            feature_importances.append(f_imp)
            error_types.append(etype)
            next_state, _, done, _ = env.step(action)
            state = next_state
            if done:
                break
        actions = np.array(actions)
        feature_importances = np.array(feature_importances)
        action_rate = {
            'no_action': np.sum(actions == 0) / len(actions),
            'repair': np.sum(actions == 1) / len(actions),
            'delete': np.sum(actions == 2) / len(actions)
        }
        action_counts = {
            'no_action': np.sum(actions == 0),
            'repair': np.sum(actions == 1),
            'delete': np.sum(actions == 2)
        }
        high_importance_mask = feature_importances > 0.5
        low_importance_mask = feature_importances <= 0.5
        high_importance_actions = {
            'no_action': np.sum(actions[high_importance_mask] == 0) / max(1, np.sum(high_importance_mask)),
            'repair': np.sum(actions[high_importance_mask] == 1) / max(1, np.sum(high_importance_mask)),
            'delete': np.sum(actions[high_importance_mask] == 2) / max(1, np.sum(high_importance_mask))
        }
        low_importance_actions = {
            'no_action': np.sum(actions[low_importance_mask] == 0) / max(1, np.sum(low_importance_mask)),
            'repair': np.sum(actions[low_importance_mask] == 1) / max(1, np.sum(low_importance_mask)),
            'delete': np.sum(actions[low_importance_mask] == 2) / max(1, np.sum(low_importance_mask))
        }
        missing_mask = np.array(error_types) == 0
        outlier_mask = np.array(error_types) == 1
        missing_actions = {
            'no_action': np.sum(actions[missing_mask] == 0) / max(1, np.sum(missing_mask)),
            'repair': np.sum(actions[missing_mask] == 1) / max(1, np.sum(missing_mask)),
            'delete': np.sum(actions[missing_mask] == 2) / max(1, np.sum(missing_mask))
        }
        outlier_actions = {
            'no_action': np.sum(actions[outlier_mask] == 0) / max(1, np.sum(outlier_mask)),
            'repair': np.sum(actions[outlier_mask] == 1) / max(1, np.sum(outlier_mask)),
            'delete': np.sum(actions[outlier_mask] == 2) / max(1, np.sum(outlier_mask))
        }
        final_performance = env._evaluate_model(env.df)
        results.append({
            'error_rate': error_rate,
            'overall_actions': action_rate,
            'high_importance_actions': high_importance_actions,
            'low_importance_actions': low_importance_actions,
            'missing_actions': missing_actions,
            'outlier_actions': outlier_actions,
            'final_performance': final_performance,
            'strategy_performance': strat_perf,
            'action_counts': action_counts
        })
        visualize_strategy_comparison(results, task_type=task_type)

        # 每训练完当前错误率后保存模型到本地
        agent.model.save(model_path)
        print(f"Model saved to {model_path} after training at error rate {error_rate}")

    return results


# def run_experiment(task_type='classification', n_episodes=100, error_rates=[0.05, 0.1, 0.2, 0.3, 0.4],
#                    model_type='random_forest'):
#     results = []
#     for error_rate in tqdm(error_rates, desc="Testing error rates"):
#         # try:
#         print(f"\nTesting error rate: {error_rate}")
#         clean_df = generate_clean_data(task_type=task_type)
#         visualize_data(clean_df, task_type)
#         injector = ErrorInjector(clean_df)
#         df_with_errors = injector.inject_missing_values(error_rate=error_rate/3)
#         # 如有需要，可同时注入异常值和噪声：
#         df_with_errors = injector.inject_outliers(error_rate=error_rate/3)
#         df_with_errors = injector.inject_noise(error_rate=error_rate/3, noise_level=0.1)
#         visualize_data(df_with_errors, task_type)
#         if task_type == 'classification':
#             ml_model = RandomForestClassifier(n_estimators=10, random_state=42)
#         else:
#             ml_model = RandomForestRegressor(n_estimators=10, random_state=42)
#
#         env = DataCleaningEnv(df_with_errors, ml_model, injector.error_locations, task_type=task_type,
#                               model_type=model_type)
#         agent = DQNAgent(state_size=5, action_size=3)
#         X = clean_df.drop('target', axis=1)
#         y = clean_df['target']
#         X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
#         evaluator = ModelEvaluator(X_train, y_train, X_val, y_val, task_type=task_type, model_type=model_type)
#         # strat_perf = evaluate_strategies(task_type, df_with_errors, injector.error_locations, evaluator, env, agent)
#         # print(strat_perf)
#         for e in tqdm(range(n_episodes), desc="Training episodes", leave=False):
#             state = env.reset()
#             for _ in range(len(injector.error_locations)):
#                 action = agent.act(state)
#                 next_state, reward, done, _ = env.step(action)
#                 agent.remember(state, action, reward, next_state, done)
#                 state = next_state
#                 if done:
#                     break
#             agent.replay(batch_size=32)
#             if (e + 1) % 10 == 0:
#                 print(f"Episode: {e + 1}/{n_episodes}")
#
#         strat_perf = evaluate_strategies(task_type, df_with_errors, injector.error_locations, evaluator, env, agent)
#
#         actions = []
#         feature_importances = []
#         error_types = []
#         for _ in range(len(injector.error_locations)):
#             action = agent.act(state)
#             actions.append(action)
#             etype, f_imp, _, _, _ = state
#             feature_importances.append(f_imp)
#             error_types.append(etype)
#             next_state, _, done, _ = env.step(action)
#             state = next_state
#             if done:
#                 break
#         actions = np.array(actions)
#         feature_importances = np.array(feature_importances)
#         action_counts = {
#             'no_action': np.sum(actions == 0) / len(actions),
#             'repair': np.sum(actions == 1) / len(actions),
#             'delete': np.sum(actions == 2) / len(actions)
#         }
#         high_importance_mask = feature_importances > 0.5
#         low_importance_mask = feature_importances <= 0.5
#         high_importance_actions = {
#             'no_action': np.sum(actions[high_importance_mask] == 0) / max(1, np.sum(high_importance_mask)),
#             'repair': np.sum(actions[high_importance_mask] == 1) / max(1, np.sum(high_importance_mask)),
#             'delete': np.sum(actions[high_importance_mask] == 2) / max(1, np.sum(high_importance_mask))
#         }
#         low_importance_actions = {
#             'no_action': np.sum(actions[low_importance_mask] == 0) / max(1, np.sum(low_importance_mask)),
#             'repair': np.sum(actions[low_importance_mask] == 1) / max(1, np.sum(low_importance_mask)),
#             'delete': np.sum(actions[low_importance_mask] == 2) / max(1, np.sum(low_importance_mask))
#         }
#         missing_mask = np.array(error_types) == 0
#         outlier_mask = np.array(error_types) == 1
#         missing_actions = {
#             'no_action': np.sum(actions[missing_mask] == 0) / max(1, np.sum(missing_mask)),
#             'repair': np.sum(actions[missing_mask] == 1) / max(1, np.sum(missing_mask)),
#             'delete': np.sum(actions[missing_mask] == 2) / max(1, np.sum(missing_mask))
#         }
#         outlier_actions = {
#             'no_action': np.sum(actions[outlier_mask] == 0) / max(1, np.sum(outlier_mask)),
#             'repair': np.sum(actions[outlier_mask] == 1) / max(1, np.sum(outlier_mask)),
#             'delete': np.sum(actions[outlier_mask] == 2) / max(1, np.sum(outlier_mask))
#         }
#         final_performance = env._evaluate_model(env.df)
#         results.append({
#             'error_rate': error_rate,
#             'overall_actions': action_counts,
#             'high_importance_actions': high_importance_actions,
#             'low_importance_actions': low_importance_actions,
#             'missing_actions': missing_actions,
#             'outlier_actions': outlier_actions,
#             'final_performance': final_performance,
#             'strategy_performance': strat_perf
#         })
#         visualize_strategy_comparison(results, task_type=task_type)
#     # except Exception as e:
#     #     print(f"Error at error_rate {error_rate}: {e}")
#     #     continue
#     return results


# =============================================================================
# 8. 可视化函数：包括策略对比和整体结果
# =============================================================================
def visualize_strategy_comparison(results, task_type='classification'):
    """
    针对每个错误率，绘制柱状图比较四种策略（Do Nothing, Delete All, Repair All, RL Optimal）的性能指标
    （以 accuracy 为例，仅适用于分类任务）。
    """
    for res in results:
        er = res['error_rate']
        strat_perf = res['strategy_performance']
        if task_type == 'classification':
            strategies = ['Do Nothing', 'Delete All', 'Repair All', 'RL Optimal']
            accuracies = [strat_perf['do_nothing']["accuracy"],
                          strat_perf['delete_all']["accuracy"],
                          strat_perf['repair_all']["accuracy"],
                          strat_perf['RL_optimal']["accuracy"]]
            plt.figure(figsize=(8, 6))
            plt.bar(strategies, accuracies, color=['gray', 'red', 'blue', 'green'])
            plt.xlabel("Strategy")
            plt.ylabel("Accuracy")
            plt.title(f"Performance Comparison at Error Rate: {er}")
            plt.ylim(0, 1)
            for i, acc in enumerate(accuracies):
                plt.text(i, acc + 0.02, f"{acc:.3f}", ha='center')
            plt.tight_layout()
            plt.savefig('strategy_comparison' + str(er) + '.png')
            # plt.show()


def visualize_results(results, task_type='classification'):
    """
    根据实验结果生成多幅图表
    参数:
      results: run_experiment返回的结果列表
      task_type: 'classification'或'regression'
    """
    error_rates = [r['error_rate'] for r in results]

    # 图1：整体动作分布
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(error_rates, [r['overall_actions']['no_action'] for r in results], 'o-', label='No Action')
    plt.plot(error_rates, [r['overall_actions']['repair'] for r in results], 's-', label='Repair')
    plt.plot(error_rates, [r['overall_actions']['delete'] for r in results], '^-', label='Delete')
    plt.xlabel('Error Rate')
    plt.ylabel('Action Proportion')
    plt.title('Overall Action Distribution')
    plt.legend()
    plt.grid(True)

    # 图2：高重要性特征下的动作分布
    plt.subplot(2, 2, 2)
    plt.plot(error_rates, [r['high_importance_actions']['no_action'] for r in results], 'o-', label='No Action')
    plt.plot(error_rates, [r['high_importance_actions']['repair'] for r in results], 's-', label='Repair')
    plt.plot(error_rates, [r['high_importance_actions']['delete'] for r in results], '^-', label='Delete')
    plt.xlabel('Error Rate')
    plt.ylabel('Action Proportion')
    plt.title('High Importance Actions')
    plt.legend()
    plt.grid(True)

    # 图3：低重要性特征下的动作分布
    plt.subplot(2, 2, 3)
    plt.plot(error_rates, [r['low_importance_actions']['no_action'] for r in results], 'o-', label='No Action')
    plt.plot(error_rates, [r['low_importance_actions']['repair'] for r in results], 's-', label='Repair')
    plt.plot(error_rates, [r['low_importance_actions']['delete'] for r in results], '^-', label='Delete')
    plt.xlabel('Error Rate')
    plt.ylabel('Action Proportion')
    plt.title('Low Importance Actions')
    plt.legend()
    plt.grid(True)

    # 图4：不同错误类型下的动作分布
    plt.subplot(2, 2, 4)
    plt.plot(error_rates, [r['missing_actions']['repair'] for r in results], 'o-', label='Repair Missing')
    plt.plot(error_rates, [r['missing_actions']['delete'] for r in results], 's-', label='Delete Missing')
    plt.plot(error_rates, [r['outlier_actions']['repair'] for r in results], '^-', label='Repair Outlier')
    plt.plot(error_rates, [r['outlier_actions']['delete'] for r in results], 'D-', label='Delete Outlier')
    plt.xlabel('Error Rate')
    plt.ylabel('Action Proportion')
    plt.title('Actions by Error Type')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('action_distribution'+str(error_rates)+'.png')
    # plt.show()

    # 图5：容忍度界限图（展示修复+删除比例）
    plt.figure(figsize=(10, 6))
    process_rates = [(r['overall_actions']['repair'] + r['overall_actions']['delete']) for r in results]
    repair_vs_delete = []
    for r in results:
        total_process = r['overall_actions']['repair'] + r['overall_actions']['delete']
        repair_vs_delete.append(r['overall_actions']['repair'] / total_process if total_process > 0 else 0)
    plt.plot(error_rates, process_rates, 'o-', color='blue', label='处理比例 (修复+删除)')
    plt.axhline(y=0.5, color='r', linestyle='--', label='容忍界限 (50%)')
    tolerance_threshold = None
    for i in range(len(error_rates) - 1):
        if process_rates[i] < 0.5 and process_rates[i + 1] >= 0.5:
            tolerance_threshold = error_rates[i] + (error_rates[i + 1] - error_rates[i]) * (0.5 - process_rates[i]) / (
                    process_rates[i + 1] - process_rates[i])
            break
    if tolerance_threshold:
        plt.axvline(x=tolerance_threshold, color='g', linestyle='--',
                    label=f'错误率容忍界限: {tolerance_threshold:.3f}')
    plt.xlabel('Error Rate')
    plt.ylabel('处理操作比例')
    plt.title('数据错误容忍度界限分析')
    plt.legend()
    plt.grid(True)
    plt.savefig('tolerance_threshold'+str(error_rates)+'.png')
    # plt.show()

    # 图6：修复 vs 删除平衡点图
    plt.figure(figsize=(10, 6))
    plt.plot(error_rates, repair_vs_delete, 's-', color='purple', label='修复占比')
    plt.axhline(y=0.5, color='r', linestyle='--', label='修复/删除平衡点 (50%)')
    repair_threshold = None
    for i in range(len(error_rates) - 1):
        if repair_vs_delete[i] >= 0.5 and repair_vs_delete[i + 1] < 0.5:
            repair_threshold = error_rates[i] + (error_rates[i + 1] - error_rates[i]) * (0.5 - repair_vs_delete[i]) / (
                    repair_vs_delete[i + 1] - repair_vs_delete[i])
            break
    if repair_threshold:
        plt.axvline(x=repair_threshold, color='g', linestyle='--', label=f'修复/删除界限: {repair_threshold:.3f}')
    plt.xlabel('Error Rate')
    plt.ylabel('修复占比')
    plt.title('修复 vs 删除策略界限分析')
    plt.legend()
    plt.grid(True)
    plt.savefig('repair_vs_delete_threshold'+str(error_rates)+'.png')
    # plt.show()

    # 图7：性能指标与错误率关系图
    plt.figure(figsize=(10, 6))
    if task_type == 'classification':
        accuracies = [r['final_performance']['accuracy'] for r in results]
        f1_scores = [r['final_performance']['f1_score'] for r in results]
        plt.plot(error_rates, accuracies, 'o-', label='Accuracy')
        plt.plot(error_rates, f1_scores, 's-', label='F1 Score')
        plt.ylabel('Performance')
        plt.title('Classification Performance vs Error Rate')
    else:
        mses = [r['final_performance']['mse'] for r in results]
        r2_scores = [r['final_performance']['r2_score'] for r in results]
        plt.plot(error_rates, mses, 'o-', label='MSE')
        plt.plot(error_rates, r2_scores, 's-', label='R2 Score')
        plt.ylabel('Performance')
        plt.title('Regression Performance vs Error Rate')
    plt.xlabel('Error Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_vs_error_rate'+str(error_rates)+'.png')
    # plt.show()

    visualize_action_counts(results)



def analyze_thresholds(results):
    """
    分析实验结果，计算各项容忍度界限
    返回一个字典，包含：
      - overall_tolerance: 总体数据质量容忍界限
      - repair_vs_delete: 修复与删除策略的临界点
      - key_feature_tolerance: 关键特征的容忍界限
      - non_key_feature_tolerance: 非关键特征的容忍界限
      - missing_tolerance: 缺失值的容忍界限
      - outlier_tolerance: 异常值的容忍界限
    """
    error_rates = [r['error_rate'] for r in results]
    process_rates = [(r['overall_actions']['repair'] + r['overall_actions']['delete']) for r in results]
    repair_vs_delete = []
    for r in results:
        total_process = r['overall_actions']['repair'] + r['overall_actions']['delete']
        repair_vs_delete.append(r['overall_actions']['repair'] / total_process if total_process > 0 else 0)

    tolerance_threshold = None
    for i in range(len(error_rates) - 1):
        if process_rates[i] < 0.5 and process_rates[i + 1] >= 0.5:
            tolerance_threshold = error_rates[i] + (error_rates[i + 1] - error_rates[i]) * (0.5 - process_rates[i]) / (
                    process_rates[i + 1] - process_rates[i])
            break

    repair_threshold = None
    for i in range(len(error_rates) - 1):
        if repair_vs_delete[i] >= 0.5 and repair_vs_delete[i + 1] < 0.5:
            repair_threshold = error_rates[i] + (error_rates[i + 1] - error_rates[i]) * (0.5 - repair_vs_delete[i]) / (
                    repair_vs_delete[i + 1] - repair_vs_delete[i])
            break

    high_importance_process = []
    low_importance_process = []
    for r in results:
        high_process = r['high_importance_actions']['repair'] + r['high_importance_actions']['delete']
        low_process = r['low_importance_actions']['repair'] + r['low_importance_actions']['delete']
        high_importance_process.append(high_process)
        low_importance_process.append(low_process)

    key_feature_threshold = None
    for i in range(len(error_rates) - 1):
        if high_importance_process[i] < 0.5 and high_importance_process[i + 1] >= 0.5:
            key_feature_threshold = error_rates[i] + (error_rates[i + 1] - error_rates[i]) * (
                    0.5 - high_importance_process[i]) / (
                                            high_importance_process[i + 1] - high_importance_process[i])
            break

    non_key_feature_threshold = None
    for i in range(len(error_rates) - 1):
        if low_importance_process[i] < 0.5 and low_importance_process[i + 1] >= 0.5:
            non_key_feature_threshold = error_rates[i] + (error_rates[i + 1] - error_rates[i]) * (
                    0.5 - low_importance_process[i]) / (low_importance_process[i + 1] - low_importance_process[i])
            break

    missing_process = []
    outlier_process = []
    for r in results:
        missing_proc = r['missing_actions']['repair'] + r['missing_actions']['delete']
        outlier_proc = r['outlier_actions']['repair'] + r['outlier_actions']['delete']
        missing_process.append(missing_proc)
        outlier_process.append(outlier_proc)

    missing_threshold = None
    for i in range(len(error_rates) - 1):
        if missing_process[i] < 0.5 and missing_process[i + 1] >= 0.5:
            missing_threshold = error_rates[i] + (error_rates[i + 1] - error_rates[i]) * (0.5 - missing_process[i]) / (
                    missing_process[i + 1] - missing_process[i])
            break

    outlier_threshold = None
    for i in range(len(error_rates) - 1):
        if outlier_process[i] < 0.5 and outlier_process[i + 1] >= 0.5:
            outlier_threshold = error_rates[i] + (error_rates[i + 1] - error_rates[i]) * (0.5 - outlier_process[i]) / (
                    outlier_process[i + 1] - outlier_process[i])
            break

    return {
        'overall_tolerance': tolerance_threshold,
        'repair_vs_delete': repair_threshold,
        'key_feature_tolerance': key_feature_threshold,
        'non_key_feature_tolerance': non_key_feature_threshold,
        'missing_tolerance': missing_threshold,
        'outlier_tolerance': outlier_threshold
    }


def generate_report(thresholds):
    """
    生成研究报告，将各个容忍界限及策略建议打印出来
    参数:
      thresholds: analyze_thresholds返回的字典
    """
    print("\n" + "=" * 80)
    print("         数据多样性与真实性视角的模型低质数据容忍度界限研究")
    print("=" * 80)

    print("\n1. 总体容忍度界限分析")
    print("-" * 50)
    if thresholds['overall_tolerance'] is not None:
        print(f"总体数据质量容忍界限: {thresholds['overall_tolerance']:.3f}")
        print(f"当错误率超过 {thresholds['overall_tolerance'] * 100:.1f}% 时，需要进行数据预处理")
    else:
        print("未找到明确的容忍界限，可能需要测试更高的错误率")

    print("\n2. 修复 vs 删除策略界限")
    print("-" * 50)
    if thresholds['repair_vs_delete'] is not None:
        print(f"修复/删除界限: {thresholds['repair_vs_delete']:.3f}")
        print(f"当错误率低于 {thresholds['repair_vs_delete'] * 100:.1f}% 时，修复策略更优")
        print(f"当错误率高于 {thresholds['repair_vs_delete'] * 100:.1f}% 时，删除策略更优")
    else:
        print("未找到明确的修复/删除界限，可能需要扩展实验范围")

    print("\n3. 特征重要性分析")
    print("-" * 50)
    if thresholds['key_feature_tolerance'] is not None and thresholds['non_key_feature_tolerance'] is not None:
        print(f"关键特征容忍界限: {thresholds['key_feature_tolerance']:.3f}")
        print(f"非关键特征容忍界限: {thresholds['non_key_feature_tolerance']:.3f}")
        diff = thresholds['non_key_feature_tolerance'] - thresholds['key_feature_tolerance']
        if diff > 0:
            print(f"结论: 关键特征的容忍度较低 (差异: {diff:.3f})")
            print("     模型对关键特征的数据质量要求更高")
        else:
            print("结论: 未发现关键特征与非关键特征的容忍度显著差异")
    else:
        print("未找到明确的特征重要性相关界限")

    print("\n4. 错误类型分析")
    print("-" * 50)
    if thresholds['missing_tolerance'] is not None and thresholds['outlier_tolerance'] is not None:
        print(f"缺失值容忍界限: {thresholds['missing_tolerance']:.3f}")
        print(f"异常值容忍界限: {thresholds['outlier_tolerance']:.3f}")
        diff = thresholds['outlier_tolerance'] - thresholds['missing_tolerance']
        if abs(diff) > 0.05:
            if diff > 0:
                print(f"结论: 模型对缺失值的容忍度较低 (差异: {diff:.3f})")
                print("     应优先处理缺失值问题")
            else:
                print(f"结论: 模型对异常值的容忍度较低 (差异: {abs(diff):.3f})")
                print("     应优先处理异常值问题")
        else:
            print("结论: 模型对缺失值和异常值的容忍度相近")
    else:
        print("未找到明确的错误类型相关界限")

    print("\n5. 总体研究结论")
    print("-" * 50)
    print("根据实验结果，建议的数据预处理策略如下:")
    if thresholds['overall_tolerance'] is not None:
        print(f"1. 当总体错误率低于 {thresholds['overall_tolerance'] * 100:.1f}% 时，可不做预处理")
    if thresholds['repair_vs_delete'] is not None:
        print(f"2. 需要预处理时：")
        print(f"   - 错误率低于 {thresholds['repair_vs_delete'] * 100:.1f}% 时，建议采用修复策略")
        print(f"   - 错误率高于 {thresholds['repair_vs_delete'] * 100:.1f}% 时，建议采用删除策略")
    if thresholds['key_feature_tolerance'] is not None and thresholds['non_key_feature_tolerance'] is not None:
        print(f"3. 针对关键特征，即使错误率较低（低于 {thresholds['key_feature_tolerance'] * 100:.1f}%）也应关注；")
        print(f"   非关键特征可容忍更高错误率（最高可达 {thresholds['non_key_feature_tolerance'] * 100:.1f}%）")
    print("\n" + "=" * 80)
    print("                              研究报告结束")
    print("=" * 80)


# =============================================================================
# 9. 主函数：整合所有模块，运行实验、生成图表、分析阈值与输出报告
# =============================================================================
def main():
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    task_type = 'classification'  # 'classification' 或 'regression'
    n_episodes = 50  # 每个错误率下的RL训练轮数
    # 设定较高的错误率以观察模型性能随错误率变化的趋势
    error_rates = [0.8, 0.9]
    # error_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # 设定下游评估模型类型，例如：
    # 对于分类任务可选：'random_forest', 'svm', 'logistic_regression'
    # 对于回归任务可选：'random_forest', 'svm', 'linear_regression'
    model_type = 'svm'

    print("开始运行实验...")
    print(f"任务类型: {task_type}")
    print(f"训练轮数: {n_episodes}")
    print(f"错误率范围: {str(error_rates)}")
    print(f"下游评估模型类型: {model_type}")

    # 运行实验：对每个错误率分别进行RL训练和策略对比评估
    results = run_experiment(task_type=task_type, n_episodes=n_episodes, error_rates=error_rates, model_type=model_type)

    print("生成可视化结果...")
    visualize_results(results, task_type=task_type)
    visualize_strategy_comparison(results, task_type=task_type)

    print("分析容忍度界限...")
    thresholds = analyze_thresholds(results)
    generate_report(thresholds)

    print("保存实验结果...")

    def convert_to_serializable(val):
        if isinstance(val, dict):
            return {k: convert_to_serializable(v) for k, v in val.items()}
        elif isinstance(val, (np.integer, np.floating)):
            return float(val)
        else:
            return val

    serializable_results = [convert_to_serializable(r) for r in results]
    with open('experiment_results.json', 'w') as f:
        json.dump({
            'results': serializable_results,
            'thresholds': convert_to_serializable(thresholds)
        }, f, indent=2)
    print("实验完成！结果已保存到 'experiment_results.json'")


if __name__ == "__main__":
    main()
