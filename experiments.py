# 1. 实验配置和数据生成函数
from DQN_extract import *
import matplotlib.patches as mpatches
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as path_effects

def setup_experiment(error_rates, models, task_type, n_samples, n_features,
                     missing_ratio, outlier_ratio, noise_ratio):
    """
    设置实验配置并返回初始化的数据结构
    """
    # 存储实验结果
    results = {model: {er: {} for er in error_rates} for model in models}
    action_dist = {model: {er: {} for er in error_rates} for model in models}
    tolerances = {model: [] for model in models}

    return results, action_dist, tolerances


# 2. 特征重要性计算函数
def calculate_feature_importance(df, task_type):
    """使用随机森林计算特征重要性"""
    X = df.drop('target', axis=1)
    y = df['target']

    if task_type == 'classification':
        model = RandomForestClassifier(n_estimators=50, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=50, random_state=42)

    model.fit(X, y)
    importances = model.feature_importances_

    return {feature: importance for feature, importance in zip(X.columns, importances)}


# 3. 运行单个错误率的实验
def run_single_error_rate_experiment(error_rate, models, task_type, n_samples, n_features,
                                     missing_ratio, outlier_ratio, noise_ratio, results,
                                     action_dist, tolerances):
    """运行单个错误率的实验并更新结果"""
    print(f"Running experiment with error rate: {error_rate}")

    # 训练阶段：生成干净数据集和计算特征重要性
    train_clean_df = generate_clean_data(task_type=task_type, n_samples=n_samples, n_features=n_features)
    feature_importance = calculate_feature_importance(train_clean_df, task_type)
    print(f"Feature importance: {feature_importance}")

    # 创建训练用错误注入器
    train_injector = ErrorInjector(train_clean_df)

    # 计算每种错误类型的错误率
    missing_err_rate = error_rate * missing_ratio
    outlier_err_rate = error_rate * outlier_ratio
    noise_err_rate = error_rate * noise_ratio

    # 注入训练数据错误（按特征重要性）
    train_injector.inject_missing_values(error_rate=missing_err_rate, feature_importance=feature_importance)
    train_injector.inject_outliers(error_rate=outlier_err_rate, feature_importance=feature_importance)
    train_injector.inject_noise(error_rate=noise_err_rate, feature_importance=feature_importance)

    # 获取训练用错误数据
    train_df_with_errors = train_injector.df

    # 准备训练数据
    X_train_full = train_clean_df.drop('target', axis=1)
    y_train_full = train_clean_df['target']
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=42)

    # 对每个模型运行实验
    for model_name in models:
        print(f"  Testing model: {model_name}")
        # 初始化ML模型
        ml_model = initialize_ml_model(task_type, model_name)

        # 创建训练环境和评估器
        train_env = DataCleaningEnv(train_df_with_errors.copy(), ml_model, train_injector.error_locations,
                                    task_type=task_type, model_type=model_name)

        # 训练RL代理
        agent,_ = train_rl_agent(train_env, train_injector.error_locations)

        # 测试阶段：生成新的干净测试数据和注入错误
        test_clean_df, test_df_with_errors, test_injector = generate_test_data(
            task_type, n_samples, n_features, missing_err_rate, outlier_err_rate,
            noise_err_rate, feature_importance)

        # 准备测试评估器和环境
        test_evaluator, test_env = setup_test_environment(
            test_clean_df, test_df_with_errors, test_injector, ml_model,
            task_type, model_name)

        # 评估不同策略
        strat_perf = evaluate_strategies(
            task_type, test_df_with_errors,
            test_injector.error_locations, test_evaluator, test_env, agent)

        # 记录结果
        update_results(results, action_dist, tolerances, model_name, error_rate,
                       task_type, strat_perf, test_env, test_injector, agent)

    return results, action_dist, tolerances


# 4. 模型初始化函数
def initialize_ml_model(task_type, model_name):
    """初始化机器学习模型"""
    if task_type == 'classification':
        if model_name == 'random_forest':
            return RandomForestClassifier(n_estimators=10, random_state=42)
        elif model_name == 'svm':
            return SVC(probability=True, random_state=42)
        else:
            return LogisticRegression(random_state=42)
    else:
        if model_name == 'random_forest':
            return RandomForestRegressor(n_estimators=10, random_state=42)
        elif model_name == 'svm':
            return SVR()
        else:
            return LinearRegression()


# 5. RL代理训练函数
def train_rl_agent(train_env, error_locations, n_episodes=80, model_name="default_agent",
                   reload_model=False, models_dir="saved_models", batch_size=32,
                   save_interval=None, learning_rate=0.001, exploration_rate=None, verbose=True):
    """
    训练RL代理，支持进度显示、模型保存和加载

    参数:
        train_env: 训练环境
        error_locations: 错误位置字典
        n_episodes: 训练轮次
        model_name: 模型名称，用于保存和加载
        reload_model: 是否加载已有模型
        models_dir: 模型保存目录
        batch_size: 经验回放批量大小
        save_interval: 模型保存间隔（轮次），None表示使用默认值
        learning_rate: 学习率，如果加载模型则忽略
        exploration_rate: 探索率，None表示使用DQNAgent默认值
        verbose: 是否显示详细进度

    返回:
        训练好的RL代理
    """
    # 确保模型目录存在
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # 构建完整的模型路径
    model_path = os.path.join(models_dir, f"{model_name}.h5")

    # 计算保存间隔
    if save_interval is None:
        save_interval = max(1, n_episodes // 5)  # 默认在20%、40%、60%、80%和100%保存

    # 初始化代理
    agent = DQNAgent(state_size=5, action_size=3)

    # 如果提供了自定义探索率
    if exploration_rate is not None:
        agent.epsilon = exploration_rate

    # 如果设置了加载模型且模型文件存在，则加载
    loaded_model = False
    if reload_model and os.path.exists(model_path):
        try:
            agent.model = tf.keras.models.load_model(model_path, compile=False)
            agent.model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
            loaded_model = True
            if verbose:
                print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training new model instead.")

    # 如果没有加载模型，显示使用的学习率
    if not loaded_model and verbose:
        print(f"Training new model with learning rate: {learning_rate}")

    # 创建训练日志
    training_log = {
        'episode_rewards': [],
        'episode_steps': [],
        'model_path': model_path,
        'loaded_model': loaded_model
    }

    # 使用tqdm显示训练进度
    episodes_range = tqdm(range(n_episodes), desc="Training episodes") if verbose else range(n_episodes)

    for e in episodes_range:
        state = train_env.reset()
        steps = 0
        total_reward = 0

        # 内部步骤进度条
        step_iterator = range(len(error_locations))
        if verbose and n_episodes <= 10:  # 只有当轮次较少时才显示步骤进度条
            step_iterator = tqdm(step_iterator, desc=f"Episode {e + 1} steps", leave=False)

        for _ in step_iterator:
            action = agent.act(state)
            next_state, reward, done, _ = train_env.step(action)
            agent.remember(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state
            steps += 1

            if done:
                break

        # 记录本轮的步数和奖励
        training_log['episode_rewards'].append(total_reward)
        training_log['episode_steps'].append(steps)

        # 只有当经验池足够大时才进行回放
        if len(agent.memory) > batch_size:
            agent.replay(batch_size=batch_size)

        # 在训练过程中显示信息
        if verbose and (e + 1) % max(1, n_episodes // 10) == 0:
            avg_reward = np.mean(training_log['episode_rewards'][-10:])
            avg_steps = np.mean(training_log['episode_steps'][-10:])
            print(
                f"Episode {e + 1}/{n_episodes} - Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}, Epsilon: {agent.epsilon:.4f}")

        # 定期保存模型
        if (e + 1) % save_interval == 0 or e == n_episodes - 1:
            agent.model.save(model_path)
            if verbose:
                print(f"Model saved to {model_path} after episode {e + 1}/{n_episodes}")

    # 完成训练后保存模型
    agent.model.save(model_path)
    if verbose:
        print(f"Training completed. Final model saved to {model_path}")

    return agent, training_log


# 6. 生成测试数据函数
def generate_test_data(task_type, n_samples, n_features, missing_err_rate,
                       outlier_err_rate, noise_err_rate, feature_importance):
    """生成测试数据集并注入错误"""
    test_clean_df = generate_clean_data(task_type=task_type, n_samples=n_samples, n_features=n_features)

    # 创建测试用错误注入器（使用相同的特征重要性）
    test_injector = ErrorInjector(test_clean_df)

    # 注入测试数据错误
    test_injector.inject_missing_values(error_rate=missing_err_rate)
    test_injector.inject_outliers(error_rate=outlier_err_rate)
    test_injector.inject_noise(error_rate=noise_err_rate)

    # 获取测试用错误数据
    test_df_with_errors = test_injector.df

    return test_clean_df, test_df_with_errors, test_injector


# 7. 设置测试环境函数
def setup_test_environment(test_clean_df, test_df_with_errors, test_injector,
                           ml_model, task_type, model_name):
    """设置测试评估器和环境"""
    # 准备测试评估器
    X_test_full = test_clean_df.drop('target', axis=1)
    y_test_full = test_clean_df['target']
    X_train_test, X_val_test, y_train_test, y_val_test = train_test_split(
        X_test_full, y_test_full, test_size=0.3, random_state=42)

    test_evaluator = ModelEvaluator(
        X_train_test, y_train_test, X_val_test, y_val_test,
        task_type=task_type, model_type=model_name)

    # 创建测试环境
    test_env = DataCleaningEnv(
        test_df_with_errors.copy(), ml_model, test_injector.error_locations,
        task_type=task_type, model_type=model_name)

    return test_evaluator, test_env


# 8. 更新结果函数
def update_results(results, action_dist, tolerances, model_name, error_rate,
                   task_type, strat_perf, test_env, test_injector, agent):
    """更新实验结果数据结构"""
    # 记录性能指标
    metric_key = 'accuracy' if task_type == 'classification' else 'r2_score'

    results[model_name][error_rate] = {
        'do_nothing': strat_perf['do_nothing'][metric_key],
        'delete_all': strat_perf['delete_all'][metric_key],
        'repair_all': strat_perf['repair_all'][metric_key],
        'rl_optimal': strat_perf['RL_optimal'][metric_key]
    }

    # 计算容忍度 (RL策略提升/理想提升比例)
    rl_gain = results[model_name][error_rate]['rl_optimal'] - results[model_name][error_rate]['do_nothing']
    ideal_gain = results[model_name][error_rate]['repair_all'] - results[model_name][error_rate]['do_nothing']
    tolerance = rl_gain / ideal_gain if ideal_gain > 0 else 1.0
    tolerances[model_name].append(tolerance)

    # 收集动作分布
    test_env.df = test_env.df.copy()
    test_env.current_error_idx = 0
    actions = []

    state = test_env.reset()
    for _ in range(len(test_injector.error_locations)):
        action = agent.act(state)
        actions.append(action)
        next_state, _, done, _ = test_env.step(action)
        state = next_state
        if done: break

    # 计算动作比例
    actions = np.array(actions)
    action_dist[model_name][error_rate] = {
        'no_action': np.sum(actions == 0) / len(actions),
        'repair': np.sum(actions == 1) / len(actions),
        'delete': np.sum(actions == 2) / len(actions)
    }

    return results, action_dist, tolerances


# 9. 绘制动作分布图函数
def plot_action_distribution(action_dist, models, error_rates, enhanced=True):
    """绘制动作分布对比图"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bar_width = 0.065
    index = np.arange(3)  # 3种动作
    action_names = ['No Action', 'Repair', 'Delete']

    # 颜色映射 - 使用对比度更高的颜色
    if enhanced:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 更鲜明的颜色
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    hatches = ['', '///', 'xxx', '...', '+++']  # 用于区分错误率

    # 绘制条形图
    for m_idx, model in enumerate(models):
        for e_idx, er in enumerate(error_rates[::1]):  # 选择部分错误率以避免过度拥挤
            values = [action_dist[model][er]['no_action'],
                      action_dist[model][er]['repair'],
                      action_dist[model][er]['delete']]

            # 计算条形位置
            offset = (m_idx * len(error_rates[::1]) + e_idx) * bar_width
            pos = index + offset - (len(models) * len(error_rates[::1]) - 1) * bar_width / 2

            # 绘制条形 - 调整透明度以增强可见性
            bars = ax.bar(pos, values, bar_width * 0.95,
                          color=colors[m_idx % len(colors)],
                          # color='white',
                          alpha=0.7 + 0.3 * e_idx / len(error_rates[::1]),
                          hatch=hatches[e_idx % len(hatches)],
                          # edgecolor=colors[m_idx % len(colors)],
                          # edgecolor='black',
                          label=f"{model.replace('_', ' ').title()} ({int(er * 100)}%)" if e_idx == 0 else "")
            for bar in bars:
                original_edge_color = bar.get_edgecolor()
                bar.set_edgecolor(original_edge_color)  # eg: 模型色
                bar.set_linewidth(0.01)
                bar.set_path_effects([
                    path_effects.withStroke(linewidth=3.5, foreground='black')  # 👈 外描边黑色
                ])

            # 为所有条形添加数字标签
            for i, v in enumerate(values):
                # 根据值的大小调整标签位置和颜色
                if v > 0.15:  # 更宽松的阈值，显示更多标签
                    text_color = 'black' if v > 0.3 else 'black'
                    ax.text(pos[i], v + 0.02, f"{v:.2f}", ha='center', va='bottom',
                            fontsize=7, rotation=90, color=text_color, fontweight='bold')

    # 设置图表属性 - 增强标题和标签
    ax.set_xlabel('Action Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Proportion', fontsize=12, fontweight='bold')
    ax.set_title('Action Distribution by Model and Error Rate', fontsize=14, fontweight='bold')
    ax.set_xticks(index)
    ax.set_xticklabels(action_names, fontsize=10, fontweight='bold')

    # 获取数据最大值，设置合适的y轴范围
    data_max = 0
    for model_dist in action_dist.values():
        for er_dist in model_dist.values():
            current_max = max(er_dist.values())
            if current_max > data_max:
                data_max = current_max

    # 设置y轴上限，确保所有标签都可见
    if data_max < 0.95:
        y_max = min(1.0, data_max * 1.15)  # 增加更多空间
    else:
        y_max = 1.05

    ax.set_ylim(0, y_max)

    # 添加网格线以提高可读性
    ax.grid(True, linestyle='--', alpha=0.3)

    # 添加模型图例 - 改进布局和可见性
    handles, labels = ax.get_legend_handles_labels()
    l1 = ax.legend(handles, labels, loc='upper center', fontsize=9,
                   ncol=len(models), bbox_to_anchor=(0.5, -0.12), framealpha=0.8)

    # 添加错误率图例 - 使用更明显的标记
    error_patches = []
    for e_idx, er in enumerate(error_rates[::1]):
        # patch = mpatches.Patch(color='white', hatch=hatches[e_idx % len(hatches)],
        #                        label=f"{int(er * 100)}%", alpha=0.7)
        patch = mpatches.Patch(facecolor='white', hatch=hatches[e_idx % len(hatches)], edgecolor='gray', label=f"{int(er * 100)}%")

        error_patches.append(patch)

    l2 = ax.legend(handles=error_patches, loc='upper right', fontsize=9,
                   title="Error Rate", title_fontsize=10, framealpha=0.8)
    ax.add_artist(l1)
    plt.tight_layout(rect=[0, 0.03, 1, 1])  # [left, bottom, right, top]

    # plt.tight_layout()  # 增加边距，确保所有元素可见
    # 获取最左边和最右边的条形位置
    total_bars = len(models) * len(error_rates)
    bar_span = total_bars * bar_width
    start = index[0] - bar_span / 2
    end = index[-1] + bar_span / 2 + bar_width

    # 设置 x 轴范围，稍微留出一点边距
    ax.set_xlim(start, end)

    return fig, ax


def plot_performance_comparison(results, tolerances, models, error_rates, task_type, enhanced=True):
    """使用分组条形图绘制多错误率策略性能对比图"""
    fig, ax = plt.subplots(figsize=(8, 5))

    # 设置条形图参数
    bar_width = 0.08  # 条形宽度
    opacity = 0.8  # 条形透明度

    # 使用对比度更高的颜色
    if enhanced:
        model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 模型颜色
    else:
        model_colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    # 性能指标标签
    metric_label = 'Accuracy' if task_type == 'classification' else 'R² Score'

    # 策略名称
    strategy_names = {
        'do_nothing': 'Do Nothing',
        'delete_all': 'Delete All',
        'repair_all': 'Repair All',
        'rl_optimal': 'RL Optimal'
    }

    # 设置x轴位置
    index = np.arange(len(error_rates))  # 每个错误率一个位置

    # 主要关注的策略
    highlighted_strategies = ['rl_optimal', 'repair_all', 'delete_all','do_nothing']

    # 为每个模型绘制条形和折线
    for m_idx, model in enumerate(models):
        model_color = model_colors[m_idx % len(model_colors)]

        # 提取RL策略的性能数据用于折线
        rl_values = [results[model][er]['rl_optimal'] for er in error_rates]

        # 绘制每种策略的条形
        for s_idx, strategy in enumerate(highlighted_strategies):
            perf_values = [results[model][er][strategy] for er in error_rates]

            # 计算条形位置
            offset = (m_idx * len(highlighted_strategies) + s_idx) * bar_width
            positions = index + offset - (len(models) * len(highlighted_strategies) - 1) * bar_width / 2

            # 设置条形样式
            if strategy == 'rl_optimal':
                color = model_color
                hatch = ''
                alpha = 0.9
                edge_color = 'black'
                line_width = 1
                zorder = 10
            elif strategy == 'repair_all':
                color = 'white'
                hatch = '///'
                alpha = 0.7
                edge_color = model_color
                line_width = 1
                zorder = 5
            elif strategy == 'delete_all':
                color = 'white'
                hatch = 'xxx'
                alpha = 0.7
                edge_color = model_color
                line_width = 1
                zorder = 5
            else:
                color = 'white'
                hatch = '...'
                alpha = 0.7
                edge_color = model_color
                line_width = 1
                zorder = 5

            # 绘制条形
            label = f"{model.replace('_', ' ').title()} - {strategy_names[strategy]}" if s_idx == 0 else None
            bars = ax.bar(
                positions,
                perf_values,
                bar_width * 0.9,
                alpha=alpha,
                color=color,
                hatch=hatch,
                edgecolor=edge_color,
                linewidth=line_width,
                label=label,
                zorder=zorder
            )

            # 为条形添加数值标签
            for i, v in enumerate(perf_values):
                # 设置标签显示阈值
                if v > 0.3:  # 根据数据调整
                    ax.text(
                        positions[i],
                        v + 0.01,
                        f"{v:.2f}",
                        ha='center',
                        va='bottom',
                        fontsize=7,
                        color='black',
                        fontweight='bold' if strategy == 'rl_optimal' else 'normal',
                        rotation=90
                    )

        # 添加趋势线 - 只为RL策略添加
        if len(error_rates) > 2:  # 只有当点足够多时才添加趋势线
            trend_positions = index + (m_idx * len(highlighted_strategies)) * bar_width
            ax.plot(
                trend_positions,
                rl_values,
                color=model_color,
                linestyle='-',
                alpha=0.5,
                linewidth=1.5,
                zorder=15
            )

    # 标注容忍度阈值点
    for m_idx, model in enumerate(models):
        # 找到容忍度下降到0.8的点
        threshold = 0.8
        tolerance_array = np.array(tolerances[model])
        threshold_idx = np.abs(tolerance_array - threshold).argmin()
        threshold_er = error_rates[threshold_idx]
        threshold_perf = results[model][threshold_er]['rl_optimal']

        # 找到对应的x轴位置
        er_idx = error_rates.index(threshold_er)
        bar_offset = (m_idx * len(highlighted_strategies)) * bar_width
        x_pos = index[er_idx] + bar_offset - (len(models) * len(highlighted_strategies) - 1) * bar_width / 2

        # 标注阈值点
        ax.scatter(
            [x_pos],
            [threshold_perf + 0.05],
            s=80,
            marker='v',
            facecolors=model_colors[m_idx % len(model_colors)],
            edgecolors='black',
            linewidth=1.5,
            zorder=20
        )
        ax.text(
            x_pos,
            threshold_perf + 0.06,
            f"T={threshold:.1f}",
            color=model_colors[m_idx % len(model_colors)],
            ha='center',
            fontsize=8,
            fontweight='bold'
        )

    # 设置图表属性
    ax.set_xlabel('Error Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
    ax.set_title('Strategy Performance by Model and Error Rate', fontsize=14, fontweight='bold')

    # 设置x轴刻度位置和标签
    ax.set_xticks(index)
    ax.set_xticklabels([f"{er:.1f}" for er in error_rates], fontsize=10)

    # 计算合适的y轴范围
    all_values = []
    for model in results:
        for er in results[model]:
            all_values.extend(results[model][er].values())

    y_min = max(0, min(all_values) - 0.05)
    y_max = min(1.0, max(all_values) + 0.15)  # 增加足够空间用于标签
    ax.set_ylim(y_min, y_max)

    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.set_axisbelow(True)  # 网格线置于图形元素之下

    # 设置刻度标签字体
    ax.tick_params(axis='both', which='major', labelsize=10)

    # 创建图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = ax.legend(
        by_label.values(),
        by_label.keys(),
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=len(models),
        fontsize=9,
        framealpha=0.8
    )

    # 添加策略图例
    rl_patch = mpatches.Patch(color='gray', label='RL Optimal')
    repair_patch = mpatches.Patch(facecolor='white', hatch='///', edgecolor='gray', label='Repair All')
    delete_patch = mpatches.Patch(facecolor='white', hatch='xxx', edgecolor='gray', label='Delete All')
    nothing_patch = mpatches.Patch(facecolor='white', hatch='...', edgecolor='gray', label='DoNothing')
    strategy_legend = ax.legend(
        handles=[rl_patch, repair_patch,delete_patch,nothing_patch],
        loc='upper right',
        title='Strategies',
        fontsize=9,
        framealpha=0.8,
        title_fontsize=10
    )

    # 确保两个图例都显示
    ax.add_artist(legend)
    # 调整 x 轴范围以减少左右空隙
    total_groups = len(models) * len(highlighted_strategies)
    total_bar_span = total_groups * bar_width
    left_bound = index[0] - total_bar_span / 2
    right_bound = index[-1] + total_bar_span / 2 + bar_width

    # 留出一点边距
    ax.set_xlim(left_bound - 0.05, right_bound + 0.05)

    plt.tight_layout(rect=[0, 0.03, 1, 1])  # [left, bottom, right, top]

    return fig, ax


# 10. 绘制性能对比图函数
# def plot_performance_comparison(results, tolerances, models, error_rates, task_type, enhanced=True):
#     """绘制多错误率策略性能对比图"""
#     fig, ax = plt.subplots(figsize=(7, 4.5))
#
#     # 不同策略的线型
#     strategies = ['do_nothing', 'delete_all', 'repair_all', 'rl_optimal']
#     strategy_names = ['Do Nothing', 'Delete All', 'Repair All', 'RL Optimal']
#     line_styles = [':', '--', '-.', '-']
#     marker_styles = ['o', 's', '^', 'D']
#
#     # 使用对比度更高的颜色
#     if enhanced:
#         colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 更鲜明的颜色
#     else:
#         colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
#
#     # 性能指标标签
#     metric_label = 'Accuracy' if task_type == 'classification' else 'R² Score'
#
#     # 绘制性能折线图
#     for m_idx, model in enumerate(models):
#         for s_idx, strategy in enumerate(strategies):
#             # 收集数据点
#             perf_values = [results[model][er][strategy] for er in error_rates]
#
#             # 设置线条样式
#             line_style = line_styles[s_idx]
#             marker = marker_styles[s_idx]
#             color = colors[m_idx % len(colors)]
#
#             # 根据策略重要性调整线条样式
#             if strategy in ['rl_optimal', 'repair_all']:
#                 label = f"{model.replace('_', ' ').title()} - {strategy_names[s_idx]}" if m_idx == 0 else None
#                 # 更明显的线条
#                 lw = 2.0 if strategy == 'rl_optimal' else 1.5
#                 alpha = 1.0
#                 zorder = 10
#             else:
#                 label = f"{strategy_names[s_idx]}" if m_idx == 0 else None
#                 # 更细的线条
#                 lw = 1.0
#                 alpha = 0.6
#                 zorder = 5
#
#             # 绘制折线 - 增强线条可见性
#             line = ax.plot(error_rates, perf_values, marker=marker, linestyle=line_style,
#                            color=color, linewidth=lw, markersize=5, label=label, alpha=alpha, zorder=zorder)
#
#             # 在终点为RL策略添加标签，增加文本可见性
#             if strategy == 'rl_optimal':
#                 ax.text(error_rates[-1] + 0.02, perf_values[-1],
#                         f"{model.replace('_', ' ').title()}", color=color,
#                         fontsize=9, fontweight='bold', ha='left', va='center')
#
#     # 标注关键容忍度阈值点 - 增强可见性
#     for m_idx, model in enumerate(models):
#         # 找到容忍度下降到0.8的点
#         threshold = 0.8
#         tolerance_array = np.array(tolerances[model])
#         # 找到最接近阈值的错误率位置
#         threshold_idx = np.abs(tolerance_array - threshold).argmin()
#         threshold_er = error_rates[threshold_idx]
#         threshold_perf = results[model][threshold_er]['rl_optimal']
#
#         # 标注阈值点 - 使用更大更明显的标记
#         ax.scatter([threshold_er], [threshold_perf], s=100,
#                    facecolors='none', edgecolors=colors[m_idx % len(colors)],
#                    linewidth=2.0, zorder=15)
#         ax.text(threshold_er, threshold_perf - 0.04,
#                 f"T={threshold:.1f}", color=colors[m_idx % len(colors)],
#                 ha='center', fontsize=9, fontweight='bold')
#
#     # 设置图表属性 - 增强标题和标签
#     ax.set_xlabel('Error Rate', fontsize=12, fontweight='bold')
#     ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
#     ax.set_title('Strategy Performance vs Error Rate', fontsize=14, fontweight='bold')
#
#     # 计算合适的y轴范围
#     all_values = []
#     for model in results:
#         for er in results[model]:
#             all_values.extend(results[model][er].values())
#
#     y_min = max(0, min(all_values) - 0.05)
#     y_max = min(1.0, max(all_values) + 0.05)
#
#     # 增加额外空间，确保所有标签可见
#     ax.set_ylim(y_min, y_max + 0.05)
#     ax.set_xlim(min(error_rates) - 0.02, max(error_rates) + 0.07)  # 增加右侧空间
#
#     # 添加网格线 - 增强对比度
#     ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
#
#     # 设置刻度标签字体
#     ax.tick_params(axis='both', which='major', labelsize=10)
#
#     # 创建分组图例 - 增强可见性
#     # 1. 策略图例
#     strategy_lines = []
#     for i, name in enumerate(strategy_names):
#         if name in ['RL Optimal', 'Repair All']:
#             line = plt.Line2D([0], [0], color='black', linestyle=line_styles[i],
#                               marker=marker_styles[i], markersize=5,
#                               linewidth=2.0 if name == 'RL Optimal' else 1.5,
#                               label=name)
#         else:
#             line = plt.Line2D([0], [0], color='black', linestyle=line_styles[i],
#                               marker=marker_styles[i], markersize=5,
#                               linewidth=1.0, alpha=0.6, label=name)
#         strategy_lines.append(line)
#
#     # 将策略图例放在左下角，增加透明度
#     l1 = ax.legend(handles=strategy_lines, loc='lower left',
#                    title='Strategies', fontsize=9, framealpha=0.8,
#                    title_fontsize=10)
#
#     # 2. 模型图例 (放在右上角)
#     model_patches = [mpatches.Patch(color=colors[i % len(colors)],
#                                     label=model.replace('_', ' ').title(),
#                                     alpha=0.9)
#                      for i, model in enumerate(models)]
#
#     # 增加模型图例的可见性
#     l2 = ax.legend(handles=model_patches, loc='upper right',
#                    title='Models', fontsize=9, framealpha=0.8,
#                    title_fontsize=10)
#
#     # 确保两个图例都显示
#     ax.add_artist(l1)
#
#     plt.tight_layout(pad=1.1)  # 增加边距，确保所有元素可见
#
#     return fig, ax


# 11. 结果分析函数
def analyze_results(results, action_dist, tolerances, models, error_rates, task_type):
    """分析实验结果并打印关键量化数据"""
    print("\n===== 关键量化结果 =====")

    # 1. 低错误率场景下的动作分布
    low_error = min(error_rates)
    print(f"\n1. 低错误率({low_error:.1f})下的动作分布:")
    for model in models:
        no_action = action_dist[model][low_error]['no_action'] * 100
        repair = action_dist[model][low_error]['repair'] * 100
        delete = action_dist[model][low_error]['delete'] * 100
        print(f"   {model}: 不作为={no_action:.1f}%, 修复={repair:.1f}%, 删除={delete:.1f}%")

    # 2. 高错误率场景下的动作分布
    high_error = max(error_rates)
    print(f"\n2. 高错误率({high_error:.1f})下的动作分布:")
    for model in models:
        no_action = action_dist[model][high_error]['no_action'] * 100
        repair = action_dist[model][high_error]['repair'] * 100
        delete = action_dist[model][high_error]['delete'] * 100
        print(f"   {model}: 不作为={no_action:.1f}%, 修复={repair:.1f}%, 删除={delete:.1f}%")

    # 3. 模型容忍度比较
    print("\n3. 不同模型的平均容忍度:")
    for model in models:
        avg_tolerance = np.mean(tolerances[model]) * 100
        print(f"   {model}: 平均容忍度={avg_tolerance:.1f}%")

    # 4. RL策略与全部修复的性能比较
    print("\n4. RL策略与全部修复的性能比较:")
    for model in models:
        avg_rl = np.mean([results[model][er]['rl_optimal'] for er in error_rates]) * 100
        avg_repair = np.mean([results[model][er]['repair_all'] for er in error_rates]) * 100
        ratio = avg_rl / avg_repair * 100
        print(f"   {model}: RL={avg_rl:.1f}%, 全部修复={avg_repair:.1f}%, 比例={ratio:.1f}%")

    # 5. 动作成本效益分析
    print("\n5. 动作成本效益分析:")
    for model in models:
        # 计算平均修复成本节省 (与全部修复相比)
        avg_repair_saved = np.mean([1 - action_dist[model][er]['repair'] for er in error_rates]) * 100
        # 计算平均删除成本节省 (与全部删除相比)
        avg_delete_saved = np.mean([1 - action_dist[model][er]['delete'] for er in error_rates]) * 100
        # 平均性能比例
        avg_perf_ratio = np.mean([results[model][er]['rl_optimal'] / results[model][er]['repair_all']
                                  for er in error_rates]) * 100
        print(f"   {model}: 减少修复={avg_repair_saved:.1f}%, 减少删除={avg_delete_saved:.1f}%, "
              f"性能保持={avg_perf_ratio:.1f}%")

    # 6. 错误率阈值分析
    print("\n6. 错误率阈值分析:")
    for model in models:
        # 找到性能下降超过10%的错误率阈值
        baseline_perf = results[model][min(error_rates)]['repair_all']
        threshold_er = max(error_rates)
        for er in error_rates:
            current_perf = results[model][er]['rl_optimal']
            perf_drop = (baseline_perf - current_perf) / baseline_perf
            if perf_drop > 0.1:  # 性能下降超过10%
                threshold_er = er
                break
        print(f"   {model}: 性能显著下降阈值={threshold_er:.1f}")

    # 7. 不同模型对不同错误类型的敏感度分析
    print("\n7. 模型敏感度分析:")
    sensitivity = {}
    for model in models:
        # 错误率增加时，RL修复操作增加的速率
        low_repair = action_dist[model][min(error_rates)]['repair']
        high_repair = action_dist[model][max(error_rates)]['repair']
        repair_sensitivity = (high_repair - low_repair) / (max(error_rates) - min(error_rates))

        # 错误率增加时，性能下降的速率
        low_perf = results[model][min(error_rates)]['rl_optimal']
        high_perf = results[model][max(error_rates)]['rl_optimal']
        perf_sensitivity = (low_perf - high_perf) / (max(error_rates) - min(error_rates))

        sensitivity[model] = {
            'repair_sensitivity': repair_sensitivity,
            'perf_sensitivity': perf_sensitivity
        }

        print(f"   {model}: 修复敏感度={repair_sensitivity:.2f}, 性能敏感度={perf_sensitivity:.2f}")

    # 8. 模型容忍度排名
    model_tolerance_avg = {model: np.mean(tolerances[model]) for model in models}
    sorted_models = sorted(model_tolerance_avg.items(), key=lambda x: x[1], reverse=True)

    print("\n8. 模型容忍度排名:")
    for rank, (model, tolerance) in enumerate(sorted_models, 1):
        print(f"   第{rank}名: {model} (容忍度={tolerance:.2f})")

    return sensitivity


# 12. 完整实验运行函数
# 12. 完整实验运行函数
# def run_tolerance_experiment(error_rates=[0.1, 0.2, 0.3, 0.4, 0.5],
#                              models=['random_forest', 'svm', 'logistic_regression'],
#                              task_type='classification',
#                              n_samples=1000,
#                              n_features=5,
#                              missing_ratio=0.33,
#                              outlier_ratio=0.33,
#                              noise_ratio=0.34,
#                              enhanced_visuals=True):
#     """
#     运行多错误率多模型容忍度实验并绘制结果图
#
#     参数:
#         error_rates: 要测试的错误率列表
#         models: 要测试的机器学习模型列表
#         task_type: 'classification' 或 'regression'
#         n_samples: 数据集样本数
#         n_features: 数据集特征数
#         missing_ratio: 缺失值错误的比例
#         outlier_ratio: 异常值错误的比例
#         noise_ratio: 噪声错误的比例
#         enhanced_visuals: 是否使用增强的可视化效果
#
#     返回:
#         fig1, fig2, results_data: 两个图形对象和一个包含详细结果的字典
#     """
#     # 初始化结果数据结构
#     results, action_dist, tolerances = setup_experiment(
#         error_rates, models, task_type, n_samples, n_features,
#         missing_ratio, outlier_ratio, noise_ratio)
#
#     # 对每个错误率运行实验
#     for error_rate in error_rates:
#         results, action_dist, tolerances = run_single_error_rate_experiment(
#             error_rate, models, task_type, n_samples, n_features,
#             missing_ratio, outlier_ratio, noise_ratio,
#             results, action_dist, tolerances)
#
#     # 绘制动作分布图
#     fig1, ax1 = plot_action_distribution(
#         action_dist, models, error_rates, enhanced=enhanced_visuals)
#
#     # 绘制性能对比图
#     fig2, ax2 = plot_performance_comparison(
#         results, tolerances, models, error_rates, task_type, enhanced=enhanced_visuals)
#
#     # 分析并输出结果
#     sensitivity = analyze_results(
#         results, action_dist, tolerances, models, error_rates, task_type)
#
#     # 返回图形和完整结果数据
#     results_data = {
#         'results': results,
#         'action_dist': action_dist,
#         'tolerances': tolerances,
#         'sensitivity': sensitivity
#     }
#
#     return fig1, fig2, results_data
#
def run_tolerance_experiment(error_rates=[0.1, 0.2, 0.3, 0.4, 0.5],
                             models=['random_forest', 'svm', 'logistic_regression'],
                             task_type='classification',
                             n_samples=1000,
                             n_features=5,
                             missing_ratio=0.33,
                             outlier_ratio=0.33,
                             noise_ratio=0.34,
                             enhanced_visuals=True):
    """
    运行多错误率多模型容忍度实验并绘制结果图
    """
    # 初始化结果数据结构
    results, action_dist, tolerances = setup_experiment(
        error_rates, models, task_type, n_samples, n_features,
        missing_ratio, outlier_ratio, noise_ratio)

    # 1. 在实验开始时生成训练和测试的干净数据
    print("生成干净的训练和测试数据...")
    train_clean_df = generate_clean_data(task_type=task_type, n_samples=n_samples, n_features=n_features)
    # 计算特征重要性（只需计算一次）
    feature_importance = calculate_feature_importance(train_clean_df, task_type)
    print(f"特征重要性: {feature_importance}")

    # 生成额外的测试数据（确保与训练数据不同）
    test_clean_df = generate_clean_data(task_type=task_type, n_samples=n_samples, n_features=n_features)

    # 对每个错误率运行实验
    for error_rate in error_rates:
        print(f"Running experiment with error rate: {error_rate}")

        # 计算每种错误类型的错误率
        missing_err_rate = error_rate * missing_ratio
        outlier_err_rate = error_rate * outlier_ratio
        noise_err_rate = error_rate * noise_ratio

        # 训练阶段：使用训练数据注入错误
        train_injector = ErrorInjector(train_clean_df.copy())
        train_injector.inject_missing_values(error_rate=missing_err_rate, feature_importance=feature_importance)
        train_injector.inject_outliers(error_rate=outlier_err_rate, feature_importance=feature_importance)
        train_injector.inject_noise(error_rate=noise_err_rate, feature_importance=feature_importance)
        train_df_with_errors = train_injector.df

        # 准备训练数据分割
        X_train_full = train_clean_df.drop('target', axis=1)
        y_train_full = train_clean_df['target']
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=42)

        # 对每个模型运行实验
        for model_name in models:
            print(f"  Testing model: {model_name}")
            # 初始化ML模型
            ml_model = initialize_ml_model(task_type, model_name)

            # 创建训练环境和评估器
            train_env = DataCleaningEnv(train_df_with_errors.copy(), ml_model, train_injector.error_locations,
                                        task_type=task_type, model_type=model_name)

            # 训练RL代理
            agent, _ = train_rl_agent(train_env, train_injector.error_locations)

            # 测试阶段：使用测试数据注入相同程度的错误
            test_injector = ErrorInjector(test_clean_df.copy())
            test_injector.inject_missing_values(error_rate=missing_err_rate, feature_importance=feature_importance)
            test_injector.inject_outliers(error_rate=outlier_err_rate, feature_importance=feature_importance)
            test_injector.inject_noise(error_rate=noise_err_rate, feature_importance=feature_importance)
            test_df_with_errors = test_injector.df

            # 准备测试评估器和环境
            test_evaluator, test_env = setup_test_environment(
                test_clean_df, test_df_with_errors, test_injector, ml_model,
                task_type, model_name)

            # 评估不同策略
            strat_perf = evaluate_strategies(
                task_type, test_df_with_errors,
                test_injector.error_locations, test_evaluator, test_env, agent)

            # 记录结果
            update_results(results, action_dist, tolerances, model_name, error_rate,
                           task_type, strat_perf, test_env, test_injector, agent)

    # 绘制动作分布图
    fig1, ax1 = plot_action_distribution(
        action_dist, models, error_rates, enhanced=enhanced_visuals)

    # 绘制性能对比图
    fig2, ax2 = plot_performance_comparison(
        results, tolerances, models, error_rates, task_type, enhanced=enhanced_visuals)

    # 分析并输出结果
    sensitivity = analyze_results(
        results, action_dist, tolerances, models, error_rates, task_type)

    # 输出完整的结果数据
    print("\n===== 完整实验结果 =====")
    print("\nAction Distribution:")
    import json
    print(json.dumps(action_dist, indent=2, default=str))
    print("\nPerformance Results:")
    print(json.dumps(results, indent=2, default=str))

    # 返回图形和完整结果数据
    results_data = {
        'results': results,
        'action_dist': action_dist,
        'tolerances': tolerances,
        'sensitivity': sensitivity
    }

    return fig1, fig2, results_data


# 示例调用
if __name__ == "__main__":
    # 运行实验并生成图表
    fig1, fig2, results_data = run_tolerance_experiment(
        error_rates=[0.1, 0.3, 0.5, 0.7, 0.9],
        # error_rates=[0.5],
        models=['random_forest', 'svm', 'logistic_regression'],
        task_type='classification',
        n_samples=1000,  # 减少样本量以加快实验速度
        n_features=5,
        enhanced_visuals=True  # 使用增强的可视化效果
    )

    # 保存图表为高质量PDF
    fig1.savefig('action_distribution_enhanced.pdf', bbox_inches='tight', dpi=600)
    fig2.savefig('performance_comparison_enhanced.pdf', bbox_inches='tight', dpi=600)

    # 展示图表
    plt.figure(fig1.number)
    plt.show()
    plt.figure(fig2.number)
    plt.show()

    # 访问详细结果示例
    rf_tolerance = results_data['tolerances']['random_forest']
    svm_actions = results_data['action_dist']['svm']
    lr_performance = results_data['results']['logistic_regression']

    # print(f"\nRandom Forest 容忍度: {rf_tolerance}")
    # print(f"SVM 在错误率 0.7 的动作分布: {svm_actions[0.7]}")
    # print(f"Logistic Regression 在错误率 0.9 的性能: {lr_performance[0.9]}")
