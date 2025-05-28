import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 忽略用户警告
warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 读取宝可梦信息数据并删除 sprites 列
pokemon_df = pd.read_csv("data/Pokemon/final_pokemon.csv").drop(columns=["sprites"])
# 读取对战数据
combats_df = pd.read_csv("data/Pokemon/final_combats.csv")

# 将 Legendary 列转换为整数类型（允许缺失）
pokemon_df['Legendary'] = pokemon_df['Legendary'].astype('Int64')

# 打印数据前几行
print(pokemon_df.head())
print(combats_df.head())

# 绘制世代分布图
plt.figure(figsize=(4, 3))
sns.histplot(pokemon_df['Generation'], bins=range(pokemon_df['Generation'].min(), pokemon_df['Generation'].max() + 2), kde=False)
plt.title('宝可梦世代分布')
plt.xlabel('世代')
plt.ylabel('数量')
plt.xticks(range(pokemon_df['Generation'].min(), pokemon_df['Generation'].max() + 1))
plt.tight_layout()
plt.show()

# 绘制主类型和副类型的饼图
pokemon_df['Type 1'].value_counts().plot(kind='pie', title='主类型分布')
pokemon_df['Type 2'].value_counts().plot(kind='pie', title='副类型分布')

# 创建类型组合的交叉表并绘制热力图
heatmap_data = pd.crosstab(pokemon_df['Type 1'], pokemon_df['Type 2'])

plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="coolwarm", cbar_kws={'label': '数量'})
plt.title('宝可梦类型组合热力图')
plt.ylabel('主类型')
plt.xlabel('副类型')
plt.tight_layout()
plt.show()

# 计算数值属性之间的相关性矩阵并可视化
int_columns = pokemon_df.select_dtypes(include='int64').drop(columns=['#', 'Generation', 'Legendary'])
corr_matrix = int_columns.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("宝可梦属性相关性矩阵")
plt.tight_layout()
plt.show()

# 绘制各数值属性的分布直方图及中位数、四分位数线
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))
axes = axes.flatten()

for i, col in enumerate(int_columns.columns):
    ax = axes[i]
    sns.histplot(int_columns[col], bins=40, edgecolor='black', ax=ax, kde=True)
    median = int_columns[col].median()
    q25 = int_columns[col].quantile(0.25)
    q75 = int_columns[col].quantile(0.75)
    ax.axvline(median, color='red', linestyle='--', label='中位数')
    ax.axvline(q25, color='green', linestyle=':', label='25% 分位数')
    ax.axvline(q75, color='blue', linestyle=':', label='75% 分位数')
    ax.set_title(col)
    ax.legend()

plt.suptitle("宝可梦属性分布", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# 示例：找出与编号 1 和 558 对战的记录（不同胜者的情况）
combats_df[
    ((combats_df["First_pokemon"] == 1) | (combats_df["Second_pokemon"] == 1)) &
    ((combats_df["First_pokemon"] == 558) | (combats_df["Second_pokemon"] == 558))
]

# 重命名属性列以便后续合并区分
pokemon1 = pokemon_df.add_suffix('_1')
pokemon2 = pokemon_df.add_suffix('_2')

# 合并第一只和第二只宝可梦的数据
merged_df_1 = combats_df.merge(pokemon1, left_on='First_pokemon', right_on='#_1')
merged_df = merged_df_1.merge(pokemon2, left_on='Second_pokemon', right_on='#_2')

# 创建标签列：第一只宝可梦是否胜出
merged_df['label'] = (merged_df['Winner'] == merged_df['First_pokemon']).astype('int64')

# 计算各数值属性的差值并删除原始列
stat_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', "Legendary", 'height', 'weight',
             'base_experience']
for stat in stat_cols:
    merged_df[f'{stat}_diff'] = merged_df[f'{stat}_1'] - merged_df[f'{stat}_2']
    merged_df.drop(columns=[f'{stat}_1', f'{stat}_2'], inplace=True)

# 将类型列转换为分类类型
for col in ['Type 1_1', 'Type 1_2', 'Type 2_1', 'Type 2_2']:
    merged_df[col] = merged_df[col].astype('category')

# 查看合并后数据的信息
merged_df.info()

from sklearn.model_selection import train_test_split

# 构建特征和标签
features = [f'{stat}_diff' for stat in stat_cols] + ['Type 1_1', 'Type 1_2', 'Type 2_1', 'Type 2_2']
X = merged_df[features]
y = merged_df['label']

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 打印标签的均值（胜率）
print(y_train.mean())
print(y_test.mean())

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

# 对分类变量进行编码（Logistic 回归不支持 category 类型）
categorical_cols = ['Type 1_1', 'Type 1_2', 'Type 2_1', 'Type 2_2']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# 定义预处理器：对分类变量使用序数编码，其余数值特征直接通过
preprocessor = ColumnTransformer([
    ('cat', OrdinalEncoder(), categorical_cols),
    ('num', 'passthrough', numeric_cols)
])

# 构建管道：预处理 → 逻辑回归
logreg_pipeline = make_pipeline(
    preprocessor,
    LogisticRegression(max_iter=1000)
)

# 使用交叉验证评估逻辑回归的准确率
logreg_acc = cross_val_score(logreg_pipeline, X, y, cv=5, scoring='accuracy')
print(f"\n[LogReg] 准确率得分: {logreg_acc}")
print(f"[LogReg] 平均准确率: {logreg_acc.mean():.4f}")

# 使用交叉验证评估 ROC AUC
logreg_auc = cross_val_score(logreg_pipeline, X, y, cv=5, scoring='roc_auc')
print(f"\n[LogReg] ROC AUC 得分: {logreg_auc}")
print(f"[LogReg] 平均 ROC AUC: {logreg_auc.mean():.4f}")

from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score, roc_curve

# 构建 XGBoost 模型
baseline_model = XGBClassifier(enable_categorical=True, use_label_encoder=False, eval_metric='logloss',
                      random_state=42, n_jobs=1,
                      tree_method='hist', deterministic=True)
baseline_model.fit(X_train, y_train)

# 模型预测
y_pred = baseline_model.predict(X_test)
y_proba = baseline_model.predict_proba(X_test)[:, 1]

# 输出评估指标
print("准确率:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("F1 分数:", f1_score(y_test, y_pred))

# 交叉验证评估准确率
accuracy_scores = cross_val_score(baseline_model, X, y, cv=5, scoring='accuracy')
print(f"\n交叉验证准确率得分: {accuracy_scores}")
print(f"平均准确率: {accuracy_scores.mean():.4f}")

# 交叉验证评估 AUC
auc_scores = cross_val_score(baseline_model, X, y, cv=5, scoring='roc_auc')
print(f"\n交叉验证 ROC AUC 得分: {auc_scores}")
print(f"平均 ROC AUC: {auc_scores.mean():.4f}")

# 特征重要性可视化
plot_importance(baseline_model)
plt.figure(figsize=(8, 6))
sns.histplot(y_proba, bins=50, kde=True, color='blue', edgecolor='black')
plt.title('预测概率分布图 (y_proba)')
plt.xlabel('预测为第一宝可梦胜出的概率')
plt.ylabel('频率')
plt.tight_layout()
plt.show()

# 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
