import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping

# 忽略用户警告
warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# =============================================
# 数据加载和预处理
# =============================================
# 读取宝可梦信息数据并删除 sprites 列
pokemon_df = pd.read_csv("data/Pokemon/final_pokemon.csv").drop(columns=["sprites"])
# 读取对战数据
combats_df = pd.read_csv("data/Pokemon/final_combats.csv")

# 将 Legendary 列转换为整数类型（允许缺失）
pokemon_df['Legendary'] = pokemon_df['Legendary'].astype('Int64')

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

# =============================================
# 特征工程
# =============================================
# 构建特征和标签
features = [f'{stat}_diff' for stat in stat_cols] + ['Type 1_1', 'Type 1_2', 'Type 2_1', 'Type 2_2']
X = merged_df[features]
y = merged_df['label']

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 对分类变量进行编码
categorical_cols = ['Type 1_1', 'Type 1_2', 'Type 2_1', 'Type 2_2']
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# 定义预处理器：对分类变量使用序数编码，数值特征标准化
preprocessor = ColumnTransformer([
    ('cat', OrdinalEncoder(), categorical_cols),
    ('num', StandardScaler(), numeric_cols)
])

# 预处理数据
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# =============================================
# 神经网络模型定义
# =============================================
def create_nn_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_processed.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# 创建Keras分类器包装器用于交叉验证
nn_model = KerasClassifier(
    build_fn=create_nn_model,
    epochs=100,
    batch_size=64,
    verbose=0,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
)

# =============================================
# 交叉验证评估
# =============================================
# print("\n正在进行交叉验证...")
# nn_acc = cross_val_score(nn_model, X_train_processed, y_train, cv=5, scoring='accuracy')
# print(f"\n[Neural Network] 准确率得分: {nn_acc}")
# print(f"[Neural Network] 平均准确率: {nn_acc.mean():.4f}")
#
# nn_auc = cross_val_score(nn_model, X_train_processed, y_train, cv=5, scoring='roc_auc')
# print(f"\n[Neural Network] ROC AUC 得分: {nn_auc}")
# print(f"[Neural Network] 平均 ROC AUC: {nn_auc.mean():.4f}")

# =============================================
# 完整训练和评估
# =============================================
print("\n训练最终模型...")
final_model = create_nn_model()
history = final_model.fit(
    X_train_processed,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=1
)

# 绘制训练曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('训练和验证准确率')
plt.xlabel('Epoch')
plt.ylabel('准确率')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('训练和验证损失')
plt.xlabel('Epoch')
plt.ylabel('损失')
plt.legend()
plt.tight_layout()
plt.show()

# 在测试集上评估
print("\n测试集评估...")
y_pred = (final_model.predict(X_test_processed) > 0.5).astype(int)
y_proba = final_model.predict(X_test_processed)

# 输出评估指标
print("\n测试集评估结果:")
print("准确率:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("F1 分数:", f1_score(y_test, y_pred))

# 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title('神经网络混淆矩阵')
plt.show()

# 预测概率分布图
plt.figure(figsize=(8, 6))
sns.histplot(y_proba, bins=50, kde=True, color='blue', edgecolor='black')
plt.title('神经网络预测概率分布图')
plt.xlabel('预测为第一宝可梦胜出的概率')
plt.ylabel('频率')
plt.tight_layout()
plt.show()