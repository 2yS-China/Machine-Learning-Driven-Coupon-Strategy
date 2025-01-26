import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import xgboost as xgb
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
# 读取Excel文件
file_path = './data/data.xlsx'
data = pd.read_excel(file_path)

# 数据初始化
data['使用状态'].fillna(2, inplace=True) #将使用状态为空的值重新赋值为-1，代表None,未领取
data['实付金额'] = data['实付金额'].abs() #将实付金额列中的负数转换为其绝对值
# data['使用状态'] = data['使用状态'].replace({1: '已使用', 0: '领取未使用', 2: '未领取'}) #将使用状态列中的数值1和0转换为分类标签'已使用'和'领取未使用'

'''
    第一问，同时显示所有的用户ID在图表中好像不是很现实所以显示前20个，但是将整个统计用户ID，下单次数，消费总额，订单平均金额进行保存
'''
# 统计用户ID，下单次数，消费总额，订单平均金额
# user_statistics = df.groupby('用户ID').agg({
#     '实付金额': ['sum', 'mean'],
#     '订单ID': 'count'
# }).reset_index()
# user_statistics.columns = ['用户ID', '消费金额总额', '订单平均金额', '下单次数']
# # 可视化前100个用户的消费金额总额、订单平均金额和下单次数
# plt.figure(figsize=(15, 8))
# # 消费金额总额
# plt.subplot(3, 1, 1)
# sns.barplot(x='用户ID', y='消费金额总额', data=user_statistics.head(20))
# plt.title('前20个用户的消费金额总额')
# # 订单平均金额
# plt.subplot(3, 1, 2)
# sns.barplot(x='用户ID', y='订单平均金额', data=user_statistics.head(20))
# plt.title('前20个用户的订单平均金额')
# # 下单次数
# plt.subplot(3, 1, 3)
# sns.barplot(x='用户ID', y='下单次数', data=user_statistics.head(20))
# plt.title('前20个用户的下单次数')
# plt.tight_layout()
# plt.show()
#
# # 提取付款日期中的年份和月份
# df['年份'] = df['付款日期'].dt.year
# df['月份'] = df['付款日期'].dt.month
# monthly_statistics = df.groupby(['年份', '月份'])['实付金额'].sum().reset_index()
# plt.figure(figsize=(12, 6))
# sns.barplot(x='月份', y='实付金额', hue='年份', data=monthly_statistics)
# plt.title('每月消费总额统计')
# plt.xlabel('月份')
# plt.ylabel('消费总额')
# plt.show()

#可视化用户地址分布情况和各地址下的消费数量分布情况

# 使用groupby方法按省份、城市和用户ID分组，并计算每个地址下的用户数量
# user_distribution_by_address = df.groupby(['省份', '用户ID']).size().reset_index(name='用户数量')
#
# # 统计每个地址下用户的分布情况
# address_user_distribution = user_distribution_by_address.groupby(['省份']).agg({
#     '用户数量': 'count'
# }).reset_index()
#
# # 可视化每个地址下用户的分布情况
# plt.figure(figsize=(20, 10), dpi=100)
# sns.barplot(x='省份', y='用户数量', data=address_user_distribution, palette='viridis', dodge=False)
# plt.xticks(rotation=45, fontsize=12)
# plt.yticks(fontsize=12)
# plt.title('每个省份下用户的分布情况')
# plt.xlabel('城市')
# plt.ylabel('用户数量')
# plt.show()


# 用户特征
# 消费总金额
# user_sum_spend = data.groupby('用户ID')['实付金额'].sum().reset_index()
# user_sum_spend.columns = ['用户ID', '总消费金额']
# # 平均消费金额
# user_avg_spend = data.groupby('用户ID')['实付金额'].mean().reset_index()
# user_avg_spend.columns = ['用户ID', '平均消费金额']
# # 购买频率
# user_purchase_freq = data.groupby('用户ID').size().reset_index(name='购买频率')
# # 订单数
# user_dingdan = data.groupby('用户ID')['订单ID'].nunique().reset_index()
# user_dingdan.columns = ['用户ID', '订单数']
# # 购买商品数量
# user_nunber = data.groupby('用户ID')['购买数量'].sum().reset_index()
# user_nunber.columns = ['用户ID', '购买商品总数']
# # 购买商品数量平均数量
# user_nunber = data.groupby('用户ID')['购买数量'].mean().reset_index()
# user_nunber.columns = ['用户ID', '购买商品平均数']
# # 商户特征
# # 总销售额
# merchant_total_sales = data.groupby('商家ID')['实付金额'].sum().reset_index()
# merchant_total_sales.columns = ['商家ID', '总销售额']
# # 平均订单金额
# merchant_avg_order = data.groupby('商家ID')['实付金额'].mean().reset_index()
# merchant_avg_order.columns = ['商家ID', '平均订单金额']
#
# # 合并特征
# user_features = pd.merge(user_avg_spend, user_purchase_freq, on='用户ID', how='left')
# user_features = pd.merge(user_sum_spend, user_features, on='用户ID', how='left')




# 用户特征
'''
    第二问
'''
# 用户特征
user_features = data.groupby('用户ID').agg({
    '订单ID': 'nunique',
    '实付金额': ['sum', 'mean'],
    '购买数量': ['sum', 'mean'],
    '邮费': ['sum', 'mean', lambda x: (x == 0).sum()]
})

user_features.columns = ['订单数', '总金额', '平均订单金额', '购买商品总数', '购买商品平均数', '邮费金额总数', '邮费金额平均数', '包邮订单数']
user_features['优惠券使用数量'] = data.groupby('用户ID')['使用状态'].apply(lambda x: (x == 1).sum())
user_features['优惠券使用率'] = data.groupby('用户ID')['使用状态'].apply(lambda x: (x == 1).sum() / len(x))
user_features['邮费占总金额比率'] = user_features['邮费金额总数'] / user_features['总金额']
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# 商户特征
merchant_features = data.groupby('商家ID').agg({
    '订单ID': 'nunique',
    '实付金额': ['sum', 'mean'],
    '购买数量': ['sum', 'mean'],
    '邮费': ['sum', 'mean', lambda x: (x == 0).sum()]
})
merchant_features.columns = ['订单数', '总金额', '平均订单金额', '售出商品总数', '售出商品平均数', '邮费金额总数', '邮费金额平均数', '包邮订单数']
merchant_features['优惠券使用数量'] = data.groupby('商家ID')['使用状态'].apply(lambda x: (x == 1).sum())
merchant_features['优惠券使用率'] = data.groupby('商家ID')['使用状态'].apply(lambda x: (x == 1).sum() / len(x))
merchant_features['邮费占总金额比率'] = merchant_features['邮费金额总数'] / merchant_features['总金额']



# 优惠券特征
# coupon_usage = data.groupby(['用户ID', '使用状态']).size().unstack(fill_value=0)
# coupon_usage['优惠券使用率'] = coupon_usage['已使用'] / (coupon_usage['已使用'] + coupon_usage['领取未使用'] + coupon_usage['未领取'])

# 考虑付款实际，城市
data['付款日期'] = pd.to_datetime(data['付款日期'])
data['年月'] = data['付款日期'].dt.to_period('M')
merchant_time_analysis = data.groupby(['商家ID', '年月']).agg({
    '订单ID': 'nunique',
    '实付金额': ['sum', 'mean'],
    '购买数量': ['sum', 'mean']
}).reset_index()
merchant_time_analysis.columns = ['商家ID', '年月', '订单数', '总金额', '平均订单金额', '售出商品总数', '售出商品平均数']


province_city_analysis = data.groupby(['省份', '城市', '年月']).agg({
    '订单ID': 'nunique',
    '实付金额': ['sum', 'mean'],
    '购买数量': ['sum', 'mean']
}).reset_index()
province_city_analysis.columns = ['省份', '城市', '年月', '订单数', '总金额', '平均订单金额', '商品总数', '商品平均数']

'''
    第三问
'''
reference_date = data['付款日期'].max() + pd.Timedelta(days=1)
rfm = data.groupby('用户ID').agg({
    '付款日期': lambda x: (reference_date - x.max()).days,
    '订单ID': 'nunique',
    '实付金额': 'sum'
}).reset_index()

rfm.columns = ['用户ID', 'Recency', 'Frequency', 'Monetary']

# Ensuring that the RFM metrics are numeric
rfm['Recency'] = pd.to_numeric(rfm['Recency'], errors='coerce')
rfm['Frequency'] = pd.to_numeric(rfm['Frequency'], errors='coerce')
rfm['Monetary'] = pd.to_numeric(rfm['Monetary'], errors='coerce')
#

quantiles = rfm[['Recency', 'Frequency', 'Monetary']].quantile(q=[0.25, 0.5, 0.75])

def rfm_score(x, p, d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]:
        return 3
    else:
        return 4

rfm['R_Score'] = rfm['Recency'].apply(rfm_score, args=('Recency', quantiles))
rfm['F_Score'] = rfm['Frequency'].apply(rfm_score, args=('Frequency', quantiles))
rfm['M_Score'] = rfm['Monetary'].apply(rfm_score, args=('Monetary', quantiles))
rfm['RFM_Segment'] = rfm['R_Score'].map(str) + rfm['F_Score'].map(str) + rfm['M_Score'].map(str)
rfm['RFM_Score'] = rfm[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)

# 识别高价值和低价值客户
high_value_customers = rfm[rfm['RFM_Score'] >= rfm['RFM_Score'].quantile(0.75)]
mid_value_customers = rfm[(rfm['RFM_Score'] > rfm['RFM_Score'].quantile(0.25)) & (rfm['RFM_Score'] < rfm['RFM_Score'].quantile(0.75))]
low_value_customers = rfm[rfm['RFM_Score'] <= rfm['RFM_Score'].quantile(0.25)]
print("高价值用户：")
print(high_value_customers.head())
print("潜在发展用户：")
print(mid_value_customers.head())
print("需关注用户：")
print(low_value_customers.head())
#
# '''
#     第四问
# '''

data['使用状态'].fillna(2, inplace=True)

data['付款日期'] = pd.to_datetime(data['付款日期'])
data['年月'] = data['付款日期'].dt.to_period('M')
data['年'] = data['年月'].dt.year
data['月'] = data['年月'].dt.month
data.drop('年月', axis=1, inplace=True)

data['付款日期'] = pd.to_datetime(data['付款日期'])
data['付款日期'] = data['付款日期'].astype(np.int64)

label_encoder = LabelEncoder()
for column in ['省份', '城市']:
    data[column] = label_encoder.fit_transform(data[column])


X = data.drop(['订单ID', '用户ID', '使用状态'], axis=1)
y = data['使用状态']

#划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 随机森林
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)

#
# 获取概率预测而不是简单分类
y_pred_proba = model.predict_proba(X_test)[:, 1]  # 获取测试数据中的用户使用优惠券的概率

# 将预测概率与测试集相对应的行合并
test_data = X_test.copy()
test_data['predicted_probability'] = y_pred_proba

# 将预测概率分箱
bins = np.linspace(0, 1, 11)
test_data['probability_bin'] = pd.cut(test_data['predicted_probability'], bins, labels=range(10), include_lowest=True)

# 为每个箱子设定优惠券发放比例
coupon_allocation = {i: i * 10 for i in range(10)}

# 分配优惠券的函数
def allocate_coupons(probability_bin, allocation):
    # 处理NaN值
    if pd.isna(probability_bin):
        return False
    threshold = allocation[probability_bin]
    return np.random.rand() < threshold / 100

test_data['receive_coupon'] = test_data['probability_bin'].apply(lambda x: allocate_coupons(x, coupon_allocation))

# 打印分析报告
print("优惠券发放策略：")
print(test_data[['probability_bin', 'receive_coupon']])




# XGBoost模型
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# 模型评估
y_pred = xgb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score for each class: {f1}")

DNN模型


data['使用状态'].fillna(2, inplace=True)
data['年'] = data['付款日期'].dt.year
data['月'] = data['付款日期'].dt.month
data.drop(['年月'], axis=1, inplace=True, errors='ignore')
data['付款日期'] = pd.to_datetime(data['付款日期']).view(np.int64)

# 标签编码
label_encoder = LabelEncoder()
for column in ['省份', '城市']:
    data[column] = label_encoder.fit_transform(data[column])

# 特征和目标变量
X = data.drop(['订单ID', '用户ID', '使用状态'], axis=1)
y = label_encoder.fit_transform(data['使用状态'])

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#构建PyTorch Dataset
class CouponDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = CouponDataset(X_train, y_train)
test_dataset = CouponDataset(X_test, y_test)

#构建DNN模型
class DNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = DNN(X_train.shape[1], len(np.unique(y)))

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
def train_model(model, train_dataset, criterion, optimizer, epochs=10):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for epoch in range(epochs):
        for features, labels in train_loader:
            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

train_model(model, train_dataset, criterion, optimizer, epochs=10)

# 评估模型
def evaluate_model(model, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    y_true = []
    y_pred = []
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.numpy())
            y_true.extend(labels.numpy())
    return y_true, y_pred

y_true, y_pred = evaluate_model(model, test_dataset)

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 计算评估指标
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')
report = classification_report(y_test, y_pred)

# 对于多分类问题，AUC的计算需要将标签二值化
y_true_binarized = label_binarize(y_true, classes=np.unique(y_true))
y_pred_binarized = label_binarize(y_pred, classes=np.unique(y_true))

# 计算每个类的AUC，并计算平均值
n_classes = y_true_binarized.shape[1]
auc = roc_auc_score(y_true_binarized, y_pred_binarized, average='macro', multi_class='ovr')

# 打印评估结果
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
print(f"Confusion Matrix:\n{cm}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score for each class: {f1}")
