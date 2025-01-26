import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import numpy as np
# 读取数据
file_path = './data/data.xlsx'
data = pd.read_excel(file_path)
print(data[:10])
data['使用状态'].fillna(2, inplace=True)
data['付款日期'] = pd.to_datetime(data['付款日期'])
data['年月'] = data['付款日期'].dt.to_period('M')
data['实付金额'] = data['实付金额'].abs()
label_encoder = LabelEncoder()
for column in ['省份', '城市']:
    data[column] = label_encoder.fit_transform(data[column])
# 用户维度分析
# 计算每个用户的消费总额
user_total_spending = data.groupby('用户ID')['实付金额'].sum()
# 计算每个用户的平均订单金额
average_order_amount = data.groupby('用户ID')['实付金额'].mean()
# 地域维度分析
# 各省份、城市的消费金额和消费数量
province_spending = data.groupby('省份')['实付金额'].sum()
city_spending = data.groupby(['省份', '城市'])['实付金额'].sum()
province_order_count = data.groupby('省份').size()
city_order_count = data.groupby(['省份', '城市']).size()
# 订单维度分析
# 每笔订单的购买数量分布
order_quantity_distribution = data['购买数量'].value_counts()
# 实付金额分布
payment_distribution = data['实付金额'].value_counts().nlargest(100)
# 邮费分布（包邮比例）
free_shipping_ratio = (data['邮费'] == 0).mean()
# 优惠券使用率
coupon_usage_rate = (data['使用状态'] == 1).mean()
coupon_usage = data['使用状态'].value_counts()
plt.figure(figsize=(10, 10))
coupon_usage.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title('优惠券使用率')
plt.ylabel('')
plt.show()
# 商品维度分析
# 各商家销售情况
merchant_sales = data.groupby('商家ID')['实付金额'].sum().nlargest(100)
merchant_order_count = data.groupby('商家ID').size().nlargest(100)
# 可视化示例：用户消费总额分布图
plt.rcParams["font.sans-serif"] = ["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"] = False #该语句解决图像中的“-”负号的乱码问题
data['付款日期'] = pd.to_datetime(data['付款日期']).dt.date
sales_data = data.groupby('付款日期')['购买数量'].sum().reset_index()
plt.figure(figsize=(10, 6))
plt.plot(sales_data['付款日期'], sales_data['购买数量'])
plt.title('订单销量随付款日期的变化')
plt.xlabel('付款日期')
plt.ylabel('销量')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
# #
plt.figure(figsize=(20, 10), dpi=100)
user_total_spending.hist(bins=50)
plt.title('用户消费总额分布')
plt.xlabel('消费总额')
plt.ylabel('用户数量')
plt.show()
# 用户平均订单金额分布
plt.figure(figsize=(20, 10), dpi=100)
average_order_amount.hist(bins=50)
plt.title('用户平均订单金额分布')
plt.xlabel('平均订单金额')
plt.ylabel('用户数量')
plt.show()

# # 省份消费金额分布
plt.figure(figsize=(20, 10), dpi=100)
province_spending.plot(kind='bar')
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.title('省份消费金额分布')
plt.xlabel('省份')
plt.ylabel('消费金额')
plt.show()

# 省份消费数量分布
plt.figure(figsize=(20, 10), dpi=100)
province_order_count.plot(kind='bar')
plt.title('省份消费数量分布')
plt.xlabel('省份')
plt.ylabel('订单数量')
plt.xticks(rotation=45)
plt.show()
# 订单维度分析的可视化
#购买数量分布
plt.figure(figsize=(20, 10))
order_quantity_distribution.plot(kind='bar')
plt.title('购买数量分布')
plt.xlabel('购买数量')
plt.ylabel('订单数量')
plt.show()
#
# # 实付金额分布
plt.figure(figsize=(20, 10))
payment_distribution.plot(kind='bar')
plt.title('实付金额分布')
plt.xlabel('实付金额')
plt.ylabel('订单数量')
plt.show()
#
# # 邮费分布
plt.figure(figsize=(20, 10))
data['邮费'].value_counts().plot(kind='bar')
plt.title('邮费分布')
plt.xlabel('邮费')
plt.ylabel('订单数量')
plt.show()
#
# # 优惠券使用率
coupon_usage = data['使用状态'].value_counts()
plt.figure(figsize=(10, 10))
coupon_usage.plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title('优惠券使用率')
plt.ylabel('')
plt.show()
#
# 商品维度分析的可视化
# 商家销售情况
plt.figure(figsize=(10, 6))
merchant_sales.plot(kind='bar')
plt.title('商家销售情况')
plt.xlabel('商家ID')
plt.ylabel('销售额')
plt.xticks(rotation=90)
plt.show()

# 商品维度分析的可视化
# 商家销售情况
plt.figure(figsize=(20, 10))
merchant_sales.plot(kind='bar')
plt.title('商家销售情况')
plt.xlabel('商家ID')
plt.ylabel('销售额')
plt.xticks(rotation=90)
plt.show()
#
# # 商家订单数量
plt.figure(figsize=(20, 10))
merchant_order_count.plot(kind='bar')
plt.title('商家订单数量')
plt.xlabel('商家ID')
plt.ylabel('订单数量')
plt.xticks(rotation=90)
plt.show()

# # # 箱线图：每个用户的消费金额分布
plt.figure(figsize=(20, 10))
sns.boxplot(x=data['用户ID'], y=data['实付金额'])
plt.title('每个用户的消费金额分布')
plt.xlabel('') # 隐藏x轴标签
plt.ylabel('实付金额')
plt.xticks(rotation=90)  # 如果用户ID较多，可以旋转标签以便阅读
plt.show()
#
# # 饼图：不同省份的用户数量占比
province_counts = data.drop_duplicates(subset=['用户ID', '省份'])['省份'].value_counts()
plt.figure(figsize=(10, 6))
province_counts.plot(kind='pie', autopct='%1.1f%%')
plt.title('不同省份的用户数量占比')
plt.ylabel('')  # 隐藏y轴标签
plt.show()
#
# # 热图：不同省份和城市的消费金额分布
province_city_spending = data.pivot_table(index='省份', columns='城市', values='实付金额', aggfunc='sum')
plt.figure(figsize=(20, 10))
sns.heatmap(province_city_spending, annot=True, fmt='.0f')
plt.title('不同省份和城市的消费金额分布')
plt.xlabel('城市')
plt.ylabel('省份')
plt.show()

