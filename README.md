import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression

# 直接加载数据集
loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

# 定义模型
model = LinearRegression()
# 学习参数
model.fit(data_X, data_y)
# 计算预测值
result = model.predict(data_X)
print(data_y[:4])#前四个真实值
print(result[:4])#前四个预测值
