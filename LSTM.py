from pyspark.sql import SparkSession

import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import Callback
from sklearn.metrics import mean_squared_error


"""
spark = SparkSession.builder.appName('predict_model').getOrCreate()
df = spark.read.csv('1.csv')
df.printSchema()
"""

data = pd.read_csv('1.csv')
ratio = 4/5

data = data[data['country']=='afghanistan'].loc[:, 'confirmed']

# 数据预处理
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.values.reshape(-1,1))

# 划分训练集和测试集
train_size = int(len(data_scaled) * ratio)
train_data, test_data = data_scaled[0:train_size], data_scaled[train_size:]

def create_dataset(dataset, time_steps=1):
    X, y = [], []
    for i in range(len(dataset)-time_steps):
        a = dataset[i:(i+time_steps)]
        X.append(a)
        y.append(dataset[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 10
X_train, y_train = create_dataset(train_data, time_steps)
X_test, y_test = create_dataset(test_data, time_steps)

# 调整输入数据的形状
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 创建Seq2Seq模型
model = Sequential()
model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 创建回调函数来计算准确率
class AccuracyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # 在每个 epoch 结束时计算指标
        y_true = scaler.inverse_transform(y_test)
        y_pred = scaler.inverse_transform(model.predict(X_test))

        print(y_true)
        print('*'*100)
        print(y_pred)

        # 计算均方根误差（RMSE）
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        print(f'\nRMSE on test set: {rmse:.4f}')
        exit()

# 训练模型
accuracy_callback = AccuracyCallback()
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2, callbacks=[accuracy_callback])

# 在测试集上进行预测
predicted_values = model.predict(X_test)

# 反向转换预测值
predicted_values = scaler.inverse_transform(np.reshape(predicted_values, (predicted_values.shape[0], 1)))

y_true = scaler.inverse_transform(y_test)

# 输出真实值和预测值
for true, pred in zip(y_true[:10], predicted_values[:10]):
    print(f'True: {true}, Predicted: {pred[0]}')