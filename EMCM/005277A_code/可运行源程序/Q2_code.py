import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 1. 读取训练集和测试集
train_df = pd.read_csv(r'C:\Users\58407\Desktop\Q2_train_data.csv', parse_dates=['utc_timestamp'])
test_df = pd.read_csv(r'C:\Users\58407\Desktop\Q2_test_data.csv', parse_dates=['utc_timestamp'])

# 2. 确保时间排序并设置索引
train_df = train_df.sort_values('utc_timestamp').set_index('utc_timestamp')
test_df = test_df.sort_values('utc_timestamp').set_index('utc_timestamp')

# 3. 提取目标列（假设列名是 'power'）
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df[['AT_solar_generation_actual']])
test_scaled = scaler.transform(test_df[['AT_solar_generation_actual']])

# 4. 构造序列数据
SEQ_LEN = 96  # 1天数据（15分钟间隔）
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, SEQ_LEN)
X_test, y_test = create_sequences(test_scaled, SEQ_LEN)

X_train = X_train.reshape((-1, SEQ_LEN, 1))
X_test = X_test.reshape((-1, SEQ_LEN, 1))

# 5. 构建 LSTM 模型
model = Sequential([
    LSTM(64, input_shape=(SEQ_LEN, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# 6. 预测并反归一化
y_pred = model.predict(X_test)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler.inverse_transform(y_pred)

# 7. 构建表2格式
start_datetime = test_df.index[SEQ_LEN]  # 起报第一时刻
interval = timedelta(minutes=15)
rows = []

# 预测共7天，每天96个点
for day in range(7):
    issue_time = (start_datetime + timedelta(days=day)).replace(hour=0, minute=0)
    forecast_start = issue_time + timedelta(days=1)
    for i in range(96):
        idx = day * 96 + i
        forecast_time = forecast_start + i * interval
        actual = y_test_inv[idx][0]
        pred = y_pred_inv[idx][0]
        rows.append([
            issue_time.strftime('%Y/%m/%d/%H:%M'),
            forecast_time.strftime('%Y/%m/%d/%H:%M'),
            round(actual, 2),
            round(pred, 2)
        ])

df_result = pd.DataFrame(rows, columns=['起报时间', '预报时间', '实际功率(MW)', '预测功率(MW)'])
df_result.to_csv('功率预测结果_表2格式.csv', index=False, encoding='utf-8-sig')
print(" 表2格式已保存：功率预测结果_表2格式.csv")

# 8. 白昼时段误差评估（06:00–18:00）
df_result['预报时间'] = pd.to_datetime(df_result['预报时间'], format='%Y/%m/%d/%H:%M')
daylight_df = df_result[df_result['预报时间'].dt.hour.between(6, 18)]

mse = mean_squared_error(daylight_df['实际功率(MW)'], daylight_df['预测功率(MW)'])
mae = mean_absolute_error(daylight_df['实际功率(MW)'], daylight_df['预测功率(MW)'])

print(f'白昼时段 MSE: {mse:.4f}, MAE: {mae:.4f}')

plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统常用黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示方块问题

# 获取前7天的实际时间戳（SEQ_LEN之后开始）
time_index = test_df.index[SEQ_LEN:SEQ_LEN + 96 * 7]  # 前7天，每天96点
# 9. 可视化
plt.figure(figsize=(12, 5))
plt.plot(time_index,y_test_inv[:96*7], label='实际功率', linewidth=1.2)
plt.plot(time_index,y_pred_inv[:96*7], label='预测功率', linewidth=1.2)
plt.title('LSTM-XGBoost 融合模型预测功率（前7天）')
plt.xlabel('时间步')
plt.ylabel('功率 (MW)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
