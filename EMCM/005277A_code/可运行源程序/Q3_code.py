import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
import joblib
from datetime import timedelta

# 1. 读取历史功率数据（15分钟间隔）
train_power = pd.read_csv(r'C:\Users\58407\Desktop\2_train_data.csv', parse_dates=['utc_timestamp']).set_index('utc_timestamp')
test_power = pd.read_csv(r'C:\Users\58407\Desktop\2_test_data.csv', parse_dates=['utc_timestamp']).set_index('utc_timestamp')

# 2. 读取NWP数据（1小时间隔）
train_nwp = pd.read_csv(r'C:\Users\58407\Desktop\Q3_train_data.csv', parse_dates=['utc_timestamp']).set_index('utc_timestamp')
test_nwp = pd.read_csv(r'C:\Users\58407\Desktop\Q3_test_data.csv', parse_dates=['utc_timestamp']).set_index('utc_timestamp')

# 3. NWP插值到15分钟间隔
def interpolate_nwp(nwp_df):
    new_idx = pd.date_range(start=nwp_df.index.min(), end=nwp_df.index.max(), freq='15min')
    nwp_df = nwp_df.reindex(nwp_df.index.union(new_idx))
    nwp_df_interp = nwp_df.interpolate(method='time').reindex(new_idx)
    return nwp_df_interp

train_nwp_15min = interpolate_nwp(train_nwp)
test_nwp_15min = interpolate_nwp(test_nwp)

# 4. 合并功率和NWP，内连接对齐时间戳
train_all = train_power.join(train_nwp_15min, how='inner')
test_all = test_power.join(test_nwp_15min, how='inner')

# 5. 选取特征和目标
feature_cols_nwp = ['AT_radiation_diffuse_horizontal', 'AT_radiation_direct_horizontal', 'AT_temperature']
target_col = 'AT_solar_generation_actual'

# 6. 归一化（功率和NWP分开归一化）
scaler_power = MinMaxScaler()
scaler_nwp = MinMaxScaler()

train_power_scaled = scaler_power.fit_transform(train_all[[target_col]])
test_power_scaled = scaler_power.transform(test_all[[target_col]])

train_nwp_scaled = scaler_nwp.fit_transform(train_all[feature_cols_nwp])
test_nwp_scaled = scaler_nwp.transform(test_all[feature_cols_nwp])

# 7. 构造序列数据函数
SEQ_LEN = 96  # 1天序列长度（15分钟间隔）

def create_sequences(power_data, nwp_data, seq_len):
    X_power, X_nwp, y = [], [], []
    for i in range(len(power_data) - seq_len):
        X_power.append(power_data[i:i+seq_len])      # 历史功率序列输入LSTM
        X_nwp.append(nwp_data[i+seq_len])            # 对应时刻NWP特征输入XGBoost
        y.append(power_data[i+seq_len])               # 目标值
    return np.array(X_power), np.array(X_nwp), np.array(y)

X_train_power, X_train_nwp, y_train = create_sequences(train_power_scaled, train_nwp_scaled, SEQ_LEN)
X_test_power, X_test_nwp, y_test = create_sequences(test_power_scaled, test_nwp_scaled, SEQ_LEN)

# 8. LSTM模型提取历史功率特征
input_power = Input(shape=(SEQ_LEN, 1))
x = LSTM(64)(input_power)
lstm_model = Model(inputs=input_power, outputs=x)

# 训练LSTM提取特征
lstm_model.compile(optimizer='adam', loss='mse')
early_stop = EarlyStopping(patience=3, restore_best_weights=True)
lstm_model.fit(X_train_power.reshape(-1, SEQ_LEN, 1), y_train,
               epochs=20, batch_size=32, validation_split=0.1, callbacks=[early_stop])

# 9. LSTM提取训练集和测试集特征（用于XGBoost）
train_lstm_feat = lstm_model.predict(X_train_power.reshape(-1, SEQ_LEN, 1))
test_lstm_feat = lstm_model.predict(X_test_power.reshape(-1, SEQ_LEN, 1))

# 10. 组合特征给XGBoost（LSTM输出特征 + NWP特征）
X_train_xgb = np.hstack([train_lstm_feat, X_train_nwp])
X_test_xgb = np.hstack([test_lstm_feat, X_test_nwp])

# 11. 训练XGBoost回归模型
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, learning_rate=0.1)
xgb_model.fit(X_train_xgb, y_train)

# 12. 预测
y_pred = xgb_model.predict(X_test_xgb)

# 13. 反归一化功率
y_test_inv = scaler_power.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_inv = scaler_power.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# 14. 保存模型
lstm_model.save('lstm_model.h5')
joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(scaler_power, 'scaler_power.pkl')
joblib.dump(scaler_nwp, 'scaler_nwp.pkl')

print("模型已保存。")

# 15. 生成表2格式结果（7天预测，每天96个15分钟点）
start_datetime = test_all.index[SEQ_LEN]
interval = timedelta(minutes=15)
rows = []

for day in range(7):
    issue_time = (start_datetime + timedelta(days=day)).replace(hour=0, minute=0)
    forecast_start = issue_time + timedelta(days=1)
    for i in range(96):
        idx = day * 96 + i
        if idx >= len(y_test_inv):  # 防止越界
            break
        forecast_time = forecast_start + i * interval
        rows.append([
            issue_time.strftime('%Y/%m/%d/%H:%M'),
            forecast_time.strftime('%Y/%m/%d/%H:%M'),
            round(y_test_inv[idx], 2),
            round(y_pred_inv[idx], 2)
        ])

df_result = pd.DataFrame(rows, columns=['起报时间', '预报时间', '实际功率(MW)', '预测功率(MW)'])
df_result.to_csv('Q3_功率预测结果_表2格式.csv', index=False, encoding='utf-8-sig')
print("表2格式预测结果已保存：Q3_功率预测结果_表2格式.csv")

# 16. 白昼时段误差统计（06:00~18:00）
df_result['预报时间'] = pd.to_datetime(df_result['预报时间'], format='%Y/%m/%d/%H:%M')
daylight_df = df_result[(df_result['预报时间'].dt.hour >= 6) & (df_result['预报时间'].dt.hour <= 18)]

mse = mean_squared_error(daylight_df['实际功率(MW)'], daylight_df['预测功率(MW)'])
mae = mean_absolute_error(daylight_df['实际功率(MW)'], daylight_df['预测功率(MW)'])
print(f'白昼时段 MSE: {mse:.4f}, MAE: {mae:.4f}')

import matplotlib.pyplot as plt

# 设置中文字体显示（适用于Windows系统）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示方块的问题

# 获取前7天的实际时间戳（SEQ_LEN之后开始）
time_index = test_power.index[SEQ_LEN:SEQ_LEN + 96 * 7]  # 前7天，每天96点

# 绘图：LSTM-XGBoost预测 vs 实际功率
plt.figure(figsize=(14, 5))
plt.plot(time_index, y_test_inv[:96*7], label='实际功率', linewidth=1.2)
plt.plot(time_index, y_pred_inv[:96*7], label='预测功率', linewidth=1.2)
plt.title('LSTM 负荷预测 - 前7天（15分钟分辨率）')
plt.xlabel('时间')
plt.ylabel('功率 (MW)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
