import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
import joblib
from datetime import timedelta
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取历史功率数据（15分钟间隔）
train_power = pd.read_csv(r'C:\Users\58407\Desktop\2_train_data.csv', parse_dates=['utc_timestamp']).set_index('utc_timestamp')
test_power = pd.read_csv(r'C:\Users\58407\Desktop\2_test_data.csv', parse_dates=['utc_timestamp']).set_index('utc_timestamp')

# 2. 读取NWP数据（1小时间隔）假设有多个空间点（如3个）
# 多空间点NWP示例列: 'grid1_AT_temperature', 'grid2_AT_temperature', 'grid3_AT_temperature' 等
train_nwp_multi = pd.read_csv(r'C:\Users\58407\Desktop\Q3_train_data.csv', parse_dates=['utc_timestamp']).set_index('utc_timestamp')
test_nwp_multi = pd.read_csv(r'C:\Users\58407\Desktop\Q3_test_data.csv', parse_dates=['utc_timestamp']).set_index('utc_timestamp')

# 3. NWP插值到15分钟间隔（时间插值）
def interpolate_nwp(nwp_df):
    new_idx = pd.date_range(start=nwp_df.index.min(), end=nwp_df.index.max(), freq='15min')
    nwp_df = nwp_df.reindex(nwp_df.index.union(new_idx))
    nwp_df_interp = nwp_df.interpolate(method='time').reindex(new_idx)
    return nwp_df_interp

train_nwp_multi_15min = interpolate_nwp(train_nwp_multi)
test_nwp_multi_15min = interpolate_nwp(test_nwp_multi)

# 4. 空间降尺度：训练一个简单的MLP模型把多个网格点NWP映射到目标点NWP（假设目标点NWP在train_nwp_15min中）
# 这里目标点NWP（3个变量）在文件Q3_train_data.csv中,假设已插值为train_nwp_15min
train_nwp_target = pd.read_csv(r'C:\Users\58407\Desktop\Q3_train_data.csv', parse_dates=['utc_timestamp']).set_index('utc_timestamp')
test_nwp_target = pd.read_csv(r'C:\Users\58407\Desktop\Q3_test_data.csv', parse_dates=['utc_timestamp']).set_index('utc_timestamp')
train_nwp_target_15min = interpolate_nwp(train_nwp_target)
test_nwp_target_15min = interpolate_nwp(test_nwp_target)

# 取训练集和测试集对应时间交集
train_idx = train_nwp_multi_15min.index.intersection(train_nwp_target_15min.index)
test_idx = test_nwp_multi_15min.index.intersection(test_nwp_target_15min.index)

train_nwp_multi_15min = train_nwp_multi_15min.loc[train_idx]
train_nwp_target_15min = train_nwp_target_15min.loc[train_idx]

test_nwp_multi_15min = test_nwp_multi_15min.loc[test_idx]
test_nwp_target_15min = test_nwp_target_15min.loc[test_idx]

# 特征列（多个网格点气象变量，示例以温度为例，替换成你的实际列名）
multi_feature_cols = [col for col in train_nwp_multi_15min.columns if 'temperature' in col or 'radiation' in col or 'diffuse' in col or 'direct' in col]

target_feature_cols = ['AT_radiation_diffuse_horizontal', 'AT_radiation_direct_horizontal', 'AT_temperature']

# 归一化多点NWP特征和目标点NWP特征
scaler_multi_nwp = MinMaxScaler()
scaler_target_nwp = MinMaxScaler()

train_multi_nwp_scaled = scaler_multi_nwp.fit_transform(train_nwp_multi_15min[multi_feature_cols])
train_target_nwp_scaled = scaler_target_nwp.fit_transform(train_nwp_target_15min[target_feature_cols])

test_multi_nwp_scaled = scaler_multi_nwp.transform(test_nwp_multi_15min[multi_feature_cols])
test_target_nwp_scaled = scaler_target_nwp.transform(test_nwp_target_15min[target_feature_cols])

# 空间降尺度MLP模型定义
input_dim = len(multi_feature_cols)
output_dim = len(target_feature_cols)

sd_input = Input(shape=(input_dim,))
x = Dense(64, activation='relu')(sd_input)
x = Dense(32, activation='relu')(x)
sd_output = Dense(output_dim)(x)

sd_model = Model(sd_input, sd_output)
sd_model.compile(optimizer='adam', loss='mse')

# 训练空间降尺度模型
early_stop_sd = EarlyStopping(patience=5, restore_best_weights=True)
sd_model.fit(train_multi_nwp_scaled, train_target_nwp_scaled, 
             validation_split=0.1, epochs=50, batch_size=64, callbacks=[early_stop_sd])

# 用空间降尺度模型预测训练集和测试集目标点NWP
train_nwp_downscaled = sd_model.predict(train_multi_nwp_scaled)
test_nwp_downscaled = sd_model.predict(test_multi_nwp_scaled)

# 反归一化降尺度NWP（可选，用于评估降尺度误差）
train_nwp_downscaled_inv = scaler_target_nwp.inverse_transform(train_nwp_downscaled)
test_nwp_downscaled_inv = scaler_target_nwp.inverse_transform(test_nwp_downscaled)

# 5. 读取功率数据，并对齐时间
#train_power.index = train_power.index.tz_localize('UTC')  # 给 train_power 添加 UTC 时区
#train_power = train_power.loc[train_idx]
#test_power.index = test_power.index.tz_localize('UTC')  # 添加 UTC 时区
#test_power = test_power.loc[test_idx]
valid_train_idx = train_idx.intersection(train_power.index)
train_power = train_power.loc[valid_train_idx]
valid_test_idx = train_idx.intersection(test_power.index)
test_power = test_power.loc[valid_test_idx]

#train_power = train_power.loc[train_idx]
#test_power = test_power.loc[test_idx]

# 6. 功率归一化
scaler_power = MinMaxScaler()
train_power_scaled = scaler_power.fit_transform(train_power[[ 'AT_solar_generation_actual' ]])
test_power_scaled = scaler_power.transform(test_power[[ 'AT_solar_generation_actual' ]])

# 7. 构造序列数据函数（改为用降尺度后的NWP特征）
SEQ_LEN = 96  # 1天长度

def create_sequences(power_data, nwp_data, seq_len):
    X_power, X_nwp, y = [], [], []
    for i in range(len(power_data) - seq_len):
        X_power.append(power_data[i:i+seq_len])      # 功率序列
        X_nwp.append(nwp_data[i+seq_len])            # 降尺度后NWP特征，当前时刻
        y.append(power_data[i+seq_len])               # 目标功率
    return np.array(X_power), np.array(X_nwp), np.array(y)

X_train_power, X_train_nwp, y_train = create_sequences(train_power_scaled, train_nwp_downscaled, SEQ_LEN)
X_test_power, X_test_nwp, y_test = create_sequences(test_power_scaled, test_nwp_downscaled, SEQ_LEN)

# 8. LSTM模型提取功率序列特征
input_power = Input(shape=(SEQ_LEN, 1))
x = LSTM(64)(input_power)
lstm_model = Model(inputs=input_power, outputs=x)
lstm_model.compile(optimizer='adam', loss='mse')

early_stop_lstm = EarlyStopping(patience=3, restore_best_weights=True)
lstm_model.fit(X_train_power.reshape(-1, SEQ_LEN, 1), y_train,
               epochs=20, batch_size=32, validation_split=0.1, callbacks=[early_stop_lstm])

# 9. LSTM提取特征用于XGBoost
train_lstm_feat = lstm_model.predict(X_train_power.reshape(-1, SEQ_LEN, 1))
test_lstm_feat = lstm_model.predict(X_test_power.reshape(-1, SEQ_LEN, 1))

# 10. 组合XGBoost输入（LSTM特征 + 降尺度后的NWP特征）
X_train_xgb = np.hstack([train_lstm_feat, X_train_nwp])
X_test_xgb = np.hstack([test_lstm_feat, X_test_nwp])

# 11. 训练XGBoost模型
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, learning_rate=0.1)
xgb_model.fit(X_train_xgb, y_train)

# 12. 预测
y_pred = xgb_model.predict(X_test_xgb)

# 13. 反归一化功率
y_test_inv = scaler_power.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_inv = scaler_power.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# 14. 保存模型和归一化器
lstm_model.save('lstm_model.h5')
sd_model.save('nwp_space_downscale_model.h5')
joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(scaler_power, 'scaler_power.pkl')
joblib.dump(scaler_multi_nwp, 'scaler_multi_nwp.pkl')
joblib.dump(scaler_target_nwp, 'scaler_target_nwp.pkl')

print("模型已保存。")

# 15. 生成表2格式结果（7天预测，每天96个15分钟点）
start_datetime = test_power.index[SEQ_LEN]
interval = timedelta(minutes=15)
rows = []

for day in range(7):
    issue_time = (start_datetime + timedelta(days=day)).replace(hour=0, minute=0)
    forecast_start = issue_time + timedelta(days=1)
    for i in range(96):
        idx = day * 96 + i
        if idx >= len(y_test_inv):
            break
        forecast_time = forecast_start + i * interval
        rows.append([
            issue_time.strftime('%Y/%m/%d/%H:%M'),
            forecast_time.strftime('%Y/%m/%d/%H:%M'),
            round(y_test_inv[idx], 2),
            round(y_pred_inv[idx], 2)
        ])

df_result = pd.DataFrame(rows, columns=['起报时间', '预报时间', '实际功率(MW)', '预测功率(MW)'])
df_result.to_csv('Q4_功率预测结果_表2格式.csv', index=False, encoding='utf-8-sig')
print("表2格式预测结果已保存：Q4_功率预测结果_表2格式.csv")

# 16. 白昼时段误差统计（06:00~18:00）
df_result['预报时间'] = pd.to_datetime(df_result['预报时间'], format='%Y/%m/%d/%H:%M')
daylight_df = df_result[(df_result['预报时间'].dt.hour >= 6) & (df_result['预报时间'].dt.hour <= 18)]

mse = mean_squared_error(daylight_df['实际功率(MW)'], daylight_df['预测功率(MW)'])
mae = mean_absolute_error(daylight_df['实际功率(MW)'], daylight_df['预测功率(MW)'])
print(f'白昼时段 MSE: {mse:.4f}, MAE: {mae:.4f}')

# 17. 绘图
time_index = test_power.index[SEQ_LEN:SEQ_LEN + 96 * 7]

plt.figure(figsize=(14, 5))
plt.plot(time_index, y_test_inv[:96*7], label='实际功率', linewidth=1.2)
plt.plot(time_index, y_pred_inv[:96*7], label='预测功率', linewidth=1.2)
plt.title('LSTM-XGBoost + NWP空间降尺度模型预测功率（前7天）')
plt.xlabel('时间')
plt.ylabel('功率 (MW)')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

