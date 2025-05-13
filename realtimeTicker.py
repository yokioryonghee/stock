import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 1. 데이터 수집
ticker = 'TSLA'  # S&P 500
data = yf.download(ticker, start='2010-01-01', end='2025-05-08')

# 2. 전처리
# 'Close' 가격만 사용
data = data[['Close']]

# 결측값 처리
data.dropna(inplace=True)

# MinMaxScaler로 데이터 스케일링 (0과 1 사이 값으로 변환)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. 시계열 데이터 준비 (LSTM 입력 데이터)
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60  # 60일 간의 데이터를 보고 다음날 예측
X, y = create_dataset(scaled_data, time_step)

# 데이터 차원을 LSTM 입력에 맞게 조정 (samples, time steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# 4. 학습/테스트 데이터 분할
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 5. LSTM 모델 생성
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(100))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# 6. 모델 학습
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# 7. 예측
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 스케일링 복원
train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
