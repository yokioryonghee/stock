import yfinance as yf
import pandas as pd

# 개별 종목 리스트
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

# 종목 데이터 수집 함수
def get_stock_data(ticker, start='2020-01-01', end='2025-05-08'):
    return yf.download(ticker, start=start, end=end)

# 여러 종목 데이터 수집
stock_data = {ticker: get_stock_data(ticker) for ticker in tickers}

# 데이터 확인 (애플의 데이터)
print(stock_data['AAPL'].head())

# 기술적 지표 계산 함수
def add_technical_indicators(df):
    df['MA_50'] = df['Close'].rolling(window=50).mean()  # 50일 이동 평균
    df['MA_200'] = df['Close'].rolling(window=200).mean()  # 200일 이동 평균
    df['RSI'] = calculate_rsi(df['Close'])  # RSI 추가
    return df

# RSI 계산 함수
def calculate_rsi(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 각 종목에 기술적 지표 추가
stock_data = {ticker: add_technical_indicators(df) for ticker, df in stock_data.items()}

# 애플 종목의 데이터 확인
print(stock_data['AAPL'].tail())

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 투자 결정(매수/매도) 모델 학습 함수
def train_investment_model(df):
    # 피처와 타깃 설정 (기술적 지표를 피처로 사용, 가격 상승/하락을 타깃으로 설정)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)  # 다음 날 주가 상승 여부 (1: 상승, 0: 하락)
    X = df[['MA_50', 'MA_200', 'RSI']].dropna()  # 기술적 지표를 피처로 사용
    y = df['Target'].dropna()  # 타깃 값
    
    X, y = X.iloc[:len(y)], y.iloc[:len(X)]


    # 학습 및 테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 랜덤 포레스트 모델 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 테스트 데이터로 모델 성능 평가
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    return model

# 애플 종목의 투자 결정 모델 학습
aapl_model = train_investment_model(stock_data['AAPL'])

# 실시간 매수/매도 신호 함수
def generate_investment_signal(model, df):
    # 가장 최근 데이터에 대해 예측
    latest_data = df[['MA_50', 'MA_200', 'RSI']].dropna().iloc[-1].values.reshape(1, -1)
    signal = model.predict(latest_data)
    
    if signal == 1:
        print("Buy Signal: Predicted Price Increase")
    else:
        print("Sell Signal: Predicted Price Decrease")

# 애플 종목에 대한 실시간 투자 신호
generate_investment_signal(aapl_model, stock_data['AAPL'])

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 실시간 그래프 업데이트 함수
def animate(i):
    # 최신 데이터를 가져옴
    latest_data = yf.download('AAPL', period='1d', interval='1m')
    stock_data['AAPL']._append(latest_data)
    
    # 차트 그리기
    plt.cla()
    plt.plot(stock_data['AAPL']['Close'], label='Close Price')
    plt.plot(stock_data['AAPL']['MA_50'], label='50-day MA')
    plt.plot(stock_data['AAPL']['MA_200'], label='200-day MA')
    plt.legend(loc='upper left')
    plt.tight_layout()

# 그래프 생성
ani = animation.FuncAnimation(plt.gcf(), animate, interval=600)  # 1분마다 업데이트
plt.show()

