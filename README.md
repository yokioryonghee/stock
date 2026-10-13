# Stock direction forecasting experiments

일봉 가격과 거래량으로 미래 방향을 분류하는 Random Forest와 LSTM 실험입니다. 학습, 모델 선택용 검증, 최종 테스트를 시간 순서로 분리합니다.

## 설치와 오프라인 실행

Python 3.11 이상을 사용합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python realtimeTicker2.py --demo
python realtimeTicker.py --demo --epochs 2
python "realtimeTicker5(0512).py" --demo --epochs 2
```

합성 데모는 실행 경로 검사입니다. 실제 시장 성능을 보여주지 않습니다. 결과는 `artifacts/forecast.json`에 저장됩니다. Windows는 `.\.venv\Scripts\Activate.ps1`로 활성화합니다.

## 실제 일봉 실험

```bash
python forecasting.py --ticker AAPL --start 2015-01-01 --model both --horizon 5 --epochs 15 --output artifacts/aapl.json
```

Yahoo Finance 연결이 필요합니다. 또는 `--csv`로 date index와 open,high,low,close,volume 열이 있는 CSV를 전달합니다. 현재 실시간 시세나 주문 실행은 구현하지 않습니다.

## 검증 방식

- 특징과 타깃은 동일한 날짜 index로 정렬합니다. 아직 관측되지 않은 미래 라벨은 missing으로 남깁니다.
- 약 60% train / 20% validation / 20% test를 사용하고, 미래 라벨이 다음 구간을 침범하는 표본을 제외합니다.
- LSTM scaler는 학습 입력에만 fit합니다. checkpoint는 validation loss로 선택하고 최종 test는 선택에 사용하지 않습니다.
- RF 튜닝은 train 안에서 TimeSeriesSplit과 horizon gap을 사용합니다.
- 최종 정확도와 학습 구간 다수 클래스 기준선을 함께 출력합니다. 정확도는 수익성이 아닙니다.

```bash
python -m pip install pytest
python -m pytest -q
```

선택된 종목·기간의 한 번의 평가만으로 미래 우위가 입증되지는 않습니다. 실제 수익성, 거래 비용과 주문 운영은 이 실험의 범위 밖입니다.
