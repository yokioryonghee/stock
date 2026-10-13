"""Chronological direction forecasting, with separate validation and test blocks."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score

FEATURES=['return_1d','momentum_5','momentum_20','volatility_20','range','volume_change']

def prepare(raw,horizon=1):
    if not isinstance(horizon,int) or horizon<1: raise ValueError('Positive integer horizon required')
    df=raw.copy()
    if isinstance(df.columns,pd.MultiIndex):
        fields={'Open','High','Low','Close','Volume'}
        levels=[i for i in range(df.columns.nlevels) if fields<=set(df.columns.get_level_values(i))]
        if len(levels)!=1: raise ValueError('Cannot identify OHLCV fields')
        field=levels[0]
        if any(df.columns.get_level_values(i).nunique()!=1 for i in range(df.columns.nlevels) if i!=field): raise ValueError('Single asset required')
        df.columns=df.columns.get_level_values(field)
    df.columns=[str(c).lower() for c in df.columns]
    required=['open','high','low','close','volume']
    if not set(required)<=set(df): raise ValueError('OHLCV columns required')
    df.index=pd.to_datetime(df.index,errors='raise'); df=df.sort_index()
    if df.index.duplicated().any(): raise ValueError('Duplicate dates')
    df[required]=df[required].apply(pd.to_numeric,errors='raise')
    if not np.isfinite(df[required]).all().all() or (df[required[:4]]<=0).any().any() or (df.volume<0).any(): raise ValueError('Invalid OHLCV')
    close=df.close
    df['return_1d']=close.pct_change(fill_method=None)
    df['momentum_5']=close.pct_change(5,fill_method=None)
    df['momentum_20']=close.pct_change(20,fill_method=None)
    df['volatility_20']=df.return_1d.rolling(20).std()
    df['range']=(df.high-df.low)/close
    df['volume_change']=df.volume.pct_change(fill_method=None).replace([np.inf,-np.inf],np.nan).fillna(0)
    future=close.shift(-horizon)
    df['target']=(future>close).astype(float).where(future.notna())
    df['label_end']=pd.Series(df.index,index=df.index).shift(-horizon)
    return df.dropna(subset=FEATURES).copy()

def chronological_splits(frame):
    frame=frame.dropna(subset=['target','label_end'])
    if len(frame)<150: raise ValueError('At least 150 labeled feature rows required')
    val_start=frame.index[int(len(frame)*.6)]
    test_start=frame.index[int(len(frame)*.8)]
    train=frame[(frame.index<val_start)&(frame.label_end<val_start)]
    val=frame[(frame.index>=val_start)&(frame.index<test_start)&(frame.label_end<test_start)]
    test=frame[frame.index>=test_start]
    if min(len(train),len(val),len(test))<10: raise ValueError('Too little history after label purge')
    return train,val,test

def run_random_forest(frame,horizon=1,tune=True):
    train,val,test=chronological_splits(frame)
    if train.target.nunique()<2: raise ValueError('Training needs both direction classes')
    model=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=1,class_weight='balanced')
    if tune:
        model=GridSearchCV(model,{'max_depth':[3,6,None]},cv=TimeSeriesSplit(n_splits=3,gap=horizon),scoring='balanced_accuracy',n_jobs=1)
    model.fit(train[FEATURES],train.target.astype(int))
    majority=int(train.target.mode().iloc[0])
    return {'model':'random_forest','train_count':len(train),'validation_count':len(val),'test_count':len(test),
            'last_train_label':str(train.label_end.max()),'first_validation_signal':str(val.index.min()),
            'last_validation_label':str(val.label_end.max()),'first_test_signal':str(test.index.min()),
            'validation_accuracy':float(accuracy_score(val.target,model.predict(val[FEATURES]))),
            'test_accuracy':float(accuracy_score(test.target,model.predict(test[FEATURES]))),
            'test_balanced_accuracy':float(balanced_accuracy_score(test.target,model.predict(test[FEATURES]))),
            'train_majority_test_accuracy':float(accuracy_score(test.target,np.full(len(test),majority))),
            'best_params':getattr(model,'best_params_',{}),'scope':'direction classification; no profit claim'}

def run_lstm(frame,epochs=15,lookback=30,seed=42):
    import torch
    from torch import nn
    if epochs<1 or lookback<2: raise ValueError('Positive epochs and lookback>=2 required')
    torch.manual_seed(seed); torch.set_num_threads(2)
    train,val,test=chronological_splits(frame)
    scaler=StandardScaler().fit(frame.loc[:train.index.max(),FEATURES])
    x=scaler.transform(frame[FEATURES]).astype(np.float32)
    def windows(block):
        indices=frame.index.get_indexer(block.index)
        keep=indices>=lookback-1; indices=indices[keep]
        if not len(indices): raise ValueError('No complete historical windows')
        return torch.tensor(np.stack([x[i-lookback+1:i+1] for i in indices])),torch.tensor(block.target.to_numpy()[keep],dtype=torch.float32)
    xt,yt=windows(train); xv,yv=windows(val); xs,ys=windows(test)
    class DirectionLSTM(nn.Module):
        def __init__(self):
            super().__init__(); self.lstm=nn.LSTM(len(FEATURES),32,batch_first=True); self.head=nn.Linear(32,1)
        def forward(self,input):
            output,_=self.lstm(input); return self.head(output[:,-1]).squeeze(-1)
    model=DirectionLSTM(); optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    loss_fn=nn.BCEWithLogitsLoss(); best=float('inf'); state=None
    generator=torch.Generator().manual_seed(seed)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xt,yt),batch_size=32,shuffle=True,generator=generator)
    for _ in range(epochs):
        model.train()
        for xb,yb in loader:
            loss=loss_fn(model(xb),yb); optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): score=float(loss_fn(model(xv),yv))
        if score<best: best=score; state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state); model.eval()
    with torch.no_grad(): prediction=(model(xs)>0).numpy().astype(int)
    return {'model':'lstm','test_accuracy':float(accuracy_score(ys.numpy(),prediction)),
            'test_count':len(ys),'validation_loss':best,
            'train_majority_test_accuracy':float(accuracy_score(ys.numpy(),np.full(len(ys),int(train.target.mode().iloc[0])))),
            'scope':'validation-selected model; untouched test; no profit claim'}

def demo_bars(n=600):
    rng=np.random.default_rng(42); close=100*np.exp(np.cumsum(rng.normal(0,.01,n)))
    op=close*np.exp(rng.normal(0,.003,n))
    return pd.DataFrame({'open':op,'high':np.maximum(op,close)*1.01,'low':np.minimum(op,close)*.99,
                         'close':close,'volume':rng.integers(100,10000,n)},index=pd.bdate_range('2020-01-01',periods=n))

def main(default_model='rf'):
    p=argparse.ArgumentParser(); p.add_argument('--ticker',default='AAPL'); p.add_argument('--csv')
    p.add_argument('--start',default='2015-01-01'); p.add_argument('--end'); p.add_argument('--demo',action='store_true')
    p.add_argument('--model',choices=['rf','lstm','both'],default=default_model)
    p.add_argument('--horizon',type=int,default=1); p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--output',default='artifacts/forecast.json'); a=p.parse_args()
    if a.demo: raw=demo_bars()
    elif a.csv: raw=pd.read_csv(a.csv,index_col=0,parse_dates=True)
    else:
        import yfinance as yf
        raw=yf.download(a.ticker,start=a.start,end=a.end,auto_adjust=True,multi_level_index=False,progress=False,timeout=20)
        if raw is None or raw.empty: p.error('No market data returned')
    frame=prepare(raw,a.horizon); results=[]
    if a.model in ('rf','both'): results.append(run_random_forest(frame,a.horizon))
    if a.model in ('lstm','both'): results.append(run_lstm(frame,a.epochs))
    report={'symbol':a.ticker,'synthetic':a.demo,'horizon_sessions':a.horizon,'results':results}
    path=Path(a.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
