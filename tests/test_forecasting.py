import numpy as np
import pandas as pd
from forecasting import prepare,chronological_splits,demo_bars,FEATURES,run_random_forest,run_lstm

def test_targets_keep_their_dates_and_unknown_tail():
    raw=demo_bars(); frame=prepare(raw,horizon=5)
    expected=(raw.close.shift(-5)>raw.close).astype(float).where(raw.close.shift(-5).notna())
    pd.testing.assert_series_equal(frame.target,expected.loc[frame.index],check_names=False)
    assert frame.target.tail(5).isna().all()

def test_features_causal_and_purged_splits():
    raw=demo_bars(); frame=prepare(raw,5); changed=raw.copy(); changed.iloc[400:,:4]*=100
    modified=prepare(changed,5)
    pd.testing.assert_frame_equal(frame.iloc[:380][FEATURES],modified.iloc[:380][FEATURES])
    train,val,test=chronological_splits(frame)
    assert train.label_end.max()<val.index.min()
    assert val.label_end.max()<test.index.min()

def test_models_run_and_report_untouched_test():
    frame=prepare(demo_bars())
    result=run_random_forest(frame,tune=True)
    assert 0<=result['test_accuracy']<=1 and result['best_params']
    lstm=run_lstm(frame,epochs=1,lookback=8)
    assert 0<=lstm['test_accuracy']<=1
