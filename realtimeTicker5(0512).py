# Filename: multi_model_SOXL_predictor_with_gridsearch.py

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV # <<< ADDED for Hyperparameter Tuning

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

# --- Configuration ---
TICKER = 'AAPL' # Changed back to SOXL as per your last code
START_DATE = '2015-01-01'
END_DATE = '2025-05-13'
TIME_STEP = 60
PREDICT_STEPS = 1
TRAIN_SPLIT_PERCENT = 0.8
LSTM_UNITS = 50 # From your last snippet
DROPOUT_RATE = 0.2 # From your last snippet
EPOCHS = 50      # REDUCED for faster GridSearchCV testing initially, you can increase
BATCH_SIZE = 20  # From your last snippet
BEST_LSTM_WEIGHTS_FILEPATH = 'best_lstm_SOXL_model_tuned.weights.h5'

# Feature Columns
FEATURE_COLUMNS = [
    'Close', 'Volume', 'MA_50', 'MA_200', 'RSI', 'MACD',
    'Log_Return', 'Return_Volatility_10D', 'ATR_14', 'Close_vs_MA20_Ratio'
]
NUM_FEATURES = len(FEATURE_COLUMNS) # Will be updated dynamically

# --- Helper Functions (calculate_atr, add_all_technical_indicators, create_lstm_sequences) ---
# (These functions remain largely the same as in the previous 'multi_model_SOXL_predictor.py'
#  Make sure they are correctly defined as in the script you confirmed was working before this request)

def calculate_atr(high_series, low_series, close_series, period=14):
    if not all(isinstance(s, pd.Series) for s in [high_series, low_series, close_series]):
        if isinstance(close_series, pd.Series):
            return pd.Series(np.nan, index=close_series.index, dtype='float64')
        return pd.Series(dtype='float64')
    if len(high_series) < period + 1:
        return pd.Series([np.nan] * len(high_series), index=high_series.index)
    tr1 = pd.DataFrame(high_series - low_series)
    tr2 = pd.DataFrame(abs(high_series - close_series.shift(1)))
    tr3 = pd.DataFrame(abs(low_series - close_series.shift(1)))
    tr_df = pd.concat([tr1, tr2, tr3], axis=1)
    true_range = tr_df.max(axis=1, skipna=False)
    atr = true_range.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return atr

def add_all_technical_indicators(df_input):
    df = df_input.copy()
    df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True, errors='ignore')

    if not all(col in df.columns for col in ['high', 'low', 'close']):
        print("Error: DataFrame missing 'high', 'low', or 'close' columns for TA calculation.")
        # Attempt to create them if 'close' exists and others don't (less ideal)
        if 'close' in df.columns:
            if 'high' not in df.columns: df['high'] = df['close']
            if 'low' not in df.columns: df['low'] = df['close']
        else:
            return df # Cannot proceed

    close_prices = pd.to_numeric(df['close'], errors='coerce')
    if close_prices.isnull().any():
         print("Warning: NaNs found in Close prices during TA calculation after to_numeric.")

    df['MA_20'] = close_prices.rolling(window=20, min_periods=1).mean()
    df['MA_50'] = close_prices.rolling(window=50, min_periods=1).mean()
    df['MA_200'] = close_prices.rolling(window=200, min_periods=1).mean()

    delta = close_prices.diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=13).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=13).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100.0 - (100.0 / (1.0 + rs.replace([np.inf, -np.inf], np.nan)))
    df['RSI'] = df['RSI'].clip(0, 100).fillna(50)

    ema_12 = close_prices.ewm(span=12, adjust=False).mean()
    ema_26 = close_prices.ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD'] = df['MACD_Line'] - df['MACD_Signal'] # This is MACD Histogram

    df['Log_Return'] = np.log(df['close'] / df['close'].shift(1))
    if 'Log_Return' in df.columns: # Check because shift(1) creates NaN in first row
        df['Return_Volatility_10D'] = df['Log_Return'].rolling(window=10, min_periods=1).std()

    df['ATR_14'] = calculate_atr(df['high'], df['low'], df['close'], period=14)

    if 'MA_20' in df.columns and 'close' in df.columns:
        # Ensure MA_20 is not zero to avoid division by zero
        df['Close_vs_MA20_Ratio'] = df['close'] / df['MA_20'].replace({0: np.nan}) # Replace 0 with NaN before division
        df['Close_vs_MA20_Ratio'] = df['Close_vs_MA20_Ratio'].replace([np.inf, -np.inf], np.nan)
    else:
        df['Close_vs_MA20_Ratio'] = np.nan
    return df

def create_lstm_sequences(data_scaled, target_col_index, time_step=60, predict_steps=1):
    X, y = [], []
    if data_scaled.shape[0] <= time_step + predict_steps - 1:
        print(f"LSTM Warning: Not enough data ({data_scaled.shape[0]}) for sequences with time_step={time_step}, predict_steps={predict_steps}.")
        return np.array(X), np.array(y)
    for i in range(len(data_scaled) - time_step - predict_steps + 1):
        X.append(data_scaled[i:(i + time_step), :])
        y.append(data_scaled[i + time_step + predict_steps - 1, target_col_index])
    return np.array(X), np.array(y)

# --- Main Script ---
if __name__ == '__main__':
    print(f"Downloading initial data for {TICKER}...")
    initial_data_df = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    if not isinstance(initial_data_df, pd.DataFrame) or initial_data_df.empty:
        print("ERROR: yfinance download failed. Exiting.")
        exit()

    if isinstance(initial_data_df.columns, pd.MultiIndex):
        initial_data_df.columns = initial_data_df.columns.get_level_values(0)
    if isinstance(initial_data_df.index, pd.DatetimeIndex) and initial_data_df.index.tz is not None:
        initial_data_df.index = initial_data_df.index.tz_localize(None)

    required_ohlcv = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in initial_data_df.columns for col in required_ohlcv):
        print(f"ERROR: Downloaded data missing one or more of {required_ohlcv}. Auto_adjust=True should provide these.")
        print(f"Columns found: {initial_data_df.columns}")
        # If auto_adjust=True but still missing, yfinance might have an issue or data for ticker is sparse
        # If you set auto_adjust=False, ensure you handle 'Adj Close' appropriately.
        exit()

    print("Calculating technical indicators...")
    data_with_ta = add_all_technical_indicators(initial_data_df.copy())

    # Determine actual feature columns after TA calculation (lowercase)
    actual_feature_columns = []
    for col_config in FEATURE_COLUMNS:
        col_lower = col_config.lower()
        # Check if lowercase version exists and is numeric
        if col_lower in data_with_ta.columns and pd.api.types.is_numeric_dtype(data_with_ta[col_lower]):
            actual_feature_columns.append(col_lower)
        # Fallback to original case if that exists and is numeric (shouldn't happen if rename to lower works)
        elif col_config in data_with_ta.columns and pd.api.types.is_numeric_dtype(data_with_ta[col_config]):
            actual_feature_columns.append(col_config)
            if col_lower != col_config: # If config was mixed case but only original case found
                print(f"Warning: Feature '{col_config}' found with original casing, not lowercase '{col_lower}'.")
        else:
            print(f"Warning: Feature '{col_config}' (or '{col_lower}') not found or not numeric after TA. Will be filled with NaN if critical.")
            data_with_ta[col_lower] = np.nan # Ensure column exists for selection
            actual_feature_columns.append(col_lower)


    if not actual_feature_columns:
        print("ERROR: No valid feature columns available after TA. Exiting.")
        exit()

    # Ensure 'close' is in actual_feature_columns if it's needed for target derivation
    close_col_name_in_df = 'close' # Since we rename to lowercase
    if close_col_name_in_df not in actual_feature_columns:
        # This can happen if 'Close' was not in the original FEATURE_COLUMNS or failed to be created
        if 'close' in data_with_ta.columns and pd.api.types.is_numeric_dtype(data_with_ta['close']):
            actual_feature_columns.append('close') # Add it if it exists but wasn't in config list
            print(f"Info: Added '{close_col_name_in_df}' to features for target derivation.")
        else:
            print(f"ERROR: Crucial column '{close_col_name_in_df}' for target is missing or not numeric. Exiting.")
            exit()
    
    # Update NUM_FEATURES based on actual columns being used
    NUM_FEATURES = len(actual_feature_columns)
    print(f"Using actual features ({NUM_FEATURES} total): {actual_feature_columns}")


    feature_data = data_with_ta[actual_feature_columns].copy()
    
    initial_len = len(feature_data)
    feature_data.dropna(inplace=True) # Drop rows with ANY NaN in the selected actual_feature_columns
    final_len = len(feature_data)
    print(f"Dropped {initial_len - final_len} rows with NaNs from feature_data.")

    if final_len < TIME_STEP + PREDICT_STEPS:
         print(f"Error: Not enough data ({final_len}) for TIME_STEP {TIME_STEP}. Try earlier START_DATE.")
         exit()

    print("Feature data sample (after TA and dropna):")
    print(feature_data.head())

    feature_data['Future_Close'] = feature_data[close_col_name_in_df].shift(-PREDICT_STEPS)
    feature_data.dropna(inplace=True)
    
    if feature_data.empty:
        print("Error: No data left after creating Future_Close and dropping NaNs.")
        exit()

    y_direction_target = (feature_data['Future_Close'] > feature_data[close_col_name_in_df]).astype(int)
    X_features_for_model_df = feature_data[actual_feature_columns].copy() # Use the confirmed list

    if len(X_features_for_model_df) != len(y_direction_target):
        print(f"Aligning X and y after Future_Close creation...")
        X_features_for_model_df = X_features_for_model_df.loc[y_direction_target.index]

    train_split_idx = int(len(X_features_for_model_df) * TRAIN_SPLIT_PERCENT)
    X_train_df = X_features_for_model_df.iloc[:train_split_idx]
    X_test_df = X_features_for_model_df.iloc[train_split_idx:]
    y_train_direction = y_direction_target.iloc[:train_split_idx]
    y_test_direction = y_direction_target.iloc[train_split_idx:]
    
    print(f"Total samples for modeling: {len(X_features_for_model_df)}")
    print(f"Training samples: {len(X_train_df)}, Testing samples: {len(X_test_df)}")

    if len(X_train_df) == 0 or len(X_test_df) == 0:
        print("ERROR: Train or Test data is empty. Exiting.")
        exit()

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler_X.fit_transform(X_train_df)
    X_test_scaled = scaler_X.transform(X_test_df)

    close_col_idx_in_features = actual_feature_columns.index(close_col_name_in_df)

    temp_X_train_lstm_seq, temp_y_train_lstm_seq_scaled = [], []
    for i in range(len(X_train_scaled) - TIME_STEP - PREDICT_STEPS + 1):
        temp_X_train_lstm_seq.append(X_train_scaled[i:(i + TIME_STEP), :])
        temp_y_train_lstm_seq_scaled.append(X_train_scaled[i + TIME_STEP + PREDICT_STEPS - 1, close_col_idx_in_features])
    X_train_lstm_seq = np.array(temp_X_train_lstm_seq)
    y_train_lstm_seq_scaled = np.array(temp_y_train_lstm_seq_scaled)

    temp_X_test_lstm_seq, temp_y_test_lstm_seq_scaled = [], []
    if len(X_test_scaled) > TIME_STEP + PREDICT_STEPS -1: # Check if enough data for test sequences
        for i in range(len(X_test_scaled) - TIME_STEP - PREDICT_STEPS + 1):
            temp_X_test_lstm_seq.append(X_test_scaled[i:(i + TIME_STEP), :])
            temp_y_test_lstm_seq_scaled.append(X_test_scaled[i + TIME_STEP + PREDICT_STEPS - 1, close_col_idx_in_features])
    X_test_lstm_seq = np.array(temp_X_test_lstm_seq)
    y_test_lstm_seq_scaled = np.array(temp_y_test_lstm_seq_scaled)

    y_test_direction_lstm_aligned = np.array([])
    if len(y_test_direction) >= TIME_STEP + PREDICT_STEPS -1 + len(X_test_lstm_seq) and len(X_test_lstm_seq)>0 :
        y_test_direction_lstm_aligned = y_test_direction.iloc[TIME_STEP + PREDICT_STEPS - 1 : TIME_STEP + PREDICT_STEPS - 1 + len(X_test_lstm_seq)].values
    elif len(X_test_lstm_seq) > 0: # If some test sequences were made but not enough for full alignment
        y_test_direction_lstm_aligned = y_test_direction.iloc[TIME_STEP + PREDICT_STEPS - 1 :].values[:len(X_test_lstm_seq)]


    print(f"LSTM Train sequences: X={X_train_lstm_seq.shape}, y={y_train_lstm_seq_scaled.shape}")
    print(f"LSTM Test sequences: X={X_test_lstm_seq.shape}, y_scaled={y_test_lstm_seq_scaled.shape}, y_direction_aligned={y_test_direction_lstm_aligned.shape}")

    X_train_rf_tabular = X_train_scaled[TIME_STEP -1 : len(X_train_scaled) - PREDICT_STEPS, :]
    y_train_rf_direction_aligned = y_train_direction.iloc[:len(X_train_rf_tabular)].values
    X_test_rf_tabular = X_test_scaled[TIME_STEP -1 : len(X_test_scaled) - PREDICT_STEPS, :]
    y_test_rf_direction_aligned = y_test_direction.iloc[:len(X_test_rf_tabular)].values
    
    print(f"RF Train samples: X={X_train_rf_tabular.shape}, y={y_train_rf_direction_aligned.shape}")
    print(f"RF Test samples: X={X_test_rf_tabular.shape}, y={y_test_rf_direction_aligned.shape}")

    if X_train_lstm_seq.shape[0] < BATCH_SIZE or X_train_rf_tabular.shape[0] == 0 :
        print("Not enough data to train models. Exiting.")
        exit()

    # ========== LSTM Model ==========
    print("\n--- Training and Evaluating LSTM Model ---")
    model_lstm = Sequential([
        Input(shape=(TIME_STEP, NUM_FEATURES)),
        LSTM(LSTM_UNITS, return_sequences=True),
        Dropout(DROPOUT_RATE),
        LSTM(LSTM_UNITS, return_sequences=False),
        Dropout(DROPOUT_RATE),
        Dense(1)
    ])
    model_lstm.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    lstm_accuracy = 0.0
    history_lstm = None
    if X_train_lstm_seq.shape[0] >= BATCH_SIZE:
        validation_data_lstm = (X_test_lstm_seq, y_test_lstm_seq_scaled) if X_test_lstm_seq.shape[0] > 0 and y_test_lstm_seq_scaled.shape[0] > 0 else None
        monitor_metric_lstm = 'val_loss' if validation_data_lstm else 'loss'
        
        checkpoint_lstm = ModelCheckpoint(
            BEST_LSTM_WEIGHTS_FILEPATH, monitor=monitor_metric_lstm,
            save_best_only=True, save_weights_only=True, mode='min', verbose=0
        )
        print(f"Starting LSTM training for {EPOCHS} epochs...")
        history_lstm = model_lstm.fit(
            X_train_lstm_seq, y_train_lstm_seq_scaled, epochs=EPOCHS, batch_size=BATCH_SIZE,
            validation_data=validation_data_lstm, callbacks=[checkpoint_lstm], verbose=1,
            shuffle=True
        )
        if history_lstm:
            try:
                model_lstm.load_weights(BEST_LSTM_WEIGHTS_FILEPATH)
                print("LSTM: Best weights loaded.")
            except Exception as e:
                print(f"LSTM: Warning - Could not load best weights ({e}).")

        if X_test_lstm_seq.shape[0] > 0 and len(y_test_direction_lstm_aligned) > 0:
            predicted_scaled_close_lstm = model_lstm.predict(X_test_lstm_seq)
            prev_scaled_close_lstm_test = X_test_lstm_seq[:, -1, close_col_idx_in_features]
            predicted_direction_lstm = predicted_scaled_close_lstm.flatten() > prev_scaled_close_lstm_test
            
            actual_direction_lstm_eval = y_test_direction_lstm_aligned[:len(predicted_direction_lstm)]
            if len(predicted_direction_lstm) > 0 and len(actual_direction_lstm_eval) == len(predicted_direction_lstm) :
                lstm_accuracy = np.mean(predicted_direction_lstm == actual_direction_lstm_eval) * 100
            print(f"LSTM Model Accuracy (Predicting Direction): {lstm_accuracy:.2f}%")
        else:
            print("LSTM: No test data or aligned direction data for evaluation.")
    else:
        print("LSTM: Skipping training - not enough data for a batch.")

    # ========== Random Forest Model with GridSearchCV ==========
    print("\n--- Training and Evaluating Random Forest Model with GridSearchCV ---")
    rf_accuracy_tuned = 0.0
    best_rf_params = {}

    if X_train_rf_tabular.shape[0] > 0:
        print("Setting up GridSearchCV for Random Forest...")
        # Define a smaller parameter grid for faster initial testing
        param_grid_rf = {
            'n_estimators': [50, 100, 200, 300, 400, 500, 1000], # Reduced options
            'max_depth': [None, 10, 20, 30, 40, 50],    # Reduced options
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 3, 4],
            
            # 'max_features': ['sqrt', 'log2'] # Using 'sqrt' (formerly 'auto') is default and often good
        }

        rf_model_for_grid = RandomForestClassifier(random_state=42, class_weight='balanced')
        
        # Use fewer folds for cv for faster testing, e.g., cv=3
        # n_jobs=-1 uses all available cores
        grid_search_rf = GridSearchCV(estimator=rf_model_for_grid, param_grid=param_grid_rf, 
                                      cv=3, n_jobs=-1, verbose=1, scoring='accuracy') 
                                      # Scoring can be 'roc_auc', 'f1' etc.

        print("Starting GridSearchCV for Random Forest (this may take time)...")
        grid_search_rf.fit(X_train_rf_tabular, y_train_rf_direction_aligned)

        best_rf_params = grid_search_rf.best_params_
        print(f"Random Forest: Best parameters found: {best_rf_params}")

        best_model_rf = grid_search_rf.best_estimator_ # This is the RF model trained with best params

        if X_test_rf_tabular.shape[0] > 0 and len(y_test_rf_direction_aligned) > 0:
            actual_rf_test_directions = y_test_rf_direction_aligned[:len(X_test_rf_tabular)]
            if len(actual_rf_test_directions) == len(X_test_rf_tabular):
                predicted_direction_rf_tuned = best_model_rf.predict(X_test_rf_tabular)
                rf_accuracy_tuned = accuracy_score(actual_rf_test_directions, predicted_direction_rf_tuned) * 100
                print(f"Tuned Random Forest Model Accuracy: {rf_accuracy_tuned:.2f}%")
            else:
                 print("RF (Tuned): Length mismatch for accuracy calculation.")
        else:
            print("Random Forest (Tuned): No test data for evaluation.")
    else:
        print("Random Forest: Skipping GridSearchCV - not enough training data.")


    # --- Final Comparison ---
    print("\n\n--- Model Performance Comparison (Directional Accuracy) ---")
    print(f"  LSTM Model (Epochs: {EPOCHS}, Batch: {BATCH_SIZE}): {lstm_accuracy:.2f}%")
    print(f"  Tuned Random Forest Model (Best Params: {best_rf_params}): {rf_accuracy_tuned:.2f}%")

    # Basic comparison
    if lstm_accuracy == 0.0 and rf_accuracy_tuned == 0.0:
        print("\nNeither model produced results for comparison.")
    elif lstm_accuracy > rf_accuracy_tuned:
        print("\nLSTM performed better for directional accuracy in this run.")
    elif rf_accuracy_tuned > lstm_accuracy:
        print("\nTuned Random Forest performed better for directional accuracy in this run.")
    else:
        print("\nModels performed similarly for directional accuracy.")
    
    print("\nMulti-model comparison script with GridSearchCV finished.")