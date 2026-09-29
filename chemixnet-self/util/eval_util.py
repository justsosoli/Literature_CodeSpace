from util.dl_util import *
from util.chemixnet_model_util import *
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import time
import tensorflow as tf

SEED = 1024
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def split_data_and_train_model(model, X1, X2, Y, vocab_size, max_len, prefix, batch_size, epochs, data_handle_method = None, gate = None):
    X1 = np.array(X1)
    X2 = np.array(X2)
    Y = np.array(Y)

    # 这行代码的作用是将 X1, X2, Y 这三组数据按照 8:2 的比例（test_size=0.2）划分为训练集和测试集，并且保证划分的随机性可复现（random_state=SEED）。
    # train_test_split 是用来随机划分数据的常用方法，返回值中 X1_train 和 X1_test 分别是原始 X1 对应的训练和测试部分，以此类推。划分后的各部分可用于后续模型训练和评估。
    X1_train, X1_test, X2_train, X2_test, Y_train, Y_test = train_test_split(X1, X2, Y, test_size=0.2, random_state=SEED)

    model_name = model.__name__
    print(model_name)

    if 'rnn' in model_name:
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    else:
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    time_start = time.time()
    if 'hiv' in prefix or 'tox' in prefix:
        metric = 'accuracy'
        model_category = 'classification'
    else:
        metric = 'mae'
        model_category = 'regression'

    # 数据处理，标准化 / 归一化
    y_train_raw = Y_train
    y_test_raw = Y_test
    if data_handle_method == 'standard':
        y_mean = float(np.mean(y_train_raw))
        y_std  = float(np.std(y_train_raw)) + 1e-12
        y_train_scaled = (y_train_raw - y_mean) / y_std
        y_test_scaled  = (y_test_raw  - y_mean) / y_std
        def _restore_data(data):
            return data * y_std + y_mean
    elif data_handle_method == 'normalize':
        y_min = float(np.min(y_train_raw))
        y_max = float(np.max(y_train_raw))
        y_train_scaled = (y_train_raw - y_min) / (y_max - y_min + 1e-12)
        y_test_scaled  = (y_test_raw  - y_min) / (y_max - y_min + 1e-12)
        def _restore_data(data):
            return data * (y_max - y_min) + y_min
    else:
        y_train_scaled = y_train_raw
        y_test_scaled  = y_test_raw
        def _restore_data(data):
            return data

    # 根据模型 确定训练策略
    if 'mlp' in model_name:
        if 'cnn' in model_name or 'rnn' in model_name:
            if gate:
                model = model(vocab_size = vocab_size, gate = gate, input_shape = max_len, task_type = model_category)
                model_name = model_name.replace('rnn', gate)
            else:
                model = model(vocab_size = vocab_size, input_shape = max_len, task_type = model_category)
            history = model.fit([X1_train, X2_train], y_train_scaled, shuffle=True, validation_split = 0.1, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping])
            X_test = [X1_test, X2_test]
        else:
            model = model(vocab_size = vocab_size, input_shape = max_len, task_type = model_category)
            history = model.fit(X2_train, y_train_scaled, shuffle=True, validation_split = 0.1, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping])
            X_test = X2_test
    
    else:
        if gate:
            model = model(vocab_size = vocab_size, gate = gate, input_shape = max_len, task_type = model_category)
            model_name = model_name.replace('rnn', gate)
        else:
            model = model(vocab_size = vocab_size, input_shape = max_len, task_type = model_category)
        history = model.fit(X1_train, y_train_scaled, shuffle=True, validation_split = 0.1, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping])
        X_test = X1_test
    
    time_end = time.time()
    time_cost = time_end - time_start
    print(f"Time cost: {time_cost} seconds")
    print(f"Model: {model_name}")
    print(f"Model category: {model_category}")
    print(f"Metric: {metric}")
    print(f"History: {history}")
    
    metrics = model.evaluate(X_test, y_test_scaled, verbose=0)

    if model_category == 'regression':
        y_pred = model.predict(X_test, verbose=0).reshape(-1)
        y_pred = _restore_data(y_pred)

        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        r2 = r2_score(Y_test, y_pred)
        mae = mean_absolute_error(Y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
        print(f"R2: {r2:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
    else:
        # 分类任务：拿到sigmoid输出概率，sklearn计算指标
        y_pred_prob = model.predict(X_test, verbose=0).reshape(-1)
        y_pred = (y_pred_prob > 0.5).astype(int)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
        accuracy = accuracy_score(Y_test, y_pred)
        precision = precision_score(Y_test, y_pred, zero_division=0)
        recall = recall_score(Y_test, y_pred, zero_division=0)
        auc = roc_auc_score(Y_test, y_pred_prob)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"AUC: {auc:.4f}")
    return Y_test, y_pred