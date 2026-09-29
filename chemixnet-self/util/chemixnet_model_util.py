from tensorflow.python.feature_column.sequence_feature_column import fc
from util.dl_util import *
from keras.layers import (
    LSTM, GRU, Input, concatenate, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense
)
from keras.models import Model, Sequential
from keras import Input
import keras
import tensorflow as tf



OPTIMIZER = "adam"
LR = 1e-3
EMBEDDING_DIM = 128
DROPOUT_RATE = 0.

def _determined_model_name(model_name):
    if model_name is None:
        model_name = "UNKNOWN_MODEL"
    else:
        model_name = model_name
    return model_name


def _return_task_type(task_type):
    if task_type == "regression":
        return "linear"
    elif task_type == "classification":
        return "sigmoid"
    else:
        raise ValueError(f"Invalid task type: {task_type}")

def _complete_task(model, task_type):
    if task_type == "classification":
        precision = tf.keras.metrics.Precision(name="precision")
        recall = tf.keras.metrics.Recall(name="recall")
        auc = tf.keras.metrics.AUC(name="auc")
        model.compile(
            loss="binary_crossentropy",
            optimizer=OPTIMIZER,
            metrics=["accuracy", precision, recall, auc],
        )
    else:  # regression
        model.compile(
            loss="mse",
            optimizer=OPTIMIZER,
            metrics=['mae'],
        )
    model.summary()
    return model


def CNN_MODEL(task_type, input_shape, vocab_size, model_name = 'CNN_MODEL',):
    task_type = _return_task_type(task_type)

    model_name = _determined_model_name(model_name)
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    model.add(Embedding(vocab_size, output_dim=EMBEDDING_DIM))

    model.add(Conv1D(filters=512, kernel_size=3, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=256, kernel_size=3, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(GlobalMaxPooling1D())

    model.add(Dense(units=1, activation=task_type))
    model = _complete_task(model, task_type)
    return model

def RNN_MODEL(model_name, task_type, input_shape, vocab_size, gate, units_num):
    task_type = _return_task_type(task_type)
    model_name = _determined_model_name(model_name)

    model = Sequential()
    model.add(keras.Input(shape=(input_shape,)))
    model.add(Embedding(vocab_size, EMBEDDING_DIM))

    layer_cls = {"LSTM": LSTM, "GRU": GRU}[gate]

    # 第一层RNN：return_sequences=True，输出序列
    model.add(layer_cls(units=units_num, return_sequences=True))
    # 第二层RNN：return_sequences=True
    model.add(layer_cls(units=units_num, return_sequences=True))
    # 第三层RNN：默认return_sequences=False，输出最后一个状态，不再是序列
    model.add(layer_cls(units=units_num))

    # 经过三层RNN后，最后一层默认只输出最后一个状态（不是序列），shape为(batch, units_num)，
    # 通常这种情形之后就不需要再Flatten、GlobalAveragePooling1D、GlobalMaxPooling1D等展平或池化操作了，
    # 因为输出已经是一维特征向量，适合直接送入Dense层。
    # 只有当最后一层RNN的return_sequences=True时，才需要加池化，否则不加

    model.add(Dense(1, activation=_return_task_type(task_type)))
    return _complete_task(model, task_type)
    

def mlp_model(task_type, model_name = 'MLP_MODEL', fp = 'maccs', dropout = 0.):
    task_type = _return_task_type(task_type)
    model_name = _determined_model_name(model_name)

    fc_model = Sequential()
    
    input_dim = 167 if "maccs" in fp else 1024
    
    hidden_dims = [1024, 512, 256, 128, 64, 32, 1]
    fc_model.add(Dense(hidden_dims[0], activation = tf.nn.relu, input_dim = input_dim))

    for h in hidden_dims[1:]:
        from tensorflow.keras.layers import Dropout  # 导入Dropout依赖
        fc_model.add(Dropout(dropout))
   
        fc_model.add(Dense(h, activation=tf.nn.relu))

    fc_model.add(Dense(1, activation=_return_task_type(task_type)))
    return _complete_task(fc_model, task_type)

