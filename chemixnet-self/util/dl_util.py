from keras.src.legacy.preprocessing.text import Tokenizer

def standardize(data):
    """
    标准化函数：将数据按列进行标准化, 使每一列的均值为0，标准差为1
    :param data: 输入为numpy数组
    :return: 标准化后的数据、均值、标准差
    """
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std == 0] = 1  # 防止出现除以零
    standardized_data = (data - mean) / std
    return standardized_data, mean, std

def normalize(data):
    """
    归一化函数：将数据按列归一化到[0,1]区间
    :param data: 输入为numpy数组
    :return: 归一化后的数据、最小值、最大值
    """
    min_val = data.min(axis=0)
    max_val = data.max(axis=0)
    range_val = max_val - min_val
    range_val[range_val == 0] = 1  # 防止出现除以零
    normalized_data = (data - min_val) / range_val
    return normalized_data, min_val, max_val


def generate_math_code_tokenizer(docs, encode_type = None):
    # 这句代码的作用是创建一个 Keras 的分词器（Tokenizer）对象，
    # 参数 char_level=True 表示以字符为单位进行分词（即将文本拆解成单个字符）；
    # lower=False 表示不将文本转换为小写（即保持原有大小写）；
    # filters='' 表示不过滤任何字符（即不会去除文本中的任何符号或标点）。
    t = Tokenizer(char_level=True, lower=False, filters='')
    t.fit_on_texts(docs)

    
    if encode_type == 'vocab_size':
        return len(t.word_index) + 1
    
    encoded_docs = t.texts_to_sequences(docs)
    return encoded_docs
    