from keras import Input, Model
from keras.layers import Conv2D, Lambda, Activation, SeparableConv2D, MaxPooling2D, Conv1D, MaxPooling1D, \
    GlobalMaxPool1D, Dense, TimeDistributed, GRU, concatenate, Dropout, LSTM, Bidirectional, UpSampling1D, \
    BatchNormalization, Reshape, Concatenate
import tensorflow as tf
import os
from keras.utils import plot_model

NETWORK_PACKETS = 16
NETWORK1_BYTES = 64
NETWORK3_FLOWSIZE = 512


class MyFusionNet:
    def __init__(self, model_name='MyFusionNet'):
        self.model_name = model_name
    
    # 统计特征提取网络Bi-LSTM
    def network1(self, inputs):
        # seq_input = Input(shape=(NETWORK_PACKETS, NETWORK1_BYTES))
        # 使用了双向长短期记忆（BidirectionalLSTM）网络层。
        # 这个层的目标是从输入数据中学习和提取特征。
        # units = 256表示LSTM层中的神经元数量为256，return_sequences=True表示返回的是整个序列，而不仅仅是最后的输出。
        x = Bidirectional(LSTM(units=256, return_sequences=True))(inputs)
        # 使用了Dropout层，目的是在训练过程中随机丢弃一部分神经元（本例中为20%），以防止过拟合。
        x = Dropout(0.2)(x)
        # 使用了一个双向LSTM层，只返回最后的输出。
        x = Bidirectional(LSTM(units=256))(x)
        # 使用了Dropout层，目的是在训练过程中随机丢弃一部分神经元（本例中为20%），以防止过拟合。
        x = Dropout(0.2)(x)
        # 使用了全连接层（Dense），神经元数量为256，激活函数为ReLU，用于增加模型的非线性。
        x = Dense(units=256, activation='relu')(x)
        return x
        # model = Model(inputs=seq_input, outputs=x)
        # return model
    
    # 时序特征提取网络Bi-LSTM
    def network2(self, inputs):
        # seq_input = Input(shape=(NETWORK_PACKETS, 2))
        x = Bidirectional(LSTM(units=256, return_sequences=True))(inputs)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(units=256))(x)
        x = Dropout(0.2)(x)
        x = Dense(units=256, activation='relu')(x)
        return x
        # model = Model(inputs=seq_input, outputs=x)
        # return model
    
    # 空间特征提取网络Conv1D
    def network3(self, inputs):
        # seq_input = Input(shape=(1, NETWORK3_FLOWSIZE))
        x = UpSampling1D(size=16)(inputs)
        x = Conv1D(filters=500, kernel_size=3, padding='valid', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = UpSampling1D(size=2)(x)
        x = Conv1D(filters=500, kernel_size=3, padding='valid', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=256, kernel_size=3, padding='valid', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=256, kernel_size=1, padding='valid', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Reshape((256, ))(x)
        x = Dense(units=256, activation='relu')(x)
        return x
        # model = Model(inputs=seq_input, outputs=x)
        # return model

    def fusion(self):
        # 定义输入层
        input1 = Input(shape=(NETWORK_PACKETS, NETWORK1_BYTES), name='in1')
        input2 = Input(shape=(NETWORK_PACKETS, 2), name='in2')
        input3 = Input(shape=(1, NETWORK3_FLOWSIZE), name='in3')

        # 三个模型
        net1 = self.network1(input1)
        net2 = self.network2(input2)
        net3 = self.network3(input3)
        
        # 使用Keras的Concatenate函数将net1，net2和net3这三个网络的输出进行拼接。
        # 这三个网络的输出将会被合并成一个更大的向量，然后作为后续层的输入
        concat = Concatenate()([net1, net2, net3])

        x = Dropout(0.2)(concat)
        x = Dense(units=256, activation='relu')(x)
        x = Dense(units=256)(x)
        predict_layer = Dense(5, name='dense_layer', activation='softmax')(x)
        
        # 使用Keras的Model函数来创建一个模型
        model = Model(inputs=[input1, input2, input3], outputs=predict_layer)
        return model, self.model_name


class MyFusionNet2:
    def __init__(self, model_name='MyFusionNet2'):
        self.model_name = model_name

    def network1(self, inputs):
        # seq_input = Input(shape=(NETWORK_PACKETS, NETWORK1_BYTES))
        x = Bidirectional(LSTM(units=256))(inputs)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(units=256))(x)
        # x = Dropout(0.2)(x)
        # x = Dense(units=256, activation='relu')(x)
        return x
        # model = Model(inputs=seq_input, outputs=x)
        # return model, ''

    def network2(self, inputs):
        # seq_input = Input(shape=(NETWORK_PACKETS, 2))
        x = Bidirectional(LSTM(units=256, return_sequences=True))(inputs)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(units=256))(x)
        x = Dropout(0.2)(x)
        x = Dense(units=256, activation='relu')(x)
        return x
        # model = Model(inputs=seq_input, outputs=x)
        # return model

    def network3(self, inputs):
        # seq_input = Input(shape=(1, NETWORK3_FLOWSIZE))
        x = UpSampling1D(size=16)(inputs)
        x = Conv1D(filters=512, kernel_size=3, padding='valid', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = UpSampling1D(size=2)(x)
        x = Conv1D(filters=512, kernel_size=3, padding='valid', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=256, kernel_size=3, padding='valid', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=4)(x)
        # x = Conv1D(filters=256, kernel_size=1, padding='valid', activation='relu')(x)
        # x = BatchNormalization()(x)
        # x = MaxPooling1D(pool_size=2)(x)
        x = Reshape((256, ))(x)
        x = Dense(units=256, activation='relu')(x)

        return x
        # model = Model(inputs=seq_input, outputs=x)
        # return model, ''

    def fusion(self):
        input1 = Input(shape=(NETWORK_PACKETS, NETWORK1_BYTES), name='in1')
        input2 = Input(shape=(NETWORK_PACKETS, 2), name='in2')
        input3 = Input(shape=(1, NETWORK3_FLOWSIZE), name='in3')

        net1 = self.network1(input1)
        net2 = self.network2(input2)
        net3 = self.network3(input3)

        concat = Concatenate()([net1, net2, net3])

        x = Dropout(0.2)(concat)
        x = Dense(units=256, activation='relu')(x)
        x = Dense(units=256)(x)
        predict_layer = Dense(5, name='dense_layer', activation='softmax')(x)

        model = Model(inputs=[input1, input2, input3], outputs=predict_layer)
        return model, self.model_name



if __name__ == '__main__':
    model, _ = MyFusionNet2().network1('')
    model.summary()
