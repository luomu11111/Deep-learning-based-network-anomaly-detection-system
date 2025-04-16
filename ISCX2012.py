import numpy as np
import pickle
import glob
import random
from keras import callbacks
from keras import Input, Model
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from timeit import default_timer as timer
from tensorflow.python.keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
# from tensorflow import compat
#
# config = compat.v1.ConfigProto(gpu_options=compat.v1.GPUOptions(allow_growth=True))
# sess = compat.v1.Session(config=config)


# MODEL
NETWORK_PACKETS = 16
NETWORK1_BYTES = 64
NETWORK3_FLOWSIZE = 512

MODEL_NAME = ''
MINI_BATCH = 200
DIVIDE_RATE = 1 / 9  # validation//train sample ratio in total sample
CLASS_NUM = 5
EPOCH = 30
# EPOCH = 5
STEP_PER_EPOCH = 0
VALIDATION_STEP = 0

CHECKPOINTS_DIR = 'checkpoints-new-2012/'


def load_data(root_path):
    # global重要!
    global STEP_PER_EPOCH
    global VALIDATION_STEP
    start_time = timer()
    label_list = []
    v1_list = []
    v2_list = []
    v3_list = []
    label_packets = []
    # 读取npy文件
    label = np.load(root_path + 'label.npy')
    v1 = np.load(root_path + 'vector1.npy')
    v2 = np.load(root_path + 'vector2.npy')
    v3 = np.load(root_path + 'vector3.npy')
    for i in range(len(label)):
        label_packets.append([label[i], v1[i], v2[i], v3[i]])
    # 打乱顺序
    random.shuffle(label_packets)
    for i in label_packets:
        label_list.append(i[0])
        v1_list.append(i[1])
        v2_list.append(i[2])
        v3_list.append(i[3])
    STEP_PER_EPOCH = int((len(label_list) / MINI_BATCH) * (1 - DIVIDE_RATE))# 周期步数(占九分之八)
    VALIDATION_STEP = int((len(label_list) / MINI_BATCH) * DIVIDE_RATE)# 验证步数（占九分之一）
    print('***', v1_list)
    # print(STEP_PER_EPOCH, VALIDATION_STEP)
    # print('label_num:', len(label_list))
    # print('v1_packet_num: %d, v2: %d, v3: %d' % (len(v1_list), len(v2_list), len(v3_list)))
    end_time = timer()
    print(f'Time cost of loading data:{end_time - start_time} seconds')
    return label_list, v1_list, v2_list, v3_list


def model_structure():
    # 执行fusion()与__init__()
    global MODEL_NAME
    from fusion_model import MyFusionNet
    model, model_name = MyFusionNet().fusion()
    MODEL_NAME = model_name
    return model


def dataset_generator(v1, v2, v3, label, indices, batch_size):
    # print(v1)
    # print(v2)
    # print(v3)
    v1_batch = np.zeros((batch_size, NETWORK_PACKETS, NETWORK1_BYTES))
    v2_batch = np.zeros((batch_size, NETWORK_PACKETS, 2))
    v3_batch = np.zeros((batch_size, 1, NETWORK3_FLOWSIZE))
    y_batch = np.zeros((batch_size, CLASS_NUM), dtype=np.int64)
    batch_idx = 0
    while True:
        for idx in indices:
            # print(idx)
            # 取出选定的训练集数据
            v1_batch[batch_idx] = v1[idx]
            v2_batch[batch_idx] = v2[idx]
            v3_batch[batch_idx] = v3[idx]
            # 转化分类标签为独热编码
            y_batch[batch_idx] = np_utils.to_categorical(label[idx], num_classes=CLASS_NUM)
            batch_idx += 1
            # 每batch_size个数据为一轮
            if batch_idx == batch_size:
                batch_idx = 0
                # print(v1_batch)
                # print(v2_batch)
                # print(v3_batch)
                # print(y_batch)
                # print(len(v1_batch), len(v2_batch), len(v3_batch), len(y_batch))
                # print(indices, batch_size)
                yield [v1_batch, v2_batch, v3_batch], y_batch


def callbacks_method(K):
    '''
    :param K: K-fold number
    :return:
    '''
    from tools.CallbacksMethod import CustomHistory
    import time
    # 调用CustomHistory这个自己写的回调函数
    customhistory_cb = CustomHistory(MODEL_NAME)
    
    if K != None:
        weight_file = CHECKPOINTS_DIR + str(MODEL_NAME) + '_' + str(
            time.strftime("%b%d", time.localtime())) + f'_K{K}' + '_{epoch:02d}_{val_loss:.2f}.hdf5'
    else:
        weight_file = CHECKPOINTS_DIR + str(MODEL_NAME) + '_' + str(
            time.strftime("%b%d", time.localtime())) + '_{epoch:02d}_{val_loss:.2f}.hdf5'
    # 为在每个epoch结束时保存模型的权重到一个HDF5文件中。
    # 文件的名称包含了模型的名字、当前的日期、当前的epoch数，以及验证损失（val_loss）。
    # save_best_only = False指无论val_loss是否改善，模型的权重都会被保存。
    check_cb = callbacks.ModelCheckpoint(weight_file, monitor='val_loss', verbose=0, save_best_only=False,
                                         save_weights_only=True, mode='min')
    # 为在验证损失（val_loss）连续5个epoch没有改善时停止训练。
    # 可以防止模型过拟合，
    earlystop_cb = callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    # 为将epoch的结果（如损失和准确率）记录到一个CSV文件中。
    csvlogger_cb = callbacks.CSVLogger(filename='log/{}.csv'.format(MODEL_NAME), append=True)
    return check_cb, earlystop_cb, csvlogger_cb, customhistory_cb


def train_model(model, train_data_generator, validation_data_generator, K=None):
    # 打印模型信息
    model.summary()
    # 编译模型,使用的优化器是Adam，损失函数是分类交叉熵，评估指标是准确率。
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    # model.compile("RMSProp", "categorical_crossentropy", metrics=["accuracy"])
    # 定义四个回调函数，在训练时用到
    check_cb, earlystop_cb, csvlogger_cb, customhistory_cb = callbacks_method(K)
    # # 使用生成器训练模型
    # model.fit_generator(generator=train_data_generator,
    #                     steps_per_epoch=STEP_PER_EPOCH,
    #                     epochs=EPOCH,
    #                     callbacks=[check_cb, earlystop_cb, csvlogger_cb, customhistory_cb],
    #                     validation_data=validation_data_generator,
    #                     validation_steps=VALIDATION_STEP)
    # 使用生成器训练模型
    model.fit(train_data_generator,
              workers=1,
              steps_per_epoch=STEP_PER_EPOCH,
              epochs=EPOCH,
              callbacks=[check_cb, earlystop_cb, csvlogger_cb, customhistory_cb],
              validation_data=validation_data_generator,
              validation_steps=VALIDATION_STEP)


def run(dataset_path):
    label_list, v1_list, v2_list, v3_list = load_data(dataset_path)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)# 分层五折交叉验证
    K_start = 0
    # 分成五部分，每次循环取四部分为训练集，一部分为验证集，并且每一部分里面的类别（label_list）比例都与原始数据类别比例相同
    # v1_list和label_list已经足够，因为data_generator返回的是三个数据,且索引相同
    for train_indices, test_indices in kfold.split(v1_list, label_list):
        model = model_structure()
        # 生成训练数据
        train_data_generator = dataset_generator(v1_list, v2_list, v3_list, label_list, train_indices, MINI_BATCH)
        # 生成验证数据
        validation_data_generator = dataset_generator(v1_list, v2_list, v3_list, label_list, test_indices, MINI_BATCH)
        train_model(model, train_data_generator, validation_data_generator, K_start)
        K_start += 1

    # for train_indices, test_indices in kfold.split(input_texts, target_texts):
    #     model = model_structure()
    #     # train_data_generator = dataset_generator2(input_texts, target_texts, train_indices, MINI_BATCH)
    #     # validation_data_generator = dataset_generator2(input_texts, target_texts, test_indices, MINI_BATCH)
    #     train_model(model, train_data_generator, validation_data_generator, K_start)
    #     K_start += 1


if __name__ == '__main__':
    run(r'D:\360MoveData\Users\LENOVO\Desktop\大创\数据集\ISCX2012\Dataset_test/')
