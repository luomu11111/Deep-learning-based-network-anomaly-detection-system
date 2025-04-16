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

# MODEL
NETWORK_PACKETS = 16
NETWORK1_BYTES = 64
# NETWORK3_FLOWSIZE = 500
NETWORK3_FLOWSIZE = 512

MODEL_NAME = ''
MINI_BATCH = 300
DIVIDE_RATE = 1 / 9  # validation//train sample ratio in total sample
CLASS_NUM = 5
EPOCH = 20
STEP_PER_EPOCH = 0
VALIDATION_STEP = 0

CHECKPOINTS_DIR = 'checkpoints-new-2017/'


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

    label = np.load(root_path + 'label.npy')
    v1 = np.load(root_path + 'vector1.npy')
    v2 = np.load(root_path + 'vector2.npy')
    v3 = np.load(root_path + 'vector3.npy')
    for i in range(len(label)):
        label_packets.append([label[i], v1[i], v2[i], v3[i]])
    random.shuffle(label_packets)
    for i in label_packets:
        label_list.append(i[0])
        v1_list.append(i[1])
        v2_list.append(i[2])
        v3_list.append(i[3])
    STEP_PER_EPOCH = int(len(label_list) / MINI_BATCH * (1 - DIVIDE_RATE))
    VALIDATION_STEP = int(len(label_list) / MINI_BATCH * DIVIDE_RATE)
    print('label_num:', len(label_list))
    print('v1_packet_num: %d, v2: %d, v3: %d' % (len(v1_list), len(v2_list), len(v3_list)))
    end_time = timer()
    print(f'Time cost of loading data:{end_time - start_time} seconds')
    return label_list, v1_list, v2_list, v3_list


def model_structure():
    global MODEL_NAME
    # from fusion_model import MyFusionNet
    from fusion_model import MyFusionNet2
    model, model_name = MyFusionNet2().fusion()
    MODEL_NAME = model_name
    return model


def dataset_generator(v1, v2, v3, label, indices, batch_size):
    v1_batch = np.zeros((batch_size, NETWORK_PACKETS, NETWORK1_BYTES))
    v2_batch = np.zeros((batch_size, NETWORK_PACKETS, 2))
    v3_batch = np.zeros((batch_size, 1, NETWORK3_FLOWSIZE))
    y_batch = np.zeros((batch_size, CLASS_NUM), dtype=np.int64)
    batch_idx = 0
    while True:
        for idx in indices:
            v1_batch[batch_idx] = v1[idx]
            v2_batch[batch_idx] = v2[idx]
            v3_batch[batch_idx] = v3[idx]
            y_batch[batch_idx] = np_utils.to_categorical(label[idx], num_classes=CLASS_NUM)
            batch_idx += 1
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
    customhistory_cb = CustomHistory(MODEL_NAME)
    if K != None:
        weight_file = CHECKPOINTS_DIR + str(MODEL_NAME) + '_' +  str(
            time.strftime("%b%d", time.localtime())) + f'_K{K}' + '_{epoch:02d}_{val_loss:.2f}.hdf5'
    else:
        weight_file = CHECKPOINTS_DIR + str(MODEL_NAME) + '_' + str(
            time.strftime("%b%d", time.localtime())) + '_{epoch:02d}_{val_loss:.2f}.hdf5'
    check_cb = callbacks.ModelCheckpoint(weight_file, monitor='val_loss', verbose=0, save_best_only=False,
                                         save_weights_only=True, mode='min')
    earlystop_cb = callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    csvlogger_cb = callbacks.CSVLogger(filename='log/{}.csv'.format(MODEL_NAME), append=True)
    return check_cb, earlystop_cb, csvlogger_cb, customhistory_cb


def train_model(model, train_data_generator, validation_data_generator, K=None):
    model.summary()
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    # model.compile("RMSProp", "categorical_crossentropy", metrics=["accuracy"])
    check_cb, earlystop_cb, csvlogger_cb, customhistory_cb = callbacks_method(K)
    model.fit_generator(generator=train_data_generator,
                        steps_per_epoch=STEP_PER_EPOCH,
                        epochs=EPOCH,
                        callbacks=[check_cb, earlystop_cb, csvlogger_cb, customhistory_cb],
                        validation_data=validation_data_generator,
                        validation_steps=VALIDATION_STEP)


def run(dataset_path):
    label_list, v1_list, v2_list, v3_list = load_data(dataset_path)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    K_start = 0
    # v1_list和label_list已经足够，因为data_generator返回的是三个数据,且索引相同
    for train_indices, test_indices in kfold.split(v1_list, label_list):
        model = model_structure()
        train_data_generator = dataset_generator(v1_list, v2_list, v3_list, label_list, train_indices, MINI_BATCH)
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
    # run(r'G:\dataset\CIC-IDS-2017\FusionDataset\dataset/')
    run(r'D:\360MoveData\Users\LENOVO\Documents\WeChat Files\wxid_rzqu5f9qoazm22\FileStorage\File\2024-02\dataset/')