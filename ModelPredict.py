import numpy as np
import pickle
import glob
import random
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn import metrics

MINI_BATCH = 300
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

DATASET_NAME = ''
MODEL_NAME = ''
LABEL_NAME = ''
LABEL_CLASS = {}
CLASS_NUM = 0


def model_structure():
    if MODEL_NAME == 'MyFusionNet':
        from fusion_model import MyFusionNet
        model, _ = MyFusionNet(MODEL_NAME).fusion()
        return model
    elif MODEL_NAME == 'MyFusionNet2':
        from fusion_model import MyFusionNet2
        model, _ = MyFusionNet2(MODEL_NAME).fusion()
        return model
    raise ValueError(f"model name :{MODEL_NAME} is invalid")


def choose_label_name():
    global LABEL_NAME, LABEL_CLASS, CLASS_NUM
    print(MODEL_NAME, DATASET_NAME)
    if MODEL_NAME == 'MyFusionNet' or MODEL_NAME == 'MyFusionNet2':
        if DATASET_NAME == 'CICIDS2017':
            LABEL_CLASS = {0: 'Normal', 1: 'BruteForce', 2: 'DoS', 3: 'DDoS', 4: 'PortScan'}
            LABEL_NAME = ['Normal', 'BruteForce', 'DoS', 'DDoS', 'PortScan']
            CLASS_NUM = len(LABEL_CLASS)
        elif DATASET_NAME == 'ISCX2012':
            LABEL_CLASS = {0: 'Normal', 1: 'BFSSH', 2: 'Infilt', 3: 'HttpDoS', 4: 'DDoS'}
            LABEL_NAME = ['Normal', 'BFSSH', 'Infilt', 'HttpDoS', 'DDoS']
            CLASS_NUM = len(LABEL_CLASS)
        else:
            raise ValueError(f"MODEL_NAME:{MODEL_NAME} & DATASET_NAME: {DATASET_NAME} is not supported")
    else:
        raise ValueError(f"MODEL_NAME:{MODEL_NAME} is not supported in choose_label_name")


def load_data_p(test_dir_path):
    '''
    仅供predict时load data用
    :return:
    '''
    start_time = timer()
    # label_list = []
    # v1_list = []
    # v2_list = []
    # v3_list = []
    # label_packets = []

    label = np.load(test_dir_path + 'label.npy')
    v1 = np.load(test_dir_path + 'vector1.npy')
    v2 = np.load(test_dir_path + 'vector2.npy')
    v3 = np.load(test_dir_path + 'vector3.npy')
    # for i in range(len(label)):
    #     label_packets.append([label[i], v1[i], v2[i], v3[i]])
    # random.shuffle(label_packets)
    # for i in label_packets:
    #     label_list.append(i[0])
    #     v1_list.append(i[1])
    #     v2_list.append(i[2])
    #     v3_list.append(i[3])
    print('label_num:', len(label))
    # print('v1_packet_num: %d, v2: %d, v3: %d' % (len(v1_list), len(v2_list), len(v3_list)))
    end_time = timer()
    print(f'Time cost of loading data:{end_time - start_time} seconds')
    # 注意输出一定要是NUMPY ARRAY
    return label, v1, v2, v3


def update_confusion_matrix(confusion_matrix, actual_lb, predict_lb):
    # 真实值和预测值为两个轴，每出现一个坐标，给对应位置加1
    for idx, value in enumerate(actual_lb):
        p_value = predict_lb[idx]
        confusion_matrix[value, p_value] += 1
    return confusion_matrix


def draw_CM_2(cm, labels_name, title, isChinese=False, title_size=20, label_size=18, isManyClasses=False):
    '''
    第二种风格的CM图
    :return:
    '''
    # 打印NumPy数组时，浮点数将会被四舍五入到小数点后两位。
    np.set_printoptions(precision=2)
    # 归一化操作
    # .astype('float')将混淆矩阵cm的数据类型转换为浮点数。
    # .sum(axis=1)计算混淆矩阵每一行的和。
    # [:, np.newaxis]将和的结果转换为列向量。
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure(figsize=(12, 8), dpi=120)
    ind_array = np.arange(len(labels_name))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.001:
            if y_val == x_val:
                plt.text(x_val, y_val, "%0.3f" % (c,), color='white', fontsize=7 if isManyClasses else 15, va='center',
                         ha='center', fontdict={'family': 'Times New Roman'})
            else:
                plt.text(x_val, y_val, "%0.3f" % (c,), color='black', fontsize=7 if isManyClasses else 15, va='center',
                         ha='center', fontdict={'family': 'Times New Roman' })
    # offset the tick
    tick_marks = np.array(range(len(labels_name))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    # plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.binary)
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontdict={'weight': 'normal', 'size': title_size})
    plt.colorbar()
    xlocations = np.array(range(len(labels_name)))
    plt.xticks(xlocations, labels_name, fontproperties='Times New Roman', rotation=45, ha='right')
    plt.yticks(xlocations, labels_name, fontproperties='Times New Roman')
    plt.tick_params(labelsize=15)

    if isChinese:
        plt.ylabel('真实标签', fontdict={'family': 'SimSun', 'weight': 'normal', 'size': label_size})
        plt.xlabel('预测标签', fontdict={'family': 'SimSun', 'weight': 'normal', 'size': label_size})
    else:
        plt.ylabel('True label', fontdict={'weight': 'normal', 'size': label_size})
        plt.xlabel('Predicted label', fontdict={'weight': 'normal', 'size': label_size})
    # show confusion matrix
    # fig.autofmt_xdate()  # x轴斜着打印
    plt.savefig(f'./evaluation/{MODEL_NAME}_Confusion_Matrix_style2_{DATASET_NAME}.png', format='png', dpi=600)
    plt.show()
    # disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot().figure_.show()


def complement_minor(matrix, classnum):
    '''
    解析混淆矩阵，返回TN,TP，FP,FN四元列表
    此时的正常流量就是选中的流量，恶意流量就是其他的流量
    TP是恶意流量被正确识别的数量，那么TN就是正常流量被正确识别的数量，
    FN是恶意流量被错误识别为正常流量的数量，FP是正常流量被错误识别为恶意流量的数量
    :param matrix:
    :return:
    '''
    TN_list = []
    TP_list = []
    FP_list = []
    FN_list = []
    for i in range(classnum):
        TN = matrix[:i, :].sum() + matrix[i + 1:, :].sum() - matrix[:, i].sum() + matrix[i, i].sum()
        TP = matrix[i, i]
        FP = matrix[i].sum() - matrix[i, i].sum()
        FN = matrix[:, i].sum() - matrix[i, i].sum()
        TN_list.append(TN)
        TP_list.append(TP)
        FP_list.append(FP)
        FN_list.append(FN)
        print(f'{LABEL_CLASS[i]}的TN数为{TN}，TP数为{TP},FP数为{FP},FN数为{FN}')
    return TN_list, TP_list, FP_list, FN_list


def get_result_cm(confuse_matrix, classnum):
    '''

    :param confuse_matrix:
    :return:
    '''
    total_element_sum = sum(map(sum, confuse_matrix))
    ma_len = len(confuse_matrix)
    # print(ma_len)
    '''
    以下说明及注释部分的代码仅适用于ISCX2012数据集
    虽然共有5种流量，但是如果把流量看成两类，即正常流量和恶意流量，则可以用二分类方法得到
    overall_accuracy,overall_tpr,overall_fpr
    其中tpr= TP/(TP+FN)  fpr= FP/(FP+TN) ，这里的TP是恶意流量被正确识别的数量，那么TN就是正常流量被正确识别的数量，
    FN是恶意流量被错误识别为正常流量的数量，FP是正常流量被错误识别为恶意流量的数量
    '''

    overall_TP = confuse_matrix[1:, 1:].sum()
    overall_FN = confuse_matrix[0].sum() - confuse_matrix[0, 0].sum()
    overall_TN = confuse_matrix[0, 0].sum()
    overall_FP = confuse_matrix[:, 0].sum() - confuse_matrix[0, 0].sum()

    print(f'overall_TP={overall_TP}, overall_FN={overall_FN}, overall_TN={overall_TN}, overall_FP={overall_FP}')

    overall_DR = overall_TP / (overall_TP + overall_FN)
    overall_accuracy = (overall_TP + overall_TN) / (overall_FN + overall_TN + overall_FP + overall_TP)
    overall_FAR = overall_FP / (overall_FP + overall_TN)
    overall2write = "overall_DR=%f,overall_accuracy=%f,overall_FAR=%f" % (overall_DR, overall_accuracy, overall_FAR)
    print(overall2write)

    TN_list, TP_list, FP_list, FN_list = complement_minor(confuse_matrix, classnum)
    result2write = []
    for i in range(ma_len):
        category = LABEL_CLASS[i]
        TP = TP_list[i]
        TN = TN_list[i]
        FP = FP_list[i]
        FN = FN_list[i]
        precision = TP / (TP + FP)
        if (TP + FN) == 0:
            DetectionRate == 0
        else:
            DetectionRate = TP / (TP + FN)  # 实际上为recall
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        Falsealarmrate = FP / (FP + TN)  # FAR
        F1_score = 2 * precision * DetectionRate / (precision + DetectionRate)
        result = "category \"%s\" result:TP=%d,FP=%d,TN=%d,FN=%d,precision=%f,DetectionRate=%f,accuracy=%f,FAR=%f,F1-score=%f \n" % (
            category, TP, FP, TN, FN, precision, DetectionRate, accuracy, Falsealarmrate, F1_score)
        print(result)
        result2write.append(result)
    with open('evaluation/{}_evaluation_result.txt'.format(MODEL_NAME), 'w') as f:
        f.writelines(result2write)
        f.write(overall2write)


def data_generator(input1_data, input2_data, input3_data, batch_size):
    while True:
        for i in range(0, len(input1_data), batch_size):
            batch_input1 = input1_data[i:i + batch_size]
            batch_input2 = input2_data[i:i + batch_size]
            batch_input3 = input3_data[i:i + batch_size]
            yield [batch_input1, batch_input2, batch_input3]


def model_predictor(weight_path, test_data_path, model_name, dataset_name, isChinese=False):
    global DATASET_NAME
    global MODEL_NAME
    DATASET_NAME = dataset_name
    MODEL_NAME = model_name
    choose_label_name()  # 根据模型选择数据标签字典
    pred_start_time = timer()
    model = model_structure()
    model.load_weights(weight_path)
    label_list, v1_list, v2_list, v3_list = load_data_p(test_data_path)

    total_test_num = len(label_list)
    print('Total Count: %d' % total_test_num)
    # data_single_one = []
    # data_all = []
    # for i in range(len(v1_list)):
    #     # data_single_one.append([v1_list[i], v2_list[i], v3_list[i]])
    #     data_all.append([v1_list[i], v2_list[i], v3_list[i]])
    # print(len(data_all))
    # print(len(data_all[0]))
    print(len(v1_list), len(v2_list), len(v3_list))
    pred_raw = model.predict([v1_list, v2_list, v3_list], verbose=1)  # 未经处理的pred数据
    print('Shape:', pred_raw.shape)
    print(pred_raw)
    pred_label = np.argmax(pred_raw, axis=1)  # 经过处理的pred数据，取pred_raw中值最大的元素的索引为label值
    pred_end_time = timer()
    pred_efficiency = (pred_end_time - pred_start_time) / total_test_num * 10000
    print(f'预测时间为{pred_efficiency} seconds /10000 samples')

    label = np.array(label_list, dtype=np.int8)
    init_matrix = np.zeros((CLASS_NUM, CLASS_NUM), dtype=int)
    confuse_matrix = update_confusion_matrix(init_matrix, label, pred_label)  # 生成混淆矩阵
    if isChinese:
        manyclasses = True if DATASET_NAME == 'USTC-TFC2016' else False
        # draw_CM_2(confuse_matrix, LABEL_NAME, f'{dataset_name}测试集上归一化的混淆矩阵', isChinese=True,
        #           isManyClasses=manyclasses)
        draw_CM_2(confuse_matrix, LABEL_NAME, f'', isChinese=True,
                  isManyClasses=manyclasses)
    else:
        draw_CM_2(confuse_matrix, LABEL_NAME, f'Normalized Confusion Matrix of {DATASET_NAME} Test Dataset')
    print(confuse_matrix)
    with open(f'evaluation/{MODEL_NAME}-confuse_matrix.pkl', 'wb') as f:
        pickle.dump(confuse_matrix, f)
        f.close()

    get_result_cm(confuse_matrix, classnum=CLASS_NUM)

    # 保存预测值和真实值
    with open(f'evaluation/{MODEL_NAME}_pred_raw.pkl', 'wb') as f:
        pickle.dump(pred_raw, f)
        print('pred_raw保存成功')
        f.close()
    with open(f'evaluation/{MODEL_NAME}_labels.pkl', 'wb') as f:
        pickle.dump(label, f)
        print('label保存成功')
        f.close()


def model_saver(weight_path, save_path):
    model = model_structure()
    model.load_weights(weight_path)
    model.save(save_path, save_format='tf')


if __name__ == '__main__':
    # model_predictor(weight_path='checkpoints/MyFusionNet_Apr26_K4_17_0.03.hdf5',
    #                 test_data_path='H:/dataset/CIC-IDS-2017/FusionDataset/test/', model_name='MyFusionNet',
    #                 dataset_name='CICIDS2017', isChinese=True)
    model_predictor(weight_path='checkpoints-new-2012/MyFusionNet_Mar13_K0_10_0.07.hdf5',
                    test_data_path=r'D:\360MoveData\Users\LENOVO\Desktop\大创\数据集\ISCX2012\test_test/', model_name='MyFusionNet',
                    dataset_name='ISCX2012', isChinese=True)
    # model_predictor(weight_path='checkpoints-new-2017/MyFusionNet2_Feb04_K0_02_8.20.hdf5',
    #                 test_data_path='D:\\360MoveData\\Users\\LENOVO\\Documents\\WeChat Files\\wxid_rzqu5f9qoazm22\\FileStorage\\File\\2024-02\\dataset/', model_name='MyFusionNet2',
    #                 dataset_name='CICIDS2017', isChinese=True)
    MODEL_NAME = 'MyFusionNet2'
    # model_saver(weight_path='checkpoints/MyFusionNet_Apr26_K4_17_0.03.hdf5',
    #                 save_path="G:/modelSave/flow1")
    pass

