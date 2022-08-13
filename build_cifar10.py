import random
from pickled import *
from load_data import *

DATA_LEN = 3072
CHANNEL_LEN = 1024
SHAPE = 32  # 圖像大小


def getFirstLevelDirList(image_data_folder, save_path):
    root_dirs = []
    for root, dirs, files in os.walk(image_data_folder):
        root_dirs.append(root)
    root_dirs = root_dirs[1:]

    f = open(save_path, 'w')
    k = 0
    for root_dir in root_dirs:
        f.write('%s %s\n' % (root_dir, k))
        k = k + 1
    f.close()

    return root_dirs


def getSecondList(first_level_dir, save_path):
    j = 0
    f = open(save_path, 'w')
    for path in first_level_dir:
        for root, dirs, files in os.walk(path):
            for file in files:
                f.write('%s/%s %i\n' % (root, file, j))

        j = j + 1
    f.close()


def divideTheDataset(my_cifar_image, ratio, my_cifar_train, my_cifar_test):
    with open(my_cifar_image) as f:
        list = f.readlines()
        random.shuffle(list)
        set_num = int(float(len(list)) * ratio)
        test_list = list[:set_num]
        train_list = list[set_num:]
    f.close()

    with open(my_cifar_train, 'w') as f2:
        for i in train_list:
            f2.write(i)
    f2.close()
    with open(my_cifar_test, 'w') as f3:
        for i in test_list:
            f3.write(i)
    f3.close()


def imread(im_path, color):
    if color == "GRAYSCALE":
        im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    else:
        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
    if SHAPE is not None:
        assert isinstance(SHAPE, int)
        im = cv2.resize(im, (SHAPE, SHAPE))

    return im


def read_data(filename, color):
    if os.path.isdir(filename):
        print("Can't found data file!")
    else:
        f = open(filename)
        lines = f.read().splitlines()
        count = len(lines)
        data = np.zeros((count, DATA_LEN), dtype=np.uint8)
        lst = [ln.split(' ')[0] for ln in lines]
        new_lst = []
        for l in lst:
            a = l.split('/')
            l = a[len(a) - 1]
            new_lst.append(l)
        # print(new_lst)
        label = [int(ln.split(' ')[1]) for ln in lines]

        idx = 0
        c = CHANNEL_LEN
        for ln in lines:
            fname, lab = ln.split(' ')
            im = imread(os.path.join(fname), color)
            if color == 'GRAYSCALE':
                data[idx, :c] = np.reshape(im, c)
                label[idx] = int(lab)
                idx = idx + 1
            if color == 'RGB':
                data[idx, :c] = np.reshape(im[:, :, 0], c)
                data[idx, c:2 * c] = np.reshape(im[:, :, 1], c)
                data[idx, 2 * c:] = np.reshape(im[:, :, 2], c)
                label[idx] = int(lab)
                idx = idx + 1

        return data, label, new_lst


def pickled(savepath, data, label, fnames, bin_num=1, mode="train"):
    assert os.path.isdir(savepath)
    total_num = len(fnames)
    samples_per_bin = int(total_num / bin_num)
    assert samples_per_bin > 0
    idx = 0
    for i in range(bin_num):
        start = i * samples_per_bin
        end = (i + 1) * samples_per_bin

        if end <= total_num:
            dict = {'data': data[start:end, :],
                    'labels': label[start:end],
                    'filenames': fnames[start:end]}
        else:
            dict = {'data': data[start:, :],
                    'labels': label[start:],
                    'filenames': fnames[start:]}
        if mode == "train":
            dict['batch_label'] = "training batch {} of {}".format(idx, bin_num)
            with open(os.path.join(savepath, 'data_batch_' + str(idx)), 'wb') as fi:
                cPickle.dump(dict, fi)
            idx = idx + 1
        else:
            dict['batch_label'] = "testing batch {} of {}".format(idx, bin_num)
            with open(os.path.join(savepath, 'test_batch'), 'wb') as fi:
                cPickle.dump(dict, fi)


def unpickled(filename):
    assert os.path.isfile(filename)
    with open(filename, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def batchesMeta(batches_meta, label_list_path, my_cifar_train):
    with open(label_list_path) as f0:
        lines = f0.read().splitlines()
    f0.close()
    lst = [ln.split(' ')[0] for ln in lines]
    lable_names = []
    for l in lst:
        a = l.split('\\')
        lable_names.append(a[len(a) - 1])

    with open(my_cifar_train) as f1:
        num_cases_per_batch = len(f1.read().splitlines())
    f1.close()
    dictCow = {'num_cases_per_batch': num_cases_per_batch,  # 每个batch包含的样本数量
               'label_names': lable_names,  # 类别索引，将类别索引表（object_list.txt）中的label_names:填进去
               'num_vis': 3072}  # 这里不要动

    with open(batches_meta, 'wb') as f2:
        pickle.dump(dictCow, f2)
    f2.close()


if __name__ == '__main__':
    # 图像数据的目录
    image_data_folder = '../OriginalImageDataset/my_cifar10'
    # 保存临时文件及生成的数据集的目录
    save_path = './cifar/'
    # ratio为拆分阈值，0.2则是前20%为测试集，剩下的是训练集
    split_ratio = 0.2
    # 生成的train的batch数量
    train_batch_num = 5
    # 图像色彩模式
    COLORMODE = 'GRAYSCALE'
    # COLORMODE = 'RGB'

    # 生成label和索引的对应表
    label_list_path = f'./{save_path}/label_list'
    first_level_dir = getFirstLevelDirList(image_data_folder, label_list_path)

    # 生成图像和label索引的对应表
    my_cifar_image = f'./{save_path}/my_cifar_image'
    getSecondList(first_level_dir, my_cifar_image)

    # 划分数据集
    my_cifar_train = f'./{save_path}/my_cifar_train'
    my_cifar_test = f'./{save_path}/my_cifar_test'
    divideTheDataset(my_cifar_image, split_ratio, my_cifar_train, my_cifar_test)

    # 生成训练batch
    data, label, lst = read_data(my_cifar_train, COLORMODE)
    pickled(save_path, data, label, lst, bin_num=train_batch_num, mode='train')

    # 生成测试batch
    data, label, lst = read_data(my_cifar_test, COLORMODE)
    pickled(save_path, data, label, lst, mode='test')

    # 生成batches.meta
    batches_meta = f'{save_path}/batches.meta'
    batchesMeta(batches_meta, label_list_path, my_cifar_train)

    train = unpickled(f'{save_path}/data_batch_0')
    test = unpickled(f'{save_path}/test_batch')
    meta = unpickled(f'{save_path}/batches.meta')

    print(train.get('batch_label'))
    print(test.get('batch_label'))
    print(meta)
