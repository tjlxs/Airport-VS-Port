import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from itertools import groupby
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" #
from collections import defaultdict



# 适用于所有类别图片在同一个文件夹中的情况，本例不使用此函数
def get_files(file_dir):
    """

    :param file_dir: file directory
    :return:list of images and labels
    """
    airport=[]
    port = []
    beach = []

    label_airport = []  # 0
    label_port = []  # 1
    label_beach = [] # 2
    for file in os.listdir(file_dir):
        name = file.split(sep='_')
        name = name.split(sep='-') # 部分文件名以'-'相隔
        if name[0]=='airport':
            airport.append(file_dir+file)
            label_airport.append(0)
        if name[0]=='port':
            port.append(file_dir+file)
            label_port.append(1)
        if name[0]=='beach':
            beach.append(file_dir+file)
            label_beach.append(2)
        # ...
        # ...
    print('There are %d airports\nThere are %d ports\nThere are %d beaches' % (len(airport), len(port),
                                                                               len(beach)))

    image_list = np.hstack((airport, port,))
    label_list = np.hstack((label_airport, label_port))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()  # 转置操作
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    """
    获得批次
    :param image:list type
    :param label:list type
    :param image_W:image width
    :param image_H:image height
    :param batch_size:batch size
    :param capacity:the maximum elements in queue
    :return:
    image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
    label_batch: 1D tensor [batch_size], dtype=tf.int32
    """

    image = tf.cast(image, tf.string) # list形式，要转换为tensorflow可识别的
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label],shuffle=True) # 因为image 和 label 是分开的

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])  # 读取队列中的图像
    image = tf.image.decode_jpeg(image_contents, channels=3)  # 解码这些图像



    """
    data argumentation should go to here
    
    """
    # image = tf.image.resize_images(image, [600,600], method=tf.image.ResizeMethod.BICUBIC)
    # #效果非常的差原因找到了，是因为resize之后返回的图像是float格式
    # 需要转化为unit8才能正常显示
    #image = tf.image.resize_images(image, [400, 400], method=tf.image.ResizeMethod.BICUBIC)
    #image = tf.cast(image, tf.uint8)


    # 如果需要填充、裁剪可以选择下面的函数
    # image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.resize_images(image,[image_W,image_H]) # 缩放图片， 以加快训练速度

    # if you want to test the generated batches of images, you might want to comment the following line.
    # 如果想看到正常的图片，请注释掉111行（标准化）和 126行（image_batch = tf.cast(image_batch, tf.float32)）
    # 训练时不要注释掉！
    image = tf.image.per_image_standardization(image) #数据标准化

    # 生成batch

    image_batch, label_batch = tf.train.batch([image,label],
                                              batch_size=batch_size,
                                              num_threads=2,
                                              capacity=capacity)
    # you can also use shuffle_batch
    #    image_batch, label_batch = tf.train.shuffle_batch([image,label],
    #                                                      batch_size=BATCH_SIZE,
    #                                                      num_threads=64,
    #                                                      capacity=CAPACITY,
    #                                                      min_after_dequeue=CAPACITY-1)

    label_batch = tf.reshape(label_batch, [batch_size]) # 讲label_batch reshape成 [batch_size]这么多行的的tensor

    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch



def new_getfiles(train_dir):

    training_image_list=[]
    training_label_list=[]
    testing_image_list=[]
    testing_label_list=[]

    image_filenames = glob.glob(train_dir)
    label_with_filedir = map(lambda filename:(filename.split('\\')[5],filename), image_filenames )
    for label, filedirs in groupby(label_with_filedir, lambda x :x[0]):
        for i, filedir in enumerate(filedirs):
            if i % 5 == 0:
                testing_label_list.append(label)
                testing_image_list.append(filedir[1])
            else:
                training_label_list.append(label)
                training_image_list.append(filedir[1])
        print("There are %d %s images " % (i,label))
    print(' %d images in testing dataset and  %d images in training dataset' % (len(testing_label_list),
                                                                                len(training_label_list)))
    # convert the label from string to int
    testing_numlabel=tf.to_int32(tf.argmax(tf.to_int32(tf.stack([tf.equal(testing_label_list, ['Airport']),
                                                                 tf.equal(testing_label_list, ['Beach']),
                                                                 tf.equal(testing_label_list, ['Bridge']),
                                                                 tf.equal(testing_label_list, ['Commercial']),
                                                                 tf.equal(testing_label_list, ['Desert']),
                                                                 tf.equal(testing_label_list, ['Farmland']),
                                                                 tf.equal(testing_label_list, ['footballField']),
                                                                 tf.equal(testing_label_list, ['Forest']),
                                                                 tf.equal(testing_label_list, ['Industrial']),
                                                                 tf.equal(testing_label_list, ['Meadow']),
                                                                 tf.equal(testing_label_list, ['Mountain']),
                                                                 tf.equal(testing_label_list, ['Park']),
                                                                 tf.equal(testing_label_list, ['Parking']),
                                                                 tf.equal(testing_label_list, ['Pond']),
                                                                 tf.equal(testing_label_list, ['Port']),
                                                                 tf.equal(testing_label_list, ['railwayStation']),
                                                                 tf.equal(testing_label_list, ['Residential']),
                                                                 tf.equal(testing_label_list, ['River']),
                                                                 tf.equal(testing_label_list, ['Viaduct'])]))))

    training_numlabel=tf.to_int32(tf.argmax(tf.to_int32(tf.stack([tf.equal(training_label_list, ['Airport']),
                                                                  tf.equal(training_label_list, ['Beach']),
                                                                  tf.equal(training_label_list, ['Bridge']),
                                                                  tf.equal(training_label_list, ['Commercial']),
                                                                  tf.equal(training_label_list, ['Desert']),
                                                                  tf.equal(training_label_list, ['Farmland']),
                                                                  tf.equal(training_label_list, ['footballField']),
                                                                  tf.equal(training_label_list, ['Forest']),
                                                                  tf.equal(training_label_list, ['Industrial']),
                                                                  tf.equal(training_label_list, ['Meadow']),
                                                                  tf.equal(training_label_list, ['Mountain']),
                                                                  tf.equal(training_label_list, ['Park']),
                                                                  tf.equal(training_label_list, ['Parking']),
                                                                  tf.equal(training_label_list, ['Pond']),
                                                                  tf.equal(training_label_list, ['Port']),
                                                                  tf.equal(training_label_list, ['railwayStation']),
                                                                  tf.equal(training_label_list, ['Residential']),
                                                                  tf.equal(training_label_list, ['River']),
                                                                  tf.equal(training_label_list, ['Viaduct'])]))))
    return training_image_list,training_numlabel,testing_image_list,testing_numlabel

def get_freeTest_images(image_files_dir):
    '''
    the freeTest images fold shoul have only image documents
    :param image_files_dir:
    :return:
    '''
    freeTest_list=[]
    for file in os.listdir(image_files_dir):
        freeTest_list.append(image_files_dir+file)

    print(freeTest_list)
    return  freeTest_list






# train_dir = 'D:\\DL\\RS_data\\WH-RSDataset\\RSDataset\\*\\*.jpg'
# train, train_label, test, test_label=new_getfiles(train_dir)
# print(train,train_label)
# sess=tf.Session()
# print(sess.run(train_label))

#  TEST
# To test the generated batches of images
# When training the model, DO comment the following codes


# BATCH_SIZE = 10
# CAPACITY = 256
# IMG_W = 500
# IMG_H = 500
#
#
# train_dir='D:\\DL\\RS_data\\WH-RSDataset\\RSDataset\\*\\*.jpg'
#
# testing_numlabel, testing_image_list, training_numlabel, training_image_list = new_getfiles(train_dir)
# image_batch, label_batch = get_batch(training_image_list, training_numlabel, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
# with tf.Session() as sess:
#     i = 0
#     coord = tf.train.Coordinator()
#
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#     test= coord.should_stop()
#
#     try:
#         while not test and i < 1:
#
#             img, label = sess.run([image_batch, label_batch])
#
#             # just test one batch
#             for j in np.arange(BATCH_SIZE):
#                 print('label: %d' %label[j])
#                 plt.figure(j)
#                 plt.imshow(img[j,:,:,:])
#                 plt.show()
#             i += 1
#
#     except tf.errors.OutOfRangeError:
#         print('done!')
#     finally:
#         coord.request_stop()
#
#     coord.join(threads)
#     sess.close()



























