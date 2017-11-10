from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import input_data
import model

IMAGE_W=300 # the size of the input of THE cnn
IMAGE_H=300

# 读取随机选择的图片
def get_one_image(train):
    '''
    Randomly pick one image from training data
    Return: ndarray
    '''
    n = len(train)
    print(n)
    ind = np.random.randint(0, n)
    img_dir = train[ind]
    print(img_dir)

    image = Image.open(img_dir)
    plt.imshow(image)
    image = image.resize([IMAGE_W, IMAGE_H])
    image = np.array(image)

    return image

def prediciton_Image(image_array):
        with tf.Graph().as_default():
            BATCH_SIZE = 1
            N_CLASSES = 19

            image = tf.cast(image_array, tf.float32)
            image = tf.image.per_image_standardization(image)
            image = tf.reshape(image, [1, IMAGE_W, IMAGE_H, 3])
            logit = model.inference(image, BATCH_SIZE, N_CLASSES)

            logit = tf.nn.softmax(logit)

            x = tf.placeholder(tf.float32, shape=[IMAGE_W, IMAGE_H, 3])

            logs_train_dir = 'D:\\DL\\Airport-VS-Port\\logs\\train'

            saver = tf.train.Saver()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:

                print("Reading checkpoints...")
                ckpt = tf.train.get_checkpoint_state(logs_train_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Loading success, global_step is %s' % global_step)
                else:
                    print('No checkpoint file found')

                prediction = sess.run(logit, feed_dict={x: image_array})
                print('prediction:', prediction)
                max_index = np.argmax(prediction)
                pre_Class = {0: "Airport", 1: "Beach", 2: "Bridge", 3: "Commercial", 4: "Desert", 5: "Farmland",
                             6: "footballField",
                             7: "Forest", 8: "Industrial", 9: "Meadow", 10: "Mountain", 11: "Park", 12: "Parking",
                             13: "Pond", 14: "Port", 15: "railwayStation", 16: "Residential", 17: "River",
                             18: "Viaduct"}

                print('This is a %s with possibility %.6f' % (pre_Class[max_index], prediction[0, max_index]))



                # if max_index == 0:
                #     print('This is a Airport with possibility %.6f' % prediction[:, 0])
                # else:
                #     print('This is a Port with possibility %.6f' % prediction[:, 1])
def evaluate_one_image():
    '''Test one image against the saved models and parameters
    测试一张数据集中test部分的数据
    '''

    train_dir = 'D:\\DL\\RS_data\\WH-RSDataset\\RSDataset\\*\\*.jpg'
    train, train_label, test, test_label = input_data.new_getfiles(train_dir)
    image_array = get_one_image(test) # 从测试数据集中选取一张图片进行验证
    prediciton_Image(image_array)

def evaluate_freeTest_image():
        '''
        测试一张其他来源数据数据
        :return:
        '''
        freeTest_dir = 'C:\\Users\\gaohan\\Desktop\\test data from random source\\'
        testImagelist = input_data.get_freeTest_images(freeTest_dir)
        image_array = get_one_image(testImagelist)
        prediciton_Image(image_array)





