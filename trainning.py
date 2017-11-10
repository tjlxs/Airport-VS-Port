import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" #
import numpy as np
import tensorflow as tf
import input_data
import model
import datetime

# %%

N_CLASSES = 19
IMG_W = 300  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 300
BATCH_SIZE = 100
CAPACITY = 500
MAX_STEP = 500  # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001  # with current parameters, it is suggested to use learning rate<0.0001


# %%
def run_training():
    starttime = datetime.datetime.now()
    # you need to change the directories to yours.
    train_dir = 'D:\\DL\\RS_data\\WH-RSDataset\\RSDataset\\*\\*.jpg'
    logs_train_dir = 'D:\\DL\\Airport-VS-Port\\logs\\train'

    train_image, train_label, test_image, test_label= input_data.new_getfiles(train_dir)

    train_batch, train_label_batch = input_data.get_batch(train_image,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE,
                                                          CAPACITY)
    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.trainning(train_loss, learning_rate)
    train__acc = model.evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    # sess = tf.Session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config) # allowing dynamic memory growth as follows
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    test = coord.should_stop()
    try:
        for step in np.arange(MAX_STEP):
            if test:
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])

            if step % 1 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    # coord.join(threads)
    sess.close()
    endtime=datetime.datetime.now()
    print(endtime-starttime).seconds






