# coding:utf-8
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import os
from urllib import request


def model():
    x = tf.placeholder(tf.uint8, [None, 20, 20, 1], name="x")  # 20x20x1
    y = tf.placeholder(tf.int64, [None], name="y")
    h_00 = tf.divide(tf.cast(x, tf.float32), 256., name="h_00")  # 20x20x1
    h_01 = tf.layers.conv2d(h_00, 20, 5, (1, 1), "valid", activation=tf.nn.relu, name="conv2d_01")  # 16x16x20
    h_02 = tf.layers.max_pooling2d(h_01, 2, 2, "valid", name="pool_02")  # 8x8x20
    h_03 = tf.layers.conv2d(h_02, 50, 5, (1, 1), "valid", activation=tf.nn.relu, name="conv2d_03")  # 4x4x50
    h_04 = tf.layers.max_pooling2d(h_03, 2, 2, "valid", name="pool_04")  # 2x2x50
    h_05 = tf.layers.flatten(h_04, name="flatten_05")  # 200
    h_06 = tf.layers.dense(h_05, 500, activation=tf.nn.relu, name="dense_06")  # 500
    h_07 = tf.layers.dropout(h_06, 0.5, name="dropout_07")  # 500
    h_08 = tf.layers.dense(h_07, 32, name="dense_08")  # 32
    loss = tf.losses.sparse_softmax_cross_entropy(y, h_08)
    predict = tf.argmax(h_08, axis=1, name="predict")
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y, predict), tf.float32))
    return x, y, loss, predict, accuracy


def visualize(images, labels=None, label2text=str, height=20, width=20, channel=1, pad=1):
    """
    将多张图片连标签一起放置在一张图片上
    :param images: 多张图片数据，np.ndarry(dtype=np.uint8)
    :param labels: 图片对应标签，np.ndarry(dtype=np.int64)
    :param label2text: 标签转字符串函数
    :param height: 单张图片高度，int
    :param width: 单张图片宽度，int
    :param channel: 图片通道数，int
    :param pad: 图片边距，int
    :return: PIL.Image
    """
    size = len(images)
    num_h = int(np.ceil(np.sqrt(size)))
    num_v = int(np.ceil(size / num_h).astype(np.int))
    image = np.zeros((num_v * height + (num_v + 1) * pad, num_h * width + (num_h + 1) * pad, channel))
    for idx, img in enumerate(images):
        i = idx // num_h
        j = idx % num_v
        image[pad + i * (height + pad):pad + i * (height + pad) + height, pad + j * (width + pad):pad + j * (width + pad) + width, :] = img
    if channel == 1:
        img = Image.fromarray(image.reshape(image.shape[:-1]))
    else:
        img = Image.fromarray(image)
    if labels is not None:
        draw = ImageDraw.Draw(img)
        for idx, label in enumerate(labels):
            i = idx // num_h
            j = idx % num_v
            draw.text((j * (width + pad) + pad, i * (height + pad) + pad), label2text(label), fill=0)
    return img


def train(start_step=0, restore=False):
    data = np.load("data/data.npz")
    train_x = data["train_x"]
    train_y = data["train_y"]
    test_x = data["test_x"]
    test_y = data["test_y"]
    string = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"

    m_x, m_y, m_loss, m_predict, m_accuracy = model()

    batch_size = 64
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(learning_rate=0.0001, global_step=global_step, decay_steps=100, decay_rate=0.99)
    op = tf.train.AdamOptimizer(lr).minimize(m_loss)

    tf.summary.scalar("loss", m_loss)
    tf.summary.scalar("accuracy", m_accuracy)
    merged_summary_op = tf.summary.merge_all()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3  # 占用GPU90%的显存
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("log", sess.graph)
    saver = tf.train.Saver(max_to_keep=5)
    if restore:
        saver.restore(sess, "model/model.ckpt-%d" % start_step)
    for step in range(start_step, 3001):
        index = np.random.choice(16000, batch_size)
        if step % 10 == 0:
            merged_summary, v_loss = sess.run([merged_summary_op, m_loss], feed_dict={m_x: train_x[index], m_y: train_y[index]})
            print("step %6d, loss = %f" % (step, v_loss))
            summary_writer.add_summary(merged_summary, step)
        if step % 100 == 0:
            v_predict, v_accuracy = sess.run([m_predict, m_accuracy], feed_dict={m_x: test_x, m_y: test_y})
            image = visualize(test_x, v_predict, lambda i: string[i])
            image.convert("RGB").save("sample/%06d.jpg" % step)
            print("step %6d, accuracy = %f" % (step, v_accuracy))
            saver.save(sess, "model/model.ckpt", global_step=step)
        sess.run(op, feed_dict={m_x: train_x[index], m_y: train_y[index]})


def test():
    table = [255 if i > 140 else i for i in range(256)]
    url = "http://mis.teach.ustc.edu.cn/randomImage.do?date='" + str(np.random.randint(2147483647)) + "'"
    req = request.urlopen(url)
    try:
        request.urlretrieve(url, "/tmp/tmp.jpg")
    except IOError:
        print("IOError")
    finally:
        req.close()
    img = Image.open("/tmp/tmp.jpg").convert('L').point(table)
    images = np.zeros([4, 20, 20, 1])
    images[0, :, :, 0] = np.array(img.crop((00, 0, 20, 20)))
    images[1, :, :, 0] = np.array(img.crop((20, 0, 40, 20)))
    images[2, :, :, 0] = np.array(img.crop((40, 0, 60, 20)))
    images[3, :, :, 0] = np.array(img.crop((60, 0, 80, 20)))

    labels = np.array(list("23456789ABCDEFGHJKLMNPQRSTUVWXYZ"))

    m_x, _, _, m_predict, _ = model()

    with tf.device("/cpu"):
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, "model/model.ckpt-%d" % 3000)
        v_predict = sess.run(m_predict, feed_dict={m_x: images})
        print("".join(labels[v_predict]))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    train()
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    test()
