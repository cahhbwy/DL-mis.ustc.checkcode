# coding:utf-8
# DCGAN without batch normalization
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw


def discriminator(image, label, reuse=None):
    with tf.variable_scope("discriminator", reuse=reuse):
        d_01 = tf.layers.conv2d(image, 80, 5, (1, 1), "valid", activation=tf.nn.relu, name="dis_01")  # 16x16x80
        d_03 = tf.layers.max_pooling2d(d_01, 2, 2, "valid", name="dis_03")  # 8x8x80
        d_04 = tf.layers.conv2d(d_03, 200, 5, (1, 1), "valid", activation=tf.nn.relu, name="dis_04")  # 4x4x200
        d_06 = tf.layers.max_pooling2d(d_04, 2, 2, "valid", name="dis_06")  # 2x2x200
        d_07 = tf.layers.flatten(d_06, name="dis_07")  # 800
        d_08 = tf.layers.dense(d_07, 2000, activation=tf.nn.relu, name="dis_08")  # 2000
        d_09 = tf.layers.dropout(d_08, 0.2, name="dis_09")  # 2000
        d_10 = tf.layers.dense(d_09, 1, name="dis_10")  # 1
        loss = tf.losses.sigmoid_cross_entropy(label, d_10)
        return loss


def generator_0(rand, reuse=None):
    with tf.variable_scope("generator", reuse=reuse):
        g_00 = tf.layers.dense(rand, 5 * 5 * 80, activation=tf.nn.leaky_relu, name="gen_00")  # 2000
        g_01 = tf.reshape(g_00, [-1, 5, 5, 80], name="gen_01")  # 5x5x80
        g_02 = tf.layers.conv2d_transpose(g_01, 40, 5, (2, 2), "same", activation=tf.nn.leaky_relu, name="gen_02")  # 10x10x40

        g_03 = tf.layers.dense(rand, 10 * 10 * 40, activation=tf.nn.leaky_relu, name="gen_03")  # 4000
        g_04 = tf.reshape(g_03, [-1, 10, 10, 40], name="gen_04")  # 10x10x40
        g_05 = tf.concat([g_02, g_04], axis=3, name="gen_05")  # 10x10x80
        g_06 = tf.layers.conv2d_transpose(g_05, 20, 5, (2, 2), "same", activation=tf.nn.leaky_relu, name="gen_06")  # 20x20x20

        g_07 = tf.layers.dense(rand, 20 * 20 * 20, activation=tf.nn.leaky_relu, name="gen_07")  # 8000
        g_08 = tf.reshape(g_07, [-1, 20, 20, 20], name="gen_08")  # 20x20x20
        g_09 = tf.concat([g_06, g_08], axis=3, name="gen_09")  # 20x20x40
        g_10 = tf.layers.conv2d(g_09, 20, 3, (1, 1), "same", activation=tf.nn.leaky_relu, name="gen_10")  # 20x20x20
        g_11 = tf.layers.conv2d(g_10, 1, 3, (1, 1), "same", activation=tf.nn.tanh, name="gen_11")  # 20x20x1    数据范围 -1 ~ +1
        return g_11


def generator_1(rand, reuse=None):
    with tf.variable_scope("generator", reuse=reuse):
        g_00 = tf.layers.dense(rand, 2000, activation=tf.nn.leaky_relu, name="gen_00")  # 2000
        g_01 = tf.reshape(g_00, [-1, 5, 5, 80], name="gen_01")  # 5x5x80
        g_02 = tf.layers.conv2d_transpose(g_01, 40, 5, (2, 2), "same", activation=tf.nn.leaky_relu, name="gen_02")  # 10x10x40
        g_03 = tf.layers.conv2d_transpose(g_02, 20, 5, (2, 2), "same", activation=tf.nn.leaky_relu, name="gen_03")  # 20x20x20
        g_04 = tf.layers.conv2d(g_03, 10, 5, (1, 1), "same", activation=tf.nn.leaky_relu, name="gen_04")  # 20x20x10
        g_05 = tf.layers.conv2d(g_04, 1, 5, (1, 1), "same", activation=tf.nn.tanh, name="gen_05")  # 20x20x1
        return g_05


def model(rand, batch_size):
    generator = generator_1
    real_image_uint8 = tf.placeholder(tf.uint8, [None, 20, 20, 1])
    fake_image_float = generator(rand)
    fake_image_uint8 = tf.cast(tf.clip_by_value(tf.cast((fake_image_float + 1.) / 2. * 256., tf.int32), 0, 255), tf.uint8)
    real_image_float = (tf.cast(real_image_uint8, tf.float32) / 256. - 0.5) / 0.5
    dis_loss = discriminator(real_image_float, tf.ones([batch_size, 1])) + discriminator(fake_image_float, tf.zeros([batch_size, 1]), True)
    gen_loss = discriminator(fake_image_float, tf.ones([batch_size, 1]), True)
    return real_image_uint8, fake_image_uint8, dis_loss, gen_loss


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
        assert len(images) == len(labels)
        draw = ImageDraw.Draw(img)
        for idx, label in enumerate(labels):
            i = idx // num_h
            j = idx % num_v
            draw.text((j * (width + pad) + pad, i * (height + pad) + pad), label2text(label), fill=0)
    return img


def train(start_step, restore):
    rand_size = 100
    batch_size = 256

    data = np.load("data/data.npz")
    image_data = data["train_x"]
    size = len(image_data)

    m_rand = tf.placeholder(tf.float32, [None, rand_size])
    m_real_image, m_fake_image, m_dis_loss, m_gen_loss = model(m_rand, batch_size)
    tf.summary.scalar("dis_loss", m_dis_loss)
    tf.summary.scalar("gen_loss", m_gen_loss)
    merged_summary_op = tf.summary.merge_all()

    dis_vars = [var for var in tf.trainable_variables() if "dis" in var.name]
    gen_vars = [var for var in tf.trainable_variables() if "gen" in var.name]

    global_step = tf.Variable(0, trainable=False)
    dis_lr = tf.train.exponential_decay(learning_rate=0.00004, global_step=global_step, decay_steps=100, decay_rate=0.99)
    gen_lr = tf.train.exponential_decay(learning_rate=0.00020, global_step=global_step, decay_steps=100, decay_rate=0.95)

    dis_op = tf.train.AdamOptimizer(dis_lr).minimize(m_dis_loss, var_list=dis_vars)
    gen_op = tf.train.AdamOptimizer(gen_lr).minimize(m_gen_loss, var_list=gen_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("log", sess.graph)
    saver = tf.train.Saver(max_to_keep=10)
    if restore:
        saver.restore(sess, "model/model.ckpt-%d" % start_step)

    n_dis = 1
    n_gen = 5
    v_sample_rand = np.random.uniform(-1., 1., (batch_size, rand_size))
    for step in range(start_step, 5001):
        if step % 10 == 0:
            index = np.random.choice(size, batch_size)
            v_real_image = image_data[index]
            v_rand = np.random.uniform(-1., 1., (batch_size, rand_size))
            merged_summary, v_dis_loss, v_gen_loss = sess.run([merged_summary_op, m_dis_loss, m_gen_loss], feed_dict={m_rand: v_rand, m_real_image: v_real_image})
            print("step %6d, dis_loss = %f, gen_loss = %f" % (step, v_dis_loss, v_gen_loss))
            summary_writer.add_summary(merged_summary, step)
        if step % 100 == 0:
            v_fake_image = sess.run(m_fake_image, feed_dict={m_rand: v_sample_rand})
            image = visualize(v_fake_image)
            image.convert("RGB").save("sample/%06d.jpg" % step)
            saver.save(sess, "model/model.ckpt", global_step=step)
        for _ in range(n_dis):
            index = np.random.choice(size, batch_size)
            v_real_image = image_data[index]
            v_rand = np.random.uniform(-1., 1., (batch_size, rand_size))
            sess.run(dis_op, feed_dict={m_rand: v_rand, m_real_image: v_real_image, global_step: step})
        for _ in range(n_gen):
            index = np.random.choice(size, batch_size)
            v_real_image = image_data[index]
            v_rand = np.random.uniform(-1., 1., (batch_size, rand_size))
            sess.run(gen_op, feed_dict={m_rand: v_rand, m_real_image: v_real_image, global_step: step})


if __name__ == '__main__':
    train(0, False)
