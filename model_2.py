# coding:utf-8
# DCGAN without batch normalization
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw


def discriminator(d_00, reuse=None):
    with tf.variable_scope("discriminator", reuse=reuse):
        d_01 = tf.layers.conv2d(d_00, 20, 5, (1, 1), "valid", activation=tf.nn.relu, name="dis_01")  # 16x16x20
        d_03 = tf.layers.max_pooling2d(d_01, 2, 2, "valid", name="dis_03")  # 8x8x20
        d_04 = tf.layers.conv2d(d_03, 50, 5, (1, 1), "valid", activation=tf.nn.relu, name="dis_04")  # 4x4x50
        d_06 = tf.layers.max_pooling2d(d_04, 2, 2, "valid", name="dis_06")  # 2x2x50
        d_07 = tf.layers.flatten(d_06, name="dis_07")  # 200
        d_08 = tf.layers.dense(d_07, 500, activation=tf.nn.relu, name="dis_08")  # 500
        d_09 = tf.layers.dropout(d_08, 0.2, name="dis_09")  # 500
        d_10 = tf.layers.dense(d_09, 1, name="dis_10")  # 1
        return d_10


def generator(rand, reuse=None):
    with tf.variable_scope("generator", reuse=reuse):
        g_00 = tf.layers.dense(rand, 5 * 5 * 20, activation=tf.nn.leaky_relu, name="gen_00")  # 500
        g_01 = tf.reshape(g_00, [-1, 5, 5, 20], name="gen_01")  # 5x5x20
        g_02 = tf.layers.conv2d_transpose(g_01, 10, 5, (2, 2), "same", activation=tf.nn.leaky_relu, name="gen_02")  # 10x10x10

        g_03 = tf.layers.dense(rand, 10 * 10 * 10, activation=tf.nn.leaky_relu, name="gen_03")  # 1000
        g_04 = tf.reshape(g_03, [-1, 10, 10, 10], name="gen_04")  # 10x10x10
        g_05 = tf.concat([g_02, g_04], axis=3, name="gen_05")  # 10x10x20
        g_06 = tf.layers.conv2d_transpose(g_05, 5, 5, (2, 2), "same", activation=tf.nn.leaky_relu, name="gen_06")  # 20x20x5

        g_07 = tf.layers.dense(rand, 20 * 20 * 5, activation=tf.nn.leaky_relu, name="gen_07")  # 2000
        g_08 = tf.reshape(g_07, [-1, 20, 20, 5], name="gen_08")  # 20x20x5
        g_09 = tf.concat([g_06, g_08], axis=3, name="gen_09")  # 20x20x10
        g_10 = tf.layers.conv2d(g_09, 5, 3, (1, 1), "same", activation=tf.nn.leaky_relu, name="gen_10")  # 20x20x5
        g_11 = tf.layers.conv2d(g_10, 1, 3, (1, 1), "same", activation=tf.nn.sigmoid, name="gen_11")  # 20x20x1
        return g_11


def model(batch_size):
    rand = tf.placeholder(tf.float32, [None, 100])
    real_image_uint8 = tf.placeholder(tf.uint8, [None, 20, 20, 1])
    fake_image_float = generator(rand)
    fake_image_uint8 = tf.cast(tf.multiply(fake_image_float, 255.), tf.uint8)
    real_image_float = tf.divide(tf.cast(real_image_uint8, tf.float32), 255., name="dis_00")
    real = discriminator(real_image_float)
    fake = discriminator(fake_image_float, True)
    dis_loss = tf.reduce_mean(fake) - tf.reduce_mean(real)
    gen_loss = -tf.reduce_mean(fake)
    alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
    interpolates = alpha * real_image_float + (1. - alpha) * fake_image_float
    gradients = tf.gradients(discriminator(interpolates, True), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), [1, 2, 3]))    # 除第0维之外的所有维度
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    dis_loss += 10 * gradient_penalty
    return rand, real_image_uint8, fake_image_uint8, dis_loss, gen_loss


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

    m_rand, m_real_image, m_fake_image, m_dis_loss, m_gen_loss = model(batch_size)
    tf.summary.scalar("dis_loss", m_dis_loss)
    tf.summary.scalar("gen_loss", m_gen_loss)
    merged_summary_op = tf.summary.merge_all()

    dis_vars = [var for var in tf.trainable_variables() if "dis" in var.name]
    gen_vars = [var for var in tf.trainable_variables() if "gen" in var.name]

    global_step = tf.Variable(0, trainable=False)
    dis_lr = tf.train.exponential_decay(learning_rate=0.00005, global_step=global_step, decay_steps=100, decay_rate=0.99)
    gen_lr = tf.train.exponential_decay(learning_rate=0.00025, global_step=global_step, decay_steps=100, decay_rate=0.95)

    dis_op = tf.train.RMSPropOptimizer(dis_lr).minimize(m_dis_loss, var_list=dis_vars, colocate_gradients_with_ops=True)
    gen_op = tf.train.RMSPropOptimizer(gen_lr).minimize(m_gen_loss, var_list=gen_vars, colocate_gradients_with_ops=True)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("log", sess.graph)
    saver = tf.train.Saver(max_to_keep=10)
    if restore:
        saver.restore(sess, "model/model.ckpt-%d" % start_step)

    v_sample_rand = np.random.uniform(-1., 1., (batch_size, rand_size))
    for step in range(start_step, 5001):
        if step < 5 or step % 100 == 0:
            n_dis = 100
            n_gen = 1
        else:
            n_dis = 5
            n_gen = 1
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
