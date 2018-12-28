# coding:utf-8
# Conditional WDCGAN-GP
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw


def discriminator_0(image, label, batch_size, reuse=None):
    with tf.variable_scope("discriminator", reuse=reuse):
        d_00 = tf.layers.conv2d(image, 32, 3, (1, 1), "same", activation=tf.nn.leaky_relu, name="dis_00")  # 20x20x32
        d_01 = tf.layers.max_pooling2d(d_00, 2, 2, "valid", name="dis_01")  # 10x10x32
        d_02 = tf.concat([d_01, tf.ones([batch_size, 10, 10, 32]) * tf.reshape(label, [batch_size, 1, 1, 32])], axis=3, name="dis_02")  # 10x10x64
        d_03 = tf.layers.conv2d(d_02, 32, 3, (1, 1), "same", activation=tf.nn.leaky_relu, name="dis_03")  # 10x10x32
        d_04 = tf.layers.max_pooling2d(d_03, 2, 2, "valid", name="dis_04")  # 5x5x32
        d_05 = tf.concat([d_04, tf.ones([batch_size, 5, 5, 32]) * tf.reshape(label, [batch_size, 1, 1, 32])], axis=3, name="dis_05")  # 5x5x64
        d_06 = tf.layers.conv2d(d_05, 1, 5, (1, 1), "valid", name="dis_06")  # 1x1x1
        d_07 = tf.layers.flatten(d_06, name="dis_07")  # 1
        return d_07


def generator_0(rand, label, batch_size, reuse=None):
    with tf.variable_scope("generator", reuse=reuse):
        g_00 = tf.layers.dense(rand, 5 * 5 * 96, activation=tf.nn.leaky_relu, name="gen_00")  # 2400
        g_01 = tf.reshape(g_00, [batch_size, 5, 5, 96], name="gen_01")  # 5x5x96
        g_02 = tf.concat([g_01, tf.ones([batch_size, 5, 5, 32]) * tf.reshape(label, [batch_size, 1, 1, 32])], axis=3, name="gen_02")  # 5x5x128
        g_03 = tf.layers.conv2d_transpose(g_02, 96, 5, (2, 2), "same", activation=tf.nn.leaky_relu, name="gen_03")  # 10x10x96
        g_04 = tf.concat([g_03, tf.ones([batch_size, 10, 10, 32]) * tf.reshape(label, [batch_size, 1, 1, 32])], axis=3, name="gen_04")  # 10x10x128
        g_05 = tf.layers.conv2d_transpose(g_04, 32, 5, (2, 2), "same", activation=tf.nn.leaky_relu, name="gen_05")  # 20x20x32
        g_06 = tf.concat([g_05, tf.ones([batch_size, 20, 20, 32]) * tf.reshape(label, [batch_size, 1, 1, 32])], axis=3, name="gen_06")  # 20x20x64
        g_07 = tf.layers.conv2d(g_06, 1, 3, (1, 1), "same", activation=tf.nn.tanh, name="gen_07")  # 20x20x1
        return g_07


def model(rand, batch_size):
    discriminator = discriminator_0
    generator = generator_0
    real_image_uint8 = tf.placeholder(tf.uint8, [None, 20, 20, 1])
    real_image_label = tf.placeholder(tf.float32, [None, 32])
    fake_image_label = tf.placeholder(tf.float32, [None, 32])
    fake_image_float = generator(rand, fake_image_label, batch_size)
    fake_image_uint8 = tf.cast(tf.clip_by_value(tf.cast((fake_image_float + 1.) / 2. * 256., tf.int32), 0, 255), tf.uint8)
    real_image_float = (tf.cast(real_image_uint8, tf.float32) / 256. - 0.5) / 0.5
    real = discriminator(real_image_float, real_image_label, batch_size)
    fake = discriminator(fake_image_float, fake_image_label, batch_size, True)
    # dis_loss = tf.reduce_mean(fake) - tf.reduce_mean(real) + 10. * (tf.losses.sigmoid_cross_entropy(tf.zeros([batch_size, 1]), fake) + tf.losses.sigmoid_cross_entropy(tf.ones([batch_size, 1]), real))
    # gen_loss = -tf.reduce_mean(fake) + 10. * tf.losses.sigmoid_cross_entropy(tf.ones([batch_size, 1]), fake)
    # alpha = tf.random_uniform(shape=[batch_size], minval=0., maxval=1.)
    # interpolates_image = tf.reshape(alpha, [batch_size, 1, 1, 1]) * real_image_float + (1. - tf.reshape(alpha, [batch_size, 1, 1, 1])) * fake_image_float
    # interpolates_label = tf.reshape(alpha, [batch_size, 1]) * real_image_label + (1. - tf.reshape(alpha, [batch_size, 1])) * fake_image_label
    # gradients_image, gradients_label = tf.gradients(discriminator(interpolates_image, interpolates_label, batch_size, True), [interpolates_image, interpolates_label])
    # slopes_image = tf.sqrt(tf.reduce_sum(tf.square(gradients_image), [1, 2, 3]))  # 除第0维之外的所有维度
    # slopes_label = tf.sqrt(tf.reduce_sum(tf.square(gradients_label), [1]))  # 除第0维之外的所有维度
    # gradient_penalty = tf.reduce_mean((slopes_image - 1.) ** 2) + tf.reduce_mean((slopes_label - 1.) ** 2)
    # dis_loss += 10 * gradient_penalty
    dis_loss = tf.losses.sigmoid_cross_entropy(tf.ones([batch_size, 1]), real) + tf.losses.sigmoid_cross_entropy(tf.zeros([batch_size, 1]), fake)
    gen_loss = tf.losses.sigmoid_cross_entropy(tf.ones([batch_size, 1]), fake)
    return real_image_uint8, real_image_label, fake_image_uint8, fake_image_label, dis_loss, gen_loss


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
    rand_size = 1024
    batch_size = 256

    data = np.load("data/data.npz")
    image_data = data["train_x"]
    label_data = data["train_y"]
    size = len(image_data)

    m_rand = tf.placeholder(tf.float32, [None, rand_size])
    m_real_image, m_real_label, m_fake_image, m_fake_label, m_dis_loss, m_gen_loss = model(m_rand, batch_size)
    tf.summary.scalar("dis_loss", m_dis_loss)
    tf.summary.scalar("gen_loss", m_gen_loss)
    merged_summary_op = tf.summary.merge_all()

    dis_vars = [var for var in tf.trainable_variables() if "dis" in var.name]
    gen_vars = [var for var in tf.trainable_variables() if "gen" in var.name]

    global_step = tf.Variable(0, trainable=False)
    dis_lr = tf.train.exponential_decay(learning_rate=0.0001, global_step=global_step, decay_steps=100, decay_rate=0.95)
    gen_lr = tf.train.exponential_decay(learning_rate=0.0005, global_step=global_step, decay_steps=100, decay_rate=0.95)

    # dis_op = tf.train.RMSPropOptimizer(dis_lr).minimize(m_dis_loss, var_list=dis_vars, colocate_gradients_with_ops=True)
    # gen_op = tf.train.RMSPropOptimizer(gen_lr).minimize(m_gen_loss, var_list=gen_vars, colocate_gradients_with_ops=True)
    dis_op = tf.train.AdamOptimizer(dis_lr).minimize(m_dis_loss, var_list=dis_vars, colocate_gradients_with_ops=True)
    gen_op = tf.train.AdamOptimizer(gen_lr).minimize(m_gen_loss, var_list=gen_vars, colocate_gradients_with_ops=True)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("log", sess.graph)
    saver = tf.train.Saver(max_to_keep=10)
    if restore:
        saver.restore(sess, "model/model.ckpt-%d" % start_step)

    v_sample_rand = np.random.uniform(-1., 1., (batch_size, rand_size))
    v_sample_label = np.zeros([batch_size, 32])
    v_sample_label[np.arange(batch_size), np.repeat(np.arange(32), batch_size // 32)] = 1
    n_dis = 1
    n_gen = 5
    for step in range(start_step, 5001):
        # if step < 5 or step % 100 == 0:
        #     n_dis = 100
        #     n_gen = 1
        # else:
        #     n_dis = 5
        #     n_gen = 1
        if step % 10 == 0:
            index = np.random.choice(size, batch_size)
            v_real_image = image_data[index]
            v_real_label = np.zeros([batch_size, 32])
            v_real_label[np.arange(batch_size), label_data[index]] = 1.
            v_rand = np.random.uniform(-1., 1., (batch_size, rand_size))
            v_fake_label = np.zeros([batch_size, 32])
            v_fake_label[np.arange(batch_size), np.random.randint(32, size=batch_size)] = 1
            merged_summary, v_dis_loss, v_gen_loss = sess.run([merged_summary_op, m_dis_loss, m_gen_loss], feed_dict={m_rand: v_rand, m_real_image: v_real_image, m_real_label: v_real_label, m_fake_label: v_fake_label})
            print("step %6d, dis_loss = %f, gen_loss = %f" % (step, v_dis_loss, v_gen_loss))
            summary_writer.add_summary(merged_summary, step)
        if step % 100 == 0:
            v_fake_image = sess.run(m_fake_image, feed_dict={m_rand: v_sample_rand, m_fake_label: v_sample_label})
            image = visualize(v_fake_image, np.repeat(np.arange(32), batch_size // 32), lambda x: "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"[x])
            image.convert("RGB").save("sample/%06d.jpg" % step)
            saver.save(sess, "model/model.ckpt", global_step=step)
        for _ in range(n_dis):
            index = np.random.choice(size, batch_size)
            v_real_image = image_data[index]
            v_real_label = np.zeros([batch_size, 32])
            v_real_label[np.arange(batch_size), label_data[index]] = 1.
            v_rand = np.random.uniform(-1., 1., (batch_size, rand_size))
            v_fake_label = np.zeros([batch_size, 32])
            v_fake_label[np.arange(batch_size), np.random.randint(32, size=batch_size)] = 1
            sess.run(dis_op, feed_dict={m_rand: v_rand, m_real_image: v_real_image, m_real_label: v_real_label, m_fake_label: v_fake_label, global_step: step})
        for _ in range(n_gen):
            index = np.random.choice(size, batch_size)
            v_real_image = image_data[index]
            v_real_label = np.zeros([batch_size, 32])
            v_real_label[np.arange(batch_size), label_data[index]] = 1.
            v_rand = np.random.uniform(-1., 1., (batch_size, rand_size))
            v_fake_label = np.zeros([batch_size, 32])
            v_fake_label[np.arange(batch_size), np.random.randint(32, size=batch_size)] = 1
            sess.run(gen_op, feed_dict={m_rand: v_rand, m_real_image: v_real_image, m_real_label: v_real_label, m_fake_label: v_fake_label, global_step: step})


if __name__ == '__main__':
    train(0, False)
