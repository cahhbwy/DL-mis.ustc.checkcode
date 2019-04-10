# coding:utf-8
# Conditional DCGAN
import tensorflow as tf
import numpy as np
from util import visualize
import os


def discriminator(batch_size, image, label_code, training=True, reuse=tf.AUTO_REUSE):
    ki = tf.initializers.random_normal(stddev=0.01)
    with tf.variable_scope("discriminator", reuse=reuse):
        # d_00 = tf.concat([image, tf.ones([batch_size, 20, 20, 32]) * tf.reshape(label_code, [batch_size, 1, 1, 32])], 3, name="dis_00")  # 20x20x33
        d_01 = tf.layers.conv2d(image, 32, 5, (2, 2), "same", activation=tf.nn.leaky_relu, kernel_initializer=ki, name="dis_01")  # 10x10x32
        d_02 = tf.concat([d_01, tf.ones([batch_size, 10, 10, 32]) * tf.reshape(label_code, [batch_size, 1, 1, 32])], 3, name="dis_02")  # 10x10x64
        d_03 = tf.layers.conv2d(d_02, 64, 5, (2, 2), "same", activation=tf.nn.leaky_relu, kernel_initializer=ki, name="dis_03")  # 5x5x64
        d_04 = tf.layers.flatten(d_03, name="dis_04")  # 1600
        # d_05 = tf.concat([d_04, label_code], 1, name="dis_05")  # 1632
        d_06 = tf.layers.dense(d_04, 128, kernel_initializer=ki, name="dis_06")  # 128
        d_07 = tf.layers.batch_normalization(d_06, momentum=0.9, training=training, name="dis_07")  # 128
        d_08 = tf.nn.leaky_relu(d_07, name="dis_08")
        d_09 = tf.concat([d_08, label_code], 1, name="dis_09")  # 160
        d_10 = tf.layers.dense(d_09, 1, kernel_initializer=ki, name="dis_10")  # 1
        return d_10


def generator(batch_size, rand_z, label_code, training=True, reuse=tf.AUTO_REUSE):
    ki = tf.initializers.random_normal(stddev=0.01)
    with tf.variable_scope("generator", reuse=reuse):
        g_00 = tf.concat([rand_z, label_code], 1, name="gen_00")  # rand_z_size + 32
        g_01 = tf.layers.dense(g_00, 1600, kernel_initializer=ki, name="gen_01")  # 1600
        g_02 = tf.layers.batch_normalization(g_01, momentum=0.9, training=training, name="gen_02")  # 1600
        g_03 = tf.nn.relu(g_02, name="gen_03")  # 1600
        g_04 = tf.reshape(g_03, [-1, 5, 5, 64], name="gen_04")  # 5x5x64
        g_05 = tf.concat([g_04, tf.ones([batch_size, 5, 5, 32]) * tf.reshape(label_code, [batch_size, 1, 1, 32])], 3, name="gen_05")  # 5x5x96
        g_06 = tf.layers.conv2d_transpose(g_05, 32, 5, (2, 2), "same", kernel_initializer=ki, name="gen_06")  # 10x10x32
        g_07 = tf.layers.batch_normalization(g_06, momentum=0.9, training=training, name="gen_07")  # 10x10x32
        g_08 = tf.nn.relu(g_07, name="gen_08")  # 10x10x32
        # g_09 = tf.concat([g_08, tf.ones([batch_size, 10, 10, 32]) * tf.reshape(label_code, [batch_size, 1, 1, 32])], 3, name="gen_09")  # 10x10x64
        g_10 = tf.layers.conv2d_transpose(g_08, 1, 5, (2, 2), "same", activation=tf.nn.tanh, kernel_initializer=ki, name="gen_10")  # 20x20x32
        return g_10


def model(batch_size, rand_z_size=500):
    rand_z = tf.placeholder(tf.float32, [None, rand_z_size], name="rand_z")
    training = tf.placeholder(tf.bool)
    real_label = tf.placeholder(tf.int64, [None], name="real_label")
    real_label_code = tf.one_hot(real_label, 32, on_value=0.04, off_value=-0.04, dtype=tf.float32, name="real_label_code")
    fake_label = tf.placeholder(tf.int64, [None], name="fake_label")
    fake_label_code = tf.one_hot(fake_label, 32, on_value=0.04, off_value=-0.04, dtype=tf.float32, name="fake_label_code")
    real_image_uint8 = tf.placeholder(tf.uint8, [None, 20, 20, 1], name="real_image_uint8")
    real_image_float = tf.divide(tf.cast(real_image_uint8, tf.float32) - 128., 128., name="real_image_float")
    fake_image_float = generator(batch_size, rand_z, fake_label_code, training)
    fake_image_uint8 = tf.cast(tf.clip_by_value(tf.cast(tf.multiply(fake_image_float + 1.0, 128.), tf.int32), 0, 255), tf.uint8)
    dis_loss = tf.losses.sigmoid_cross_entropy(tf.ones([batch_size, 1]), discriminator(batch_size, real_image_float, real_label_code, training)) + \
               tf.losses.sigmoid_cross_entropy(tf.zeros([batch_size, 1]), discriminator(batch_size, fake_image_float, fake_label_code, training))
    gen_loss = tf.losses.sigmoid_cross_entropy(tf.ones([batch_size, 1]), discriminator(batch_size, fake_image_float, fake_label_code, training))
    return training, rand_z, real_image_uint8, fake_image_uint8, real_label, fake_label, dis_loss, gen_loss


def train(start_step, restore):
    batch_size = 256
    rand_z_size = 128

    data = np.load("data/data.npz")
    image_data = data["train_x"]
    label_data = data["train_y"]
    size = len(image_data)

    m_training, m_rand_z, m_real_image, m_fake_image, m_real_label, m_fake_label, m_dis_loss, m_gen_loss = model(batch_size, rand_z_size)
    tf.summary.scalar("dis_loss", m_dis_loss)
    tf.summary.scalar("gen_loss", m_gen_loss)
    merged_summary_op = tf.summary.merge_all()

    dis_vars = [var for var in tf.trainable_variables() if "dis" in var.name]
    gen_vars = [var for var in tf.trainable_variables() if "gen" in var.name]

    global_step = tf.Variable(0, trainable=False)
    dis_lr = tf.train.exponential_decay(learning_rate=0.0003, global_step=global_step, decay_steps=100, decay_rate=0.75)
    gen_lr = tf.train.exponential_decay(learning_rate=0.0003, global_step=global_step, decay_steps=100, decay_rate=0.75)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        dis_op = tf.train.AdamOptimizer(dis_lr).minimize(m_dis_loss, var_list=dis_vars)
        gen_op = tf.train.AdamOptimizer(gen_lr).minimize(m_gen_loss, var_list=gen_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("log", sess.graph)
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
    if restore:
        saver.restore(sess, "model/model.ckpt-%d" % start_step)

    v_sample_rand = np.random.uniform(-1., 1., (batch_size, rand_z_size))
    v_sample_label = np.repeat(np.arange(32), 8)
    assert len(v_sample_label) == batch_size
    n_dis = 1
    n_gen = 5
    for step in range(start_step, 2001):
        if step % 10 == 0:
            index = np.random.choice(size, batch_size)
            v_real_image = image_data[index]
            v_real_label = label_data[index]
            v_fake_label = np.random.randint(0, 32, batch_size)
            v_rand_z = np.random.uniform(-1., 1., (batch_size, rand_z_size))
            merged_summary, v_dis_loss, v_gen_loss = sess.run([merged_summary_op, m_dis_loss, m_gen_loss], feed_dict={m_rand_z: v_rand_z, m_real_image: v_real_image, m_real_label: v_real_label, m_fake_label: v_fake_label, m_training: False})
            print("step %6d, dis_loss = %f, gen_loss = %f" % (step, v_dis_loss, v_gen_loss))
            summary_writer.add_summary(merged_summary, step)
        if step % 100 == 0:
            v_fake_image = sess.run(m_fake_image, feed_dict={m_rand_z: v_sample_rand, m_fake_label: v_sample_label, m_training: False})
            image = visualize(v_fake_image)
            image.convert("RGB").save("sample/%06d.jpg" % step)
            saver.save(sess, "model/model.ckpt", global_step=step)
        for _ in range(n_dis):
            index = np.random.choice(size, batch_size)
            v_real_image = image_data[index]
            v_real_label = label_data[index]
            v_fake_label = np.random.randint(0, 32, batch_size)
            v_rand_z = np.random.uniform(-1., 1., (batch_size, rand_z_size))
            sess.run(dis_op, feed_dict={m_rand_z: v_rand_z, m_real_image: v_real_image, m_real_label: v_real_label, m_fake_label: v_fake_label, global_step: step, m_training: True})
        for _ in range(n_gen):
            index = np.random.choice(size, batch_size)
            v_real_image = image_data[index]
            v_real_label = label_data[index]
            v_fake_label = np.random.randint(0, 32, batch_size)
            v_rand_z = np.random.uniform(-1., 1., (batch_size, rand_z_size))
            sess.run(gen_op, feed_dict={m_rand_z: v_rand_z, m_real_image: v_real_image, m_real_label: v_real_label, m_fake_label: v_fake_label, global_step: step, m_training: True})


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    train(0, False)
