# coding:utf-8
# DCGAN with batch normalization
import tensorflow as tf
import numpy as np
from util import visualize
import os


def discriminator(image, training=True, reuse=tf.AUTO_REUSE):
    ki = tf.initializers.random_normal(stddev=0.01)
    with tf.variable_scope("discriminator", reuse=reuse):
        d_00 = tf.layers.conv2d(image, 32, 5, (2, 2), "same", activation=tf.nn.leaky_relu, kernel_initializer=ki, name="dis_00")  # 10x10x32
        d_01 = tf.layers.conv2d(d_00, 64, 5, (2, 2), "same", activation=tf.nn.leaky_relu, kernel_initializer=ki, name="dis_01")  # 5x5x64
        d_02 = tf.layers.flatten(d_01, name="dis_02")  # 1600
        d_03 = tf.layers.dense(d_02, 128, name="dis_03")  # 128
        d_04 = tf.layers.batch_normalization(d_03, training=training, name="dis_04")  # 128
        d_05 = tf.nn.leaky_relu(d_04, name="dis_05")  # 128
        d_06 = tf.layers.dense(d_05, 1, kernel_initializer=ki, name="dis_06")  # 1
        return d_06


def generator(rand_z, training=True, reuse=tf.AUTO_REUSE):
    ki = tf.initializers.random_normal(stddev=0.01)
    with tf.variable_scope("generator", reuse=reuse):
        g_00 = tf.layers.dense(rand_z, 1600, kernel_initializer=ki, name="gen_00")  # 1600
        g_01 = tf.layers.batch_normalization(g_00, training=training, name="gen_01")  # 1600
        g_02 = tf.nn.relu(g_01, name="gen_02")  # 1600
        g_03 = tf.reshape(g_02, [-1, 5, 5, 64], name="gen_03")  # 5x5x64
        g_04 = tf.layers.conv2d_transpose(g_03, 32, 5, (2, 2), "same", kernel_initializer=ki, name="gen_04")  # 10x10x32
        g_05 = tf.layers.batch_normalization(g_04, training=training, name="gen_05")  # 10x10x32
        g_06 = tf.nn.relu(g_05, name="gen_06")  # 10x10x32
        g_07 = tf.layers.conv2d_transpose(g_06, 1, 5, (2, 2), "same", activation=tf.nn.sigmoid, kernel_initializer=ki, name="gen_07")  # 20x20x1
        return g_07


def model(batch_size: int, rand_z_size: int):
    rand_z = tf.placeholder(tf.float32, [None, rand_z_size], name="rand_z")
    training = tf.placeholder(tf.bool)
    real_image_uint8 = tf.placeholder(tf.uint8, [None, 20, 20, 1], name="real_image_uint8")
    real_image_float = tf.divide(tf.cast(real_image_uint8, tf.float32), 256., name="real_image_float")
    fake_image_float = generator(rand_z, training)
    fake_image_uint8 = tf.cast(tf.clip_by_value(tf.cast(tf.multiply(fake_image_float, 256.), tf.int32), 0, 255), tf.uint8)
    dis_loss = tf.losses.sigmoid_cross_entropy(tf.ones([batch_size, 1]), discriminator(real_image_float, training)) + \
               tf.losses.sigmoid_cross_entropy(tf.zeros([batch_size, 1]), discriminator(fake_image_float, training))
    gen_loss = tf.losses.sigmoid_cross_entropy(tf.ones([batch_size, 1]), discriminator(fake_image_float, training))
    return training, rand_z, real_image_uint8, fake_image_uint8, dis_loss, gen_loss


def train(start_step, restore):
    batch_size = 64
    rand_z_size = 128

    data = np.load("data/data.npz")
    image_data = data["train_x"]
    size = len(image_data)

    m_training, m_rand_z, m_real_image, m_fake_image, m_dis_loss, m_gen_loss = model(batch_size, rand_z_size)
    tf.summary.scalar("dis_loss", m_dis_loss)
    tf.summary.scalar("gen_loss", m_gen_loss)
    merged_summary_op = tf.summary.merge_all()

    dis_vars = [var for var in tf.trainable_variables() if "dis" in var.name]
    gen_vars = [var for var in tf.trainable_variables() if "gen" in var.name]

    global_step = tf.Variable(0, trainable=False)
    dis_lr = tf.train.exponential_decay(learning_rate=0.0002, global_step=global_step, decay_steps=100, decay_rate=0.95)
    gen_lr = tf.train.exponential_decay(learning_rate=0.0005, global_step=global_step, decay_steps=100, decay_rate=0.95)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        dis_op = tf.train.AdamOptimizer(dis_lr, beta1=0.5, beta2=0.9).minimize(m_dis_loss, var_list=dis_vars)
        gen_op = tf.train.AdamOptimizer(gen_lr, beta1=0.5, beta2=0.9).minimize(m_gen_loss, var_list=gen_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("log", sess.graph)
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
    if restore:
        saver.restore(sess, "model/model.ckpt-%d" % start_step)

    n_dis = 1
    n_gen = 5
    v_sample_rand = np.random.uniform(-1., 1., (256, rand_z_size))
    for step in range(start_step, 10001):
        if step % 10 == 0:
            index = np.random.choice(size, batch_size)
            v_real_image = image_data[index]
            v_rand_z = np.random.uniform(-1., 1., (batch_size, rand_z_size))
            merged_summary, v_dis_loss, v_gen_loss = sess.run([merged_summary_op, m_dis_loss, m_gen_loss], feed_dict={m_rand_z: v_rand_z, m_real_image: v_real_image, m_training: False})
            print("step %6d, dis_loss = %f, gen_loss = %f" % (step, v_dis_loss, v_gen_loss))
            summary_writer.add_summary(merged_summary, step)
        if step % 100 == 0:
            v_fake_image = sess.run(m_fake_image, feed_dict={m_rand_z: v_sample_rand, m_training: False})
            image = visualize(v_fake_image)
            image.convert("RGB").save("sample/%06d.jpg" % step)
            saver.save(sess, "model/model.ckpt", global_step=step)
        for _ in range(n_dis):
            index = np.random.choice(size, batch_size)
            v_real_image = image_data[index]
            v_rand_z = np.random.uniform(-1., 1., (batch_size, rand_z_size))
            sess.run(dis_op, feed_dict={m_rand_z: v_rand_z, m_real_image: v_real_image, global_step: step, m_training: True})
        for _ in range(n_gen):
            index = np.random.choice(size, batch_size)
            v_real_image = image_data[index]
            v_rand_z = np.random.uniform(-1., 1., (batch_size, rand_z_size))
            sess.run(gen_op, feed_dict={m_rand_z: v_rand_z, m_real_image: v_real_image, global_step: step, m_training: True})


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    train(0, False)
