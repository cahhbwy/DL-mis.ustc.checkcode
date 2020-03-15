# coding:utf-8
# Condition DCGAN

import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers, activations, metrics, models
from util import visualize, load_data


def make_discriminator():
    images = layers.Input(shape=(20, 20, 1), dtype=tf.float32, name='images')  # (20, 20, 1)
    labels = layers.Input(shape=(1,), dtype=tf.int32, name='labels')  # 1
    hidden = tf.concat([
        images,
        tf.reshape(tf.one_hot(labels, depth=32, on_value=1.0, off_value=-1.0), (tf.shape(labels)[0], 1, 1, 32)) * tf.ones(shape=(tf.shape(labels)[0], 20, 20, 32))
    ], axis=3)
    hidden = layers.Conv2D(filters=32, kernel_size=5, strides=(2, 2), padding='same', activation=tf.nn.leaky_relu, name='conv2d_01')(hidden)  # (10, 10, 64)
    hidden = layers.BatchNormalization(name='bn_01')(hidden)
    hidden = layers.Conv2D(filters=64, kernel_size=5, strides=(2, 2), padding='same', activation=tf.nn.leaky_relu, name='conv2d_02')(hidden)  # (5, 5, 128)
    hidden = layers.Flatten(name='flatten')(hidden)  # 1600
    hidden = layers.Dense(units=256, name='dense_01')(hidden)  # 64
    hidden = layers.BatchNormalization(name='bn_03')(hidden)  # 64
    hidden = layers.Activation(activation=tf.nn.leaky_relu)(hidden)  # 64
    output = layers.Dense(units=1, name='dense_02')(hidden)  # 1
    return models.Model(inputs=[images, labels], outputs=[output], name='discriminator')


def make_generator(noise_length):
    noises = layers.Input(shape=(noise_length,), name='noises')  # noises_shape
    labels = layers.Input(shape=(), dtype=tf.int32, name='labels')
    hidden = tf.concat([noises, tf.one_hot(labels, depth=32, on_value=1.0, off_value=-1.0)], axis=1)
    hidden = layers.Dense(units=64 * 5 * 5, activation=activations.relu, name='dense_01')(hidden)  # 1600
    hidden = layers.Reshape(target_shape=(5, 5, 64), name='reshape')(hidden)  # (5, 5, 64)
    hidden = tf.concat([hidden, tf.reshape(tf.one_hot(labels, depth=32, on_value=0.1, off_value=-0.1), (tf.shape(labels)[0], 1, 1, 32)) * tf.ones((tf.shape(labels)[0], 5, 5, 32), name="ones_02")], axis=3)
    hidden = layers.Conv2DTranspose(filters=32, kernel_size=5, strides=(2, 2), padding='same', activation=activations.relu, name='deconv2d_01')(hidden)  # (10, 10, 32)
    output = layers.Conv2DTranspose(filters=1, kernel_size=5, strides=(2, 2), padding='same', activation=activations.tanh, name='deconv2d_02')(hidden)  # (20, 20, 1)
    return models.Model(inputs=[noises, labels], outputs=[output], name='generator')


def train(start_step=0, restore=False):
    batch_size = 64
    noise_length = 128
    epochs = 16
    num_examples = 256
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sample_noise = tf.random.normal([num_examples, noise_length])
    sample_label = np.repeat(np.arange(0, 32, dtype=np.int32), 8)

    train_ds = load_data(batch_size)

    model_dis = make_discriminator()
    model_gen = make_generator(noise_length)

    lr_dis = optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0005, decay_steps=1000, decay_rate=0.95, staircase=False)
    lr_gen = optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0005, decay_steps=1000, decay_rate=0.95, staircase=False)

    optimizer_dis = optimizers.Adam(learning_rate=lr_dis, beta_1=0.5, beta_2=0.90)
    optimizer_gen = optimizers.Adam(learning_rate=lr_gen, beta_1=0.5, beta_2=0.90)

    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer_gen=optimizer_gen, optimizer_dis=optimizer_dis, model_gen=model_gen, model_dis=model_dis)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, f"model/", max_to_keep=5)

    if restore:
        try:
            checkpoint.restore(f"model/ckpt-{start_step}")
            print(f"Restored from model/ckpt-{start_step}")
        except tf.errors.NotFoundError:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            if checkpoint_manager.latest_checkpoint:
                start_step = checkpoint.step.numpy()
                print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
            else:
                start_step = 0
                print("Initializing from scratch.")

    train_loss_dis = metrics.Mean(name='train_loss_dis')
    train_loss_gen = metrics.Mean(name='train_loss_gen')

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 20, 20, 1), dtype=tf.uint8, name='image_real'),
        tf.TensorSpec(shape=(None,), dtype=tf.int32, name='label_real')
    ])
    def train_step(image_real, label_real):
        image_real = tf.subtract(tf.divide(tf.cast(image_real, tf.float32), 127.5), 1)
        noises = tf.random.normal([batch_size, noise_length])
        label_fake = tf.random.uniform(shape=(batch_size,), minval=0, maxval=32, dtype=tf.int32, name='label_fake')
        with tf.GradientTape() as gt_dis, tf.GradientTape() as gt_gen:
            image_fake = model_gen([noises, label_fake])
            output_real = model_dis([image_real, label_real])
            output_fake = model_dis([image_fake, label_fake])
            loss_dis = losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(output_real), output_real) + losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(output_fake), output_fake)
            loss_gen = losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(output_fake), output_fake)
        gradients_dis = gt_dis.gradient(loss_dis, model_dis.trainable_variables)
        gradients_gen = gt_gen.gradient(loss_gen, model_gen.trainable_variables)
        optimizer_dis.apply_gradients(zip(gradients_dis, model_dis.trainable_variables))
        optimizer_gen.apply_gradients(zip(gradients_gen, model_gen.trainable_variables))
        train_loss_dis(loss_dis)
        train_loss_gen(loss_gen)

    log_dis = f"log/{current_time}/dis"
    log_gen = f"log/{current_time}/gen"
    log_sample = f"log/{current_time}/sample"
    summary_writer_dis = tf.summary.create_file_writer(log_dis)
    summary_writer_gen = tf.summary.create_file_writer(log_gen)
    summary_writer_sample = tf.summary.create_file_writer(log_sample)

    for epoch in range(start_step, epochs):
        for image_batch, label_batch in train_ds:
            train_step(image_batch, label_batch)
        with summary_writer_dis.as_default():
            tf.summary.scalar('Discriminator Loss', train_loss_dis.result(), step=epoch)
        with summary_writer_gen.as_default():
            tf.summary.scalar('Generator Loss', train_loss_gen.result(), step=epoch)
        print(f"Epoch {epoch}, Gen Loss: {train_loss_gen.result()}, Dis Loss: {train_loss_dis.result()}")
        samples = model_gen([sample_noise, sample_label]).numpy()
        img = visualize(((samples + 1.) * 127.5).astype("uint8"))
        img.save(f"sample/{epoch:06d}.jpg")
        with summary_writer_sample.as_default():
            tf.summary.image("sample", tf.expand_dims(tf.convert_to_tensor(np.array(img)), 0), step=epoch)
        checkpoint.step.assign_add(1)
        checkpoint_manager.save()
        train_loss_dis.reset_states()
        train_loss_gen.reset_states()
    model_gen.save(f"model/{current_time}-generator.hdf5", save_format="hdf5")
    model_dis.save(f"model/{current_time}-discriminator.hdf5", save_format="hdf5")


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    train(0, False)
