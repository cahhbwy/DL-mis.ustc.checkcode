# coding:utf-8
# WDCGAN-GP

import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, activations, models
from util import visualize, load_data


def make_discriminator():
    images = layers.Input(shape=(20, 20, 1), dtype=tf.float32, name='images')  # (20, 20, 1)
    hidden = layers.Conv2D(filters=64, kernel_size=5, strides=(2, 2), padding='same', activation=tf.nn.leaky_relu, name='conv2d_01')(images)  # (10, 10, 32)
    hidden = layers.Conv2D(filters=128, kernel_size=5, strides=(2, 2), padding='same', activation=tf.nn.leaky_relu, name='conv2d_02')(hidden)  # (5, 5, 64)
    hidden = layers.Flatten(name='flatten')(hidden)  # 1600
    hidden = layers.Dense(units=1024, activation=tf.nn.leaky_relu, name='dense_01')(hidden)  # 64
    output = layers.Dense(units=1, name='dense_02')(hidden)  # 1
    return models.Model(inputs=[images], outputs=[output], name='discriminator')


def make_generator(noise_length):
    noises = layers.Input(shape=(noise_length,), name='noises')  # noises_shape
    hidden = layers.Dense(units=128 * 5 * 5, activation=activations.relu, use_bias=False, name='dense_01')(noises)  # 1600
    hidden = layers.Reshape(target_shape=(5, 5, 128), name='reshape')(hidden)  # (5, 5, 64)
    hidden = layers.Conv2DTranspose(filters=64, kernel_size=5, strides=(2, 2), padding='same', activation=activations.relu, name='deconv2d_01')(hidden)  # (10, 10, 32)
    hidden = layers.Conv2DTranspose(filters=32, kernel_size=5, strides=(2, 2), padding='same', activation=activations.relu, name='deconv2d_02')(hidden)  # (10, 10, 32)
    output = layers.Conv2D(filters=1, kernel_size=5, strides=(1, 1), padding='same', activation=activations.tanh, name='conv2d_01')(hidden)  # (20, 20, 1)
    return models.Model(inputs=[noises], outputs=[output], name='generator')


def train(start_step=0, restore=False):
    batch_size = 64
    noise_length = 128
    epochs = 100
    num_examples = 64
    gradient_penalty_weight = 10.0
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    noise_seed = tf.random.normal([num_examples, noise_length])

    train_ds = load_data(batch_size, with_label=False)

    model_dis = make_discriminator()
    model_gen = make_generator(noise_length)

    lr_dis = optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0005, decay_steps=100, decay_rate=0.99, staircase=False)
    lr_gen = optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0025, decay_steps=100, decay_rate=0.99, staircase=False)

    optimizer_dis = optimizers.RMSprop(learning_rate=lr_dis)
    optimizer_gen = optimizers.RMSprop(learning_rate=lr_gen)

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

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 20, 20, 1), dtype=tf.uint8, name='image_real')])
    def train_step(image_real):
        image_real = tf.subtract(tf.divide(tf.cast(image_real, tf.float32), 127.5), 1)
        noises = tf.random.normal([tf.shape(image_real)[0], noise_length])
        with tf.GradientTape() as gt_dis, tf.GradientTape() as gt_gen:
            image_fake = model_gen(noises, training=False)
            output_real = model_dis(image_real, training=True)
            output_fake = model_dis(image_fake, training=True)
            alpha = tf.random.uniform(shape=[tf.shape(image_real)[0], 1, 1, 1], minval=0., maxval=1.)
            interpolates = alpha * image_real + (1. - alpha) * image_fake
            with tf.GradientTape() as gt_penalty:
                gt_penalty.watch(interpolates)
                pred = model_dis(interpolates)
            gradients = gt_penalty.gradient(pred, [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_mean(tf.square(gradients), axis=[1, 2, 3]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            loss_dis = tf.reduce_mean(output_fake) - tf.reduce_mean(output_real) + gradient_penalty_weight * gradient_penalty
            loss_gen = -tf.reduce_mean(output_fake)
        gradients_dis = gt_dis.gradient(loss_dis, model_dis.trainable_variables)
        gradients_gen = gt_gen.gradient(loss_gen, model_gen.trainable_variables)
        optimizer_dis.apply_gradients(zip(gradients_dis, model_dis.trainable_variables))
        optimizer_gen.apply_gradients(zip(gradients_gen, model_gen.trainable_variables))
        return loss_dis, loss_gen

    log_dis = f"log/{current_time}/dis"
    log_gen = f"log/{current_time}/gen"
    log_sample = f"log/{current_time}/sample"
    summary_writer_dis = tf.summary.create_file_writer(log_dis)
    summary_writer_gen = tf.summary.create_file_writer(log_gen)
    summary_writer_sample = tf.summary.create_file_writer(log_sample)

    train_loss_dis = tf.convert_to_tensor(0.0)
    train_loss_gen = tf.convert_to_tensor(0.0)
    for epoch in range(start_step, epochs):
        for image_batch in train_ds:
            train_loss_dis, train_loss_gen = train_step(image_batch)
        with summary_writer_dis.as_default():
            tf.summary.scalar('Discriminator Loss', train_loss_dis.numpy(), step=epoch)
        with summary_writer_gen.as_default():
            tf.summary.scalar('Generator Loss', train_loss_gen.numpy(), step=epoch)
        print(f"Epoch {epoch}, Gen Loss: {train_loss_gen.numpy()}, Dis Loss: {train_loss_dis.numpy()}")
        samples = model_gen(noise_seed, training=False).numpy()
        img = visualize(((samples + 1.) * 127.5).astype("uint8"))
        img.save(f"sample/{epoch:06d}.jpg")
        with summary_writer_sample.as_default():
            tf.summary.image("sample", tf.expand_dims(tf.convert_to_tensor(np.array(img)), 0), step=epoch)
        checkpoint.step.assign_add(1)
        checkpoint_manager.save()
    model_gen.save("model/generator.hdf5")
    model_dis.save("model/discriminator.hdf5")


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train(0, False)
