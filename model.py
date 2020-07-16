# coding:utf-8
import numpy as np
import os
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.keras import activations, layers, losses, metrics, models, optimizers
import datetime


def load_data(batch_size):
    data = np.load("data/data.npz")
    train_x = data["train_x"]
    train_y = data["train_y"].astype(np.int32)
    test_x = data["test_x"]
    test_y = data["test_y"].astype(np.int32)
    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(16000).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).shuffle(16000).batch(batch_size)
    return train_ds, test_ds


def make_model():
    images = layers.Input(shape=[20, 20, 1], dtype=tf.uint8)  # 20x20x1
    hidden = tf.image.convert_image_dtype(images, tf.float32)
    hidden = layers.Conv2D(filters=20, kernel_size=5, strides=(1, 1), padding='same', activation=activations.relu, name='conv2d_1')(hidden)  # 20x20x20
    hidden = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool2d_1')(hidden)  # 10x10x20
    hidden = layers.Conv2D(filters=50, kernel_size=5, strides=(1, 1), padding='same', activation=activations.relu, name='conv2d_2')(hidden)  # 10x10x50
    hidden = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool2d_2')(hidden)  # 5x5x50
    hidden = layers.Flatten(name='flatten_1')(hidden)  # 1250
    hidden = layers.Dense(units=500, activation=activations.relu, name='dense_1')(hidden)  # 500
    hidden = layers.Dropout(rate=0.5, name='dropout_1')(hidden)  # 500
    predictions = layers.Dense(units=32, activation=activations.softmax, name='dense_2')(hidden)  # 32
    model = models.Model(inputs=[images], outputs=[predictions])
    return model


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
        img = Image.fromarray(image.reshape(image.shape[:-1])).convert("RGB")
    else:
        img = Image.fromarray(image)
    if labels is not None:
        draw = ImageDraw.Draw(img)
        for idx, label in enumerate(labels):
            i = idx // num_h
            j = idx % num_v
            draw.text((j * (width + pad) + pad, i * (height + pad) + pad), label2text(label), fill=(255, 0, 0))
    return img


def train(start_step=0, restore=False):
    batch_size = 64
    epochs = 10
    train_ds, test_ds = load_data(batch_size)

    sample_images = tf.concat([images for images, _ in test_ds], axis=0).numpy()

    model = make_model()
    model.summary()

    learning_rate = optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0001, decay_steps=100, decay_rate=0.99, staircase=True)
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, 'model/', max_to_keep=3)

    if restore:
        try:
            checkpoint.restore(f'model/ckpt-{start_step}')
            print(f"Restored from model/ckpt-{start_step}")
        except tf.errors.NotFoundError:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            if checkpoint_manager.latest_checkpoint:
                start_step = checkpoint.step.numpy()
                print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
            else:
                start_step = 0
                print("Initializing from scratch.")

    loss_object = losses.SparseCategoricalCrossentropy(from_logits=False)
    train_loss = metrics.Mean(name='train_loss')
    train_accuracy = metrics.SparseCategoricalAccuracy(name='train_accuracy')

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 20, 20, 1), dtype=tf.uint8, name="train_images"), tf.TensorSpec(shape=(None,), dtype=tf.int32, name="train_labels")])
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(labels, predictions)

    test_loss = metrics.Mean(name='test_loss')
    test_accuracy = metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 20, 20, 1), dtype=tf.uint8, name="test_images"), tf.TensorSpec(shape=(None,), dtype=tf.int32, name="test_labels")])
    def test_step(images, labels):
        predictions = model(images)
        test_loss(loss_object(labels, predictions))
        test_accuracy(labels, predictions)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_info = f"log/{current_time}/info"
    log_train = f"log/{current_time}/train"
    log_test = f"log/{current_time}/test"
    log_sample = f"log/{current_time}/sample"
    summary_writer_info = tf.summary.create_file_writer(log_info)
    summary_writer_train = tf.summary.create_file_writer(log_train)
    summary_writer_test = tf.summary.create_file_writer(log_test)
    summary_writer_sample = tf.summary.create_file_writer(log_sample)

    string = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"

    with summary_writer_info.as_default():
        tf.summary.trace_on(graph=True, profiler=True)

        @tf.function(input_signature=[tf.TensorSpec(shape=(None, 20, 20, 1), dtype=tf.uint8, name="sample_images"), tf.TensorSpec(shape=(None,), dtype=tf.int32, name="sample_labels")])
        def create_graph(images, labels):
            predictions = model(images)
            return tf.reduce_mean(loss_object(labels, predictions))

        _ = create_graph(tf.zeros((batch_size, 20, 20, 1), tf.uint8), tf.zeros((batch_size,), tf.int32))
        tf.summary.trace_export(name="graph", step=0, profiler_outdir=log_info)
        tf.summary.trace_off()

    for epoch in range(start_step, epochs):
        for train_images, train_labels in train_ds:
            train_step(train_images, train_labels)
        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)
        print(f"Epoch {epoch + 1}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result() * 100}%, Test Loss: {test_loss.result()}, Test Accuracy: {test_accuracy.result() * 100}%")
        with summary_writer_train.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
        with summary_writer_test.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
        with summary_writer_sample.as_default():
            sample_predictions = tf.argmax(model(sample_images), axis=1).numpy()
            sample = visualize(sample_images, sample_predictions, lambda i: string[i])
            tf.summary.image("sample", tf.expand_dims(tf.convert_to_tensor(np.array(sample)), 0), step=epoch)
        checkpoint.step.assign_add(1)
        checkpoint_manager.save()

        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

    model.save("model/final.hdf5", save_format="hdf5")


def test():
    model = models.load_model("model/final.hdf5")
    data = np.load("data/data.npz")
    test_x = data["test_x"]

    @tf.function
    def predict(images):
        return tf.argmax(model(images), axis=1)

    string = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    predictions = predict(test_x)
    image = visualize(test_x, predictions, lambda i: string[i])
    image.save("sample/final.jpg")


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train(start_step=0, restore=False)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # test()
