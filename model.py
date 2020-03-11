# coding:utf-8
import numpy as np
import os
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.keras import activations, layers, losses, metrics, models, optimizers, callbacks
import datetime


def load_data(batch_size):
    data = np.load("data/data.npz")
    train_x = data["train_x"]
    train_y = data["train_y"]
    test_x = data["test_x"]
    test_y = data["test_y"]
    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(16000).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).shuffle(16000).batch(batch_size)
    return train_ds, test_ds


class MyModel(models.Model):
    def __init__(self):
        super(MyModel, self).__init__(name='MyModel')
        self.conv2d_01 = layers.Conv2D(filters=20, kernel_size=5, strides=(1, 1), padding='same', activation=activations.relu, name='conv2d_01')
        self.pool2d_01 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool2d_01')
        self.conv2d_02 = layers.Conv2D(filters=50, kernel_size=5, strides=(1, 1), padding='same', activation=activations.relu, name='conv2d_02')
        self.pool2d_02 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool2d_02')
        self.flatten = layers.Flatten(name='flatten_01')
        self.dense_01 = layers.Dense(units=500, activation=activations.relu, name='dense_01')
        self.dropout = layers.Dropout(rate=0.5, name='dropout')
        self.dense_02 = layers.Dense(units=32, activation=activations.softmax, name='dense_02')

    def call(self, input_tensor, training=True):
        x = tf.divide(tf.cast(input_tensor, tf.float32), 256.)
        x = self.conv2d_01(x)
        x = self.pool2d_01(x)
        x = self.conv2d_02(x)
        x = self.pool2d_02(x)
        x = self.flatten(x)
        x = self.dense_01(x)
        x = self.dropout(x)
        x = self.dense_02(x)
        return x


def make_model():
    images = layers.Input(shape=[20, 20, 1], dtype=tf.uint8)  # 20x20x1
    hidden00 = tf.divide(tf.cast(images, tf.float32), 256.)
    hidden01 = layers.Conv2D(filters=20, kernel_size=5, strides=(1, 1), padding='same', activation=activations.relu, name='conv2d_1')(hidden00)  # 20x20x20
    hidden02 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool2d_1')(hidden01)  # 10x10x20
    hidden03 = layers.Conv2D(filters=50, kernel_size=5, strides=(1, 1), padding='same', activation=activations.relu, name='conv2d_2')(hidden02)  # 10x10x50
    hidden04 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pool2d_2')(hidden03)  # 5x5x50
    hidden05 = layers.Flatten(name='flatten_1')(hidden04)  # 1250
    hidden06 = layers.Dense(units=500, activation=activations.relu, name='dense_1')(hidden05)  # 500
    hidden07 = layers.Dropout(rate=0.5, name='dropout_1')(hidden06)  # 500
    predictions = layers.Dense(units=32, activation=activations.softmax, name='dense_2')(hidden07)  # 32
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


def train(start_step=0, restore=False, oop=False, auto=False):
    batch_size = 64
    train_ds, test_ds = load_data(batch_size)

    sample_images = tf.concat([images for images, _ in test_ds], axis=0).numpy()

    if oop:
        model = MyModel()
        model.build(input_shape=(batch_size, 20, 20, 1))
    else:
        model = make_model()
    model.summary()

    if auto:
        data = np.load("data/data.npz")
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = callbacks.TensorBoard(log_dir=f"log/{current_time}", write_images=True)
        checkpoint_callback = callbacks.ModelCheckpoint(filepath="model/ckpt-{epoch}", verbose=1)
        model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss=losses.SparseCategoricalCrossentropy(from_logits=False), metrics=[metrics.sparse_categorical_accuracy])
        training_history = model.fit(x=data["train_x"], y=data["train_y"], batch_size=64, epochs=10, validation_data=(data["test_x"], data["test_y"]), callbacks=[tensorboard_callback, checkpoint_callback])
        print("Average test loss: ", np.average(training_history.history['loss']))
    else:
        loss_object = losses.SparseCategoricalCrossentropy(from_logits=False)
        learning_rate = optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0001, decay_steps=100, decay_rate=0.99, staircase=True)
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=model)
        manager = tf.train.CheckpointManager(ckpt, 'model/', max_to_keep=3)

        if restore:
            try:
                ckpt.restore(f'model/ckpt-{start_step}')
                print(f"Restored from model/ckpt-{start_step}")
            except tf.errors.NotFoundError:
                ckpt.restore(manager.latest_checkpoint)
                if manager.latest_checkpoint:
                    start_step = ckpt.step.numpy()
                    print("Restored from {}".format(manager.latest_checkpoint))
                else:
                    start_step = 0
                    print("Initializing from scratch.")

        train_loss = metrics.Mean(name='train_loss')
        train_accuracy = metrics.SparseCategoricalAccuracy(name='train_accuracy')

        @tf.function
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

        @tf.function
        def test_step(images, labels):
            predictions = model(images)
            test_loss(loss_object(labels, predictions))
            test_accuracy(labels, predictions)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        info_log_dir = f"log/{current_time}/info"
        train_log_dir = f"log/{current_time}/train"
        test_log_dir = f"log/{current_time}/test"
        sample_log_dir = f"log/{current_time}/sample"
        info_summary_writer = tf.summary.create_file_writer(info_log_dir)
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        sample_summary_writer = tf.summary.create_file_writer(sample_log_dir)

        string = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"

        with info_summary_writer.as_default():
            tf.summary.trace_on(graph=True, profiler=True)

            @tf.function
            def predict(images):
                return model(images)

            predict(tf.zeros((batch_size, 20, 20, 1), tf.uint8))
            tf.summary.trace_export(name="graph", step=0, profiler_outdir=info_log_dir)
            tf.summary.trace_off()
        EPOCHS = 10
        for epoch in range(start_step, EPOCHS):
            for train_images, train_labels in train_ds:
                train_step(train_images, train_labels)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
            for test_images, test_labels in test_ds:
                test_step(test_images, test_labels)
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
            print(f"Epoch {epoch + 1}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result() * 100}%, Test Loss: {test_loss.result()}, Test Accuracy: {test_accuracy.result() * 100}%")
            with sample_summary_writer.as_default():
                sample_predictions = tf.argmax(model(sample_images), axis=1).numpy()
                sample = visualize(sample_images, sample_predictions, lambda i: string[i])
                tf.summary.image("sample", tf.expand_dims(tf.convert_to_tensor(np.array(sample)), 0), step=epoch)
            ckpt.step.assign_add(1)
            manager.save()

            train_loss.reset_states()
            train_accuracy.reset_states()
            test_loss.reset_states()
            test_accuracy.reset_states()

    if oop:
        model.save("model/", save_format="tf")
    else:
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
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # train(start_step=0, restore=False, oop=False, auto=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    test()
