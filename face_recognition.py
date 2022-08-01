import cv2
import os
import random
import numpy as np
from kivy.clock import mainthread
from matplotlib import pyplot as plt
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.keras.metrics import Precision, Recall
import tensorflow as tf
import environment as env

import uuid

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

if not os.path.exists(POS_PATH):
    os.makedirs(POS_PATH)

if not os.path.exists(NEG_PATH):
    os.makedirs(NEG_PATH)

if not os.path.exists(ANC_PATH):
    os.makedirs(ANC_PATH)


def data_augmentation(data_augmentation_pos_pb, data_augmentation_anc_pb):
    def data_aug(img, is_pos=True):
        data = []
        for i in range(9):
            img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1, 2))
            img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1, 3))
            img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100), np.random.randint(100)))
            img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100,
                                                         seed=(np.random.randint(100), np.random.randint(100)))
            img = tf.image.stateless_random_saturation(img, lower=0.9, upper=1,
                                                       seed=(np.random.randint(100), np.random.randint(100)))

            data.append(img)

            env.index += 1
            if is_pos:
                update_pb(data_augmentation_pos_pb, env.index)
            else:
                update_pb(data_augmentation_anc_pb, env.index)

        return data

    env.index = 0
    for file_name in os.listdir(os.path.join(POS_PATH)):
        img_path = os.path.join(POS_PATH, file_name)
        img = cv2.imread(img_path)
        augmented_images = data_aug(img)

        for image in augmented_images:
            cv2.imwrite(os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())

    env.index = 0
    for file_name in os.listdir(os.path.join(ANC_PATH)):
        img_path = os.path.join(ANC_PATH, file_name)
        img = cv2.imread(img_path)
        augmented_images = data_aug(img, is_pos=False)

        for image in augmented_images:
            cv2.imwrite(os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())


def get_image_directory(getting_images_pb):
    anchor = tf.data.Dataset.list_files(ANC_PATH + '\*.jpg').take(3000)
    update_pb(getting_images_pb, 1)

    positive = tf.data.Dataset.list_files(POS_PATH + '\*.jpg').take(3000)
    update_pb(getting_images_pb, 2)

    negative = tf.data.Dataset.list_files(NEG_PATH + '\*.jpg').take(3000)
    update_pb(getting_images_pb, 3)

    return anchor, positive, negative


def preprocess(file_path):
    byte_image = tf.io.read_file(file_path)
    image = tf.io.decode_jpeg(byte_image)
    image = tf.image.resize(image, (100, 100))
    image = image / 255.0
    return image


def create_labeled_dataset(anchor, positive, negative, creating_labeled_ds_pb):
    positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    update_pb(creating_labeled_ds_pb, 1)

    negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
    update_pb(creating_labeled_ds_pb, 2)

    data = positives.concatenate(negatives)
    update_pb(creating_labeled_ds_pb, 3)

    return data


def preprocess_twin(input_image, validation_image, label):
    return (preprocess(input_image), preprocess(validation_image), label)


def build_dataset_pipeline(data, build_dataset_pipeline_pb):
    data = data.map(preprocess_twin)
    update_pb(build_dataset_pipeline_pb, 1)

    data = data.cache()
    update_pb(build_dataset_pipeline_pb, 2)

    data = data.shuffle(buffer_size=5000)
    update_pb(build_dataset_pipeline_pb, 3)

    return data


def make_train_data(data, make_train_test_data_pb):
    train_data = data.take(round(len(data) * .7))
    update_pb(make_train_test_data_pb, 1)

    train_data = train_data.batch(16)
    update_pb(make_train_test_data_pb, 2)

    train_data = train_data.prefetch(8)
    update_pb(make_train_test_data_pb, 3)

    return train_data


def make_test_data(data, make_train_test_data_pb):
    test_data = data.skip(round(len(data) * .7))
    update_pb(make_train_test_data_pb, 4)

    test_data = test_data.take(round(len(data) * .3))
    update_pb(make_train_test_data_pb, 5)

    test_data = test_data.batch(16)
    update_pb(make_train_test_data_pb, 6)

    test_data = test_data.prefetch(8)
    update_pb(make_train_test_data_pb, 7)

    return test_data


def make_embedding(make_model_pb):
    input = Input(shape=(100, 100, 3), name='input_image')
    update_pb(make_model_pb, 1)

    conv1 = Conv2D(64, (10, 10), activation='relu')(input)
    mp1 = MaxPooling2D(64, (2, 2), padding='same')(conv1)
    update_pb(make_model_pb, 2)

    conv2 = Conv2D(128, (7, 7), activation='relu')(mp1)
    mp2 = MaxPooling2D(64, (2, 2), padding='same')(conv2)
    update_pb(make_model_pb, 3)

    conv3 = Conv2D(128, (4, 4), activation='relu')(mp2)
    mp3 = MaxPooling2D(64, (2, 2), padding='same')(conv3)
    update_pb(make_model_pb, 4)

    conv4 = Conv2D(256, (4, 4), activation='relu')(mp3)
    flat = Flatten()(conv4)
    dense = Dense(4096, activation='sigmoid')(flat)
    update_pb(make_model_pb, 5)

    return Model(inputs=[input], outputs=[dense], name='embedding')


class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    # similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def make_siamese_model(make_model_pb):
    embedding = make_embedding(make_model_pb)
    input_image = Input(name='input_image', shape=(100, 100, 3))
    validation_image = Input(name='validation_image', shape=(100, 100, 3))
    update_pb(make_model_pb, 6)

    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    update_pb(make_model_pb, 7)

    classifier = Dense(1, activation='sigmoid')(distances)
    update_pb(make_model_pb, 8)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


def train_the_model(siamese_model, train_data, train_pb):
    binary_cross_loss = tf.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(1e-4)
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(opt=optimizer, siamese_model=siamese_model)

    env.index = 0

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            X = batch[:2]
            y = batch[2]

            y_hat = siamese_model(X, training=True)
            loss = binary_cross_loss(y, y_hat)
        print(loss)

        grad = tape.gradient(loss, siamese_model.trainable_variables)
        optimizer.apply_gradients(zip(grad, siamese_model.trainable_variables))

        return loss

    def train(data, EPOCHS):
        flag = False
        loss = None
        for epoch in range(1, EPOCHS + 1):
            print('\nEpoch {}/{}'.format(epoch, EPOCHS))

            r = Recall()
            p = Precision()

            progbar = tf.keras.utils.Progbar(len(data))

            for idx, batch in enumerate(data):
                loss = train_step(batch)
                y_hat = siamese_model.predict(batch[:2], verbose=0)

                r.update_state(batch[2], y_hat)
                p.update_state(batch[2], y_hat)

                env.index += 1
                update_pb(train_pb, env.index)

                progbar.update(idx + 1)

                if epoch >= 5 and loss.numpy() < 0.0001 and r.result().numpy() > 0.9999 and p.result().numpy() > 0.9999:
                    flag = True
                    update_pb(train_pb, 13150)
                    break

            print(epoch, flag)
            print(loss.numpy(), r.result().numpy(), p.result().numpy())

            if flag:
                break

            if epoch % 10 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

    EPOCHS = 50
    train(train_data, EPOCHS)

    return siamese_model


def save_the_model(model, name):
    model.save(name + '.h5')


def load_the_model(name, custom_objects=None):
    if custom_objects is None:
        custom_objects = {'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy}
    # m = tf.keras.models.load_model(name + '.h5', custom_objects=custom_objects)
    # m.comp
    return tf.keras.models.load_model(name + '.h5', custom_objects=custom_objects)


def verify(model, detection_threshold, verification_threshold, cap, label):
    update_label(label, 'Verifying')

    SAVEPATH = os.path.join('application_data', 'input_image.jpg')
    ret, frame = cap.read()
    frame = frame[120:120 + 250, 200:200 + 250, :]
    cv2.imwrite(SAVEPATH, frame)

    results = []
    for image in os.listdir(os.path.join('application_data', 'verification_images')):
        input_img = preprocess(os.path.join('application_data', 'input_image.jpg'))
        validation_img = preprocess(os.path.join('application_data', 'verification_images', image))

        result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    detection = np.sum(np.array(results) > detection_threshold)

    verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
    verified = verification > verification_threshold

    text = 'Verified' if verified else 'Unverified'
    update_label(label, text)

    print(results)
    print(np.sum(np.squeeze(results) < 0.9))
    print(detection)
    print(verification)
    print(model)

    return results, verified


def find_screen(list, name):
    inx_ = 0
    print(list)
    for i in list:
        if i.name == name:
            index = inx_
            break
        inx_ += 1

    return index


def take_images(path, capture, button, pb):
    index = 0
    while index < 0:
        PATH = os.path.join('data', path)
        ret, frame = capture.read()
        frame = frame[120:120 + 250, 200:200 + 250, :]
        cv2.imwrite(os.path.join(PATH, '{}.jpg'.format(uuid.uuid1())), frame)
        index += 1
        update_pb(pb, index)

    enable_button(button)


@mainthread
def enable_button(button):
    button.disabled = False


@mainthread
def update_pb(pb, value):
    pb.value = value


@mainthread
def update_label(label, text):
    label.text = text


def run_train(self_):
    # data_augmentation(self_.data_augmentation_pos_pb, self_.data_augmentation_anc_pb)
    anchor, positive, negative = get_image_directory(self_.getting_images_pb)
    data = create_labeled_dataset(anchor, positive, negative, self_.creating_labeled_ds_pb)
    data = build_dataset_pipeline(data, self_.build_dataset_pipeline_pb)

    train_data = make_train_data(data, self_.make_train_test_data_pb)
    test_data = make_test_data(data, self_.make_train_test_data_pb)

    siamese_model = make_siamese_model(self_.make_model_pb)
    siamese_model = train_the_model(siamese_model, train_data, self_.train_pb)

    self_.save_button.disabled = False
    self_.model = siamese_model

    print('done')
