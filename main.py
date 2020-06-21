import tensorflow.compat.v1 as tf
import random
import numpy as np
from train_images import train_images
from train_labels import train_labels

from inference_images import inference_images
from inference_labels import inference_labels

tf.disable_v2_behavior()
tf.disable_eager_execution()

def model_fn_build():
    def model_fn(features, labels, mode, params):
        tf.logging.info("***-----------------Build the training mode----------------------***")
        #tf.logging.info("*** Build the training mode ***")
        logits = tf.layers.dense(inputs=features, name='layer_fc1', units=10, activation=tf.nn.softmax, use_bias=True)

        if mode == tf.estimator.ModeKeys.PREDICT:
            spec = tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=logits)
        else:
            cross_entropy = -tf.reduce_sum(labels * tf.log(logits))

            loss = tf.reduce_mean(cross_entropy)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
            train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step())

            metrics = \
            {
                "accuracy": tf.metrics.accuracy(tf.argmax(labels, axis=1), tf.argmax(logits, axis=1))
            }

            # Wrap all of this in an EstimatorSpec.
            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=metrics)
        return spec

    return model_fn


def generate_dummydata(size):
    images = []
    labels = []
    for i in range(size):
        images.append([random.uniform(0, 1) for __ in range(784)])
        labels.append([random.uniform(0, 1) for __ in range(10)])

    return (images, labels)

def input_fn_dummy():
    return tf.data.Dataset.from_tensor_slices(generate_dummydata(2)).batch(2)

def input_fn():
    train_images_data = train_images('./train_data/train-images-idx3-ubyte')
    images_number = train_images_data.get_images_number()
    images = train_images_data.read_images(images_number)
    train_labels_data = train_labels('./train_data/train-labels-idx1-ubyte')
    labels = train_labels_data.read_labels(images_number)
    return tf.data.Dataset.from_tensor_slices((images,labels)).batch(128)

def input_fn_inference():
    inference_images_data = inference_images('test_data/t10k-images-idx3-ubyte')
    images_number = inference_images_data.get_images_number()
    images = inference_images_data.read_images(images_number)
    return tf.data.Dataset.from_tensor_slices(images).batch(128)

def get_prediction_labels():
    inference_labels_data = inference_labels('test_data/t10k-labels-idx1-ubyte')
    inference_images_data = inference_images('test_data/t10k-images-idx3-ubyte')
    images_number = inference_images_data.get_images_number()
    labels = inference_labels_data.read_labels(images_number)
    return labels

if __name__ == "__main__":
    model_fn = model_fn_build()
    params = {}
    model = tf.estimator.Estimator(model_fn=model_fn,
                                   params=params,
                                   model_dir="./output/")

    res = model.predict(input_fn_inference)
    label = get_prediction_labels()
    #print(list(res))
    index = 0
    right_count = 0
    for item in list(res):
        print("----------------")
        print(np.argmax(item))
        print(label[index])
        if np.argmax(item) == np.argmax(label[index]):
            right_count += 1
        index = index + 1
    print("accuracy is: {}".format(float(right_count)/index))
