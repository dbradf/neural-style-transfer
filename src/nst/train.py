import time

import click
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg19
from keras import backend as K
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from imageio import imsave


IMG_HEIGHT = 720
IMG_WIDTH = 600


def preprocess_image(image_path):
    img = load_img(image_path)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)

    return img


def deprocess_image(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def content_loss(base, combination):
    return K.sum(K.square(combination - base))


def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination):
    s = gram_matrix(style)
    c = gram_matrix(combination)
    channels = 3
    size = IMG_WIDTH * IMG_HEIGHT

    return K.sum(K.square(s - c)) / (4. * channels ** 2) * (size ** 2)


def total_variation_loss(x):
    a = K.square(x[:, :IMG_HEIGHT - 1, :IMG_WIDTH - 1, :] - x[:, 1:, :IMG_WIDTH - 1, :])
    b = K.square(x[:, :IMG_HEIGHT - 1, :IMG_WIDTH - 1, :] - x[:, :IMG_HEIGHT - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


def extract_layer_features(layer_dict, layer_name, idx0, idx1):
    layer_features = layer_dict[layer_name]
    features0 = layer_features[idx0, :, :, :]
    features1 = layer_features[idx1, :, :, :]

    return features0, features1


@click.command()
@click.option("--target-image", "target_image_path")
@click.option("--style-image")
@click.option("--iterations", default=20)
@click.option("--output")
def main(target_image_path, style_image, iterations, output):
    target_image = K.constant(preprocess_image(target_image_path))
    style_reference_image = K.constant(preprocess_image(style_image))
    combination_image = K.placeholder((1, IMG_HEIGHT, IMG_WIDTH, 3))

    input_tensor = K.concatenate([target_image, style_reference_image, combination_image], axis=0)

    model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)

    outputs_dict = {layer.name: layer.output for layer in model.layers}
    content_layer = 'block5_conv2'
    style_layers = [f'block{i}_conv1' for i in range(1, 6)]
    weights = {
        'total_variation': 1e-4,
        'style': 1.,
        'content': 0.025,
    }
    # total_variation_weight = 1e-4
    # style_weight = 1.
    content_weight = 0.025
    loss = K.variable(0.)
    target_image_features, combination_features = extract_layer_features(outputs_dict, content_layer, 0, 2)
    # layer_features = outputs_dict[content_layer]
    # target_image_features = layer_features[0, :, :, :]
    # combination_features = layer_features[2, :, :, :]
    loss = loss + weights['content'] * content_loss(target_image_features, combination_features)

    for layer_name in style_layers:
        style_reference_features, combination_features = extract_layer_features(outputs_dict,
                                                                                layer_name, 1, 2)
        # layer_features = outputs_dict[layer_name]
        # style_reference_features = layer_features[1, :, :, :]
        # combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss = loss + (weights['style'] / len(style_layers)) * sl

    loss = loss + weights['total_variation'] * total_variation_loss(combination_image)

    grads = K.gradients(loss, combination_image)[0]
    fetch_loss_and_grads = K.function([combination_image], [loss, grads])

    class Evaluator(object):

        def __init__(self):
            self.loss_value = None
            self.grad_values = None

        def loss(self, x):
            assert self.loss_value is None
            x = x.reshape((1, IMG_HEIGHT, IMG_WIDTH, 3))
            outs = fetch_loss_and_grads([x])
            loss_value = outs[0]
            grad_values = outs[1].flatten().astype('float64')
            self.loss_value = loss_value
            self.grad_values = grad_values
            return self.loss_value

        def grads(self, x):
            assert self.loss_value is not None
            grad_values = np.copy(self.grad_values)
            self.loss_value = None
            self.grad_values = None
            return grad_values

    evaluator = Evaluator()

    x = preprocess_image(target_image_path)
    x = x.flatten()
    for i in range(iterations):
        print(f"start of iteration {i}")
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x, fprime=evaluator.grads, maxfun=20)
        print(f'current loss value: {min_val}')

        img = x.copy().reshape((IMG_HEIGHT, IMG_WIDTH, 3))
        img = deprocess_image(img)
        fname = f"{output}_at_iteration_{i}.png"
        imsave(fname, img)
        end_time = time.time()

        print(f'iteration {i} completed in {end_time - start_time}')
