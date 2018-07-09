import argparse
import tensorflow as tf

CROP_SIZE = 256  # scale_size = CROP_SIZE
ngf = 64
ndf = 64


def preprocess(image):
    with tf.name_scope('preprocess'):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope('deprocess'):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    # if a.separable_conv:
    #     return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    # else:
    return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def lrelu(x, a):
    with tf.name_scope('lrelu'):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))
    # with tf.variable_scope('batchnorm'):
    #     # this block looks like it has 3 inputs on the graph unless we do this
    #     input = tf.identity(input)
    #
    #     channels = input.get_shape()[3]
    #     offset = tf.get_variable('offset', [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
    #     scale = tf.get_variable('scale', [channels], dtype=tf.float32,
    #                             initializer=tf.random_normal_initializer(1.0, 0.02))
    #     mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
    #     variance_epsilon = 1e-5
    #     normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
    #     return normalized


def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    # if a.separable_conv:
    #     _b, h, w, _c = batch_input.shape
    #     resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #     return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    # else:
    return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def process_image(x):
    with tf.name_scope('load_images'):
        raw_input = tf.image.convert_image_dtype(x, dtype=tf.float32)

        raw_input.set_shape([None, None, 3])

        # break apart image pair and move to range [-1, 1]
        width = tf.shape(raw_input)[1]  # [height, width, channels]
        a_images = preprocess(raw_input[:, :width // 2, :])
        b_images = preprocess(raw_input[:, width // 2:, :])

    inputs, targets = [a_images, b_images]

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    def transform(image):
        r = image

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [CROP_SIZE, CROP_SIZE], method=tf.image.ResizeMethod.AREA)

        return r

    with tf.name_scope('input_images'):
        input_images = tf.expand_dims(transform(inputs), 0)

    with tf.name_scope('target_images'):
        target_images = tf.expand_dims(transform(targets), 0)

    return input_images, target_images

    # Tensor('batch:1', shape=(1, 256, 256, 3), dtype=float32) -> 1 batch size


def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope('encoder_1'):
        output = gen_conv(generator_inputs, ngf)
        layers.append(output)

    layer_specs = [
        ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope('encoder_%d' % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (ngf * 8, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (ngf * 8, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (ngf * 8, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (ngf * 8, 0.0),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (ngf * 4, 0.0),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (ngf * 2, 0.0),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (ngf, 0.0),  # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope('decoder_%d' % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope('decoder_1'):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def create_model(inputs, targets):
    with tf.variable_scope('generator'): # as scope
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    return outputs


def convert(image):
    return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True, name='output')  # output tensor


def generate_output(x):
    with tf.name_scope('generate_output'):
        test_inputs, test_targets = process_image(x)

        # inputs and targets are [batch_size, height, width, channels]
        model = create_model(test_inputs, test_targets)

        # deprocess files
        outputs = deprocess(model)

        # reverse any processing on images so they can be written to disk or displayed to user
        converted_outputs = convert(outputs)
    return converted_outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-input', dest='input_folder', type=str, help='Model folder to import.')
    parser.add_argument('--model-output', dest='output_folder', type=str, help='Model (reduced) folder to export.')
    args = parser.parse_args()

    x = tf.placeholder(tf.uint8, shape=(256, 512, 3), name='image_tensor')  # input tensor
    y = generate_output(x)

    with tf.Session() as sess:
        # Restore original model
        saver = tf.train.Saver()
        checkpoint = tf.train.latest_checkpoint(args.input_folder)
        saver.restore(sess, checkpoint)

        # Export reduced model used for prediction
        saver = tf.train.Saver()
        saver.save(sess, '{}/reduced_model'.format(args.output_folder))
        print("Model is exported to {}".format(checkpoint))
