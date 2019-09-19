import tensorflow as tf

def cut(num_classes, size=(256, 256), num_cuts=12):

    def cut_fn(image, mask):
        image_shape = tf.shape(image)

        limit_height = tf.math.maximum(0, image_shape[0] - size[0])
        limit_width = tf.math.maximum(0, image_shape[1] - size[1])

        def big_fn(x):
            x = tf.cond(limit_height > 0,
                    lambda: tf.random.uniform((1,), minval=0, maxval=limit_height, dtype=tf.dtypes.int32)[0],
                    lambda: 0)

            y = tf.cond(limit_width > 0,
                    lambda: tf.random.uniform((1,), minval=0, maxval=limit_width, dtype=tf.dtypes.int32)[0],
                    lambda: 0)

            h = tf.math.minimum(image_shape[0], size[0] + x)
            w = tf.math.minimum(image_shape[1], size[1] + y)

            new = tf.pad(image[x:h, y:w],
                     [[0, size[0] - (h - x)], [0, size[1] - (w - y)], [0, 0]])
            new_mask = tf.pad(mask[x:h, y:w],
                          [[0, size[0] - (h - x)], [0, size[1] - (w - y)], [0, 0]])
            new_mask = tf.math.reduce_max(new_mask, axis=-1)

            return new, tf.one_hot(new_mask, num_classes, axis=-1)

        result = tf.map_fn(big_fn, tf.range(num_cuts), dtype=(tf.float32, tf.float32))

        return result

    return cut_fn
