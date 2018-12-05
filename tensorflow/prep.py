import tensorflow as tf

def read_lines(fname):
    data = open(fname).readlines()
    fnames = []
    labels = []
    for line in data:
        fnames.append(line.split()[0])
        labels.append(int(line.split()[1]))
    return fnames, labels

def train_image_process(fname):
    image_string = tf.read_file(fname)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_images(image, [256, 256])
    image = tf.random_crop(image, [224, 224, 3])
    image = tf.image.random_flip_left_right(image)
    return image

def train_prep(fname, label):
    feature_dict = {"source":train_image_process(fname["source"]), \
                    "target":train_image_process(fname["target"]), \
                    "ad_s_label":fname["ad_s_label"], 
                    "ad_t_label":fname["ad_t_label"]}
    return feature_dict, label
    
def test_prep(fname, label):
    image_string = tf.read_file(fname)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_images(image, [256, 256])
    image = tf.image.central_crop(image, 0.875)
    return image, label   
