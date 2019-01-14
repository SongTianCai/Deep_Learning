import tensorflow as tf



tf_path = ['./save_tfrecords/my_image_train.tfrecords','./save_tfrecords/other_image_train.tfrecords']

tf.logging.set_verbosity(tf.logging.INFO)

def parse(record):
    '''TFrecord解析'''
    features = tf.parse_single_example(
        record,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    # 对图片进行解码
    decode_image = tf.decode_raw(features['image'], tf.uint8)
    image_reshape = tf.reshape(decode_image, [64, 64, 3])
    tf.cast(image_reshape,tf.float32)
    return {'image':image_reshape}, features['label']


# data_set = tf.data.TFRecordDataset(tf_path)
# data_set = data_set.map(parse)
# data_set = data_set.shuffle(buffer_size=256)
# data_set = data_set.batch(128)
# #定义遍历数据的迭代器
# iterator = data_set.make_one_shot_iterator()
# image,label = iterator.get_next()
data = tf.data.TFRecordDataset(tf_path)
data = data.map(parse)
data = data.shuffle(buffer_size=20000)
data = data.batch(128)
iterator = data.make_one_shot_iterator()
image_batches,label_batches = iterator.get_next()


with tf.Session() as sess:
    for i in range(4):
        print(sess.run([image_batches,label_batches]))



