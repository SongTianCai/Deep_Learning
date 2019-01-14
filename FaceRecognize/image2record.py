import tensorflow as tf
import os

my_faces_path='./my_faces'
other_faces_path='./other_face'
my_face_test = './my_test'
other_test = './other_test'
train_my_save_path ='./save_tfrecords/my_image_train.tfrecords'
train_other_save_path ='./save_tfrecords/other_image_train.tfrecords'
test_my_save_path ='./save_tfrecords/my_image_test.tfrecords'
test_other_save_path ='./save_tfrecords/other_image_test.tfrecords'


file_name = os.listdir(other_faces_path)

file_list = [os.path.join(other_faces_path,file) for file in file_name]

#1.构造文件队列
file_queue = tf.train.string_input_producer(file_list)

#2.文件阅读器
reader = tf.WholeFileReader()
_,value = reader.read(file_queue)

#3.解码
image = tf.image.decode_jpeg(value)

#4图片处理
image_reshape = tf.reshape(image,[64,64,3])

#5批处理
image_batch = tf.train.batch([image_reshape],batch_size=len(file_list),num_threads=4,capacity=len(file_list))



def make_records(image_batches,save_path,label=1):
    '''保存为TFrecords'''
    #1.建立TFrecords存储器

    writer = tf.python_io.TFRecordWriter(save_path)
    for i in range(len(file_list)):
        image = image_batches[i].tostring()

        print(i)

        example = tf.train.Example(features=tf.train.Features(feature={
            'image':tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])),
            'label':tf.train.Feature(int64_list = tf.train.Int64List(value = [label]))
        }))


        writer.write(example.SerializeToString())
    writer.close()


with tf.Session() as sess:

    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess,coord=coord)
    my_face = sess.run(image_batch)
    print('开始存入')
    make_records(image_batches=my_face,save_path=train_other_save_path,label=0)

    print('结束')
    coord.request_stop()
    coord.join(thread)


