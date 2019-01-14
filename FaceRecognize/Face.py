import tensorflow as tf
import cv2
import numpy as np
train_file = ['./save_tfrecords/my_image_train.tfrecords','./save_tfrecords/other_image_train.tfrecords']
test_file = ['./save_tfrecords/my_image_test.tfrecords','./save_tfrecords/other_image_test.tfrecords']
mode_dir = './mode/'

tf.enable_eager_execution()

tf.logging.set_verbosity(tf.logging.INFO)

def parse(records):
    features = tf.parse_single_example(
        records,
        features={
            'image':tf.FixedLenFeature([],tf.string),
            'label':tf.FixedLenFeature([],tf.int64)
        }
    )
    decode_image = tf.decode_raw(features['image'],tf.uint8)
    decode_image=tf.cast(decode_image,tf.float32)
    image_reshape = tf.reshape(decode_image,[64,64,3])
    return {'image':image_reshape},features['label']

def Net_work(x,is_training):
    '''定义网络结构'''
    # x = tf.reshape(x,[-1,64,64,3])

    net =tf.layers.conv2d(x,32,5,padding='SAME',activation=tf.nn.relu)
    net =tf.layers.max_pooling2d(net,2,2,padding='SAME')
    net =tf.layers.conv2d(net,64,3,padding='SAME',activation=tf.nn.relu)
    net =tf.layers.max_pooling2d(net,2,2,padding='SAME')
    net =tf.layers.flatten(net)
    net =tf.layers.dense(net,1024)
    net =tf.layers.dropout(net,rate=0.4,training=is_training)
    return tf.layers.dense(net,2)


def my_input_fn(file_path,perfrom_shuffle=False,repeat_count=1):
    data = tf.data.TFRecordDataset(file_path)
    data = data.map(parse)
    if perfrom_shuffle:
        data = data.shuffle(buffer_size=20000)
    data = data.repeat(repeat_count)
    data = data.batch(128)
    iterator = data.make_one_shot_iterator()
    image_batches,label_batches = iterator.get_next()
    print(image_batches,label_batches)
    return image_batches,label_batches


def model_fn(features,labels,mode,params):
    #定义前向传播
    predict = Net_work(features['image'],mode == tf.estimator.ModeKeys.TRAIN)

    #如果在预测模式，那么只需要将结果返回
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions = {'result':tf.argmax(predict,1)}
        )

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=predict,labels=labels
        )
    )
    optimizer  = tf.train.AdadeltaOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())

    #定义评测标准：
    eval_metric_ops = {
        'my_metric':tf.metrics.accuracy(tf.argmax(predict,1),labels)
    }
    return  tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )

model_params = {'learning_rate':0.01}
estimator = tf.estimator.Estimator(model_fn=model_fn,params=model_params,model_dir=mode_dir)

#estimator.train(input_fn=lambda :my_input_fn(train_file,True,10))
# test_result = estimator.evaluate(input_fn=lambda :my_input_fn(test_file,False,1))
#
# accuracy_score = test_result['my_metric']
# print('\n Test accuracy: %g %%' % (accuracy_score*100))

#使用OpenCV 检测人脸
haar=cv2.CascadeClassifier('D:/opencv-3.4/data/haarcascades/haarcascade_frontalface_default.xml')

camera = cv2.VideoCapture(0)

n = 1
while 1:
    if (n <= 20000):
        print('It`s processing %s image.' % n)
        # 读帧
        success, img = camera.read()

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray_img, 1.3, 5)
        for f_x, f_y, f_w, f_h in faces:
            face = img[f_y:f_y + f_h, f_x:f_x + f_w]
            face = cv2.resize(face, (64, 64))

            test_x = np.array([face])
            test_x = test_x.astype(np.float32)

            predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={'image': test_x}, shuffle=False)
            prediction = estimator.predict(input_fn=predict_input_fn)
            for i, p in enumerate(prediction):
                print('Prediction %s: %s' % (i + 1, p['result']))

                if p['result'] == 1:
                    cv2.putText(img, 'Letao Song', (f_x, f_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)  # 显示名字
                    img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (255, 0, 0), 2)
                else:
                    img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (0, 255, 0), 2)
                    cv2.putText(img, 'UnRecognize', (f_x, f_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            n += 1
        cv2.imshow('img', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
camera.release()
cv2.destroyAllWindows()


# predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={'image':image},
#                                                       num_epochs=1,
#                                                       shuffle=True
#                                                       )
#
#
# predictions = estimator.predict(input_fn=predict_input_fn)





