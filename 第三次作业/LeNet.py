import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # 导入数据集
 
'''LeNet_5, 可以识别图片中的手写数字, 针对灰度图像训练的'''
class LeNet_5:
    def __init__(self):
        self.in_x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1], name="in_x")
        self.in_y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="in_y")
 
        # 卷积层 (batch, 28, 28, 1) -> (batch, 24, 24, 6)
        self.conv1 = tf.layers.Conv2D(filters=6, kernel_size=5, strides=(1, 1),
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(1 / 6)))
        # 池化层 (batch, 24, 24, 6) -> (batch, 12, 12, 6)
        self.pool1 = tf.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))
        # 卷积层 (batch, 12, 12, 6) -> (batch, 8, 8, 16)
        self.conv2 = tf.layers.Conv2D(filters=16, kernel_size=5, strides=(1, 1),
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(1 / 16)))
        # 池化层 (batch, 8, 8, 16) -> (batch, 4, 4, 16)
        self.pool2 = tf.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))
        # 全链接层 (batch, 4*4*16(256)) -> (batch, 120)
        self.fc1 = tf.layers.Dense(120, kernel_initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(1 / 120)))
        # 全链接层 (batch, 120) -> (batch, 10)
        self.fc2 = tf.layers.Dense(10, kernel_initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(1 / 10)))
 
    def forward(self):  # 因为是还原LeNet5 所以使用sigmoid
        self.conv1_out = tf.nn.sigmoid(self.conv1(self.in_x))  # 将图片传入 conv1
        self.pool1_out = self.pool1(self.conv1_out)  # 将 conv1 的输出传入 pool1
        self.conv2_out = tf.nn.sigmoid(self.conv2(self.pool1_out))  # 将 pool1 的输出传入 conv2
        self.pool2_out = self.pool2(self.conv2_out)  # 将 conv2 的输出传入 pool2
        self.flat = tf.reshape(self.pool2_out, shape=[-1, 256])  # 将 pool2 的输出reshape成 (batch, -1(-1指这里的256,具体看计算出的图大小))
        self.fc1_out = tf.nn.sigmoid(self.fc1(self.flat))  # 将 reshape 后的图传入 fc1
        self.fc2_out = tf.nn.softmax(self.fc2(self.fc1_out))  # 将 fc1 的输出传入 fc2
 
    def backward(self):  # 后向计算
        self.loss = tf.reduce_mean((self.fc2_out - self.in_y) ** 2)  # 均方差计算损失
        self.opt = tf.train.AdamOptimizer().minimize(self.loss)  # 使用Adam优化器优化损失
 
    def acc(self):  # 精度计算(可不写, 不影响网络使用)
        self.acc1 = tf.equal(tf.argmax(self.fc2_out, 1), tf.argmax(self.in_y, 1))
        self.accaracy = tf.reduce_mean(tf.cast(self.acc1, dtype=tf.float32))
 
 
if __name__ == '__main__':
    net = LeNet_5()  # 创建LeNet_5的对象
    net.forward()  # 执行前向计算
    net.backward()  # 执行后向计算
    net.acc()  # 执行精度计算
    init = tf.global_variables_initializer()  # 初始化所有tensorflow变量
    with tf.Session() as sess:
        sess.run(init)
        for i in range(10000):
            train_x, train_y = mnist.train.next_batch(100)  # 取出mnist训练集的 100 批数据和标签
            train_x_flat = train_x.reshape([-1, 28, 28, 1])  # 将数据整型
            # 将数据传入网络,并得到计算后的精度和损失
            acc, loss, _ = sess.run(fetches=[net.accaracy, net.loss, net.opt],
                                    feed_dict={net.in_x: train_x_flat, net.in_y: train_y})
            if i % 100 == 0:  # 每训练100次打印一次训练集精度和损失
                print("训练集精度:|", acc)
                print("训练集损失:|", loss)
                test_x, test_y = mnist.test.next_batch(100)  # 取出100批测试集数据进行测试
                test_x_flat = test_x.reshape([-1, 28, 28, 1])  # 同上
                # 同上
                test_acc, test_loss = sess.run(fetches=[net.accaracy, net.loss],
                                               feed_dict={net.in_x: test_x_flat, net.in_y: test_y})
                print('----------')
                print("验证集精度:|", test_acc)  # 打印验证集精度
                print("验证集损失:|", test_loss)  # 打印验证集损失
                print('--------------------')