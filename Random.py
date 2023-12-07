from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add
from tensorflow.keras.layers import Input, MaxPooling2D, GlobalAveragePooling2D, Flatten
from tensorflow.keras.layers import Dense, Dropout, Softmax
from tensorflow.keras.models import Model
import numpy as np

'''
K = np.random.randint(0, 2, (4, 4))
K = np.triu(K)
print('K:\n',K)
随机矩阵K=
    [[1 1 0 1]
    [0 0 1 1]
    [0 0 1 0]
    [0 0 0 1]]
'''





def Conv_BN_Relu(filters, kernel_size, strides, input_layer):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

# Random18网络对应的模块1和模块2
def resiidual_1_or_2(input_x, filters, flag):
    if flag == '1':
        # 主路
        x = Conv_BN_Relu(filters, (3, 3), 1, input_x)
        y1 = Add()([x, input_x])
        x = Conv_BN_Relu(filters, (3, 3), 1, y1)
        y2 = Add()([x, input_x])
        x = Conv_BN_Relu(filters, (3, 3), 1, y2)
        y3 = Add()([x, y1])
        x = Conv_BN_Relu(filters, (3, 3), 1, y3)
        y4 = Add()([x, y3])
        y4 = Add()([y4, y1])

        # 输出

        y = Add()([y4, input_x])

        return y
    elif flag == '2':
        # 主路
        x = Conv_BN_Relu(filters, (3, 3), 2, input_x)
        input_x = Conv_BN_Relu(filters, (1,1), 2, input_x)
        y1 = Add()([x, input_x])
        x = Conv_BN_Relu(filters, (3, 3), 1, y1)
        y2 = Add()([x, input_x])
        x = Conv_BN_Relu(filters, (3, 3), 1, y2)
        y3 = Add()([x, y1])
        x = Conv_BN_Relu(filters, (3, 3), 1, y3)
        y4 = Add()([x, y3])
        y4 = Add()([y4, y1])


        # 输出
        y = Add()([y4, input_x])

        return y






# 第一层
input_layer = Input((300, 300, 3))
conv1 = Conv_BN_Relu(64, (7, 7), 1, input_layer)
conv1_Maxpooling = MaxPooling2D((3, 3), strides=2, padding='same')(conv1)

# conv2_x
x = resiidual_1_or_2(conv1_Maxpooling, 64, '2')
x = resiidual_1_or_2(x, 64, '1')

# conv3_x
x = resiidual_1_or_2(x, 128, '2')
x = resiidual_1_or_2(x, 128, '1')

'''# conv4_x
x = resiidual_1_or_2(x, 256, '2')
x = resiidual_1_or_2(x, 256, '1')

# conv5_x
x = resiidual_1_or_2(x, 512, '2')
x = resiidual_1_or_2(x, 512, '1')'''

# 最后一层
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dense(1000)(x)
x = Dropout(0.5)(x)
y = Softmax(axis=-1)(x)

model = Model([input_layer], [y])

model.summary()

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
plot_model(model, to_file='Random_Resnet_18.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))

import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import random
import pathlib

# 1.读取图片及标签，将标签编码
# 2.运用tensorflow框架进行训练


data_dir = 'D:/ybybyb/研/elpv-dataset-master/train image-2'  # 数据集路径
data_root = pathlib.Path(data_dir)
for item in data_root.iterdir():  # 在目录中进行迭代
    print(item)

all_image_path = list(data_root.glob('*/*')) # 提取所有目录中的所有文件
print(all_image_path[0:2])

all_image_path = [str(path) for path in all_image_path] # 列表推导式
print(all_image_path[10:12])

random.shuffle(all_image_path) # 乱序
print(all_image_path[10:12])

image_count = len(all_image_path)
print(image_count) # 432

label_names = sorted(item.name for item in data_root.glob('*/'))
print(label_names)

label_to_index = dict((name, index) for index, name in enumerate(label_names))  # enumerate 既遍历索引又遍历值
print(label_to_index)


all_image_label = [label_to_index[pathlib.Path(p).parent.name] for p in all_image_path]
print(all_image_label[:5])

index_to_label = dict((value, key) for (key, value) in label_to_index.items())
print(index_to_label)


def load_prepross_image(image_path):  # 图片预处理函数(读取图片，解码图片，转换图片大小格式)
    image_raw = tf.io.read_file(image_path)
    image_tensor = tf.image.decode_jpeg(image_raw, channels=3)
    image_tensor = tf.image.resize(image_tensor, [300, 300])
    image_tensor = tf.cast(image_tensor, tf.float32)
    img = image_tensor/255
    return img

path_dataset = tf.data.Dataset.from_tensor_slices(all_image_path) # 切片
image_dataset = path_dataset.map(load_prepross_image) # 预处理图片
label_dataset = tf.data.Dataset.from_tensor_slices(all_image_label) # 切片
dataset = tf.data.Dataset.zip((image_dataset, label_dataset)) # 数据集
test_count = int(image_count*0.3) # 测试集个数
train_count = image_count - test_count # 训练集个数
train_dataset = dataset.skip(test_count) # 训练集
test_dataset = dataset.take(test_count) # 测试集
BATCH_SIZE = 32
train_dataset = train_dataset.repeat().shuffle(buffer_size=train_count).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model.compile(tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['acc'])
step_per_epoch = train_count//BATCH_SIZE
validation_steps = test_count//BATCH_SIZE
history = model.fit(train_dataset, epochs=10, steps_per_epoch=step_per_epoch, validation_data=test_dataset, validation_steps=validation_steps)

model.summary()
print('keys:',history.history.keys())
plt.plot(history.epoch, history.history.get('acc'), label = 'acc')
plt.plot(history.epoch, history.history.get('val_acc'), label = 'val_acc')
plt.legend()
plt.show()

