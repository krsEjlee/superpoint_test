# # import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# # from tensorflow import layers as tfl
# from tensorflow.compat.v1 import layers as tfl



# def vgg_block(inputs, filters, kernel_size, name, data_format, training=False,
#               batch_normalization=True, kernel_reg=0., **params):
#     with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
#         x = tfl.conv2d(inputs, filters, kernel_size, name='conv',
#                        kernel_regularizer=tf.contrib.layers.l2_regularizer(kernel_reg),
#                        data_format=data_format, **params)
#         if batch_normalization:
#             x = tfl.batch_normalization(
#                     x, training=training, name='bn', fused=True,
#                     axis=1 if data_format == 'channels_first' else -1)
#     return x


# def vgg_backbone(inputs, **config):
#     params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
#                    'activation': tf.nn.relu, 'batch_normalization': True,
#                    'training': config['training'],
#                    'kernel_reg': config.get('kernel_reg', 0.)}
#     params_pool = {'padding': 'SAME', 'data_format': config['data_format']}

#     with tf.variable_scope('vgg', reuse=tf.AUTO_REUSE):
#         x = vgg_block(inputs, 64, 3, 'conv1_1', **params_conv)
#         x = vgg_block(x, 64, 3, 'conv1_2', **params_conv)
#         x = tfl.max_pooling2d(x, 2, 2, name='pool1', **params_pool)

#         x = vgg_block(x, 64, 3, 'conv2_1', **params_conv)
#         x = vgg_block(x, 64, 3, 'conv2_2', **params_conv)
#         x = tfl.max_pooling2d(x, 2, 2, name='pool2', **params_pool)

#         x = vgg_block(x, 128, 3, 'conv3_1', **params_conv)
#         x = vgg_block(x, 128, 3, 'conv3_2', **params_conv)
#         x = tfl.max_pooling2d(x, 2, 2, name='pool3', **params_pool)

#         x = vgg_block(x, 128, 3, 'conv4_1', **params_conv)
#         x = vgg_block(x, 128, 3, 'conv4_2', **params_conv)

#     return x
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.compat.v1 import layers as tfl

def vgg_block(inputs, filters, kernel_size, name, data_format, training=False,
              batch_normalization=True, kernel_reg=0., **params):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        x = tfl.conv2d(inputs, filters, kernel_size, name='conv',
                       # 변경된 부분: tf.keras.regularizers.l2 사용
                       kernel_regularizer=tf.keras.regularizers.l2(kernel_reg),
                       data_format=data_format, **params)
        if batch_normalization:
            x = tfl.batch_normalization(
                    x, training=training, name='bn', fused=True,
                    axis=1 if data_format == 'channels_first' else -1)
    return x

def vgg_backbone(inputs, **config):
    # 모든 블록에 공통 파라미터 설정
    params_conv = {
        'padding': 'SAME', 
        'data_format': config['data_format'],
        'activation': tf.nn.relu, 
        'batch_normalization': True,
        'training': config['training'],
        'kernel_reg': config.get('kernel_reg', 0.)
    }
    params_pool = {'padding': 'SAME', 'data_format': config['data_format']}
    
    # 네트워크 구조 표준화 및 최적화
    with tf.variable_scope('vgg', reuse=tf.AUTO_REUSE):
        # 첫 번째 블록: 입력 -> 64 채널
        x = vgg_block(inputs, 64, 3, 'conv1_1', **params_conv)
        x = vgg_block(x, 64, 3, 'conv1_2', **params_conv)
        x = tfl.max_pooling2d(x, 2, 2, name='pool1', **params_pool)
        
        # 두 번째 블록: 64 -> 64 채널 (더 효율적인 채널 수로 조정)
        x = vgg_block(x, 64, 3, 'conv2_1', **params_conv)
        x = vgg_block(x, 64, 3, 'conv2_2', **params_conv)
        x = tfl.max_pooling2d(x, 2, 2, name='pool2', **params_pool)
        
        # 세 번째 블록: 64 -> 128 채널
        x = vgg_block(x, 128, 3, 'conv3_1', **params_conv)
        x = vgg_block(x, 128, 3, 'conv3_2', **params_conv)
        x = tfl.max_pooling2d(x, 2, 2, name='pool3', **params_pool)
        
        # 네 번째 블록: 128 -> 128 채널
        x = vgg_block(x, 128, 3, 'conv4_1', **params_conv)
        x = vgg_block(x, 128, 3, 'conv4_2', **params_conv)
    
    return x