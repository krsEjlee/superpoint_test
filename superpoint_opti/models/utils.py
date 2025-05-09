# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .homographies import warp_points
from .backbones.vgg import vgg_block


def detector_head(inputs, **config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'batch_normalization': True,
                   'training': config['training'],
                   'kernel_reg': config.get('kernel_reg', 0.)}
    cfirst = config['data_format'] == 'channels_first'
    cindex = 1 if cfirst else -1  # index of the channel

    with tf.variable_scope('detector', reuse=tf.AUTO_REUSE):
        x = vgg_block(inputs, 256, 3, 'conv1',
                      activation=tf.nn.relu, **params_conv)
        x = vgg_block(x, 1+pow(config['grid_size'], 2), 1, 'conv2',
                      activation=None, **params_conv)

        prob = tf.nn.softmax(x, axis=cindex)
        # Strip the extra “no interest point” dustbin
        prob = prob[:, :-1, :, :] if cfirst else prob[:, :, :, :-1]
        prob = tf.depth_to_space(
                prob, config['grid_size'], data_format='NCHW' if cfirst else 'NHWC')
        prob = tf.squeeze(prob, axis=cindex)

    return {'logits': x, 'prob': prob}


# def descriptor_head(inputs, **config):
#     params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
#                    'batch_normalization': True,
#                    'training': config['training'],
#                    'kernel_reg': config.get('kernel_reg', 0.)}
#     cfirst = config['data_format'] == 'channels_first'
#     cindex = 1 if cfirst else -1  # index of the channel

#     with tf.variable_scope('descriptor', reuse=tf.AUTO_REUSE):
#         x = vgg_block(inputs, 256, 3, 'conv1',
#                       activation=tf.nn.relu, **params_conv)
#         x = vgg_block(x, config['descriptor_size'], 1, 'conv2',
#                       activation=None, **params_conv)

#         desc = tf.transpose(x, [0, 2, 3, 1]) if cfirst else x
#         desc = tf.image.resize_bilinear(
#             desc, config['grid_size'] * tf.shape(desc)[1:3])
#         desc = tf.transpose(desc, [0, 3, 1, 2]) if cfirst else desc
#         desc = tf.nn.l2_normalize(desc, cindex)

#     return {'descriptors_raw': x, 'descriptors': desc}

def descriptor_head(inputs, **config):
    """최적화된 디스크립터 헤드"""
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'batch_normalization': True,
                   'training': config['training'],
                   'kernel_reg': config.get('kernel_reg', 0.)}
    cfirst = config['data_format'] == 'channels_first'
    cindex = 1 if cfirst else -1  # 채널 인덱스

    with tf.variable_scope('descriptor', reuse=tf.AUTO_REUSE):
        # 계산 최적화를 위한 채널 수 감소
        mid_channels = min(256, config['descriptor_size'])
        
        # 첫 번째 합성곱 레이어
        x = vgg_block(inputs, mid_channels, 3, 'conv1',
                      activation=tf.nn.relu, **params_conv)
                      
        # 최종 디스크립터로의 합성곱
        x = vgg_block(x, config['descriptor_size'], 1, 'conv2',
                      activation=None, **params_conv)

        # 채널 순서 조정 및 크기 조정
        if cfirst:
            desc = tf.transpose(x, [0, 2, 3, 1])
        else:
            desc = x
            
        # 효율적인 리사이징 - 계산 비용이 큰 작업
        target_shape = tf.multiply(
            tf.shape(desc)[1:3], 
            config['grid_size']
        )
        desc = tf.image.resize_bilinear(desc, target_shape)
        
        # 원래 채널 순서로 복원
        if cfirst:
            desc = tf.transpose(desc, [0, 3, 1, 2])
            
        # L2 정규화 - 안정적인 설명자 생성
        desc = tf.nn.l2_normalize(desc, cindex)

    return {'descriptors_raw': x, 'descriptors': desc}

def detector_loss(keypoint_map, logits, valid_mask=None, **config):
    # Convert the boolean labels to indices including the "no interest point" dustbin
    labels = tf.to_float(keypoint_map[..., tf.newaxis])  # for GPU
    labels = tf.space_to_depth(labels, config['grid_size'])
    shape = tf.concat([tf.shape(labels)[:3], [1]], axis=0)
    labels = tf.concat([2*labels, tf.ones(shape)], 3)
    # Add a small random matrix to randomly break ties in argmax
    labels = tf.argmax(labels + tf.random_uniform(tf.shape(labels), 0, 0.1),
                       axis=3)

    # Mask the pixels if bordering artifacts appear
    valid_mask = tf.ones_like(keypoint_map) if valid_mask is None else valid_mask
    valid_mask = tf.to_float(valid_mask[..., tf.newaxis])  # for GPU
    valid_mask = tf.space_to_depth(valid_mask, config['grid_size'])
    valid_mask = tf.reduce_prod(valid_mask, axis=3)  # AND along the channel dim

    loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits, weights=valid_mask)
    return loss


def descriptor_loss(descriptors, warped_descriptors, homographies,
                    valid_mask=None, **config):
    # Compute the position of the center pixel of every cell in the image
    (batch_size, Hc, Wc) = tf.unstack(tf.to_int32(tf.shape(descriptors)[:3]))
    coord_cells = tf.stack(tf.meshgrid(
        tf.range(Hc), tf.range(Wc), indexing='ij'), axis=-1)
    coord_cells = coord_cells * config['grid_size'] + config['grid_size'] // 2  # (Hc, Wc, 2)
    # coord_cells is now a grid containing the coordinates of the Hc x Wc
    # center pixels of the 8x8 cells of the image

    # Compute the position of the warped center pixels
    warped_coord_cells = warp_points(tf.reshape(coord_cells, [-1, 2]), homographies)
    # warped_coord_cells is now a list of the warped coordinates of all the center
    # pixels of the 8x8 cells of the image, shape (N, Hc x Wc, 2)

    # Compute the pairwise distances and filter the ones less than a threshold
    # The distance is just the pairwise norm of the difference of the two grids
    # Using shape broadcasting, cell_distances has shape (N, Hc, Wc, Hc, Wc)
    coord_cells = tf.to_float(tf.reshape(coord_cells, [1, 1, 1, Hc, Wc, 2]))
    warped_coord_cells = tf.reshape(warped_coord_cells,
                                    [batch_size, Hc, Wc, 1, 1, 2])
    cell_distances = tf.norm(coord_cells - warped_coord_cells, axis=-1)
    s = tf.to_float(tf.less_equal(cell_distances, config['grid_size'] - 0.5))
    # s[id_batch, h, w, h', w'] == 1 if the point of coordinates (h, w) warped by the
    # homography is at a distance from (h', w') less than config['grid_size']
    # and 0 otherwise

    # Normalize the descriptors and
    # compute the pairwise dot product between descriptors: d^t * d'
    descriptors = tf.reshape(descriptors, [batch_size, Hc, Wc, 1, 1, -1])
    descriptors = tf.nn.l2_normalize(descriptors, -1)
    warped_descriptors = tf.reshape(warped_descriptors,
                                    [batch_size, 1, 1, Hc, Wc, -1])
    warped_descriptors = tf.nn.l2_normalize(warped_descriptors, -1)
    dot_product_desc = tf.reduce_sum(descriptors * warped_descriptors, -1)
    dot_product_desc = tf.nn.relu(dot_product_desc)
    dot_product_desc = tf.reshape(tf.nn.l2_normalize(
        tf.reshape(dot_product_desc, [batch_size, Hc, Wc, Hc * Wc]),
        3), [batch_size, Hc, Wc, Hc, Wc])
    dot_product_desc = tf.reshape(tf.nn.l2_normalize(
        tf.reshape(dot_product_desc, [batch_size, Hc * Wc, Hc, Wc]),
        1), [batch_size, Hc, Wc, Hc, Wc])
    # dot_product_desc[id_batch, h, w, h', w'] is the dot product between the
    # descriptor at position (h, w) in the original descriptors map and the
    # descriptor at position (h', w') in the warped image

    # Compute the loss
    positive_dist = tf.maximum(0., config['positive_margin'] - dot_product_desc)
    negative_dist = tf.maximum(0., dot_product_desc - config['negative_margin'])
    loss = config['lambda_d'] * s * positive_dist + (1 - s) * negative_dist

    # Mask the pixels if bordering artifacts appear
    valid_mask = tf.ones([batch_size,
                          Hc * config['grid_size'],
                          Wc * config['grid_size']], tf.float32)\
        if valid_mask is None else valid_mask
    valid_mask = tf.to_float(valid_mask[..., tf.newaxis])  # for GPU
    valid_mask = tf.space_to_depth(valid_mask, config['grid_size'])
    valid_mask = tf.reduce_prod(valid_mask, axis=3)  # AND along the channel dim
    valid_mask = tf.reshape(valid_mask, [batch_size, 1, 1, Hc, Wc])

    normalization = tf.reduce_sum(valid_mask) * tf.to_float(Hc * Wc)
    # Summaries for debugging
    # tf.summary.scalar('nb_positive', tf.reduce_sum(valid_mask * s) / normalization)
    # tf.summary.scalar('nb_negative', tf.reduce_sum(valid_mask * (1 - s)) / normalization)
    tf.summary.scalar('positive_dist', tf.reduce_sum(valid_mask * config['lambda_d'] *
                                                     s * positive_dist) / normalization)
    tf.summary.scalar('negative_dist', tf.reduce_sum(valid_mask * (1 - s) *
                                                     negative_dist) / normalization)
    loss = tf.reduce_sum(valid_mask * loss) / normalization
    return loss


def spatial_nms(prob, size):
    """Performs non maximum suppression on the heatmap using max-pooling. This method is
    faster than box_nms, but does not suppress contiguous that have the same probability
    value.

    Arguments:
        prob: the probability heatmap, with shape `[H, W]`.
        size: a scalar, the size of the pooling window.
    """

    with tf.name_scope('spatial_nms'):
        prob = tf.expand_dims(tf.expand_dims(prob, axis=0), axis=-1)
        pooled = tf.nn.max_pool(
                prob, ksize=[1, size, size, 1], strides=[1, 1, 1, 1], padding='SAME')
        prob = tf.where(tf.equal(prob, pooled), prob, tf.zeros_like(prob))
        return tf.squeeze(prob)

def box_nms(prob, size, iou=0.1, min_prob=0.01, keep_top_k=0):
    """
    효율적인 비최대 억제 구현
    """
    with tf.name_scope('box_nms'):
        # 임계값보다 큰 확률을 가진 점들만 고려하여 메모리 사용량 감소
        pts = tf.to_float(tf.where(tf.greater_equal(prob, min_prob)))
        
        if tf.shape(pts)[0] == 0:
            return tf.zeros_like(prob)
            
        size = tf.constant(size/2.)
        
        # 더 효율적인 박스 생성
        boxes = tf.concat([pts-size, pts+size], axis=1)
        scores = tf.gather_nd(prob, tf.to_int32(pts))
        
        # CPU에서 NMS 실행 (GPU에서 지원하지 않을 수 있음)
        with tf.device('/cpu:0'):
            indices = tf.image.non_max_suppression(
                boxes, scores, tf.shape(boxes)[0], iou)
                
        # 상위 K개 결과만 유지
        if keep_top_k > 0:
            # 효율적인 top_k 구현
            k = tf.minimum(tf.shape(indices)[0], tf.constant(keep_top_k))
            indices = indices[:k]
            
        # 선택된 점에 대한 새 확률 맵 생성
        selected_pts = tf.gather(pts, indices)
        selected_scores = tf.gather(scores, indices)
        prob = tf.scatter_nd(tf.to_int32(selected_pts), selected_scores, tf.shape(prob))
    
    return prob
