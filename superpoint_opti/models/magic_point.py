# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .base_model import BaseModel, Mode
from .backbones.vgg import vgg_backbone
from .utils import detector_head, detector_loss, box_nms
from .homographies import homography_adaptation


class MagicPoint(BaseModel):
    input_spec = {
            'image': {'shape': [None, None, None, 1], 'type': tf.float32}
    }
    required_config_keys = []
    default_config = {
            'data_format': 'channels_first',
            'kernel_reg': 0.,
            'grid_size': 8,
            'detection_threshold': 0.4,
            'homography_adaptation': {'num': 0},
            'nms': 0,
            'top_k': 0
    }

    # def _model(self, inputs, mode, **config):
    #     config['training'] = (mode == Mode.TRAIN)
    #     image = inputs['image']

    #     def net(image):
    #         if config['data_format'] == 'channels_first':
    #             image = tf.transpose(image, [0, 3, 1, 2])
    #         features = vgg_backbone(image, **config)
    #         outputs = detector_head(features, **config)
    #         return outputs

    #     if (mode == Mode.PRED) and config['homography_adaptation']['num']:
    #         outputs = homography_adaptation(image, net, config['homography_adaptation'])
    #     else:
    #         outputs = net(image)

    #     prob = outputs['prob']
    #     if config['nms']:
    #         prob = tf.map_fn(lambda p: box_nms(p, config['nms'],
    #                                            min_prob=config['detection_threshold'],
    #                                            keep_top_k=config['top_k']), prob)
    #         outputs['prob_nms'] = prob
    #     pred = tf.to_int32(tf.greater_equal(prob, config['detection_threshold']))
    #     outputs['pred'] = pred

    #     return outputs

    def _model(self, inputs, mode, **config):
        config['training'] = (mode == Mode.TRAIN)
        image = inputs['image']

        def net(image):
            if config['data_format'] == 'channels_first':
                image = tf.transpose(image, [0, 3, 1, 2])
                
            # 특징 추출
            features = vgg_backbone(image, **config)
            
            # 병렬 헤드 계산
            with tf.name_scope('heads'):
                # 디텍터 헤드
                detector_outputs = detector_head(features, **config)
                
                # 추론 모드일 때만 디스크립터 계산 (필요한 경우)
                if mode == Mode.PRED and config.get('compute_descriptors', True):
                    descriptor_outputs = descriptor_head(features, **config)
                    outputs = {**detector_outputs, **descriptor_outputs}
                else:
                    outputs = detector_outputs
                    
            return outputs

        # 호모그래피 적응 사용 시
        if (mode == Mode.PRED) and config['homography_adaptation']['num']:
            outputs = homography_adaptation(image, net, config['homography_adaptation'])
        else:
            outputs = net(image)

        # NMS 적용 및 최종 예측 생성
        prob = outputs['prob']
        if config['nms']:
            # NMS 최적화 - 임계값 적용된 확률에 대해서만 실행
            mask = tf.greater_equal(prob, config['detection_threshold'])
            if tf.reduce_any(mask):
                prob_nms = tf.map_fn(
                    lambda p: box_nms(
                        p, 
                        config['nms'],
                        min_prob=config['detection_threshold'],
                        keep_top_k=config['top_k']
                    ), 
                    prob
                )
                outputs['prob_nms'] = prob_nms
            else:
                outputs['prob_nms'] = prob
        
        # 최종 예측 생성
        pred = tf.to_int32(tf.greater_equal(
            outputs.get('prob_nms', prob), 
            config['detection_threshold']
        ))
        outputs['pred'] = pred

        return outputs

    def _loss(self, outputs, inputs, **config):
        if config['data_format'] == 'channels_first':
            outputs['logits'] = tf.transpose(outputs['logits'], [0, 2, 3, 1])
        return detector_loss(inputs['keypoint_map'], outputs['logits'],
                             valid_mask=inputs['valid_mask'], **config)

    def _metrics(self, outputs, inputs, **config):
        # 유효한 영역에 대해서만 계산하기 위해 valid_mask를 곱합니다.
        pred = tf.cast(inputs['valid_mask'] * outputs['pred'], tf.float32)
        labels = tf.cast(inputs['keypoint_map'], tf.float32)
        
        # True Positive 계산
        tp = tf.reduce_sum(pred * labels)
        
        # 분모가 0이 되는 경우를 방지하기 위해 작은 상수 epsilon 추가
        epsilon = 1e-7
        precision = tf.reduce_sum(pred * labels) / (tf.reduce_sum(pred) + epsilon)
        recall = tf.reduce_sum(pred * labels) / (tf.reduce_sum(labels) + epsilon)
        
        return {'precision': precision, 'recall': recall}

