import logging
import yaml
import os
import sys
import argparse
import numpy as np
import datetime
import json
from contextlib import contextmanager
from json import dumps as pprint

from superpoint.datasets import get_dataset
from superpoint.models import get_model
from superpoint.utils.stdout_capturing import capture_outputs
from superpoint.settings import EXPER_PATH

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 경로 설정
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 로깅 설정
logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                   datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

# def train(config, n_iter, output_dir, pretrained_dir=None,
#           checkpoint_name='model.ckpt'):
#     checkpoint_path = os.path.join(output_dir, checkpoint_name)
#     with _init_graph(config) as net:
#         if pretrained_dir is not None:
#             net.load(pretrained_dir)
#         try:
#             net.train(n_iter, output_dir=output_dir,
#                       validation_interval=config.get('validation_interval', 100),
#                       save_interval=config.get('save_interval', None),
#                       checkpoint_path=checkpoint_path,
#                       keep_checkpoints=config.get('keep_checkpoints', 1))
#         except KeyboardInterrupt:
#             logging.info('Got Keyboard Interrupt, saving model and closing.')
#         net.save(os.path.join(output_dir, checkpoint_name))

def train(config, n_iter, output_dir, pretrained_dir=None,
          checkpoint_name='model.ckpt'):#'sp_v6'):#
    checkpoint_path = os.path.join(output_dir, checkpoint_name)
    model_name = os.path.basename(output_dir)
    
    with _init_graph(config) as net:
        if pretrained_dir is not None:
            # pretrained_dir가 디렉토리인지 파일인지 확인하고, 
            # 둘 중 하나라도 지원하도록 합니다.
            if os.path.isdir(pretrained_dir):
                net.load(pretrained_dir)
            elif os.path.isfile(pretrained_dir) and (pretrained_dir.endswith('.pb') or pretrained_dir.endswith('.ckpt')):
                net.load(pretrained_dir)
            else:
                raise ValueError('지원되지 않는 pretrained 모델 형식입니다: ' + pretrained_dir)
        
        start_time = datetime.datetime.now()
        
        try:
            net.train(n_iter, output_dir=output_dir,
                      validation_interval=config.get('validation_interval', 100),
                      save_interval=config.get('save_interval', None),
                      checkpoint_path=checkpoint_path,
                      keep_checkpoints=config.get('keep_checkpoints', 1))
        except KeyboardInterrupt:
            logging.info('Got Keyboard Interrupt, saving model and closing.')
        
        net.save(checkpoint_path)
        
        # 학습 후 모델 정보 수집
        end_time = datetime.datetime.now()
        training_duration = (end_time - start_time).total_seconds() / 60.0  # 분 단위
        
        # 모델 타입 결정
        model_type = "magic_point" if config['model']['name'] == 'magic_point' else "super_point"
        
        # 데이터셋 정보
        dataset = config['data']['name']
        
        # 기타 정보 (설정 파일, 가중치 파일)
        config_file = os.path.abspath(os.path.join(output_dir, 'config.yml'))
        weights_file = os.path.abspath(checkpoint_path)
        
        # 학습 파라미터 수집
        train_params = {
            "batch_size": config['model'].get('batch_size', None),
            "learning_rate": config['model'].get('learning_rate', None),
            "train_iterations": n_iter,
            "detection_threshold": config['model'].get('detection_threshold', None),
            "nms_radius": config['model'].get('nms', None),
            "validation_interval": config.get('validation_interval', None),
            "augmentation_photo": config['data'].get('augmentation', {}).get('photometric', {}).get('enable', False),
            "augmentation_homo": config['data'].get('augmentation', {}).get('homographic', {}).get('enable', False),
            "warped_pair": config['data'].get('warped_pair', {}).get('enable', False) if 'warped_pair' in config['data'] else False,
        }
        
        # SuperPoint 특화 파라미터
        if model_type == 'super_point':
            train_params.update({
                "lambda_d": config['model'].get('lambda_d', None),
                "positive_margin": config['model'].get('positive_margin', None),
                "negative_margin": config['model'].get('negative_margin', None),
                "lambda_loss": config['model'].get('lambda_loss', None)
            })
        
        # 모델 기본 정보
        model_data = {
            "model_name": model_name,
            "model_type": model_type,
            "dataset": dataset,
            "base_model": os.path.basename(pretrained_dir) if pretrained_dir else "",
            "training_date": datetime.datetime.now().strftime('%Y-%m-%d'),
            "config_file": config_file,
            "weights_file": weights_file,
            "comments": f"학습 시간: {training_duration:.2f}분, 반복 횟수: {n_iter}"
        }
        
        # 모델 정보와 학습 파라미터를 JSON 파일로 저장
        model_info_path = os.path.join(output_dir, "model_info.json")
        with open(model_info_path, 'w') as f:
            model_info = {**model_data, **train_params}
            json.dump(model_info, f, indent=2)
        
        # 엑셀 파일 업데이트
        try:
            # 평가 스크립트가 있는 디렉토리 경로 (상대 경로로 추정)
            update_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../evaluation/superpoint_update_script.py"))
            
            if os.path.exists(update_script_path):
                # 명령어 구성
                cmd = [
                    "python", update_script_path, "add",
                    "--model_name", model_name,
                    "--model_type", model_type,
                    "--dataset", dataset,
                    "--config_file", config_file,
                    "--weights_file", weights_file
                ]
                
                if pretrained_dir:
                    cmd.extend(["--base_model", os.path.basename(pretrained_dir)])
                
                # 명령 실행
                import subprocess
                subprocess.run(cmd)
                logging.info(f"모델 정보가 엑셀 파일에 업데이트되었습니다.")
            else:
                logging.warning(f"업데이트 스크립트를 찾을 수 없습니다: {update_script_path}")
        except Exception as e:
            logging.error(f"모델 정보 업데이트 중 오류 발생: {str(e)}")


def evaluate(config, output_dir, n_iter=None):
    with _init_graph(config) as net:
        net.load(output_dir)
        results = net.evaluate(config.get('eval_set', 'test'), max_iterations=n_iter)
    return results


def predict(config, output_dir, n_iter):
    pred = []
    data = []
    with _init_graph(config, with_dataset=True) as (net, dataset):
        if net.trainable:
            net.load(output_dir)
        test_set = dataset.get_test_set()
        for _ in range(n_iter):
            data.append(next(test_set))
            pred.append(net.predict(data[-1], keys='*'))
    return pred, data


def set_seed(seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)


def get_num_gpus():
    return len(os.environ.get('CUDA_VISIBLE_DEVICES', '1').split(','))


@contextmanager
def _init_graph(config, with_dataset=False):
    set_seed(config.get('seed', int.from_bytes(os.urandom(4), byteorder='big')))
    n_gpus = get_num_gpus()
    logging.info('Number of GPUs detected: {}'.format(n_gpus))

    dataset = get_dataset(config['data']['name'])(**config['data'])
    model = get_model(config['model']['name'])(
            data={} if with_dataset else dataset.get_tf_datasets(),
            n_gpus=n_gpus, **config['model'])
    model.__enter__()
    if with_dataset:
        yield model, dataset
    else:
        yield model
    model.__exit__()
    tf.reset_default_graph()


def _cli_train(config, output_dir, args):
    assert 'train_iter' in config

    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    if args.pretrained_model is not None:
        pretrained_dir = os.path.join(EXPER_PATH, args.pretrained_model)
        if not os.path.exists(pretrained_dir):
            raise ValueError("Missing pretrained model: " + pretrained_dir)
    else:
        pretrained_dir = None
        
    train(config, config['train_iter'], output_dir, pretrained_dir)

    if args.eval:
        _cli_eval(config, output_dir, args)


def _cli_eval(config, output_dir, args):
    # Load model config from previous experiment
    with open(os.path.join(output_dir, 'config.yml'), 'r') as f:
        model_config = yaml.load(f)['model']
    model_config.update(config.get('model', {}))
    config['model'] = model_config

    results = evaluate(config, output_dir, n_iter=config.get('eval_iter'))

    # Print and export results
    logging.info('Evaluation results: \n{}'.format(
        pprint(results, indent=2, default=str)))
    with open(os.path.join(output_dir, 'eval.txt'), 'a') as f:
        f.write('Evaluation for {} dataset:\n'.format(config['data']['name']))
        for r, v in results.items():
            f.write('\t{}:\n\t\t{}\n'.format(r, v))
        f.write('\n')


# TODO
def _cli_pred(config, args):
    raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Training command
    p_train = subparsers.add_parser('train')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.add_argument('--eval', action='store_true')
    p_train.add_argument('--pretrained_model', type=str, default=None)
    p_train.set_defaults(func=_cli_train)

    # Evaluation command
    p_train = subparsers.add_parser('evaluate')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.set_defaults(func=_cli_eval)

    # Inference command
    p_train = subparsers.add_parser('predict')
    p_train.add_argument('config', type=str)
    p_train.add_argument('exper_name', type=str)
    p_train.set_defaults(func=_cli_pred)

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with capture_outputs(os.path.join(output_dir, 'log')):
        logging.info('Running command {}'.format(args.command.upper()))
        args.func(config, output_dir, args)
