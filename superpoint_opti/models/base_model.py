from abc import ABCMeta, abstractmethod
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.client import timeline
import numpy as np
from tqdm import tqdm
import os.path as osp
import itertools

from superpoint.utils.tools import dict_update


class Mode:
    TRAIN = 'train'
    EVAL = 'eval'
    PRED = 'pred'


class BaseModel(metaclass=ABCMeta):
    """Base model class.

    Arguments:
        data: A dictionary of `tf.data.Dataset` objects, can include the keys
            `"training"`, `"validation"`, and `"test"`.
        n_gpus: An integer, the number of GPUs available.
        data_shape: A dictionary, where the keys are the input features of the prediction
            network and the values are the associated shapes. Only required if `data` is
            empty or `None`.
        config: A dictionary containing the configuration parameters.
            Entries `"batch_size"` and `"learning_rate"` are required.

    Models should inherit from this class and implement the following methods:
        `_model`, `_loss`, and `_metrics`.
    Additionally, the following static attributes should be defined:
        input_spec: A dictionary, where the keys are the input features (e.g. `"image"`)
            and the associated values are dictionaries containing `"shape"` (list of
            dimensions, e.g. `[N, H, W, C]` where `None` indicates an unconstrained
            dimension) and `"type"` (e.g. `tf.float32`).
        required_config_keys: A list containing the required configuration entries.
        default_config: A dictionary of potential default configuration values.
    """
    dataset_names = set(['training', 'validation', 'test'])
    required_baseconfig = ['batch_size', 'learning_rate']
    _default_config = {'eval_batch_size': 1, 'pred_batch_size': 1}

    @abstractmethod
    def _model(self, inputs, mode, **config):
        """Implements the graph of the model.

        This method is called three times: for training, evaluation and prediction (see
        the `mode` argument) and can return different tensors depending on the mode.
        It is a good practice to support both NCHW (channels first) and NHWC (channels
        last) data formats using a dedicated configuration entry.

        Arguments:
            inputs: A dictionary of input features, where the keys are their names
                (e.g. `"image"`) and the values of type `tf.Tensor`. Same keys as in the
                datasets given during the object instantiation.
            mode: An attribute of the `Mode` class, either `Mode.TRAIN`, `Mode.EVAL` or
                `Mode.PRED`.
            config: A configuration dictionary, given during the object instantiantion.

        Returns:
            A dictionary of outputs, where the keys are their names (e.g. `"logits"`) and
            the values are the corresponding `tf.Tensor`.
        """
        raise NotImplementedError

    @abstractmethod
    def _loss(self, outputs, inputs, **config):
        """Implements the sub-graph computing the training loss.

        This method is called on the outputs of the `_model` method in training mode.

        Arguments:
            outputs: A dictionary, as retuned by `_model` called with `mode=Mode.TRAIN`.
            inputs: A dictionary of input features (see same as for `_model`).
            config: A configuration dictionary.

        Returns:
            A tensor corresponding to the loss to be minimized during training.
        """
        raise NotImplementedError

    @abstractmethod
    def _metrics(self, outputs, inputs, **config):
        """Implements the sub-graph computing the evaluation metrics.

        This method is called on the outputs of the `_model` method in evaluation mode.

        Arguments:
            outputs: A dictionary, as retuned by `_model` called with `mode=Mode.EVAL`.
            inputs: A dictionary of input features (see same as for `_model`).
            config: A configuration dictionary.

        Returns:
            A dictionary of metrics, where the keys are their names (e.g. "`accuracy`")
            and the values are the corresponding `tf.Tensor`.
        """
        raise NotImplementedError

    def __init__(self, data={}, n_gpus=1, data_shape=None, **config):
        self.datasets = data
        self.data_shape = data_shape
        self.n_gpus = n_gpus
        self.graph = tf.get_default_graph()
        self.name = self.__class__.__name__.lower()  # get child name
        self.trainable = getattr(self, 'trainable', True)

        # Update config
        self.config = dict_update(self._default_config,
                                  getattr(self, 'default_config', {}))
        self.config = dict_update(self.config, config)

        required = self.required_baseconfig + getattr(self, 'required_config_keys', [])
        for r in required:
            assert r in self.config, 'Required configuration entry: \'{}\''.format(r)
        assert set(self.datasets) <= self.dataset_names, \
            'Unknown dataset name: {}'.format(set(self.datasets)-self.dataset_names)
        assert n_gpus >= 0

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self._build_graph()

    def _unstack_nested_dict(self, d, num):
        return {k: self._unstack_nested_dict(v, num) if isinstance(v, dict)
                else tf.unstack(v, num=num, axis=0) for k, v in d.items()}

    def _shard_nested_dict(self, d, num):
        shards = [{} for _ in range(num)]
        for k, v in d.items():
            if isinstance(v, dict):
                stack = self._shard_nested_dict(v, num)
            else:
                stack = [tf.stack(v[i::num]) for i in range(num)]
            shards = [{**s, k: stack[i]} for i, s in enumerate(shards)]
        return shards

    def _gpu_tower(self, data, mode, batch_size):
        # 분기 추가: 0개의, GPU를 사용하거나 데이터 병렬성을 위해 여러 개 사용
        n_shards = max(1, self.n_gpus)
        device = 'cpu' if self.n_gpus == 0 else 'gpu'
        
        with tf.device('/cpu:0'):
            with tf.name_scope('{}_data_sharding'.format(mode)):
                shards = self._unstack_nested_dict(data, batch_size*n_shards)
                shards = self._shard_nested_dict(shards, n_shards)

        # 각 GPU에 대한 타워 생성
        tower_losses = []
        tower_gradvars = []
        tower_preds = []
        tower_metrics = []
        
        # 소프트 배치 옵션 추가
        soft_placement = True  # 이렇게 설정하면 GPU에서 불가능한 연산은 자동으로 CPU로 이동
        
        for i in range(n_shards):
            worker = '/{}:{}'.format(device, i)
            device_setter = tf.train.replica_device_setter(
                    worker_device=worker, ps_device='/cpu:0', ps_tasks=1)
            with tf.name_scope('{}_tower{}'.format(mode, i)) as scope:
                # 소프트 배치 옵션을 사용하는 컨텍스트 관리자 추가
                with tf.device(device_setter), tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE), \
                    tf.device(lambda op: self._maybe_move_to_cpu(op, worker)):
                    net_outputs = self._model(shards[i], mode, **self.config)
                    if mode == Mode.TRAIN:
                        loss = self._loss(net_outputs, shards[i], **self.config)
                        loss += tf.reduce_sum(
                                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope))
                        model_params = tf.trainable_variables()
                        grad = tf.gradients(loss, model_params)
                        tower_losses.append(loss)
                        tower_gradvars.append(zip(grad, model_params))
                        if i == 0:
                            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
                    elif mode == Mode.EVAL:
                        tower_metrics.append(self._metrics(
                            net_outputs, shards[i], **self.config))
                    else:
                        tower_preds.append(net_outputs)

        if mode == Mode.TRAIN:
            return tower_losses, tower_gradvars, update_ops
        elif mode == Mode.EVAL:
            return tower_metrics
        else:
            # 타워 예측의 인터리브
            return {k: tf.stack(
                [v for z in zip(*[tf.unstack(p[k], num=batch_size) for p in tower_preds])
                for v in z]) for k in tower_preds[0]}
             
    # 이 메서드를 추가하여 특정 연산을 CPU로 이동
    def _maybe_move_to_cpu(self, op, worker_device):
        # 이 연산들은 GPU에서 지원되지 않으므로 CPU로 강제 이동
        cpu_ops = ['DenseToDenseSetOperation', 'StringSplit', 'StringJoin', 'StringToHashBucket']
        for cpu_op in cpu_ops:
            if cpu_op in op.type:
                return '/cpu:0'
        return worker_device

    def _train_graph(self, data):
        tower_losses, tower_gradvars, update_ops = self._gpu_tower(
                data, Mode.TRAIN, self.config['batch_size'])

        # Perform the consolidation on CPU
        gradvars = []
        with tf.device('/cpu:0'):
            # Average losses and gradients
            with tf.name_scope('tower_averaging'):
                all_grads = {}
                for grad, var in itertools.chain(*tower_gradvars):
                    if grad is not None:
                        all_grads.setdefault(var, []).append(grad)
                for var, grads in all_grads.items():
                    if len(grads) == 1:
                        avg_grad = grads[0]
                    else:
                        avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
                    gradvars.append((avg_grad, var))
                self.loss = tf.reduce_mean(tower_losses)
                tf.summary.scalar('loss', self.loss)

            # Create optimizer ops
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            opt = tf.train.AdamOptimizer(self.config['learning_rate'])
            with tf.control_dependencies(update_ops):
                self.trainer = opt.apply_gradients(
                        gradvars, global_step=self.global_step)

    def _eval_graph(self, data):
        tower_metrics = self._gpu_tower(data, Mode.EVAL, self.config['eval_batch_size'])
        with tf.device('/cpu:0'):
            self.metrics = {m: tf.reduce_mean(tf.stack([t[m] for t in tower_metrics]))
                            for m in tower_metrics[0]}

    def _pred_graph(self, data):
        pred_out = self._gpu_tower(data, Mode.PRED, self.config['pred_batch_size'])
        self.pred_out = {n: tf.identity(p, name=n) for n, p in pred_out.items()}

    def _build_graph(self):
        # Training and evaluation network, if tf datasets provided
        if self.datasets:
            # Generate iterators for the given tf datasets
            self.dataset_iterators = {}
            with tf.device('/cpu:0'):
                for n, d in self.datasets.items():
                    output_shapes = d.output_shapes
                    if n == 'training':
                        train_batch = self.config['batch_size']*self.n_gpus
                        d = d.repeat().padded_batch(
                                train_batch, output_shapes).prefetch(train_batch)
                        self.dataset_iterators[n] = d.make_one_shot_iterator()
                    else:
                        d = d.padded_batch(self.config['eval_batch_size']*self.n_gpus,
                                           output_shapes)
                        self.dataset_iterators[n] = d.make_initializable_iterator()
                    output_types = d.output_types
                    output_shapes = d.output_shapes
                    self.datasets[n] = d

                    # Perform compatibility checks with the inputs of the child model
                    for i, spec in self.input_spec.items():
                        assert i in output_shapes
                        tf.TensorShape(output_shapes[i]).assert_is_compatible_with(
                                tf.TensorShape(spec['shape']))

                # Used for input shapes of the prediction network
                if self.data_shape is None:
                    self.data_shape = output_shapes

                # Handle for the feedable iterator
                self.handle = tf.placeholder(tf.string, shape=[])
                iterator = tf.data.Iterator.from_string_handle(
                        self.handle, output_types, output_shapes)
                data = iterator.get_next()

            # Build the actual training and evaluation models
            if self.trainable:
                self._train_graph(data)
            self._eval_graph(data)
            self.summaries = tf.summary.merge_all()

        # Prediction network with feed_dict
        if self.data_shape is None:
            self.data_shape = {i: spec['shape'] for i, spec in self.input_spec.items()}
        self.pred_in = {i: tf.placeholder(spec['type'], shape=self.data_shape[i], name=i)
                        for i, spec in self.input_spec.items()}
        self._pred_graph(self.pred_in)

        # Start session
        sess_config = tf.ConfigProto(device_count={'GPU': self.n_gpus},
                                     allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        # Register tf dataset handles
        if self.datasets:
            self.dataset_handles = {}
            for n, i in self.dataset_iterators.items():
                self.dataset_handles[n] = self.sess.run(i.string_handle())

        self.sess.run([tf.global_variables_initializer(),
                       tf.local_variables_initializer()])

    def train(self, iterations, validation_interval=100, output_dir=None, profile=False,
          save_interval=None, checkpoint_path=None, keep_checkpoints=1):
        assert self.trainable, 'Model is not trainable.'
        assert 'training' in self.datasets, 'Training dataset is required.'
        
        # 텐서보드 설정
        if output_dir is not None:
            train_writer = tf.summary.FileWriter(output_dir)
            
        # 모델 저장 설정
        if not hasattr(self, 'saver'):
            with tf.device('/cpu:0'):
                self.saver = tf.train.Saver(save_relative_paths=True,
                                            max_to_keep=keep_checkpoints)
        
        # 그래프 최종화
        if not self.graph.finalized:
            self.graph.finalize()
            
        # 프로파일링 설정
        options = None
        run_metadata = None
        if profile:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

        # 학습률 감소 설정
        lr_decay_steps = self.config.get('lr_decay_steps', 0)
        lr_decay_rate = self.config.get('lr_decay_rate', 0.9)
        if lr_decay_steps > 0:
            # 학습률 감소 스케줄러 추가
            lr = tf.train.exponential_decay(
                self.config['learning_rate'],
                self.global_step,
                lr_decay_steps,
                lr_decay_rate,
                staircase=True)
            # 옵티마이저 학습률 업데이트
            self.optimizer = tf.train.AdamOptimizer(lr)
            with tf.control_dependencies(self.update_ops):
                self.trainer = self.optimizer.apply_gradients(
                    self.gradvars, global_step=self.global_step)

        # 조기 중단 변수 초기화
        best_val_loss = float('inf')
        patience = self.config.get('early_stopping_patience', 0)
        patience_counter = 0
        
        tf.logging.info('Start training')
        for i in range(iterations):
            # 학습 단계 실행
            loss, summaries, _ = self.sess.run(
                [self.loss, self.summaries, self.trainer],
                feed_dict={self.handle: self.dataset_handles['training']},
                options=options, run_metadata=run_metadata)
                
            # 모델 저장
            if save_interval and checkpoint_path and (i+1) % save_interval == 0:
                self.save(checkpoint_path)
                
            # 검증 실행
            if 'validation' in self.datasets and i % validation_interval == 0:
                metrics = self.evaluate('validation', mute=True)
                val_loss = metrics.get('loss', 0)
                
                # 로깅
                tf.logging.info(
                    'Iter {:4d}: loss {:.4f}'.format(i, loss) +
                    ''.join([', {} {:.4f}'.format(m, metrics[m]) for m in metrics]))
                    
                # 텐서보드에 추가
                if output_dir is not None:
                    train_writer.add_summary(summaries, i)
                    metrics_summaries = tf.Summary(value=[
                        tf.Summary.Value(tag=m, simple_value=v)
                        for m, v in metrics.items()])
                    train_writer.add_summary(metrics_summaries, i)
                    
                    # 프로파일링 정보 저장
                    if profile and i != 0:
                        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                        chrome_trace = fetched_timeline.generate_chrome_trace_format()
                        with open(osp.join(output_dir,
                                        'profile_{}.json'.format(i)), 'w') as f:
                            f.write(chrome_trace)
                
                # 조기 중단 로직
                if patience > 0:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # 가장 좋은 모델 저장
                        if checkpoint_path:
                            self.save(checkpoint_path + '_best')
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            tf.logging.info(f'Early stopping at iteration {i}')
                            break
                            
    tf.logging.info('Training finished')

    def predict(self, data, keys='pred', batch=False):
        assert set(data.keys()) >= set(self.input_spec.keys())
        if isinstance(keys, str):
            if keys == '*':
                op = self.pred_out  # just gather all outputs
            else:
                op = self.pred_out[keys]
        else:
            op = {k: self.pred_out[k] for k in keys}
        if not batch:  # add batch dimension
            data = {d: [v] for d, v in data.items()}
        feed = {self.pred_in[i]: data[i] for i in self.input_spec}
        pred = self.sess.run(op, feed_dict=feed)
        if not batch:  # remove batch dimension
            if isinstance(pred, dict):
                pred = {p: v[0] for p, v in pred.items()}
            else:
                pred = pred[0]
        return pred

    def evaluate(self, dataset, max_iterations=None, mute=False):
        assert dataset in self.datasets
        self.sess.run(self.dataset_iterators[dataset].initializer)

        if not mute:
            tf.logging.info('Starting evaluation of dataset \'{}\''.format(dataset))
            if max_iterations:
                pbar = tqdm(total=max_iterations, ascii=True)
        i = 0
        metrics = []
        while True:
            try:
                metrics.append(self.sess.run(self.metrics,
                               feed_dict={self.handle: self.dataset_handles[dataset]}))
            except tf.errors.OutOfRangeError:
                break
            if max_iterations:
                i += 1
                if not mute:
                    pbar.update(1)
                if i == max_iterations:
                    break
        if not mute:
            tf.logging.info('Finished evaluation')
            if max_iterations:
                pbar.close()

        # List of dicts to dict of lists
        metrics = dict(zip(metrics[0], zip(*[m.values() for m in metrics])))
        metrics = {m: np.nanmean(metrics[m], axis=0) for m in metrics}
        return metrics

    def load(self, checkpoint_path):
        with tf.device('/cpu:0'):
            saver = tf.train.Saver(save_relative_paths=True)
        if tf.gfile.IsDirectory(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
            if checkpoint_path is None:
                raise ValueError('Checkpoint directory is empty.')
        saver.restore(self.sess, checkpoint_path)

    def save(self, checkpoint_path):
        step = self.sess.run(self.global_step)
        tf.logging.info('Saving checkpoint for iteration #{}'.format(step))
        self.saver.save(self.sess, checkpoint_path, write_meta_graph=False,
                        global_step=step)

    def close(self):
        self.sess.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()