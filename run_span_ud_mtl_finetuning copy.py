# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT classification or regression finetuning runner in TF 2.x."""

import enum
import functools
import json
import math
import os
from collections import Counter

# Import libraries
from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow as tf
import tensorflow_addons as tfa

from official.common import distribute_utils
from official.modeling import performance
from official.nlp import optimization
from official.nlp.bert import bert_models
from official.nlp.bert import common_flags
from official.nlp.bert import configs as bert_configs
from official.nlp.bert import input_pipeline
from official.nlp.bert import model_saving_utils
from official.utils.misc import keras_utils

from official.nlp.bert import tokenization_morp_utagger_ver
from official.nlp.data import create_span_ud_nv_finetuning_data


# test 할 때는 'train_and_eval'=>predict으로 변경
flags.DEFINE_enum(
    'mode', 'train_and_eval', ['train_and_eval', 'export_only', 'predict'],
    'One of {"train_and_eval", "export_only", "predict"}. `train_and_eval`: '
    'trains the model and evaluates in the meantime. '
    '`export_only`: will take the latest checkpoint inside '
    'model_dir and export a `SavedModel`. `predict`: takes a checkpoint and '
    'restores the model to output predictions on the test set.')

# flags.DEFINE_integer(
#     'max_seq_length', 512,
#     'The maximum total input sequence length after WordPiece tokenization. '
#     'Sequences longer than this will be truncated, and sequences shorter '
#     'than this will be padded.')

flags.DEFINE_string('train_data_path', r'C:\Users\klplab\Desktop\ubert\train_span_ud_nv_ej',
                    'Path to training data for BERT classifier.')

flags.DEFINE_string('meta_data_path', r'C:\Users\klplab\Desktop\ubert\express_span_ud_nv_ej.json',
                    'Path to training data for BERT classifier.')
# validation
# flags.DEFINE_string('eval_data_path', r'C:\Users\klplab\Desktop\ubert\validation_span_ud_nv_ej',
#                     'Path to evaluation data for BERT classifier.')

# test
flags.DEFINE_string('eval_data_path', r'C:\Users\klplab\Desktop\ubert\test_span_ud_nv_ej',
                    'Path to evaluation data for BERT classifier.')

flags.DEFINE_string('eval_file_path', r'C:\Users\klplab\Desktop\ubert\test_span_ud_nv_ej.txt',
                    'Path to evaluation data for BERT classifier.')

# flags.DEFINE_string("vocab_file", 'D:/UTagger_bert_model/update_token_list_1018_V2.txt',
#                     "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_integer('train_data_size', None, 'Number of training samples '
                     'to use. If None, uses the full train data. '
                     '(default: None).')

flags.DEFINE_string('predict_checkpoint_path', None,
                    'Path to the checkpoint for predictions.')

flags.DEFINE_integer(
    'num_eval_per_epoch', 1,
    'Number of evaluations per epoch. The purpose of this flag is to provide '
    'more granular evaluation scores and checkpoints. For example, if original '
    'data has N samples and num_eval_per_epoch is n, then each epoch will be '
    'evaluated every N/n samples.')

flags.DEFINE_integer('train_batch_size', 16, 'Batch size for training.')
flags.DEFINE_integer('eval_batch_size', 16, 'Batch size for evaluation.')

common_flags.define_common_bert_flags()

FLAGS = flags.FLAGS

def export_classifier(model_export_path, num_label, bert_config,
                      model_dir):
  """Exports a trained model as a `SavedModel` for inference.

  Args:
    model_export_path: a string specifying the path to the SavedModel directory.
    input_meta_data: dictionary containing meta data about input and model.
    bert_config: Bert configuration file to define core bert layers.
    model_dir: The directory where the model weights and training/evaluation
      summaries are stored.

  Raises:
    Export path is not specified, got an empty string or None.
  """
  if not model_export_path:
    raise ValueError('Export path is not specified: %s' % model_export_path)
  if not model_dir:
    raise ValueError('Export path is not specified: %s' % model_dir)

  # Export uses float32 for now, even if training uses mixed precision.
  tf.keras.mixed_precision.set_global_policy('float32')
  classifier_model = bert_models.classifier_token_model(
      bert_config,
      num_label,
      hub_module_url=FLAGS.hub_module_url,
      hub_module_trainable=False)[0]

  model_saving_utils.export_bert_model(
      model_export_path, model=classifier_model, checkpoint_dir=model_dir)


def get_dataset_fn(input_file_pattern,
                   max_seq_length,
                   global_batch_size,
                   is_training,
                   label_type=tf.int64,
                   include_sample_weights=False,
                   num_samples=None):
  """Gets a closure to create a dataset."""

  def _dataset_fn(ctx=None):
    """Returns tf.data.Dataset for distributed BERT pretraining."""
    batch_size = ctx.get_per_replica_batch_size(
        global_batch_size) if ctx else global_batch_size
    dataset = input_pipeline.create_span_ud_mtl_dataset(
        tf.io.gfile.glob(input_file_pattern),
        max_seq_length,
        batch_size,
        is_training=is_training,
        input_pipeline_context=ctx,
        label_type=label_type,
        include_sample_weights=include_sample_weights,
        num_samples=num_samples)
    return dataset

  return _dataset_fn

# 전부 변경, 비동기 방식 바꿀것 why don't running async
def get_predictions_and_labels(strategy,
                               trained_model,
                               eval_input_fn,
                               is_regression=False,
                               return_probs=False):
  """Obtains predictions of trained model on evaluation data.

  Note that list of labels is returned along with the predictions because the
  order changes on distributing dataset over TPU pods.

  Args:
    strategy: Distribution strategy.
    trained_model: Trained model with preloaded weights.
    eval_input_fn: Input function for evaluation data.
    is_regression: Whether it is a regression task.
    return_probs: Whether to return probabilities of classes.

  Returns:
    predictions: List of predictions.
    labels: List of gold labels corresponding to predictions.
  """

  @tf.function
  def test_step(iterator):
    """Computes predictions on distributed devices."""

    def _test_step_fn(inputs):
      """Replicated predictions."""
      inputs, labels = inputs
      logits = trained_model(inputs, training=False)
      if not is_regression:
        probabilities1 = tf.nn.softmax(logits['start_logits'])
        probabilities2 = tf.nn.softmax(logits['end_logits'])
        domain_pred = tf.nn.softmax(logits['tags'])
        return probabilities1, probabilities2, domain_pred, labels
      else:
        return logits, labels

    start, end, domain, labels = strategy.run(_test_step_fn, args=(next(iterator),))
    # outputs: current batch logits as a tuple of shard logits
    start = tf.nest.map_structure(strategy.experimental_local_results,
                                    start)
    end = tf.nest.map_structure(strategy.experimental_local_results,
                                    end)
    domain = tf.nest.map_structure(strategy.experimental_local_results,
                                    domain)
    labels = tf.nest.map_structure(strategy.experimental_local_results, labels)
    return start, end, domain, labels

  def _run_evaluation(test_iterator):
    """Runs evaluation steps."""
    preds_s, preds_e, golds_s, golds_e, preds_d, golds_d = list(), list(), list(), list(), list(), list()
    try:
      with tf.experimental.async_scope():
        while True:
          probabilities_s, probabilities_e, domain_pred, labels = test_step(test_iterator)
          for cur_probs_s, cur_probs_e, cur_probs_d, cur_labels_s, cur_labels_e, cur_labels_d in zip(probabilities_s, probabilities_e, domain_pred, labels['start_label'], labels['end_label'], labels['domain']):
            if return_probs:
              preds_s.extend(cur_probs_s.numpy().tolist())
              preds_e.extend(cur_probs_e.numpy().tolist())
              preds_d.extend(cur_probs_d.numpy().tolist())
            else:
              preds_s.extend(tf.math.argmax(cur_probs_s, axis=1).numpy())
              preds_e.extend(tf.math.argmax(cur_probs_e, axis=1).numpy())
              preds_d.extend(tf.math.argmax(cur_probs_d, axis=1).numpy())
            golds_s.extend(cur_labels_s.numpy().tolist())
            golds_e.extend(cur_labels_e.numpy().tolist())
            golds_d.extend(cur_labels_d.numpy().tolist())
    except (StopIteration, tf.errors.OutOfRangeError):
      tf.experimental.async_clear_error()
    return preds_s, preds_e, preds_d, golds_s, golds_e, golds_d

  test_iter = iter(strategy.distribute_datasets_from_function(eval_input_fn))
  predictions_s, predictions_e, predictions_d, labels_s, labels_e, labels_d = _run_evaluation(test_iter)

  return predictions_s, predictions_e, predictions_d, labels_s, labels_e, labels_d

#class weight
pos_weight = [0.1747, 0, 5.6782, 1.2238, 8.055, 1.0822, 31.8255, 4.9222]

def get_loss_fn(num_classes):
  """Gets the classification loss function."""
  def classification_loss_fn(start_label, end_label, tag_label, start_logits, end_logits, tag_logits, attention_mask):
    """Classification loss."""

    tag_label = tf.squeeze(tag_label)
    log_probs_tags = tf.nn.log_softmax(tag_logits, axis=-1)
    one_hot_labels_tags = tf.one_hot(
        tf.cast(tag_label, dtype=tf.int32), depth=4, dtype=tf.float32)
    per_example_loss_tags = -tf.reduce_sum(
        tf.cast(one_hot_labels_tags, dtype=tf.float32) * log_probs_tags, axis=-1)
    per_example_loss_tags = tf.reduce_mean(per_example_loss_tags)

    active_loss = attention_mask==1
    start_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=start_logits, labels=start_label)
    end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=end_logits, labels=end_label)
    active_start_loss = tf.boolean_mask(start_loss, active_loss)
    active_end_loss = tf.boolean_mask(end_loss, active_loss)

    return (tf.reduce_mean((active_start_loss +active_end_loss) / 2), per_example_loss_tags)

  return classification_loss_fn

def run_keras_compile_fit(model_dir,
                          strategy,
                          model_fn,
                          train_input_fn,
                          eval_input_fn,
                          loss_fn,
                          metric_fn,
                          init_checkpoint,
                          epochs,
                          steps_per_epoch,
                          steps_per_loop,
                          eval_steps,
                          training_callbacks=True,
                          custom_callbacks=None):
  """Runs BERT classifier model using Keras compile/fit API."""

  with strategy.scope():
    training_dataset = train_input_fn()
    evaluation_dataset = eval_input_fn() if eval_input_fn else None
    bert_model, sub_model = model_fn()
    optimizer = bert_model.optimizer

    if init_checkpoint:
      checkpoint = tf.train.Checkpoint(model=sub_model, encoder=sub_model)
      # checkpoint.read(init_checkpoint).assert_existing_objects_matched()
      checkpoint.read(init_checkpoint).expect_partial()

    #crf일때만 필요 없음.
    # if not isinstance(metric_fn, (list, tuple)):
    #   metric_fn = [metric_fn]

    
    bert_model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        #crf일때만 필요 없음.
        # metrics=[fn() for fn in metric_fn],
        # crf
        metrics=metric_fn,
        steps_per_execution=steps_per_loop)
  
    summary_dir = os.path.join(model_dir, 'summaries')
    summary_callback = tf.keras.callbacks.TensorBoard(summary_dir)
    checkpoint = tf.train.Checkpoint(model=bert_model, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=model_dir,
        max_to_keep=None,
        step_counter=optimizer.iterations,
        checkpoint_interval=0)
    checkpoint_callback = keras_utils.SimpleCheckpoint(checkpoint_manager)

    if training_callbacks:
      if custom_callbacks is not None:
        custom_callbacks += [summary_callback, checkpoint_callback]
      else:
        custom_callbacks = [summary_callback, checkpoint_callback]
    # bert_model.summary()
    history = bert_model.fit(
        x=training_dataset,
        validation_data=evaluation_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_steps=eval_steps,
        callbacks=custom_callbacks)
        
    stats = {'total_training_steps': steps_per_epoch * epochs}
    if 'loss' in history.history:
      stats['train_loss'] = history.history['loss'][-1]
    if 'val_accuracy' in history.history:
      stats['eval_metrics'] = history.history['val_accuracy'][-1]
    return bert_model, stats


def run_bert_classifier(strategy,
                        bert_config,
                        model_dir,
                        epochs,
                        steps_per_epoch,
                        steps_per_loop,
                        eval_steps,
                        warmup_steps,
                        initial_lr,
                        init_checkpoint,
                        train_input_fn,
                        eval_input_fn,
                        num_classes,
                        training_callbacks=True,
                        custom_callbacks=None,
                        custom_metrics=None):
  """Run BERT classifier training using low-level API."""
  max_seq_length = FLAGS.max_seq_length
  is_regression = num_classes == 1

  def _get_classifier_model():
    """Gets a classifier model."""
    classifier_model, core_model = (
        bert_models.classifier_token_model(
            bert_config,
            num_classes,
            max_seq_length,
            hub_module_url=FLAGS.hub_module_url,
            hub_module_trainable=FLAGS.hub_module_trainable))
    optimizer = optimization.create_optimizer(initial_lr,
                                              steps_per_epoch * epochs,
                                              warmup_steps, FLAGS.end_lr,
                                              FLAGS.optimizer_type)
    classifier_model.optimizer = performance.configure_optimizer(
        optimizer,
        use_float16=common_flags.use_float16(),
        use_graph_rewrite=common_flags.use_graph_rewrite())
    return classifier_model, core_model

  # tf.keras.losses objects accept optional sample_weight arguments (eg. coming
  # from the dataset) to compute weighted loss, as used for the regression
  # tasks. The classification tasks, using the custom get_loss_fn don't accept
  # sample weights though.
  loss_fn = (tf.keras.losses.MeanSquaredError() if is_regression
             else get_loss_fn(num_classes))

  # loss_fn = (tf.keras.losses.BinaryCrossentropy(from_logits=True) if is_regression
  #            else get_loss_fn(num_classes))

  # Defines evaluation metrics function, which will create metrics in the
  # correct device and strategy scope.
  if custom_metrics:
    metric_fn = custom_metrics
  elif is_regression:
    # metric_fn = functools.partial(
    #     tf.keras.metrics.MeanSquaredError,
    #     'mean_squared_error',
    #     dtype=tf.float32)
    metric_fn = functools.partial(
        tf.keras.metrics.BinaryAccuracy,
        'binary_accuracy',
        dtype=tf.float32) 
  else:
    # metric_fn = functools.partial(
    #     tf.keras.metrics.SparseCategoricalAccuracy,
    #     'accuracy',
    #     dtype=tf.float32)
    # crf
    metric_fn = custom_metric_fn

  # Start training using Keras compile/fit API.
  logging.info('Training using TF 2.x Keras compile/fit API with '
               'distribution strategy.')
  return run_keras_compile_fit(
      model_dir,
      strategy,
      _get_classifier_model,
      train_input_fn,
      eval_input_fn,
      loss_fn,
      metric_fn,
      init_checkpoint,
      epochs,
      steps_per_epoch,
      steps_per_loop,
      eval_steps,
      training_callbacks=training_callbacks,
      custom_callbacks=custom_callbacks)

# tf.config.run_functions_eagerly(True) 

def bert_extract_item(start_logits, end_logits):
    S = []
    for i, s_l in enumerate(start_logits):
        if s_l == 0:
            continue
        for j, e_l in enumerate(end_logits[i:]):
            if s_l == e_l:
                S.append((s_l, i, i + j))
                break
    return S

class SpanEntityScore(object):
    def __init__(self, id2label):
        self.id2label = id2label
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([self.id2label[x[0]] for x in self.origins])
        found_counter = Counter([self.id2label[x[0]] for x in self.founds])
        right_counter = Counter([self.id2label[x[0]] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, true_subject, pred_subject):
        self.origins.extend(true_subject)
        self.founds.extend(pred_subject)
        self.rights.extend([pre_entity for pre_entity in pred_subject if pre_entity in true_subject])


def custom_metric_fn(start_label, end_label, tag_label, start_pred, end_pred, tag_pred, attention_mask) :
  active_loss = attention_mask==1
  start_pred_sf = tf.math.softmax(start_pred, axis=-1)
  start_answer = tf.keras.metrics.sparse_categorical_accuracy(start_label, start_pred_sf ) # b * seq
  end_pred_sf = tf.math.softmax(end_pred, axis=-1)
  end_answer = tf.keras.metrics.sparse_categorical_accuracy(end_label, end_pred_sf ) # b * seq
  start_numeric = tf.boolean_mask(start_answer, active_loss)
  end_numeric = tf.boolean_mask(end_answer, active_loss)

  tag_pred_sf = tf.math.softmax(tag_pred, axis=-1)
  tag_answer = tf.keras.metrics.sparse_categorical_accuracy(tag_label, tag_pred_sf ) # b * 1

  return tf.reduce_mean(start_numeric), tf.reduce_mean(end_numeric), tf.reduce_mean(tag_answer)


def run_bert(strategy,
             model_config,
             eval_len,
             meta_datas,
             train_input_fn=None,
             eval_input_fn=None,
             init_checkpoint=None,
             custom_callbacks=None,
             custom_metrics=None):
  """Run BERT training."""
  # Enables XLA in Session Config. Should not be set for TPU.
  keras_utils.set_session_config(FLAGS.enable_xla)
  performance.set_mixed_precision_policy(common_flags.dtype())

  epochs = FLAGS.num_train_epochs * FLAGS.num_eval_per_epoch
  train_data_size = (meta_datas['train_data_size'] // FLAGS.num_eval_per_epoch)

  if FLAGS.train_data_size:
    train_data_size = min(train_data_size, FLAGS.train_data_size)
    logging.info('Updated train_data_size: %s', train_data_size)
  steps_per_epoch = int(train_data_size / FLAGS.train_batch_size)
  warmup_steps = int(epochs * train_data_size * 0.1 / FLAGS.train_batch_size)
  eval_steps = int(
      math.ceil(eval_len / FLAGS.eval_batch_size))

  if not strategy:
    raise ValueError('Distribution strategy has not been specified.')

  if not custom_callbacks:
    custom_callbacks = []

  if FLAGS.log_steps:
    custom_callbacks.append(
        keras_utils.TimeHistory(
            batch_size=FLAGS.train_batch_size,
            log_steps=FLAGS.log_steps,
            logdir=FLAGS.model_dir))

  trained_model, _ = run_bert_classifier(
      strategy,
      model_config,
      FLAGS.model_dir,
      epochs,
      steps_per_epoch,
      FLAGS.steps_per_loop,
      eval_steps,
      warmup_steps,
      FLAGS.learning_rate,
      init_checkpoint or FLAGS.init_checkpoint,
      train_input_fn,
      eval_input_fn,
      meta_datas['num_classes'],
      custom_callbacks=custom_callbacks,
      custom_metrics=custom_metrics)

  if FLAGS.model_export_path:
    model_saving_utils.export_bert_model(
        FLAGS.model_export_path, model=trained_model)
  return trained_model


def str_to_int(exlist):
  result = []
  for i in exlist:
    result.append(int(i))
  return result

def main(_) :

    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)

    FLAGS.model_dir = 'C:\\Users\\klplab\\Desktop\\ubert\\express_span_up_down_mtl_weight2_df\\'
    FLAGS.distribution_strategy = 'one_device'

    if not FLAGS.model_dir:
        logging.error("Not define model_dir")
        return

    bert_config = bert_configs.BertConfig.from_json_file(FLAGS.bert_config_file)

    meta_datas = None
    # meta data read 
    with tf.io.gfile.GFile(FLAGS.meta_data_path, 'rb') as reader:
        meta_datas = json.loads(reader.read().decode('utf-8'))

    if FLAGS.mode == 'export_only':
        export_classifier(FLAGS.model_export_path, meta_datas['num_classes'], bert_config,
                        FLAGS.model_dir)
        return

    strategy = distribute_utils.get_distribution_strategy(
        distribution_strategy=FLAGS.distribution_strategy,
        num_gpus=FLAGS.num_gpus,
        tpu_address=FLAGS.tpu)

    eval_input_fn = get_dataset_fn(
        FLAGS.eval_data_path,
        FLAGS.max_seq_length,
        FLAGS.eval_batch_size,
        num_samples=meta_datas['num_classes'],
        is_training=False,)
    
    # eval의 경우 meta에 없음
    eval_len = sum(1 for _ in tf.data.TFRecordDataset(FLAGS.eval_data_path))

    if FLAGS.mode == 'predict':
        with strategy.scope():
            classifier_model = bert_models.classifier_token_model(
                bert_config=bert_config, num_labels=meta_datas['num_classes'], max_seq_length=FLAGS.max_seq_length)[0]
            checkpoint = tf.train.Checkpoint(model=classifier_model)
            latest_checkpoint_file = (
                FLAGS.predict_checkpoint_path or
                tf.train.latest_checkpoint(FLAGS.model_dir))
            assert latest_checkpoint_file
            logging.info('Checkpoint file %s found and restoring from '
                        'checkpoint', latest_checkpoint_file)
            checkpoint.restore(
                latest_checkpoint_file).assert_existing_objects_matched()
            preds_s, preds_e, preds_d, _, _, _ = get_predictions_and_labels(
                strategy,
                classifier_model,
                eval_input_fn,
                is_regression=(meta_datas['num_classes'] == 1),
                return_probs=True)

        eval_datas = create_span_ud_nv_finetuning_data.read_Seq2Seq(input_file=FLAGS.eval_file_path, 
                        read_function=create_span_ud_nv_finetuning_data.read_tsv())
        tokenizer = tokenization_morp_utagger_ver.KoreanMorpsTokenizer(
                        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

        output_predict_file = os.path.join(FLAGS.model_dir, 'test_results.json')
        with tf.io.gfile.GFile(output_predict_file, 'w') as writer:
            logging.info('***** Predict results *****')
            # 결과를 저장하는 부분
            save_ner_result = dict()

            labels_dic = create_span_ud_nv_finetuning_data.get_labels()
            # reverse_dic
            v_k_labels_dic = dict()


            for key, value in labels_dic.items() :
              v_k_labels_dic[value] = key

            def convert_logits_to_labels(logits,label=True) :
              result_labels = []
              # softmax_result = tf.nn.softmax(logits) 
              if label == True:
                for logit in logits:
                  best_index = sorted(enumerate(logit), key=lambda x:x[1], reverse=True)[0]
                  result_labels.append(best_index[0])
              else:
                best_index = sorted(enumerate(logits), key=lambda x:x[1], reverse=True)[0]
                result_labels.append(best_index[0])

              return result_labels

            answer_tags = []
            result_tags = []
            # mlp 부분 고쳐야 함

            label_list =['O','PDT','MOV','TRV']
            tag_list = ['PDT','MOV','TRV','NONE']
            id2label = {i: label for i, label in enumerate(label_list)}
            metric = SpanEntityScore(id2label)

            for idx, (probabilities_s, probabilities_e, probabilities_d) in enumerate(zip(preds_s,preds_e)):
              result_labels_start = convert_logits_to_labels(probabilities_s)
              result_labels_end = convert_logits_to_labels(probabilities_e)
              result_labels_domain = probabilities_d.index(max(probabilities_d))
              eval_data = eval_datas[idx]
              eval_tokens, ej_nums, morp_nums = tokenizer.tokenize_and_ej_morp_number(eval_data.text)
              eval_len = len(eval_data.text.split())

              # morp_nums 를 통해 정답 축소, 정답의 처음과 끝은 제거 CLS, SEP 토큰임
              fixed_answer_start, fixed_answer_end = dict(), dict()
              result_labels_start = result_labels_start[1: eval_len + 1]
              result_labels_end = result_labels_end[1: eval_len + 1]

              start = str_to_int(eval_data.starts)
              end = str_to_int(eval_data.ends)

              R = bert_extract_item(result_labels_start, result_labels_end)
              T = bert_extract_item(start, end)

              save_ner_result[idx] = {'tag_text': eval_data.text,
                                      'answer' : str(T),
                                      'pred': str(R),
                                      'answer_tag': eval_data.domain,
                                      'pred_tag': tag_list[result_labels_domain]}
              
              if tag_list[result_labels_domain] == eval_data.domain:
                domain_score+=1

              metric.update(true_subject=T, pred_subject=R)

            json_dump = json.dumps(save_ner_result, indent=4, ensure_ascii=False, sort_keys=False)
            writer.write(json_dump)
            writer.close()
        
            
            logging.info("***** Calculation accuracy ****** %s" , str((domain_score/idx)*100))
            logging.info("***** Calculation F1 score ******")

            # with open(FLAGS.model_dir + '\\f1-score.json', 'w') as w_f1 :
            #   w_f1.write(metric.update(true_subject=T, pred_subject=R))
            #   w_f1.close()
            print("\n")
            eval_info, entity_info = metric.result()

            results = {f'{key}': value for key, value in eval_info.items()}
            print("***** Eval results *****")
            info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
            print(info)
            print("***** Entity results *****")
            for key in sorted(entity_info.keys()):
                print("******* %s results ********" % key)
                info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
                print(info)
        return

    if FLAGS.mode != 'train_and_eval':
        raise ValueError('Unsupported mode is specified: %s' % FLAGS.mode)

    train_input_fn = get_dataset_fn(
        FLAGS.train_data_path,
        FLAGS.max_seq_length,
        FLAGS.train_batch_size,
        is_training=True,
        label_type=tf.int64,
        num_samples=FLAGS.train_data_size)
    run_bert(
        strategy,
        bert_config,
        eval_len,
        meta_datas,
        train_input_fn,
        eval_input_fn,
        )

if __name__ == '__main__' :
    app.run(main) 