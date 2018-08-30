import network
import tensorflow as tf
import argparse
import numpy as np
import os
import os.path as osp
from prep import train_prep, test_prep, read_lines
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.eager import context
import random

def inv_lr_decay(learning_rate, global_step, gamma, power, name=None):
    if global_step is None:
        raise ValueError("global_step is required for inv_decay.")
    with ops.name_scope(name, "InvDecay", \
                      [learning_rate, global_step, gamma, power]) as name:
        learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
        dtype = learning_rate.dtype
        gamma = math_ops.cast(gamma, dtype)
        power = math_ops.cast(power, dtype)

        def decayed_lr(global_step):
            global_step = math_ops.cast(global_step, dtype)
            base = math_ops.multiply(gamma, global_step)
            return math_ops.multiply( \
                    learning_rate, math_ops.pow(1+base, -power), name=name)
        if not context.executing_eagerly():
            decayed_lr = decayed_lr(global_step)
        return decayed_lr

def cdan_model_fn(features, labels, mode, params):
  model_class = params["model"]
  resnet_size = params["resnet_size"]
  num_classes = params["num_classes"]
  weight_decay = params["weight_decay"]
  loss_scale = params["loss_scale"]
  momentum = params["momentum"]
  base_lr = params["base_lr"]
  batch_size = params["batch_size"]

  model = model_class(resnet_size, data_format="channels_last", num_classes=num_classes)

  if mode == tf.estimator.ModeKeys.PREDICT:
    
    logits, hidden_features = model(features, mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.cast(logits, tf.float32)

    predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    # Return the predictions and the specification for serving a SavedModel
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        })
  elif mode == tf.estimator.ModeKeys.TRAIN:
    s_input = features["source"]
    t_input = features["target"]
    s_label = labels
    ad_s_label = features["ad_s_label"]
    ad_t_label = features["ad_t_label"]
    model = model_class(resnet_size, data_format="channels_last", num_classes=num_classes)
    logits, hidden_features = model(tf.concat((s_input, t_input), 0), mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.cast(logits, tf.float32)
    ad_labels = tf.cast(tf.concat((ad_s_label, ad_t_label), 0), tf.float32)
    mid_point = tf.shape(s_input)[0]

    predictions = {
      'classes': tf.argmax(tf.slice(logits, [0, 0], [mid_point, num_classes]), axis=1),
      'probabilities': tf.nn.softmax(tf.slice(logits, [0, 0], [mid_point, num_classes]), name='softmax_tensor')
    }

    global_step = tf.train.get_or_create_global_step()

    ad_net = network.AdversarialNetwork(global_step)
    ad_out = ad_net(hidden_features, mode == tf.estimator.ModeKeys.TRAIN)

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      logits=tf.slice(logits, [0, 0], [mid_point, num_classes]), labels=s_label)
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    adversarial_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=ad_out, labels=ad_labels))
    tf.identity(cross_entropy, name='adversarial_loss')
    tf.summary.scalar('adversarial_loss', adversarial_loss)

    def exclude_batch_norm(name):
        return 'batch_normalization' not in name

    l2_loss = weight_decay * tf.add_n(
      # loss is computed using fp32 for numerical stability.
      [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
       if exclude_batch_norm(v.name)])
    tf.summary.scalar('l2_loss', l2_loss)
    loss = cross_entropy + l2_loss + adversarial_loss

    learning_rate = inv_lr_decay(base_lr, global_step, gamma=0.001, power=0.75)
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=momentum
    )

    def _grad_filter(gvs):
        return [(g, v) for g, v in gvs if not ('dense' in v.name)], [(g, v) for g, v in gvs if 'dense' in v.name]

    if loss_scale != 1:
        scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)
        dense_grad_vars, other_grad_vars = _grad_filter(scaled_grad_vars)
        other_grad_vars = [(grad / loss_scale, var)
                        for grad, var in other_grad_vars]
        dense_grad_vars = [(grad / loss_scale * 10.0, var)
                        for grad, var in dense_grad_vars]     
        minimize_op = optimizer.apply_gradients(dense_grad_vars+other_grad_vars, global_step)
    else:
        grad_vars = optimizer.compute_gradients(loss)
        dense_grad_vars, other_grad_vars = _grad_filter(grad_vars)
        other_grad_vars = [(grad, var)
                        for grad, var in other_grad_vars]
        dense_grad_vars = [(grad * 10.0, var)
                        for grad, var in dense_grad_vars]  
        minimize_op = optimizer.apply_gradients(dense_grad_vars+other_grad_vars, global_step)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)
    accuracy = tf.metrics.accuracy(s_label, predictions['classes'])

    metrics = {'accuracy': accuracy}

    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

  else:
    logits, hidden_features = model(features, mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.cast(logits, tf.float32)

    predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    train_op = None

    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      logits=logits, labels=labels)
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    def exclude_batch_norm(name):
        return 'batch_normalization' not in name

    l2_loss = weight_decay * tf.add_n(
      # loss is computed using fp32 for numerical stability.
      [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
       if exclude_batch_norm(v.name)])
    tf.summary.scalar('l2_loss', l2_loss)
    loss = cross_entropy + l2_loss

    accuracy = tf.metrics.accuracy(labels, predictions['classes'])

    metrics = {'accuracy': accuracy}

    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', help="Options: ResNet18,34,50,101,152; AlexNet")
    parser.add_argument('--dset', type=str, default='office', help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='../../pytorch/data/office/amazon_list.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='../../pytorch/data/office/webcam_list.txt', help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='../snapshot/tf_cdan', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--pretrained_model_checkpoint_path', type=str, default='../snapshot/20180601_resnet_v2_imagenet_checkpoint', help='')
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--high', type=float, default=1.0, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=36, help="learning rate")
    args = parser.parse_args()

    if int(args.gpu_id) < 0:
        sess_conf = tf.ConfigProto(device_count = {"GPU": 0})
        sess_conf.gpu_options.allow_growth = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        sess_conf = tf.ConfigProto()
        sess_conf.gpu_options.allow_growth = True

    os.system('mkdir -p ' + args.output_dir)

    log_file = open(osp.join(args.output_dir, "log_file.txt"), "w")

    s_fnames, s_labels = read_lines(args.s_dset_path)
    t_fnames, t_labels = read_lines(args.t_dset_path)
    test_fnames, test_labels = read_lines(args.t_dset_path)

    #s_iter = s_dset.make_one_shot_iterator()
    #t_iter = t_dset.make_one_shot_iterator()
    #test_iter = test_dset.make_initializable_iterator()

    if len(s_fnames) > len(t_fnames):
        t_n_fnames = t_fnames * (len(s_fnames) // len(t_fnames))
        s_sample = True
    else:
        s_fnames *= len(t_fnames) // len(s_fnames)

    def input_fn_train():
        repeat_num = 20
        t_input_fnames = []
        s_input_fnames = []
        s_input_labels = []
        for j in range(repeat_num):
            if s_sample:
                sample_index = random.sample(range(len(s_fnames)), len(t_n_fnames))
                s_fnames_sample = [s_fnames[i] for i in sample_index]
                s_labels_sample = [s_labels[i] for i in sample_index]
                s_input_fnames += s_fnames_sample
                t_input_fnames += t_n_fnames
                s_input_labels += s_labels_sample 
            else:
                sample_index = random.sample(range(len(t_fnames)), len(s_fnames))
                t_fnames_sample = [t_fnames[i] for i in sample_index]
                s_input_fnames += s_fnames
                t_input_fnames += t_fnames_sample
                s_input_labels += s_labels
           
        features = {"source":s_input_fnames, "target":t_input_fnames, "ad_s_label":[[1]]*len(s_input_fnames), "ad_t_label":[[0]]*len(t_input_fnames)}
        labels = s_input_labels
        return tf.data.Dataset.from_tensor_slices((features, labels)).shuffle(buffer_size=1000).map(train_prep, num_parallel_calls=4).batch(args.batch_size)

    def input_fn_test():
        return tf.data.Dataset.from_tensor_slices((t_fnames, t_labels)).map(test_prep, num_parallel_calls=4).batch(4).make_one_shot_iterator().get_next()


    #sess = tf.Session(config=sess_conf)
    #features, labels = sess.run(input_fn_train().make_one_shot_iterator().get_next())

    run_config = tf.estimator.RunConfig(session_config=sess_conf)

    warm_start_settings = tf.estimator.WarmStartSettings(
        args.pretrained_model_checkpoint_path,
        vars_to_warm_start='^(?!.*dense)')
    classifier = tf.estimator.Estimator(
        model_fn=cdan_model_fn, model_dir=args.output_dir, config=run_config,
        warm_start_from=warm_start_settings, params={
        'model':network.ResNetModel,
        'resnet_size':50,
        "weight_decay":0.0005,
        'num_classes':31,
        'loss_scale':1,
        'momentum':0.9,
        'base_lr':0.001,
        'batch_size':args.batch_size
        })

    for epochs in range(100):
        print("epochs {:d}".format(epochs))
        classifier.train(input_fn=input_fn_train, steps=500)
        print("caozhangjie start test")
        eval_results = classifier.evaluate(input_fn=input_fn_test)
        print(eval_results)
    #graph_def = tf.saved_model.loader.load(sess, ["serve"], "../snapshot/20180601_resnet_v2_imagenet_savedmodel/1527887769/")
    #_ = tf.import_graph_def(graph_def, name='')
