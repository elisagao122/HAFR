from __future__ import absolute_import
from __future__ import division
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import math
import numpy as np
import tensorflow as tf
from multiprocessing import Pool
from multiprocessing import cpu_count
import argparse
import logging
from time import time
from time import strftime
from time import localtime
from Dataset import Dataset
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
_user_input = None
_item_input_pos = None
_batch_size = None
_index = None
_model = None
_sess = None
_dataset = None
_eval_phase = None
_K = None
_feed_dict = None


def parse_args():
    parser = argparse.ArgumentParser(description="Run HAFR.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='data',
                        help='Choose a dataset.')
    parser.add_argument('--val_verbose', type=int, default=10,
                        help='Evaluate per X epochs for validation set.')
    parser.add_argument('--test_verbose', type=int, default=10,
                        help='Evaluate per X epochs for test set.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--dns', type=int, default=1,
                        help='number of negative sample for each positive in dns.')
    parser.add_argument('--reg', type=float, default=0.1,
                        help='Regularization for user and item embeddings.')
    parser.add_argument('--reg_image', type=float, default=0.01,
                        help='Regularization for image embeddings.')
    parser.add_argument('--reg_w', type=float, default=1,
                        help='Regularization for mlp w.')
    parser.add_argument('--reg_h', type=float, default=1,
                        help='Regularization for mlp h.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--pretrain', type=int, default=1,
                        help='Use the pretraining weights or not')
    parser.add_argument('--ckpt', type=int, default=10,
                        help='Save the model per X epochs.')
    parser.add_argument('--weight_size', type=int, default=64,
                        help='weight_size')
    parser.add_argument('--save_folder', nargs="?", default='save',
                    help='Choose a save folder in pretrain')
    parser.add_argument('--restore_folder', nargs="?", default='restore',
                    help='Choose a restore folder in pretrain')
    return parser.parse_args()


# data sampling and shuffling
def sampling(dataset):
    _user_input, _item_input_pos, _ingre_input_pos, _ingre_num_pos, _image_input_pos = [], [], [], [], []
    for (u, i) in dataset.trainMatrix.keys():
        _user_input.append(u)
        _item_input_pos.append(i)
	_ingre_input_pos.append(dataset.ingreCodeDict[i])
	_ingre_num_pos.append(dataset.ingreNum[i])
	_image_input_pos.append(dataset.embImage[i])
    return _user_input, _item_input_pos, _ingre_input_pos, _ingre_num_pos,  _image_input_pos


def shuffle(samples, batch_size, dataset, model):
    global _user_input
    global _item_input_pos
    global _ingre_input_pos
    global _ingre_num_pos
    global _image_input_pos
    global _batch_size
    global _index
    global _model
    global _dataset
    global _num_batch
    _user_input, _item_input_pos, _ingre_input_pos, _ingre_num_pos,  _image_input_pos = samples
    _batch_size = batch_size
    _index = range(len(_user_input))
    _model = model
    _dataset = dataset
    num_batch = len(_user_input) // _batch_size
    _num_batch = num_batch
    np.random.shuffle(_index)
    neg_index = []
    for p in range(2):
        for i in range(num_batch):
            user_batch, item_batch, ingre_batch, ingre_num_batch, image_batch = [], [], [], [], []
            user_neg_batch, item_neg_batch, ingre_neg_batch, ingre_num_neg_batch, image_neg_batch = [], [], [], [], []
            begin = i * _batch_size
            for idx in range(begin, begin + _batch_size):
                user_batch.append(_user_input[_index[idx]])
                item_batch.append(_item_input_pos[_index[idx]])
                ingre_batch.append(_ingre_input_pos[_index[idx]])
                ingre_num_batch.append(_ingre_num_pos[_index[idx]])
                image_batch.append(_image_input_pos[_index[idx]])
                for dns in range(_model.dns):
                    user = _user_input[_index[idx]]
                    user_neg_batch.append(user)
		    gtItem_list = _dataset.validTestRatings[user]
                    if p == 0:
                        j = np.random.randint(_dataset.num_items)
                        while j in _dataset.trainList[_user_input[_index[idx]]] or j in gtItem_list:
                                j = np.random.randint(_dataset.num_items)
                        neg_index.append(j)
                    else:
                        j = neg_index[idx]
                    item_neg_batch.append(j)
                    ingre_neg_batch.append(_dataset.ingreCodeDict[j])
                    ingre_num_neg_batch.append(_dataset.ingreNum[j])
                    image_neg_batch.append(_dataset.embImage[j])
            yield  np.array(user_batch)[:, None], np.array(item_batch)[:, None], ingre_batch, np.array(ingre_num_batch)[:, None], image_batch, \
                   np.array(user_neg_batch)[:, None], np.array(item_neg_batch)[:, None], ingre_neg_batch, np.array(ingre_num_neg_batch)[:, None], image_neg_batch


# prediction model
class HAFR:
    def __init__(self, num_users, num_items, num_cold, num_ingredients, image_size, args):
        self.num_items = num_items
	self.num_cold = num_cold
        self.num_users = num_users
        self.embedding_size = args.embed_size
        self.learning_rate = args.lr
        self.reg = args.reg
        self.dns = args.dns
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.num_ingredients = num_ingredients
	self.image_size = image_size
	self.reg_image = args.reg_image
	self.reg_w = args.reg_w
	self.reg_h = args.reg_h
	self.weight_size = args.weight_size

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
            self.item_input_pos = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_pos")
            self.ingre_input_pos = tf.placeholder(tf.int32, shape=[None, None], name="ingre_input_pos")
            self.ingre_num_pos = tf.placeholder(tf.int32, shape=[None, 1], name="ingre_num_pos")
            self.image_input_pos = tf.placeholder(tf.float32, shape=[None, self.image_size], name="image_input_pos")
            self.item_input_neg = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_neg")
            self.ingre_input_neg = tf.placeholder(tf.int32, shape=[None, None], name="ingre_input_neg")
            self.ingre_num_neg = tf.placeholder(tf.int32, shape=[None, 1], name="ingre_num_neg")
            self.image_input_neg = tf.placeholder(tf.float32, shape=[None, self.image_size], name="image_input_neg")
	    self.keep_prob = tf.placeholder(tf.float32, name="dropout_keep")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")

    def _create_variables(self):
        with tf.name_scope("embedding"):
	    self.embedding_P = tf.Variable(
                tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                name='embedding_P', dtype=tf.float32, trainable=True)  # (users, embedding_size)
	    self.embedding_Q1 = tf.Variable(
                tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                name='embedding_Q1', dtype=tf.float32)  # (items, embedding_size)
            self.embedding_Q2 = tf.Variable(
                tf.zeros(shape=[self.num_cold, self.embedding_size]),
                name='embedding_Q2', dtype=tf.float32, trainable=False)  # (items, embedding_size)
            self.embedding_Q = tf.concat([self.embedding_Q1, self.embedding_Q2], 0, name='embedding_Q')

            self.c1 = tf.Variable(tf.truncated_normal(shape=[self.num_ingredients, self.embedding_size], mean=0.0, stddev=0.01), name='embedding_ingredient_c1', dtype=tf.float32, trainable=True)
            self.c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2')
            self.embedding_ingredient = tf.concat([self.c1, self.c2], 0, name='embedding_ingredient') 

                ####image maghts_pathp weights and bias
            self.W_image = tf.Variable(tf.truncated_normal(shape=[self.image_size, self.embedding_size], mean=0.0, stddev=tf.sqrt(tf.div(2.0, self.image_size + self.embedding_size))),name='Weights_for_image_map', dtype=tf.float32, trainable=True)
            self.b_image = tf.Variable(tf.truncated_normal(shape=[1, self.embedding_size], mean=0.0, stddev=tf.sqrt(tf.div(2.0, self.image_size + self.embedding_size))),name='Bias_for_image_map', dtype=tf.float32, trainable=True)

            self.W_concat = tf.Variable(tf.truncated_normal(shape=[self.embedding_size*3, self.embedding_size], mean=0.0, stddev=tf.sqrt(tf.div(2.0, 4*self.embedding_size))),name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            self.b_concat = tf.Variable(tf.truncated_normal(shape=[1, self.embedding_size], mean=0.0, stddev=tf.sqrt(tf.div(2.0, 2*self.embedding_size))),name='Bias_for_MLP', dtype=tf.float32, trainable=True)

            self.h = tf.Variable(tf.truncated_normal(shape=[self.embedding_size,1], mean=0.0, stddev=tf.sqrt(tf.div(2.0, 2*self.embedding_size))),name='h_for_MLP', dtype=tf.float32, trainable=True)

	    self.W_att_ingre = tf.Variable(tf.truncated_normal(shape=[self.embedding_size*3, self.weight_size], mean=0.0, stddev=tf.sqrt(tf.div(2.0, 3*self.embedding_size+self.weight_size))),name='Weights_for_ingre_attMLP', dtype=tf.float32, trainable=True)
            self.b_att_ingre = tf.Variable(tf.truncated_normal(shape=[1, self.weight_size], mean=0.0, stddev=tf.sqrt(tf.div(2.0, self.weight_size + self.embedding_size))),name='Bias_for_ingre_attMLP', dtype=tf.float32, trainable=True)

            self.v = tf.Variable(tf.ones([self.weight_size, 1]), name='v_for_MLP', dtype=tf.float32, trainable=True)

	    
	    self.W_att_com = tf.Variable(tf.truncated_normal(shape=[self.embedding_size*2, self.weight_size], mean=0.0, stddev=tf.sqrt(tf.div(2.0, 2*self.embedding_size+self.weight_size))),name='Weights_for_com_attMLP', dtype=tf.float32, trainable=True)
            self.b_att_com = tf.Variable(tf.truncated_normal(shape=[1, self.weight_size], mean=0.0, stddev=tf.sqrt(tf.div(2.0, self.weight_size + self.embedding_size))),name='Bias_for_com_attMLP', dtype=tf.float32, trainable=True)

            self.v_c = tf.Variable(tf.ones([self.weight_size, 1]), name='v_com_att_for_MLP', dtype=tf.float32, trainable=True)
	   

    def _attention_ingredient_level(self, q_, embedding_p, image_embed, item_ingre_num):
        with tf.name_scope("attention_ingredient_level"):
		b = tf.shape(q_)[0]
		n = tf.shape(q_)[1]
		expand_p = tf.expand_dims(embedding_p, 1)
		tile_p = tf.tile(expand_p, [1,n,1]) # (b,n,e)
		expand_image = tf.expand_dims(image_embed, 1)
		tile_image = tf.tile(expand_image, [1,n,1]) # (b,n,e)
		concat_v = tf.concat([q_, tile_p, tile_image],2) # (b,n,3e)
		MLP_output = tf.matmul(tf.reshape(concat_v, [b*n,-1]), self.W_att_ingre) + self.b_att_ingre
		MLP_output = tf.nn.tanh( MLP_output )
		A_ = tf.reshape(tf.matmul(MLP_output, self.v),[b,n])
		smooth = -1e12
                self.num_idx = tf.reduce_sum(item_ingre_num ,1)
                mask_mat = tf.sequence_mask(self.num_idx, maxlen = n, dtype = tf.float32) # (b, n) 
                mask_mat = tf.ones_like(mask_mat) - mask_mat
                self.m = mask_mat * smooth
                self.A = tf.nn.softmax(A_ + self.m)
                self.A = tf.expand_dims(self.A, 2) #(b,n,1)
                self.A_sum = tf.reduce_sum(self.A,1)

		return tf.reduce_sum(self.A * q_, 1) # (b,n,e) -->(b,e)


    def _attention_id_ingre_image(self, embedding_p,  embedding_q, embedding_ingre_att, image_embed):
	b = tf.shape(embedding_p)[0]
	cp1 = tf.concat([embedding_p, embedding_q], 1)
	cp2 = tf.concat([embedding_p, embedding_ingre_att], 1)
	cp3 = tf.concat([embedding_p, image_embed], 1)
	cp = tf.concat([cp1,cp2,cp3], 0)
	c_hidden_output = tf.matmul(cp, self.W_att_com) + self.b_att_com
	c_hidden_output = tf.nn.tanh(c_hidden_output)
	c_mlp_output = tf.reshape(tf.matmul(c_hidden_output, self.v_c), [b,-1])
	B = tf.nn.softmax(c_mlp_output)
	self.B = tf.expand_dims(B, 2) # (b,3,1)
	self.ce1 = tf.expand_dims(embedding_q, 1) # (b,1,e)
	self.ce2 = tf.expand_dims(embedding_ingre_att, 1) # (b,1,e)
	self.ce3 = tf.expand_dims(image_embed, 1)
	self.ce = tf.concat([self.ce1, self.ce2, self.ce3], 1) #(b,3,e)
	return tf.reduce_sum(self.B * self.ce, 1) #(b,e)

    def _attention_user(self, embedding_p):
        return embedding_p

    def _create_inference(self, item_input, ingre_input, image_input, ingre_num):
        with tf.name_scope("inference"):
            self.embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
            self.embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, item_input), 1)  
            self.embedding_ingre = tf.nn.embedding_lookup(self.embedding_ingredient, ingre_input) 
            self.image_embed = tf.matmul(image_input, self.W_image) + self.b_image

            self.embedding_ingre_att = self._attention_ingredient_level(self.embedding_ingre, self.embedding_p, self.image_embed, ingre_num)
            self.item_att_fin = self._attention_id_ingre_image(self.embedding_p, self.embedding_q, self.embedding_ingre_att, self.image_embed)
            self.embedding_p_att = self._attention_user(self.embedding_p)

            self.user_item_concat = tf.concat([self.embedding_p_att, self.item_att_fin, self.embedding_p_att*self.item_att_fin], 1)
            self.hidden_input = tf.matmul(self.user_item_concat, self.W_concat) + self.b_concat
            MLP_output = tf.nn.dropout(self.hidden_input, self.keep_prob)
	    MLP_output = tf.nn.relu( MLP_output )

            return tf.matmul(MLP_output, self.h), self.embedding_p, self.embedding_q, self.embedding_ingre  # (b, embedding_size) * (embedding_size, 1)



    def _create_loss(self):
        with tf.name_scope("loss"):
	    self.output,self.embed_pos_p, self.embed_pos_q, self.embed_pos_ingre = self._create_inference(self.item_input_pos, self.ingre_input_pos, self.image_input_pos, self.ingre_num_pos)
            self.output_neg, self.embed_neg_p, self.embed_neg_q, self.embed_neg_ingre = self._create_inference(self.item_input_neg, self.ingre_input_neg, self.image_input_neg, self.ingre_num_neg)
            self.result = self.output - self.output_neg
            self.loss = tf.reduce_sum(tf.nn.softplus(-self.result))

            # loss to be omptimized
	    self.opt_loss = self.loss + self.reg * (
                        tf.reduce_sum(tf.square(self.embed_pos_p)) + tf.reduce_sum(tf.square(self.embed_pos_q)) + tf.reduce_sum(tf.square(self.embed_pos_ingre)) + \
                        tf.reduce_sum(tf.square(self.embed_neg_q)) + tf.reduce_sum(tf.square(self.embed_neg_ingre))) + self.reg_image * (
			tf.reduce_sum(tf.square(self.W_image))) + self.reg_w * (tf.reduce_sum(tf.square(self.W_concat))) + self.reg_h * (tf.reduce_sum(tf.square(self.h)))

    def _create_optimizer(self):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.opt_loss)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()


# training
def training(model, dataset, args, saver=None):  # saver is an object to save pq
    with tf.Session() as sess:
	ckpt_save_path = "Pretrain/embed_%d/%s/" % (args.embed_size, args.save_folder)
        ckpt_restore_path = "Pretrain/embed_64/7-d41/"
	
        if not os.path.exists(ckpt_save_path):
            os.makedirs(ckpt_save_path)
        if not os.path.exists(ckpt_restore_path):
            os.makedirs(ckpt_restore_path)

	saver_ckpt = tf.train.Saver({'embedding_P': model.embedding_P, 'embedding_Q1': model.embedding_Q1, 'embedding_Q2': model.embedding_Q2, 'c1':model.c1, \
                                         'W_image': model.W_image, 'b_image': model.b_image, \
                                         'W_concat': model.W_concat, 'b_concat': model.b_concat, \
                                         'h': model.h})

        sess.run(tf.global_variables_initializer())

        if args.pretrain:
            ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckpt_restore_path + 'checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver_ckpt.restore(sess, ckpt.model_checkpoint_path)
                logging.info("Using pretrained variables")
                print "Using pretrained variables"
	    else:
		print "Wrong in pretrained variables loading"
        else:
            logging.info("Initialized from scratch")
            print "Initialized from scratch"

	saver_ckpt_com = tf.train.Saver({'embedding_P': model.embedding_P, 'embedding_Q1': model.embedding_Q1, 'embedding_Q2': model.embedding_Q2, 'c1':model.c1, \
                                         'W_image': model.W_image, 'b_image': model.b_image, \
                                         'W_concat': model.W_concat, 'b_concat': model.b_concat, \
                                         'h': model.h, 'v':model.v, \
					 'W_att_ingre': model.W_att_ingre, 'b_att_ingre': model.b_att_ingre, \
					 'W_att_com': model.W_att_com, 'b_att_com': model.b_att_com, 'v_c': model.v_c}, max_to_keep = 30)
        # sample the data
        samples = sampling(dataset)

        # initialize the max_ndcg to memorize the best result
        max_auc = 0
        best_res = {}

	max_valid_auc = 0
	best_valid_res = {}
	best_model = None

        # train by epoch
        for epoch_count in range(1, 1 + args.epochs):
	    print "Epoch %d begins in %s" % (epoch_count, datetime.datetime.now())
            # initialize for training batches
            batch_begin = time()
            batches = shuffle(samples, args.batch_size, dataset, model)
            batch_time = time() - batch_begin
            pre_loss, pre_obj, prev_acc = training_loss_acc(model, sess, batches)

            # training the model
	    loss_begin = time()
            train_loss, train_obj, post_acc, train_time = training_batch(model, sess, batches, epoch_count, batch_time, pre_loss, pre_obj, prev_acc)
            print "train_time = %.1fs" % train_time
    	    loss_time = time() - loss_begin

	    if epoch_count % args.val_verbose == 0:
                valid_feed_dicts = init_eval_model(model, dataset, "Valid")
                auc, cur_res, valid_time = output_evaluate(model, sess, dataset, valid_feed_dicts, epoch_count, "Valid")
                # print and log the best result for validation set
                if max_valid_auc < auc:
                        max_valid_auc = auc
                        best_valid_res['result'] = cur_res
                        best_valid_res['epoch'] = epoch_count
			best_model = model
			best_epoch = epoch_count
	    

            if model.epochs == epoch_count:
                print "Epoch %d is the best epoch for validation set " % best_valid_res['epoch']
		
                test_feed_dicts = init_eval_model(best_model, dataset, "Test")
                print "test starting in %s" % datetime.datetime.now()
                auc, cur_res, test_time = output_evaluate(best_model, sess, dataset, test_feed_dicts, best_epoch, "Test")
                print "test ending in %s" % datetime.datetime.now()
                max_auc = auc
                best_res['result'] = cur_res
                best_res['epoch'] = best_epoch
		
                print "For epoch %d [%.1fs], the result for test set is " % (best_res['epoch'], test_time)
                for idx, (recall_k, ndcg_k, auc_k) in enumerate(np.swapaxes(best_res['result'], 0, 1)):
                    res = "K = %d: Recall = %.4f, NDCG = %.4f, AUC = %.4f" % ((idx + 1)*10, recall_k, ndcg_k, auc_k)
                    print res

            # save the embedding weights
            if args.ckpt > 0 and epoch_count % args.ckpt == 0:
                saver_ckpt_com.save(sess, ckpt_save_path + 'weights', global_step=epoch_count)


def output_evaluate(model, sess, dataset, eval_feed_dicts, epoch_count, eval_phase):
    eval_begin = time()
    result = evaluate(model, sess, dataset, eval_feed_dicts, eval_phase)
    eval_time = time() - eval_begin

    recall, ndcg, auc = np.swapaxes(result, 0, 1)[-1]
    return auc, result, eval_time



# input: model, sess, batches
# output: training_loss
def training_loss_acc(model, sess, train_batches):
    train_loss = 0.0
    train_obj = 0.0
    acc = 0
    idx = 0
    for train_batch in train_batches:
	idx += 1
    	user_input, item_input_pos, ingre_input_pos, ingre_num_pos, image_input_pos, user_input_neg, item_input_neg, ingre_input_neg, ingre_num_neg, image_input_neg = list(map(list, zip(*zip(*train_batch))))
        feed_dict = {model.user_input: user_input,
                     model.item_input_pos: item_input_pos,
                     model.ingre_input_pos: ingre_input_pos,
                     model.ingre_num_pos: ingre_num_pos,
                     model.image_input_pos: image_input_pos,
                     model.item_input_neg: item_input_neg,
                     model.ingre_input_neg: ingre_input_neg,
                     model.ingre_num_neg: ingre_num_neg,
                     model.image_input_neg: image_input_neg,
                     model.keep_prob: 0.5,
                     model.train_phase: True}
        loss, obj, output_pos, output_neg = sess.run([model.loss, model.opt_loss, model.output, model.output_neg], feed_dict)
        train_loss += loss
        train_obj += obj
        acc += ((output_pos - output_neg) > 0).sum() / len(output_pos)
	if idx == _num_batch:
		break
    num_batch = _num_batch 
    return train_loss / num_batch, train_obj / num_batch, acc / num_batch

def training_batch(model, sess, train_batches, epoch_count, batch_time, pre_loss, pre_obj, prev_acc):
    train_loss = 0.0
    train_obj = 0.0
    acc = 0
    train_begin = time()

    idx = 0
    tmp_feed = None
    for train_batch in train_batches:
	idx += 1
    	user_input, item_input_pos, ingre_input_pos, ingre_num_pos, image_input_pos, user_input_neg, item_input_neg, ingre_input_neg, ingre_num_neg, image_input_neg = list(map(list, zip(*zip(*train_batch))))
        feed_dict = {model.user_input: user_input,
                     model.item_input_pos: item_input_pos,
                     model.ingre_input_pos: ingre_input_pos,
                     model.ingre_num_pos: ingre_num_pos,
                     model.image_input_pos: image_input_pos,
                     model.item_input_neg: item_input_neg,
                     model.ingre_input_neg: ingre_input_neg,
                     model.ingre_num_neg: ingre_num_neg,
                     model.image_input_neg: image_input_neg,
                     model.keep_prob: 0.5,
                     model.train_phase: True}
        _, loss, obj, output_pos, output_neg = sess.run([model.optimizer, model.loss, model.opt_loss, model.output, model.output_neg], feed_dict)
        train_loss += loss
        train_obj += obj
        acc += ((output_pos - output_neg) > 0).sum() / len(output_pos)
	if idx == _num_batch:
		tmp_feed = feed_dict
    feed_dict = tmp_feed
    train_time = time() - train_begin
    num_batch = idx
    train_loss, train_obj, post_acc = train_loss / num_batch, train_obj / num_batch, acc / num_batch

    #print res
    return train_loss, train_obj, post_acc, train_time



def init_eval_model(model, dataset, eval_phase):
    begin_time = time()
    global _dataset
    global _model
    global _eval_phase
    _eval_phase = eval_phase
    _dataset = dataset
    _model = model

    if eval_phase == "Test":
        for user in range(_dataset.num_users):
            # generate items_list
            test_item_list = _dataset.testRatings[user]
            item_input = _dataset.testNegatives[user]
            for test_item in test_item_list:
                if test_item in item_input:
                        item_input.remove(test_item)
            item_input = list(item_input)
            item_input.extend(test_item_list)
            user_input = np.full(len(item_input), user, dtype='int32')[:, None]
            item_input = np.array(item_input)[:, None]
            ingre_input = []
            ingre_num_input = []
            image_input = []
            for item in item_input:
                #print item
                ingre_input.append(_dataset.ingreCodeDict[item[0]])
                ingre_num_input.append(_dataset.ingreNum[item[0]])
                image_input.append(_dataset.embImage[item[0]])
            yield user_input, item_input, ingre_input, ingre_num_input, image_input
    else:
        for idx in range(len(_dataset.valid_users)):
            user = _dataset.valid_users[idx]
            test_item_list = _dataset.validRatings[idx]
            item_input = _dataset.validNegatives[idx]
            for test_item in test_item_list:
                if test_item in item_input:
                        item_input.remove(test_item)
            item_input = list(item_input)
            item_input.extend(test_item_list)
            user_input = np.full(len(item_input), user, dtype='int32')[:, None]
            item_input = np.array(item_input)[:, None]
            ingre_input = []
            ingre_num_input = []
            image_input = []
            for item in item_input:
                #print item
                ingre_input.append(_dataset.ingreCodeDict[item[0]])
                ingre_num_input.append(_dataset.ingreNum[item[0]])
                image_input.append(_dataset.embImage[item[0]])
            yield user_input, item_input, ingre_input, ingre_num_input, image_input

def evaluate(model, sess, dataset, feed_dicts, eval_phase):
    global _model
    global _K
    global _sess
    global _dataset
    global _feed_dicts
    global _eval_phase

    _eval_phase = eval_phase
    _dataset = dataset
    _model = model
    _sess = sess
    _K = 50
    _feed_dicts = feed_dicts

    res = []
    user_idx = 0
    for user_batch in _feed_dicts:
        res.append(_eval_by_user(user_idx, user_batch))
        user_idx += 1
    res = np.array(res)
    recall, ndcg, auc = (res.mean(axis=0)).tolist()

    return recall, ndcg, auc


def _eval_by_user(user_idx, user_batch):
    # get prredictions of data in testing set
    user_input, item_input, ingre_input, ingre_num_input, image_input = list(map(list, zip(*zip(*user_batch))))
    ingre_num_input = np.array(ingre_num_input)[:, None]
    feed_dict = {_model.user_input: user_input, _model.item_input_pos: item_input, _model.ingre_input_pos: ingre_input, _model.ingre_num_pos: ingre_num_input,  _model.image_input_pos: image_input, _model.keep_prob: 1, _model.train_phase: False}
    predictions = _sess.run(_model.output, feed_dict)

    if _eval_phase == "Valid":
        gtItem_list = _dataset.validRatings[user_idx]
    else:
        gtItem_list = _dataset.testRatings[user_idx]
    pos_num = len(gtItem_list)

    neg_predict, pos_predict = predictions[:-pos_num], predictions[-pos_num:]
    rel_list = range(len(predictions)-pos_num, len(predictions), 1)

    idx_value_dict = {}
    for pre_idx, value in enumerate(predictions):
        idx_value_dict[pre_idx] = value
    sorted_idx_value_list = sorted(idx_value_dict.items(), key=lambda d: d[1], reverse=True)


    recall, ndcg, auc = [], [], []
    neg_num = 500
    auc_value = get_auc(rel_list, predictions, neg_num)

    for k in range(10,_K+1, 10):
        doc_list = []
        for idx, val in sorted_idx_value_list:
                if len(doc_list) < k:
                        doc_list.append(idx)
                else:
                        break
        assert len(doc_list) == k
        recall_value, ndcg_value = metrics(doc_list, rel_list)
        recall.append(recall_value)
        ndcg.append(ndcg_value)
	auc.append(auc_value)

    return recall, ndcg, auc

def metrics(doc_list, rel_list):
    dcg = 0.0
    hit_num = 0.0

    for i in range(len(doc_list)):
        if doc_list[i] in rel_list:
            dcg += 1/(math.log(i+2) / math.log(2))
            hit_num += 1

    idcg = 0.0
    for i in range(min(len(doc_list), len(rel_list))):
        idcg += 1/(math.log(i+2) / math.log(2))
    ndcg = dcg/ idcg
    recall = hit_num / len(rel_list)
    return recall, ndcg

def get_auc(rel_list, predictions, neg_num):
    auc_value = 0.0
    for rel in rel_list:
        for pre in predictions[0: neg_num]:
                if predictions[rel] > pre:
                        auc_value += 1
    return auc_value / (len(rel_list) * neg_num)


if __name__ == '__main__':
    args = parse_args()
    print("Arguments: %s" % (args))

    # initialize dataset
    dataset = Dataset(args.path + args.dataset)
    print "Has finished processing dataset"

    # initialize models
    model = HAFR(dataset.num_users, dataset.num_items, dataset.cold_num, dataset.num_ingredients, dataset.image_size, args)
    model.build_graph()

    # start training
    training(model, dataset, args)
