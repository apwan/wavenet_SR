#!/usr/bin/env python3
import sys, os
from typing import List
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import logging
logging.set_verbosity(logging.ERROR)

# TODO: remove dependency on sugartensor
import sugartensor as sgtf
from functools import lru_cache

# temparory fix for tensorflow < 1.5
if not hasattr(tf, 'AUTO_REUSE'):
	setattr(tf, 'AUTO_REUSE', False)
import numpy as np
import librosa

from utils import (abs_dir, files_by_suffixes,
	en_dict, en_phon, en_phnset,
	dict2obj)

from pathlib import Path

default_ckpt_dirs = dict(
	wavenet = abs_dir('./train/'),
	adapt_timit = abs_dir('./adapt_timit/')
)




'''
Vocabulary
'''

def voca_utils(index2label: List[str]):
	# vocabulary size
	voca_size = len(index2label)
	label2index = {index2label[i]: i for i in range(voca_size)}
	
	def index2str(index_list: List[int], sep: str=',') -> str: 
		# transform label index to character
		str_ = sep.join([index2label[ch] for ch in index_list if ch > 0])
		return str_


	# convert sentence to index list, no use in recognition
	def str2index(str_: str) -> List[int]:
		# clean long white spaces
		str_ = ' '.join(str_.split())
		# remove punctuation and make lower case
		str_ = str_.translate(None, string.punctuation).lower()
		res = [label2index[ch] for ch in str_ if ch in label2index] # drop OOV
		
		return res	

	def print_index(indices: List[List[int]], sep: str=','):
		print(indices)
		ret = [index2str(index_list, sep) for index_list in indices]
		print(ret)
		return ret	

	return voca_size, index2str, str2index, print_index


'''
Session management
'''

sess = None
loaded_models = {
	'wavenet': None,
	'adapt_timit': None,
	'adapt_timit128_41': None
}

def release_session():
	global sess
	if sess is not None:
		for i in loaded_models:
			loaded_models[i] = None
		sess.close()
		sess = None
		

def init_session():
	global sess
	if sess is None:
		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.3
		sess = tf.Session(config=config)
		# init variables
		sess.run(tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer()))


def load_model(ckpt_dir, recover_dict=None):
	init_session()
	# restore parameters
	if recover_dict is None:
		saver = tf.train.Saver()
	else:
		saver = tf.train.Saver(recover_dict)

	print(f'load model from ckpt_dir: {ckpt_dir}')
	saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))


def default_recover_list(name) -> list:
	recover_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
	return recover_list
	#return {v.op.name:v for v in recover_list}


def strip_recover_list(name, recover_list: list=None) -> dict:
	if recover_list is None:
		recover_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
	# temperary hack
	return {v.op.name[len(name)+1:].replace('mean_3', 'mean'):v for v in recover_list}


'''
Reusabel components
'''	


def build_conv1d_s1(input_dim, output_dim, batch_size, name, softmax=False):
	x = tf.placeholder(dtype=sgtf.sg_floatx, shape=(batch_size, None, input_dim))
	if softmax:
		logit = tf.nn.softmax(x)
	else:   
		logit = x
	logit = sgtf.sg_layer.sg_conv1d(logit, size=1, dim=output_dim, name=name)
	return x, logit


# currently no use
def wavenet_block_scopes(n, r):
	return ["block_{i}_{2**j}/" for i in range(n) for j in range(r)]		



# residual block
def res_block(tensor, size, rate, block, dim):

	with sgtf.sg_context(name=f'block_{block}_{rate}', reuse=sgtf.AUTO_REUSE):

		# filter convolution
		conv_filter = tensor.sg_aconv1d(size=size, rate=rate, act='tanh', bn=True, name='conv_filter')
		# gate convolution
		conv_gate = tensor.sg_aconv1d(size=size, rate=rate,  act='sigmoid', bn=True, name='conv_gate')

		# output by gate multiplying
		out = conv_filter * conv_gate

		# final output
		out = out.sg_conv1d(size=1, dim=dim, act='tanh', bn=True, name='conv_out')

		# residual and skip output
		return out + tensor, out


#
# logit calculating graph using atrous convolution
#
def get_logit(x, voca_size, num_blocks, num_block_layers, num_conv, num_dim):

	# expand dimension
	with sgtf.sg_context(name='front', reuse=tf.AUTO_REUSE):
		z = x.sg_conv1d(size=1, dim=num_dim, act='tanh', bn=True, name='conv_in')

	# dilated conv block loop
	skip = 0  # skip connections
	for i in range(num_blocks):
		for j in range(num_block_layers):
			r = 2 ** j
			z, s = res_block(z, size=num_conv, rate=r, block=i, dim=num_dim)
			skip += s

	logit, in_layer = None, None	

	# final logit layers
	with sgtf.sg_context(name='logit', reuse=tf.AUTO_REUSE):
		in_layer = skip.sg_conv1d(size=1, act='tanh', bn=True, name='conv_1')
		logit = in_layer.sg_conv1d(size=1, dim=voca_size, name='conv_2')

	return logit, in_layer



def build_decoder(logit, seq_len):
	# ctc decoding
	decoded, _ = tf.nn.ctc_beam_search_decoder(logit.sg_transpose(perm=[1, 0, 2]), seq_len, merge_repeated=False)
	# to dense tensor
	top = decoded[0]
	y = sgtf.sparse_to_dense(top.indices, top.dense_shape, top.values) + 1 # skip <EOS>

	return y, decoded



'''
Graph Builders
return tuple of sgtf.Variables , and
	 recover_list (no use in training) --> please use sgtf.sg_train
'''

def build_adapt_timit(name, batch_size, x, voca_size, **kwargs):
	
	with sgtf.sg_context(name=name, reuse=tf.AUTO_REUSE):
		logit = sgtf.sg_layer.sg_conv1d(x, size=1, dim=voca_size, name='conv_2')
	# sequence length except zero-padding
	seq_len = sgtf.not_equal(x.sg_sum(axis=2), 0.).sg_int().sg_sum(axis=1)
	logit_softmax = tf.nn.softmax(logit)
	y, decoded = build_decoder(logit, seq_len)

	return (x, logit, logit_softmax, y), default_recover_list(name)



def build_wavenet(name, batch_size, x, output_dim, **kwargs):
	# hyperparams
	num_blocks = 3	 # dilated blocks
	num_block_layers = 5
	num_conv = 7
	num_dim = 128	  # latent dimension
	
	# mfcc feature of audio
	

	# sequence length except zero-padding
	seq_len = sgtf.not_equal(x.sg_sum(axis=2), 0.).sg_int().sg_sum(axis=1)

	with sgtf.sg_context(name=name, reuse=sgtf.AUTO_REUSE):
		logit, in_layer = get_logit(x, voca_size=output_dim, 
			num_blocks=num_blocks, num_block_layers=num_block_layers, num_conv=num_conv, num_dim=num_dim)

	#logit = sgtf.Print(logit, [sgtf.shape(logit)])
	logit_softmax = tf.nn.softmax(logit)
	y, decoded = build_decoder(logit, seq_len)

	return (logit, logit_softmax, in_layer, y), default_recover_list(name)

'''
Model Loaders
'''
def parse_args(prefix, index2label, **kwargs):
	voca_size, index2str, str2index, print_index = voca_utils(index2label)
	batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 1
	ckpt_dir = kwargs['ckpt_dir'] if 'ckpt_dir' in kwargs else default_ckpt_dirs[prefix]
	return voca_size, print_index, batch_size, ckpt_dir


from functools import wraps

def loader_decorate(tag=None):
	def loader_decorator(func):
		_name = tag
		if tag is None:
			_name = func.__name__
		@wraps(func)
		def func_wrapper(*args, **kwargs):
			if loaded_models[_name] is not None:
				return loaded_models[_name]
			
			model = dict2obj(func(*args, **kwargs))
			loaded_models[_name] = model
			return model
		return func_wrapper
	return loader_decorator   


@lru_cache(maxsize=1)
@loader_decorate('wavenet')
def loader_wavenet(prefix, **kwargs):
	# vocabulary
	index2letter = ['<EMP>', ' ']
	index2letter.extend([chr(i) for i in range(ord('a'), ord('z')+1)])
	voca_size, print_index, batch_size, ckpt_dir = parse_args(prefix, index2letter, **kwargs)
	# build graph
	x = tf.placeholder(dtype=sgtf.sg_floatx, shape=(batch_size, None, 20))
	(logit, logit_softmax, in_layer, y), recover_list = build_wavenet(prefix, batch_size, x, voca_size)
	# because previous training do not use score name, we need to strip prefix
	recover_list = strip_recover_list(prefix, recover_list)
	#print(recover_list)
	
	# load pretrained model
	try:
		load_model(ckpt_dir, recover_list) #load_model(ckpt_dir, list(diff_set))
	except Exception as e:
		print(f'initial loading failed: {e}')
		load_model(ckpt_dir)



	run_vars = [logit, logit_softmax, in_layer, y]	
	feed_vars = [x]

	return dict(run_vars=run_vars, 
		feed_vars=feed_vars, print_label=print_index, batch_size=batch_size)
	


@loader_decorate('adapt_timit')
def loader_adapt_timit(prefix, **kwargs):
	from utils import get_en_phnset
	#vocabulary
	index2en_phon = ['<EOS>', ' ']

	if 'ipa' not in kwargs:
		kwargs['ipa'] = False
		
	added = get_en_phnset(kwargs['ipa'])
	index2en_phon.extend(added)
	
	voca_size, print_index, batch_size, ckpt_dir = parse_args(prefix, index2en_phon, **kwargs)
	x = tf.placeholder(dtype=sgtf.sg_floatx, shape=(batch_size, None, 27))
	# build graph
	(x, logit, logit_softmax, y), recover_list = build_adapt_timit(prefix, batch_size, x, voca_size)
	# because previous training do not use score name, we need to strip prefix
	recover_list = strip_recover_list(prefix, recover_list)

	
	# load pretrained model
	try:
		
		load_model(ckpt_dir, recover_list)
	except Exception as e:
		print(f'initial loading failed: {e}')
		load_model(ckpt_dir)
	
	return dict(run_vars=[logit, logit_softmax, y], 
		feed_vars=[x], print_label=print_index, batch_size=batch_size)





'''
Data processing
'''


def get_input(wav, sr):
	#sr=16000
	hop_length = 512
	if sr != 16000:
		_r = sr/16000.
		hop_length = int(512 * _r)
	n_fft = 4 * hop_length

	# get mfcc feature
	mfcc = np.expand_dims(librosa.feature.mfcc(wav, sr=sr, n_fft=n_fft, hop_length=hop_length), axis=0)
	rate = hop_length * 1000. / sr
	return np.transpose(mfcc, [0, 2, 1]), rate




'''
Predictors/recognizers
'''



def zero_padding(all_inputs, padding_axis=1, concat_axis=0):
	max_len = max([i.shape[padding_axis] for i in all_inputs])
	print(f'max seq_len {max_len}')
	for i in range(len(all_inputs)):
		pad = max_len - all_inputs[i].shape[padding_axis]
		if pad > 0:
			all_inputs[i] = np.pad(all_inputs[i], ((0,0), (0,pad), (0,0)), mode='constant')
			print(f'{i}-th input, padding {pad}, shape {all_inputs[i].shape}')

	return np.concatenate(all_inputs, axis=concat_axis)


def split_batch(all_inputs, batch_size):
	if type(all_inputs) is list:
		n = len(all_inputs)
		num_batch = n // batch_size
		print('num_batch', num_batch)
		outputs = [zero_padding(all_inputs[i:i+batch_size]) for i in range(0, batch_size * num_batch, batch_size)]
		
	else: # check all_inputs.shape[0] == batch_size
		outputs = [all_inputs]

	return outputs




def predict_adapt_timit(all_inputs, prefix='adapt_timit', **kwargs) -> list:
	

	model = loader_adapt_timit(prefix, **kwargs)
	[logit, logit_softmax, y] = model.run_vars
	x = model.feed_vars[0]	

	# run network
	all_inputs = split_batch(all_inputs, model.batch_size)
	ret = []
	for _input in all_inputs:
		outs, probs, label = sess.run(model.run_vars, feed_dict={x: _input})
		
		if 'verbose' in kwargs and kwargs['verbose']:
			# print label
			model.print_label(label)

		if 'softmax' in kwargs and kwargs['softmax']:
			# normalized probability by softmax
			ret.append(probs)
		else:	
			ret.append(outs)
	ret = [np.expand_dims(out,axis=0) for batch_out in ret for out in batch_out]		
	return ret






def predict_wavenet(mfccs, prefix='wavenet', **kwargs) -> list:


	model = loader_wavenet(prefix, **kwargs)
	logit, logit_softmax, in_layer, y = model.run_vars
	x = model.feed_vars[0]

	mfccs = split_batch(mfccs, model.batch_size)

	ret = []	

	for mfcc in mfccs:
		# run network
		outs, probs, inner_features, label = sess.run(model.run_vars, feed_dict={x: mfcc})
	
		# print label
		if 'verbose' in kwargs and kwargs['verbose']:
			model.print_label(label, sep='')

		if 'softmax' in kwargs and kwargs['softmax']:
			ret.append(probs) # normalized probability by softmax
			
		elif 'inner' in kwargs and kwargs['inner']:
			ret.append(inner_features)
		else:
			ret.append(outs)
	ret = [np.expand_dims(out,axis=0) for batch_out in ret for out in batch_out]		
	return ret


'''
Test functions
'''

def test_pretrained(input_file="./sample/vocal.mp3"):
	if len(sys.argv) > 1:
		input_file = sys.argv[1]
			
	wav, sr = librosa.load(input_file, mono=True, sr=None)
	mfcc, rate = get_input(wav, sr)
	print(predict_wavenet(mfcc, verbose=True, verbal=True))
	release_session()


def test_adapt(data_dir='./sample/'):
	if len(sys.argv) > 1:
		data_dir = sys.argv[1]
		print(sys.argv[1])
	files = files_by_suffixes(data_dir, ['wav', 'mp3', 'm4a', 'WAV'])
	files = [os.path.join(data_dir, f) for f in files]
	
	all_inputs = []
	for input_file in files:
		print(f"loading {input_file} ...")
		wav, sr = librosa.load(input_file, mono=True, sr=None)
		mfcc, _ = get_input(wav, sr)
		all_inputs.append(mfcc)
		
	batch_size = len(all_inputs)
	if batch_size > 8:
		batch_size = 4
		
	outs = predict_wavenet(all_inputs, batch_size=batch_size, verbose=True, verbal=True)
	all_inputs = [out[:,:,:-1] for out in outs] # strip last dim for blank


	predict_adapt_timit(all_inputs, batch_size=batch_size, ipa=True, verbose=True, verbal=True)   
	release_session()	 


if __name__ == "__main__":
	#test_pretrained()
	test_adapt()
	
        # Fernando in	
	

