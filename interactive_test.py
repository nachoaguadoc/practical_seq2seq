import tensorflow as tf
import numpy as np

# preprocessed data
from datasets.twitter import data
import data_utils

# load data from pickle and npy files
metadata, idx_q, idx_a = data.load_data(PATH='datasets/twitter/')
(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)

# parameters 
xseq_len = trainX.shape[-1]
yseq_len = trainY.shape[-1]
batch_size = 16
lookup = metadata['idx2w']
xvocab_size = len(metadata['idx2w'])  
yvocab_size = xvocab_size
emb_dim = 1024

import seq2seq_wrapper
import importlib
importlib.reload(seq2seq_wrapper)

# esto se tarda a lo más 1 minuto
model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/twitter/',
                               emb_dim=emb_dim,
                               num_layers=3
                               )
test_batch_gen = data_utils.rand_batch_gen(testX, testY, 256)

sess = model.restore_last_session()

while True:
    print('\n')
    question = input('Q: ')
    if question == 'exit':
        break

    input_ = np.array(data.pad_seq(question.split(' '), metadata['w2idx'], xseq_len), ndmin=2).T

    output = model.predict(sess, input_)[0]
    decoded = data_utils.decode(sequence=output, lookup=metadata['idx2w'], separator=' ').split(' ')

    print('{}{}'.format('A: ', ' '.join(decoded)))


#input_ = test_batch_gen.__next__()[0]
query = "how old are you"



'''
for ii, oi in zip(input_.T, output):
    q = data_utils.decode(sequence=ii, lookup=metadata['idx2w'], separator=' ')
    decoded = data_utils.decode(sequence=oi, lookup=metadata['idx2w'], separator=' ').split(' ')
    if decoded not in replies:
        chat = 'q : [{0}]; a : [{1}]'.format(q, ' '.join(decoded))
        print(chat)
        #conversation.append('{}\n'.format(chat))
        replies.append(decoded)

'''