import os.path
import tensorflow as tf
from tensorflow.python.platform import gfile

LOGDIR='tf_files/graph_summaries/mobilenet_v1_0.50_224'

if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)
with tf.Session() as sess:
    model_filename = 'tf_files/models/mobilenet_v1_0.50_224/frozen_graph.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    #writer = tf.train.SummaryWriter(LOGDIR, graph_def)
    writer = tf.summary.FileWriter(LOGDIR)
    writer.close()