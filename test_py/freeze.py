# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 14:51:02 2018

@author: jinglonz
"""

import os, argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util


dir = os.path.dirname(os.path.realpath(__file__))


def freeze_graph(model_folder):
    """
        Args: the model file path
    """
    
    ## get the checkpoint full path
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path
    print('/'.join(input_checkpoint.split('/')[:-1]))

    ## set the frozen graph file name
    absolute_model_folder = '/'.join(input_checkpoint.split('/')[:-1])
    
    output_graph_path = absolute_model_folder + '/frozen_model.pb'
    
    output_node_names = 'Accuracy/correct_prediction'
    
    # We clear the devices, to allow TensorFlow to control on the loading where it wants operations to be calculated
    clear_devices = True
    
    # import the meta graph and retrieve the Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
    
    # retrieve the protocol buffer graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    
    # start a session and restore the weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        
        # We use a built-in TF helper to export variables to constant
        output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names.split(','))
        
        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph_path, 'wb') as gf:
            gf.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', default="model/", type=str, help='Model folder to export')
    args = parser.parse_args()
    
    freeze_graph(args.model_folder)