# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 15:15:16 2018

@author: jinglonz
"""

import argparse
import tensorflow as tf


def load_graph(frozen_graph_filename):
    ## parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, 'rb') as gf:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(gf.read())
        
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None
        )
    return graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_graph_filename", default="model/frozen_model.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()

    graph = load_graph(args.frozen_graph_filename)
    
    # We can list operations
    #op.values() gives you a list of tensors it produces
    #op.name gives you the name
    #输入,输出结点也是operation,所以,我们可以得到operation的名字
    for op in graph.get_operations():
        print(op.name,op.values())

    print('FINISHED!!!')