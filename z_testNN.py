import tensorflow as tf


class test_NN:
    def __init__ (self, input_node, output_node, act_func):
        self.input_node = input_node
        self.output_node = output_node
        self.act_func = act_func

    def know_tensor_size(self, selected_node):
        col = tf.rank(selected_node)
        row = tf.size(selected_node) / col
        return row, col

    def set_1(self):
        self.inode_row, self.inode_col = self.know_tensor_size(self.input_node)
        self.onode_row, self.onode_col = self.know_tensor_size(self.output_node)
        self.hnode = tf.zeros([(self.inode_row + self.onode_row) / 2])

    def propagation(self):
        self.hidden