'''The class ProcData is mainly used to pre-process data
   The class DrawTree is mainly used to draw the decision tree'''

# -*- coding:utf-8 -*-
import random as rd
import pickle as pk
from codecs import open as cd_open
import matplotlib.pyplot as plt


# read config file and preprocess
class ProcData:
    def __init__(self):
        self.attr_num = 0        # attribute_num
        self.attr_name = []      # attribute name
        self.attr_index = []     # attribute index according to the order in data set
        self.train_set = []      # train set
        self.test_set = []       # test set
        self.type_pos = 0        # the label pos(the beginning or the end
        self.file_name = None    # file name
        self.type_num = 0        # the type num (label num)
        self.train_size = 0      # train sample size
        self.test_size = 0       # test sample size
        self.type_name = []      # type name list
        self.type_size = []      # every type size
        self.type_start_index = [] # every type start index
        self.offset_list = []    # offset list

    # read config file and get the parameter(This part is similiar with the first homework)
    def readConfig(self):
        f = cd_open("config.cfg", 'r', encoding="utf-8")
        conf = f.readlines()
        temp = conf[0].split(',')  # the path of the data set
        self.file_name = temp[0]
        self.type_pos = temp[1]
        temp = conf[1].split(',')  # obtain type num, attribute num, train set num, test set num
        self.type_num = int(temp[0])
        self.attr_num = int(temp[1])
        self.train_size = int(temp[2])
        self.test_size = int(temp[3])
        temp = conf[2].split(',')   # obtain the type name list by order
        temp1 = conf[3].split(',')  # obtain the sample size every type name
        temp2 = conf[4].split(',')  # obtain the start index of every type(by order)
        for i in range(0, self.type_num):
            self.type_name.append(temp[i])
            self.type_size.append(int(temp1[i]))
            self.type_start_index.append(int(temp2[i]))
        temp = conf[5].split(',')   # obtain the attribute name list
        self.attr_name = temp[:self.attr_num]
        self.attr_index = [i for i in range(0, self.attr_num)]
        f.close()

    # get the index of random train set (get sample of every type according to sampled num)
    def getIndex(self, sample_index, sample_num, offset_list, aim_set, all_sample):
        for i in range(0, sample_num):  # generate train set(size=sample_num)
            offset = rd.randint(0, self.type_size[sample_index]-1)
            while offset in offset_list:
                offset = rd.randint(0, self.type_size[sample_index]-1)
            temp = all_sample[offset + self.type_start_index[sample_index]].split(',')
            offset_list.append(offset)
            if self.type_pos == 0:
                temp = temp[1:]
            temp[-1] = str(sample_index)  # the last is the label, replaced with number by order
            aim_set.append(list(map(eval, temp)))  # need to convert the string type to float

    # write list in pkl format
    def writePkl(self, filename, lst):
        fout = open(filename, 'wb')  # pkl文件要采用流写入方式
        pk.dump(lst, fout)
        fout.close()

    # read pkl file in list
    def readPkl(self, filename):
        fin = open(filename, 'rb')
        lst = pk.load(fin)
        fin.close()
        return lst

    # data pre-process, get train/test set according to proportion
    def first_preProcess(self):
        try:
            tr_set = []
            te_set = []
            self.readConfig()
            f = open(self.file_name, 'r')  # read sample in cache
            all_sample = f.readlines()
            f.close()
            for i in range(0, self.type_num):  # assure equal proportion when sampling
                tr_num = int((self.train_size + i) / self.type_num)  # trick to classify num
                te_num = int((self.test_size + self.type_num-i-1) / self.type_num)
                if te_num> (self.type_size[i]-tr_num):
                    te_num = self.type_size[i]-tr_num
                offset_list = []  # in order to assure the chosen data will not be chosen the next time
                self.getIndex(i, tr_num, offset_list, self.train_set, all_sample)
                self.getIndex(i, te_num, offset_list, self.test_set, all_sample)
            self.writePkl("prod_data/train_set.pkl", self.train_set)  # store the data after processing
            self.writePkl("prod_data/test_set.pkl", self.test_set)
        except Exception as e:
            print(e)

    # when train/test set .pkl file exist, read directly
    def second_preProcess(self):
        self.readConfig()
        self.train_set = self.readPkl("prod_data/train_set.pkl")
        self.test_set = self.readPkl("prod_data/test_set.pkl")


# draw the decision tree
class DrawTree:
    def __init__(self, value, attr, label):
        self.attr_name = attr      # attr name list
        self.label_name = label    # label name list
        self.split_value = value   # split value list
        self.tree_width = 0        # in fact is decision tree leaf node num
        self.tree_height = 0       # in fact is decision tree depth
        self.x_offset = 0.0        # every node x offset
        self.y_offset = 0.0        # every node y offset
        self.ax = None      # fd = face color  ec = edge color lw = line width
        self.arrow_args = dict(arrowstyle='<-', fc='k', lw='0.8')
        self.dec_node = dict(boxstyle="square", fc=(0.8,0.8,0.9), ec='g')
        self.leaf_node = dict(boxstyle="round",fc=(0.9,0.8,0.9), ec='r')

    # get the leaf num of the current tree
    def get_leafNum(self, tree_dic):
        leaf_num = 0
        first_key = list(tree_dic.keys())[0]
        second_tree = tree_dic[first_key]
        for key in second_tree.keys():    # if leaf node:+1; else get leaf node recursively
            if type(second_tree[key]).__name__ == "dict":
                leaf_num += self.get_leafNum(second_tree[key])
            else:
                leaf_num += 1
        return leaf_num

    # get the depth of the current tree
    def get_treeDepth(self, tree_dic):
        max_depth = 0
        first_key = list(tree_dic.keys())[0]
        second_tree = tree_dic[first_key]
        for key in second_tree.keys():  # if leaf node:+1; else get leaf node recursively
            if type(second_tree[key]).__name__ == "dict":
                now_depth = 1+self.get_treeDepth(second_tree[key])
            else:
                now_depth = 1
            if now_depth > max_depth:
                max_depth = now_depth
        return max_depth

    # get the node complete information
    def get_nodeInfo(self, index, n_type):
        if n_type == 0:  # if the node is decision node
            return "AttrIndex=" + str(index) + "\n" + self.attr_name[index]
        elif type(index).__name__ == 'int':  # if the node is leaf node with label
            return "TypeIndex=" + str(index) + "\n" + self.label_name[index]
        else:
            return "Type=" + index

    # draw a node, in other words,it's a node in decsion tree
    def drawNode(self, nodeInfo, son_cord, parent_cord, node_args):
        self.ax.annotate(nodeInfo, xy=parent_cord,xycoords='axes fraction',xytext=son_cord,
                         textcoords='axes fraction',va="bottom", ha="center", bbox=node_args,
                         arrowprops=self.arrow_args)

    # add the text in the mid of the arrow(x,y)
    def add_arrowText(self, son_cord, parent_cord, key, text):
        x_mid = float(son_cord[0]+parent_cord[0]-0.1)/2.0
        y_mid = float(son_cord[1]+parent_cord[1]+0.1)/2.0
        self.ax.text(x_mid, y_mid, key+str(text))

    # draw the decision tree recursively(depth first strategy)
    def draw_decTree(self, now_tree, parent_cord, new_key, split_value):
        leaf_num = self.get_leafNum(now_tree)    # get leaf num(to assure coordinate)
        pt_index = list(now_tree.keys())[0]
        son_cord = (self.x_offset + (1.0+leaf_num)/(2.0*self.tree_width), self.y_offset)
        self.add_arrowText(son_cord, parent_cord, new_key, split_value)
        self.drawNode(self.get_nodeInfo(pt_index, 0), son_cord, parent_cord, self.dec_node)
        son_dict = now_tree[pt_index]   # get the son tree of this key
        self.y_offset -= 1.0/self.tree_height   # the y coordinate decrease
        for key in son_dict.keys():          # for every son tree
            if type(son_dict[key]).__name__ == 'dict':
                self.draw_decTree(son_dict[key], son_cord, key, self.split_value[pt_index])
            else:   # if the son node is not a tree(the leaf node)
                self.x_offset += 1.0/self.tree_width
                self.add_arrowText((self.x_offset, self.y_offset), son_cord, key, self.split_value[pt_index])
                self.drawNode(self.get_nodeInfo(son_dict[key], 1), (self.x_offset, self.y_offset), son_cord, self.leaf_node)
        self.y_offset += 1.0/self.tree_height

    # draw a decision tree on the subplot
    def draw_subplotInit(self, subplot, now_tree, name):
        self.ax = subplot
        self.tree_width = float(self.get_leafNum(now_tree))  # get width and height of the tree
        self.tree_height = float(self.get_treeDepth(now_tree))
        self.x_offset = -0.5 / self.tree_width
        self.y_offset = 1.0
        self.draw_decTree(now_tree, (0.5, 1.0), '', '')
        self.ax.text(0.0, 1.0, name)

    # init the draw parameter of the decision
    def execute_drawDecTree(self, dec_tree, pes_tree):
        figure = plt.figure(num=0, figsize=(10, 5), facecolor="white")
        figure.clear()     # define axis with no information
        axis_info = dict(xticks=[], yticks=[]) # get a figure with no axis
        # divide the picture into i*j blocks, the picture in k
        self.draw_subplotInit(figure.add_subplot(121, frameon=False, **axis_info), dec_tree, "DEC_TREE")
        self.draw_subplotInit(figure.add_subplot(122, frameon=False, **axis_info), pes_tree, "PEP_TREE")
        plt.show()
