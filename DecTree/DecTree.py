''' C4.5 implement decision tree (function are divided in two py file)
    The class DecTree is core function of decision tree'''

# -*- coding:utf-8 -*-
import math
from ProcDraw import *

# execute core function class (construct and verify)
class DecTree:
    def __init__(self, num, name, t_num, index, train, test):
        self.attr_num = num             # attribute_num
        self.attr_name = name           # attribute name
        self.attr_index_list = index    # attribute index list
        self.train_set = train          # train set
        self.test_set = test            # test set
        self.label_num = t_num          # type num
        self.split_value = [0 for i in range(0, self.attr_num)]

    # judge examples and decide if necessary to build tree
    def pureJudge(self, data_set):
        for item in data_set:  # the last dim of the item is label
            if item[-1] != data_set[0][-1]:
                return False
        return True

    # get the count of item by every label
    def get_labelCount(self, data_set):
        label = {i: 0 for i in range(0, self.label_num)}  # init the label by sequence in configure
        for data in data_set:
            if data[-1] in label.keys():
                label[data[-1]] += 1
        return label

    # get the attribute with highest frequency
    def get_maxLabel(self, data_set):
        label = self.get_labelCount(data_set)
        sorted_data_set = sorted(label.items(), key=lambda x: x[1], reverse=True)
        return sorted_data_set[0][0]

    # calculate the data set entropy
    def calculateEntropy(self, data_set):
        label_count = self.get_labelCount(data_set)
        entropy = 0
        for key in label_count.keys():
            if label_count[key]:     # the type count is not 0
                prob = float(label_count[key])/float(len(data_set))
                entropy -= prob*math.log(prob, 2)
        return entropy

    # split the data set by attribute index and value
    def split_attrSet(self, attr_index, split_value, data_set):
        try:
            lower_set = []
            higher_set = []
            for item in data_set:
                if item[attr_index] <= split_value:
                    lower_set.append(item)
                else:
                    higher_set.append(item)
            return lower_set, higher_set
        except Exception as e:
            print(attr_index, e)

    # calculate the information gain(when discrete use this, choose attribute use info-gain-rate)
    def calculate_InfoGain(self, split_index, sorted_set):  # a trick(use sorted set thus no need to split set)
        low_set = sorted_set[:split_index]
        high_set = sorted_set[split_index:]
        low_ratio = float(len(low_set))/float(len(sorted_set))
        high_ratio = float(len(high_set))/float(len(sorted_set)) # a trick: must add - before Entropy
        split_info = low_ratio*self.calculateEntropy(low_set)\
                    +high_ratio*self.calculateEntropy(high_set)
        info_gain = self.calculateEntropy(sorted_set)-split_info
        return info_gain

    # calculate the information gain rate
    def calculate_InfoGainRatio(self, attr_index, split_value, data_set):
        low_set, high_set = self.split_attrSet(attr_index, split_value, data_set)
        low_ratio = float(len(low_set)) / float(len(data_set))
        high_ratio = float(len(high_set)) / float(len(data_set))  # a trick: must add - before Entropy
        if low_ratio == 0 or high_ratio == 0:
            return 0.0
        split_info = low_ratio * self.calculateEntropy(low_set) \
                     + high_ratio * self.calculateEntropy(high_set)
        info_gain = self.calculateEntropy(data_set) - split_info
        penalty = -low_ratio*math.log(low_ratio, 2)-high_ratio*math.log(high_ratio, 2)
        return info_gain/penalty

    # use one attribute to discrete the data set
    def discrete_setBinary(self, attr_index, data_set):
        try:
            max_info_gain = -1
            best_split_value = -1.0
            temp_set = sorted(data_set, key=lambda x: x[attr_index])  # sort the set from small to large
            for i in range(0, len(temp_set) - 1):
                if temp_set[i][-1] != temp_set[i + 1][-1]:  # the index may be where the sorted label change
                    info_gain = self.calculate_InfoGain(i + 1, temp_set)
                    if info_gain > max_info_gain:
                        max_info_gain = info_gain
                        best_split_value = temp_set[i][attr_index]
            return best_split_value
        except Exception as e:
            print(attr_index, e)

    # get the attribute index with maxium info gain
    def get_bestAttr(self, data_set):
        max_info_gain_ratio = -1.0
        best_split_index = -1
        best_split_value = -1.0
        for attr_index in self.attr_index_list:  # use binary split way for every attr_index
            split_value = self.discrete_setBinary(attr_index, data_set)
            info_gain_ratio = self.calculate_InfoGainRatio(attr_index, split_value, data_set)
            if info_gain_ratio > max_info_gain_ratio:   # find the index with maximum info gain ratio
                best_split_index = attr_index
                best_split_value = split_value
                max_info_gain_ratio = info_gain_ratio
        return best_split_index, best_split_value

    # build decision tree recursively
    def build_decTree(self, data_set):
        if len(data_set) is 0:      # data set is null
            return "NULL"
        elif self.pureJudge(data_set):   # if pure set, classification is end
            return data_set[0][-1]       # the last element of the item is label
        elif len(self.attr_index_list) is 0:  # if attribute index list is null
            return self.get_maxLabel(data_set)  # return the attribute index with highest frequency

        best_split_index, best_split_value = self.get_bestAttr(data_set)  # using binary way
        self.split_value[best_split_index] = best_split_value
        self.attr_index_list.remove(best_split_index)   # remove the chosen index
        self.split_value[best_split_index] = best_split_value
        lower_set, higher_set = self.split_attrSet(best_split_index, best_split_value, data_set)
        tree = {best_split_index: {}}   # build tree recursively
        tree[best_split_index]['<='] = self.build_decTree(lower_set)
        tree[best_split_index]['>'] = self.build_decTree(higher_set)
        return tree

    # store decision tree
    def store_DecTree(self, tree):
        try:
            f = open("prod_data/DecTree.pkl", 'wb')
            pk.dump(tree, f)
            f.close()
            f = open("prod_data/SplitValue.pkl", 'wb')
            pk.dump(self.split_value, f)
            f.close()
        except Exception as e:
            print("There is no Tree\n", e)

    # load decison tree
    def load_DecTree(self):
        try:
            f1 = open("prod_data/DecTree.pkl", 'rb')
            tree = pk.load(f1)
            f1.close()
            f2 = open("prod_data/SplitValue.pkl", 'rb')
            split_value = pk.load(f2)
            f2.close()
            return tree, split_value
        except Exception as e:
            print("There is no Tree\n", e)

    # use decison tree to classify
    def judgeType(self, tree, element):
        try:
            attr_index = list(tree.keys())[0]
            if element[attr_index] <= self.split_value[attr_index]:
                sub_tree = tree[attr_index]['<=']
                if type(sub_tree) == dict:
                    return self.judgeType(sub_tree, element)
                else:
                    return sub_tree
            else:
                sub_tree = tree[attr_index]['>']
                if type(sub_tree) == dict:
                    return self.judgeType(sub_tree, element)
                else:
                    return sub_tree
        except Exception as e:
            print(tree.keys(), e)

    # calculate the accuracy of classify
    def calculateAccuracy(self, tree):
        count = 0
        for item in self.test_set:
            result = self.judgeType(tree, item)
            if result == item[-1]:      # if classify is right
                count += 1
        return float(count)/float(len(self.test_set))

    # get result according to the decision tree
    def get_classifyCount(self, tree, test_set, leaf_count):
        p_index = list(tree.keys())[0]
        son_dict = tree[p_index]
        for key in son_dict.keys():
            right_count = 0
            wrong_count = 0
            lower_set, higher_set = self.split_attrSet(p_index, self.split_value[p_index], test_set)
            if key == '<=':
                sub_set = lower_set
            else:
                sub_set = higher_set
            if type(son_dict[key]).__name__ == 'dict':
                self.get_classifyCount(son_dict[key], sub_set, leaf_count)
            else:
                for item in sub_set:         # count the right num and error num
                    if item[-1] == son_dict[key]:  # the label is the class with majority count(trick)
                        right_count += 1
                    else:
                        wrong_count += 1    # the node in count is leaf node
                leaf_count.append([right_count, wrong_count, son_dict[key]])

    # pessimistic error pruning
    def cut_branchPessimistic(self, in_tree, data_set):
        p_index = list(in_tree.keys())[0]
        son_dict = in_tree[p_index]
        new_tree = {p_index: {}}
        for key in son_dict.keys():      # go to the deepest non-leaf node
            if type(son_dict[key]).__name__ == 'dict':
                lower_set, higher_set = self.split_attrSet(p_index, self.split_value[p_index], data_set)
                if key == '<=':
                    sub_set = lower_set
                else:
                    sub_set = higher_set
                leaf_count = []  # calculate the classify count (current sub tree, the len=leaf num)
                self.get_classifyCount(son_dict[key], sub_set, leaf_count)
                all_count = 0.0    # the right/wrong count of the current sub(all sample num)
                wrong_count = 0.0  # if the set label is replaced with the majority count/
                for item in leaf_count:  # get the count of the current sub tree
                    all_count += item[0] + item[1]
                    wrong_count += item[1]    # get the wrong judge ratio
                et = wrong_count+0.5          # because the leaf node label is the type with majority count
                eTt = wrong_count+len(leaf_count)/2   # thus the pure operation error count in train set == wrong count
                SeTt = math.sqrt(eTt*(1-eTt/all_count))
                if et <= eTt + SeTt:   # judge if necessary to prune
                    new_tree[p_index][key] = self.get_maxLabel(sub_set)
                else:
                   new_tree[p_index][key]= self.cut_branchPessimistic(son_dict, sub_set)
            else:
                new_tree[p_index][key] = son_dict[key]
        return new_tree


if __name__ == "__main__":     # execute function
    pre = "RAW"
    load = "1LOAD"
    proc_data = ProcData()
    if pre == "RAW":  # raw data, need to be processed
        proc_data.first_preProcess()  # if it's the first time to use the data set
    else:
        proc_data.second_preProcess()  # if there exist the data sets processed
    dec_tree = DecTree(proc_data.attr_num, proc_data.attr_name, proc_data.type_num,
                       proc_data.attr_index, proc_data.train_set,proc_data.test_set)
    # could load the tree from the memory(the processed data)
    if load == "LOAD":  # must be corresponding with the data set
        tree_dict, dec_tree.split_value = dec_tree.load_DecTree()
    else:
        tree_dict = dec_tree.build_decTree(dec_tree.train_set)
        dec_tree.store_DecTree(tree_dict)
    print (tree_dict)
    print("accuracy = ", dec_tree.calculateAccuracy(tree_dict))
    # start prune the tree us pessimistic prune
    new_tree_dict = dec_tree.cut_branchPessimistic(tree_dict, dec_tree.train_set)
    print("After pessimistic prune, the accuracy = ", dec_tree.calculateAccuracy(new_tree_dict))
    # draw the tree before and after cut
    draw_tree = DrawTree(dec_tree.split_value, dec_tree.attr_name, proc_data.type_name)
    draw_tree.execute_drawDecTree(tree_dict, new_tree_dict)

