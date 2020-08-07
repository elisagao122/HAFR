import scipy.sparse as sp
import numpy as np
from time import time
import pickle
import datetime

class Dataset(object):
    '''
    Loading the data file
        trainMatrix: load rating records as sparse matrix for class Data
        trianList: load rating records as list to speed up user's feature retrieval
        testRatings: load leave-one-out rating test for class Evaluate
        testNegatives: sample the items not rated by user
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_training_file_as_matrix(path + ".train.rating")
        self.trainList = self.load_training_file_as_list(path + ".train.rating")
        self.testRatings = self.load_training_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)
        self.num_users, self.num_items = self.trainMatrix.shape
	
	self.ingreCodeDict = np.load(path+"_ingre_code_file.npy")
	self.embImage = np.load(path + "_image_features_float.npy")
	self.image_size = self.embImage.shape[1]
	
	self.validRatings, self.valid_users = self.load_valid_file_as_list(path + ".valid.rating")
        self.validNegatives = self.load_negative_file(path + ".valid.negative")
	self.validTestRatings = self.load_valid_test_file_as_dict(path+".valid.rating", path+".test.rating")
        self.num_ingredients = 33147
	self.cold_list, self.cold_num, self.train_item_list = self.get_cold_start_item_num()
	self.ingreNum = self.load_id_ingre_num(path+"_id_ingre_num_file")
	


    def load_valid_test_file_as_dict(self, valid_file, test_file):
        validTestRatings = {}
        for u in range(self.num_users):
                validTestRatings[u] = set()
        fv = open(valid_file, "r")
        for line in fv:
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                validTestRatings[u].add(i)
        fv.close()
        ft = open(test_file, "r")
        for line in ft:
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                validTestRatings[u].add(i)
        ft.close()
        return validTestRatings


    def get_cold_start_item_num(self):
        train_item_list = []
        for i_list in self.trainList:
                train_item_list.extend(i_list)
        test_item_list = []
        for r in self.testRatings:
                test_item_list.extend(r)
        valid_item_list = []
        for r in self.validRatings:
                valid_item_list.extend(r)
        c_list = list((set(test_item_list) | set(valid_item_list))- set(train_item_list))
        t_list = list(set(train_item_list))
        return c_list, len(c_list), len(t_list)
	

    def load_image(self, filename):
	fr = open(filename, 'rb')
        image_feature_dict_from_pickle = pickle.load(fr)
        fr.flush()
        fr.close()
	return image_feature_dict_from_pickle

    def load_id_ingre_code(self, filename):
	fr = open(filename, 'rb')
        dict_from_pickle = pickle.load(fr)
        fr.flush()
        fr.close()
	return dict_from_pickle

    def load_id_ingre_num(self, filename):
	fr = open(filename, "r")
	ingreNumDict = {}
	for line in fr:
		arr = line.strip().split("\t")
		ingreNumDict[int(arr[0])] = int(arr[1])
	return ingreNumDict


    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_training_file_as_matrix(self, filename):
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
		
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()
        return mat

    def load_training_file_as_list(self, filename):
        # Get number of users and items
        u_ = 0
        lists, items = [], []
        with open(filename, "r") as f:
            line = f.readline()
            index = 0
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                if u_ < u:
                    index = 0
                    lists.append(items)
                    items = []
                    u_ += 1
		index += 1
                items.append(i)
                line = f.readline()
        lists.append(items)
        return lists

    def load_valid_file_as_list(self, filename):
        # Get number of users and items
        lists, items, user_list = [], [], []
        with open(filename, "r") as f:
            line = f.readline()
            index = 0
            u_ = int(line.split("\t")[0])
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                if u_ < u:
                    index = 0
                    lists.append(items)
                    user_list.append(u_)
                    items = []
                    u_ = u
                index += 1
                items.append(i)
                line = f.readline()
        lists.append(items)
        user_list.append(u)
        return lists, user_list
