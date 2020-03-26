from pyspark import SparkContext
import os, copy
import json
import argparse
import time
import numpy as np
import xgboost as xgb
from collections import defaultdict

class XGBoost():
	'''Model-Based Recommendation System'''
	def __init__(self, num_partition):
		self._num_partition = num_partition

	def read_csv(self, spark_context, input_file_path):
		sc = spark_context
		RDD = sc.textFile(input_file_path)
		### Remove header ###
		rdd_header = RDD.first()
		header = sc.parallelize([rdd_header])
		RDD = RDD.subtract(header).repartition(self._num_partition)
		return RDD

	def read_json(self, spark_context, input_file_path):
		sc = spark_context
		RDD = sc.textFile(input_file_path)
		if 'user' in input_file_path:
			read_lambda = self.read_user
		else:
			read_lambda = self.read_business
		RDD = RDD.map(read_lambda).repartition(self._num_partition)
		return RDD

	def data_process(self, RDD, user_dict, business_dict, is_test=False):
		if is_test:
			RDD = RDD.map(lambda x: self.get_feature_test(data=x, user_dict=user_dict, business_dict=business_dict))
			dataset =  np.array(RDD.collect())
			return dataset[:,:-1].astype(np.float32), list(dataset[:,-1])
		else:
			RDD = RDD.map(lambda x: self.get_feature_train(data=x, user_dict=user_dict, business_dict=business_dict))
			dataset =  np.array(RDD.collect()).astype(np.float32)
			return dataset[:,:-1], dataset[:,-1].astype(int)

	@staticmethod
	def get_feature_train(data, user_dict, business_dict):
		data = data.split(',')
		feature = copy.deepcopy(user_dict[data[0]])
		feature.extend(business_dict[data[1]])
		feature.append(data[2])
		return feature

	@staticmethod
	def get_feature_test(data, user_dict, business_dict):
		data = data.split(',')
		feature = copy.deepcopy(user_dict[data[0]])
		feature.extend(business_dict[data[1]])
		feature.append([data[0], data[1]])
		# print(feature)
		return feature

	@staticmethod
	def read_user(data):
		data = json.loads(data)
		feature = []
		user_list = ['review_count', 'useful', 'funny', 'cool', 'fans', 'average_stars']
		for u in user_list:
			feature.append(data[u])
		user_id = data['user_id']
		return user_id, feature

	@staticmethod
	def read_business(data):
		data = json.loads(data)
		feature = []
		business_list = ['stars', 'review_count', 'is_open']
		for u in business_list:
			feature.append(data[u])
		business_id = data['business_id']
		return business_id, feature

	@staticmethod
	def print_csv(name, prediction, output_file_path):
		with open(output_file_path, 'w') as f:
			f.write("user_id, business_id, prediction\n")
			for ind, rate in enumerate(prediction):
				f.write(name[ind][0]+','+name[ind][1]+','+str(rate))
				f.write("\n")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("train_folder", default="data", help="train file folder")
	parser.add_argument("test", default="yelp_val.csv", help="validation yelp file")
	parser.add_argument("output", default="output.csv", help="output file")
	args = parser.parse_args()

	sc = SparkContext('local[*]', 'Item-Based-Collaborative-Filter')
	sc.setLogLevel("ERROR")

	start_time = time.time()
	xgb_model = XGBoost(num_partition=20)
	print("Reading Files")
	user_rdd = xgb_model.read_json(spark_context=sc, input_file_path=args.train_folder+"/user.json")
	business_rdd = xgb_model.read_json(spark_context=sc, input_file_path=args.train_folder+"/business.json")
	train_rdd = xgb_model.read_csv(spark_context=sc, input_file_path=args.train_folder+"/yelp_train.csv")
	test_rdd = xgb_model.read_csv(spark_context=sc, input_file_path=args.test)
	print("Building Dictionaries")
	user_dict = user_rdd.collectAsMap()
	business_dict = business_rdd.collectAsMap()
	print("Data Processing")
	X_train, Y_train = xgb_model.data_process(RDD=train_rdd, user_dict=user_dict, business_dict=business_dict)
	X_test, Y_test = xgb_model.data_process(RDD=test_rdd, user_dict=user_dict, business_dict=business_dict, is_test=True)

	print("Data Loaded")

	model = xgb.XGBRegressor()
	model.fit(X_train, Y_train)

	print("Train Finished")
	preds = model.predict(X_test)
	xgb_model.print_csv(name=Y_test, prediction=preds, output_file_path=args.output)


	total_time = time.time() - start_time
	print("Duration:", total_time)

if __name__ == '__main__':
	main()