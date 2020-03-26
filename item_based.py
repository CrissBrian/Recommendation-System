from pyspark import SparkContext
import os
import argparse
import time
import numpy as np
from collections import defaultdict

class IBCF(object):
	'''Item-Based Collaborative Filter'''
	def __init__(self, N, num_partition):
		self._business_dict = None
		self._user_dict = None

		self._N = N
		self._num_partition = num_partition

	def readfile(self, spark_context, input_file_path):
		sc = spark_context
		RDD = sc.textFile(input_file_path)
		### Remove header ###
		rdd_header = RDD.first()
		header = sc.parallelize([rdd_header])
		RDD = RDD.subtract(header).repartition(self._num_partition)
		return RDD

	def data_process(self, RDD):
		self._business_dict = self.get_business_dict(RDD)
		self._user_dict = self.get_user_dict(RDD)

	def inference(self, RDD):
		prediction = RDD.map(self.get_user_business) \
				.map(lambda x: self.predict(data=x, N=self._N)) \
				.collect()
		self._prediction = prediction

	def predict(self, data, N):
		user, business_x = data
		business_list = self._user_dict.get(user)
		WR = []
		R_all = []
		for business_y in business_list:
			w = self.find_correlation(bx=business_x, by=business_y)
			r = self._business_dict.get(business_y).get(user)
			R_all.append(r)
			if w > 0:
				WR.append([w,r])
		if len(WR) < N:
			bx = self._business_dict.get(business_x)
			if bx is not None:
				p = bx.get('mean')
			else:
				p = np.array(R_all).mean()
		else:
			WR = sorted(WR, reverse=True)[:N]
			WR = np.array(WR)
			W = WR[:,0]
			R = WR[:,1]
			p = np.dot(W, R) / W.sum()
		if p < 1.2:
			p = 1.2
		if p > 4.95:
			p = 4.95
		return [user, business_x, p]

	def find_correlation(self, bx, by):
		bx = self._business_dict.get(bx)
		by = self._business_dict.get(by)
		if bx is None or by is None:
			return 0
		rx = bx.pop('mean')
		ry = by.pop('mean')
		setx = set(bx)
		sety = set(by)
		x = []
		y = []
		inter = setx.intersection(sety)
		for key in inter:
			x.append(bx[key])
			y.append(by[key])
		w = self.pearson_similarity(x=np.array(x), y=np.array(y), rx=rx, ry=ry)
		bx.update({"mean": rx})
		by.update({"mean": ry})
		return w

	@staticmethod
	def pearson_similarity(x, y, rx, ry):
		if len(x) == 0:
			return 0
		x = x - rx
		y = y - ry
		up = np.dot(x, y)
		if up == 0:
			return 0
		down = np.linalg.norm(x) * np.linalg.norm(y)
		w = up / down
		return w

	def get_business_dict(self, RDD):
		business = RDD.map(self.get_business_user) \
			.groupByKey().mapValues(self.list2INTdict).collect()
		business_dict = self.list2dict(business)
		return business_dict

	def get_user_dict(self, RDD):
		user = RDD.map(self.get_user_business) \
			.groupByKey().mapValues(list).collect()
		user_dict = self.list2dict(user)
		return user_dict

	@staticmethod
	def print_csv(data, output_file_path):
		with open(output_file_path, 'w') as f:
			f.write("user_id, business_id, prediction\n")
			for i in data:
				f.write(str(i).replace("[","").replace("]","")
					.replace("'","").replace(", ",","))
				f.write("\n")

	@staticmethod
	def list2dict(data):
		dict = {}
		for x in data:
			dict.update({x[0]: x[1]})
		return dict

	@staticmethod
	def list2INTdict(data):
		dict = {}
		mean = 0
		count = 0
		for x in data:
			dict.update({x[0]: float(x[1])})
			mean +=float(x[1])
			count += 1
		mean /= count
		dict.update({"mean": mean})
		return dict

	@staticmethod
	def get_business_user(x):
		x = x.split(',')
		return [x[1],[x[0], x[2]]]

	@staticmethod
	def get_user_business(x):
		x = x.split(',')
		return [x[0],x[1]]

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("train", default="yelp_train.csv", help="train yelp file")
	parser.add_argument("test", default="yelp_val.csv", help="validation yelp file")
	parser.add_argument("output", default="output.csv", help="output file")
	args = parser.parse_args()

	sc = SparkContext('local[*]', 'Item-Based-Collaborative-Filter')
	sc.setLogLevel("ERROR")

	start_time = time.time()
	ibcf = IBCF(N=150, num_partition=20)
	### read files
	train_rdd = ibcf.readfile(spark_context=sc, input_file_path=args.train)
	val_rdd = ibcf.readfile(spark_context=sc, input_file_path=args.test)
	### do things
	ibcf.data_process(RDD=train_rdd)
	ibcf.inference(RDD=val_rdd)
	ibcf.print_csv(data=ibcf._prediction, output_file_path=args.output)

	total_time = time.time() - start_time
	print("Duration:", total_time)

if __name__ == '__main__':
	main()