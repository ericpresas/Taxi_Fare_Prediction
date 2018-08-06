import csv
import numpy as np

class Dataset:

	def __init__(self):

		self.key_names = []
		self.train_data = []
		self.validation_data = []
		self.test_data = []

	def create(self):
		with open('all/train/file_01.csv', 'rt') as csvfile:
			rows = csv.reader(csvfile, delimiter=',')
			data = []
			aux_fields = ['year', 'month', 'hour']
			for i,row in enumerate(rows):
				if (i==0):
					header = row
					self.key_names = header
				else:
					tmp_dict = {}
					for content, field in zip(row, header):
						try:
							tmp_dict[field] = float(content)
						except ValueError:
							tmp_dict[field] = content
					tmp_dict['year'] = float(tmp_dict['key'].split()[0].split('-')[0])
					tmp_dict['month'] = float(tmp_dict['key'].split()[0].split('-')[1])
					tmp_dict['hour'] = float(tmp_dict['key'].split()[1].split(':')[0])
					print('Cost viatge: ', tmp_dict['fare_amount'])
					data.append(tmp_dict)

			print(data[0])

		size = len(data)
		train_len = round(size * 0.75)
		val_len = size - train_len
		self.train_data = data[:train_len]
		self.validation_data = data[train_len:]

		print('Total: ', len(data))
		print('Train samples: ', len(self.train_data))
		print('Val samples: ', len(self.validation_data))


	def from_listdict_to_array(self, dict, pred_field):

		array = []
		labels = []
		for content in dict:
			list_ = []
			for value in list(content.items()):
				if isinstance(value[1], str)!=True:
					list_.append(value[1])
					if value[0]==pred_field:
						labels.append(value[1])
			try:
				array.append(np.array(list_, dtype='f4'))
			except:
				array.append(np.array(list_))

		return np.array(array), np.array(labels)


	 
