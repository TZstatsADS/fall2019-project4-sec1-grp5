import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
import ipdb

class ObjectiveFunction(object):
	def __init__(self, training, testing, features = 10, lmbda = 0.1):
		self.train_data = training
		self.test_data = testing
		self.num_users = 610
		self.num_movies = 9742
		self.features = features
		#self.b_i = b_i
		#self.b_u = b_u
		self.mu = None #### CALCULATE TOTAL AVERAGE RATING HERE!!
		self.lmbda = lmbda
		self.p = None
		self.q = None
		self.learning_rate = 0.1
	def matrix_factorization(self):
		np.random.seed(0)
		self.p = np.random.uniform(low=0, high=2, size=(self.num_users, self.features))
		self.q = np.random.uniform(low=0, high=2, size=(self.num_movies, self.features))
		self.r = np.matmul(self.p, self.q.T)  #SHAPE: 610 X 9472

	'''
	def temp_dyn_pred_reg(self, user_id, movie_id,time_interval):
		pred = self.mu + self.b_i[movie_id,time_interval] + self.b_u[user_id,time_interval] + np.matmul(self.q[movie_id,:], self.p[user_id,:].T)
		return pred
	'''

	def train_obj_function(self, batch_size = 1, obj_function = 'GD'):

		num_batches = len(self.train_data)//batch_size 
		train_loss_values = np.zeros((num_batches))
		validation_loss_values = np.zeros((num_batches))

		train_loss = 0

		for i in range(num_batches):
			
			#Inialize gradients
			dl_dp, dl_dq = 0,0
			#find gradients
			for j in range(batch_size):

				user_id = self.train_data[i*batch_size+j,:][0]
				movie_id = self.train_data[i*batch_size+j,:][1]
				actual_rating = self.train_data[i*batch_size+j,:][2]				
				#time = self.train_data[i*batch_size+j,:][3]
				#time_interval = self.get_time_interval(time)

				prediction = np.matmul(self.q[movie_id,:], self.p[user_id,:].T) # self.temp_dyn_pred_reg(user_id, movie_id,time_interval)
				#print(prediction)
				dpred_dq = self.p[user_id,:] ## double check this, but its based of assumption dq/dq = array of ones, so matrix multipls array of ones by p[row,:] = sum(p[row,:]) 
				dpred_dp = self.q[movie_id,:]

				if obj_function == 'GD':

					train_loss_values[i] = (actual_rating - prediction)**2 + self.GDreg(user_id, movie_id)
					#train_loss += (actual_rating - prediction)**2 + self.GDreg(user_id, movie_id)
					dl_dq += (actual_rating - prediction)*dpred_dq - self.deriv_GDreg(self.q, movie_id)
					dl_dp += (actual_rating - prediction)*dpred_dp - self.deriv_GDreg(self.p, user_id)
				
				#update p and q matrices via gradient descent, add value to train_loss array for plotting later
				self.p[user_id,:] += self.learning_rate*dl_dp
				self.q[movie_id,:] += self.learning_rate*dl_dq
		#train_loss_values[j] = train_loss
		
		#print(train_loss)
			#self.gradient_descent(self.p, dl_dp)
			#self.gradient_descent(self.q, dl_dq)
		return train_loss_values

	def GDreg(self, user_id, movie_id):
		return self.lmbda*(np.linalg.norm(self.p[user_id,:])**2 + np.linalg.norm(self.q[movie_id,:])**2)
	def deriv_GDreg(self,matrix, num):
		return 2*self.lmbda*np.linalg.norm(matrix[num,:])*matrix[num,:]

	def ALSreg(self, user_id, movie_id):
		raise NotImplementedError
	def deriv_ALSreg(self):
		raise NotImplementedError


	def get_validation_loss(self, obj_function ='GD'):
		test_loss = 0
		test_loss_values = np.zeros((len(self.test_data)))
		for j in range(len(self.test_data)):
			user_id = self.test_data[j,:][0]
			movie_id = self.test_data[j,:][1]
			actual_rating = self.test_data[j,:][2]
			prediction = np.matmul(self.q[movie_id,:], self.p[user_id,:].T) #self.temp_dyn_pred_reg(user_id, movie_id,time_interval)
			if obj_function == 'GD':
				test_loss_values[j] = (actual_rating - prediction)**2 + self.GDreg(user_id, movie_id)
			
		return np.average(test_loss_values)


	def train_gd(self, objfunc, num_epochs= 20):
			
		val_loss = np.zeros((num_epochs))
		train_loss = np.zeros((num_epochs))

		val_error = np.zeros((num_epochs))
		train_error = np.zeros((num_epochs))


		for i in range(num_epochs):
			np.random.shuffle(self.train_data)
			train_error[i] = self.predict()
			val_error[i] = self.predict(train=False)
			
			train_loss_array = self.train_obj_function(obj_function = objfunc)
			train_loss[i] = np.average(train_loss_array)
			val_loss[i] = self.get_validation_loss(obj_function = objfunc)

		#plot results
		x = np.arange(1, num_epochs+1) 

		fig, (loss, error) = plt.subplots(1, 2)
		fig.suptitle("Nonprobabilistic Gradient Descent training")
		loss.set_xlabel("# of Epochs") 
		loss.set_ylabel("loss") 
		loss.plot(x,train_loss) 
		loss.plot(x,val_loss) 
		error.set_xlabel("# of Epochs") 
		error.set_ylabel("error") 
		error.plot(x,train_error) 
		error.plot(x,val_error) 
		
		plt.show()

	#def gradient_descent(self, matrix, dl, learning_rate = 0.001):
	#	for i in range(len(matrix)):
	#		for j in range(len(matrix[0])):
	#			matrix[i,j] -= learning_rate*dl


	def post_processing(self):
		raise NotImplementedError

	def predict(self, train = True):
		self.r = np.matmul(self.p, self.q.T)
		if train:
			data = self.train_data
		else:
			data = self.test_data
		total = len(data)
		correct = 0
		for i in range(total):
			user_id = data[i,0]
			movie_id = data[i,1]
			if round(self.r[user_id, movie_id]*2)/2 == data[i,2]:
				correct +=1
		return (total-correct)/total

if __name__ == '__main__':


	data_path = 'train_test_data.xlsx'
	xl_file = pd.ExcelFile(data_path)
	df_names= [('bin1_train','bin1_test'),('bin2_train','bin2_test'),('bin3_train','bin3_test')]
	dfs = {sheet_name: xl_file.parse(sheet_name) 
		for sheet_name in xl_file.sheet_names}

	#Bin 1
	#train_df = dfs[df_names[0][0]]
	#test_df = dfs[df_names[0][1]]

	train_data1 = dfs[df_names[0][0]][['userId','movieId','rating']].to_numpy().astype(int)
	test_data1 = dfs[df_names[0][1]][['userId','movieId','rating']].to_numpy().astype(int)
	objfunc1 = ObjectiveFunction(train_data1, test_data1)
	objfunc1.matrix_factorization()
	objfunc1.train_gd('GD')


'''
	#Bin 2
	train_data2 = dfs[df_names[1][0]][['userId','movieId','rating']].to_numpy().astype(int)
	test_data2 = dfs[df_names[1][1]][['userId','movieId','rating']].to_numpy().astype(int)
	objfunc1 = ObjectiveFunction(train_data2, test_data2)
	objfunc1.matrix_factorization()
	objfunc1.train_gd('GD')

	#Bin 3
	train_data3 = dfs[df_names[2][0]][['userId','movieId','rating']].to_numpy().astype(int)
	test_data3 = dfs[df_names[2][1]][['userId','movieId','rating']].to_numpy().astype(int)
	objfunc1 = ObjectiveFunction(train_data3, test_data3)
	objfunc1.matrix_factorization()
	objfunc1.train_gd('GD')
'''

