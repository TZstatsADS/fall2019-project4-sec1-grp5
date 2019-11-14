import numpy as np
from matplotlib import pyplot as plt 

class ObjectiveFunction(object):
	def __init__(self, training, testing, b_i, b_u, features = 10, lmbda = 0.3):
		self.train_data = training
		self.test_data = testing
		self.num_users = 610
		self.num_movies = 9742
		self.features = features
		self.b_i = b_i
		self.b_u = b_u
		self.mu = None #### CALCULATE TOTAL AVERAGE RATING HERE!!
		self.lmbda = lmbda
		self.p = None
		self.q = None
	def matrix_factorization(self):
		np.random.seed(0)
		self.p = np.random.uniform(low=0, high=2.5, size=(self.num_users, self.features))
		self.q = np.random.uniform(low=0, high=2.5, size=(self.num_movies, self.features))
		self.r = np.matmul(self.p, self.q.T)  #SHAPE: 610 X 9472

	
	def temp_dyn_pred_reg(self, user_id, movie_id,time_interval):
		pred = self.mu + self.b_i[movie_id,time_interval] + self.b_u[user_id,time_interval] + np.matmul(self.q[movie_id,:], self.p[user_id,:].T)
		return pred


	def get_time_interval(self,time):
		#INSERT FUNCTION TO GET TIME INTERVAL 
		raise NotImplementedError

	def train_obj_function(self, batch_size = 12, obj_function = 'GD'):

		num_batches = len(self.train_data)//batch_size 
		train_loss_values = np.zeros((num_batches))
		validation_loss_values = np.zeros((num_batches))


		for i in range(num_batches):
			
			#Inialize gradients
			dl_dp, dl_dq = 0,0
			train_loss = 0
			#find gradients
			for j in range(batch_size):
				user_id = self.train_data[i*batch_size+j,:][0]
				movie_id = self.train_data[i*batch_size+j,:][1]
				actual_rating = self.train_data[i*batch_size+j,:][2]
				time = self.train_data[i*batch_size+j,:][3]
				time_interval = self.get_time_interval(time)
				prediction = self.temp_dyn_pred_reg(user_id, movie_id,time_interval)


				dpred_dq = np.sum(self.p[user_id,:]) ## double check this, but its based of assumption dq/dq = array of ones, so matrix multipls array of ones by p[row,:] = sum(p[row,:]) 
				dpred_dp = np.sum(self.q[movie_id,:])

				if obj_function == 'GD':
					train_loss + = (actual_rating - prediction)**2 + self.GDreg(user_id, movie_id)
					dl_dq += 2*(actual_rating - prediction)*dpred_dq + self.deriv_GDreg(self.q, j)
					dl_dp += 2*(actual_rating - prediction)*dpred_dp + self.deriv_GDreg(self.p, j)

				elif obj_function == 'ALS':

					train_loss + = (actual_rating - prediction)**2 + self.ALSreg()

					dl_dq += 2*(actual_rating - prediction)*dpred_dq + self.deriv_ALSreg(self.q, j)
					dl_dp += 2*(actual_rating - prediction)*dpred_dp + self.deriv_ALSreg(self.p, j)

			#update p and q matrices via gradient descent, add value to train_loss array for plotting later
			train_loss_values[j] = train_loss
			self.gradient_descent(self.p, dl_dp)
			self.gradient_descent(self.q, dl_dq)

		return train_loss_values

	def GDreg(self, user_id, movie_id):
		return self.lmbda*(np.linalg.norm(self.p[user_id,:])**2 + np.linalg.norm(self.q[movie_id,:])**2)
	def deriv_GDreg(self,matrix, num):
		return 2*self.lmbda*np.linalg.norm(matrix[num,:])

	def ALSreg(self, user_id, movie_id):
		raise NotImplementedError
	def deriv_ALSreg(self):
		raise NotImplementedError


	def get_validation_loss(self, obj_function ='GD'):
		test_loss = 0
		for j in range(len(self.test_data)):
			user_id = self.test_data[j,:][0]
			movie_id = self.test_data[j,:][1]
			actual_rating = self.test_data[j,:][2]
			time = self.test_data[j,:][3]
			time_interval = self.get_time_interval(time)
			prediction = self.temp_dyn_pred_reg(user_id, movie_id,time_interval)
			if obj_function == 'GD':
				test_loss + = (actual_rating - prediction)**2 + self.GDreg(user_id, movie_id)
			elif obj_function == 'ALS':
				test_loss + = (actual_rating - prediction)**2 + self.ALSreg(user_id, movie_id)
		return test_loss


	def train_gd(self, num_epochs= 20, objfunc):
			
		val_loss = np.zeros((num_epochs))
		train_loss = np.zeros((num_epochs))

		for i in range(num_epochs):
			np.shuffle(self.train_data)
			train_loss_array = self.train_obj_function(obj_function = objfunc)
			train_loss[i] = np.average(train_loss_array)
			val_loss[i] = get_validation_loss(obj_function = objfunc)
		
		#plot results
		x = np.arange(1, num_epochs) 
		plt.title("Nonprobabilistic Gradient Descent training") 
		plt.xlabel("# of Minibatches") 
		plt.ylabel("loss") 
		plt.plot(x,train_loss) 
		plt.plot(x,val_array) 
		plt.show()


	def gradient_descent(self, matrix, dl, learning_rate = 0.01):
		for i in range(len(matrix)):
			for j in range(len(matrix[0])):
				matrix[i,j] -= learning_rate*dl


	def post_processing(self):
		raise NotImplementedError

	def predict(self):
		self.r = np.matmul(self.p, self.q.T)  #SHAPE: 610 X 9472
		return self.r[user_id, movie_id]

if __name__ == '__main__':
	
	# train_data = 
	# test_data = 
	# b_i = 
	# b_u = 
	# objfunc1 = ObjectiveFunction(train_data, test_data, b_i, b_u)
	# objfunc1.matrix_factorization()
	# objfunc1.train_gd('GD')

	# objfunc2 = ObjectiveFunction(train_data, test_data, b_i, b_u)
	# objfunc2.matrix_factorization()
	# objfunc2.train_gd('ALS')



