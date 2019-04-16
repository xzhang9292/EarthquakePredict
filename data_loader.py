import numpy as np
import datetime
from multiprocessing import Pool

data_directionary = 'data/train_sub/'
processed_data_directory = 'data/train_4096/'

train_sets = [0,1,3,4,6,7,9,10,12,14,15,16]
test_sets = [2,5,8,11,13]

def load_set(ind):
	print(str(datetime.datetime.now())+': start loading training set %i'%ind)
	data = np.loadtxt(data_directionary+'train_sub_%i.csv'%ind, delimiter=',')
	print(str(datetime.datetime.now())+': preprocessing training set %i'%ind)
	#print(str(datetime.datetime.now()+': preprocessing training set %i'%ind))
	td = data[:-1,1]-data[1:,1]
	# each segment has a time difference around 0.001, where the time interval between  measurement is 4.5e-6
	sp = np.where(td>0.0001)[0]
	fsp = np.zeros(sp.shape,dtype=int)
	fsp[1:] = sp[:-1]
	fsp[0] = -1
	sp_dif = sp-fsp
	msp = np.where(sp_dif!=4096)[0]
	for p in msp:
		d_count = 4096-sp_dif[p]
		d_ind = sp[p]
		for offset in range(d_count):
			data = np.insert(data, d_ind+offset+1, data[d_ind],axis=0)
	processed_data = np.reshape(data[:,0],(-1,4096))
	processed_data = processed_data.astype(int)
	#n_data = processed_data.shape[0]
	processed_label = data[4095::4096,1]
	print('sanity check:')
	print('data shape: ', processed_data.shape)
	print('label shape: ', processed_label.shape)
	return (processed_data,processed_label)

def data_test(ind):
	print(str(datetime.datetime.now())+': start loading training set %i'%ind)
	data = np.loadtxt(data_directionary+'train_sub_%i.csv'%ind, delimiter=',')
	print(str(datetime.datetime.now())+': preprocessing training set %i'%ind)
	#print(str(datetime.datetime.now()+': preprocessing training set %i'%ind))
	td = data[:-1,1]-data[1:,1]
	# each segment has a time difference around 0.001, where the time interval between  measurement is 4.5e-6
	sp = np.where(td>0.0001)[0]
	fsp = np.zeros(sp.shape,dtype=int)
	fsp[1:] = sp[:-1]
	fsp[0] = -1
	sp_dif = sp-fsp
	msp = np.where(sp_dif!=4096)[0]
	for p in msp:
		d_count = 4096-sp_dif[p]
		print(p,d_count)
		'''
		d_ind = sp[p]
		for offset in range(d_count):
			data = np.insert(data, d_ind+offset+1, data[d_ind],axis=0)
		'''
	'''
	processed_data = np.reshape(data[:,0],(-1,4096))
	processed_data = processed_data.astype(int)
	#n_data = processed_data.shape[0]
	processed_label = data[4095::4096,1]
	print('sanity check:')
	print('data shape: ', processed_data.shape)
	print('label shape: ', processed_label.shape)
	return (processed_data,processed_label)
	'''

def process_set(ind):
	data, label = load_set(ind)
	np.savetxt(processed_data_directory+'train_sub_%i_data.csv'%ind, data, fmt='%i',delimiter=',')
	np.savetxt(processed_data_directory+'train_sub_%i_label.csv'%ind, label, fmt='%.10f',delimiter=',')

def load_preprocessed_set(ind):
	print(str(datetime.datetime.now())+': start loading data set %i'%ind)
	data = np.loadtxt(processed_data_directory+'train_sub_%i_data.csv'%ind, delimiter=',', dtype=int)
	label = np.loadtxt(processed_data_directory+'train_sub_%i_label.csv'%ind, delimiter=',', dtype=float)
	return (ind,(data, label))

def get_data():
	train_data = {}
	test_data = {}
	n_train_data = 0
	n_test_data = 0 
	workerpool = Pool(8)
	traindataresult = workerpool.map(load_preprocessed_set, train_sets)
	#workerpool.join()
	for train in traindataresult:
		dkey, data = train 
		train_data[dkey] = data
		n_train_data += len(train_data[dkey][1])
	testdataresult = workerpool.map(load_preprocessed_set, test_sets)
	for test in testdataresult:
		dkey, data = test
		test_data[dkey] = data
		n_test_data += len(test_data[dkey][1])
	workerpool.close()
	workerpool.join()
	return {'train':train_data, 'test':test_data, 'n_train':n_train_data, 'n_test':n_test_data}

def get_brief_data():
	train_data = {}
	test_data = {}
	n_train_data = 0
	n_test_data = 0 
	workerpool = Pool(8)
	traindataresult = workerpool.map(load_preprocessed_set, [16])
	#workerpool.join()
	for train in traindataresult:
		dkey, data = train 
		train_data[dkey] = data
		n_train_data += len(train_data[dkey][1])
	testdataresult = workerpool.map(load_preprocessed_set, [0])
	for test in testdataresult:
		dkey, data = test
		test_data[dkey] = data
		n_test_data += len(test_data[dkey][1])
	workerpool.close()
	workerpool.join()
	return {'train':train_data, 'test':test_data, 'n_train':n_train_data, 'n_test':n_test_data}

def get_data_batch(data, batch_size):
	# every 150000 can fit in 36 * 4096 + 2544
	data_ind = []
	for k in data.keys():
		subset, _ = data[k]
		n_d = subset.shape[0]
		subdata_ind = np.zeros((n_d-36+1,2),dtype=int)
		subdata_ind[:,0] = k
		subdata_ind[:,1] = np.asarray(list(range(n_d-36+1)))
		data_ind.append(subdata_ind)
	data_ind = np.concatenate(data_ind,axis=0)
	np.random.shuffle(data_ind)
	total_n_d = data_ind.shape[0]
	for i in range(0,total_n_d,batch_size):
		data_range = data_ind[i:min(i+batch_size,total_n_d),:]
		batchdata = np.zeros((batch_size, 36* 4096),dtype=int)
		batchlabel = np.zeros((batch_size, 1), dtype=float)
		for j in range(data_range.shape[0]):
			subdata, sublabel = data[data_range[j,0]]
			batchdata[j,:] = (subdata[data_range[j,1]:data_range[j,1]+36,:]).flatten()
			batchlabel[j,0] = sublabel[data_range[j,1]+35]
		yield (batchdata, batchlabel)

def get_data_vector_batch(data, batch_size):
	# every 150000 can fit in 36 * 4096 + 2544
	data_ind = []
	for k in data.keys():
		subset, _ = data[k]
		n_d = subset.shape[0]
		subdata_ind = np.zeros((n_d,2),dtype=int)
		subdata_ind[:,0] = k
		subdata_ind[:,1] = np.asarray(list(range(n_d)))
		data_ind.append(subdata_ind)
	data_ind = np.concatenate(data_ind,axis=0)
	np.random.shuffle(data_ind)
	total_n_d = data_ind.shape[0]
	for i in range(0,total_n_d,batch_size):
		data_range = data_ind[i:min(i+batch_size,total_n_d),:]
		batchdata = np.zeros((batch_size, 4096),dtype=int)
		batchlabel = np.zeros((batch_size, 1), dtype=float)
		for j in range(data_range.shape[0]):
			subdata, sublabel = data[data_range[j,0]]
			batchdata[j,:] = subdata[data_range[j,1],:]
			batchlabel[j,0] = sublabel[data_range[j,1]]
		yield (batchdata, batchlabel)

def get_data_sts():
	data = get_data()
	train_data = data['train']
	result = 0
	max_range = 0
	for k in train_data.keys():
		train_sub = train_data[k]
		result += np.sum(train_sub[0])
		set_max = np.max(np.abs(train_sub[0]))
		if set_max> max_range:
			max_range = set_max
			print(k)
	return result/(data['n_train']*4096),max_range




#def get_training_data():
if __name__ == "__main__":
	print(get_data_sts())
	'''
	for i in train_sets:
		data_test(i)
	for i in test_sets:
		data_test(i)
	data = load_preprocessed_set(0)
	data = {0:data}
	counter = 0
	for i in get_data_batch(data, 10):
		d, l = i
		print(d)
		print(l)
		break
	'''
	#get_data()