import numpy as np
import re

'''Channel generation'''
def channel_generator(path_list):
	channel = np.zeros([Number_antenna, 1], dtype=np.complex)
	channel_adj=np.zeros([Number_antenna, 1], dtype=np.complex)
	a_x = lambda phi_az, phi_ele, M: [np.exp(1j * np.pi * np.sin(phi_ele) * np.cos(phi_az) * i) for i in range(M)]
	for channel_path in path_list:
		path_loss_dB, path_phase, path_delay, AOA_ele_D, AOA_az_D = channel_path
		AOA_ele = AOA_ele_D / 180 * np.pi
		AOA_az = AOA_az_D / 180 * np.pi
		path_phase = path_phase /( Key *180 * np.pi)
		a_A_x = np.array(a_x(AOA_az, AOA_ele, Number_antenna), dtype=np.complex).reshape([Number_antenna, 1])
		path_loss = 10 ** (path_loss_dB / 10)
		channel += np.sqrt(path_loss) * np.exp(1j * (path_phase + 2 * np.pi * path_delay * B)) * a_A_x
		channel_adj+=np.sqrt(path_loss) * np.exp(1j * (path_phase + 2 *(2)* np.pi * path_delay * B)) * a_A_x
	return channel.reshape((Number_antenna,)),channel_adj.reshape((Number_antenna,))

'''Rename the output file'''
import shutil
def rename_output_file():
	for id_area in ID_inf:
		for bs_fre in enumerate(BS_frequency):
			filename = 'F://CODES//CODE_META_WIRELESS//PMData/try3.paths.t001_' + str(bs_fre[0] + 1).zfill(
				2) + '.r0' + id_area + '.p2m'
			newfilename = 'F://CODES//CODE_META_WIRELESS//RePMData//BS' + str(bs_fre[1]) + 'area' + id_area + '.p2m'
			shutil.copy(filename, newfilename)
			print(filename)

'''Exactraction of path parameter from .p2m file'''
import scipy.io as scio
def Exactraction_path_parameters():
	for id_area in ID_inf:
		channel_matrix_per_area = []
		for fre in BS_frequency:
			filename =  'F://CODES//CODE_META_WIRELESS//RePMData//BS' + str(fre) + 'area' + id_area + '.p2m'
			channel_infor_envir = open(filename, 'r')
			channel_infor_content = channel_infor_envir.readlines()
			num_users_index = 0
			channel_sample_list = []
			found_num_user_flag = 1
			found_new_user_flag = 1
			found_position_flag=0
			for line in enumerate(channel_infor_content):
				if line[1][0] != '#':
					if found_num_user_flag:
						found_num_user_flag = 0
						number_user_str = line[1][0:4]
						Number_users_given_total = int(number_user_str)
						# print('Found the total number of users  for frequency ', fre, ' area', id_area, ' : ',
						# 	  Number_users_given_total,'num_ant',Number_antenna)

					if line[1][0:2] == 'Tx':
						if found_new_user_flag == 1:
							found_new_user_flag = 0
							num_users_index += 1
							# print('find a new user: ', num_users_index)
							path_list_per_user = []
							path_number_extract = channel_infor_content[line[0] - 3]
							if path_number_extract.find('.') == -1:
								path_number_str = re.findall('[-Ee0-9.]+', path_number_extract)
								userindex, pathnumber = [int(i) for i in path_number_str[0:2]]
								# print('the', num_users_index, '-th  user has ', pathnumber,
								# 	  'paths and its orginial index is', userindex)
								path_infor = channel_infor_content[line[0] - 1]
								data_path_str = re.findall('[-Ee0-9.]+', path_infor)
								data_path = [float(i) for i in data_path_str[2:7]]
								path_list_per_user.append(data_path)


							else:
								num_users_index -= 1
								# print('one user has no path and num_users_index-=1: ', num_users_index)

						else:
							'''position data'''
							if len(path_list_per_user)==1:# and found_position_flag==0:
								position_infor = channel_infor_content[line[0] - 2]
								position_str = re.findall('[-Ee0-9.]+', position_infor)
								px, py, pz = [float(i) for i in position_str]
								if pz==2.0: #'''Receiver point'''
									# print('Right receiver point')
									position_per_user=np.array([px,py])
									# position_data = np.concatenate((position_per_user, supvector))
									found_position_flag=1
								else:
									print('find no LOS path')
									print(len(path_list_per_user))
									print(id_area,fre,num_users_index,'pz',pz)
									raise SystemExit('abnormal sample')
							path_infor = channel_infor_content[line[0] - 1]
							data_path_str = re.findall('[-Ee0-9.]+', path_infor)
							data_path = [float(i) for i in data_path_str[2:7]]
							path_list_per_user.append(data_path)
							if len(path_list_per_user) == Max_path or len(path_list_per_user) == pathnumber:
								found_new_user_flag = 1
								if found_position_flag==1:
									found_position_flag = 0
									csi,csi_adj = channel_generator(path_list_per_user)
									csi_pos=np.array([csi,position_per_user,csi_adj])
									channel_sample_list.append(csi_pos)
									# print('channel_sample_list',np.asarray(channel_sample_list).shape)
								else:
									raise SystemExit('position error')

								# print('user number +1, channel finished')
					# print(np.asarray(channel_sample_list).shape)
					if len(channel_sample_list) == Number_users_given_total:
						print('all users has paths and then break')
						print('the number of users in datasets is: ', len(channel_sample_list))
						break
					elif len(channel_sample_list) == Max_users:
						print('the number of users is enough: ', Max_users, 'and then break')
						print('the number of users in datasets is: ', len(channel_sample_list))
						break
			channel_matrix_per_area.append(channel_sample_list)
			print('channel_sample_list', np.asarray(channel_sample_list).shape)
			print('channel_matrix_per_area', np.asarray(channel_matrix_per_area).shape)
			dataNew = './DataSave/channel_matrix_area_pos_adj' + str(Key)+str(id_area)+str(Number_antenna)+'.mat'
			scio.savemat(dataNew, {'channel_matrix_area_pos_adj': channel_matrix_per_area})
			channel_infor_envir.close()

'''channel_matrix_per_area shape (frequency, user number, antenna) e.g., (34, 1800, 64)'''

'''Globel parameters'''
Max_path = 30  # max number of paths
Max_users=50000
Number_antenna = 64
B = 2e4
# BS_frequency=np.arange(1000,2982,60)
BS_frequency=np.arange(1900,2025,120)
# ID_area=np.arange(37,86,1)
ID_area=np.arange(37,86,1)
ID_inf=[str(ind) for ind in ID_area]
supvector=np.zeros((1,Number_antenna-2))
Key=1
# rename_output_file()
# Exactraction_path_parameters()
time_lenth=6  # infact  is 3-1
channel_pairs_task = []
for id_area in ID_inf:
	dataNew = './DataSave/channel_matrix_area_pos_adj' + str(Key)+str(id_area) +str(Number_antenna)+'.mat'
	data_matrix = scio.loadmat(dataNew)
	channel_matrix_area_pos = data_matrix['channel_matrix_area_pos_adj']
	temp_num_user = np.shape(channel_matrix_area_pos)[1]
	gap_list=6*np.arange(1,time_lenth)   #fixed parameter
	for task_index in np.arange(gap_list[-1],temp_num_user):
		# print(channel_matrix_area_pos.shape)
		channel_matrix_task=channel_matrix_area_pos[:,task_index,:]
		# print(channel_matrix_task.shape)
		channel_uplink = np.squeeze(channel_matrix_task[0,0],axis=0)
		channel_downlink =  np.squeeze(channel_matrix_task[1,0],axis=0)
		channel_sub =  np.squeeze(channel_matrix_task[1,2],axis=0)
		if (channel_matrix_task[0,1]==channel_matrix_task[1,1]).all()==False:
			raise SystemExit('position data error')
		position_xy=np.squeeze(channel_matrix_task[0,1])
		channel_his_list=[channel_matrix_area_pos[1, task_index - gap, 0] for gap in gap_list]
		hist_long_post=np.squeeze(channel_matrix_area_pos[1, task_index - gap_list[-1], 1])
		# print(hist_long_post.shape)
		# print(hist_long_post)
		if position_xy[0]-hist_long_post[0]==0.5*(gap_list[-1]//6):
			channel_time=[np.squeeze(his_data) for his_data in channel_his_list]
			# for his_data in channel_his_list:
			# 	print('his_data', his_data.shape)
			# 	channel_hist = np.squeeze(his_data)
			# 	channel_time.append(channel_hist)

			channel_time=np.asarray(channel_time)
			# print('channel_time', np.asarray(channel_time).shape)
		else:
			print('task_index: ',task_index,position_xy)
			raise SystemExit('time data error')
		pair = np.array([channel_uplink, channel_downlink, position_xy,channel_sub,channel_time])
		channel_pairs_task.append(pair)
	print(np.asarray(channel_pairs_task).shape)


x1,x2=np.asarray(channel_pairs_task).shape
#
dataNew = './DataSave/samples_pos_adj_time' +str(time_lenth)+str(Key)+str(Number_antenna)+'_'+str(x1)+'.mat'
scio.savemat(dataNew, {'channel_pairs_task': np.asarray(channel_pairs_task)})
print('stored in ',dataNew)


'''test position data code'''
# dataNew = './DataSave/channel_matrix_area_pos_adj' + str(Key) + str(37) + str(Number_antenna) + '.mat'
# # scio.savemat(dataNew, {'channel_matrix_area_pos': channel_matrix_per_area})
# data=scio.loadmat(dataNew)
# channel_matrix_area_pos=data['channel_matrix_area_pos_adj']
#
# for ii in range(np.shape(channel_matrix_area_pos)[1]//100):
# 	posi=channel_matrix_area_pos[1,ii,1]
# 	print(posi)


# dataNew = './DataSave/samples_pos_adj' +str(Key)+str(Number_antenna)+'_'+str(64974)+'.mat'
# print(dataNew)
# data=scio.loadmat(dataNew)
# channel_pairs_task=data['channel_pairs_task']
# print(channel_pairs_task.shape)
# def nmsetest(vec1,vec2):
# 	return np.square(np.linalg.norm(vec1-vec2,2))/np.square(np.linalg.norm(vec2,2))
# m1=0
# m2=0
# m3=0
# m0=0
# time=10
# for index in range(time):#channel_pairs_task.shape//500):
# 	m1+=nmsetest(np.squeeze(channel_pairs_task[index,0],axis=0),np.squeeze(channel_pairs_task[index,1],axis=0))/time
# 	m0+=nmsetest(np.squeeze(channel_pairs_task[index,1],axis=0),np.squeeze(channel_pairs_task[index,1],axis=0))/time
# 	m2+=nmsetest(np.squeeze(channel_pairs_task[index,3],axis=0),np.squeeze(channel_pairs_task[index,1],axis=0))/time
# 	m3+=nmsetest(np.squeeze(channel_pairs_task[index,4],axis=0),np.squeeze(channel_pairs_task[index,1],axis=0))/time
# print(m0,m1,m2,m3)

