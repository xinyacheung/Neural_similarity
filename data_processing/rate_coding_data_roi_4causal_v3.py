# rate coding (the rate of firing or number of spikes)
import scipy.io
import glob
import numpy as np
import re
import pandas as pd
import sys, os
# os.chdir(os.path.dirname(__file__))
from joblib import Parallel, delayed
import time

start_time = time.time()
num_f = 30 # number of neurons/features
bin_size = 0.05

def time_bin_spike_trains(spikes,label,region,isolated_neurons,t0,t1):
    isolated_neurons=list(isolated_neurons)
    # print(spikes.shape) #[trial, num. of neuron in one roi or lobe]
    # print(spikes[1,:].reshape(1,-1).shape) #shape=[1, num. of neuron in one roi or lobe]
    trial_num=0
    for trial in range(spikes.shape[0]): # each trial
        roi_vec_temp = np.zeros((1, num_f))
        if (label.iloc[trial,0],label.iloc[trial,1]) in comb:
            temp_spike = spikes[trial,:].reshape(1,-1)
            k=0
            for idx in range(spikes.shape[1]): # each neuron
                if isolated_neurons[idx]==1:
                    continue                  #if neuron wasn't isolated then skip
                spike = temp_spike[0,idx]
                if isinstance(spike,float) or isinstance(spike,int):
                    continue
                else:
                    temp = [a for i, a in enumerate(spike) if a >= t0 and a < t1]
                    #spike trains from [t0, to, t1] -- t1-t0 = bin_size
                    roi_vec_temp[0,k] = len(temp)
                    k += 1
                if k==num_f:
                    break
            if k >0:
                if trial_num == 0:
                    roi_vec = roi_vec_temp
                else:
                    roi_vec = np.concatenate((roi_vec, roi_vec_temp), axis=0)
                trial_num += 1

    try: #not every session includes nodes in all six regions.
        return roi_vec
    except:
        return 1

def time_bin_spike_trains_two(spikes,label,region,isolated_neurons,t0,t1,task_index):
    iidx =  label.iloc[:,task_index]
    isolated_neurons=list(isolated_neurons)
    # print(spikes.shape) #[trial, num. of neuron in one roi or lobe]
    # print(spikes[1,:].reshape(1,-1).shape) #shape=[1, num. of neuron in one roi or lobe]
    trial_num=0
    for trial in range(spikes.shape[0]): # fist is color, second is motion
        roi_vec_temp = np.zeros((1, num_f))
        if label.iloc[trial,task_index] in [30,90]: # each trial
            temp_spike = spikes[trial,:].reshape(1,-1)
            k=0
            for idx in range(spikes.shape[1]): # each neuron
                if isolated_neurons[idx]==1:
                    continue                  #if neuron wasn't isolated then skip
                spike = temp_spike[0,idx]
                if isinstance(spike,float) or isinstance(spike,int):
                    continue
                else:
                    temp = [a for i, a in enumerate(spike) if a>=t0 and a<t1]
                    #spike trains from [t0, to, t1] -- t1-t0 = bin_size
                    roi_vec_temp[0, k] = len(temp)
                    k += 1
                if k == num_f:
                    break
                if k > 0:
                    if trial_num == 0:
                        roi_vec_1 = roi_vec_temp
                    else:
                        roi_vec_1 = np.concatenate((roi_vec_1, roi_vec_temp), axis=0)
                    trial_num += 1

    trial_num = 0
    for trial in range(spikes.shape[0]):
        roi_vec_temp = np.zeros((1, num_f))
        if label.iloc[trial,task_index] in [-30,-90]:
            temp_spike = spikes[trial,:].reshape(1,-1)
            k = 0
            for idx in range(spikes.shape[1]):
                if isolated_neurons[idx]==1:
                    continue                  #if neuron wasn't isolated then skip # use 1
                spike = temp_spike[0,idx]
                if isinstance(spike,float) or isinstance(spike,int):
                    continue
                else:
                    temp = [a for i, a in enumerate(spike) if a >= t0 and a < t1]

                    roi_vec_temp[0, k] = len(temp)
                    k += 1
                if k == num_f:
                    break
                if k > 0:
                    if trial_num == 0:
                        roi_vec_0 = roi_vec_temp
                    else:
                        roi_vec_0 = np.concatenate((roi_vec_0, roi_vec_temp), axis=0)
                    trial_num += 1
    try:
        return roi_vec_1,roi_vec_0
    except:
        return 1,1

def main(t0,t1):
    first1=0
    first2=0
    first3=0

    for file_name_i in data_files:
        print(f'{data_files.index(file_name_i)}/{len(data_files)}')
        mat = scipy.io.loadmat(file_name_i, squeeze_me=True, struct_as_record=True)
        try:
            sess_num = re.search(r'(/)([0-9]*)(.mat)', file_name_i).group(2)
        except:
            sess_num = re.search(r'(/)([0-9]*_[0-9]*)(.mat)', file_name_i).group(2)

        unitInfo = pd.read_csv(parent_save_dir + '/unitInfo' + '/' + sess_num + '.txt', sep=',')

        trialInfo_allTasks = pd.read_csv(parent_save_dir + '/trialInfo' + '/' + sess_num + '.txt', sep=',')

        trialInfo = pd.read_csv(parent_save_dir + '/mini_data/trial_info' + sess_num + '.csv', sep=',')

        badTrials = trialInfo_allTasks.badTrials.copy()
        spikes = mat['spikeTimes'].copy()
        spikes = spikes[trialInfo_allTasks.task == 'mocol', :]
        badTrials = badTrials[trialInfo_allTasks.task == 'mocol']
        spikes = spikes[badTrials == 0, :]

        spikes = spikes[:, unitInfo.area == region]

        condition = trialInfo.rule.copy()
        spikes_motion = spikes[condition == 'motion']
        labels_motion = trialInfo[['color', 'direction']].copy()  # color & direction coordinates
        labels_motion = labels_motion[condition == 'motion']
        spikes_color = spikes[condition == 'color']
        labels_color = trialInfo[['color', 'direction']].copy()  # color & direction coordinates
        labels_color = labels_color[condition == 'color']

        labels_all = trialInfo[['color', 'direction']].copy()
        spikes_all = spikes

        isolated_neurons = unitInfo.isolated
        isolated_neurons = isolated_neurons[unitInfo.area == region]

        # roi_vec_m = time_bin_spike_trains(spikes_motion, labels_motion, region, isolated_neurons, t0, t1)
        # roi_vec_c = time_bin_spike_trains(spikes_color, labels_color, region, isolated_neurons, t0, t1)
        #
        # if isinstance(roi_vec_m, int) or isinstance(roi_vec_c, int):
        #     t = 1
        # else:
        #     if first1 == 0:
        #         roi_vec_m_temp = roi_vec_m
        #         roi_vec_c_temp = roi_vec_c
        #         first1 += 1
        #     else:
        #         roi_vec_m_temp = np.concatenate((roi_vec_m_temp, roi_vec_m), axis=0)
        #         roi_vec_c_temp = np.concatenate((roi_vec_c_temp, roi_vec_m), axis=0)

        roi_vec_1, roi_vec_0 = time_bin_spike_trains_two(spikes_all, labels_all, region, isolated_neurons, t0, t1,
                                                         task_index=0)
        if isinstance(roi_vec_0, int):
            t = 1
        else:
            if first2 == 0:
                roi_vec_1_temp = roi_vec_1
                roi_vec_0_temp = roi_vec_0
                first2 += 1
            else:
                roi_vec_1_temp = np.concatenate((roi_vec_1_temp, roi_vec_1), axis=0)
                roi_vec_0_temp = np.concatenate((roi_vec_0_temp, roi_vec_0), axis=0)

        roi_vec_1, roi_vec_0 = time_bin_spike_trains_two(spikes_all, labels_all, region, isolated_neurons, t0, t1,
                                                         task_index=1)
        if isinstance(roi_vec_0, int):
            t = 1
        else:
            if first3 == 0:
                roi_vec_1_temp2 = roi_vec_1
                roi_vec_0_temp2 = roi_vec_0
                first3 += 1
            else:
                roi_vec_1_temp2 = np.concatenate((roi_vec_1_temp2, roi_vec_1), axis=0)
                roi_vec_0_temp2 = np.concatenate((roi_vec_0_temp2, roi_vec_0), axis=0)

    np.save(save_dir + f'/color/1_{round(t1, 2)}_{region}_{bin_size}.npy', roi_vec_1_temp)
    np.save(save_dir + f'/color/0_{round(t1, 2)}_{region}_{bin_size}.npy', roi_vec_0_temp)

    np.save(save_dir + f'/motion/1_{round(t1, 2)}_{region}_{bin_size}.npy', roi_vec_1_temp2)
    np.save(save_dir + f'/motion/0_{round(t1, 2)}_{region}_{bin_size}.npy', roi_vec_0_temp2)

    # np.save(save_dir+f'/task/color_{round(t1,2)}_{region}_{bin_size}_1.npy',roi_vec_m_temp)
    # np.save(save_dir + f'/task/motion_{round(t1, 2)}_{region}_{bin_size}_1.npy', roi_vec_m_temp)


parent_save_dir = '/home/zhang/lovelab/scripts/save_dir'
save_dir = parent_save_dir + '/mini_data_causal_v3' #'/mini_data_roiwell_50ms' # new is [][] instead of [,] #first dataset is in 20ms (0-200ms)
data_dir = '/home/zhang/lovelab/scripts/Siegel2015data'

mat_files_match = data_dir + '/*.mat'
data_files = glob.glob(mat_files_match)
data_files.sort()
ref_dists = pd.read_csv(parent_save_dir + '/reference_dists.csv', sep=',')

# fist is color, second is motion
comb = [(90, 90), (90, 30), (90, -30), (90, -90), (30, 90), (30, 30), (30, -30), (30, -90),
        (-30, 90), (-30, 30), (-30, -30), (-30, -90), (-90, 90), (-90, 30), (-90, -30), (-90, -90)]


# t0 = np.arange(-0.1,0.22,0.02)
# t1 = np.arange(-0.08,0.24,0.02)

# t0 = np.arange(-0.52,-0.12+0.02,bin_size)
# t1 = np.arange(-0.5,-0.1+0.02,bin_size)

# t0 = np.arange(-1,-0.52,bin_size)
# t1 = np.arange(-1+0.02,-0.5,bin_size)
# time_range = [(t0[i],t1[i]) for i in range(t0.shape[0])]

# t0 = np.arange(-0.05,0.20+0.05,0.025)
# t1 = t0 + bin_size
# time_range = [(t0[i],t1[i]) for i in range(t0.shape[0])]


# time_range = [(-1,0)]
t0 = np.arange(-0.08,0.23,0.01)
t1 =t0 + bin_size
time_range = [(t0[i],t1[i]) for i in range(t0.shape[0])]
print(time_range)
print('ok')

for region in ['PFC','FEF','LIP','MT','IT','V4']:
    njobs = 27
    Parallel(n_jobs=njobs)( delayed(main) (t0,t1) for t0,t1 in time_range )

