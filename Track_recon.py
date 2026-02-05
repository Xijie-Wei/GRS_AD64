from unpack_package import UnpackPackage
import ROOT
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel
import numpy as np
from scipy.stats import norm
plt.rcParams['text.usetex'] = True

use_external_noise_file = True
inte_range = np.array([-5,6])#area used to calculate integartion
bg_range = 20# use wavedatapoint[0:bg_range] to calculate background
time_stamp_offset = 0
#-------------------------------------------------------------------------------------------------------

data_info,trigger_info = UnpackPackage("data_file/RAW_data_20260203_171011.bin")

pack_pointer_board_channel_timeStamp = data_info["PackagePointer"]
pack_pointer_board_channel_timeStamp_valid = data_info["PackagePointerValid"]

existed_board_id = data_info["ExistedBoardId"]
existed_channel_id = data_info["ExistedChannelId"]
existed_time_stamp = data_info["ExistedTimeStamp"]

board_id = data_info["BoardId"]
sub_pack_channel_id_int = data_info["SubPackageChannelId"]
sub_pack_trigger_source_stamp = data_info["SubPackageTriggerStamp"]
sub_pack_trigger_source_count = data_info["SubPackageTriggerCount"]

wave_sample_data = data_info["WaveSampleData"]
wave_sample_data_valid = data_info["WaveSampleDataValid"]

internal_event_stamp = trigger_info["EventIntTimeStamp"]
internal_event_stamp_valid = trigger_info["EventIntTimeStampValid"]

#------------------------------------------------------------------------------------------------------

# Read in board channel potision infomation
board_channel_location_relation_file = np.loadtxt('BoardChannelLocationRelation.csv')

board_channel_location_relation = {
    'BoardId': board_channel_location_relation_file[0,:].astype(np.int32),
    'ChannelId': board_channel_location_relation_file[1,:].astype(np.int32),
    'LocationId': board_channel_location_relation_file[2,:].astype(np.int32),
    'State': board_channel_location_relation_file[3,:].astype(np.bool_), # True for L and False for C
}

#print(board_channel_location_relation)

#find maxima number of packages in an event
num_hit_event = np.zeros(internal_event_stamp.shape[0]) # number of hits in a event
for event_idx in range(internal_event_stamp.shape[0]):
    event_time_stamps = internal_event_stamp[event_idx,:][internal_event_stamp_valid[event_idx,:]]
    for time_stamp in event_time_stamps:
        pack_idxs = pack_pointer_board_channel_timeStamp[:,:,np.where(existed_time_stamp==time_stamp)][pack_pointer_board_channel_timeStamp_valid[:,:,np.where(existed_time_stamp==time_stamp)]].flatten()
        num_hit_event[event_idx] += pack_idxs.shape[0]
num_hit_event = num_hit_event.astype(np.int32)

#-----------------------------------------------------------------------------------------------
x_hits = np.zeros((internal_event_stamp.shape[0],np.max(num_hit_event)))# x position of hits in a event
y_hits = np.zeros((internal_event_stamp.shape[0],np.max(num_hit_event)))# y position of hits in a event
time_stamp_hits = np.zeros((internal_event_stamp.shape[0],np.max(num_hit_event)))# time stampe hits in a event

# true mean the index is used
x_hits_valid = np.zeros((internal_event_stamp.shape[0],np.max(num_hit_event)),dtype = np.bool_)
y_hits_valid = np.zeros((internal_event_stamp.shape[0],np.max(num_hit_event)),dtype = np.bool_)
time_stamp_valid = np.zeros((internal_event_stamp.shape[0],np.max(num_hit_event)),dtype = np.bool_)

for event_idx in range(internal_event_stamp.shape[0]):
    event_time_stamps = internal_event_stamp[event_idx,:][internal_event_stamp_valid[event_idx,:]]
    idx = 0
    for time_stamp in event_time_stamps:
        pack_idxs = pack_pointer_board_channel_timeStamp[:,:,np.where(existed_time_stamp==time_stamp)][pack_pointer_board_channel_timeStamp_valid[:,:,np.where(existed_time_stamp==time_stamp)]].flatten()
        #print(pack_idxs)
        for pack_idx in pack_idxs:
            output_data = wave_sample_data[pack_idx][wave_sample_data_valid[pack_idx]]

            this_channel_id = sub_pack_channel_id_int[pack_idx]
            this_board_id = board_id[0,pack_idx]

            convert_idx = np.logical_and(board_channel_location_relation['BoardId'] == this_board_id, 
                                         board_channel_location_relation['ChannelId'] == this_channel_id, 
                                         )#index in convert csv
            if not convert_idx.any() : continue
            location = board_channel_location_relation['LocationId'][convert_idx]
            state = board_channel_location_relation['State'][convert_idx]

            #rint(location)
            #print(board_channel_location_relation['BoardId'] == this_board_id)
            #print(board_channel_location_relation['ChannelId'] == this_channel_id)
            

            if state:
                y_hits[event_idx,idx] = location
                y_hits_valid[event_idx,idx] = True
            else:
                x_hits[event_idx,idx] = location
                x_hits_valid[event_idx,idx] = True

            time_stamp_hits[event_idx,idx] = time_stamp + np.argmax(output_data) - time_stamp_offset
            time_stamp_valid[event_idx,idx] = True
            
            idx+=1

#test code
idx_test = 3
print(f'x signal: {x_hits[idx_test][x_hits_valid[idx_test,:]]} at timeestampe {time_stamp_hits[idx_test][x_hits_valid[idx_test,:]]}')
print(f'y signal: {y_hits[idx_test][y_hits_valid[idx_test,:]]} at timeestampe {time_stamp_hits[idx_test][y_hits_valid[idx_test,:]]}')
event_existed_time_stamp = np.unique(time_stamp_hits[idx_test][time_stamp_valid[idx_test]])
idx_time_stamp_test = 1

print(y_hits[idx_test][y_hits_valid[idx_test,:]][time_stamp_hits[idx_test][y_hits_valid[idx_test,:]]==event_existed_time_stamp[idx_time_stamp_test]])
print(x_hits[idx_test][x_hits_valid[idx_test,:]][time_stamp_hits[idx_test][x_hits_valid[idx_test,:]]==event_existed_time_stamp[idx_time_stamp_test]])

plt.figure()
#plt.hlines(y_hits[idx_test][y_hits_valid[idx_test,:]][time_stamp_hits[idx_test][y_hits_valid[idx_test,:]]==event_existed_time_stamp[idx_time_stamp_test]],0,64)
#plt.vlines(x_hits[idx_test][x_hits_valid[idx_test,:]][time_stamp_hits[idx_test][x_hits_valid[idx_test,:]]==event_existed_time_stamp[idx_time_stamp_test]],0,64)

plt.hlines(y_hits[idx_test][y_hits_valid[idx_test,:]],0,64)
plt.vlines(x_hits[idx_test][x_hits_valid[idx_test,:]],0,64)
plt.savefig('output/hit_test.png')

for idx_test in range(100):
    idx_time_stamp_test = 0
    event_existed_time_stamp = np.unique(time_stamp_hits[idx_test][time_stamp_valid[idx_test]])
    '''
    plt.figure()
    plt.hlines(y_hits[idx_test][y_hits_valid[idx_test,:]][time_stamp_hits[idx_test][y_hits_valid[idx_test,:]]==event_existed_time_stamp[idx_time_stamp_test]],0,64)
    plt.vlines(x_hits[idx_test][x_hits_valid[idx_test,:]][time_stamp_hits[idx_test][x_hits_valid[idx_test,:]]==event_existed_time_stamp[idx_time_stamp_test]],0,64)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'output/hit_test/hit_test{idx_test}_xy.png')
    plt.close()
    '''
    fig,axs = plt.subplots(nrows=1,ncols=2,sharey=True)
    axs[0].plot(y_hits[idx_test][y_hits_valid[idx_test,:]],time_stamp_hits[idx_test][y_hits_valid[idx_test,:]]-np.min(time_stamp_hits[idx_test][time_stamp_valid[idx_test]]),'.',color = 'c')
    axs[0].set_xlabel('y')
    axs[0].set_ylabel('dt')

    axs[1].plot(x_hits[idx_test][x_hits_valid[idx_test,:]],time_stamp_hits[idx_test][x_hits_valid[idx_test,:]]-np.min(time_stamp_hits[idx_test][time_stamp_valid[idx_test]]),'.',color = 'r')
    axs[1].set_xlabel('x')
    
    plt.savefig(f'output/hit_test/hit_test{idx_test}_xyt.png')
    plt.close()