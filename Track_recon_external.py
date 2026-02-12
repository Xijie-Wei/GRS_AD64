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

data_info,trigger_info = UnpackPackage("data_file/RAW_data_20251226_151216.bin")

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

event_num = trigger_info["EventNum"]
event_ext_trig_counts_num = trigger_info["EventExtTriggerCountNum"]
event_ext_trig_counts = trigger_info["EventExtTriggerCount"]
event_ext_trig_stamp = trigger_info["EventExtTriggerTimeStamp"]
event_ext_trig_stamp_exceed = trigger_info["EventExtTriggerTimeStampExcceed"]

ext_tri_count = trigger_info["ExistedExtTriggerCount"]
ext_tri_source_stamp = trigger_info["ExistedExtTriggerStamp"]

#------------------------------------------------------------------------------------------------------

#print(event_ext_trig_stamp.shape)
#print(event_ext_trig_counts)

# Read in board channel potision infomation
board_channel_location_relation_file = np.loadtxt('BoardChannelLocationRelation.csv')

board_channel_location_relation = {
    'BoardId': board_channel_location_relation_file[0,:].astype(np.int32),
    'ChannelId': board_channel_location_relation_file[1,:].astype(np.int32),
    'LocationId': board_channel_location_relation_file[2,:].astype(np.int32),
    'Group': board_channel_location_relation_file[3,:].astype(np.int32),
    'State': board_channel_location_relation_file[4,:].astype(np.bool_), # True for L and False for C
}

#print(board_channel_location_relation)

#find maxima number of packages in an event
num_hit_event = np.zeros(event_num) # number of hits in a event
for event_idx in range(event_num):
    for this_count in event_ext_trig_counts[event_idx,0:event_ext_trig_counts_num[event_idx]]:
        pack_idxs = np.where(sub_pack_trigger_source_count == this_count)[0]
        num_hit_event[event_idx] += pack_idxs.shape[0]
num_hit_event = num_hit_event.astype(np.int32)

#-----------------------------------------------------------------------------------------------
x_hits = np.zeros((event_num,np.max(num_hit_event),6,3))*np.nan# x position of hits in a event
y_hits = np.zeros((event_num,np.max(num_hit_event),6,3))*np.nan# y position of hits in a event
T0 = np.zeros(event_num)

# true mean the index is used
x_hits_valid = np.zeros((event_num,np.max(num_hit_event)),dtype = np.bool_)
y_hits_valid = np.zeros((event_num,np.max(num_hit_event)),dtype = np.bool_)

for event_idx in range (event_num):
    idx = 0
    
    for this_count in event_ext_trig_counts[event_idx,0:event_ext_trig_counts_num[event_idx]]:
        pack_idxs = np.where(sub_pack_trigger_source_count == this_count)[0]
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
            group = board_channel_location_relation['Group'][convert_idx]
            #rint(location)
            #print(board_channel_location_relation['BoardId'] == this_board_id)
            #print(board_channel_location_relation['ChannelId'] == this_channel_id)

            time_stamp = ext_tri_source_stamp[ext_tri_count == sub_pack_trigger_source_count[pack_idx]] * 0.1


            if np.max(output_data) == 4095:
                Time_wave_peak = np.round(np.mean(np.where(output_data==4095)[0])).astype(np.int32)
            else: Time_wave_peak = np.argmax(output_data)
            #Time_wave_peak = 0

            if state:
                y_hits[event_idx,idx,:,1] = location
                y_hits[event_idx,idx,:,0] = (np.array([group,group+7,group+14,group+21,group+28,group+35]) - 1)[:,0]
                y_hits[event_idx,idx,:,2] = time_stamp + Time_wave_peak - time_stamp_offset
                y_hits_valid[event_idx,idx] = True
            else:
                x_hits[event_idx,idx,:,0] = location
                x_hits[event_idx,idx,:,1] = (np.array([group,group+7,group+14,group+21,group+28,group+35]) - 1)[:,0]
                x_hits[event_idx,idx,:,2] = time_stamp + Time_wave_peak - time_stamp_offset
                x_hits_valid[event_idx,idx] = True
            
            idx+=1
    
    T0[event_idx] = np.nanmin((x_hits[event_idx,:,:,2],y_hits[event_idx,:,:,2]))
    x_hits[event_idx,:,:,2] -= T0[event_idx]
    y_hits[event_idx,:,:,2] -= T0[event_idx]

#test
t = 0
dt = 20
for TesEventId in range(100):
    fig,axs = plt.subplots(1,2,sharey=True)

    axs[0].plot(x_hits[TesEventId,x_hits_valid[TesEventId],:,0],x_hits[TesEventId,x_hits_valid[TesEventId],:,2],'.')
    axs[0].axhline(t,ls = '-',alpha = 0.3)
    axs[0].axhline(t+dt,ls = '-',alpha = 0.3)
    axs[0].set_ylabel('dt')
    axs[0].set_xlabel('x')

    axs[1].plot(y_hits[TesEventId,y_hits_valid[TesEventId],:,1],y_hits[TesEventId,y_hits_valid[TesEventId],:,2],'.')
    axs[1].axhline(t,ls = '-',alpha = 0.3)
    axs[1].axhline(t+dt,ls = '-',alpha = 0.3)
    axs[1].set_xlabel('y')
    plt.savefig(f'output/hit_test/EventId{TesEventId}')
    plt.close()

    #print(np.logical_and([x_hits[TesEventId,x_hits_valid[TesEventId],:,2]<= dt+50],[x_hits[TesEventId,x_hits_valid[TesEventId],:,2]>=dt]).shape)
    plt.figure(figsize= [5,5])
    plt.plot(x_hits[TesEventId,x_hits_valid[TesEventId],:,0][np.logical_and([x_hits[TesEventId,x_hits_valid[TesEventId],:,2]<= dt+t],[x_hits[TesEventId,x_hits_valid[TesEventId],:,2]>=t])[0]],
             x_hits[TesEventId,x_hits_valid[TesEventId],:,1][np.logical_and([x_hits[TesEventId,x_hits_valid[TesEventId],:,2]<= dt+t],[x_hits[TesEventId,x_hits_valid[TesEventId],:,2]>=t])[0]],'.',color='c')
    plt.plot(y_hits[TesEventId,y_hits_valid[TesEventId],:,0][np.logical_and([y_hits[TesEventId,y_hits_valid[TesEventId],:,2]<= dt+t],[y_hits[TesEventId,y_hits_valid[TesEventId],:,2]>=t])[0]],
             y_hits[TesEventId,y_hits_valid[TesEventId],:,1][np.logical_and([y_hits[TesEventId,y_hits_valid[TesEventId],:,2]<= dt+t],[y_hits[TesEventId,y_hits_valid[TesEventId],:,2]>=t])[0]],'+',color='r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-2,43])
    plt.ylim([-2,43])
    plt.xticks([0,5,10,15,20,25,30,35,40])
    plt.yticks([0,5,10,15,20,25,30,35,40])
    plt.grid(visible = True)
    plt.savefig(f'output/hit_test/EventId{TesEventId}_xy')
    plt.close()