import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
from tqdm import tqdm
#from scipy.signal import find_peaks
from scipy.stats import norm
#from sklearn.mixture import GaussianMixture
#import ruptures as rpt
from sklearn.cluster import DBSCAN
import ROOT
from lmfit.models import GaussianModel

plt.rcParams['text.usetex'] = True

event_max_interval = 1000# Maxima time allowed for two external trigger file to be considered in same event(unit 2.5ns)
interval_trigger_interval = 20# Maxima time allowed for two Inrtnal trigger file to be considered in same event(unit 25ns)
inte_range = np.array([-5,6])#area used to calculate integartion
bg_range = 20# use wavedatapoint[0:bg_range] to calculate background
use_external_noise_file = True # set True if use external noise file, (stored in Noise_level/)

#load file 
file_name = "data_file/RAW_data_20251229_184323.bin"
file = np.fromfile(file_name,dtype=np.uint8)# common decoding
file_head = np.fromfile(file_name,dtype=(np.void,4))# for finding general package head

idx_head = np.where(file_head==np.array([b'\x1E'b'\xAD'b'\xC0'b'\xDE'],dtype=(np.void,4)))*np.array([4])# index of head of package
idx_end = np.where(file_head==np.array([b'\x5A'b'\x5A'b'\x5A'b'\x5A'],dtype=(np.void,4)))*np.array([4])# index of end of package
print(f"{idx_head.shape[1]} package heading found")
print(f"{idx_end.shape[1]} package end found")

valid_length = np.min([idx_head.shape[1],idx_end.shape[1]])

# validation check
pack_info = np.unpackbits(np.stack((file[idx_head+np.array([6])].flatten(),file[idx_head+np.array([7])].flatten()),axis=1),axis=1)
P = pack_info[:,0]# even odd 1=> last 4bytes all 0; 0=> last 4bytes all data
E = pack_info[:,1]# pack error 1=> packaing error 0; 0=> no error
pack_length = pack_info[:,4:16]# package_length 12bit
convert_matr = np.logspace(11,0,num=12,base=2)# materix convert 12bit to int
pack_length_int = np.sum(pack_length*convert_matr,axis =1)# package_length int
length_check = np.equal(pack_length_int[0:valid_length]*8,(idx_end[:,0:valid_length]-idx_head[:,0:valid_length]+4))

if idx_head.shape[1] != idx_end.shape[1]: print("Warning: dismatch between number of package headings and package ends, ignore the last heading")
idx_head = idx_head[:,0:valid_length]
idx_end = idx_end[:,0:valid_length]
#print(idx_head.shape)
#print(idx_end.shape)


print(f"Packing check: {E}")
print(f"Found pack at {np.where(E.astype(bool))[0]} with error, total {np.count_nonzero(np.logical_not(E.astype(bool)))} packages pass packing check")
print(f"Length check: {length_check}")
print(f"Found pack at {np.where(np.logical_not(length_check))[1]} with error, total {np.count_nonzero(length_check)} packages pass length check")
# unpack package detail
board_id = file[idx_head+np.array([9])]
pack_id = file[idx_head+np.array([10])]
#print(pack_id)
#print(board_id)

existed_board_id = np.unique(board_id)
print(f"Board with ID {existed_board_id} founded in the file")
existed_pack_id = np.unique(pack_id)
print(f"Package with ID {existed_pack_id} founded in the file")

# Open external noise file(if needed)
if use_external_noise_file:
    external_noise_mean = np.zeros([existed_board_id.shape[0],64])
    external_noise_sigma = np.zeros([existed_board_id.shape[0],64])
    for external_noise_board_id in range(existed_board_id.shape[0]):
        external_noise_mean[external_noise_board_id] = np.loadtxt(f'Noise_level/Board{existed_board_id[external_noise_board_id]}_noise_mean')
        external_noise_sigma[external_noise_board_id] = np.loadtxt(f'Noise_level/Board{existed_board_id[external_noise_board_id]}_noise_sigma')

existed_board_id_new = np.zeros(existed_board_id.shape[0])
for idx,new_id in enumerate(np.array([254,18,16,19,2,5,17,15,28,13])):
    existed_board_id_new[existed_board_id == new_id] = idx
print(f"New id applied {existed_board_id_new}({existed_board_id_new.shape[0]})")

print("Relation ship between old and new ids")
for idx in range(existed_board_id_new.shape[0]):
    print(f"{existed_board_id[idx]} -> {existed_board_id_new[idx]}")

# find maxima repeat number of package ids
pack_id_repeat = np.zeros([existed_board_id.shape[0],existed_pack_id.shape[0]])
for idx_baord_id in range(0,existed_board_id.shape[0]):
    address =  np.where(board_id == existed_board_id[idx_baord_id])
    for idx_pack_id in range(0,existed_pack_id.shape[0]):
        repeated_id = np.where(pack_id[address] == existed_pack_id[idx_pack_id])
        pack_id_repeat[idx_baord_id,idx_pack_id] = len(repeated_id[0])
        
pack_id_count = int(np.max(pack_id_repeat))
print(f"Maxima {pack_id_count} repeats of pack id found")


genral_pack_pointer = np.zeros([existed_board_id.shape[0],existed_pack_id.shape[0],pack_id_count])
# Pointer pointing idx_head, idx_head[genral_pack_pointer[i,j,k]] means address of package with board id i and package id j and k-th repeat in file  
genral_pack_pointer_valid = np.zeros([existed_board_id.shape[0],existed_pack_id.shape[0],pack_id_count],dtype=bool)

# Find pointer value
for idx_baord_id in range(0,existed_board_id.shape[0]):
    #print(existed_board_id[idx_baord_id])
    address =  np.where(board_id == existed_board_id[idx_baord_id])
    for idx_pack_id in range(0,existed_pack_id.shape[0]):
        pack_address = np.where(pack_id[0,address[1]] == existed_pack_id[idx_pack_id])
        genral_pack_pointer[idx_baord_id,idx_pack_id,0:len(pack_address[0])] = address[1][pack_address[0][0:len(pack_address[0])]]
        genral_pack_pointer_valid[idx_baord_id,idx_pack_id,0:len(pack_address[0])] = True

genral_pack_pointer = genral_pack_pointer.astype(np.int32)
#print(genral_pack_pointer)
#print(genral_pack_pointer_valid)
#test code
"""
#test board id validation
print(existed_board_id)
test_id = idx_head[0,genral_pack_pointer[9,0,0][genral_pack_pointer_valid[9,0,0]]][0]
print(test_id)
print(file[test_id:test_id+20])

#test valid function validation
print(genral_pack_pointer[1,0,:][genral_pack_pointer_valid[1,0,:]])
"""

# ! imnportant as external trigger pack would only be the first 1to 3 packs. no specicic procress is taken here.
sub_pack_head = np.unpackbits(file[idx_head+np.array([12])].T,axis=1)
# package type
sub_pack_type = sub_pack_head[:,0:2]
if_data_package = np.logical_and(np.equal(sub_pack_type,np.array([1,0]))[:,0],np.equal(sub_pack_type,np.array([1,0]))[:,1])# true if the package is a data package else a trigger package
print(f"Package at {np.where(np.logical_not(if_data_package))[0]} are external trigger package ({np.where(np.logical_not(if_data_package))[0].shape[0]} packages)")
# channel number
sub_pack_channel_id = sub_pack_head[:,2:8]
convert_matr2 = np.logspace(5,0,num=6,base=2)# materix convert 6bit to int
sub_pack_channel_id_int = np.sum(sub_pack_channel_id * convert_matr2,axis=1).astype(np.int32)

sub_pack_length = file[idx_head+np.array([13])]*np.array([16])
sub_pack_id = file[idx_head+np.array([14])]*np.array([256])+file[idx_head+np.array([15])]

# trigger souce
convert_matr3 = np.logspace(2,0,num=3,base=2)# materix convert 3bit to int
sub_pack_trigger_source = np.sum(np.unpackbits(file[idx_head+np.array([19])].T,axis=1)[:,5:8] * convert_matr3,axis=1).astype(np.int32)

# trigger souce count
sub_pack_trigger_source_count = (file[idx_head+np.array([20])]*np.array([256])+file[idx_head+np.array([21])])[0]
#print(sub_pack_trigger_source_count)
print(f"Trigger source count of {np.unique(sub_pack_trigger_source_count)} found")

# trigger souce time stamp
convert_matr4 = np.logspace(2*4*5,0,num=6,base=2)# materix convert 6byte to int
#print(convert_matr4)
sub_pack_trigger_source_stamp = np.sum(np.array([
                                          file[idx_head+np.array([22])][0],
                                          file[idx_head+np.array([23])][0],
                                          file[idx_head+np.array([24])][0],
                                          file[idx_head+np.array([25])][0],
                                          file[idx_head+np.array([26])][0],
                                          file[idx_head+np.array([27])][0]
                                          ]).T*convert_matr4,axis=1).astype(np.int64)
#sub_pack_trigger_source_stamp = np.array([hex(stamp) for stamp in sub_pack_trigger_source_stamp])
#print(sub_pack_trigger_source_stamp.T)
#print(sub_pack_trigger_source_stamp.shape)

wave_sample_data = np.zeros([idx_end.shape[1],np.max(sub_pack_length)]) 
wave_sample_data_valid = np.zeros([idx_end.shape[1],np.max(sub_pack_length)],dtype=bool) 
#print(wave_sample_data.shape)
for idx_sub_pack in range(idx_head.shape[1]):
    if not if_data_package[idx_sub_pack] : continue
    this_sub_pack_length = sub_pack_length[:,idx_sub_pack][0]*2
    if P[idx_sub_pack].astype(bool): this_data = file[idx_head[:,idx_sub_pack][0]+28:idx_head[:,idx_sub_pack][0]+24+this_sub_pack_length]
    else: this_data = file[idx_head[:,idx_sub_pack][0]+28:idx_head[:,idx_sub_pack][0]+28+this_sub_pack_length]
    this_data_front = this_data[::2]
    this_data_rear = this_data[1::2]
    wave_sample_data[idx_sub_pack,0:int(this_sub_pack_length/2)] = this_data_front*np.array([256])+this_data_rear
    wave_sample_data_valid[idx_sub_pack,0:int(this_sub_pack_length/2)] = True

#print(wave_sample_data)

#test code
"""
idx = genral_pack_pointer[1,1,0][genral_pack_pointer_valid[1,1,0]]
plt.plot(wave_sample_data[idx][wave_sample_data_valid[idx]])
plt.xlabel("Time")
plt.ylabel("Data")
plt.title(f"Board id: {board_id[0,idx][0]},channel id: {sub_pack_channel_id_int[idx][0]},time stamp: {sub_pack_trigger_source_stamp[idx[0]]}")
plt.savefig("output")
"""

#now use board id, channel id and time stamp to label a package
existed_channel_id = np.unique(sub_pack_channel_id_int)
existed_time_stamp = np.unique(sub_pack_trigger_source_stamp)
print(f"Channel id of {existed_channel_id} found")
print(f"Time stamp of {existed_time_stamp} found, ({existed_time_stamp.shape[0]} time stamps)")

#this pointer uses board id, channel if and time stamp to label a package
pack_pointer_board_channel_timeStamp = np.zeros([existed_board_id.shape[0],existed_channel_id.shape[0],existed_time_stamp.shape[0]],dtype=np.int32)
pack_pointer_board_channel_timeStamp_valid = np.zeros([existed_board_id.shape[0],existed_channel_id.shape[0],existed_time_stamp.shape[0]],dtype=bool)
#as we do not expect more than 3 external trigger pack found 
ext_tri = np.zeros(np.where(np.logical_not(if_data_package))[0].shape[0]).astype(np.int32)
num_ext_tri = 0

for idx_board in range(existed_board_id.shape[0]):
    print(f'Processing Board id {existed_board_id[idx_board]} ({idx_board+1}/{existed_board_id.shape[0]})')
    for idx_pack in tqdm(range(existed_pack_id.shape[0])):
        for idx_repeat in range(genral_pack_pointer.shape[2]):
            #print(f"{idx_board},{idx_pack},{idx_repeat}")
            if not genral_pack_pointer_valid[idx_board,idx_pack,idx_repeat]: continue
            file_pointer = genral_pack_pointer[idx_board,idx_pack,idx_repeat]
            if not length_check[0,file_pointer]: continue
            if P[file_pointer].astype(bool): continue
            if not if_data_package[file_pointer]: 
                ext_tri[num_ext_tri] = file_pointer
                num_ext_tri += 1
                continue
            #print(file_pointer)
            this_channel = sub_pack_channel_id_int[file_pointer]
            this_timeStamp = sub_pack_trigger_source_stamp[file_pointer]
            #print(np.where(existed_channel_id==this_channel)[0])
            #print(np.where(existed_time_stamp==this_timeStamp)[0])
            pack_pointer_board_channel_timeStamp[idx_board,
                                                np.where(existed_channel_id==this_channel)[0].astype(np.int32),
                                                np.where(existed_time_stamp==this_timeStamp)[0].astype(np.int32)
                                                ] = file_pointer
            pack_pointer_board_channel_timeStamp_valid[idx_board,
                                                       np.where(existed_channel_id==this_channel)[0].astype(np.int32),
                                                       np.where(existed_time_stamp==this_timeStamp)[0].astype(np.int32)
                                                       ] = True

print(f"Number of valid data package pointer: {np.count_nonzero(pack_pointer_board_channel_timeStamp_valid)}")

# process ext tri pack(if found)
if num_ext_tri != 0:
    print(f"{num_ext_tri} external trigger package found")
    exceed=np.zeros(num_ext_tri)
    ext_tri_count=np.zeros(num_ext_tri)
    ext_tri_stamp_exceed=np.zeros(num_ext_tri)
    ext_tri_source_stamp=np.zeros(num_ext_tri)
    for idx_tri in range(num_ext_tri):
        #print(f"Raw package{file[idx_head[0,ext_tri[idx_tri]]:idx_head[0,ext_tri[idx_tri]]+32]}")
        #print(ext_tri[idx_tri])
        exceed[idx_tri] = np.unpackbits(file[idx_head[0,ext_tri[idx_tri]]+13])[-1]
        ext_tri_count[idx_tri] = file[idx_head[0,ext_tri[idx_tri]]+14]*256 + file[idx_head[0,ext_tri[idx_tri]]+15]
        ext_tri_stamp_exceed[idx_tri] = np.unpackbits(file[idx_head[0,ext_tri[idx_tri]]+21])[-1]
        ext_tri_source_stamp[idx_tri] = sub_pack_trigger_source_stamp[ext_tri[idx_tri]]
        print(f"External trigger {idx_tri+1}. Trigger count: {ext_tri_count[idx_tri]} (exceed:{exceed[idx_tri]}),Trigger time stamp: {ext_tri_source_stamp[idx_tri]}(exceed:{ext_tri_stamp_exceed[idx_tri]})")
    #print(ext_tri_source_stamp)
    
    # event finder
    event_ext_trig_counts = np.zeros([num_ext_tri,3])*np.nan
    event_ext_trig_counts_num = np.zeros(num_ext_tri,dtype=np.int32)
    event_ext_trig_stamp = np.zeros(num_ext_tri)*np.nan
    event_ext_trig_stamp_exceed = np.zeros(num_ext_tri)*np.nan
    event_ext_trig_stamp[0] = ext_tri_source_stamp[0]
    event_ext_trig_stamp_exceed[0] = ext_tri_stamp_exceed[0]
    event_ext_trig_counts[0,0] = ext_tri_count[0]
    event_ext_trig_counts_num[0] = 1
    for idx_tri in range(num_ext_tri):
        if np.nanmin(np.abs(ext_tri_source_stamp[idx_tri]-event_ext_trig_stamp)) > event_max_interval: 
            event_ext_trig_stamp[idx_tri] = ext_tri_source_stamp[idx_tri]
            event_ext_trig_stamp_exceed[idx_tri] = ext_tri_stamp_exceed[idx_tri]
            event_ext_trig_counts[idx_tri,0] = ext_tri_count[idx_tri]
            event_ext_trig_counts_num[idx_tri] = 1
        else:
            if event_ext_trig_counts_num[np.nanargmin(np.abs(ext_tri_source_stamp[idx_tri]-event_ext_trig_stamp))] < 2:
                #print(event_ext_trig_counts_num[np.nanargmin(np.abs(ext_tri_source_stamp[idx_tri]-event_ext_trig_stamp))])
                event_ext_trig_counts[np.nanargmin(np.abs(ext_tri_source_stamp[idx_tri]-event_ext_trig_stamp)),event_ext_trig_counts_num[np.nanargmin(np.abs(ext_tri_source_stamp[idx_tri]-event_ext_trig_stamp))]] = ext_tri_count[idx_tri]
                #print(event_ext_trig_counts[np.nanargmin(np.abs(ext_tri_source_stamp[idx_tri]-event_ext_trig_stamp)),:])
                event_ext_trig_counts_num[np.nanargmin(np.abs(ext_tri_source_stamp[idx_tri]-event_ext_trig_stamp))] += 1 
            else:
                event_ext_trig_stamp[idx_tri] = ext_tri_source_stamp[idx_tri]
                event_ext_trig_stamp_exceed[idx_tri] = ext_tri_stamp_exceed[idx_tri]
                event_ext_trig_counts[idx_tri,0] = ext_tri_count[idx_tri]
                event_ext_trig_counts_num[idx_tri] = 1
    event_ext_trig_counts_num = event_ext_trig_counts_num[~np.isnan(event_ext_trig_counts[:,0])]
    event_ext_trig_counts = event_ext_trig_counts[~np.isnan(event_ext_trig_counts[:,0]),:]
    event_ext_trig_stamp = event_ext_trig_stamp[~np.isnan(event_ext_trig_stamp)]
    event_ext_trig_stamp_exceed = event_ext_trig_stamp_exceed[~np.isnan(event_ext_trig_stamp_exceed)]
    #print(filtered_ext_trig_stamp)
    #print(filtered_ext_trig_stamp.shape)
    #print(filtered_ext_trig_count.shape)
    #print(filtered_ext_trig_exceed.shape)
    #print(filtered_ext_trig_sstamp_exceed.shape)
    event_num = event_ext_trig_counts.shape[0]
    print(f"{event_num} events found")

else: 
    print("No external trigger package found")
    # event finder for internal trigger
    '''
    stamps_intervals = np.diff(existed_time_stamp)
    cumulative_interval = 0
    stamp_marks = [0]
    for this_idx,stamp_interval in enumerate(stamps_intervals):
        if stamp_interval > interval_trigger_interval / 1.2:
            stamp_marks.append(this_idx+1)
            cumulative_interval = 0
        else:
            cumulative_interval += stamp_interval
        if cumulative_interval > interval_trigger_interval:
            stamp_marks.append(this_idx+1)
            cumulative_interval = 0
    stamp_marks.append(this_idx+1)
    stamp_marks = np.array(stamp_marks)
    '''
    '''
    algo = rpt.Pelt(model="l2",min_size=2).fit(np.diff(existed_time_stamp))  
    stamp_marks = np.append(0,np.array(algo.predict(pen=10)))
    print(stamp_marks)
    '''    
    '''
    internal_event_stamp = np.zeros([stamp_marks.shape[0]-1,np.max(np.diff(stamp_marks))])*np.nan
    internal_event_stamp_valid = np.zeros([stamp_marks.shape[0]-1,np.max(np.diff(stamp_marks))],dtype = bool)
    for this_idx,stamp_mark in enumerate(stamp_marks[1:]):
        internal_event_stamp[this_idx,0:(stamp_mark-stamp_marks[this_idx])] = existed_time_stamp[stamp_marks[this_idx]:stamp_mark]
        internal_event_stamp_valid[this_idx,0:stamp_mark-stamp_marks[this_idx]]  = True
    '''
    clustering = DBSCAN(eps=interval_trigger_interval, min_samples=2).fit(np.reshape(existed_time_stamp,(-1,1)))
    print(clustering.labels_)
    existed_labels,counts = np.unique(clustering.labels_ ,return_counts=True )
    internal_event_stamp = np.zeros([existed_labels.shape[0],np.max(counts)])*np.nan
    internal_event_stamp_valid = np.zeros([existed_labels.shape[0],np.max(counts)],dtype = bool)
    for idx,label in enumerate(existed_labels):
        internal_event_stamp[idx,0:counts[idx]] = existed_time_stamp[clustering.labels_  == label]
        internal_event_stamp_valid[idx,0:counts[idx]] = True
    print(f'{internal_event_stamp.shape[0]} event found')
    #print(internal_event_stamp)
    #print(internal_event_stamp_valid)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# calculate area under line for internal trigger
area_under_line = np.zeros(internal_event_stamp.shape[0])
valid_package_count = np.zeros(internal_event_stamp.shape[0])
valid_channel_count = np.zeros([internal_event_stamp.shape[0],64])
print_max = []
for event_idx in range(internal_event_stamp.shape[0]):
    event_time_stamps = internal_event_stamp[event_idx,:][internal_event_stamp_valid[event_idx,:]]
    for time_stamp in event_time_stamps:
        pack_idxs = pack_pointer_board_channel_timeStamp[:,:,np.where(existed_time_stamp==time_stamp)][pack_pointer_board_channel_timeStamp_valid[:,:,np.where(existed_time_stamp==time_stamp)]].flatten()
        #print(pack_idxs)
        for pack_idx in pack_idxs:
            output_data = wave_sample_data[pack_idx][wave_sample_data_valid[pack_idx]]
            if use_external_noise_file:
                bg = external_noise_mean[np.where(existed_board_id==board_id[0,pack_idx]),sub_pack_channel_id_int[pack_idx]]
                bg_std = external_noise_sigma[np.where(existed_board_id==board_id[0,pack_idx]),sub_pack_channel_id_int[pack_idx]]
            else:
                bg = np.mean(output_data[0:bg_range])
                bg_std = np.std(output_data[0:bg_range])
            #if np.max(output_data) < bg + 10*bg_std:continue
            #found maxima
            if np.max(output_data) == 4095:
                idx_max = np.round(np.mean(np.where(output_data==4095)[0])).astype(np.int32)
            else: idx_max = np.argmax(output_data)

            area = np.mean(output_data[idx_max+inte_range[0]:idx_max+inte_range[1]]) - bg
            if area < 0: continue
            area_under_line[event_idx] += area
            valid_package_count[event_idx] += 1 
            valid_channel_count[event_idx,sub_pack_channel_id_int[np.where(existed_board_id==board_id[0,pack_idx])]] += 1
            print_max.append(np.max(output_data))

#area_under_line = area_under_line[valid_package_count>2]
#alid_channel_count = valid_channel_count[valid_package_count>2,:]
#valid_package_count = valid_package_count[valid_package_count>2]

print(f'Total valid event: {np.count_nonzero(valid_package_count)}')
print(f'Total valid package: {np.sum(valid_package_count)},(average = {np.mean(valid_package_count)})')

fitting_bound1 = [3000,6000]
fitting_bound2 = [1400,2400]

num_bins = 500
(mu,sigma) = norm.fit(area_under_line[np.logical_and(area_under_line>fitting_bound1[0],area_under_line<fitting_bound1[1])])
(mu2,sigma2) = norm.fit(area_under_line[np.logical_and(area_under_line>fitting_bound2[0],area_under_line<fitting_bound2[1])])
fig = plt.figure(figsize=[7,4.8])
ax = fig.add_subplot(111)

ax.hist(area_under_line,density=True,bins = num_bins,histtype='step',range = [0,7000],color="#445D6C80")
#x = np.linspace(1000,9000,1000)
#plt.plot(x,norm.pdf(x,mu,sigma)*np.count_nonzero(area_under_line[np.logical_and(area_under_line>4400,area_under_line<7600)])/area_under_line.shape[0]+
#         norm.pdf(x,mu2,sigma2)*np.count_nonzero(area_under_line[np.logical_and(area_under_line>2000,area_under_line<3800)])/area_under_line.shape[0]
#         ,'--',label = r"$n = n_1 + n_2$"
#         )
peak_main = np.count_nonzero(area_under_line[np.logical_and(area_under_line>fitting_bound1[0],area_under_line<fitting_bound1[1])])/area_under_line.shape[0]
x = np.linspace(fitting_bound1[0]-800,fitting_bound1[1]+800,1000)
normal1 = ax.plot(x,norm.pdf(x,mu,sigma)*peak_main
         ,'--',label = r"$n_1 \sim N(\mu_1,{\sigma_1}^2)$"
         )
#peak_main = np.max(norm.pdf(x,mu,sigma)*peak_main)
peak_main = np.trapz(norm.pdf(x,mu,sigma)*peak_main,x)

peak_excape = np.count_nonzero(area_under_line[np.logical_and(area_under_line>fitting_bound2[0],area_under_line<fitting_bound2[1])])/area_under_line.shape[0]
x = np.linspace(fitting_bound2[0]-800,fitting_bound2[1]+800,1000)
normal2 = ax.plot(x,norm.pdf(x,mu2,sigma2)*peak_excape
         ,'--',label = r"$n_2 \sim N(\mu_2,{\sigma_2}^2)$"
         )
#peak_excape=np.max(norm.pdf(x,mu2,sigma2)*peak_excape)
peak_excape = np.trapz(norm.pdf(x,mu2,sigma2)*peak_excape,x)

plt.xlabel(r"$Q[LSB]$")
plt.ylabel(r"$Number \ density$")


# load simulation data
simu_data = ROOT.TFile.Open('output0.root')
tree = simu_data.Get("TEdep")

Edep = []
#tree.Print()
print(tree.GetEntries())
for entry in tree:
    #print(entry)
    Edep.append(entry.fEdep)

Edep = np.array(Edep)
Edep=Edep[Edep>0]*1e3
#print(Edep.shape)


ax_top = ax.twinx()
ax_top.set_ylabel("$Count$")
ax_top = ax_top.twiny()


ax2 = plt.gca()

ax_top.set_xlabel("$Deposit \ energy \ [keV]$")
ax2.spines["top"].set_color("#000000FF")
ax2.spines["right"].set_color("#000000FF")
G4_simu = ax_top.hist(Edep,bins = 48,range = [0,7200*5.9/mu],histtype='step',color = "#000000FF",label = r"$Geant4 \ data$",align = 'right')
simu_hist,simu_bins = np.histogram(Edep,bins = num_bins)
simu_peak_main =  np.count_nonzero(Edep[np.logical_and(Edep>4,Edep<6)])/Edep.shape[0]
simu_peak_escape =  np.count_nonzero(Edep[np.logical_and(Edep>2,Edep<4)])/Edep.shape[0]

'''
mannul_bg = np.random.rand(round(np.count_nonzero(Edep[np.logical_and(Edep>4,Edep<6)]) * 0.00005 / 0.0006))*5.9
print(mannul_bg)
Edepbg = np.append(Edep,mannul_bg)
G4_simu = ax_top.hist(Edepbg,bins = 64,range = [0,10000*5.9/mu],histtype='step',color = "#787878FF",label = r"$Geant4 \ data + bg$")

simu_peak_main_bg =  np.count_nonzero(Edepbg[np.logical_and(Edepbg>4,Edepbg<6)])/Edepbg.shape[0]
simu_peak_escape_bg =  np.count_nonzero(Edepbg[np.logical_and(Edepbg>2,Edepbg<4)])/Edepbg.shape[0]
'''
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax_top.get_legend_handles_labels()
ax_top.legend(lines + lines2, labels + labels2, loc=0)

print(f"Simu P-p ratio = {simu_peak_main/simu_peak_escape:.2f}")
plt.figtext(0.15,0.8,rf'$\\ Total \ event:{internal_event_stamp.shape[0]} \\ Valid \ event:{np.size(area_under_line[~np.isnan(area_under_line)])} \\ Normal \ Fitting \\ \mu_1  = {mu:.1f} \ \sigma_1 = {sigma:.1f} \\ \mu_2  = {mu2:.1f} \ \sigma_2 = {sigma2:.1f} \\ P-p \ ratio = {peak_main/peak_excape:.2f} \\ P-p \ ratio \ (Simu) = {simu_peak_main/simu_peak_escape:.2f}$')

plt.savefig('output/spectrum.png')

fig = plt.figure(figsize=[7,4.8])
ax = fig.add_subplot(111)

Q_hist,Q_bins = np.histogram(area_under_line,bins = num_bins,range = [0,7000])
Q_bins_center = 0.5*(Q_bins[1:]+Q_bins[:-1])
noise = np.mean(Q_hist[np.logical_not(np.logical_or(
    np.logical_and(Q_bins_center>fitting_bound1[0],Q_bins_center<fitting_bound1[1]),
    np.logical_and(Q_bins_center>fitting_bound2[0],Q_bins_center<fitting_bound2[1]))
    )])
Q_hist = Q_hist - noise
Q_hist[Q_hist<0] = 0

model = GaussianModel()
par = model.guess(Q_hist[np.logical_and(Q_bins_center>fitting_bound1[0],Q_bins_center<fitting_bound1[1])],
                     x=Q_bins_center[np.logical_and(Q_bins_center>fitting_bound1[0],Q_bins_center<fitting_bound1[1])])
fitting1 = model.fit(Q_hist[np.logical_and(Q_bins_center>fitting_bound1[0],Q_bins_center<fitting_bound1[1])],
                     par,
                     x=Q_bins_center[np.logical_and(Q_bins_center>fitting_bound1[0],Q_bins_center<fitting_bound1[1])]
                     )
#print(fitting1.fit_report())

model = GaussianModel()
par = model.guess(Q_hist[np.logical_and(Q_bins_center>fitting_bound2[0],Q_bins_center<fitting_bound2[1])],
                     x=Q_bins_center[np.logical_and(Q_bins_center>fitting_bound2[0],Q_bins_center<fitting_bound2[1])])
fitting2 = model.fit(Q_hist[np.logical_and(Q_bins_center>fitting_bound2[0],Q_bins_center<fitting_bound2[1])],
                     par,
                     x=Q_bins_center[np.logical_and(Q_bins_center>fitting_bound2[0],Q_bins_center<fitting_bound2[1])]
                     )
#print(fitting2.fit_report())
ax.step(Q_bins_center,Q_hist,color="#445D6C80")
peak_main = np.sum(Q_hist[np.logical_and(Q_bins_center>fitting_bound1[0],Q_bins_center<fitting_bound1[1])])
peak_escape = np.sum(Q_hist[np.logical_and(Q_bins_center>fitting_bound2[0],Q_bins_center<fitting_bound2[1])])

ax.plot(Q_bins_center[np.logical_and(Q_bins_center>fitting_bound1[0],Q_bins_center<fitting_bound1[1])],
        fitting1.best_fit,'--')
ax.plot(Q_bins_center[np.logical_and(Q_bins_center>fitting_bound2[0],Q_bins_center<fitting_bound2[1])],
        fitting2.best_fit,'--')

print(peak_main/peak_escape)

#fitting1.plot()
#fitting2.plot()

print(fitting1.params['center'].value)
print(fitting1.params['sigma'].value)
print(fitting2.params['center'].value)
print(fitting2.params['sigma'].value)

plt.savefig('output/spectrum_filtered.png')

'''
plt.figure()
plt.hist(print_max,bins = 48,histtype='step',range = [0,4096])
plt.xlabel(r"$Q[LSB]$")
plt.ylabel(r"$Count$")
plt.savefig('output/pack_maxima.png')
'''

bins = np.linspace(0,20,21)
#print(np.unique(valid_package_count,return_counts=True))
plt.figure()
#plt.hist(np.count_nonzero(valid_channel_count,axis=1),bins=bins,histtype='step',align='right')
plt.hist(valid_package_count,bins=bins,histtype='step',align='left')
plt.xticks(bins)
plt.xlabel(r"$Number \ of \ package \ in \ an \ event$")
plt.ylabel(r"$Count$")
plt.savefig('output/package_used.png')




'''
bins = np.linspace(0,63,64)
plt.figure()
plt.hist(sub_pack_channel_id_int[board_id[0,:] == 254],bins = bins)
#plt.xticks(bins)
plt.xlabel(r"$Channel \ of \ package$")
plt.ylabel(r"$Count$")
plt.savefig('output/channel_used_254.png')
'''
'''
bins = np.linspace(0,64,65)
for this_board_id in existed_board_id:
    hist,_ = np.histogram(sub_pack_channel_id_int[board_id[0,:] == this_board_id],bins = bins)
    #print(hist)
    np.savetxt(f'output/Board{this_board_id}_channel_count',np.round((np.stack((bins[0:64],hist),axis = -1))),header ='ChannelID Count',fmt = ['%i','%i'])
'''

"""
#find time difference with a external trigger
area_under_line = np.zeros(event_num)
valid_pack_count = np.zeros(event_num)
num_pack = np.zeros(event_num)
difference_reco = np.zeros(event_num)
for event_idx in tqdm(range (event_num)):
    #test_ext_tri_idx = 0
    #print(f'{event_idx+1}: External trigger counts considered in same event: {event_ext_trig_counts[event_idx,0:event_ext_trig_counts_num[event_idx]]}')
    difference = np.array([])

    for this_count in event_ext_trig_counts[event_idx,0:event_ext_trig_counts_num[event_idx]]:
        pack_idxs = np.where(sub_pack_trigger_source_count == this_count)[0]
        pack_idxs_valid = if_data_package[pack_idxs]
        difference = np.append(difference, event_ext_trig_stamp[event_idx] * 2.5 - sub_pack_trigger_source_stamp[pack_idxs][if_data_package[pack_idxs]] * 25)
        for pack_idx in pack_idxs:
            output_data = wave_sample_data[pack_idx][wave_sample_data_valid[pack_idx]]
            if np.size(output_data) == 0: continue
            #if np.max(output_data) < 300: continue
            #print(output_data)
            if use_external_noise_file:
                bg = external_noise_mean[np.where(existed_board_id==board_id[0,pack_idx]),sub_pack_channel_id_int[pack_idx]]
                bg_std = external_noise_sigma[np.where(existed_board_id==board_id[0,pack_idx]),sub_pack_channel_id_int[pack_idx]]
            else:
                bg = np.mean(output_data[0:bg_range])
                bg_std = np.std(output_data[0:bg_range])
            #print(bg)
            #print(bg_std)
            #found maxima
            if np.max(output_data) == 4095:
                idx_max = np.round(np.mean(np.where(output_data==4095)[0])).astype(np.int32)
            else: idx_max = np.argmax(output_data)

            

            area = np.max(output_data)
            # #if (area/(25*(inte_range[1]-inte_range[0]))) / bg < 1.2: continue
            # if np.mean(output_data[idx_max+inte_range[0]:idx_max+inte_range[1]]) < bg + 7 * bg_std: continue
            if np.max(output_data) < bg + 7 * bg_std: continue
            
            #area = np.trapz(output_data[idx_max+inte_range[0]:idx_max+inte_range[1]],dx = 25)# unit of dx = ns
            #print((area/(25*(inte_range[1]-inte_range[0]))) / bg )
            
            valid_pack_count[event_idx] +=1
            area_under_line[event_idx] += area
    num_pack[event_idx] = np.unique(difference).shape[0]
    #if num_pack[event_idx] != 128: valid_pack_count[event_idx] = 0
    #print(np.unique(difference).shape[0])
    #print(difference)
    #if valid_pack_count < 10:area_under_line[event_idx] += np.nan
    #print(f'    Total number of package: {difference.shape[0]}')
    #print(f'    {valid_pack_count} Valid package found')
    #print(f'    Time difference: {np.diff(np.unique(difference))}')
    #print(f'    Sum of area under peak: {area_under_line[event_idx]}')

area_under_line = area_under_line[valid_pack_count > 1]
valid_pack_count = valid_pack_count[valid_pack_count > 1]
#area_under_line = area_under_line[valid_pack_count < 20]
#valid_pack_count = valid_pack_count[valid_pack_count < 20]

plt.figure()
plt.hist(valid_pack_count,bins = 129,histtype='step',range = [0,128])
plt.xlabel(r"$Number \ of \ packages \ in \ an \ event$")
plt.ylabel(r"$Count$")
plt.savefig('output/channel_used.png')



plt.figure()
plt.hist(num_pack,histtype='step')
plt.xlabel(r"$Number \ of \ trigger \ in \ an \ event$")
plt.ylabel(r"$Count$")
plt.savefig('output/time_stamp_event.png')

#print(f"{np.unique(num_pack,return_counts=True)}")



num_bins = 72
ratio_inti = 2
throld = 1

print(f'Valid event: {np.size(area_under_line)} (area > 0)')
#cut_num = np.sort(area_under_line)[round(throld*area_under_line.shape[0]-1)]
#area_under_line = area_under_line[area_under_line<=cut_num]
#print(f'Cutting throld: {cut_num} Valid event: {np.size(area_under_line)}')
hist,bins = np.histogram(area_under_line,bins=num_bins)
#fitting
#(mu,sigma) = norm.fit(area_under_line[np.logical_and(area_under_line>25000,area_under_line<35000)])

#(mu,sigma) = norm.fit(area_under_line[np.logical_and(area_under_line > 20000 , area_under_line < 40000)])
#(mu2,sigma2) = norm.fit(area_under_line[area_under_line>ratio_inti*bins[np.argmax(hist)]])

#gmm = GaussianMixture(n_components=2,max_iter = 10000).fit(area_under_line.reshape(-1, 1)) 
#print(gmm.weights_) 
#print(gmm.means_) 
#print(gmm.covariances_)
plt.figure()
plt.hist(area_under_line,bins=num_bins,histtype='step')#,range = [0,2e7])
xlim,xmax = plt.xlim()
x = np.linspace(25000,35000,1000)
#print(np.trapz(hist,dx = bins[1]-bins[0]))

#plt.plot(x,(bins[1]-bins[0])*np.sum(hist[np.logical_and(bins>25000,bins<35000)[0:num_bins]])*norm.pdf(x,mu,sigma))

#plt.plot(x,gmm.weights_[0]*(bins[1]-bins[0])*np.sum(hist)*norm.pdf(x,gmm.means_[0,0],np.sqrt(gmm.covariances_[0,0,0])))
#plt.plot(x,gmm.weights_[1]*(bins[1]-bins[0])*np.sum(hist)*norm.pdf(x,gmm.means_[1,0],np.sqrt(gmm.covariances_[1,0,0])))
#plt.plot(x,gmm.weights_[0]*(bins[1]-bins[0])*np.sum(hist)*norm.pdf(x,gmm.means_[0,0],np.sqrt(gmm.covariances_[0,0,0]))+
#         gmm.weights_[1]*(bins[1]-bins[0])*np.sum(hist)*norm.pdf(x,gmm.means_[1,0],np.sqrt(gmm.covariances_[1,0,0])))
#plt.axvline(ratio_inti*bins[np.argmax(hist)],0,0.5,linestyle = '--',color = 'red')
plt.xlabel(r"$Q[LSB]$")
plt.ylabel(r"$Count$")
#plt.figtext(0.55,0.8,rf'$\\ Total \ event:{event_idx} \\ Valid \ event:{np.size(area_under_line[~np.isnan(area_under_line)])} \\ Integration \ width:{inte_range[1]-inte_range[0]-1} \\ Ignore \ 0<area<{throld}\times Maxima \\ Normal \ Fitting \\ \mu _1  = {gmm.means_[0,0]:.0f} \ \mu _2 = {gmm.means_[1,0]:.0f}\\ \sigma _1 = {np.sqrt(gmm.covariances_[0,0,0]):.0f} \ \sigma _2 = {np.sqrt(gmm.covariances_[1,0,0]):.0f}$')
#plt.figtext(0.55,0.8,rf'$\\ Total \ event:{event_idx} \\ Valid \ event:{np.size(area_under_line[~np.isnan(area_under_line)])} \\ Integration \ width:{inte_range[1]-inte_range[0]-1} \\ Consider \ 0<area<{throld}\times Maxima \\ Normal \ Fitting \\ \mu  = {mu:0f}\\ \sigma = {sigma:0f}$')
plt.figtext(0.55,0.8,rf'$\\ Total \ event:{event_idx} \\ Valid \ event:{np.size(area_under_line[~np.isnan(area_under_line)])}$')
plt.savefig("output/spectrum.png")
plt.savefig("output/spectrum.png")
"""

"""
#print(sub_pack_trigger_source_stamp)
board_id_idx_test = np.where(existed_board_id==254)[0]
idx = pack_pointer_board_channel_timeStamp[board_id_idx_test][pack_pointer_board_channel_timeStamp_valid[board_id_idx_test]]
print(idx)
print(sub_pack_trigger_source_stamp[idx])
print(sub_pack_trigger_source_count[idx])
"""

"""
idx_g_board = []
idx_g_channel = []
idx_g_timestamp = []
for idx_board_tri in range(existed_board_id.shape[0]):
    for idx_channel_tri in range(existed_channel_id.shape[0]):
        for idx_timestamp_tri in range(existed_time_stamp.shape[0]):
            if not pack_pointer_board_channel_timeStamp_valid[idx_board_tri,idx_channel_tri,idx_timestamp_tri]: continue
            idx = pack_pointer_board_channel_timeStamp[idx_board_tri,idx_channel_tri,idx_timestamp_tri]
            if not existed_board_id[idx_board_tri] == 254: continue
            #if not existed_channel_id[idx_channel_tri] == 32: continue
            if np.max(wave_sample_data[idx][wave_sample_data_valid[idx]]) >= 300:
                idx_g_board.append(idx_board_tri)
                idx_g_channel.append(idx_channel_tri)
                idx_g_timestamp.append(idx_timestamp_tri)
idx_g_board = np.array(idx_g_board).astype(np.int32)
idx_g_channel = np.array(idx_g_channel).astype(np.int32)
idx_g_timestamp = np.array(idx_g_timestamp).astype(np.int32)
print(f"{idx_g_board.shape[0]} package labeled")
#print(np.unique(existed_board_id[idx_g_500_board]))
#print(idx_g_500_board)
'''
plt.figure()
plt.hist(existed_board_id_new[idx_g_board],bins = 4,range=(0,4))
plt.ylabel(r"$Count$")
plt.xlabel(r"$Board ID$")
plt.savefig("Hist_board")

plt.figure()
plt.hist(existed_channel_id[idx_g_channel],bins = 64,range=(0,63))
plt.ylabel(r"$Count$")
plt.xlabel(r"$Channel ID$")
plt.savefig("Hist_channel")
'''

areas = np.zeros(num_ext_tri+1)
areas_count = np.zeros(num_ext_tri+1)
cut_low = 0
cut_high = 0
for idx in tqdm(range(idx_g_board.shape[0])):
    board_id_idx_test = np.where(existed_board_id==254)[0]#idx_g_board[idx]#0#
    channel_id_idx_test = idx_g_channel[idx]#0#np.where(existed_channel_id==47)[0]
    #print(np.where(pack_pointer_board_channel_timeStamp_valid[board_id_idx_test,channel_id_idx_test,:]))
    timeStamp_idx_test = idx_g_timestamp[idx]#np.where(pack_pointer_board_channel_timeStamp_valid[0,0,:])[0][0]#np.where(existed_time_stamp=='0xf8d5610')[0]
    #print(timeStamp_idx_test)

    board_id_test = existed_board_id[board_id_idx_test]
    channel_id_test = existed_channel_id[channel_id_idx_test]
    timeStamp_test = existed_time_stamp[timeStamp_idx_test]

    #print(f"Test board id: {board_id_test},test channel_id: {channel_id_test}, test timeStamp: {timeStamp_test}")

    if pack_pointer_board_channel_timeStamp_valid[board_id_idx_test,channel_id_idx_test,timeStamp_idx_test]:
        #if not idx % (137*5) == 0 : continue
        #print("Pack found")
        this_idx = pack_pointer_board_channel_timeStamp[board_id_idx_test,channel_id_idx_test,timeStamp_idx_test]
        this_tri_count = sub_pack_trigger_source_count[this_idx]
        output_data = wave_sample_data[this_idx][wave_sample_data_valid[this_idx]]

        #bg_infro
        bg = np.mean(output_data[0:bg_range])
        '''
        #found maxima
        if np.max(output_data) == 4095:
            idx_max = np.round(np.mean(np.where(output_data==4095)[0])).astype(np.int32)
        else: idx_max = np.argmax(output_data)
        '''
        algo = rpt.Dynp(model="l2", min_size=3).fit(output_data)
        idx_max = algo.predict(n_bkps=2)[0:2]
        #print(idx_max)

        '''
        peaks,__ = find_peaks(output_data,width = 16)
        if np.size(peaks) == 0 :
            cut_low+=1
            continue 
        if np.size(peaks) > 1 :
            cut_high+=1
            continue 
        else:idx_max = peaks[0]
        '''

        #area = np.trapz(output_data[idx_max+inte_range[0]:idx_max+inte_range[1]],dx = 25)# unit of dx = ns
        #area_m_bg = np.trapz(output_data[idx_max+inte_range[0]:idx_max+inte_range[1]]-bg,dx = 25)# unit of dx = ns
        
        #output_mean = np.mean(output_data)
        #output_std = np.std(output_data)

        #areas[this_tri_count] += area_m_bg
        #areas_count[this_tri_count] += 1
        
        plt.figure()
        #plt.text(300,10,rf'$\sigma = {output_std:.3f},\\ mean = {output_mean:.3f}$')
        plt.plot(output_data)
        #plt.annotate(text = rf'$Area: {area:.1f} \enspace LSB \times ns \\ Area \enspace - \enspace background: {area_m_bg:.1f} \enspace LSB \times ns \\Index \enspace of \enspace Maxima: {idx_max}$',xy=(idx_max+inte_range[1]+1,np.max(output_data)*0.7))
        #plt.fill_between(np.arange(idx_max+inte_range[0],idx_max+inte_range[1]+1),output_data[idx_max+inte_range[0]:idx_max+inte_range[1]+1],color = "c",hatch='//',alpha=0.3,label=r"$Area_{signal}$")
        plt.axvline(idx_max[0],ls='--',color = 'r')
        plt.axvline(idx_max[1],ls='--',color = 'r')
        #plt.axvline(idx_max+inte_range[0],ls='--',color = 'r')
        #plt.axvline(idx_max+inte_range[1],ls='--',color = 'r')
        
        plt.fill_between(np.arange(0,bg_range+1),output_data[0:bg_range+1],color = "g",hatch='//',alpha=0.3,label = r"$Area_{noise}$")
        plt.axhline(bg,ls='--',color = 'g',label=r"$Average \enspace noise$")
        plt.axvline(bg_range,ymax=0.5,ls='--',color = 'k')
        
        plt.legend()

        plt.xlabel(r'$Time (\times 25ns)$')
        plt.ylabel(r'$Data$')
        plt.ylim(bottom=0)
        plt.title(rf'$Board Id: {board_id[0,this_idx]}(New Id:{existed_board_id_new[existed_board_id == board_id[0,this_idx]].astype(np.int32)[0]}),Channel Id: {sub_pack_channel_id_int[this_idx]},Time Stamp: {sub_pack_trigger_source_stamp[this_idx]}$')
        plt.savefig(f"output/Wave_samples/B{board_id_test}C{channel_id_test}T{timeStamp_test}.png")
        plt.close()
        idx_head_test = idx_head[0,idx]
        #print(f"{output_std}")
        #print(file[idx_head_test:idx_head_test+800])
       

    else: print("Pack not found")
"""
"""
print(cut_low)
print(cut_high)
(mu,sigma) = norm.fit(areas[areas_count>10])
print(np.unique(areas))
plt.figure()
plt.hist(areas[areas_count>10],bins=24,range=(0,1.1e7),histtype='step',density=True,label=r'$Without \enspace background$')
plt.plot(np.linspace(0,1.1e7,24), norm.pdf(np.linspace(0,1.1e7,24), mu, sigma))
plt.ylabel(r'$Count$')
plt.xlabel(r'$Normalized \enspace Area$')
#plt.legend()
plt.savefig("spectrum")
"""

"""
# statistical anaylsis, This is used to found out different bg noice accross different channel and different board
for idx_board_sta in range(existed_board_id.shape[0]):
    channel_noise_mean = np.zeros([existed_channel_id.shape[0]])
    channel_noise_sigma = np.zeros([existed_channel_id.shape[0]])
    channel_noise_size = np.zeros([existed_channel_id.shape[0]])
    for idx_channel_sta in range(existed_channel_id.shape[0]):
        idx = pack_pointer_board_channel_timeStamp[idx_board_sta,idx_channel_sta,pack_pointer_board_channel_timeStamp_valid[idx_board_sta,idx_channel_sta,:]]
        channel_noise_size[idx_channel_sta] = wave_sample_data[idx].shape[0]
        if np.logical_not(channel_noise_size[idx_channel_sta]==0):
            channel_noise_mean[idx_channel_sta] = np.mean(wave_sample_data[idx])
            channel_noise_sigma[idx_channel_sta] = np.std(wave_sample_data[idx])
        else:
            channel_noise_mean[idx_channel_sta] = np.nan
            channel_noise_sigma[idx_channel_sta] = np.nan
    np.savetxt(f'Noise_level/Board{board_id[0,idx][0]}_noise_mean',channel_noise_mean)
    np.savetxt(f'Noise_level/Board{board_id[0,idx][0]}_noise_sigma',channel_noise_sigma)
    fig,ax1 = plt.subplots(2,1,sharex=True)
    ax1[0].plot(channel_noise_mean,label = r'$\overline{noise}$',color='b')
    ax2 = ax1[0].twinx()
    ax2.plot(channel_noise_sigma,label = r'$\sigma$',color='r')
    fig.legend()
    
    ax1[0].set_ylabel(r'$Data$')
    ax1[0].set_title(rf'$Board \enspace ID \enspace {board_id[0,idx][0]}$')
    
    ax1[1].step(existed_channel_id,channel_noise_size,where = 'mid')
    ax1[1].set_ylabel(r'$Package \enspace count$')
    ax1[1].set_xlabel(r'$Channel \enspace ID$')
    for idx_channel_sta in range(existed_channel_id.shape[0]):
        if channel_noise_size[idx_channel_sta] == 0:
            if idx_channel_sta < existed_channel_id.shape[0]/2:
                ax1[1].annotate(text = rf'$Suspected \enspace error \enspace \\ at \enspace channel \enspace id \enspace{existed_channel_id[idx_channel_sta]}$',
                            xy = (existed_channel_id[idx_channel_sta],0),
                            xytext = (existed_channel_id[idx_channel_sta]+5,150),
                            size = 10,
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3",linestyle = '--'),
                            )
            else:
                ax1[1].annotate(text = rf'$Suspected \enspace error \enspace \\ at \enspace channel \enspace id \enspace{existed_channel_id[idx_channel_sta]}$',
                            xy = (existed_channel_id[idx_channel_sta],0),
                            xytext = (existed_channel_id[idx_channel_sta]-25,150),
                            size = 10,
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3",linestyle = '--'),
                            )
    fig.savefig(f"output/noise_boardID{board_id[0,idx][0]}")
"""