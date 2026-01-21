import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN


def UnpackPackage(file_name,event_max_interval=1000,internal_trigger_interval=20):
    '''
    Docstring for UnpackPackage
    This package is used to unpack the binary file generated from the GRS_AD64
    Based on version 2.0 2019.06 data sheet
    requires numpy tqdm and sklean.cluster
    
    :param file_name: File name of the binary file
    :param event_max_interval: Maxima time allowed for two external trigger file to be considered in same event(unit 2.5ns)
    :param internal_trigger_interval: Maxima time allowed for two Inrtnal trigger file to be considered in same event(unit 25ns)
    
    :return Package_data: dictionary contian package information see line 326 for difinations
    :return Trigger_data: dictionary contain trigger information see line 271 or line 321 for difinations
    '''

    #load file 
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
        Trigger_data = {
            "EventNum": event_num, # Number of events
            "EventExtTriggerCountNum": event_ext_trig_counts_num, # Number of trigger counts in a event
            "EventExtTriggerCount": event_ext_trig_counts, # Trigger counts in a event
            "EventExtTriggerTimeStamp": event_ext_trig_stamp, # Time stamp in of the event
            "EventExtTriggerTimeStampExcceed": event_ext_trig_stamp_exceed # If the time stamp exceeds
        }

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
        clustering = DBSCAN(eps=internal_trigger_interval, min_samples=2).fit(np.reshape(existed_time_stamp,(-1,1)))
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
        Trigger_data = {
            "EventNum": internal_event_stamp.shape[0], # Number of events
            "EventIntTimeStamp": internal_event_stamp, # Time stamp in a event
            "EventIntTimeStampValid": internal_event_stamp_valid # If this element is used
        }

    Package_data = {
        "PackagePointer": pack_pointer_board_channel_timeStamp, # Pointer labeling package with board id channel id and time stamp
        "PackagePointerValid": pack_pointer_board_channel_timeStamp_valid, # True means this pointer is used

        "ExistedBoardId": existed_board_id, # All possible board id existed in the file
        "ExistedChannelId": existed_channel_id, # All possible channel id existed in the file
        "ExistedTimeStamp": existed_time_stamp, # All possible time stamp existed in the file

        "BoardId": board_id, # Board id of the package
        "SubPackageChannelId": sub_pack_channel_id_int, # Subpakage channel id
        "SubPackageTriggerStamp": sub_pack_trigger_source_stamp, # Subpakage trigger stamp
        "SubPackageTriggerCount": sub_pack_trigger_source_count, # Subpakage trigger count

        "WaveSampleData": wave_sample_data, # Wave sample data of the package pointed by the pointer
        "WaveSampleDataValid": wave_sample_data_valid, # True if this element in the wave sample is used
    }

    return Package_data,Trigger_data