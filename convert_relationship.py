# This py file is used to convert channel_board ID relation from excel to a csv file

import pandas as pd
import numpy as np

board_id_list = [16,254,15,19,28,5,17,18,2,13]

board_ids = np.array([])
channel_ids = np.array([])
locations = np.array([])
group = np.array([])

for bord_id in board_id_list:
    board_ids = np.append(board_ids,np.repeat(bord_id,64))
board_ids = board_ids.astype(np.int32)
print(board_ids)

# read ids
for idx in range(10):
    sheet = pd.read_excel('Pad排列说明文档与DAQ通道对应关系/GasTPC通道与pad对应关系_edited.xlsx',sheet_name=f'J{idx+1}--Board No {board_id_list[idx]}',header = 0,usecols='A:E')
    #print(sheet)
    #print(sheet['定义']=='Signal')
    #print(sheet['通道号'][sheet['定义']=='Signal'])
    #print(sheet['行列号'][sheet['定义']=='Signal'])
    #print(sheet)

    channel_ids = np.append(channel_ids,sheet['通道号'][sheet['定义']=='Signal'].to_numpy())
    locations = np.append(locations,sheet['行列号'][sheet['定义']=='Signal'].to_numpy())
    group = np.append(group,sheet['组号'][sheet['定义']=='Signal'].to_numpy())

channel_ids = channel_ids.astype(np.int32)

print(channel_ids)
print(locations)

board_channel_location_relation = {
    'BoardId': np.array([],dtype=np.int32),
    'ChannelId': np.array([],dtype=np.int32),
    'LocationId': np.array([],dtype=np.int32),
    'Group': np.array([],dtype=np.int32),
    'State': np.array([],dtype=np.bool_), # True for L and False for C
}
# process data
for idx, location in enumerate(locations):
    if location == 'GND': continue
    State = location[0] == 'L'
    potition = int(location[1:])
    #print(group[idx])
    group_number = int(group[idx][1:])
    print(f'State:{State} location:{potition}')
    board_channel_location_relation['BoardId'] = np.append(board_channel_location_relation['BoardId'],board_ids[idx])
    board_channel_location_relation['ChannelId'] = np.append(board_channel_location_relation['ChannelId'],channel_ids[idx])
    board_channel_location_relation['State'] = np.append(board_channel_location_relation['State'],State)
    board_channel_location_relation['Group'] = np.append(board_channel_location_relation['Group'],group_number)
    board_channel_location_relation['LocationId'] = np.append(board_channel_location_relation['LocationId'],potition)


print(board_channel_location_relation)
np.savetxt('BoardChannelLocationRelation.csv',(board_channel_location_relation['BoardId'],# first row
                                               board_channel_location_relation['ChannelId'],# second row
                                               board_channel_location_relation['LocationId'],# third row
                                               board_channel_location_relation['Group'],# fourth row
                                               board_channel_location_relation['State']# fifth row (True for L and False for C)
                                               ), fmt='%i')