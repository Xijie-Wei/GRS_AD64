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
#-------------------------------------------------------------------------------------------------------

data_info,trigger_info = UnpackPackage("data_file/RAW_data_20251229_184323.bin")

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

# Open external noise file(if needed)
if use_external_noise_file:
    external_noise_mean = np.zeros([existed_board_id.shape[0],64])
    external_noise_sigma = np.zeros([existed_board_id.shape[0],64])
    for external_noise_board_id in range(existed_board_id.shape[0]):
        external_noise_mean[external_noise_board_id] = np.loadtxt(f'Noise_level/Board{existed_board_id[external_noise_board_id]}_noise_mean')
        external_noise_sigma[external_noise_board_id] = np.loadtxt(f'Noise_level/Board{existed_board_id[external_noise_board_id]}_noise_sigma')


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

"""
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
"""
plt.figtext(0.15,0.8,rf'$\\ Total \ event:{internal_event_stamp.shape[0]} \\ Valid \ event:{np.size(area_under_line[~np.isnan(area_under_line)])} \\ Normal \ Fitting \\ \mu_1  = {mu:.1f} \ \sigma_1 = {sigma:.1f} \\ \mu_2  = {mu2:.1f} \ \sigma_2 = {sigma2:.1f} \\ P-p \ ratio = {peak_main/peak_excape:.2f}$')
plt.savefig('output/spectrum.png')

#---------------------------------------------------------------------------------------------
fitting_bound1 = [3000,5000]
fitting_bound2 = [1400,2200]


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


model = GaussianModel()
par = model.guess(Q_hist[np.logical_and(Q_bins_center>fitting_bound2[0],Q_bins_center<fitting_bound2[1])],
                     x=Q_bins_center[np.logical_and(Q_bins_center>fitting_bound2[0],Q_bins_center<fitting_bound2[1])])
fitting2 = model.fit(Q_hist[np.logical_and(Q_bins_center>fitting_bound2[0],Q_bins_center<fitting_bound2[1])],
                     par,
                     x=Q_bins_center[np.logical_and(Q_bins_center>fitting_bound2[0],Q_bins_center<fitting_bound2[1])]
                     )

ax.step(Q_bins_center,Q_hist,color="#445D6C80")
peak_main = np.sum(Q_hist[np.logical_and(Q_bins_center>fitting_bound1[0],Q_bins_center<fitting_bound1[1])])
peak_escape = np.sum(Q_hist[np.logical_and(Q_bins_center>fitting_bound2[0],Q_bins_center<fitting_bound2[1])])

ax.plot(Q_bins_center[np.logical_and(Q_bins_center>fitting_bound1[0],Q_bins_center<fitting_bound1[1])],
        fitting1.best_fit,'--')
ax.plot(Q_bins_center[np.logical_and(Q_bins_center>fitting_bound2[0],Q_bins_center<fitting_bound2[1])],
        fitting2.best_fit,'--')

ax.set_xlabel(r"$Q[LSB]$")
ax.set_ylabel(r"$Count$")
ax.set_ylim(bottom = 0)
print(peak_main/peak_escape)

#fitting1.plot()
#fitting2.plot()

#print(fitting1.params['center'].value)
#print(fitting1.params['sigma'].value)
#print(fitting2.params['center'].value)
#print(fitting2.params['sigma'].value)
mu1 = fitting1.params['center'].value
sigma1 = fitting1.params['sigma'].value
mu2 = fitting2.params['center'].value
sigma2 = fitting2.params['sigma'].value

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

pos = ax.get_position()
ax_ru = fig.add_axes(pos, frameon=False)
ax_ru.xaxis.set_ticks_position('top')
ax_ru.xaxis.set_label_position('top')
ax_ru.yaxis.set_ticks_position('right')
ax_ru.yaxis.set_label_position('right')
ax_ru.spines["top"].set_color("#000000FF")
ax_ru.spines["right"].set_color("#000000FF")

ax_ru.set_xlabel(r"$Deposit \ energy \ [keV]$")
ax_ru.set_ylabel(r"$Count (\times 10^6)$")

G4_simu = ax_ru.hist(Edep,bins = 48,range = [0,7200*5.9/mu],histtype='step',color = "#000000FF",label = r"$Geant4 \ data$",align = 'right')
simu_hist,simu_bins = np.histogram(Edep,bins = num_bins)
simu_peak_main =  np.count_nonzero(Edep[np.logical_and(Edep>4,Edep<6)])/Edep.shape[0]
simu_peak_escape =  np.count_nonzero(Edep[np.logical_and(Edep>2,Edep<4)])/Edep.shape[0]

offset = ax_ru.yaxis.get_offset_text()
offset.set_position((1.08, 0.5))
offset.set_horizontalalignment('center')
offset.set_visible(False)

'''
mannul_bg = np.random.rand(round(np.count_nonzero(Edep[np.logical_and(Edep>4,Edep<6)]) * 0.00005 / 0.0006))*5.9
print(mannul_bg)
Edepbg = np.append(Edep,mannul_bg)
G4_simu = ax_top.hist(Edepbg,bins = 64,range = [0,10000*5.9/mu],histtype='step',color = "#787878FF",label = r"$Geant4 \ data + bg$")

simu_peak_main_bg =  np.count_nonzero(Edepbg[np.logical_and(Edepbg>4,Edepbg<6)])/Edepbg.shape[0]
simu_peak_escape_bg =  np.count_nonzero(Edepbg[np.logical_and(Edepbg>2,Edepbg<4)])/Edepbg.shape[0]
'''
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax_ru.get_legend_handles_labels()
#ax_top.legend(lines + lines2, labels + labels2, loc=0)


print(f"Simu P-p ratio = {simu_peak_main/simu_peak_escape:.2f}")

print(fitting1.fit_report())
print(fitting2.fit_report())
plt.figtext(0.15,0.8,rf'$\\ Total \ event:{internal_event_stamp.shape[0]} \\ Valid \ event:{np.size(area_under_line[~np.isnan(area_under_line)])} \\ Normal \ Fitting \\ \mu_1  = {mu1:.1f} \ \sigma_1 = {sigma1:.1f} \\ \mu_2  = {mu2:.1f} \ \sigma_2 = {sigma2:.1f} \\ P-p \ ratio = {peak_main/peak_escape:.2f} \\ P-p \ ratio \ (Simu) {simu_peak_main/simu_peak_escape:.2f}$')

plt.savefig('output/spectrum_filtered.png')