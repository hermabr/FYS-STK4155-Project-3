import matplotlib.pyplot as plt
import numpy as np

adl_duration  = [3,3,5,3,4,5,5,3,3,6,7,6,5,5,5,3,4,8,8,4,5,4,3,2,3,3,3,2,4,5,8,6,6,6,9,6,5,6,4,5]

number_of_bins = 8

# plt.hist(adl_duration, bins = number_of_bins, alpha = 0.7, label = 'ADL')



fall_duration = [5,3,7,3,5,3,5,3,6,4,4,3,2,2,2,1,3,2,3,3,1,1,2,2,2,2,3,2,3,2]

plt.hist([adl_duration, fall_duration], number_of_bins, label=['ADL adjusted', 'fall'])

# plt.hist(fall_duration, bins = number_of_bins, alpha = 0.5, label = 'fall')
plt.title('Distribution of video length')
plt.xlabel('Duration (seconds)')
plt.ylabel('Count')
plt.legend()
plt.savefig('duration_distribution_adjusted_adl.pdf')
plt.show()





# pyplot.hist(x, bins, alpha=0.5, label='x')
# pyplot.hist(y, bins, alpha=0.5, label='y')



adl_duration_original = [5,6,6,5,6,7,6,6,5,10,10,8,8,7,9,8,7,8,8,9,9,8,7,2,3,3,3,2,4,13,8,6,6,6,9,11,11,11,9,11]

plt.hist([adl_duration_original, fall_duration], number_of_bins, label=['ADL original', 'fall'])

# plt.hist(fall_duration, bins = number_of_bins, alpha = 0.5, label = 'fall')
plt.title('Distribution of video length')
plt.xlabel('Duration (seconds)')
plt.ylabel('Count')
plt.legend()
plt.savefig('duration_distribution_original_adl.pdf')
plt.show()

print(f'original ADL videos: mean duratoin = {np.mean(adl_duration_original)}, max = {np.max(adl_duration_original)}, min = {np.min(adl_duration_original)}')
print(f'adjusted ADL videos: mean duratoin = {np.mean(adl_duration)}, max = {np.max(adl_duration)}, min = {np.min(adl_duration)}')
print(f'fall videos: mean duratoin = {np.mean(fall_duration)}, max = {np.max(fall_duration)}, min = {np.min(fall_duration)}')
