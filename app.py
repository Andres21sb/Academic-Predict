import pandas as pd
import time
import matplotlib.pyplot as plt


#metrics for read_csv
start_time = time.time()
# Load the data
#data = pd.read_csv('DATA/Dataset.csv')
data = pd.read_csv('DATA/MiniDataset.csv')
end_time = time.time()

print("Time taken to read the data: ", end_time - start_time)

# print data size
print('data size -> ',data.shape)

#data description
print('Data description in progress...')

#save output of data.describe() to a txt file
with open('Results/describeMini.txt', 'w') as f:
    f.write(data.describe().to_string())
    
print('Data description completed and saved to Results/describeMini.txt')

#plot bar graph for data target and save it
data['Target'].value_counts().plot(kind='bar')
plt.title('Target distribution')
plt.savefig('Results/targetDistributionMini.png')
print('Target distribution plot saved to Results/targetDistributionMini.png')
