# ECE285FA18_BestGroup
GAN Project

utils.py contains scripts to import data and do some basic processing

**Loading dataset example**
```
path = glob.glob('./drive/My Drive/ECE285_Proj/datasets/VEDAI/*')
files = scan_dataset(path)
training_set, testing_set = create_subsets(files,'./drive/My Drive/ECE285_Proj/datasets/VEDAI/', use_validation = False)

print('Training files: ' + str(training_set))
print('Testing files: ' + str(testing_set))

rgb, infra = read_VEDAI(training_set, path)
print('RGB shape: '+ str(rgb.shape))
print('Infra shape: '+ str(infra.shape))

training_data = combine_rgb_infra(rgb,infra)
print('Data shape: '+ str(training_data.shape))
```
