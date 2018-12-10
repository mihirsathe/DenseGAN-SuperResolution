# ECE285FA18_BestGroup
GAN Project

utils.py contains scripts to import data and do some basic processing

**Steps to connect to VM** <br>
1. Install G-Cloud CLI SDK <br>
https://cloud.google.com/sdk/ <br>
2. Save the instance name<br>
MAC:```export INSTANCE_NAME="tensorflow-1-vm"``` <br>
3. Connect to the instance <br>
MAC:```gcloud compute ssh $INSTANCE_NAME -- -L 8080:localhost:8080``` <br>
WINDOWS:```gcloud compute ssh tensorflow-1-vm -- -L 8080:localhost:8080``` <br>
4. Open Jupyter notebook in browser by visiting:<br>
localhost:8080 <br>



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

training_data = normalize(combine_rgb_infra(rgb,infra))
print('Data shape: '+ str(training_data.shape))
```
