# ECE285FA18_BestGroup
GAN Project

utils.py contains scripts to import data and do some basic processing

**Steps to connect to VM** <br>
1. Install G-Cloud CLI SDK <br>
https://cloud.google.com/sdk/ <br>
2. Connect to the instance <br>
```gcloud compute ssh ece285-tf-gpu-vm -- -L 8080:localhost:8080``` <br>
3. Open Jupyter notebook in browser by visiting:<br>
localhost:8080 <br>



**Loading dataset example**
```
import utils
path = glob.glob('./drive/My Drive/ECE285_Proj/datasets/VEDAI/*')
files = utils.scan_dataset(path)
training_set, testing_set = utils.create_subsets(files,'./drive/My Drive/ECE285_Proj/datasets/VEDAI/', use_validation = False)

rgb, infra = utils.read_VEDAI(training_set, path)
print('RGB shape: '+ str(rgb.shape))
print('Infra shape: '+ str(infra.shape))

training_data = normalize(combine_rgb_infra(rgb,infra))
print('Data shape: '+ str(training_data.shape))
 ```
 OR
 ```
num_images = 50
files = utils.scan_dataset(data_dir, num_images)
training_set, testing_set = utils.create_subsets(files, data_dir, use_validation=False)
im_hr, im_lr, batch_idx = utils.load_data(0, training_set, data_dir, True, len(training_set))
 ```


**Installation Instructions**
All package requirements are in a pip requirements.txt file that can be installed with:
```
pip install -r requirements.txt
```
