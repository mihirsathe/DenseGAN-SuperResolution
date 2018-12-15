# ECE285FA18_BestGroup
Super Resolution GAN Project

**Steps to connect to VM** <br>
1. Install G-Cloud CLI SDK <br>
https://cloud.google.com/sdk/ <br>
2. Connect to the instance <br>
```gcloud compute ssh ece285-tf-gpu-vm -- -L 8080:localhost:8080``` <br>
3. Open Jupyter notebook in browser by visiting:<br>
localhost:8080 <br>

**Installation Instructions**
All package requirements are in a pip requirements.txt file that can be installed with:
```
pip install -r requirements.txt
```


**Loading dataset example**
utils.py contains scripts to import data and do some basic processing

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

**Training**
```
import DenseSRGAN
dir_pfx = './' # base file path
gan = DenseSRGAN.DenseSRGAN(dir_pfx,im_hr,im_lr,proj_pfx="OH",gpu_list=[1,3,5,7],dropout_rate=0.3,num_epochs_trained=450,weights_path='./weights/OH/')

gan.train(epochs=1000,verbose=False,bench_idx=2560,batch_size=16,save_interval=10,view_interval=2)
```
**Predictions**
```
# 1117, 
ix = 10

img = gan.gen.predict(im_lr[ix:ix+1,:,:,:]).squeeze()
img = (img + 1)/2
plt.figure().suptitle('RGB+Infra', fontsize=20)
plt.subplot(1,2,1)
plt.imshow(im_hr[ix,:,:,:])
plt.subplot(1,2,2)
plt.imshow(img)
```
