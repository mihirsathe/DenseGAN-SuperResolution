# ECE285FA18_BestGroup
Super Resolution GAN Project<br>

**Code Orginization** <br>
- DenseSRGAN - Code Folder<br>
    - loss_logging - Folder for stored loss values<br>
    - weights - Folder for stored model weights<br>
    - DenseBlock.py - Module implementing dense block<br>
    - DenseSRGAN.py - Module implementing super resolution GAN<br>
    - DenseSRGAN_Demo - Demo of code<br>
    - OH_DenseSRGAN.ipynb - Notebook used for training on VEDAI (overhead) dataset<br>
    - VEH_DenseSRGAN.ipynb - Notebook used for training on VEDAI vehicle patches<br>
    - visualize_loss.ipynb - Module used for loss visulization and plotting<br>
- gan-training-2.gif - Gif used in README<br>
- utils.py - Module containing utility functions for importing data etc.<br>

![prediction](https://github.com/mihirsathe/ECE285FA18_BestGroup/blob/master/pred_epoch_930_sample_6.png)<br>
**Steps to connect to development VM** <br>
N.B. Will only work if you've been approved to access the VM. 
1. Install G-Cloud CLI SDK <br>
https://cloud.google.com/sdk/ <br>
2. Connect to the instance <br>
```gcloud compute ssh ece285-tf-gpu-vm -- -L 8080:localhost:8080``` <br>
3. Open Jupyter notebook in browser by visiting:<br>
localhost:8080 <br>

**Installation Instructions**<br>
All package requirements are in a pip requirements.txt file that can be installed with:
```
pip install -r requirements.txt
```


**Loading dataset example**<br>
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
![training](https://github.com/mihirsathe/ECE285FA18_BestGroup/blob/submission/gan-training-2.gif)<br>
**Predictions**
```
im_hr_patched,im_lr_patched = utils.get_img_patches(im_hr,im_lr,i)


pred_patches = gan.gen.predict(im_lr_patched)

pred_img_stiched = utils.restitch_image_patches(pred_patches,(1024,1024,4))
orig_img_stitched = utils.restitch_image_patches(im_hr_patched,(1024,1024,4))

plt.figure().suptitle('Original vs Predicted DenseSRGAN')
plt.subplot(1,2,1)
plt.imshow(orig_img_stitched[:,:,0:3])
plt.title('Original Image')
plt.subplot(1,2,2)
plt.imshow(pred_img_stiched[:,:,0:3])
plt.title('Predicted Image')
plt.savefig('{0}{1}pred_epoch_1100_sample_{2}'.format(dir_pfx,'figures/',i))
```
