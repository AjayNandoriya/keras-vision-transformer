# %% [markdown]
# # Oxford IIIT image segmentation with SwinUNET

# %%
from typing import Any
import numpy as np
from glob import glob
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras

# %%
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, concatenate

# %%
import sys
sys.path.append('../')

from keras_vision_transformer import swin_layers
from keras_vision_transformer import transformer_layers
from keras_vision_transformer import utils

# %% [markdown]
# # Data and problem statement

# %% [markdown]
# This example applies the dataset of [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) (Parkhi et al. 2012). This dataset contains images of pets and their pixel-wise mask that indicates (1) pixels belonging to the pet, (2) pixels bordering the pet, and (3) surrounding pixels.
# 
# A semantic segmentation problem is proposed; it takes images as inputs and predicts the classification probability of the three pixel-wise masks.

# %%
# the indicator of a fresh run
first_time_running = False

# user-specified working directory
filepath = '/system/drive/oxford_iiit/'

filepath = r'C:\dev\repos\ImageDecomposition\keras-vision-transformer\examples\datasets\SEMs\\'
import os
out_model_fname = os.path.join(filepath,'model.h5')

# %%
if first_time_running:
    # downloading and executing data files
    import tarfile
    import urllib.request
    
    filename_image = filepath+'images.tar.gz'
    filename_target = filepath+'annotations.tar.gz'
    
    urllib.request.urlretrieve('http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz', filename_image);
    urllib.request.urlretrieve('https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz', filename_target);
    
    with tarfile.open(filename_image, "r:gz") as tar_io:
        tar_io.extractall(path=filepath)
    with tarfile.open(filename_target, "r:gz") as tar_io:
        tar_io.extractall(path=filepath)

# %% [markdown]
# # The Swin-UNET

# %% [markdown]
# Two functions are provided for customizing the Swin-UNET:
#     
# * `swin_transformer_stack`: a function that stacks multiple Swin Transformers.
# * `swin_unet_2d_base`: the base architecture of the Swin-UNET with down-/upsampling levels and skip connections.

# %%
def swin_transformer_stack(X, stack_num, embed_dim, num_patch, num_heads, window_size, num_mlp, shift_window=True, prefix=''):
    '''
    Stacked Swin Transformers that share the same token size.
    
    Alternated Window-MSA and Swin-MSA will be configured if `shift_window=True`, Window-MSA only otherwise.
    *Dropout is turned off.
    '''
    # Turn-off dropouts
    mlp_drop_rate = 0 # Droupout after each MLP layer
    attn_drop_rate = 0 # Dropout after Swin-Attention
    proj_drop_rate = 0 # Dropout at the end of each Swin-Attention block, i.e., after linear projections
    drop_path_rate = 0 # Drop-path within skip-connections
    
    qkv_bias = True # Convert embedded patches to query, key, and values with a learnable additive value
    qk_scale = None # None: Re-scale query based on embed dimensions per attention head # Float for user specified scaling factor
    
    if shift_window:
        shift_size = window_size // 2
    else:
        shift_size = 0
    
    for i in range(stack_num):
    
        if i % 2 == 0:
            shift_size_temp = 0
        else:
            shift_size_temp = shift_size

        X = swin_layers.SwinTransformerBlock(dim=embed_dim, 
                                             num_patch=num_patch, 
                                             num_heads=num_heads, 
                                             window_size=window_size, 
                                             shift_size=shift_size_temp, 
                                             num_mlp=num_mlp, 
                                             qkv_bias=qkv_bias, 
                                             qk_scale=qk_scale,
                                             mlp_drop=mlp_drop_rate, 
                                             attn_drop=attn_drop_rate, 
                                             proj_drop=proj_drop_rate, 
                                             drop_path_prob=drop_path_rate, 
                                             prefix='name{}'.format(i))(X)
    return X


def swin_unet_2d_base(input_tensor, filter_num_begin, depth, stack_num_down, stack_num_up, 
                      patch_size, num_heads, window_size, num_mlp, shift_window=True, prefix='swin_unet'):
    '''
    The base of Swin-UNET.
    
    The general structure:
    
    1. Input image --> a sequence of patches --> tokenize these patches
    2. Downsampling: swin-transformer --> patch merging (pooling)
    3. Upsampling: concatenate --> swin-transfprmer --> patch expanding (unpooling)
    4. Model head
    
    '''
    # Compute number be patches to be embeded
    input_size = input_tensor.shape.as_list()[1:]
    num_patch_x = input_size[0]//patch_size[0]
    num_patch_y = input_size[1]//patch_size[1]
    
    # Number of Embedded dimensions
    embed_dim = filter_num_begin
    
    depth_ = depth
    
    X_skip = []

    X = input_tensor
    
    # Patch extraction
    X = transformer_layers.patch_extract(patch_size)(X)

    # Embed patches to tokens
    X = transformer_layers.patch_embedding(num_patch_x*num_patch_y, embed_dim)(X)
    
    # The first Swin Transformer stack
    X = swin_transformer_stack(X, 
                               stack_num=stack_num_down, 
                               embed_dim=embed_dim, 
                               num_patch=(num_patch_x, num_patch_y), 
                               num_heads=num_heads[0], 
                               window_size=window_size[0], 
                               num_mlp=num_mlp, 
                               shift_window=shift_window, 
                               prefix='{}_swin_down0'.format(prefix))
    X_skip.append(X)
    
    # Downsampling blocks
    for i in range(depth_-1):
        
        # Patch merging
        X = transformer_layers.patch_merging((num_patch_x, num_patch_y), embed_dim=embed_dim, prefix='down{}'.format(i))(X)
        
        # update token shape info
        embed_dim = embed_dim*2
        num_patch_x = num_patch_x//2
        num_patch_y = num_patch_y//2
        
        # Swin Transformer stacks
        X = swin_transformer_stack(X, 
                                   stack_num=stack_num_down, 
                                   embed_dim=embed_dim, 
                                   num_patch=(num_patch_x, num_patch_y), 
                                   num_heads=num_heads[i+1], 
                                   window_size=window_size[i+1], 
                                   num_mlp=num_mlp, 
                                   shift_window=shift_window, 
                                   prefix='{}_swin_down{}'.format(prefix, i+1))
        
        # Store tensors for concat
        X_skip.append(X)
        
    # reverse indexing encoded tensors and hyperparams
    X_skip = X_skip[::-1]
    num_heads = num_heads[::-1]
    window_size = window_size[::-1]
    
    # upsampling begins at the deepest available tensor
    X = X_skip[0]
    
    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]
    
    depth_decode = len(X_decode)
    
    for i in range(depth_decode):
        
        # Patch expanding
        X = transformer_layers.patch_expanding(num_patch=(num_patch_x, num_patch_y), 
                                               embed_dim=embed_dim, 
                                               upsample_rate=2, 
                                               return_vector=True)(X)
        

        # update token shape info
        embed_dim = embed_dim//2
        num_patch_x = num_patch_x*2
        num_patch_y = num_patch_y*2
        
        # Concatenation and linear projection
        X = concatenate([X, X_decode[i]], axis=-1, name='{}_concat_{}'.format(prefix, i))
        X = Dense(embed_dim, use_bias=False, name='{}_concat_linear_proj_{}'.format(prefix, i))(X)
        
        # Swin Transformer stacks
        X = swin_transformer_stack(X, 
                                   stack_num=stack_num_up, 
                                   embed_dim=embed_dim, 
                                   num_patch=(num_patch_x, num_patch_y), 
                                   num_heads=num_heads[i], 
                                   window_size=window_size[i], 
                                   num_mlp=num_mlp, 
                                   shift_window=shift_window, 
                                   prefix='{}_swin_up{}'.format(prefix, i))
        
    # The last expanding layer; it produces full-size feature maps based on the patch size
    # !!! <--- "patch_size[0]" is used; it assumes patch_size = (size, size)
    
    X = transformer_layers.patch_expanding(num_patch=(num_patch_x, num_patch_y), 
                                           embed_dim=embed_dim, 
                                           upsample_rate=patch_size[0], 
                                           return_vector=False)(X)
    
    return X

# %% [markdown]
# ## Hyperparameters
# 
# Hyperparameters of the Swin-UNET are listed as follows:

# %%

def create_swinunet_model():

    filter_num_begin = 128     # number of channels in the first downsampling block; it is also the number of embedded dimensions
    depth = 4                  # the depth of SwinUNET; depth=4 means three down/upsampling levels and a bottom level 
    stack_num_down = 2         # number of Swin Transformers per downsampling level
    stack_num_up = 2           # number of Swin Transformers per upsampling level
    patch_size = (4, 4)        # Extract 4-by-4 patches from the input image. Height and width of the patch must be equal.
    num_heads = [4, 8, 8, 8]   # number of attention heads per down/upsampling level
    window_size = [4, 2, 2, 2] # the size of attention window per down/upsampling level
    num_mlp = 512              # number of MLP nodes within the Transformer
    shift_window=True          # Apply window shifting, i.e., Swin-MSA

    # %% [markdown]
    # ## Model configuration

    # %%
    # Input section
    input_size = (128, 128, 3)
    IN = Input(input_size)

    # Base architecture
    X = swin_unet_2d_base(IN, filter_num_begin, depth, stack_num_down, stack_num_up, 
                        patch_size, num_heads, window_size, num_mlp, 
                        shift_window=shift_window, prefix='swin_unet')

    # %%
    # Output section
    n_labels = 2
    OUT = Conv2D(n_labels, kernel_size=1, use_bias=False, activation='softmax')(X)

    # Model configuration
    model = Model(inputs=[IN,], outputs=[OUT,])

    # %%
    # Optimization
    # <---- !!! gradient clipping is important
    opt = keras.optimizers.Adam(learning_rate=1e-4, clipvalue=0.5)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt)
    return model

model = create_swinunet_model()

if os.path.isfile(out_model_fname):
    model.load_weights(out_model_fname)
# %% [markdown]
# ## Data pre-processing
# 
# The input of RGB images are resized to 128-by-128 through the nearest neighbour scheme, and then normalized to the interval of [0, 1]. The training target of pixel-wise masks are resized similarly.
# 
# A random split is applied with 80%, 10%, 10% of the samples are assigned for training, validation, and testing, respectively.

# %%
class DG(tf.keras.utils.Sequence):
    def __init__(self, filepath:str, shuffle:bool=True, img_size:list=[128,128], batch_size:int=1, in_channels:int=3, out_category:int=2) -> None:
        self.shuffle = shuffle
        self.in_names = np.array(sorted(glob(filepath+'images/*.jpg')))
        self.out_names = np.array(sorted(glob(filepath+'annotations/trimaps/*.png')))
        self.batch_size=  batch_size
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_category = out_category
        self.N = len(self.in_names)
        
    def augment(self, in_img, out_img):
        if self.shuffle:
            x = np.random.randint(0,in_img.shape[1]-self.img_size[1])
            y = np.random.randint(0,in_img.shape[0]-self.img_size[0])
        else:
            x,y = 0,0

        X = in_img[y:(y+self.img_size[0]), x:(x+self.img_size[1]),...]
        Y = out_img[y:(y+self.img_size[0]), x:(x+self.img_size[1]),...]
        return X,Y

    def __getitem__(self, index):
        X = np.zeros((self.batch_size, self.img_size[0],self.img_size[1], self.in_channels), np.float32)
        Y = np.zeros((self.batch_size, self.img_size[0],self.img_size[1], self.out_category), np.float32)
        for ib in range(self.batch_size):
            i = (index*self.batch_size + ib)%(self.N)
            in_img = cv2.imread(self.in_names[i]).astype(np.float32)/255
            out_img = cv2.imread(self.out_names[i],0).astype(np.float32)/255
            in_img, out_img = self.augment(in_img, out_img)

            out_img = tf.keras.utils.to_categorical(np.round(out_img).astype(np.int32),self.out_category)
            X[ib,...] = in_img
            Y[ib,...] = out_img
        return X,Y
    
    def __len__(self):
        return self.N*100
    
dg = DG(filepath=filepath)

model.fit(dg, epochs=10)
model.save(out_model_fname)

for i in range(len(dg)):
    X,Y  = dg.__getitem__(i)
    out_img = model.predict(X)
    fig, axes = plt.subplots(1,3, sharex=True, sharey=True)
    axes[0].imshow(X[0,:,:,0])
    axes[1].imshow(out_img[0,:,:,1])
    axes[2].imshow(Y[0,:,:,1])
    plt.show()
    pass

# %%
def input_data_process(input_array):
    '''converting pixel vales to [0, 1]'''
    return input_array/255.

def target_data_process(target_array):
    '''Converting tri-mask of {1, 2, 3} to three categories.'''
    return keras.utils.to_categorical(np.round(target_array/255).astype(np.int) )

# %%
sample_names = np.array(sorted(glob(filepath+'images/*.jpg')))
label_names = np.array(sorted(glob(filepath+'annotations/trimaps/*.png')))

L = len(sample_names)
ind_all = utils.shuffle_ind(L)

L_train = int(0.8*L); L_valid = int(0.1*L); L_test = L - L_train - L_valid
ind_train = ind_all[:L_train]; ind_valid = ind_all[L_train:L_train+L_valid]; ind_test = ind_all[L_train+L_valid:]

ind_train, ind_valid, ind_test= ind_all,ind_all,ind_all
L_train, L_valid, L_test = L,L,L
print("Training:validation:testing = {}:{}:{}".format(L_train, L_valid, L_test))

# %%
valid_input = input_data_process(utils.image_to_array(sample_names[ind_valid], size=128, channel=3))
valid_target = target_data_process(utils.image_to_array(label_names[ind_valid], size=128, channel=1))

# %%
test_input = input_data_process(utils.image_to_array(sample_names[ind_test], size=128, channel=3))
test_target = target_data_process(utils.image_to_array(label_names[ind_test], size=128, channel=1))

# %% [markdown]
# ## Training
# 
# The segmentation model is trained with fixed 15 epoches. Each epoch containts 100 batches and each batch contains 32 samples.
# 
# *The training process here is far from systematic, and is provided for illustration purposes only.*

# %%
N_epoch = 0 # number of epoches
N_batch = 100 # number of batches per epoch
N_sample = 32 # number of samples per batch

tol = 0 # current early stopping patience
max_tol = 3 # the max-allowed early stopping patience
min_del = 0 # the lowest acceptable loss value reduction 

# loop over epoches
for epoch in range(N_epoch):
    
    # initial loss record
    if epoch == 0:
        y_pred = model.predict([valid_input])
        record = np.mean(keras.losses.categorical_crossentropy(valid_target, y_pred))
        print('\tInitial loss = {}'.format(record))
    
    # loop over batches
    for step in range(N_batch):
        # selecting smaples for the current batch
        ind_train_shuffle = utils.shuffle_ind(L_train)[:N_sample]
        
        # batch data formation
        ## augmentation is not applied
        train_input = input_data_process(
            utils.image_to_array(sample_names[ind_train][ind_train_shuffle], size=128, channel=3))
        train_target = target_data_process(
            utils.image_to_array(label_names[ind_train][ind_train_shuffle], size=128, channel=1))
        
        # train on batch
        loss_ = model.train_on_batch([train_input,], [train_target,])
#         if np.isnan(loss_):
#             print("Training blow-up")

        # ** training loss is not stored ** #
        
    # epoch-end validation
    y_pred = model.predict([valid_input])
    record_temp = np.mean(keras.losses.categorical_crossentropy(valid_target, y_pred))
    # ** validation loss is not stored ** #
    
    # if loss is reduced
    if record - record_temp > min_del:
        print('Validation performance is improved from {} to {}'.format(record, record_temp))
        record = record_temp; # update the loss record
        tol = 0; # refresh early stopping patience
        # ** model checkpoint is not stored ** #

    # if loss not reduced
    else:
        print('Validation performance {} is NOT improved'.format(record_temp))
        tol += 1
        if tol >= max_tol:
            print('Early stopping')
            break;
        else:
            # Pass to the next epoch
            continue;

# %% [markdown]
# ## Evaluation
# 
# The testing set performance is evaluated.
model.save(out_model_fname)

# %%
dg = DG(filepath=filepath)
y_pred = model.predict([test_input,])
print('Testing set cross-entropy loss = {}'.format(np.mean(keras.losses.categorical_crossentropy(test_target, y_pred))))

# %% [markdown]
# **Example of outputs**

# %%
import matplotlib.pyplot as plt
# %matplotlib inline

# %%
def ax_decorate_box(ax):
    [j.set_linewidth(0) for j in ax.spines.values()]
    ax.tick_params(axis="both", which="both", bottom=False, top=False, 
                   labelbottom=False, left=False, right=False, labelleft=False)
    return ax

# %%
i_sample = 0

fig, AX = plt.subplots(1, 4, figsize=(13, (13-0.2)/4))
plt.subplots_adjust(0, 0, 1, 1, hspace=0, wspace=0.1)
for ax in AX:
    ax = ax_decorate_box(ax)
AX[0].imshow(np.mean(test_input[i_sample, ...,], axis=-1), cmap=plt.cm.gray)
AX[1].imshow(y_pred[i_sample, ..., 0], cmap=plt.cm.jet)
AX[2].imshow(y_pred[i_sample, ..., 1], cmap=plt.cm.jet)
AX[3].imshow(test_target[i_sample, ..., 1], cmap=plt.cm.jet)

AX[0].set_title("Original", fontsize=14)
AX[1].set_title("Pixels belong to the object", fontsize=14)
AX[2].set_title("Surrounding pixels", fontsize=14)
AX[3].set_title("Bordering pixels", fontsize=14);

plt.show()
# %%



