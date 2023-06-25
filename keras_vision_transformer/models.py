import os
import tensorflow as tf

if __package__ is None or __package__ == '':
    import sys
    PACKAGE_PATH = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(PACKAGE_PATH)

from keras_vision_transformer import swin_layers, transformer_layers

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
        X = tf.concat([X, X_decode[i]], axis=-1, name='{}_concat_{}'.format(prefix, i))
        X = tf.keras.layers.Dense(embed_dim, use_bias=False, name='{}_concat_linear_proj_{}'.format(prefix, i))(X)
        
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

#  [markdown]
# ## Hyperparameters
# 
# Hyperparameters of the Swin-UNET are listed as follows:

# 

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

    #  [markdown]
    # ## Model configuration

    # 
    # Input section
    input_size = (128, 128, 3)
    IN = tf.keras.layers.Input(input_size)

    # Base architecture
    X = swin_unet_2d_base(IN, filter_num_begin, depth, stack_num_down, stack_num_up, 
                        patch_size, num_heads, window_size, num_mlp, 
                        shift_window=shift_window, prefix='swin_unet')

    # 
    # Output section
    n_labels = 2
    OUT = tf.keras.layers.Conv2D(n_labels, kernel_size=1, use_bias=False, activation='softmax')(X)

    # Model configuration
    model = tf.keras.models.Model(inputs=[IN,], outputs=[OUT,])

    # 
    # Optimization
    # <---- !!! gradient clipping is important
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4, clipvalue=0.5)
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=opt)
    return model


def create_swinunet_gen_model():

    filter_num_begin = 128     # number of channels in the first downsampling block; it is also the number of embedded dimensions
    depth = 4                  # the depth of SwinUNET; depth=4 means three down/upsampling levels and a bottom level 
    stack_num_down = 2         # number of Swin Transformers per downsampling level
    stack_num_up = 2           # number of Swin Transformers per upsampling level
    patch_size = (4, 4)        # Extract 4-by-4 patches from the input image. Height and width of the patch must be equal.
    num_heads = [4, 8, 8, 8]   # number of attention heads per down/upsampling level
    window_size = [4, 2, 2, 2] # the size of attention window per down/upsampling level
    num_mlp = 512              # number of MLP nodes within the Transformer
    shift_window=True          # Apply window shifting, i.e., Swin-MSA

    #  [markdown]
    # ## Model configuration

    # 
    # Input section
    input_size = (128, 128, 1)
    IN = tf.keras.layers.Input(input_size)

    # Base architecture
    X = swin_unet_2d_base(IN, filter_num_begin, depth, stack_num_down, stack_num_up, 
                        patch_size, num_heads, window_size, num_mlp, 
                        shift_window=shift_window, prefix='swin_unet')

    # 
    # Output section
    n_labels = 1
    OUT = tf.keras.layers.Conv2D(n_labels, kernel_size=1, use_bias=False, activation='sigmoid')(X)

    # Model configuration
    model = tf.keras.models.Model(inputs=[IN,], outputs=[OUT,])

    # 
    # Optimization
    # <---- !!! gradient clipping is important
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4, clipvalue=0.5)
    model.compile(loss=tf.keras.losses.mse, optimizer=opt)
    return model


def create_conv_unet()->tf.keras.models.Model:
    filters = [4,8,16,16,16,16]
    input_size = (None, None, 1)
    # input_size = (128, 128, 1)
    IN = tf.keras.layers.Input(input_size)

    encode_imgs = []
    x = tf.keras.layers.Conv2D(filters[0], (3,3), padding='SAME', activation='relu')(IN)
    encode_imgs.append(x)
    for i, n_filter in enumerate(filters):
        x = tf.keras.layers.Conv2D(n_filter, (3,3), strides=(2,2), padding='SAME', activation='relu')(encode_imgs[-1])
        encode_imgs.append(x)
    
    x = tf.keras.layers.Conv2D(filters[-1], (3,3), strides=(1,1), padding='SAME', activation='relu')(encode_imgs[-1])
    decode_imgs = [x] 
    for i in range(len(filters)-1, -1, -1):
        n_filter = filters[i]
        x = tf.keras.layers.UpSampling2D()(decode_imgs[-1])
        x = tf.concat([encode_imgs[i], x],axis=-1)
        x = tf.keras.layers.Conv2D(n_filter, (3,3), strides=(1,1), padding='SAME', activation='relu')(x)
        decode_imgs.append(x)
    
    x = decode_imgs[-1]

    OUT = tf.keras.layers.Conv2D(1, kernel_size=1, use_bias=False, activation='sigmoid')(x)

    # Model configuration
    model = tf.keras.models.Model(inputs=[IN,], outputs=[OUT,])

    # 
    # Optimization
    # <---- !!! gradient clipping is important
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4, clipvalue=0.5)
    model.compile(loss=tf.keras.losses.mse, optimizer=opt)
    return model

def create_conv_style_unet()->tf.keras.models.Model:
    filters = [4,8,16,16,16,16]
    input_size = (None, None, 2)
    # input_size = (128, 128, 1)
    IN = tf.keras.layers.Input(input_size)

    IN1 = IN[:,:,:,:1]
    IN2 = IN[:,:,:,1:2]
    encode_imgs = []
    x = tf.keras.layers.Conv2D(filters[0], (3,3), padding='SAME', activation='relu')(IN1)
    encode_imgs.append(x)
    for i, n_filter in enumerate(filters):
        x = tf.keras.layers.Conv2D(n_filter, (3,3), strides=(2,2), padding='SAME', activation='relu')(encode_imgs[-1])
        encode_imgs.append(x)
    
    x2 = tf.keras.layers.Conv2D(filters[0], (3,3), padding='SAME', activation='relu')(IN2)
    encode_img2s = [x2]
    for i, n_filter in enumerate(filters):
        x = tf.keras.layers.Conv2D(n_filter, (3,3), strides=(2,2), padding='SAME', activation='relu')(encode_img2s[-1])
        encode_img2s.append(x)
    x2 = tf.keras.layers.Conv2D(filters[-1], (3,3), strides=(1,1), padding='SAME', activation='relu')(encode_img2s[-1])

    x = tf.concat([encode_imgs[-1],encode_img2s[-1]], axis=-1)
    x = tf.keras.layers.Conv2D(filters[-1], (3,3), strides=(1,1), padding='SAME', activation='relu')(x)
    decode_imgs = [x] 
    for i in range(len(filters)-1, -1, -1):
        n_filter = filters[i]
        x = tf.keras.layers.UpSampling2D()(decode_imgs[-1])
        x = tf.concat([encode_imgs[i], x],axis=-1)
        x = tf.keras.layers.Conv2D(n_filter, (3,3), strides=(1,1), padding='SAME', activation='relu')(x)
        decode_imgs.append(x)
    
    x = decode_imgs[-1]

    OUT = tf.keras.layers.Conv2D(1, kernel_size=1, use_bias=False, activation='sigmoid')(x)

    # Model configuration
    model = tf.keras.models.Model(inputs=[IN,], outputs=[OUT,])

    # 
    # Optimization
    # <---- !!! gradient clipping is important
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4, clipvalue=0.5)
    model.compile(loss=tf.keras.losses.mse, optimizer=opt)
    return model