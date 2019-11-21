from __future__ import division
import tensorflow as tf
import numpy as np


class SuperModel(tf.keras.Model):
    def __init__(self):
        super(SuperModel,self).__init__()
        
        self.DISP_SCALING = 10
        self.MIN_DISP = 0.01
        
        self.pose_conv1 = tf.keras.layers.Conv2D (16, (7, 7), activation=tf.nn.relu, strides=(2, 2), padding='same', name='pose_conv1')
        self.pose_conv2 = tf.keras.layers.Conv2D (32, (5, 5), activation=tf.nn.relu, strides=(2, 2), padding='same' ,name='pose_conv2')
        self.pose_conv3 = tf.keras.layers.Conv2D (64, (3, 3), activation=tf.nn.relu, strides=(2, 2), padding='same' ,name='pose_conv3')
        self.pose_conv4 = tf.keras.layers.Conv2D (128, (3, 3), activation=tf.nn.relu, strides=(2, 2), padding='same' ,name='pose_conv4')
        self.pose_conv5 = tf.keras.layers.Conv2D (256, (3, 3), activation=tf.nn.relu, strides=(2, 2), padding='same' ,name='pose_conv5')
        self.pose_conv6 = tf.keras.layers.Conv2D (256, (3, 3), activation=tf.nn.relu, strides=(2, 2), padding='same' ,name='pose_conv6')
        self.pose_conv7 = tf.keras.layers.Conv2D (256, (3, 3), activation=tf.nn.relu, strides=(2, 2), padding='same' ,name='pose_conv7')
        self.pose_last = tf.keras.layers.Conv2D (12, (1, 1), activation=None, strides=(1, 1), padding='same' ,name='pose_conv_last') #6*num_source

        
        self.exp_upconv5 = tf.keras.layers.Conv2DTranspose  (256, (3, 3), activation=tf.nn.relu, strides=(2, 2), padding='same' ,name='exp_upconv5')
        self.exp_upconv4 = tf.keras.layers.Conv2DTranspose (128, (3, 3), activation=tf.nn.relu, strides=(2, 2), padding='same' ,name='exp_upconv4')
        self.mask4 = tf.keras.layers.Conv2D (2 * 2, (3, 3), activation=None, strides=(1, 1), padding='same' ,name='mask4')
        self.exp_upconv3 = tf.keras.layers.Conv2DTranspose (64, (3, 3), activation=tf.nn.relu, strides=(2, 2), padding='same' ,name='exp_upconv3')
        self.mask3 = tf.keras.layers.Conv2D (2 * 2, (3, 3), activation=None, strides=(1, 1), padding='same' ,name='mask3')
        self.exp_upconv2 = tf.keras.layers.Conv2DTranspose (32, (5, 5), activation=tf.nn.relu, strides=(2, 2), padding='same' ,name='exp_upconv2')
        self.mask2 = tf.keras.layers.Conv2D (2 * 2, (5, 5), activation=None, strides=(1, 1), padding='same' ,name='mask2')

        self.exp_upconv1 = tf.keras.layers.Conv2DTranspose (16, (7, 7), activation=tf.nn.relu, strides=(2, 2), padding='same' ,name='exp_upconv1')
        self.mask1 = tf.keras.layers.Conv2D (2 * 2, (7, 7), activation=None, strides=(1, 1), padding='same' ,name='mask1')

        
        
        self.disp_conv1 = tf.keras.layers.Conv2D (32, (7, 7), activation=tf.nn.relu, strides=(2, 2), padding='same' ,name='disp_conv1')
        self.disp_conv1b = tf.keras.layers.Conv2D(32, (7, 7), activation=tf.nn.relu, strides=(1, 1), padding='same' ,name='disp_conv1b')
        self.disp_conv2 = tf.keras.layers.Conv2D (64, (5, 5), activation=tf.nn.relu, strides=(2, 2), padding='same' ,name='disp_conv2')
        self.disp_conv2b = tf.keras.layers.Conv2D(64, (5, 5), activation=tf.nn.relu, strides=(1, 1), padding='same' ,name='disp_conv2b')
        self.disp_conv3 = tf.keras.layers.Conv2D (128, (3, 3), activation=tf.nn.relu, strides=(2, 2), padding='same' ,name='disp_conv3')
        self.disp_conv3b = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu, strides=(1, 1), padding='same' ,name='disp_conv3b')
        self.disp_conv4 = tf.keras.layers.Conv2D (256, (3, 3), activation=tf.nn.relu, strides=(2, 2), padding='same' ,name='disp_conv4')
        self.disp_conv4b = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.nn.relu, strides=(1, 1), padding='same' ,name='disp_conv4b')
        self.disp_conv5 = tf.keras.layers.Conv2D (512, (3, 3), activation=tf.nn.relu, strides=(2, 2), padding='same' ,name='disp_conv5')
        self.disp_conv5b = tf.keras.layers.Conv2D(512, (3, 3), activation=tf.nn.relu, strides=(1, 1), padding='same' ,name='disp_conv5b')
        self.disp_conv6 = tf.keras.layers.Conv2D (512, (3, 3), activation=tf.nn.relu, strides=(2, 2), padding='same' ,name='disp_conv6')
        self.disp_conv6b = tf.keras.layers.Conv2D(512, (3, 3), activation=tf.nn.relu, strides=(1, 1), padding='same' ,name='disp_conv6b')
        self.disp_conv7 = tf.keras.layers.Conv2D (512, (3, 3), activation=tf.nn.relu, strides=(2, 2), padding='same' ,name='disp_conv7')
        self.disp_conv7b = tf.keras.layers.Conv2D(512, (3, 3), activation=tf.nn.relu, strides=(1, 1), padding='same' ,name='disp_conv7b')
        
        self.disp_upcnv7 = tf.keras.layers.Conv2DTranspose (512, (3, 3), activation=tf.nn.relu, strides=(2, 2), padding='same' ,name='disp_upconv7')
        self.disp_icnv7 = tf.keras.layers.Conv2D(512, (3, 3), activation=tf.nn.relu, strides=(1, 1), padding='same' ,name='disp_iconv7')
        self.disp_upcnv6 = tf.keras.layers.Conv2DTranspose (512, (3, 3), activation=tf.nn.relu, strides=(2, 2), padding='same' ,name='disp_upconv6')
        self.disp_icnv6 = tf.keras.layers.Conv2D(512, (3, 3), activation=tf.nn.relu, strides=(1, 1), padding='same' ,name='disp_iconv6')
        self.disp_upcnv5 = tf.keras.layers.Conv2DTranspose (256, (3, 3), activation=tf.nn.relu, strides=(2, 2), padding='same' ,name='disp_upconv5')
        self.disp_icnv5 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.nn.relu, strides=(1, 1), padding='same' ,name='disp_iconv5')
        self.disp_upcnv4 = tf.keras.layers.Conv2DTranspose (128, (3, 3), activation=tf.nn.relu, strides=(2, 2), padding='same' ,name='disp_upconv4')
        self.disp_icnv4 = tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu, strides=(1, 1), padding='same' ,name='disp_iconv4')
        self.disp_upcnv3 = tf.keras.layers.Conv2DTranspose (64, (3, 3), activation=tf.nn.relu, strides=(2, 2), padding='same' ,name='disp_upconv3')
        self.disp_icnv3 = tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, strides=(1, 1), padding='same' ,name='disp_iconv3')
        self.disp_upcnv2 = tf.keras.layers.Conv2DTranspose (32, (3, 3), activation=tf.nn.relu, strides=(2, 2), padding='same' ,name='disp_upconv2')
        self.disp_icnv2 = tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu, strides=(1, 1), padding='same' ,name='disp_iconv2')
        self.disp_upcnv1 = tf.keras.layers.Conv2DTranspose (16, (3, 3), activation=tf.nn.relu, strides=(2, 2), padding='same' ,name='disp_upconv1')
        self.disp_icnv1 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu, strides=(1, 1), padding='same' ,name='disp_iconv1')
        
        self.disp_4  = tf.keras.layers.Conv2D(1, (3, 3), strides=(1, 1), activation=tf.sigmoid, padding='same' ,name='disp_4')
        self.disp_3  = tf.keras.layers.Conv2D(1, (3, 3), strides=(1, 1), activation=tf.sigmoid, padding='same' ,name='disp_3')
        self.disp_2  = tf.keras.layers.Conv2D(1, (3, 3), strides=(1, 1), activation=tf.sigmoid, padding='same' ,name='disp_2')
        self.disp  = tf.keras.layers.Conv2D(1, (3, 3), strides=(1, 1), activation=tf.sigmoid, padding='same' ,name='disp_1')
    #        self.useless = tf.Variable(tf.random.normal([100,50]), name='useless')
    @tf.function
    def resize_like(self,inputs, ref):
        iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
        rH, rW = ref.get_shape()[1], ref.get_shape()[2]
        if iH == rH and iW == rW:
            return inputs
        return tf.compat.v1.image.resize_nearest_neighbor(inputs, [rH, rW])

    @tf.function
    def __call__(self, tgt_image,src_image_stack=None, do_mask=False, which=None, training=True):

        H = tgt_image.get_shape()[1]
        W = tgt_image.get_shape()[2]
        
        x = self.disp_conv1(tgt_image)
        disp_conv1b_val = self.disp_conv1b(x)
        x = self.disp_conv2(disp_conv1b_val)
        disp_conv2b_val = self.disp_conv2b(x)
        x = self.disp_conv3(disp_conv2b_val)
        disp_conv3b_val = self.disp_conv3b(x)
        x = self.disp_conv4(disp_conv3b_val)
        disp_conv4b_val = self.disp_conv4b(x)
        x = self.disp_conv5(disp_conv4b_val)
        disp_conv5b_val = self.disp_conv5b(x)
        x = self.disp_conv6(disp_conv5b_val)
        disp_conv6b_val  = self.disp_conv6b(x)
        x = self.disp_conv7(disp_conv6b_val)
        x = self.disp_conv7b(x)
        
        x = self.disp_upcnv7(x)
        x = self.resize_like(x,disp_conv6b_val)
        x = tf.concat([x,disp_conv6b_val], axis = 3)
        x = self.disp_icnv7(x)
        
        x = self.disp_upcnv6(x)
        x = self.resize_like(x,disp_conv5b_val)
        x = tf.concat([x,disp_conv5b_val], axis = 3)
        x = self.disp_icnv6(x)
        
        x = self.disp_upcnv5(x)
        x = self.resize_like(x,disp_conv4b_val)
        x = tf.concat([x,disp_conv4b_val], axis = 3)
        x = self.disp_icnv5(x)
        
        x = self.disp_upcnv4(x)
        x = tf.concat([x,disp_conv3b_val], axis = 3)
        x = self.disp_icnv4(x)
        disp_4 = self.DISP_SCALING * self.disp_4(x) + self.MIN_DISP
        disp4_up =tf.compat.v1.image.resize_bilinear(disp_4, [np.int(H/4), np.int(W/4)])
        
        x = self.disp_upcnv3(x)
        x = tf.concat([x, disp_conv2b_val, disp4_up], axis = 3)
        x = self.disp_icnv3(x)
        disp_3 = self.DISP_SCALING * self.disp_3(x) + self.MIN_DISP
        disp3_up = tf.compat.v1.image.resize_bilinear(disp_3, [np.int(H/2), np.int(W/2)])
        
        x = self.disp_upcnv2(x)
        x = tf.concat([x, disp_conv1b_val, disp3_up], axis = 3)
        x = self.disp_icnv2(x)
        disp_2 = self.DISP_SCALING * self.disp_2(x) + self.MIN_DISP
        disp2_up = tf.compat.v1.image.resize_bilinear(disp_2, [H, W])
        
        x = self.disp_upcnv1(x)
        x = tf.concat([x, disp2_up], axis = 3)
        x = self.disp_icnv1(x)
        disp_1 = self.DISP_SCALING * self.disp(x) + self.MIN_DISP
        
        if ((training is False) and (which != 'depth')) or (training is True):
            # pose calculation
            x_in = tf.concat([tgt_image, src_image_stack], axis=3)
            x = self.pose_conv1(x_in)
            x = self.pose_conv2(x)
            x = self.pose_conv3(x)
            x = self.pose_conv4(x)
            x_5 = self.pose_conv5(x)
            x = self.pose_conv6(x_5)
            x = self.pose_conv7(x)
            x = self.pose_last(x)
            x = tf.reduce_mean(x, [1, 2])
            poses = 0.01 * tf.reshape(x, [-1, 2, 6])
        
        if do_mask:
        # explanibility masks
            x = self.exp_upconv5(x_5)
            x = self.exp_upconv4(x)
            mask4_val = x = self.mask4(x)
            x = self.exp_upconv3(x)
            mask3_val = x = self.mask3(x)
            x = self.exp_upconv2(x)
            mask2_val = x = self.mask2(x)
            x = self.exp_upconv1(x)
            mask1_val = self.mask1(x)
        else:
            mask1_val = None
            mask2_val = None
            mask3_val = None
            mask4_val = None
            
        if (training):
            return poses, [disp_1, disp_2, disp_3, disp_4], [mask1_val, mask2_val, mask3_val, mask4_val]
        else:
            pred_disp = [disp_1, disp_2, disp_3, disp_4]
            pred_depth = [1./disp for disp in pred_disp]
            pred_depth = pred_depth[0]
            if(which == 'depth'):
                return pred_depth
            elif(which == 'pose'):
                return poses
            else:
                return poses, [disp_1, disp_2, disp_3, disp_4], [mask1_val, mask2_val, mask3_val, mask4_val]


