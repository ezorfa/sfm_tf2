from __future__ import division
import tensorflow as tf
import random
import numpy as np
from nets import SuperModel
from data_loader import DataLoader
from utils import projective_inverse_warp
from utils import colorize
import os
import datetime
import time

batch_size = 4
num_source = 2
num_scales = 4
explain_reg_weight = 0.0
smooth_weight = 0.5
img_height = 128
img_width = 416
optimizer_hp = 0.0002
max_steps = 200000
steps_per_ckpt = 20000
summary_freq = 500
steps_per_save_weights = 20000


def main(_):
	

    dataset_dir = "/cdtemp/ezorfa/datasets/kitti/kitti_seq3"

    loader = DataLoader(dataset_dir, batch_size, img_height, img_width, num_source, num_scales)

    train_dataset = loader.load_train_batch()

    optimizer = tf.keras.optimizers.Adam(learning_rate=optimizer_hp, epsilon=1e-08)

    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

    # ------------------------------------------------------------------------>
    @tf.function
    def preprocess_image(image):
        # Assuming input image is uint8
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image * 2. -1.
    @tf.function
    def deprocess_image(image):
        # Assuming input image is float32
        image = (image + 1.)/2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)
    @tf.function
    def get_reference_explain_mask(downscaling):
        tmp = np.array([0,1])
        ref_exp_mask = np.tile(tmp,
                               (batch_size,
                                int(img_height/(2**downscaling)),
                                int(img_width/(2**downscaling)),
                                1))
        ref_exp_mask = tf.constant(ref_exp_mask, dtype=tf.float32)
        return ref_exp_mask
    @tf.function
    def compute_smooth_loss(pred_disp):
        def gradient(pred):
            D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
            D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
            return D_dx, D_dy
        dx, dy = gradient(pred_disp)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        return tf.reduce_mean(tf.abs(dx2)) + \
               tf.reduce_mean(tf.abs(dxdy)) + \
               tf.reduce_mean(tf.abs(dydx)) + \
               tf.reduce_mean(tf.abs(dy2))
               
    def compute_exp_reg_loss(pred, ref):
        l = tf.nn.softmax_cross_entropy_with_logits( labels=tf.reshape(ref, [-1, 2]), logits=tf.reshape(pred, [-1, 2]))
        return tf.reduce_mean(l)
    # ------------------------------------------------------------------------>
    
    @tf.function
    def compute_loss(model, x):

        src_image_stack = preprocess_image(x[0])
        tgt_image = preprocess_image(x[1])
        intrinsics = x[2]
        
        pred_poses, pred_disp, pred_exp_logits = model(tgt_image, src_image_stack)
        pred_depth = [1./d for d in pred_disp]
        
        pixel_loss = 0
        exp_loss = 0
        smooth_loss = 0
        tgt_image_all = []
        src_image_stack_all = []
        proj_image_stack_all = []
        proj_error_stack_all = []
        exp_mask_stack_all = []

        for s in range(num_scales):
            
            if explain_reg_weight > 0:
                ref_exp_mask = get_reference_explain_mask(s)
            
            curr_tgt_image = tf.compat.v1.image.resize_area(tgt_image,
                [int(img_height/(2**s)), int(img_width/(2**s))])
            curr_src_image_stack = tf.compat.v1.image.resize_area(src_image_stack,
                [int(img_height/(2**s)), int(img_width/(2**s))])
            
            if smooth_weight > 0:
                smooth_loss += smooth_weight/(2**s) * compute_smooth_loss(pred_disp[s])
            
            for i in range(num_source):
                
                curr_proj_image = projective_inverse_warp(
                curr_src_image_stack[:,:,:,3*i:3*(i+1)],
                tf.squeeze(pred_depth[s], axis=3),
                pred_poses[:,i,:],
                intrinsics[:,s,:,:])
                
                curr_proj_error = tf.abs(curr_proj_image - curr_tgt_image)

                if explain_reg_weight > 0:
                    curr_exp_logits = tf.slice(pred_exp_logits[s], [0, 0, 0, i*2], [-1, -1, -1, 2])
                    
                    exp_loss += explain_reg_weight *  compute_exp_reg_loss(curr_exp_logits, ref_exp_mask)
                    curr_exp = tf.nn.softmax(curr_exp_logits)
                                    
                if explain_reg_weight > 0:
                    pixel_loss += tf.reduce_mean(curr_proj_error * \
                        tf.expand_dims(curr_exp[:,:,:,1], -1))
                else:
                    pixel_loss += tf.reduce_mean(curr_proj_error)
                
                if i == 0:
                    proj_image_stack = curr_proj_image
                    proj_error_stack = curr_proj_error
                    if explain_reg_weight > 0:
                        exp_mask_stack = tf.expand_dims(curr_exp[:,:,:,1], -1)
                else:
                    proj_image_stack = tf.concat([proj_image_stack,
                                                  curr_proj_image], axis=3)
                                                  
                    proj_error_stack = tf.concat([proj_error_stack,
                                                      curr_proj_error], axis=3)
                    if explain_reg_weight > 0:
                        exp_mask_stack = tf.concat([exp_mask_stack, tf.expand_dims(curr_exp[:,:,:,1], -1)], axis=3)
            
            tgt_image_all.append(curr_tgt_image)
            src_image_stack_all.append(curr_src_image_stack)
            proj_image_stack_all.append(proj_image_stack)
            proj_error_stack_all.append(proj_error_stack)
            if explain_reg_weight > 0:
                exp_mask_stack_all.append(exp_mask_stack)
        total_loss = pixel_loss + smooth_loss + exp_loss
        
        return total_loss, [pred_disp, pred_poses, tgt_image_all,src_image_stack_all,proj_image_stack_all, proj_error_stack_all]
        
    @tf.function
    def compute_apply_gradients(model, x, optimizer):
        with tf.GradientTape() as tape:
            loss, summary_list = compute_loss(model, x)
            train_loss(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return summary_list

    def tb_graph(model):
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = 'logs/gradient_tape/%s' % stamp
        writer = tf.summary.create_file_writer(logdir)
        tf.summary.trace_on(graph=True, profiler=False)
        
        for sample in train_dataset:
            src_image_stack = preprocess_image(sample[0])
            tgt_image = preprocess_image(sample[1])
            pred_poses, pred_disp = model(tgt_image, src_image_stack)
            compute_apply_gradients(model, name, optimizer)
        with writer.as_default():
          tf.summary.trace_export( name="my_graph_trace",step=0,
              profiler_outdir=logdir)
    
    def collect_summaries(summary_list, step,writer):
        with writer.as_default():
            for s in range(num_scales):
                tf.summary.histogram("scale%d_depth" %s, 1./summary_list[0][s], step=step)
                #tf.summary.image('scale%d_disparity_image' % s, colorize(summary_list[0][s], cmap='plasma') ,step=step)
                tf.summary.image('scale%d_disparity_image' % s, (summary_list[0][s]) ,step=step)
                tf.summary.image('scale%d_target_image' %s, deprocess_image(summary_list[2][s]),step=step)
                for i in range(num_source):
                    if explain_reg_weight > 0:
                        tf.summary.image('scale%d_projected_image_%d' % (s, i), deprocess_image(summary_list[4][s][:, :, :, i*3:(i+1)*3]),step=step)
                    tf.summary.image(
                    'scale%d_source_image_%d' % (s, i),
                    deprocess_image(summary_list[3][s][:, :, :, i*3:(i+1)*3]),step=step)
                    tf.summary.image('scale%d_projected_image_%d' % (s, i),
                    deprocess_image(summary_list[4][s][:, :, :, i*3:(i+1)*3]),step=step)
                    tf.summary.image('scale%d_proj_error_%d' % (s, i),
                    deprocess_image(tf.clip_by_value(summary_list[5][s][:,:,:,i*3:(i+1)*3] - 1, -1, 1)),step=step)

            tf.summary.histogram("tx", summary_list[1][:,:,0], step=step)
            tf.summary.histogram("ty", summary_list[1][:,:,1], step=step)
            tf.summary.histogram("tz", summary_list[1][:,:,2], step=step)
            tf.summary.histogram("rx", summary_list[1][:,:,3], step=step)
            tf.summary.histogram("ry", summary_list[1][:,:,4], step=step)
            tf.summary.histogram("rz", summary_list[1][:,:,5], step=step)
            tf.summary.scalar('loss', train_loss.result(), step=step)


    model = SuperModel()
    
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = 'logs/gradient_tape/%s' % stamp
    writer = tf.summary.create_file_writer(logdir)
    
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=10)
    
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch and starting training. :)")
    
        
    for epoch in range(1, max_steps):
        start_time = time.time()
        for name in train_dataset:
            ckpt.step.assign_add(1)
            step = tf.dtypes.cast(ckpt.step, tf.int64)
            summary_list = compute_apply_gradients(model, name, optimizer)
#            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in model.trainable_variables])
#            tf.print(parameter_count, " : parameter count 34094704")
            
            if int(ckpt.step) % steps_per_ckpt == 0:
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

            if int(ckpt.step)% steps_per_save_weights == 0:
                print("Saved weights for step {}".format(int(ckpt.step)))
                model.save_weights("./tmp/sw", save_format='tf')
            
                
            if int(ckpt.step) % summary_freq == 0:
                print("summary recorded")
                collect_summaries(summary_list,step,writer)


        print('Epoch: {} {} {}'.format(epoch,train_loss.result(),time.time()-start_time))
        train_loss.reset_states()
    
if __name__ == '__main__':
    tf.compat.v1.app.run()
