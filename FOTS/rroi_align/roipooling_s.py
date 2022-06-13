import numpy as np
import math

# bottom_data = cp.random.randn(1,3,40,40, dtype=np.float32)			# 特征feature 
bottom_data = np.random.randn(1,1,40,40).astype(np.float32)
batch, channels, height, width = bottom_data.shape
spatial_scale = 1.0													# 原始特征和feature的比例
rois = np.array([[0, 2, 2, 10, 10],
                 [0, 2, 4, 20, 10]], dtype=np.float32)				# rois
pooled_weight = 7													# 池化之后的宽度
pooled_height = 7
def loop(bottom_data,spatial_scale,channels,height,width, pooled_height, pool_width,bottom_rois):
    bottom_data= bottom_data.reshape(-1)
    for i in range(channels*pooled_height*pool_width):
        pw = i % pooled_width
        ph = (i // pooled_width) % pooled_height
        c = (i // pooled_width // pooled_height) % channels
        num = i // pooled_width // pooled_height // channels
        roi_batch_ind = bottom_rois[num ,  0]
        roi_start_w = round(bottom_rois[num , + 1] * spatial_scale)         
        roi_start_h = round(bottom_rois[num ,+ 2] * spatial_scale)
        roi_end_w = round(bottom_rois[num ,+ 3] * spatial_scale)
        roi_end_h = round(bottom_rois[num ,+ 4] * spatial_scale)
        #Force malformed ROIs to be 1x1
        roi_width = max(roi_end_w - roi_start_w + 1, 1)
        roi_height = max(roi_end_h - roi_start_h + 1, 1)

        #pooled_weight
        rois_pooled_width = int(math.ceil((pooled_height * roi_width) / (roi_height) ))          
        bin_size_h = (roi_height)  / (pooled_height)                         # static_cast
        bin_size_w = (roi_width)   / (rois_pooled_width)

        hstart = (math.floor((ph) * bin_size_h))
        wstart = (math.floor((pw) * bin_size_w))
        hend = (math.ceil((ph + 1) * bin_size_h))
        wend = (math.ceil((pw + 1) * bin_size_w))

        #Add roi offsets and clip to input boundaries
        hstart = min(max(hstart + roi_start_h, 0), height)
        hend = min(max(hend + roi_start_h, 0), height)
        wstart = min(max(wstart + roi_start_w, 0), width)
        wend = min(max(wend + roi_start_w, 0), width)
        is_empty = (hend <= hstart) or (wend <= wstart)
        # Define an empty pooling region to be zero
        maxval = -1E+37 if is_empty else 0
        #If nothing is pooled, argmax=-1 causes nothing to be backprop'd

        maxidx = -1
        data_offset = int((roi_batch_ind * channels + c) * height * width)
        for h in range(int(hstart),int(hend),1):
            for w in range(int(wstart),int(wend), 1):
                bottom_index = h * width + w
                if (bottom_data[data_offset + bottom_index] > maxval):
                    maxval = bottom_data[data_offset + bottom_index]
                    maxidx = bottom_index

    top_data = maxval
    argmax_data = maxidx
    return top_data, argmax_data
pooled_height = 2
maxratio = (rois[:, 3] - rois[:, 1]) / (rois[:, 4] - rois[:, 2])
maxratio = maxratio.max()
pooled_width = math.ceil(pooled_height * maxratio)

top_data = np.zeros((2, 3, pooled_height, pooled_width), dtype=np.float32)		# 输出的feature map
argmax_data = np.zeros(top_data.shape, np.int32)
top_data,argmax_posi=loop(bottom_data, spatial_scale, channels, height, width,
          pooled_height, pooled_width, rois)
print(top_data.shape)