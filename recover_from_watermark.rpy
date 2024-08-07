test_videos=rp_glob('tests/*.mp4')[:1]

import numpy as np

def recover_background(composite_images, rgba_watermark):
    # Ensure the images are in the correct float32 format ranging from 0 to 1
    composite_images = composite_images.astype(np.float32)
    rgba_watermark = rgba_watermark.astype(np.float32)
    
    # Extract RGB and Alpha components of the watermark
    rgb_watermark = rgba_watermark[:, :, :3]
    alpha_watermark = rgba_watermark[:, :, 3]
    
    # Expand alpha to work across color channels
    alpha_watermark = np.expand_dims(alpha_watermark, axis=-1)
    
    # Calculate the background image using the derived formula
    # Use np.clip to ensure the resulting pixel values are still in the range [0, 1]
    background = (composite_images - alpha_watermark * rgb_watermark) / (1 - alpha_watermark)
    background = np.clip(background, 0, 1)
    
    return background
#video=load_video('shutter_cracker.webm',use_cache=True)/255
video=load_video('/Users/ryan/Downloads/vid2.mp4',use_cache=True)/255
video=load_video(random_element(test_videos),use_cache=True)/255
#video=load_video(ans,use_cache=False)/255
video=video[::10]
watermark=load_image('watermark.exr')
watermark=crop_image(shift_image(watermark,x=0,y=2),*get_image_dimensions(watermark))
recovered=recover_background(video,watermark)
analy_video=vertically_concatenated_videos(recovered,video)
save_video_mp4(analy_video,get_unique_copy_path('comparison_video'))
display_video(analy_video)