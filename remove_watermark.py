import numpy as np
from rp import *

@memoized
def get_watermark_image():
    watermark_path = with_file_name(__file__,'watermark.exr')
    watermark = load_image(watermark_path, use_cache=True)
    assert is_rgba_image(watermark), 'Without alpha, the watermark is useless'
    assert is_float_image(watermark), 'Watermark should ideally be saved with floating-point precision'
    return watermark


def remove_watermark(video):
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

        # #Do the above line in mx+b form - that way if we need to we can refactor this to export m and b and do calculation on the GPU.
        # inv_alpha = 1 - alpha_watermark
        # m = -alpha_watermark / inv_alpha
        # b = 1 / inv_alpha
        # background = m * video + b

        background = np.clip(background, 0, 1)
        
        return background

    def get_shifts():

        def cross_corr(img1, img2):
            assert is_a_matrix(img1)
            assert is_a_matrix(img2)

            # Compute the FFT of both images
            fft1 = np.fft.fft2(img1)
            fft2 = np.fft.fft2(img2, img1.shape)
            # Compute the cross-correlation in frequency domain
            cross_fft = fft1 * np.conj(fft2)
            # Compute the inverse FFT to get the cross-correlation in spatial domain
            cross_corr = np.fft.ifft2(cross_fft)
            # Shift the zero-frequency component to the center of the spectrum
            cross_corr = np.fft.fftshift(cross_corr)
            return np.real(cross_corr)

        def best_shift(frame, watermark):
            # Compute the cross-correlation between frame and watermark
            corr = cross_corr(frame, watermark)
            # Find the coordinates of the maximum correlation
            max_loc = np.unravel_index(np.argmax(corr), corr.shape)
            # Compute the shift amounts
            dy, dx = (
                max_loc[0] - watermark.shape[0] // 2,
                max_loc[1] - watermark.shape[1] // 2,
            )
            return dx, dy
        

        zwatermark = blend_images(0.5, watermark) - 0.5  # Shape: H W C
        zavg_frame = avg_frame - cv_gauss_blur(avg_frame, sigma=20)  # Shape: H W C
        zavg_frame=as_grayscale_image(zavg_frame)
        zwatermark=as_grayscale_image(zwatermark)

        return best_shift(zavg_frame, zwatermark)

    if video.dtype==np.uint8:
        #Make it a floating point video
        video=video/255

    watermark=get_watermark_image()

    avg_frame=video.mean(0)

    watermark=crop_image(watermark,*get_image_dimensions(avg_frame)) #Sometimes not a perfect match...

    best_watermark=None
        
    best_x_shift,best_y_shift=get_shifts()
    best_watermark = np.roll(watermark, (best_y_shift, best_x_shift), axis=(0, 1))

    recovered=recover_background(video,best_watermark)

    return recovered

def demo_remove_watermark():
    test_videos=rp_glob('webvid/*.mp4')
    test_videos=shuffled(test_videos)

    while test_videos:
        tic()
        video_path = test_videos.pop()

        fansi_print("Loading video from " + video_path, "green", "bold")
        video = load_video(video_path, use_cache=False)
        video = as_numpy_array(resize_list(video, length=60))

        recovered = remove_watermark(video)
        ptoc()

        #analy_video=vertically_concatenated_videos(recovered,video)
        # analy_video=labeled_images(analy_video,'dx=%i   dy=%i'%(best_x_shift,best_y_shift))

        #save_video_mp4(analy_video,get_unique_copy_path('comparison_videos/comparison_video.mp4'),framerate=30)
        #display_video(analy_video)
