import numpy as np
from rp import *

try:
    import torch
except ImportError:
    pass

__all__ = ["remove_watermark", "demo_remove_watermark"]


def _is_uint8(x):
    if   is_numpy_array (x): return x.dtype == np.uint8
    elif is_torch_tensor(x): return x.dtype == torch.uint8
    else: raise TypeError(f"Unsupported input type: {type(x)}")


def _fft2(x):
    if   is_numpy_array (x): return    np.fft.fft2(x)
    elif is_torch_tensor(x): return torch.fft.fft2(x)
    else: raise TypeError(f"Unsupported input type: {type(x)}")


def _ifft2(x):
    if   is_numpy_array (x): return    np.fft.ifft2(x)
    elif is_torch_tensor(x): return torch.fft.ifft2(x)
    else: raise TypeError(f"Unsupported input type: {type(x)}")


def _fftshift(x):
    if   is_numpy_array (x): return    np.fft.fftshift(x)
    elif is_torch_tensor(x): return torch.fft.fftshift(x)
    else: raise TypeError(f"Unsupported input type: {type(x)}")


def _clip(x, min_val, max_val):
    if   is_numpy_array (x): return    np.clip (x, min_val, max_val)
    elif is_torch_tensor(x): return torch.clamp(x, min_val, max_val)
    else: raise TypeError(f"Unsupported input type: {type(x)}")


def _roll(x, shift, dims):
    if   is_numpy_array (x): return    np.roll(x, shift, axis=dims)
    elif is_torch_tensor(x): return torch.roll(x, shift, dims=dims)
    else: raise TypeError(f"Unsupported input type: {type(x)}")


@memoized
def _get_watermark_image():
    watermark_path = with_file_name(__file__, "watermark.exr")
    watermark = load_image(watermark_path, use_cache=True)
    assert is_rgba_image(watermark), "Without alpha, the watermark is useless"
    assert is_float_image(watermark), "Watermark should ideally be saved with floating-point precision"
    return watermark


def remove_watermark(video):
    """Removes watermark from a video.

    Given an RGB video as a NumPy array or PyTorch tensor in BHW3 form, where B is num_frames,
    H and W are height and width, and 3 (channels) is for RGB. It assumes
    it's a watermarked video - matching the watermark found in watermark.exr
    (in the same folder as this python file). Currently, that watermark is
    for shutterstock videos - and is created with make_watermark_exr.py,
    also found in the same folder as this python file.

    Args:
        video: A NumPy array or PyTorch tensor representing the video frames in BHW3 format.

    Returns:
        A NumPy array or PyTorch tensor of the same shape and type as the input video, with the
        watermark removed, and floating point pixel values between 0 and 1.

    Notes:
        The function works by:
        1. Convolving the RGBA watermark over the mean of all frames in
           grayscale to locate the watermark position. This uses FFT and
           IFFT for speed. (Technically uses cross-correlation)
        2. Once the watermark shift is found, it does inverse alpha-blending
           to remove the watermark from all frames.

        The complexity is O(total num pixels in video) aka O(B * H * W).
        It is very fast and robust, even working on videos with the watermark
        upside-down.
    """

    def recover_background(composite_images, rgba_watermark):
        # Extract RGB and Alpha components of the watermark
        watermark_rgb = rgba_watermark[:, :, :3]
        watermark_alpha = rgba_watermark[:, :, 3:]

        # Calculate the background image using the derived formula
        # Use _clip to ensure the resulting pixel values are still in the range [0, 1]
        background = (composite_images - watermark_alpha * watermark_rgb) / (1 - watermark_alpha)
        background = _clip(background, 0, 1)

        return background

    def get_shifts():
        def cross_corr(img1, img2):
            assert is_a_matrix(img1)
            assert is_a_matrix(img2)

            # Compute the FFT of both images
            fft1 = _fft2(img1)
            fft2 = _fft2(img2)
            # Compute the cross-correlation in frequency domain
            cross_fft = fft1 * fft2.conj()
            # Compute the inverse FFT to get the cross-correlation in spatial domain
            cross_corr = _ifft2(cross_fft)
            # Shift the zero-frequency component to the center of the spectrum
            cross_corr = _fftshift(cross_corr)
            return cross_corr.real

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
        zavg_frame = as_grayscale_image(zavg_frame)
        zwatermark = as_grayscale_image(zwatermark)

        return best_shift(zavg_frame, zwatermark)

    if _is_uint8(video):
        video = video / 255

    watermark = _get_watermark_image()

    avg_frame = video.mean(0)

    # Make sure the watermark image is the same size as the video so we can convolve them
    watermark = crop_image(watermark, *get_image_dimensions(avg_frame))

    best_watermark = None

    best_x_shift, best_y_shift = get_shifts()
    best_watermark = _roll(watermark, (best_y_shift, best_x_shift), dims=(0, 1))

    recovered = recover_background(video, best_watermark)

    return recovered


def demo_remove_watermark(input_video_glob="webvid/*.mp4"):
    """Demonstrates the remove_watermark function on a set of videos.

    Applies remove_watermark to a set of videos specified by the given glob
    pattern, and saves comparison videos showing the original and
    watermark-removed versions to the 'comparison_videos/' directory.

    Args:
        input_video_glob: A glob pattern specifying the set of videos to
            process. Defaults to 'webvid/*.mp4'.

    Notes:
        This demo function is fast enough to run on a typical laptop CPU.
        The processed videos are saved with filenames matching the input
        video names.
    """
    test_videos = rp_glob(input_video_glob)
    test_videos = shuffled(test_videos)

    while test_videos:
        video_path = test_videos.pop()

        fansi_print("Loading video from " + video_path, "green", "bold")
        tic()
        video = load_video(video_path, use_cache=False)
        video = as_numpy_array(resize_list(video, length=60))

        ptoctic()
        recovered = remove_watermark(video)
        ptoc()

        analy_video = vertically_concatenated_videos(recovered, video)

        fansi_print(
            "Saved video at "
            + save_video_mp4(
                analy_video,
                get_unique_copy_path(
                    "comparison_videos/"
                    + with_file_extension(
                        get_file_name(
                            video_path,
                            include_file_extension=False,
                        ),
                        "mp4",
                    ),
                ),
                framerate=30,
            ),
            "green",
            "bold",
        )
        display_video(analy_video)
        

if __name__ == "__main__":
    demo_remove_watermark()