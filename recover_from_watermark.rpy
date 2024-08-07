while True:
    test_videos=rp_glob('webvid/*.mp4')
    #test_videos=rp_glob('tests/*.mp4')
    #test_videos=['tests/stock-footage-fire-flames-igniting-and-burning-slow-motion-a-line-of-real-flames-ignite-on-a-black-background.mp4']
    #test_videos=['tests/stock-footage-bright-colored-abstraction-soundlights-effect.mp4']
    #test_videos=['/Users/ryan/Downloads/soijsd.webm']
    
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
    #video=video[::10]
    video=as_numpy_array(resize_list(video,length=60))
    #video=video[as_numpy_array(resize_list(range(len(video)),length=60))]
    watermark=load_image('watermark.exr',use_cache=True)
    
    avg_frame=video.mean(0)
    
    watermark=crop_image(watermark,*get_image_dimensions(avg_frame)) #Sometimes not a perfect match...
    

    
    best_watermark=None
    best_edges_mean=10000


    # best_x_shift=None
    # best_y_shift=None
    # shift_range=30
    # shifts=range(-shift_range,shift_range+1)
    # for x_shift in shifts:
    #     for y_shift in shifts:
    #         #shifted_watermark=crop_image(shift_image(watermark,x=x_shift,y=y_shift,allow_growth=False),*get_image_dimensions(avg_frame),origin='bottom right')
    #         shifted_watermark=np.roll(np.roll(watermark,x_shift,axis=1),y_shift,axis=0)
    #         
    #         # Make calculatoins faster by only using relevant regions
    #         h, w = get_image_dimensions(avg_frame)
    #         top   =     180 - shift_range
    #         bot   = h -  80 + shift_range
    #         left  =     120 - shift_range
    #         right = w - 120 + shift_range
    #
    #         recovered_frame = recover_background(
    #             avg_frame[top:bot, left:right][None],
    #             shifted_watermark[top:bot, left:right],
    #         )[0]
    #         
    #         #SLOW!            
    #         #edges=sobel_edges(recovered_frame)#TODO:Make faster
    #         
    #         edges=recovered_frame
    #         edges=np.diff(np.diff(edges,axis=0),axis=1)**2*100
    #         
    #
    #         edges_mean=edges.mean()
    #         if edges_mean<best_edges_mean:
    #             best_edges_mean=edges_mean
    #             best_watermark=shifted_watermark
    #             best_x_shift=x_shift
    #             best_y_shift=y_shift
    #         
    #         print(x_shift,y_shift,edges.mean())
    #         #display_image(edges)
    #         #input('>>>')
        
    def get_shifts():
        zwatermark = blend_images(0.5, watermark) - 0.5  # Shape: H W C
        zavg_frame = avg_frame - cv_gauss_blur(avg_frame, sigma=20)  # Shape: H W C
        zavg_frame=as_grayscale_image(zavg_frame)
        zwatermark=as_grayscale_image(zwatermark)
        def cross_corr(img1, img2):
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
            dy, dx = max_loc[0] - watermark.shape[0] // 2, max_loc[1] - watermark.shape[1] // 2
            return dx, dy
        return best_shift(zavg_frame, zwatermark)
    best_x_shift,best_y_shift=get_shifts()
    #print(best_x_shift,best_y_shift)
    best_watermark=np.roll(np.roll(watermark,best_x_shift,axis=1),best_y_shift,axis=0)



    
    recovered=recover_background(video,best_watermark)
    analy_video=vertically_concatenated_videos(recovered,video)
    analy_video=labeled_images(analy_video,'dx=%i   dy=%i'%(best_x_shift,best_y_shift))
    save_video_mp4(analy_video,get_unique_copy_path('comparison_videos/comparison_video.mp4'),framerate=30)
    display_video(analy_video)
