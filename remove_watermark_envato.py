# 2024-10-30 09:08:00.926993
# 2024-10-16 22:43:37.305517
#Ryan Burgert 2024 - Envato Watermark Removal Demo
from rp import *


@memoized
def get_watermarks():
    watermark_root = get_path_parent(__file__)
    watermarks = load_images(
        path_join(
            watermark_root,
            "watermark_extraction",
            "extracted",
            [
                "envato_watermark_new.png",
                "envato_watermark.png",
                "videohive_watermark.png",
            ],
        ),
        use_cache=True,
    )
    watermarks=as_float_images(watermarks)
    watermarks=as_grayscale_images(watermarks)
    watermarks=as_rgb_images(watermarks)
    
    return watermarks

def get_average_frame(video):
    if isinstance(video, str):
        video = load_video(video)
    video = as_float_images(video)
    video = resize_list_to_fit(video, 30)
    video = video.mean(0)
    return video

@memoized
def sobel(watermark):
    return sobel_edges(watermark)

def watermark_scores(image):
    watermarks=get_watermarks()
    scores = 0
    watermarks = as_rgb_images(watermarks)
    watermarks = as_float_images(watermarks)

    image = as_float_image(image)
    image -= cv_box_blur(image, 20)
    image[:200] = 0
    image[330:] = 0
    image[:, :220] = 0
    image[:, -220:] = 0
    image = full_range(image)
    image-=image.mean()

    display_image(image)
    scores = []
    for watermark in watermarks:
        watermark=watermark-watermark.mean()
        scores.append(cosine_similarity(watermark, image))
    ic(scores)
    return scores


def remove_watermark(video):
    if isinstance(video,str):
        video=load_video(video)
    video=as_float_images(video)
    video=as_rgb_images(video)
    video=as_numpy_array(video)
    avg_frame=get_average_frame(video)
    watermark_index=max_valued_index(watermark_scores(avg_frame))
    watermark=get_watermarks()[watermark_index]
    
    watermark=watermark[None,:,:,:]
    
    #ALGEBRA:    
    #    video=1*watermark+(1-watermark)*new_video
    #    video-watermark=(1-watermark)*new_video
    #    (video-watermark)/(1-watermark)=new_video
    new_video=(video-watermark)
    new_video=np.clip(new_video,0,1)
    new_video=new_video/(1-watermark)
    new_video=as_byte_images(new_video)
    
    mask=cv_dilate(auto_canny(as_rgb_image(watermark[0])>.1),diameter=3,circular=True)
    
    for i,frame in enumerate(eta(new_video)):
        new_video[i]=as_rgb_image(frame)
        

    return new_video

if __name__ == "__main__":
    video_url='https://previews.customer.envatousercontent.com/h264-video-previews/3b3c4df0-a724-46c5-8f10-02d7f560a7dd/8688854.mp4'
    video=load_video(video_url)
    video=resize_list(video,100)#Only use 100 frames of the video right now because I'm impatient
    new_video=remove_watermark(video)
    preview_video=vertically_concatenated_videos(video,new_video)
    print("SAVED OUTPUT TO ",save_video_mp4(preview_video,framerate=20))
    display_video(new_video,loop=True) #press ctrl+c to exit; comment this line out if running headlessly

