# Video Watermark Removal

This repository contains code to help remove watermarks from videos. For Shutterstock videos, it's ready-to-go out of the box - good for datasets such as WebVid.

It's very fast, and runs on CPU. It doesn't use any machine learning. It's not perfect, but generally does a very good job. It assumes the watermarks are transparent and uses template matching to find the watermarks then inverse alpha-blending to remove them.
Unlike other watermark removal methods, this does not use inpainting - letting it retain more information than many other available implementations out there.

## Getting Started

### Simple Usage

You can simply git clone this repo, then use the files in it.

1. Use the `make_watermark_exr.py` file to extract the `watermark.exr` file - which is the watermark overlay. If you're trying to remove watermarks from Shutterstock videos (aka WebVid), you can skip this step as we've already included a `watermark.exr` file in this repo.

2. Import `remove_watermark.py` and use the `remove_watermark(video)` function to remove watermarks from a video. See its docstring for more information.

3. If you want to get started right away, import `remove_watermark.py` and run `remove_watermark.demo_remove_watermark()`

### Installation

You can also install this package via pip.

Simply run 
<!-- ``` -->
<!-- git clone https://github.com/RyannDaGreat/remove_watermark.git -->
<!-- cd remove_watermark -->
<!-- pip install -e . -->
<!-- ``` -->
```
pip install rp git+https://github.com/RyannDaGreat/remove_watermark.git
```

You may also need to run `brew install ffmpeg` or `sudo apt install ffmpeg`

I haven't included a `requirements.txt` file yet, but the only special library you'll need is called "rp" - which can be installed via `pip install rp` or the above command. When running this code, other needed packages will be installed on the fly as you need them.

### Usage
```
 >>> from remove_watermark import remove_watermark
 >>> import rp
 >>> video = rp.load_video('webvid/stock-footage-centennial-olympic-park-in-downtown-atlanta-georgia-water-fountains.mp4')
 >>> video_without_watermark = remove_watermark(video)
 >>> rp.save_video_mp4(video_without_watermark, 'output.mp4')
```

## Extracting the Watermark

The `make_watermark_exr.py` uses special video clips where the background is easy to predict. Here are some video URLs you can use to crack the watermark:

- https://www.shutterstock.com/video/clip-10884623-4k-pastel-pixel-animation-background-seamless-loop
- https://www.shutterstock.com/video/clip-16241737-4k-red-blue-yellow-pixel-animation-background
- https://www.shutterstock.com/video/clip-1070206990-abstract-art-glow-graphics-bright-colors
- https://www.shutterstock.com/video/clip-1099841851-color-background-animations-video-4-k-liquid
- https://www.shutterstock.com/video/clip-1052744771-abstract-8-bit-retro-computer-game-loading-screen
- https://www.shutterstock.com/video/clip-1064430046-green-screen-moving-colorful-frame

These files are also found in the `watermark_extraction/` folder. See `make_watermark_exr.py` for how that folder is used.

## Example Outputs

Below are two examples of the watermark removal results. The top half is with the watermark removed, and the bottom half is the original.

![Friends Eat Watermelon Happily and Enjoy Go on a Picnic](assets/stock-footage-friends-eat-watermelon-happily-and-enjoy-go-on-a-picnic.gif)
![Closeup of Chef Preparing and Throwing Vegetable Mix on Frying Pan on Fire Preparation Fresh](assets/stock-footage-closeup-of-chef-preparing-and-throwing-vegetable-mix-on-frying-pan-on-fire-preparation-fresh.gif)

