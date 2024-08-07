# 2024-07-30 02:48:17.279056
urls = load_text_file("urls.txt").splitlines()
urls = [
    url
    for url in random_batch(urls, 300)
    if file_exists(get_file_name(url))
    and printed(get_video_file_shape(get_file_name(url))[1:3]) == (336, 596)
]
urls=random_batch_up_to(urls,200)#For testing

def get_thumbnail(url):
    url=url[:len('https://ak.picdn.net/shutterstock/videos/')]+url[len('https://ak.picdn.net/shutterstock/videos/'):].split('/')[0]+'/thumb/1.jpg'
    return url
import numpy as np
import einops

afters = load_images(
    # "/Users/ryan/Downloads/njnkjii.webp",
    # "/Users/ryan/Downloads/thumb2.webp",
    [get_thumbnail(x) for x in urls],
    show_progress=True,
    use_cache=True,
    strict=None,
)
urls=[url for url,after in zip(urls,afters) if after is not None]
afters=[x for x in afters if x is not None]
afters=as_rgb_images(afters)

afters = (np.stack(
    resize_images(np.stack(
        resize_images(
            afters,
            size=(336, 596),
            interp="area",
        )
    )[:,:,:-1],size=(336,596)))
    / 255
)
# afters=afters[:,:,:-1]
after = afters
before = np.stack(
    [
        next(load_video_stream(x)) / 255
        for x in get_file_names(urls)#[
            #"/Users/ryan/Downloads/shuttervid.webm",
            #"/Users/ryan/Downloads/vid2.mp4",
        #]
    ]
)
before = resize_images(before, size=(336, 596), interp="area")
# display_image_slideshow([before,after])
before = einops.rearrange(before, "B H W C -> H W (B C)")
after = einops.rearrange(after, "B H W C -> H W (B C)")


def least_square_regression(Before, After, lambda_reg=0):
    # Regularization doesnt work at all ignore it

    Before = Before.reshape(*Before.shape, 1)
    X = np.concatenate([Before, np.ones_like(Before)], axis=-1)
    X_transpose = np.transpose(X, axes=(0, 1, 3, 2))
    XTX = np.matmul(X_transpose, X)

    # Create the regularization matrix
    reg_matrix = np.zeros_like(XTX)
    reg_matrix[..., 0, 0] = 1  # Regularization aimed at making m close to 1
    reg_matrix[..., 1, 1] = 1  # Regularization aimed at making b close to 0

    # Apply regularization
    XTX_reg = XTX + lambda_reg * reg_matrix

    XTX_pinv = np.linalg.pinv(XTX_reg)
    XTY = np.matmul(X_transpose, After[..., np.newaxis])
    coefficients = np.matmul(XTX_pinv, XTY)

    m = coefficients[..., 0, 0]
    b = coefficients[..., 1, 0]
    return m, b


m, b = least_square_regression(before, after)
##############3
vid = load_video("/Users/ryan/Downloads/shuttervid.webm")
vid = load_video("/Users/ryan/Downloads/vid2.mp4")
for frame in vid:
    q = m[:, :, None] * frame / 255 + b[:, :, None]
    display_image(q)
