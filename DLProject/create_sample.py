import os
import pickle
import numpy as np
import PIL.Image
import config
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

def create():
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
    with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
        _G, _D, Gs = pickle.load(f)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Pick latent vector.
    seed = np.random.randint(0, 1000)
    rnd = np.random.RandomState(seed)
    latents = rnd.randn(1, Gs.input_shape[1])

    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

    os.makedirs(config.result_dir, exist_ok=True)
    png_filename = os.path.join(config.result_dir, str(seed) + '.png')
    PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
    return seed

if __name__ == "__main__":
    create()