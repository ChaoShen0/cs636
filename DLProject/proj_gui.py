import _tkinter
from tkinter import *
from PIL import Image, ImageTk
import PIL
import create_sample
import os
import config
import dnnlib
import dnnlib.tflib as tflib
import pickle
import numpy as np

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

window = Tk()
window.geometry("780x300+300+300")

global img_png1
global img_png2
global img_png3
global seed1
global seed2

def main():

    # canvas = Canvas(window, width=780, height=300)
    # canvas.pack()
    tflib.init_tf()

    url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
    with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
        _G, _D, Gs = pickle.load(f)

    btCreate1 = Button(window, text="Create", command=lambda : pressCreate1(Gs)).grid(row = 0, column = 0, padx = 50)
    btCreate2 = Button(window, text="Create", command=lambda : pressCreate2(Gs)).grid(row = 0, column = 1, padx = 50)
    btMix = Button(window, text="Mix", command=lambda : pressMix(Gs)).grid(row = 0, column = 2, padx = 50)
    # btCreate1.pack(side = LEFT, anchor = N,  pady = 0)
    # btCreate2.pack(side = LEFT,  anchor = N, pady = 0)
    # btMix.pack(side = LEFT,  anchor = N, pady = 0)

    w_box = 256
    h_box = 256
    img_open = Image.open('white.png')
    w, h = img_open.size
    img_open_resized = resize(w, h, w_box, h_box, img_open)
    img_png1 = ImageTk.PhotoImage(img_open_resized)
    label_img1 = Label(window, image=img_png1, width=w_box, height=h_box).grid(row = 1, column = 0)
    label_img2 = Label(window, image=img_png1, width=w_box, height=h_box).grid(row=1, column=1)
    label_img3 = Label(window, image=img_png1, width=w_box, height=h_box).grid(row=1, column=2)
    # label_img1.pack(padx=5, pady=5, side=LEFT, anchor = S)
    # seed = create_sample.create()
    # # img_open = Image.open('/results/' + str(seed) + '.png')
    # img_open = Image.open(os.path.join(config.result_dir, str(seed) + '.png'))
    # w, h = img_open.size
    # img_open_resized = resize(w, h, w_box, h_box, img_open)
    # img_png = ImageTk.PhotoImage(img_open_resized)
    # # global label_img1
    # # label_img1 = Label(window, image=img_png, width=w_box, height=h_box).grid(row=1, column=0)
    # canvas.create_image(2, 2, anchor=NW, image=img_png)

    window.mainloop()


def resize(w, h, w_box, h_box, pil_image):
  f1 = 1.0*w_box/w # 1.0 forces float division in Python2
  f2 = 1.0*h_box/h
  factor = min([f1, f2])
  #print(f1, f2, factor) # test
  # use best down-sizing filter
  width = int(w*factor)
  height = int(h*factor)
  return pil_image.resize((width, height), Image.ANTIALIAS)

def pressCreate1(Gs):
    global seed1
    global img_png1
    seed1 = np.random.randint(0, 1000)
    src_latents = np.stack(np.random.RandomState(seed1).randn(Gs.input_shape[1]))
    src_latents = src_latents[np.newaxis, :]
    #print(src_latents.shape)
    src_dlatents = Gs.components.mapping.run(src_latents, None)
    #print(src_dlatents.shape)
    src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)
    canvas = PIL.Image.new('RGB', (1024, 1024), 'white')
    canvas.paste(PIL.Image.fromarray(src_images[0], 'RGB'), (0,0))
    canvas.save(os.path.join(config.result_dir, str(seed1) + '.png' ))

    img_open = Image.open(os.path.join(config.result_dir, str(seed1) + '.png'))
    w, h = img_open.size
    w_box = 256
    h_box = 256
    img_open_resized = resize(w, h, w_box, h_box, img_open)
    img_png1 = ImageTk.PhotoImage(img_open_resized)
    label1 = Label(window, image = img_png1).grid(row = 1, column = 0)
    # global img_png1
    # global seed1
    # w_box = 256
    # h_box = 256
    # seed1 = create_sample.create()
    # # img_open = Image.open('/results/' + str(seed) + '.png')
    # img_open = Image.open(os.path.join(config.result_dir, str(seed1) + '.png'))
    # w, h = img_open.size
    # img_open_resized = resize(w, h, w_box, h_box, img_open)
    # img_png1 = ImageTk.PhotoImage(img_open_resized)
    # label1 = Label(window, image = img_png1).grid(row = 1, column = 0)

    # label1.pack(side = LEFT, padx = 0, pady = 0, ipady = 0)
    # global label_img1
    # label_img1 = Label(window, image=img_png, width=w_box, height=h_box).grid(row=1, column=0)
    # canvas.create_image(2, 2, anchor=NW, image=img_png)

def pressCreate2(Gs):
    global seed2
    global img_png2
    seed2 = np.random.randint(0, 1000)
    src_latents = np.stack(np.random.RandomState(seed2).randn(Gs.input_shape[1]))
    src_latents = src_latents[np.newaxis, :]
    # print(src_latents.shape)
    src_dlatents = Gs.components.mapping.run(src_latents, None)
    # print(src_dlatents.shape)
    src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)
    canvas = PIL.Image.new('RGB', (1024, 1024), 'white')
    canvas.paste(PIL.Image.fromarray(src_images[0], 'RGB'), (0, 0))
    canvas.save(os.path.join(config.result_dir, str(seed2) + '.png'))

    img_open = Image.open(os.path.join(config.result_dir, str(seed2) + '.png'))
    w, h = img_open.size
    w_box = 256
    h_box = 256
    img_open_resized = resize(w, h, w_box, h_box, img_open)
    img_png2 = ImageTk.PhotoImage(img_open_resized)
    label2 = Label(window, image=img_png2).grid(row=1, column=1)
    # global img_png2
    # global seed2
    # w_box = 256
    # h_box = 256
    # seed2 = create_sample.create()
    # # img_open = Image.open('/results/' + str(seed) + '.png')
    # img_open = Image.open(os.path.join(config.result_dir, str(seed2) + '.png'))
    # w, h = img_open.size
    # img_open_resized = resize(w, h, w_box, h_box, img_open)
    # img_png2 = ImageTk.PhotoImage(img_open_resized)
    # label1 = Label(window, image = img_png2).grid(row = 1, column = 1)

def pressMix(Gs):

    global img_png3
    global seed1
    global seed2

    style_range = range(0, 4)
    src_latent = np.stack(np.random.RandomState(seed1).randn(Gs.input_shape[1]))
    src_latent = src_latent[np.newaxis, :]
    dst_latent = np.stack(np.random.RandomState(seed2).randn(Gs.input_shape[1]))
    dst_latent = dst_latent[np.newaxis, :]
    # print(src_latent.shape)
    src_dlatent = Gs.components.mapping.run(src_latent, None)
    dst_dlatent = Gs.components.mapping.run(dst_latent, None)
    # print(src_dlatent.shape)
    dst_dlatent[:, style_range] = src_dlatent[:, style_range]

    row_image = Gs.components.synthesis.run(dst_dlatent, randomize_noise=False, **synthesis_kwargs)
    canvas = PIL.Image.new('RGB', (1024, 1024), 'white')
    canvas.paste(PIL.Image.fromarray(row_image[0], 'RGB'), (0, 0))
    canvas.save(os.path.join(config.result_dir, 'mix.png'))
    # image = PIL.Image.fromarray(row_image, 'RGB')
    # w, h = image.size
    # row_image_resized = resize(w, h, 256, 256, image)
    img_open = Image.open(os.path.join(config.result_dir, 'mix.png'))
    w, h = img_open.size
    w_box = 256
    h_box = 256
    img_open_resized = resize(w, h, w_box, h_box, img_open)
    img_png3 = ImageTk.PhotoImage(img_open_resized)
    label3 = Label(window, image=img_png3).grid(row=1, column=2)
    # Label(window, image = row_image_resized).grid(row = 1, column = 2)

if __name__ == "__main__":
    main()


# canvas.create_image(2, 2, anchor = NW, image = img_png1)
# canvas.create_image(260, 2, anchor = NW, image = img_png1)
# canvas.create_image(520, 2, anchor = NW, image = img_png1)
# label_img1 = Label(window, image = img_png1, width=w_box, height=h_box).grid(row = 1, column = 0)
# label_img2 = Label(window, image = img_png1, width=w_box, height=h_box).grid(row = 1, column = 1)
# label_img3 = Label(window, image = img_png1, width=w_box, height=h_box).grid(row = 1, column = 2)
# label_img1.pack(padx = 5, pady = 5, side = BOTTOM)

