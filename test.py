"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from util import util
import PIL
import time
from PIL import Image
import cv2
from data.base_dataset import BaseDataset, get_transform

class livf():
    def __init__(self):
     self.transform = get_transform(opt, grayscale=0)

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    modes=opt.lv
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    #dataset2 = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset=livf()

    #print(dataset2)
    #for i , data in enumerate(dataset2):
    #    print(i,data)
    #    print(data['A'])
    #    break
    #im_siz=dataset.gsize()
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    #web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    #if opt.load_iter > 0:  # load_iter is 0 by default
    #    web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    #print('creating web directory', web_dir)
    #webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    #if opt.eval:
    #    model.eval()
    if modes=='live':
        cap = cv2.VideoCapture(0) # says we capture an image from a webcam
        while True:
            _,frame = cap.read()
            cv2.imshow("original",frame)
            h,w=frame.shape[0],frame.shape[1]
            cv2_im = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im)
            pilim=dataset.transform(pil_im)
            pilim.unsqueeze_(0)
            #print(pilim.size())
            #print(pilim)
            pdict={'A':pilim,'A_paths':["./"]}
            model.set_input_vid(pdict)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            print(visuals.items())
            for label, im_data in visuals.items():
                im = util.tensor2im(im_data)
                im=cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
                im=cv2.resize(im,(w,h))
                cv2.imshow("enhanced",im)
                k=cv2.waitKey(5)
                if k==27:
                    break
            if k==27:
                break
        cv2.destroyAllWindows()
    else:
        cap = cv2.VideoCapture(opt.lv)
        if (cap.isOpened()== False):
          print("Error opening video stream or file")
        while(cap.isOpened()):
          ret, frame = cap.read()
          if ret == True:
            h,w=frame.shape[0]//2,frame.shape[1]//2
            frame=cv2.resize(frame,(w,h))
            cv2.imshow('original',frame)
            cv2_im = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(cv2_im)
            pilim=dataset.transform(pil_im)
            pilim.unsqueeze_(0)
            pdict={'A':pilim,'A_paths':["./"]}
            model.set_input_vid(pdict)  # unpack data from data loader
            model.test()# run inference
            visuals = model.get_current_visuals()  # get image results
            for label, im_data in visuals.items():
                im = util.tensor2im(im_data)
                im=cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
                im=cv2.resize(im,(w,h))
                cv2.imshow("enhanced",im)
                k=cv2.waitKey(5)
                if k==27:
                    break
            if k==27:
                break
          else:
            break
        cap.release()
        cv2.destroyAllWindows()


    #    #img_path = model.get_image_paths()     # get image paths
    #    if i % 1 == 0:  # save images to an html file
    #        print('processing (%04d)-th image... %s' % (i, img_path))
    #    og_width,og_height=im_siz
    #    save_images(webpage, visuals, img_path, aspect_ratio=float(og_height/og_width), width=og_width)
    #webpage.save()  # save the HTML
    #cap = cv2.VideoCapture(0) # says we capture an image from a webcam
    #_,cv2_im = cap.read()
    #cv2_im = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
    #pil_im = Image.fromarray(cv2_im)