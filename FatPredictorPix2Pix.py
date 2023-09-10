# Still need to get the dataloader working


print("Beginning execution of code")
import Utilities
import os
import LoadData
import FunctionGenerator
import CorruptImage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.profiler as profiler
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, normalize, resize
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
import random
import glob as glob
import time
from datetime import date
from IPython.display import HTML
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from multiprocessing import Pool
import multiprocessing as mp

print("Imported modules")
if torch.cuda.is_available():
    plt.switch_backend("Qt5Agg")
else:
    plt.switch_backend("MacOSX")
plt.ion()

manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)

# Hyper-parameters
imsize = [128, 128]
num_epochs = 30
batch_size = 250
batch_times = 5
num_augs = 5
learning_rate = 0.00001
iterations = 10
number_of_corruption_functions = 20
corruption_function_size = (100, imsize[0], imsize[1])
display = False
size_of_feature_maps = 64
ngpu = 1
print(imsize)

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""
 
    def __init__(self, input_nc=1, output_nc=1, nf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        # construct unet structure
        # add the innermost block
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True) 
        #print(unet_block)
 
        # add intermediate block with nf * 8 filters
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(nf * 8, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
 
        # gradually reduce the number of filters from nf * 8 to nf. 
        unet_block = UnetSkipConnectionBlock(nf * 4, nf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(nf * 2, nf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(nf, nf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
         
        # add the outermost block
        self.model = UnetSkipConnectionBlock(output_nc, nf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  
 
    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
 
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
 
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
 
        self.model = nn.Sequential(*model)
 
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class Discriminator(nn.Module):
    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(Discriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
 
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
 
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw), nn.Sigmoid()]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
 
    def forward(self, input):
        """Standard forward."""
        return self.model(input)

adversarial_loss = nn.BCELoss() 
l1_loss = nn.L1Loss()

def generator_loss(generated_image, target_img, G, real_target):
    gen_loss = adversarial_loss(G, real_target)
    l1_l = l1_loss(generated_image, target_img)
    gen_total_loss = gen_loss + (100 * l1_l)
    #print(gen_loss)
    return gen_total_loss

def discriminator_loss(output, label):
    disc_loss = adversarial_loss(output, label)
    return disc_loss

def TrainPix2Pix1(num_epochs, training_paths, num_augs, batch_size, target_loss=0, debug=False):
     
    if debug:
        display = True
        num_epochs = 600
        num_augs = 10
        batch_size = 50
        iterations = 5
        batch_times = 1
        number_of_corruption_functions = 2
        epochs_per_cor_funcs = 3
        cor_range = [0.1, 0.3]

        # Get folder pairs of fat and water series
        series_folders = Utilities.series_folders(training_paths)

        name = str(date.today()) + "Adam-" + str(learning_rate) + "-"

        # Get plots ready
        fig1, ax = plt.subplots(1, 3, figsize=(24, 6))
        ax[0].set_yscale("log")
        ax[0].set_xlim(0, num_epochs)
        ax[0].set_title(f"{iterations} iterations with {batch_size} slices with {num_augs} augmentations\n{number_of_corruption_functions} corruption functions with support {cor_range[0]} to {cor_range[1]} changing every {epochs_per_cor_funcs} epochs\nTraining Loss")
        ax[1].set_xlim(0, num_epochs)
        ax[1].set_title("Training Accuracy")
        ax[2].grid(False)
        ax[2].axis('off')
        ax[2].text(0, 0, f"{model}", fontsize='medium', linespacing=1, wrap=True)
        ax[2].set_title(name)
        fig1.tight_layout(pad=2.0)
        if display:
            fig4, ax4 = plt.subplots(3)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        D_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        D_loss_plot, G_loss_plot = [], []
        for epoch in range(1, num_epochs+1):

            if epoch % epochs_per_cor_funcs == 0:

                # Get corruption functions individually
                '''
                corruption_functions = []
                for i in range(number_of_corruption_functions):
                    print("Getting corruption function " + str(i + 1))
                    rand_frac = random.uniform(0.2, 0.3)
                    t1 = time.perf_counter()
                    func = FunctionGenerator.generate_smooth_function(corruption_function_size, fraction=rand_frac)
                    t2 = time.perf_counter()
                    print(t2 - t1)
                    corruption_functions.append(func)
                '''

                # Get corruption functions concurrently
                #t1 = time.perf_counter()
                corruption_functions = FunctionGenerator.generate_multiple_smooth_functions(number_of_corruption_functions, [corruption_function_size] * number_of_corruption_functions, [random.uniform(cor_range[0], cor_range[1]) for x in range(number_of_corruption_functions)], [10] * number_of_corruption_functions, [0.4] * number_of_corruption_functions)
                #t2 = time.perf_counter()
                #print(t2 - t1)
                print("Loaded Corruption Functions")

            if epoch % 100 == 0 and epoch != 0:
                if torch.cuda.is_available():
                    path = "/home/sauram/Documents/Research/MR-fatsup/Models/" + name + str(epoch) + "of" + str(num_epochs) + ".pth5"
                    torch.save(model.state_dict(), path)
                    
                    #prediction = model(torch.Tensor.float(image_slice[0].unsqueeze(0)).to(device))
                    #make_dot(prediction.mean(), params=dict(model.named_parameters()))

            plt.pause(0.0000001)

            print("Training on epoch " + str(epoch + 1))
            running_loss = 0.0
            correct = 0
            batch_accuracy = []
            dataloader_counter = 0
            for batch_number in range(iterations):
                
                # Load each image one after the other
                '''
                # Load in batch
                batch = []
                for slice_number in range(batch_size):
                    #grabbing = time.perf_counter()
                    image_slices = Utilities.prep_random_image_from_folder(series_folders, corruption_functions, num_augs, imsize)
                    #grabbed = time.perf_counter()
                    #print(grabbed - grabbing)
                    index = 0
                    for image_slice in image_slices[0:-2]:
                        if display and index == 0:
                            ax4[0].imshow(image_slice[0].squeeze(0), cmap='gray')
                            ax4[0].set_title(f'Image with Category {image_slice[1]:.4f}')
                            ax4[1].imshow(image_slices[-2], cmap='gray')
                            ax4[1].set_title(f'Water Image')
                            ax4[2].imshow(image_slices[-1], cmap='gray')
                            ax4[2].set_title(f'Fat Image')
                            fig4.tight_layout(pad=2.0)
                            fig4.canvas.draw()
                            fig4.canvas.flush_events()
                            #intake = input("Wait for image? ")
                            
                        if continuous:
                            batch.append(image_slice)
                        else: # Categorize factors
                            if image_slice[1] <= 0.05:
                                image_slice[1] = 0
                            #elif image_group[1] <= 0.1:
                                #image_group[1] = 1
                            #elif image_group[1] <= 0.30:
                            #    image_group[1] = 2
                            else:
                                image_slice[1] = 1
                            batch.append(image_slice)
                        index += 1 

                train_dataloader = DataLoader(batch, batch_size=batch_size * num_augs, shuffle=True)
                print("Done loading batch " + str(batch_number + 1))
                '''

                # Load images concurrently
                batch = []

                num_workers = mp.cpu_count()
                with Pool(processes=num_workers) as pool:
                    
                    # Submit tasks
                    async_results = [pool.apply_async(Utilities.prep_random_image_from_folderCNN, args=(series_folders, corruption_functions, num_augs, imsize)) for _ in range(batch_size)]
                    image_collection = [ar.get() for ar in async_results]

                    # Process results
                    for image_slices in image_collection:

                        index = 0
                        for image_slice in image_slices[0:-2]:
                            if display and index == 0:
                                ax4[0].imshow(image_slice[0].squeeze(0), cmap='gray')
                                ax4[0].set_title(f'Corrupted Image')
                                ax4[1].imshow(image_slices[-2], cmap='gray')
                                ax4[1].set_title(f'Water Image')
                                ax4[2].imshow(image_slices[-1], cmap='gray')
                                ax4[2].set_title(f'Fat Image')
                                fig4.tight_layout(pad=2.0)
                                fig4.canvas.draw()
                                fig4.canvas.flush_events()
                                intake = input("Wait for image? ")
                                
                            batch.append(image_slice)
                            index += 1 
                
                train_dataloader = DataLoader(batch, batch_size=batch_size * num_augs, shuffle=True)
                print("Done loading batch " + str(batch_number + 1))
        
                D_loss_list, G_loss_list = [], []
                for input_img, target_img in train_dataloader:
                    
                    D_optimizer.zero_grad()
                    input_img = input_img.to(device)
                    target_img = target_img.to(device)
            
                    # ground truth labels real and fake
                    real_target = Variable(torch.ones(input_img.size(0), 1, 30, 30).to(device))
                    fake_target = Variable(torch.zeros(input_img.size(0), 1, 30, 30).to(device))
                    
                    # generator forward pass
                    generated_image = generator(input_img)
                    
                    # train discriminator with fake/generated images
                    disc_inp_fake = torch.cat((input_img, generated_image), 1)
                    
                    D_fake = discriminator(disc_inp_fake.detach())
                    
                    D_fake_loss   =  discriminator_loss(D_fake, fake_target)
                    
                    # train discriminator with real images
                    disc_inp_real = torch.cat((input_img, target_img), 1)
                                            
                    D_real = discriminator(disc_inp_real)
                    D_real_loss = discriminator_loss(D_real,  real_target)
            
                
                    
                    # average discriminator loss
                    D_total_loss = (D_real_loss + D_fake_loss) / 2
                    D_loss_list.append(D_total_loss)
                    # compute gradients and run optimizer step
                    D_total_loss.backward()
                    D_optimizer.step()
                    
                    
                    # Train generator with real labels
                    G_optimizer.zero_grad()
                    fake_gen = torch.cat((input_img, generated_image), 1)
                    G = discriminator(fake_gen)
                    G_loss = generator_loss(generated_image, target_img, G, real_target)                                 
                    G_loss_list.append(G_loss)
                    # compute gradients and run optimizer step
                    G_loss.backward()
                    G_optimizer.step()