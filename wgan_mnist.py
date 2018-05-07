import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.autograd as autograd
import random
import torchvision.utils as vutils


use_cuda=True
torch.manual_seed(1)
DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
LAMBDA_2 = 2 # Consistency penalty lambda2 hyperparameter
Factor_M = 0
ITERS = 50000 # How many generator iterations to train for
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)
NOISE_SZ = 128


# ==================fucntion/class definitions======================
def weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal(m.weight.data)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.normal_(0.0, 0.2)
        torch.nn.init.xavier_uniform(m.weight.data)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        preprocess = nn.Sequential(
            nn.Linear(NOISE_SZ, 4*4*4*DIM,bias=True),
            #nn.BatchNorm1d(4*4*4*DIM),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4*DIM, 2*DIM, 5, stride=2,padding=1),
            #nn.BatchNorm2d(2*DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*DIM, DIM, 5, stride=2,padding=2,output_padding=1),
            #nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 1, 5, stride=2,padding=2,output_padding=1)

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4*DIM, 4, 4)
        #print output.size()
        output = self.block1(output)
        #print output.size()
        output = output[:, :, :7, :7]
        #print output.size()
        output = self.block2(output)
        #print output.size()
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        #print output.size()
        return output.view(-1, OUTPUT_DIM)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Conv2d(1, DIM, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(),
            nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),
            #nn.BatchNorm2d(2*DIM),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(),
            nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),
            #nn.BatchNorm2d(4*DIM),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(),
        )
        self.main = main
        self.output = nn.Linear(4*4*4*DIM, 1,bias=True)

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out2 = out.view(-1, 4*4*4*DIM)
        out = self.output(out2)
        return (out.view(-1),out2)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)


def calc_gradient_penalty(netD, real_data, fake_data,LAMBDA=LAMBDA,use_cuda=True):
    real_data_flat = real_data.view(fake_data.size())
    #print(real_data_flat.size())
    alpha = torch.rand(BATCH_SIZE,1).expand(real_data_flat.size())
    #alpha = alpha.expand(real_data.size())
    if use_cuda: alpha=alpha.cuda()

    interpolates = alpha * real_data_flat + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates,_ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty
    

def generate_image(netG):
    noise = torch.randn(BATCH_SIZE, 128)
    if use_cuda:
        noise = noise.cuda(gpu)
    noisev = autograd.Variable(noise, volatile=True)
    samples = netG(noisev)
    samples = samples.view(BATCH_SIZE, 28, 28)
    # print samples.size()

    samples = samples.cpu().data.numpy()

    lib.save_images.save_images(
        samples,
        'tmp/mnist/samples_{}.png'.format(frame)
)
# ================== main script ======================

netG = Generator()
netG.apply(weights_init)
netD = Discriminator()
netD.apply(weights_init)

print netG
print netD

if use_cuda:
    netD = netD.cuda()
    netG = netG.cuda()

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

one = torch.FloatTensor([1])
mone = one * -1
zero = torch.FloatTensor([0])
if use_cuda:
    one = one.cuda()
    mone = mone.cuda()
    zero = zero.cuda()

def dataset():
    while 1:
        for images,targets in train_loader:
            #print(images)
            yield images
data=dataset()
fixed_noise = Variable(torch.randn(100, NOISE_SZ).cuda(),requires_grad = False)

for iteration in xrange(ITERS):
    ############################
    # (1) Update D network
    ###########################
    netD.train()
    for p in netD.parameters():  
        p.requires_grad = True  
    for iter_d in xrange(CRITIC_ITERS):
        _data = data.next()
        real_data = torch.Tensor(_data)
        if use_cuda:
            real_data = real_data.cuda()
        real_data_v = autograd.Variable(real_data)

        netD.zero_grad()


        # train with fake
        noise = torch.randn(BATCH_SIZE, NOISE_SZ)
        if use_cuda:
            noise = noise.cuda()
        noise_v = autograd.Variable(noise, volatile=True)  
        fake = autograd.Variable(netG(noise_v).data)
        D_fake1_1,D_fake1_2 = netD(fake)
        D_fake1 = D_fake1_1.mean()
        D_fake1.backward((one))

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
        gradient_penalty.backward()
               
        # train with real
        D_real1_1,D_real1_2 = netD(real_data_v)
        D_real1 = D_real1_1.mean()      
        D_real1.backward((mone),retain_graph=True)
        
        # train with CT penalty
        D_real2_1,D_real2_2 = netD(real_data_v)

        ct_penalty = LAMBDA_2*((D_real1_1-D_real2_1)**2)       
        ct_penalty += LAMBDA_2*0.1*((D_real1_2-D_real2_2)**2).mean(dim=1)
        ct_penalty = torch.max(Variable(torch.zeros(ct_penalty.size()).cuda() if use_cuda else torch.zeros(ct_penalty.size())),ct_penalty-Factor_M)
        ct_penalty = ct_penalty.mean()
        #print(ct_penalty)
        ct_penalty.backward()

        D_cost = -D_real1 + D_fake1 + gradient_penalty + ct_penalty
        #print(-D_real1.data[0] , D_fake1.data[0] , gradient_penalty.data[0] ,ct_penalty.data[0])
        #D_cost.backward()
        Wasserstein_D = D_real1 - D_fake1
        optimizerD.step()

    #if not FIXED_GENERATOR:
        ############################
        # (2) Update G network
        ###########################
    netD.eval()
    for p in netD.parameters():
        p.requires_grad = False  
    netG.zero_grad()

    noise = torch.randn(BATCH_SIZE, NOISE_SZ)
    if use_cuda:
        noise = noise.cuda()
    noisev = autograd.Variable(noise,requires_grad=True) 
    fake = netG(noisev)
    G,_ = netD(fake)
    G = G.mean()
    G.backward(mone)
    G_cost = -G
    optimizerG.step()
    print((iteration,D_cost.data[0],Wasserstein_D.data[0]))
    if iteration%100==0:
        fake = netG(fixed_noise).view(-1, 1,28,28)
        vutils.save_image(fake.data,
                    'fake_samples_epoch_%03d.png' %  iteration,
                    normalize=True)

torch.save(netG.state_dict(), 'netG_mnist.pth')
torch.save(netD.state_dict(), 'netD_mnist.pth')

