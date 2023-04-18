
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.datasets as dsets
import utils
from torch.autograd import Variable
# need matlab to generate CVT lattice
import matlab.engine
from torchvision.transforms import transforms
from torchvision.utils import save_image

torch.manual_seed(123)

parser = argparse.ArgumentParser(description='PyTorch TWAE on celebA')
parser.add_argument('-batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('-epochs', type=int, default=60, help='number of epochs to train (default: 40)')
# The corresponding epochs of TWAE = iterations*batchsize*region/size of dataset
parser.add_argument('-iterations', type=int, default=1200, help='number of iterations for TWAE to train (default: 1200)')
parser.add_argument('-lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument('-dim_h', type=int, default=64, help='hidden dimension (default: 64)')
parser.add_argument('-n_z', type=int, default=64, help='hidden dimension of z (default: 64)')
parser.add_argument('-alpha', type=float, default=0.2, help='regularization coef to balance global and local latent loss (default: 0.2)')
parser.add_argument('-n_channel', type=int, default=3, help='input channels (default: 1)')
parser.add_argument("-Tessellation", type=bool, default=True, help='use tessellation technique or not (default: True)')
parser.add_argument("-regularization", type=bool, default=True, help='use of regularization for tessellation (default: True)')
parser.add_argument('-distance', default='SW', choices=['SW', 'GW', 'GSW','DSW'], 
                    help='list of distance in latent space (default:sliced Wasserstein distance)')
parser.add_argument('-architecture', default='B', choices=['A','B'], help='choice of architecture, A or B')
parser.add_argument('-region', type=int, default=100, help='number of regions in CVT (default: 100)') 
args = parser.parse_args()

dataroot = 'E:\optimlgenerator\data\celeba'
tf =transforms.Compose([
        transforms.CenterCrop(140),   
        transforms.Scale(64),
           
           transforms.ToTensor(),
           
       ])

   
dataset = dsets.ImageFolder(root=dataroot, transform=tf)
train_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            )
class EncoderB(nn.Module):
    def __init__(self,args):
        super(EncoderB, self).__init__()
        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z
        
        self.main = [
            
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(self.dim_h * 8, self.n_z , 4, 1, 0, bias=False),
        ]

        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x
    
    
class EncoderA(nn.Module):
    def __init__(self,args):
        super(EncoderA, self).__init__()
        self.n_channel = args.n_channel
        self.dim_h = args.dim_h*2
        self.n_z = args.n_z
        
        self.main = [
            
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)
        self.lin1=nn.Linear(1024*4*4,self.n_z)

    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        x=x.view(-1,1024*4*4)
        x=self.lin1(x)
        return x
    
    
class DecoderA(nn.Module):
    def __init__(self,args):
        super(DecoderA, self).__init__()
        self.n_channel = args.n_channel
        self.dim_h = args.dim_h*2
        self.n_z = args.n_z
        self.lin1 = nn.Linear(self.n_z,1024*8*8)
        self.main = [
            
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2,     self.dim_h, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h),
            nn.ReLU(True),
            nn.ConvTranspose2d(    self.dim_h,      self.n_channel, 5, 1, 2, bias=False),
            nn.Sigmoid()
        ]
        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)

    def forward(self, x):
        x=self.lin1(x)
        x=x.view(-1,1024,8,8)
        for layer in self.main:
            x = layer(x)
        return x
    
    
class DecoderB(nn.Module):
    def __init__(self,args):
        super(DecoderB, self).__init__()
        self.n_channel = args.n_channel
        self.dim_h = args.dim_h
        self.n_z = args.n_z
        self.main = [
            
            nn.ConvTranspose2d(     self.n_z, self.dim_h * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2,     self.dim_h, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h),
            nn.ReLU(True),
            nn.ConvTranspose2d(    self.dim_h,      self.n_channel, 4, 2, 1, bias=False),
            nn.Sigmoid()
        ]
        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)

    def forward(self, x):
        x = x.view(-1,64,1,1)
        for layer in self.main:
            x = layer(x)
        return x
    
    
def save_gradient(U):
    gradient = []
    for p in U.parameters():
        gradient.append(p.grad)
    return gradient


def combine_gradient(U,oldgrad,alpha):
    i=0
    for p in U.parameters():
        p.grad-=alpha*oldgrad[i]
        i=i+1
        
        
class TransformNet(nn.Module):
    def __init__(self, size):
        super(TransformNet, self).__init__()
        self.size = size
        self.net = nn.Sequential(nn.Linear(self.size,self.size))
    def forward(self, input):
        out =self.net(input)
        return out/torch.sqrt(torch.sum(out**2,dim=1,keepdim=True))
    
    
def loss_latent(latentnow,latentgen, distance):
    if distance=='SW':
        wasserstein_loss = utils.sliced_wasserstein_distance(latentnow,latentgen,1000,device = 'cuda')
    if distance=='GW':
        wasserstein_loss = 0.01*utils.second_wasserstein(latentnow,latentgen)
    if distance=='GSW':
        wasserstein_loss = utils.generalized_sliced_wasserstein_distance(latentnow,latentgen,1000,device = 'cuda')
        
    return wasserstein_loss


def main():
    if args.architecture=='A':
        encoder, decoder = EncoderA(args), DecoderA(args)
    if args.architecture=='B':
        encoder, decoder = EncoderB(args), DecoderB(args)
    criterion = nn.MSELoss()
    encoder.train()
    decoder.train()
    if torch.cuda.is_available():
        encoder, decoder = encoder.cuda(), decoder.cuda()
    
    # Optimizers
    enc_optim = optim.Adam(encoder.parameters(), lr=args.lr)
    dec_optim = optim.Adam(decoder.parameters(), lr=args.lr)
    
    # CVT generation need matlab to run cvtprogram.m
    if args.Tessellation:
        eng = matlab.engine.start_matlab()
        cvtdata = np.zeros((args.n_z,args.region))
        r = eng.cvtprogram(float(args.n_z),float(args.region))
        r = np.asarray(r)
        r = r[0]
        cvtdata = r
        norm = np.linalg.norm(cvtdata,axis=0)
        cvtdata = cvtdata/norm
        lattice = np.transpose(cvtdata)
        
    if args.distance=='DSW':
        transform_net = TransformNet(args.n_z).cuda()
        op_trannet = optim.Adam(transform_net.parameters(), lr=5*args.lr, betas=(0.5, 0.999))
        
    if args.Tessellation:
        for j in range(args.iterations):
            for i in range(args.region):
                train_iter = iter(train_loader)
                train_data = next(train_iter)
                if torch.cuda.is_available():
                    train_data = train_data[0].cuda()
                z = encoder(train_data)
                if i==0:
                    latent = z.detach().cpu()
                    x_data = train_data.detach().cpu()
                else:
                    latent = torch.cat((latent,z.detach().cpu()),0)
                    x_data = torch.cat((x_data,train_data.detach().cpu()),0)
                del train_data,z
            #assign encoded data to each region
            latent = latent.view(-1,args.n_z)
            latnup = latent.numpy()
            dmatrix = utils.distancematrix(latnup, lattice)
            match,cost = utils.assignment(dmatrix,args.batch_size)
            match = torch.from_numpy(match)
            #assign sampled data to each region
            latentdata = utils.sphere_sample_total(args.n_z,1,args.batch_size*args.region)
            dmatrix1 = utils.distancematrix(latentdata, lattice)
            match1,cost1 = utils.assignment(dmatrix1,args.batch_size)
            match1 = torch.from_numpy(match1)
            if args.regularization:
                engradnew = []
                degradnew = []
                engradold = []
                degradold = []
                
            for i in range(args.region):
                if args.regularization:
                    #compute gradient with a batch sampled from the whole support
                    xdata = x_data[i*args.batch_size:(i+1)*args.batch_size,:,:,:].cuda()
                    enc_optim.zero_grad()
                    dec_optim.zero_grad()
                    latentnow = encoder(xdata)
                    latentnow = latentnow.view(-1,args.n_z)
                    latentgen = utils.sphere_sample_total(args.n_z,1,args.batch_size)
                    latentgen = torch.from_numpy(latentgen).float().cuda()
                    if args.distance=='DSW':
                        wasserstein_loss = utils.distributional_sliced_wasserstein_distance(latentnow,latentgen,transform_net,op_trannet,device = 'cuda')
                    else:
                        wasserstein_loss = loss_latent(latentnow,latentgen, args.distance)
                    x_recon = decoder(latentnow)
                    recon_loss = criterion(x_recon, xdata)
                    total_loss = wasserstein_loss + recon_loss
                    total_loss.backward()
                    engradnew = save_gradient(encoder)
                    degradnew = save_gradient(decoder)            
                    enc_optim.zero_grad()
                    dec_optim.zero_grad()
                    if i!=0:
                        xdata = x_data[(i-1)*args.batch_size:i*args.batch_size,:,:,:].cuda()
                        latentnow = encoder(xdata)
                        latentnow = latentnow.view(-1,args.n_z)
                        latentgen = utils.sphere_sample_total(args.n_z,1,args.batch_size) 
                        latentgen = torch.from_numpy(latentgen).float().cuda()
                        if args.distance=='DSW':
                            wasserstein_loss = utils.distributional_sliced_wasserstein_distance(latentnow,latentgen,transform_net,op_trannet,device = 'cuda')
                        else:
                            wasserstein_loss = loss_latent(latentnow,latentgen, args.distance)
                        x_recon = decoder(latentnow)
                        recon_loss = criterion(x_recon, xdata)
                        total_loss = args.alpha*wasserstein_loss + args.alpha*recon_loss
                        total_loss.backward()
                
                #compute gradient with a batch sampled from a local region
                xdata = x_data[match==i,:,:,:].cuda()
                latentnow = encoder(xdata)
                latentnow = latentnow.view(-1,args.n_z)
                latentgen = latentdata[match1==i,:]
                
                latentgen = torch.from_numpy(latentgen).float().cuda()
                if args.distance=='DSW':
                    wasserstein_loss = utils.distributional_sliced_wasserstein_distance(latentnow,latentgen,transform_net,op_trannet,device = 'cuda')
                else:
                    wasserstein_loss = loss_latent(latentnow,latentgen, args.distance)
                x_recon = decoder(latentnow)
                recon_loss = criterion(x_recon, xdata)
                total_loss = wasserstein_loss + recon_loss
                total_loss.backward()
                if args.regularization:
                    if i!=0:
                        combine_gradient(encoder,engradold,args.alpha)
                        combine_gradient(decoder,degradold,args.alpha)
                enc_optim.step()
                dec_optim.step()
                if args.regularization:
                    engradold = engradnew
                    degradold = degradnew
            del latent,match,latentnow,latentgen,x_data,x_recon
            print("Iteration: [%d/%d], Reconstruction Loss: %.4f, Latent loss: %.4f" %
                          (j + 1, args.iterations, recon_loss.data.item(), wasserstein_loss.data.item()))
            
            # Test and Generation
            encoder.eval()
            decoder.eval()
            test_iter = iter(train_loader)
            test_data = next(test_iter)
                    
            z_real = encoder(Variable(test_data[0]).cuda())
            reconst = decoder(z_real)
            reconst = reconst.cpu().view(-1, 3, 64, 64)
    
            if not os.path.isdir('./TWAE_%s_CVT%d/reconst_images'%(args.distance, args.region)):
                os.makedirs('./TWAE_%s_CVT%d/reconst_images'%(args.distance, args.region))
    
            save_image(test_data[0].view(-1, 3, 64, 64), './TWAE_%s_CVT%d/reconst_images/AE_input.png'%(args.distance, args.region))
            save_image(reconst.data, './TWAE_%s_CVT%d/reconst_images/AE_images_%d.png' % (args.distance, args.region, j + 1))    
            
            z_fake = utils.sphere_sample_total(args.n_z, 1, 100) 
            z_fake = torch.from_numpy(z_fake).float()
            z_fake = z_fake.cuda()
            generate = decoder(z_fake)
            generate = generate.cpu().view(-1,3,64,64)
            if not os.path.isdir('./TWAE_%s_CVT%d/generate_images'%(args.distance, args.region)):
                os.makedirs('./TWAE_%s_CVT%d/generate_images'%(args.distance, args.region))
            save_image(generate.data, './TWAE_%s_CVT%d/generate_images/AE_images_%d.png' % (args.distance, args.region, j + 1))
            if (j+1)%20==0:
                torch.save(encoder,'./TWAE_%s_CVT%d/encoder_%d.pkl'%(args.distance, args.region,j+1))
                torch.save(decoder,'./TWAE_%s_CVT%d/decoder_%d.pkl'%(args.distance, args.region,j+1))
    else:
        #WAE without tessellation for comparison
        for j in range(args.epochs):
            for batch_idx, (train_data,_) in enumerate(train_loader):
                print(batch_idx)
                enc_optim.zero_grad()
                dec_optim.zero_grad()
                train_data = train_data.cuda()
                latentnow = encoder(train_data)
                latentnow = latentnow.view(-1,args.n_z)
                latentgen = utils.sphere_sample_total(args.n_z,1,args.batch_size)
                latentgen = torch.from_numpy(latentgen).float().cuda()
                if args.distance=='DSW':
                    wasserstein_loss = utils.distributional_sliced_wasserstein_distance(latentnow,latentgen,transform_net,op_trannet,device = 'cuda')
                else:
                    wasserstein_loss = loss_latent(latentnow,latentgen, args.distance)
                x_recon = decoder(latentnow)
                recon_loss = criterion(x_recon, train_data)
                total_loss = wasserstein_loss + recon_loss
                total_loss.backward()
                enc_optim.step()
                dec_optim.step()
            print("Epoch: [%d/%d], Reconstruction Loss: %.4f, Latent loss: %.4f" %
                          (j + 1, args.epochs, recon_loss.data.item(), wasserstein_loss.data.item()))
            encoder.eval()
            decoder.eval()
            test_iter = iter(train_loader)
            test_data = next(test_iter)
                    
            z_real = encoder(Variable(test_data[0]).cuda())
            reconst = decoder(z_real)
            reconst = reconst.cpu().view(-1, 3, 64, 64)
    
            if not os.path.isdir('./WAE_%s_CVT%d/reconst_images'%(args.distance, args.region)):
                os.makedirs('./WAE_%s_CVT%d/reconst_images'%(args.distance, args.region))
    
            save_image(test_data[0].view(-1, 3, 64, 64), './WAE_%s_CVT%d/reconst_images/AE_input.png'%(args.distance, args.region))
            save_image(reconst.data, './WAE_%s_CVT%d/reconst_images/AE_images_%d.png' % (args.distance, args.region, j + 1))    
            
            z_fake = utils.sphere_sample_total(args.n_z, 1, 100) 
            z_fake = torch.from_numpy(z_fake).float()
            z_fake = z_fake.cuda()
            generate = decoder(z_fake)
            generate = generate.cpu().view(-1,3,64,64)
            if not os.path.isdir('./WAE_%s_CVT%d/generate_images'%(args.distance, args.region)):
                os.makedirs('./WAE_%s_CVT%d/generate_images'%(args.distance, args.region))
            save_image(generate.data, './WAE_%s_CVT%d/generate_images/AE_images_%d.png' % (args.distance, args.region, j + 1))
            torch.save(encoder,'./WAE_%s_CVT%d/encoder_%d.pkl'%(args.distance, args.region,j+1))
            torch.save(decoder,'./WAE_%s_CVT%d/decoder_%d.pkl'%(args.distance, args.region,j+1))
if __name__ == "__main__":
    main()

