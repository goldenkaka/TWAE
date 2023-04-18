
import numpy as np
from torch.autograd import Variable
import torch

def rand_projections(embedding_dim, num_samples=50):
    """This function generates `num_samples` random samples from the latent space's unit sphere.

        Args:
            embedding_dim (int): embedding dimensionality
            num_samples (int): number of random projection samples

        Return:
            torch.Tensor: tensor of size (num_samples, embedding_dim)
    """
    projections = [w / np.sqrt((w**2).sum())  # L2 normalization
                   for w in np.random.normal(size=(num_samples, embedding_dim))]
    projections = np.asarray(projections)
    return torch.from_numpy(projections).type(torch.FloatTensor).cuda()
def cost_matrix_slow(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)

def max_sliced_wasserstein_distance(first_samples,
                                    second_samples,
                                    p=2,
                                    max_iter=10,
                                    device='cuda'):
    theta = Variable(torch.randn(1, first_samples.shape[1]).cuda(), requires_grad=True)
    theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1))
    opt = torch.optim.Adam([theta], lr=1e-4)
    for _ in range(max_iter):
        encoded_projections = torch.matmul(first_samples, theta.transpose(0, 1))
        distribution_projections = torch.matmul(second_samples, theta.transpose(0, 1))
        wasserstein_distance = torch.abs((torch.sort(encoded_projections)[0] -
                                          torch.sort(distribution_projections)[0]))
        wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p))
        #print(wasserstein_distance)
        l = - wasserstein_distance
        opt.zero_grad()
        l.backward(retain_graph=True)
        opt.step()
        theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1))
    theta_opt = theta.data    
    encoded_projections = torch.matmul(first_samples, theta_opt.transpose(0, 1))
    distribution_projections = torch.matmul(second_samples, theta_opt.transpose(0, 1))
    wasserstein_distance = torch.abs((torch.sort(encoded_projections)[0] -
                                      torch.sort(distribution_projections)[0]))
    wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p))

    return 0.01*wasserstein_distance
def circular_function(x1, x2, theta, r, p):
    cost_matrix_1 = torch.sqrt(cost_matrix_slow(x1, theta * r))
    cost_matrix_2 = torch.sqrt(cost_matrix_slow(x2, theta * r))
    wasserstein_distance = torch.abs((torch.sort(cost_matrix_1.transpose(0, 1), dim=1)[0] -
                                      torch.sort(cost_matrix_2.transpose(0, 1), dim=1)[0]))
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1. / p)
    return torch.pow(torch.pow(wasserstein_distance, p).mean(), 1. / p)
def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mean(torch.abs(torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)))
def distributional_sliced_wasserstein_distance(first_samples, second_samples,  f, f_op, num_projections=1000,
                                               p=2, max_iter=10, lam=1, device='cuda'):
    embedding_dim = first_samples.size(1)
    pro = rand_projections(embedding_dim, num_projections).to(device)
    first_samples_detach = first_samples.detach()
    second_samples_detach = second_samples.detach()
    for _ in range(max_iter):
        projections = f(pro)
        cos = cosine_distance_torch(projections, projections)
        reg = lam * cos
        encoded_projections = first_samples_detach.matmul(projections.transpose(0, 1))
        distribution_projections = (second_samples_detach.matmul(projections.transpose(0, 1)))
        wasserstein_distance = torch.abs((torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                                          torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]))
        wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1. / p)
        wasserstein_distance = torch.pow(torch.pow(wasserstein_distance, p).mean(), 1. / p)
        loss = reg - wasserstein_distance
        f_op.zero_grad()
        loss.backward(retain_graph=True)
        f_op.step()

    projections = f(pro)
    encoded_projections = first_samples.matmul(projections.transpose(0, 1))
    distribution_projections = (second_samples.matmul(projections.transpose(0, 1)))
    wasserstein_distance = torch.abs((torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                                      torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]))
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1. / p)
    wasserstein_distance = torch.pow(torch.pow(wasserstein_distance, p).mean(), 1. / p)
    return 0.01*wasserstein_distance
def generalized_sliced_wasserstein_distance(first_samples,
                                            second_samples,
                                            r=1,
                                            num_projections=1000,
                                            p=2,
                                            device='cuda'):
    embedding_dim = first_samples.size(1)
    projections = rand_projections(embedding_dim, num_projections).to(device)
    return 0.01*circular_function(first_samples, second_samples, projections, r, p)

def _sliced_wasserstein_distance(encoded_samples,
                                 distribution_samples,
                                 num_projections=50,
                                 p=2,
                                 device='cpu'):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')

        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    # derive latent space dimension size from random samples drawn from latent prior distribution
    embedding_dim = distribution_samples.size(1)
    # generate random projections in latent space
    projections = rand_projections(embedding_dim, num_projections).to(device)
    # calculate projections through the encoded samples
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1).to(device))
    # calculate projections through the prior distribution random samples
    distribution_projections = (distribution_samples.matmul(projections.transpose(0, 1)))
    # calculate the sliced wasserstein distance by
    # sorting the samples per random projection and
    # calculating the difference between the
    # encoded samples and drawn random samples
    # per random projection
    wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
    # distance between latent space prior and encoded distributions
    # power of 2 by default for Wasserstein-2
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    # approximate mean wasserstein_distance for each projection
    return wasserstein_distance.mean()


def sliced_wasserstein_distance(encoded_samples,
                                transformed_samples,
                                num_projections=50,
                                p=2,
                                device='cpu'):
    """ Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')

        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    # derive batch size from encoded samples
    # draw random samples from latent space prior distribution

    # approximate mean wasserstein_distance between encoded and prior distributions
    # for each random projection
    swd = _sliced_wasserstein_distance(encoded_samples, transformed_samples,
                                       num_projections, p, device)
    return swd

def assignment(dmatrix,batchsize):
    datanum = dmatrix.shape[0]
    colre = np.tile(np.arange(dmatrix.shape[1]),(dmatrix.shape[0],1))
    colre = colre.reshape(dmatrix.shape[0]*dmatrix.shape[1])
    rowre = np.repeat(np.arange(dmatrix.shape[0]),dmatrix.shape[1])
    mask = np.zeros(dmatrix.shape[0]*dmatrix.shape[1])
    cl = np.zeros(dmatrix.shape[1])
    count = 0
    cost = 0
    match = np.zeros(dmatrix.shape[0])
    dmatrix = dmatrix.reshape(-1)
    dsort = np.argsort(dmatrix)
    iters = 0
    while count < datanum:
        num = dsort[iters]
        if mask[num]==0:
            match[rowre[num]] = colre[num]
            mask[rowre == rowre[num]] = 1
            cl[colre[num]] +=1
            if cl[colre[num]] == batchsize:   #100 is a max number
                mask[colre == colre[num]] = 1
            count = count + 1
            cost = cost + dmatrix[num]
        iters = iters+1
    return match,cost
def distancematrix(data,target):
    dmatrix = np.zeros(data.shape[0]*target.shape[0]).reshape(data.shape[0],target.shape[0])
    for i in range(target.shape[0]):
        distance = data-target[i,:]
        dmatrix[:,i] = np.linalg.norm(distance,axis=1)
    return dmatrix

def sphere_sample(dim,radiu,num,lattice,index):
    count = 0
    output = np.zeros((num,dim))
    j = 0
    while count < num:
        while True:
            sample = np.random.randn(int(dim))
            sample_norm = np.linalg.norm(sample)
            sample = sample/sample_norm*radiu*np.power(np.random.rand(1),1/dim)
            sample = sample + lattice[index,:]
            if np.linalg.norm(sample)<1:
                break
        saveind = 0
        
        for i in range(lattice.shape[0]):
            if i!= index:
                normal = lattice[i,:] - lattice[index,:]
                vector = sample - 0.5*(lattice[i,:] + lattice[index,:])
                if np.dot(normal,vector) > 0 :
                    saveind = 1
                    break
        
        j = j+1
        if saveind ==0:
            output[count,:] = sample
            count += 1
    print(j/50)            
    return output
def second_wasserstein(data1,data2):
    num1 = data1.shape[0]
    num2 = data2.shape[0]
    mean1 = torch.mean(data1,0)
    mean2 = torch.mean(data2,0)
    data1 = data1 - mean1
    data2 = data2 - mean2
    
    C1 = 1/num1*torch.mm(data1.t(),data1)
    C2 = 1/num2*torch.mm(data2.t(),data2)
    ident = torch.eye(data1.shape[1])
    if torch.cuda.is_available():
        ident = ident.cuda()
    C = torch.mm(C1,C2) + 0.0000000000001*ident
    U,S,V = torch.svd(C)
    meanloss = torch.norm(mean1-mean2,2)**2
    secondloss = torch.trace(C1+C2)-2*torch.sum(torch.sqrt(S))
    WD = meanloss + secondloss
    return WD
def sphere_sample_total(dim,radiu,num):
    sample = np.random.randn(int(num),int(dim))
    sample_norm = np.linalg.norm(sample,axis = 1).reshape(-1,1)
    sample = sample/sample_norm*radiu*np.power(np.random.rand(num,1),1/dim)
    return sample