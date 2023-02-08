import torch
import matplotlib
import matplotlib.pyplot as plt

def plot_1d_unormalized_function(f,range = [-10,10], bins=200):
    tt =torch.linspace(range[0],range[1],200)
    with torch.no_grad():
        values = f(tt)
        plt.plot(tt, values*bins/(torch.sum(values)*(range[1] - range[0])))

def plot_1d_unormalized_values(values,tt):
    x_min, x_max, bins = tt[0], tt[-1], tt.shape[0]
    plt.plot(tt, values*bins/(torch.sum(values)*(x_max - x_min)))

def plot_2d_function(f,range = [[-10,10],[-10,10]], bins = [50,50], alpha = 0.7):
    with torch.no_grad():
        tt_x = torch.linspace(range[0][0], range[0][1], bins[0])
        tt_y = torch.linspace(range[1][0],range[1][1], bins[1])
        mesh = torch.cartesian_prod(tt_x, tt_y)
        with torch.no_grad():
            plt.pcolormesh(tt_x,tt_y,f(mesh).numpy().reshape(bins[0],bins[1]).T, cmap = matplotlib.cm.get_cmap('viridis'), alpha = alpha, lw = 0)

def plot_likelihood_function(log_likelihood, range = [[-10,10],[-10,10]], bins = [50,50,], levels = 2 , alpha = 0.7):
    with torch.no_grad():
        tt_x = torch.linspace(range[0][0], range[0][1], bins[0])
        tt_y = torch.linspace(range[1][0],range[1][1], bins[1])
        tt_x_plus = tt_x.unsqueeze(0).unsqueeze(-1).repeat(tt_y.shape[0],1,1)
        tt_y_plus = tt_y.unsqueeze(1).unsqueeze(-1).repeat(1,tt_x.shape[0], 1)
        with torch.no_grad():
            plt.contourf(tt_x,tt_y,torch.exp(log_likelihood(tt_y_plus, tt_x_plus)), levels = levels, cmap = matplotlib.cm.get_cmap('viridis'), alpha = alpha, lw = 0)

def plot_2d_points(samples):
    plt.scatter(samples[:,0], samples[:,1])