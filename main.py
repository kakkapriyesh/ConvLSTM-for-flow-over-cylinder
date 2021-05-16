
from Net import ED, Encoder, Decoder
from Config import convlstm_encoder_params, convlstm_decoder_params #parameters imported
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
from tqdm import tqdm #used for progress bar
import numpy as np
import argparse
import os
from torch import from_numpy, tensor
os.getcwd()
import matplotlib
matplotlib.use('agg')
TIMESTAMP = "2020-03-09T00-00-00"
parser = argparse.ArgumentParser()
parser.add_argument('-clstm',
                    '--convlstm',
                    help='use convlstm as base cell',
                    action='store_true')
parser.add_argument('--batch_size',
                    default=1,
                    type=int,
                    help='mini-batch size')
parser.add_argument('-lr', default=1e-4, type=float, help='G learning rate')
parser.add_argument('-frames_input',
                    default=5,
                    type=int,
                    help='sum of input frames')
parser.add_argument('-frames_output',
                    default=5,
                    type=int,
                    help='sum of predict frames')
parser.add_argument('-epochs', default=155, type=int, help='sum of epochs')
args = parser.parse_args()

random_seed = 199
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False   #For reproducibility

save_dir = './save_model/' + TIMESTAMP

TD_I = np.load('dataset_input_Re_500_to_Re10000_8_cases.npy')  #Input and output data
TD_T = np.load('dataset_target_Re_500_to_Re10000_8_cases.npy')
class Dataset():
    def __init__(self,X,Y):

        self.len = X.shape[0]
        self.x_data = from_numpy(X[:, :])
        self.y_data = from_numpy(Y[:])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = Dataset(TD_I,TD_T)
import matplotlib.pyplot as plt
trainFolder, validFolder = torch.utils.data.random_split(dataset, (190, 10)) #split of batch

trainLoader = torch.utils.data.DataLoader(trainFolder,
                                          batch_size=args.batch_size,
                                          shuffle=False)

validLoader = torch.utils.data.DataLoader(validFolder,
                                          batch_size=args.batch_size,
                                          shuffle=False)

encoder_params = convlstm_encoder_params
decoder_params = convlstm_decoder_params  #Obtaining parameters of the net (already imported)


def train():
    '''
    main function to run the training
    '''
    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda() #Conv_nets(0)#CLSTM(1) goes to encoder file
    decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
    net = ED(encoder, decoder)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)  #batchsize should be larger than no of GPU for this to work
    net.to(device)


    cur_epoch = 0
    lossfunction = nn.MSELoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      factor=0.5,
                                                      patience=4,
                                                      verbose=True)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # mini_val_loss = np.inf
    for epoch in range(cur_epoch, args.epochs + 1):
        ###################
        # train the model #
        ###################
        t = tqdm(trainLoader, leave=False, total=len(trainLoader)) #for bars
        for i, (inputVar,targetVar) in enumerate(t):
            inputs = inputVar.float().to(device)  # B,S,C,H,W
            label = targetVar.float().to(device)  # B,S,C,H,W
            optimizer.zero_grad()
            net.train()
            pred = net(inputs)  # B,S,C,H,W
            loss = lossfunction(pred, label)
            loss_aver = loss.item() / args.batch_size
            train_losses.append(loss_aver)
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
            optimizer.step()
            t.set_postfix({
                'trainloss': '{:.8f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })   #displays progress in the bar in terms of epochs and train loss

        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            net.eval()
            t = tqdm(validLoader, leave=False, total=len(validLoader))
            for i, (inputVar, targetVar) in enumerate(t):
                inputs = inputVar.float().to(device)
                label = targetVar.float().to(device)
                pred = net(inputs)
                loss = lossfunction(pred, label)
                loss_aver = loss.item() / args.batch_size
                # record validation loss
                valid_losses.append(loss_aver)
                v_min = torch.min(label).cpu()  #for setting limits in the plots
                v_max = torch.max(label).cpu()
                levels = np.linspace(v_min, v_max, 10)  #10 is the no of contours
                if epoch % 15 ==0 and i%1==0 :
                    fig, axarr = plt.subplots(3, label.shape[1],
                                          figsize=(label.shape[1] * 5, 10))
                    for t in range(label.shape[1]):

                        # print("max values ticks is",v_min,v_max)
                        ii = axarr[0][t].contourf(inputs[0, t, 0].detach().cpu().numpy(), vmin=v_min, vmax=v_max,levels=levels)
                        im = axarr[1][t].contourf(label[0, t, 0].detach().cpu().numpy(), vmin=v_min, vmax=v_max,levels=levels)
                        it = axarr[2][t].contourf(pred[0, t, 0].detach().cpu().numpy(),vmin=v_min, vmax=v_max,levels=levels)
                    cbaxei = fig.add_axes([0.92, 0.7, 0.03, 0.2])
                    cbari = fig.colorbar(ii, cax=cbaxei)
                    cbaxes = fig.add_axes([0.92, 0.4, 0.03, 0.2])
                    cbar = fig.colorbar(im, cax=cbaxes)
                    cbaxest = fig.add_axes([0.92, 0.1, 0.03, 0.2])
                    cbat = fig.colorbar(it, cax=cbaxest)
                    plt.savefig(os.path.join('{:03d}_{:05d}.png'.format(epoch, i)))
                    plt.close()

                    output = pred
                    outputs = output.float().to(device)
                    Z = np.zeros((4, 5, 1, 64, 64))
                    Z = torch.from_numpy(Z).float().to(device)
                    for i in range(4):
                        X = net(outputs).float().to(device)
                        Z[i, :, :, :, :] = X
                        outputs = X
                    fig, axarr = plt.subplots(6, Z.shape[1],
                                              figsize=(Z.shape[1] * 5, 25))
                    v_min = torch.min(label).cpu()
                    v_max = torch.max(label).cpu()
                    levels = np.linspace(v_min, v_max, 10)
                    for i in range(Z.shape[0]):
                        for t in range(Z.shape[1]):

                            ii1 = axarr[0][t].contourf(label[0, t, 0].detach().cpu().numpy(), vmin=v_min, vmax=v_max,
                                                      levels=levels)
                            im1 = axarr[1][t].contourf(pred[0, t, 0].detach().cpu().numpy(), vmin=v_min, vmax=v_max,
                                                      levels=levels)
                            it1 = axarr[i+2][t].contourf(Z[0, t, 0].detach().cpu().numpy(), vmin=v_min, vmax=v_max,
                                                      levels=levels)
                        cbax = fig.add_axes([0.92, 0.5, 0.05, 0.2])
                        cbari = fig.colorbar(ii, cax=cbax)
                        plt.savefig(os.path.join('Feed_Forward.png'.format(epoch)))


        torch.cuda.empty_cache()
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        epoch_len = len(str(args.epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} ' +
                     f'valid_loss: {valid_loss:.6f}')  #bar at the end
        with open("avg_train_losses.txt", 'wt') as f:
            for i in avg_train_losses:
                print(i,"",epoch, file=f)

        with open("avg_valid_losses.txt", 'wt') as f:
            for i in avg_valid_losses:
                print(i,"",epoch, file=f)

        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        pla_lr_scheduler.step(valid_loss)  # lr_scheduler
        pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print("no of parameters", pytorch_total_params)

if __name__ == "__main__":
    train()
