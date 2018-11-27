import argparse
import os
import logging
import numpy as np

from MedImgDataset import ImageDataSet
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from Networks import *
from tensorboardX import SummaryWriter
import datetime
from tqdm import tqdm

# import your own newtork

def LogPrint(msg, level=20):
    logging.getLogger(__name__).log(level, msg)
    print msg

def main(a):
    ##############################
    # Error Check
    #-----------------
    mode = 0 # Training Mode
    assert os.path.isdir(a.input), "Input data directory not exist!"
    if a.train is None:
        mode = 1 # Eval mode

    ##############################
    # Training Mode
    if not mode:
        assert os.path.isdir(a.train), "Ground truth directory cannot be opened!"
        inputDataset= ImageDataSet(a.input, dtype=np.float32, verbose=True)
        inputDataset.LoadWithCatagories(a.catagoriesIndex)
        # gtDataset   = ImageDataSet(a.train, dtype=np.uint8, verbose=True)
        # trainingSet = TensorDataset(inputDataset, gtDataset)
        loader      = DataLoader(inputDataset, batch_size=a.batchsize, shuffle=True, num_workers=4, drop_last=True)
                                 # sampler=sampler.WeightedRandomSampler(np.ones(len(trainingSet)).tolist(), a.batchsize*100))

        writer = SummaryWriter("/media/storage/PytorchRuns/Wraist_"+datetime.datetime.now().strftime("%Y%m%d_%H%M"))
        # Load Checkpoint or create new network
        #-----------------------------------------
        net = inception_v3(pretrained=False, num_classes=3, aux_logits=False)
        net.train(True)
        if os.path.isfile(a.checkpoint) :
            LogPrint("Loading checkpoint " + a.checkpoint)
            net.load_state_dict(torch.load(a.checkpoint))
        elif a.checkpoint != '':
            LogPrint("Cannot locate checkpoint!")
            return
            # net = torch.load(a.checkpoint)

        trainparams = {}
        if not a.trainparams is None:
            import ast
            trainparams = ast.literal_eval(a.trainparams)


        lr = trainparams['lr'] if trainparams.has_key('lr') else 1e-5
        mm = trainparams['momentum'] if trainparams.has_key('momentum') else 0.01


        criterion = nn.NLLLoss()
        optimizer = optim.SGD([{'params': net.parameters(),
                                'lr': lr, 'momentum': mm}])
        if a.usecuda:
            criterion = criterion.cuda()
            net = net.cuda()
            # optimizer.cuda()

        lastloss = 1e32
        losses = []
        for i in xrange(a.epoch):
            E = []
            for index, samples in enumerate(loader):
                if a.usecuda:
                    s = Variable(samples[0]).cuda()
                    g = Variable(samples[1]).cuda()
                else:
                    s, g = samples[0], samples[1]

                # out = F.log_softmax(net.forward(s.unsqueeze(1)))
                out = net.forward(s.float().unsqueeze(1))
                loss = criterion(out,g.long())

                loss.backward()
                optimizer.step()
                E.append(loss.data[0])
                tqdm.write("\t[Step %04d] Loss: %.010f"%(index, loss.data[0]))
                if a.plot:
                    writer.add_scalar('Wraist/Loss', loss.data[0], i * len(loader) + index)
                    v, d = torch.max(out, dim=1)
                    accuracy = np.sum(g.data.cpu().numpy() == d.data.cpu().numpy()) / float(len(d))
                    writer.add_scalar('Wraist/Accrucy', accuracy, i * len(loader) + index)
                    # visualizeResults(s, out, g, "Wraist_%02d"%a.useCatagory)

            losses.append(E)
            if np.array(E).mean() <= lastloss:
                backuppath = "./Backup/checkpoint_Inception.pt"
                torch.save(net.state_dict(), backuppath)
                lastloss = np.array(E).mean()
            tqdm.write("[Epoch %04d] Loss: %.010f"%(i, np.array(E).mean()))

             # Decay learning rate
            if a.decay != 0:
                for pg in optimizer.param_groups:
                    pg['lr'] = pg['lr'] * np.exp(-i * a.decay / float(a.epoch))

        writer.close()

    # Evaluation mode
    else:
        import category_parser as cp

        inputDataset= ImageDataSet(a.input, dtype=np.float32, verbose=True, loadBySlices=0, resize=True)
        loader      = DataLoader(inputDataset, batch_size=a.batchsize, shuffle=False, num_workers=4)

        net = Inception3(num_classes=3, aux_logits=False)
        net.train(False)

        if a.usecuda:
            net = net.cuda()

        try:
            LogPrint("Loading checkpoint " + a.checkpoint)
            net.load_state_dict(torch.load(a.checkpoint))
        except:
            LogPrint("Cannot load state dict, terminate,")

        result = []
        for index, samples in enumerate(tqdm(loader)):
            if a.usecuda:
                s = Variable(samples, requires_grad=False).cuda()
            else:
                s = Variable(samples, requires_grad=False)
            torch.no_grad()
            # out = F.log_softmax(net.forward(s.unsqueeze(1)))
            out = net.forward(s.float().unsqueeze(1))
            v, d = torch.max(out, 1)
            result.extend(d.data.cpu().tolist())
            del s, v, d, out

        result = np.array(result).astype('int')
        # Write result to csv
        if os.path.isdir(a.output):
            outputname = a.output + '/result.csv'
        else:
            outputname = a.output

        with open(outputname, 'w') as outfile:
            outfile.write('Name,NoLabel,Others,Hexagon\n')
            cursor = 0
            for i in xrange(len(inputDataset.dataSourcePath)):
                numOfSlices = int(inputDataset.metadata[i]['dim[3]'])
                r = list(result[cursor:cursor + numOfSlices])
                cp.check_category(r)
                s = cp.category2string(r)
                while s.count(',') < 2:
                    s += ','
                cursor += numOfSlices
                outfile.write('%s,'%os.path.basename(inputDataset.dataSourcePath[i]) + s + '\n')
            outfile.close()


    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training reconstruction from less projections.")
    parser.add_argument("input", metavar='input', action='store',
                        help="Train/Target input", type=str)
    parser.add_argument("-t", "--train", metavar='train', action='store', type=str, default=None,
                        help="Required directory with target data which serve as ground truth for training. Do no" 
                             "Set this to enable training mode.")
    parser.add_argument("-o", metavar='output', dest='output', action='store', type=str, default=None,
                        help="Set where to store outputs for eval mode")
    parser.add_argument("-p", dest='plot', action='store_true', default=False,
                        help="Select whether to disply the plot for stepwise loss")
    parser.add_argument("-d", "--decayLR", dest='decay', action='store', type=float, default=0,
                        help="Set decay halflife of the learning rates.")
    parser.add_argument("-e", "--epoch", dest='epoch', action='store', type=int, default=0,
                        help="Select network epoch.")
    parser.add_argument("-s", "--steps", dest='steps', action='store', type=int, default=1000,
                        help="Specify how many steps to run per epoch.")
    parser.add_argument("-b", "--batchsize", dest='batchsize', action='store', type=int, default=5,
                        help="Specify batchsize in each iteration.")
    parser.add_argument('-c', "--useCatagory", dest='useCatagory', action='store', type=int, default=0,
                        help="Set the catagory you wish to process. Must be used with option -C or --Catagories")
    parser.add_argument("-C", "--Catagories", dest='catagoriesIndex', action='store', type=str, default=None,
                        help="Use the catagory txt file to load the data.")
    parser.add_argument("--checkpoint-suffix", dest='checkpointSuffix', action='store', type=str, default='UNET',
                        help="Set the suffix of the checkpoint that will be saved in the Backup folder")
    parser.add_argument("--load", dest='checkpoint', action='store', default='',
                        help="Specify network checkpoint.")
    parser.add_argument("--useCUDA", dest='usecuda', action='store_true',default=False,
                        help="Set whether to use CUDA or not.")
    parser.add_argument("--train-params", dest='trainparams', action='store', type=str, default=None,
                        help="Path to a file with dictionary of training parameters written inside")
    parser.add_argument("--log", dest='log', action='store', type=str, default=None,
                        help="If specified, all the messages will be written to the specified file.")

    a = parser.parse_args()

    if a.log is None:
        if not os.path.isdir("./Backup/Log"):
            os.mkdir("./Backup/Log")
        if a.train:
            a.log = "./Backup/Log/run_%03d.log"%(a.epoch)
        else:
            a.log = "./Backup/Log/eval_%03d.log"%(a.epoch)

    logging.basicConfig(format="[%(asctime)-12s - %(levelname)s] %(message)s", filename=a.log)

    main(a)
