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
import visualization
from Networks import Inception3
from tensorboardX import SummaryWriter
import datetime
from tqdm import tqdm

# import your own newtork

def LogPrint(msg, level=20):
    logging.getLogger(__name__).log(level, msg)
    print msg

def visualizeResults(input, out, gt, env='Wraist'):
    val, index = torch.max(out, 1)
    visualization.Visualize2D(input.data.cpu(), env=env, prefix='Input', nrow=1)
    visualization.Visualize2D(gt.data.cpu(), env=env, prefix='GT', displayrange=[0, 2], nrow=1)
    visualization.Visualize2D(index.squeeze().data.cpu(), env=env, prefix="OUTPUT", displayrange=[0, 2], nrow=1)
    pass

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
        net = Inception3(num_classes=3, aux_logits=False)
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
            for index, samples in enumerate(tqdm(loader)):
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
        inputDataset= ImageDataSet(a.input, dtype=np.float32, verbose=True)
        inputDataset.LoadWithCatagories(a.catagoriesIndex)
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

        gt = []
        result = []
        for index, samples in enumerate(tqdm(loader)):
            if a.usecuda:
                s = Variable(samples[0], volatile=True).cuda()
            else:
                s = Variable(samples[0], volatile=True)

            gt.extend(samples[1].tolist())
            # out = F.log_softmax(net.forward(s.unsqueeze(1)))
            out = net.forward(s.float().unsqueeze(1))
            v, d = torch.max(out, 1)
            result.extend(d.data.tolist())
            del s, v, d

        result = np.array(result).astype('int')
        real = np.array(gt).astype('int')
        print np.sum(result == real) / float(len(result))
        print "0->0", np.sum(np.multiply(result == 0, real == 0))
        print "1->1", np.sum(np.multiply(result == 1, real == 1))
        print "2->2", np.sum(np.multiply(result == 2, real == 2))
        print "2->1", np.sum(np.multiply(result == 1, real == 2))
        print "2->0", np.sum(np.multiply(result == 0, real == 2))
        print "1->2", np.sum(np.multiply(result == 2, real == 1))
        print "1->0", np.sum(np.multiply(result == 0, real == 1))
        print "0->1", np.sum(np.multiply(result == 1, real == 0))
        print "0->2", np.sum(np.multiply(result == 2, real == 0))
        pass
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
    parser.add_argument("--stage", dest='stage', default=1, action='store', type=int,
                        help="Stage 1: Feature location, Stage2: TOCI classification")
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
