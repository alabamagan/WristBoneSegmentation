import argparse
import os
import logging
import numpy as np

from MedImgDataset import ImageDataSet2D, ImageFeaturePair, Landmarks
from torch.utils.data import DataLoader, TensorDataset, sampler
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from Networks import ConvNet, WNet
import torch.nn as nn
import torch.optim as optim
import torch
import visualization
from skimage.io import imread

def LogPrint(msg, level=20):
    logging.getLogger(__name__).log(level, msg)
    print msg

def visualizeResults(out, gt):
    """

    :param Variable out:
    :param Varialbe gt:
    :return:
    """
    visualization.VisualizeMapWithLandmarks(out.cpu().data.numpy() * 255, gt.cpu().data.numpy(),
                              env="TOCI_run", N=45)
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
        assert os.path.isfile(a.train), "Ground truth directory cannot be opened!"
        if a.stage == 1:
            inputDataset= ImageDataSet2D(a.input, dtype=np.float32, verbose=True)
            gtDataset   = Landmarks(a.train)
            trainingSet = ImageFeaturePair(inputDataset, gtDataset)
            loader      = DataLoader(trainingSet, batch_size=a.batchsize, shuffle=True, num_workers=4)
                                     # sampler=sampler.WeightedRandomSampler(np.ones(len(trainingSet)).tolist(), a.batchsize*100))
        elif a.stage == 2:
            imreader = lambda x: imread(x, as_grey=False)[:,:,:2] # Discard last layer
            inputDataset = ImageFolder(a.input, loader=imreader)
            loader = DataLoader(inputDataset, batch_size=a.batchsize, shuffle=True, num_workers=4, drop_last=True)
            pass
        else:
            return


        # Load Checkpoint or create new network
        #-----------------------------------------
        net = ConvNet(inputDataset[0].size()[1]) if a.stage == 1 else WNet(5)
        net.train(True)
        if os.path.isfile(a.checkpoint):
            assert os.path.isfile(a.checkpoint)
            LogPrint("Loading checkpoint " + a.checkpoint)
            net.load_state_dict(torch.load(a.checkpoint))

        trainparams = {}
        if not a.trainparams is None:
            import ast
            trainparams = ast.literal_eval(a.trainparams)

        lr = trainparams['lr'] if trainparams.has_key('lr') else 1e-5
        mm = trainparams['momentum'] if trainparams.has_key('momentum') else 0.01


        if a.stage == 1:
            criterion = nn.SmoothL1Loss()
        elif a.stage == 2:
            criterion = nn.NLLLoss()
        else:
            pass
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

                if a.stage == 1:
                    out = net.forward(s.unsqueeze(1))
                    loss = criterion(out,g.float())
                elif a.stage == 2:
                    out = net.forward(s.permute(0, 3, 1, 2).float())
                    loss = criterion(out,g)

                # if a.stage == 2:
                #     val, ind = torch.max(out, 1)
                #     print ind, out
                # out = net.forward(s.transpose(1, 2).float()).squeeze().
                # print out
                loss.backward()
                optimizer.step()
                E.append(loss.data[0])
                print "\t[Step %04d] Loss: %.010f"%(index, loss.data[0])
                if a.plot:
                    if a.stage == 1:
                        visualizeResults(s, out)
                    elif a.stage == 2:
                        import pandas as pd
                        val, guess = torch.max(out, 1)
                        d = {'Guess': guess.cpu().data.numpy().tolist(),
                             'Ans': g.cpu().data.numpy().tolist()}
                        d = pd.DataFrame.from_dict(d)
                        print d.to_string()

            losses.append(E)
            if np.array(E).mean() < lastloss:
                backuppath = "./Backup/checkpoint_ConvNet.pt" if a.stage == 1 else "./Backup/checkpoint_WNet.pt"
                torch.save(net.state_dict(), backuppath)
                lastloss = np.array(E).mean()
            print "[Epoch %04d] Loss: %.010f"%(i, np.array(E).mean())

             # Decay learning rate
            if a.decay != 0:
                for pg in optimizer.param_groups:
                    pg['lr'] = pg['lr'] * np.exp(-i * a.decay / float(a.epoch))


    # Evaluation mode
    else:
        import pandas as pd
        if a.stage == 1:
            inputDataset= ImageDataSet2D(a.input, dtype=np.float32, verbose=True)
            loader      = DataLoader(inputDataset, batch_size=a.batchsize, shuffle=False)
            net = ConvNet(inputDataset[0].size()[1])
        else:
            imreader = lambda x: imread(x, as_grey=False)
            inputDataset = ImageDataSet2D(a.input, dtype=np.uint8, verbose=True, as_grey=False, readfunc=imreader)
            loader      = DataLoader(inputDataset, batch_size=a.batchsize, shuffle=False)
            net = WNet(5)

        if os.path.isfile(a.checkpoint):
            LogPrint("Loading parameters " + a.checkpoint)
            net.load_state_dict(torch.load(a.checkpoint))
            net.train(False)
        else:
            LogPrint("Parameters file cannot be opened!")
            return


        if a.usecuda:
            net = net.cuda()

        results = []
        for i, samples in enumerate(loader):
            s = Variable(samples)
            if a.usecuda:
                s = s.cuda()
            out = net.forward(s.unsqueeze(1)).squeeze() if a.stage == 1 else net.forward(s.permute(0, 3, 1, 2)[:,:2].float())
            if a.stage == 1:
                for j in xrange(out.data.size()[0]):
                    results.append(out[j].data.cpu().numpy())
            else:
                val, guess = torch.max(out, 1)
                results.append(guess.cpu().data.numpy().squeeze())
                del val, guess

            if a.plot:
                if a.stage == 1:
                    visualizeResults(s, out)


        if a.stage == 1:
            outdict = {'File': [], 'Proximal Phalanx': [], 'Metacarpal': [], 'Distal Phalanx': []}
            for i, res in enumerate(results):
                outdict['File'].append(os.path.basename(inputDataset.dataSourcePath[i]))
                outdict['Proximal Phalanx'].append(np.array(res[0], dtype=int).tolist())
                outdict['Metacarpal'].append(np.array(res[1], dtype=int).tolist())
                outdict['Distal Phalanx'].append(np.array(res[2], dtype=int).tolist())
            data = pd.DataFrame.from_dict(outdict)
            data = data[['File', 'Proximal Phalanx', 'Metacarpal', 'Distal Phalanx']]
            data.to_csv(a.output, index=False)
        else:
            results = np.concatenate(results, 0).squeeze()
            print results.shape
            outdict = {'File': [os.path.basename(p) for p in  inputDataset.dataSourcePath],
                       'Guess': ["TOCI%s"%(i + 4) for i in results.tolist()]}
            data = pd.DataFrame.from_dict(outdict)
            data.to_csv(a.output, index=False)
            print data.to_string



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
