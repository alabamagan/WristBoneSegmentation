import argparse
import os
import logging
import numpy as np

from MedImgDataset import ImageDataSet2D, ImageFeaturePair, Landmarks
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from Networks import ConvNet
import torch.nn as nn
import torch.optim as optim
import torch
import visualization

def LogPrint(msg, level=20):
    logging.getLogger(__name__).log(level, msg)
    print msg

def visualizeResults(out, gt):
    """

    :param Variable out:
    :param Varialbe gt:
    :return:
    """
    visualization.VisualizeMapWithLandmarks(out.cpu().data.numpy(), gt.cpu().data.numpy(),
                              env="TOCI_run")
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
        inputDataset= ImageDataSet2D(a.input, dtype=np.float32, verbose=True)
        gtDataset   = Landmarks(a.train)
        trainingSet = ImageFeaturePair(inputDataset, gtDataset)
        loader      = DataLoader(trainingSet, batch_size=a.batchsize, shuffle=True, num_workers=4)

        # print inputDataset
        # return

        # Load Checkpoint or create new network
        #-----------------------------------------
        net = ConvNet(inputDataset[0].size()[1])
        if os.path.isfile(a.checkpoint):
            LogPrint("Loading checkpoint " + a.checkpoint)
            net.load_state_dict(torch.load(a.checkpoint))

        trainparams = {}
        if not a.trainparams is None:
            import ast
            trainparams = ast.literal_eval(a.trainparams)

        lr = trainparams['lr'] if trainparams.has_key('lr') else 1e-5
        mm = trainparams['momentum'] if trainparams.has_key('momentum') else 0.01

        criterion = nn.MSELoss()
        optimizer = optim.SGD([{'params': net.parameters(),
                                'lr': lr, 'momentum': mm}])
        if a.usecuda:
            criterion = criterion.cuda()
            net = net.cuda()
            # optimizer.cuda()

        losses = []
        for i in xrange(a.epoch):
            E = []
            for index, samples in enumerate(loader):
                if a.usecuda:
                    s = Variable(samples[0]).cuda()
                    g = Variable(samples[1]).cuda()
                else:
                    s, g = samples[0], samples[1].transpose(1, 2)
                out = net.forward(s.unsqueeze(1))
                # out = net.forward(s.transpose(1, 2).float()).squeeze().
                # print out
                loss = criterion(out,g.float())
                loss.backward()
                optimizer.step()
                E.append(loss.data[0])
                print "\t[Step %04d] Loss: %.010f"%(index, loss.data[0])
                if a.plot:
                    visualizeResults(s, out)
            losses.append(E)
            torch.save(net.state_dict(), "./Backup/checkpoint_ConvNet.pt")
            print "[Epoch %04d] Loss: %.010f"%(i, np.array(E).mean())

             # Decay learning rate
            if a.decay != 0:
                for pg in optimizer.param_groups:
                    pg['lr'] = pg['lr'] * np.exp(-i * float(a.epoch)  * a.decay / float(a.epoch))


    # Evaluation mode
    else:
        import SimpleITK as sitk

        if not os.path.isdir(a.output):
            try:
                LogPrint("Cannot find output directory, creating...")
                os.mkdir(a.output)
            except:
                LogPrint("Cannot create new directory")
                return
        assert os.path.isdir(a.output), "Ground truth directory cannot be opened!"
        inputDataset= ImageDataSet(a.input, dtype=np.float32)
        loader      = DataLoader(inputDataset, batch_size=a.batchsize, shuffle=False, num_workers=4)
        net = ResNet(inputDataset[0].size()[0], inputDataset[0].size()[0], 11)
        if os.path.isfile(a.checkpoint):
            LogPrint("Loading parameters " + a.checkpoint)
            net.load_state_dict(torch.load(a.checkpoint))
            net.training = False
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
            out = net.forward(s)
            for j in xrange(out.data.size()[0]):
                results.append(out[i].data.cpu().numpy())

        for i, r in enumerate(results):
            im = sitk.GetImageFromArray(r)
            metadata = inputDataset.metadata[i]
            im = ImageDataSet.WrapImageWithMetaData(im, metadata)
            outfname = a.output + "/" + os.path.basename(inputDataset.dataSourcePath[i])
            LogPrint("Writing to " + outfname)
            sitk.WriteImage(im, outfname)

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
