import argparse
import os
import logging
import numpy as np

from MedImgDataset import ImageDataSet
from torch.utils.data import DataLoader, TensorDataset, sampler
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from Networks import UNet
from tqdm import tqdm

def LogPrint(msg, level=20):
    logging.getLogger(__name__).log(level, msg)
    print(msg)

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
        gtDataset   = ImageDataSet(a.train, dtype=np.uint8, verbose=True)
        trainingSet = TensorDataset(inputDataset, gtDataset)
        loader      = DataLoader(trainingSet, batch_size=a.batchsize, shuffle=True, num_workers=4)
                                 # sampler=sampler.WeightedRandomSampler(np.ones(len(trainingSet)).tolist(), a.batchsize*100))

        if a.useCatagory != 0:
            assert os.path.isfile(a.catagoriesIndex)
            inputDataset.UseCatagories(a.catagoriesIndex, a.useCatagory)
            gtDataset.UseCatagories(a.catagoriesIndex, a.useCatagory)

        # Load Checkpoint or create new network
        #-----------------------------------------
        net = UNet(2, in_channels=1, depth=5, start_filts=64, up_mode='upsample')
        net.train(True)
        if os.path.isfile(a.checkpoint) :
            LogPrint("Loading checkpoint " + a.checkpoint)
            net.load_state_dict(torch.load(a.checkpoint))
        elif a.checkpoint != '':
            LogPrint("Cannot locate checkpoint!")
            # net = torch.load(a.checkpoint)

        trainparams = {}
        if not a.trainparams is None:
            import ast
            trainparams = ast.literal_eval(a.trainparams)


        lr = trainparams['lr'] if 'lr' in trainparams else 1e-5
        mm = trainparams['momentum'] if 'momentum' in trainparams else 0.01

        criterion = nn.NLLLoss2d()
        optimizer = optim.SGD([{'params': net.parameters(),
                                'lr': lr, 'momentum': mm}])
        if a.usecuda:
            criterion = criterion.cuda()
            net = net.cuda()
            # optimizer.cuda()

        lastloss = 1e32
        losses = []
        for i in tqdm(range(a.epoch)):
            E = []
            for index, samples in enumerate(loader):
                if a.usecuda:
                    s = Variable(samples[0]).cuda()
                    g = Variable(samples[1]).cuda()
                else:
                    s, g = samples[0], samples[1]

                out = F.log_softmax(net.forward(s.unsqueeze(1)), dim=1)
                loss = criterion(out,g.long())


                loss.backward()
                optimizer.step()
                E.append(loss.data[0])
                tqdm.write("\t[Step %04d] Loss: %.010f"%(index, loss.data[0]))


            losses.append(E)
            if np.array(E).mean() <= lastloss:
                backuppath = "./Backup/checkpoint_NoCat.pt"
                torch.save(net.state_dict(), backuppath)
                lastloss = np.array(E).mean()
            tqdm.write("[Epoch %04d] Loss: %.010f"%(i, np.array(E).mean()))

             # Decay learning rate
            if a.decay != 0:
                for pg in optimizer.param_groups:
                    pg['lr'] = pg['lr'] * np.exp(-i * a.decay / float(a.epoch))


    # Evaluation mode
    else:
        import SimpleITK as sitk

        inputDataset= ImageDataSet(a.input, dtype=np.float32, verbose=True)
        net = UNet(2, in_channels=1, depth=5, start_filts=64, up_mode='upsample')

        indexes = []
        concat = []
        if os.path.isfile(a.catagoriesIndex):
            LogPrint("Using catagories...")
            # requires the checkpoints named as checkpoint_UNET_Cat_2.pt and checkpoint_UNET_Cat_3.pt in
            # the checkpoint directories
            if not os.path.isdir(a.checkpoint):
                LogPrint("Cannot open directory " + a.checkpoint)
                return

            for k, cp in enumerate(['checkpoint_UNET_Cat_2.pt','checkpoint_UNET_Cat_3.pt']):
                assert os.path.isfile(a.checkpoint + '/' + cp)
                LogPrint("Loading parameters " + a.checkpoint + "/" + cp)
                net.load_state_dict(torch.load(a.checkpoint + "/" + cp))
                net.train(False)
                inputDataset.UseCatagories(a.catagoriesIndex, k + 2)
                loader = DataLoader(inputDataset, batch_size=a.batchsize, shuffle=False)

                if a.usecuda:
                    net = net.cuda()

                results = []
                for i, samples in enumerate(loader):
                    s = Variable(samples)
                    if a.usecuda:
                        s = s.cuda()

                    torch.no_grad()
                    out = net.forward(s.unsqueeze(1))[0].squeeze()
                    out = F.log_softmax(out, dim=1)
                    if out.data.dim() == 3:
                        val, seg = torch.max(out, 0)
                        results.append(seg.data.cpu().numpy()[None, :])
                    elif out.data.dim() == 4:
                        val, seg = torch.max(out, 1)
                        results.append(seg.squeeze().data.cpu().numpy())
                    else:
                        LogPrint("Dimension of output is incorrect!")

                    del out, val, seg, samples
                concat.append(np.concatenate(results,0))
                indexes.append(inputDataset._itemindexes)
            concat = np.concatenate(concat,0)
            indexes = np.concatenate(indexes,0)

            for i in range(len(inputDataset.dataSourcePath)):
                LogPrint("Wroking on image: " + inputDataset.dataSourcePath[i])
                dim = [inputDataset.metadata[i]['dim[1]'],
                       inputDataset.metadata[i]['dim[2]'],
                       inputDataset.metadata[i]['dim[3]']]
                t = concat[indexes[:,0] == i]
                seg = np.zeros(dim, dtype=np.uint8)
                seg = seg.transpose(2,0,1)
                ind = indexes[indexes[:,0] == i]
                for j, l in enumerate(ind):
                    seg[l[2]-1] = t[j]

                im = ImageDataSet.WrapImageWithMetaData(seg, inputDataset.metadata[i])
                sitk.WriteImage(im, a.output + "/" + os.path.basename(inputDataset.dataSourcePath[i]))
        else:
            LogPrint("Loading parameters " + a.checkpoint)
            net.load_state_dict(torch.load(a.checkpoint))
            net.train(False)
            loader = DataLoader(inputDataset, batch_size=a.batchsize, shuffle=False)

            if a.usecuda:
                net = net.cuda()

            results = []
            for i, samples in enumerate(tqdm(loader)):
                s = Variable(samples, volatile=True)
                if a.usecuda:
                    s = s.cuda()
                out = net.forward(s.unsqueeze(1)).squeeze() if a.stage == 1 else net.forward(s.permute(0, 3, 1, 2)[:,:2].float())
                out = F.log_softmax(out, dim=1)
                if out.data.dim() == 3:
                    val, seg = torch.max(out, 0)
                    results.append(seg.data.cpu().numpy()[None, :])
                elif out.data.dim() == 4:
                    val, seg = torch.max(out, 1)
                    results.append(seg.squeeze().data.cpu().numpy())
                else:
                    LogPrint("Dimension of output is incorrect!")

                del out, val, seg, samples
            concat = np.concatenate(results,0)


            for i in range(len(inputDataset.dataSourcePath)):
                startindex = inputDataset._itemindexes[i]
                endindex = inputDataset._itemindexes[i + 1]
                LogPrint("Wroking on image: " + inputDataset.dataSourcePath[i])
                seg = np.copy(concat[startindex:endindex])

                im = ImageDataSet.WrapImageWithMetaData(seg.astype('uint8'), inputDataset.metadata[i])
                sitk.WriteImage(im, a.output + "/" + os.path.basename(inputDataset.dataSourcePath[i]))


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

    logging.basicConfig(format="[%(asctime)-12s - %(levelname)s] %(message)s", filename=a.log, level=logging.DEBUG)

    main(a)
