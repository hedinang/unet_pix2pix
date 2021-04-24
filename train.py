import numpy as np
from torch.nn import Module
import torch
import os
from torch.autograd import Variable
from networks import *


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Pix2PixHD(Module):
    def __init__(self, input_nc, label_nc=35, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d, num_D=3, getIntermFeat=False,
                 n_local_enhancers=1, lr=0.01, output_nc=3, feat_num=3, device='cuda'):
        netD_input_nc = label_nc + input_nc
        self.netD = MultiscaleDiscriminator(netD_input_nc)
        self.feat_num = feat_num
        self.num_D = 2
        self.n_layers_D = 3
        self.lambda_feat = 10.0
        if feat_num > 0:
            self.netE = Encoder(output_nc, feat_num, 16)
        self.criterionGAN = GANLoss()
        self.criterionFeat = torch.nn.L1Loss()
        self.criterionVGG = VGGLoss()
        if n_local_enhancers > 0:
            self.netG = LocalEnhancer(input_nc, output_nc)
            finetune_list = set()
            params_dict = dict(self.netG.named_parameters())
            params = []
            for key, value in params_dict.items():
                if key.startswith('model' + str(n_local_enhancers)):
                    params += [value]
                    finetune_list.add(key.split('.')[0])
        else:
            self.netG = GlobalGenerator(input_nc)
            params = list(self.netG.parameters())

        self.device = torch.device(device)
        self.netG.to(self.device)
        self.netG.apply(weights_init)
        self.netD.to(self.device)
        self.netD.apply(weights_init)
        if feat_num > 0:
            self.netE = Encoder(output_nc, feat_num, 16)
            self.netE.to(self.device)
            self.netE.apply(weights_init)
            params += list(self.netE.parameters())
        self.old_lr = lr
        self.lr = lr
        self.optimizer_G = torch.optim.Adam(params, lr=lr)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters, lr=lr)

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):
        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
        else:
            # create one-hot vector for label map
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(
                torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(
                1, label_map.data.long().cuda(), 1.0)
            if self.opt.data_type == 16:
                input_label = input_label.half()

        # get edges from instance map
        if not self.opt.no_instance:
            inst_map = inst_map.data.cuda()
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)
        input_label = Variable(input_label, volatile=infer)
        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        # instance map for feature encoding
        if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features:
                feat_map = Variable(feat_map.data.cuda())
            if self.opt.label_feat:
                inst_map = label_map.cuda()

        return input_label, inst_map, real_image, feat_map

    def forward(self, label, inst, image, feat):
        # Encode Inputs
        input_label, inst_map, real_image, feat_map = self.encode_input(
            label, inst, image, feat)
        # Fake Generation
        if self.feat_num > 0:
            if not self.opt.load_features:
                feat_map = self.netE.forward(real_image, inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else:
            input_concat = input_label
        fake_image = self.netG.forward(input_concat)
        # Fake Detection and Loss
        pred_fake_pool = self.netD.forward(
            torch.cat((input_label, fake_image), dim=1))
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)
        # Real Detection and Loss
        pred_real = self.netD.forward(
            torch.cat((input_label, real_image), dim=1))
        loss_D_real = self.criterionGAN(pred_real, True)
        # GAN loss (Fake Passability Loss)
        pred_fake = self.netD.forward(
            torch.cat((input_label, fake_image), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        feat_weights = 4.0 / (self.n_layers_D + 1)
        D_weights = 1.0 / self.num_D
        for i in range(self.num_D):
            for j in range(len(pred_fake[i])-1):
                loss_G_GAN_Feat += D_weights * feat_weights * \
                    self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
        loss_G_VGG = self.criterionVGG(
            fake_image, real_image) * self.opt.lambda_feat
        return [loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, fake_image]

    def inference(self, label, inst, image=None):
        # Encode Inputs
        image = Variable(image) if image is not None else None
        input_label, inst_map, real_image, _ = self.encode_input(
            Variable(label), Variable(inst), image, infer=True)

        # Fake Generation
        if self.use_features:
            if self.opt.use_encoded_image:
                # encode the real image to get feature map
                feat_map = self.netE.forward(real_image, inst_map)
            else:
                # sample clusters from precomputed features
                feat_map = self.sample_features(inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else:
            input_concat = input_label

        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)
        else:
            fake_image = self.netG.forward(input_concat)
        return fake_image

    def sample_features(self, inst):
        # read precomputed feature clusters
        cluster_path = os.path.join(
            self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)
        features_clustered = np.load(cluster_path, encoding='latin1').item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)
        feat_map = self.Tensor(
            inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0])

                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):
                    feat_map[idx[:, 0], idx[:, 1] + k, idx[:, 2],
                             idx[:, 3]] = feat[cluster_idx, k]
        if self.opt.data_type == 16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num//2, :]
            val = np.zeros((1, feat_num+1))
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (
            t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-
                                  1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (
            t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1,
                                  :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        if self.opt.data_type == 16:
            return edge.half()
        else:
            return edge.float()

    def save_network(self, network, label):
        filename = '{}.pth'.format(label)
        torch.save(network.state_dict(), filename)

    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if network_label == 'G':
                raise('Generator must exist!')
        else:
            # network.load_state_dict(torch.load(save_path))
            try:
                network.load_state_dict(torch.load(save_path))
            except:
                pretrained_dict = torch.load(save_path)
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {
                        k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    if self.opt.verbose:
                        print(
                            'Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print(
                        'Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    if sys.version_info >= (3, 0):
                        not_initialized = set()
                    else:
                        from sets import Set
                        not_initialized = Set()

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])

                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)

    def save(self):
        self.save_network(self.netG, 'G')
        self.save_network(self.netD, 'D')
        if self.feat_num > 0:
            self.save_network(self.netE, 'E')

    def update_fixed_params(self):
        params = list(self.netG.parameters())
        if self.feat_num > 0:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.lr)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        self.old_lr = lr


model = Pix2PixHD(3)
optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D
niter = 100
niter_decay = 100
start_epoch = 0
epoch_iter = 0
niter_fix_global = 3
dataset = None
save_epoch_freq = 2
for epoch in range(start_epoch, niter + niter_decay + 1):
    for i, data in enumerate(dataset, start=epoch_iter):
        loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, fake_image = model(
            data['label'], data['inst'], data['image'], data['feat'])
        loss_D = (loss_D_fake + loss_D_real)
        loss_G = loss_G_GAN + loss_G_GAN_Feat + loss_G_VGG
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()
    if epoch % save_epoch_freq == 0:
        print('saving the model at epoch: ', epoch)
        model.module.save()
    if (niter_fix_global != 0) and (epoch == niter_fix_global):
        model.module.update_fixed_params()
    if epoch > niter:
        model.module.update_learning_rate()
