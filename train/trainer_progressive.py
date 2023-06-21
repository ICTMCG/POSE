import os
import time
import numpy as np
import pandas as pd
import importlib

import torch
import torchvision.utils as vutils

from utils.logger import Progbar, AverageMeter
from utils.common import plot_ROC_curve, plot_hist, tsne_analyze, set_requires_grad
from utils.evaluation import evaluate_multiclass, metric_ood, compute_oscr, metric_cluster
from sklearn.metrics import confusion_matrix

from models.models import Simple_CNN
from models.augment_network import SingleLayer
from loss.loss import TripletLoss

class PGTrainer(): 
    def __init__(self, Data, device, config, opt, writer, logger, model_dir):

        # setup model, optimizer, scheduler
        self.model=Simple_CNN(class_num=config.class_num, out_feature_result=True).to(device)
        self.optimizer = torch.optim.Adam([{'params':self.model.parameters(), 'lr':config.init_lr_E},])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.step_size, gamma=config.gamma)

        # setup criterion
        Loss = importlib.import_module('loss.'+config.loss)
        self.criterion = getattr(Loss, config.loss)(config).to(device)
        self.criterionMetric = TripletLoss()
        self.criterionMSE = torch.nn.MSELoss()
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        # dataloader, configs, device, dir, logs
        self.opt = opt
        self.config = config
        self.device = device
        self.logger = logger
        self.writer = writer
        self.model_dir = model_dir
        self.train_loader = Data.train_loader
        self.tsne_loader = Data.tsne_loader
        self.board_num = 0


    def train_epoch_baseline(self, epoch):
        progbar = Progbar(len(self.train_loader), stateful_metrics=['epoch'])
        batch_time = AverageMeter()
        end = time.time()
        self.model.train()
        for batch_idx, batch in enumerate(self.train_loader):
            self.board_num += 1
            input_img_batch, label_batch, _ = batch 
            input_img = input_img_batch.reshape((-1, 3, input_img_batch.size(-2), input_img_batch.size(-1))).to(self.device)
            label = label_batch.reshape((-1)).to(self.device)
            losses = {}

            # -------- train classifier start -------- 
            self.optimizer.zero_grad()

            input_prob, input_fea = self.model(input_img, data=self.config.input_data)
            _, loss_cls = self.criterion(input_fea, input_prob, label)
            loss = loss_cls
            loss.backward()
 
            self.optimizer.step()
            self.scheduler.step()                 
            losses.update({'loss_cls':loss_cls.item()})
            # -------- train classifier end -------- 

            # -------- log and visualize -------- 
            progbar.add(1,values=[('epoch', epoch)]+[(loss_key,losses[loss_key]) for loss_key in losses.keys()]+[('lr', self.scheduler.get_lr()[0])])
            for loss_key in losses.keys():
                self.writer.add_scalars(loss_key, {'loss_key': losses[loss_key]}, self.board_num)
            batch_time.update(time.time() - end)
            end = time.time()

    def train_epoch_POSE(self, augnets, epoch):
        augnet = SingleLayer(inc=self.config.inc, kernel_size=self.config.kernel_size).to(self.device)
        optimizerA = torch.optim.Adam([{'params':augnet.parameters(), 'lr':self.config.augnet_lr}])
        schedulerA = torch.optim.lr_scheduler.StepLR(optimizerA, step_size=self.config.step_size, gamma=self.config.gamma)
        losses = {}

        progbar = Progbar(len(self.train_loader), stateful_metrics=['epoch'])
        batch_time = AverageMeter()
        end = time.time()
        for batch_idx, batch in enumerate(self.train_loader):
            self.board_num += 1
            input_img_batch, label_batch, _ = batch 
            input_img = input_img_batch.reshape((-1, 3, input_img_batch.size(-2), input_img_batch.size(-1))).to(self.device)
            label = label_batch.reshape((-1)).to(self.device)

            # -------- step1 :train augnet start -------- 
            optimizerA.zero_grad()
            set_requires_grad([augnet], True)
            set_requires_grad([self.model], False)

            # reconstruction loss
            aug_img = augnet(input_img.detach())
            loss_mse = self.criterionMSE(aug_img, input_img)
            loss_A = torch.clamp(loss_mse, self.config.mse_lowbound)
            losses.update({'loss_mse': loss_A.item()})

            _, aug_fea = self.model(aug_img, data=self.config.input_data)
            input_prob, input_fea = self.model(input_img, data=self.config.input_data)

            if len(augnets) >= 1:
                # close to known samples
                loss_close_known = torch.clamp(1 - torch.mean(self.cos(input_fea, aug_fea)), 1 - self.config.known_sim_limit) # larger similarity 
                losses.update({'loss_close_known': loss_close_known.item()})
                loss_A += self.config.w_close_known * loss_close_known

                # distant from previous augnets
                idx = np.random.randint(0, len(augnets))
                aug_img_pre = augnets[idx](input_img.detach())
                _, aug_fea_pre = self.model(aug_img_pre, data=self.config.input_data)
                loss_distant_preaug = torch.mean(self.cos(aug_fea, aug_fea_pre)) # smaller similarity 
                losses.update({'loss_distant_preaug': loss_distant_preaug.item()})
                loss_A += self.config.w_dist_pre * loss_distant_preaug

            loss_A.backward()
            optimizerA.step()
            schedulerA.step() 
            # -------- train augnet end -------- 

            # -------- step2 :train classifier start -------- 
            self.optimizer.zero_grad()
            set_requires_grad([self.model], True)
            set_requires_grad([augnet], False)
            
            input_prob, input_fea = self.model(input_img, data=self.config.input_data)
            input_prob, loss_cls = self.criterion(input_fea, input_prob, label)

            # get augnet data and labels
            aug_img = augnet(input_img) 
            augnet_label = self.config.class_num + label    
            augnet_label = augnet_label.to(self.device)            
            _, aug_fea = self.model(aug_img, data=self.config.input_data)
            merged_label = torch.cat([label, augnet_label])
            merged_fea = torch.cat([input_fea, aug_fea])

            # metric loss
            loss_metric = self.criterionMetric(merged_fea, merged_label)    
            losses.update({'loss_cls':loss_cls.item(), 'loss_metric':loss_metric.item()})        
            loss = loss_cls + loss_metric

            # classification on previous aug data
            if self.config.cls_pre and epoch >= self.config.start_cls_pre_epoch:
                idx = np.random.randint(0, len(augnets))
                aug_img_pre = augnets[idx](input_img)
                augnet_label_pre = self.config.class_num + label  
                _, aug_fea_pre = self.model(aug_img_pre, data=self.config.input_data)
                merged_label = torch.cat([label, augnet_label_pre])
                merged_fea = torch.cat([input_fea, aug_fea_pre])
                loss_metric_pre = self.criterionMetric(merged_fea, merged_label) 
                losses.update({'loss_metric_pre':loss_metric_pre.item()})
                loss += loss_metric_pre

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()                 
            # -------- train classifier end -------- 

            # -------- log and visualize -------- 
            progbar.add(1,values=[('epoch', epoch)]+[(loss_key,losses[loss_key]) for loss_key in losses.keys()]+[('lr', self.scheduler.get_lr()[0])])
            for loss_key in losses.keys():
                self.writer.add_scalars(loss_key, {'loss_key': losses[loss_key]}, self.board_num)
            if batch_idx == len(self.train_loader)-1:
                self.writer.add_image('input_img', vutils.make_grid(input_img[:self.config.class_num,:,:,:], normalize=True, scale_each=True), self.board_num)
                self.writer.add_image('aug_img', vutils.make_grid(aug_img[:self.config.class_num,:,:,:], normalize=True, scale_each=True), self.board_num)
            batch_time.update(time.time() - end)
            end = time.time()

        torch.save({'epoch': epoch, 'state_dict': augnet.state_dict()}, os.path.join(self.model_dir, 'augnet_{}.pth'.format(epoch)))

        return augnet


    def tsne_augnet(self, epoch, augnets, dataloader, run_type='tsne'):
        self.model.eval()
        progbar = Progbar(len(dataloader), stateful_metrics=['run-type'])
        print('augnets num', len(augnets))
        with torch.no_grad():
            features = None
            aug_features = None
            for i, batch in enumerate(dataloader):
                input_img_batch, label_batch, _ = batch 
                input_img = input_img_batch.reshape((-1, 3, input_img_batch.size(-2), input_img_batch.size(-1))).to(self.device)
                label = label_batch.reshape((-1)).to(self.device)
                input_img = input_img[label==1] # select one known class

                aug_imgs = []
                augnet_labels = []
                for idx, augnet in enumerate(augnets): 
                    aug_img = augnet(input_img)
                    aug_imgs.append(aug_img)
                    aug_label = idx * torch.ones(len(input_img))
                    augnet_labels.append(aug_label)
                aug_imgs = torch.cat(aug_imgs)
                augnet_labels=torch.cat(augnet_labels)

                _, feature = self.model(input_img, data=self.config.input_data)
                _, aug_feature = self.model(aug_imgs, data=self.config.input_data)

                progbar.add(1, values=[('run-type', run_type)])

                if i == 0:
                    all_augnet_labels = augnet_labels
                    features = feature.cpu().numpy()
                    aug_features = aug_feature.cpu().numpy()
                else:
                    all_augnet_labels = torch.cat([all_augnet_labels, augnet_labels])
                    features=np.vstack((features, feature.cpu().numpy()))
                    aug_features = np.vstack((aug_features, aug_feature.cpu().numpy()))

        known_labels = torch.zeros(len(features)).cpu().numpy()
        all_augnet_labels = all_augnet_labels.cpu().numpy()
        
        # plot TSNE 
        aug_classes = ['aug{}'.format(int(i)) for i in set(list(all_augnet_labels))]
        tsne_analyze([features, aug_features],[known_labels, all_augnet_labels], [['known'], aug_classes], \
                    save_path = os.path.join(self.model_dir,'epoch-{}-{}.png'.format(epoch, run_type)), feature_num=5000)


    def predict_set(self, dataloader, run_type='test'): 
        self.model.eval()
        progbar = Progbar(len(dataloader), stateful_metrics=['run-type'])
        with torch.no_grad():
            features = None
            for i, batch in enumerate(dataloader):
                input_img_batch, label_batch, _ = batch 
                input_img = input_img_batch.reshape((-1, 3, input_img_batch.size(-2), input_img_batch.size(-1))).to(self.device)
                label = label_batch.reshape((-1)).to(self.device)

                prob, feature = self.model(input_img, data=self.config.input_data)
                if 'open-set' not in run_type:
                    prob, loss_cls = self.criterion(feature, prob, label)
                    progbar.add(1, values=[('run-type', run_type),('loss_cls',loss_cls.item())])
                else:
                    prob, _ = self.criterion(feature, prob)
                    progbar.add(1, values=[('run-type', run_type)])

                if i == 0:
                    probs = prob
                    gt_labels = label
                    features = feature.cpu().numpy()
                else:
                    probs = torch.cat([probs, prob], dim=0)
                    gt_labels = torch.cat([gt_labels, label])
                    features=np.vstack((features, feature.cpu().numpy()))
    
        gt_labels = gt_labels.cpu().numpy()
        probs = probs.cpu().numpy()
        pred_labels = np.argmax(probs,axis=1)
        
        if 'open-set' in run_type:
            return features, gt_labels, probs
        else:
            results = evaluate_multiclass(gt_labels, pred_labels)
            CM = confusion_matrix(gt_labels, pred_labels)
            perf = round(results[self.config.metric], 4) * 100
            self.logger.info('%s results: %s' % (run_type, str(results)))
            self.logger.info('%s confusion matrix: %s' % (run_type, str(CM)))
            return features, gt_labels, probs, perf


    def test_out(self, epoch, feature_known, _labels_k, _pred_k, out_loader, unknown_classes, run_type, cluster=False):
        
        # predict on open-set data
        feature_unknown, _labels_u, _pred_u = self.predict_set(out_loader, run_type='open-set-{}'.format(run_type))
        
        # ood evaluation
        x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
        out_results = metric_ood(x1, x2)['Bas'] 
        _oscr_socre = compute_oscr(_pred_k, _pred_u, _labels_k)
        unknown_perf = round(out_results['AUROC'], 2)
        
        # plot confidence histogram and ROC curve
        plot_hist(x1,x2, save_path = os.path.join(self.model_dir,'epoch-{}-{}-hist.png'.format(epoch, run_type)))
        plot_ROC_curve(out_results, save_path = os.path.join(self.model_dir,'epoch-{}-{}-roc.png'.format(epoch, run_type)))

        # plot TSNE 
        if len(self.config.known_classes)+len(unknown_classes)<40:
            unknown_classes = ['unknown_{}'.format(unknown_class) for unknown_class in unknown_classes]
            known_classes = ['known_{}'.format(known_class) for known_class in self.config.known_classes]
            tsne_analyze([feature_known, feature_unknown],[_labels_k, _labels_u], [known_classes, unknown_classes], \
                    save_path = os.path.join(self.model_dir,'epoch-{}-{}-tsne.png'.format(epoch,run_type)), feature_num=1000)

        # calculate result for each unknown class
        out_result_details={}
        for i, label_u in enumerate(set(_labels_u)):
            pred_u = _pred_u[_labels_u==label_u]
            x1, x2 = np.max(_pred_k, axis=1), np.max(pred_u, axis=1)
            pred = np.argmax(pred_u, axis=1)
            pred_labels = list(set(pred))
            pred_nums = [np.sum(pred==p) for p in pred_labels]
            result = metric_ood(x1, x2, verbose=False)['Bas']
            self.logger.info("{}\t \t mostly pred class: {}\t \t average score: {}\t AUROC (%): {:.2f}".format(unknown_classes[i], 
                                                                                     self.config.known_classes[pred_labels[np.argmax(pred_nums)]],
                                                                                     np.mean(x2), result['AUROC']))
            out_result_details[str(i)] = {'unknown_class':'\t'+ unknown_classes[i],
                                        'pred_class': '\t'+ self.config.known_classes[pred_labels[np.argmax(pred_nums)]],
                                        'average_score':'\t'+ str(round(np.mean(x2),4)), 
                                        'AUROC':'\t'+ str(round(result['AUROC'],2))}
        # save detailed OSR results
        df = pd.DataFrame(out_result_details)    
        data = df.values
        data = list(map(list,zip(*data)))
        data = pd.DataFrame(data)
        data.to_csv(os.path.join(self.model_dir, 'epoch-{}_{}_result_details.csv'.format(epoch, run_type)), header = 0)
        
        if cluster:
            # calculate NMI, cluster_acc
            features = np.concatenate([feature_known, feature_unknown])
            labels = np.concatenate([_labels_k,  _labels_u+len(_labels_k)])
            class_num = len(set(_labels_k)) + len(set(_labels_u))
            NMI, cluster_acc, purity = metric_cluster(features, class_num, labels, self.config.cluster_method)
            self.logger.info("NMI: {:.2f}, cluster_acc: {:.2f}, purity: {:.2f}".format(NMI, cluster_acc, purity))

        self.logger.info("AUC: {:.2f}, OSCR: {:.2f}".format(unknown_perf, _oscr_socre*100))

        return unknown_perf, _oscr_socre*100


    def save_model(self, epoch, save_suffix='model.pth'):
            torch.save({
            'epoch': epoch,
            'model': self.model.__class__.__name__,
            'state_dict': self.model.state_dict(),
        }, os.path.join(self.model_dir, save_suffix))