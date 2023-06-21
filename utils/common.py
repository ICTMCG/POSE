import sys
import os
import glob
import random
import importlib
import numpy as np
import pandas as pd

import torch
from torch.autograd import Function
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
import seaborn as sns

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_config(config_path):
    module = importlib.import_module(config_path)
    return module.Config()

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def read_annotations(data_path, debug=False, shuffle=True):
    lines = map(str.strip, open(data_path).readlines())
    data = []
    for line in lines:
        sample_path, label = line.split('\t')
        sample_path = sample_path.replace('fanglingfei','yangtianyun')
        label = int(label)
        data.append((sample_path, label))        
    # random.shuffle(data)
    if debug:
        data=data[:1000]
    return data

def get_train_paths(data_list, config_name, run_dir):
    train_data_path = os.path.join(data_list['data_path'], data_list['train_collection'], "annotations", data_list['train_collection'] + ".txt")
    val_data_path = os.path.join(data_list['data_path'], data_list['val_collection'], "annotations", data_list['val_collection'] + ".txt")
    model_dir = os.path.join(data_list['data_path'], data_list['train_collection'],  "models", data_list['val_collection'], config_name,
                             run_dir)
    return [model_dir, train_data_path, val_data_path]

def plot_confusion_matrix(confusion, labels_name, save_path):
    plt.figure(figsize=(12, 12))
    plt.imshow(confusion, cmap=plt.cm.Blues)  # 在特定的窗口上显示图像
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index - 0.3, second_index, confusion[first_index][second_index])
    plt.tight_layout()
    plt.savefig(save_path, format='png')
    plt.close()

def plot_ROC_curve(results, save_path):
    plt.figure(figsize=(8,8))
    lw = 2
    plt.plot(results['fpr'], results['tpr'], color='darkorange',
            lw=lw, label='ROC curve (area = %0.3f)' % results['AUROC']) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path)

def plot_hist(x1, x2, save_path):
    plt.figure(figsize=(8,8))
    plt.hist([x1, x2], rwidth=0.8, bins=50, alpha=0.7, label=['known','unknown'])
    plt.legend()
    plt.savefig(save_path)

def plot_hist_seaborn(x1, x2, save_path):
    plt.figure(figsize=(8,5))
    for d, legend in zip([x1, x2], ['known','unknown']):
        sns.kdeplot(d, shade=True, alpha=.3, label=legend)
    plt.legend()
    plt.savefig(save_path)

def tsne_analyze(features, labels, classes, save_path, feature_num = 4000, do_fit = True):

    feature_known, feature_unknown = features
    labels_known, labels_unknown = labels
    known_classes, unknown_classes = classes
    known_class_num, class_num = len(set(labels_known)), len(set(labels_known)) + len(set(labels_unknown))
    
    # feature sample 
    if feature_num is not None:
        feature_num_unknown = feature_num * len(set(unknown_classes)) // len(set(known_classes))
        feature_known = feature_known[:feature_num]
        feature_unknown = feature_unknown[:feature_num_unknown]
        labels_known = labels_known[:feature_num]
        labels_unknown = labels_unknown[:feature_num_unknown]
    
    features = np.vstack((feature_known, feature_unknown))
    labels = np.hstack((labels_known, labels_unknown + len(set(labels_known))))

    # TSNE fit
    save_dir = os.path.split(save_path)[0]
    if do_fit:
        embeddings = TSNE(n_jobs=4).fit_transform(features)
        if save_dir is not None:
            np.save(os.path.join(save_dir,'embeddings.npy'), embeddings)
    else:
        embeddings=np.load(os.path.join(save_dir,'embeddings.npy'))

    index = [i for i in range(len(embeddings))]
    random.shuffle(index)
    embeddings = np.array([embeddings[index[i]] for i in range(len(index))])
    labels = [labels[index[i]] for i in range(len(index))]
    
    # Draw TSNE results
    print(f">>> draw image begin")
    plt.figure(figsize=(8,8))
    plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
    plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
    plt.subplots_adjust(top=0.75,bottom=0.25,left=0,right=0.5,hspace=0,wspace=0) 
    
    vis_x, vis_y = embeddings[:, 0], embeddings[:, 1]
    colors = [plt.cm.tab20(i) if i<20 else plt.cm.tab20b(i-20) for i in range(40)]
    for lab in range(class_num):
        color = colors[lab]
        class_index = [j for j,v in enumerate(labels) if v == lab]
        if lab < known_class_num:
            plt.scatter(vis_x[class_index], vis_y[class_index], color = color, alpha=1, marker='*')
        else:
            plt.scatter(vis_x[class_index], vis_y[class_index], color = color, alpha=0.5, marker='o', s=5)
    plt.legend(known_classes + unknown_classes, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(save_path)


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

