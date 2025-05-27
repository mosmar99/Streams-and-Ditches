import os
import math
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from fresh.models.unet import UNet
from mapio.unet import UNetDataset
from torch.utils.data import DataLoader
import mapio.graph_processing as graph_processing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
import umap
from sklearn.preprocessing import MinMaxScaler
import tifffile
import time
import tqdm
import pickle as pk
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

from skimage.filters import sobel, meijering, sato, frangi, hessian, gaussian, threshold_sauvola

def read_file_list(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def fit_pca(train_loader, best_model_path, logdir, n_components=4):
    test_model = UNet().to(device)
    test_model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_model.eval()

    intermediate = {}
    def get_features(name):
        def hook(model, input, output):
            intermediate[name] = output.detach().cpu().numpy()
        return hook
    
    test_model.down_conv5[-1].register_forward_hook(get_features("x9"))
    test_model.down_conv4[-1].register_forward_hook(get_features("x7"))
    test_model.up_conv_1[-1].register_forward_hook(get_features("u7"))

    pca_x9 = IncrementalPCA(n_components=n_components)
    pca_x7 = IncrementalPCA(n_components=n_components)
    pca_u7 = IncrementalPCA(n_components=n_components)

    for i, (batch_images, *_) in tqdm.tqdm(enumerate(train_loader)):
        # images, padding = test_model.add_padding(batch_images)
        batch_images = batch_images.to(device)

        # make predictions
        with torch.no_grad():
            _ = test_model.predict_softmax(batch_images)
            feats_x9 = intermediate["x9"]  # [B, C, H, W]
            feats_x7 = intermediate["x7"]  # [B, C, H, W]
            feats_u7 = intermediate["u7"]  # [B, C, H, W]
    
        Bx9, Cx9, Hx9, Wx9 = feats_x9.shape
        Bx7, Cx7, Hx7, Wx7 = feats_x7.shape
        Bu7, Cu7, Hu7, Wu7 = feats_u7.shape
        feats_reshaped_x9 = np.transpose(feats_x9, (0, 2, 3, 1)).reshape(-1, Cx9)  # [B*H*W, C]
        feats_reshaped_x7 = np.transpose(feats_x7, (0, 2, 3, 1)).reshape(-1, Cx7)  # [B*H*W, C]
        feats_reshaped_u7 = np.transpose(feats_u7, (0, 2, 3, 1)).reshape(-1, Cu7)  # [B*H*W, C]

        pca_x9.partial_fit(feats_reshaped_x9)
        pca_x7.partial_fit(feats_reshaped_x7)
        pca_u7.partial_fit(feats_reshaped_u7)
    
    pk.dump(pca_x9, open(os.path.join(logdir, "pca_x9.pkl"), "wb"))
    pk.dump(pca_x7, open(os.path.join(logdir, "pca_x7.pkl"), "wb"))
    pk.dump(pca_u7, open(os.path.join(logdir, "pca.pkl"), "wb"))

    return pca_x9, pca_x7, pca_u7

def apply_pca_transform(pca, features):
    B, C, H, W = features.shape
    feats_reshaped = np.transpose(features, (0, 2, 3, 1)).reshape(B * H * W, C)  # [B*H*W, C]

    # Apply PCA transform
    transformed = pca.transform(feats_reshaped)            # [B*H*W, n_components]
    transformed = transformed.reshape(B, H, W, -1)         # [B, H, W, n_components]
    transformed = np.transpose(transformed, (0, 3, 1, 2))  # [B, n_components, H, W]

    return transformed

def save_graph_data(j, img_file_name, nodes, connections, node_mask, 
                    argmax_pred_cpu, graph_dir, node_mask_dir):
    if nodes.shape[0] == 0:
        return
    deep_fmt = ('%.7f',)*(4*3)
    slope_fmt = ('%.7f',)*5
    twi_flowacc_fmt = ('%.7f',)*8
    node_fmt = ('%d', '%d', '%d', '%.7f', '%.7f', '%.7f',*slope_fmt, *twi_flowacc_fmt, *deep_fmt, '%d')
    if nodes.shape[0] != 0:
        np.savetxt(os.path.join(graph_dir, f"{img_file_name[j]}.nodes"), nodes, delimiter=",", fmt=node_fmt)
        np.savetxt(os.path.join(graph_dir, f"{img_file_name[j]}.edges"), connections, delimiter=",", fmt='%d')    
        np.savez_compressed(os.path.join(node_mask_dir, f"{img_file_name[j]}.npz"), image=node_mask, unet_pred=argmax_pred_cpu[j], image_name=img_file_name[j])

def main(logdir, fold="tf1", batch_size=8):
    train_fnames_path = f'./data/05m_folds/{fold}/train.dat'
    test_fnames_path = f'./data/05m_folds/{fold}/test.dat'
    valid_fnames_path = f'./data/05m_folds/{fold}/valid.dat'

    train_files = read_file_list(train_fnames_path)
    test_files = read_file_list(test_fnames_path)
    valid_files = read_file_list(valid_fnames_path)

    all_files = train_files + test_files + valid_files

    image_folder = './data/05m_chips/slope/'
    label_folder = './data/05m_chips/labels/'

    train_dataset = UNetDataset(all_files, image_folder, label_folder)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=8, pin_memory=True)

    # 20250513_161035
    # instantiate the model
    test_model = UNet().to(device)
    best_model_path = f'logs/UNETCV/{fold}best_unet.pth'
    test_model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_model.eval()

    intermediate = {}
    def get_features(name):
        def hook(model, input, output):
            intermediate[name] = output.detach().cpu().numpy()
        return hook
        
    test_model.down_conv5[-1].register_forward_hook(get_features("x9"))
    test_model.down_conv4[-1].register_forward_hook(get_features("x7"))
    test_model.up_conv_1[-1].register_forward_hook(get_features("u7"))

    # create a directory to save the predictions
    # pred_dir = os.path.join(logdir, 'predictions')
    graph_dir = os.path.join(logdir, 'graphs')
    node_mask_dir = os.path.join(logdir, 'reconstruction')
    # os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(node_mask_dir, exist_ok=True)

    # pca_x9, pca_x7, pca_u7 = fit_pca(train_loader, best_model_path, logdir, n_components=4)

    pca_x9 = pk.load(open(os.path.join(logdir,"pca_x9.pkl"),'rb'))
    pca_x7 = pk.load(open(os.path.join(logdir,"pca_x7.pkl"),'rb'))
    pca_u7 = pk.load(open(os.path.join(logdir,"pca.pkl"),'rb'))

    scaler = MinMaxScaler()

    # iterate over the train dataset
    print('Iterating over the train dataset...')
    with multiprocessing.Pool(processes=8) as pool, ThreadPoolExecutor(max_workers=16) as executor:
        for batch_images, batch_labels, img_file_name in tqdm.tqdm(train_loader):
            # images, padding = test_model.add_padding(batch_images)
            batch_images = batch_images.to(device)
            labels = batch_labels.squeeze(1).long().to(device)

            # make predictions
            with torch.no_grad():
                pred = test_model.predict_softmax(batch_images)
                argmax_pred = torch.argmax(pred, dim=1)

            pred_cpu = pred.cpu().numpy()
            argmax_pred_cpu = argmax_pred.cpu().numpy()
            labes_cpu = labels.cpu().numpy()
            batch_images_cpu = batch_images.cpu().numpy()

            deep_x9 = apply_pca_transform(pca_x9, intermediate["x9"])
            deep_x7 = apply_pca_transform(pca_x7, intermediate["x7"])
            deep_u7 = apply_pca_transform(pca_u7, intermediate["u7"])

            # deep_x9 = apply_pca_transform(umap_x9, intermediate["x9"])
            # deep_x7 = apply_pca_transform(umap_x7, intermediate["x7"])
            # deep_u7 = apply_pca_transform(umap_u7, intermediate["u7"])

            flow_acc = np.array([tifffile.imread(os.path.join("./data/05m_chips/flow_acc", f"{image}.tif")) for image in img_file_name])
            twi = np.array([tifffile.imread(os.path.join("./data/05m_chips/twi", f"{image}.tif")) for image in img_file_name])
            alt_label = np.array([tifffile.imread(os.path.join("./data/05m_chips/new_labels", f"{image}.tif")) for image in img_file_name])

            graphs = pool.starmap(graph_processing.image_to_graph, [
                (argmax_pred_cpu[i],
                pred_cpu[i],
                labes_cpu[i],
                deep_x9[i],
                deep_x7[i],
                deep_u7[i],
                batch_images_cpu[i],
                flow_acc[i],
                twi[i],
                alt_label[i])
                for i in range(pred_cpu.shape[0])
            ])

            _ = [scaler.partial_fit(nodes) for nodes, _, _ in graphs if nodes.shape[0] > 0]

            for j, (nodes, connections, node_mask) in enumerate(graphs):
                executor.submit(save_graph_data, j, img_file_name, nodes, connections,
                                node_mask, argmax_pred_cpu, graph_dir, node_mask_dir)

    pk.dump(scaler, open(os.path.join(logdir, "nodes_scaler.pkl"), "wb"))

if __name__ == '__main__':
    import time
    import argparse
    begin_time = time.time()

    # check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get the arguments
    parser = argparse.ArgumentParser(description='UNet Training')
    parser.add_argument('--logdir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--fold', type=str, default='tf1', help='What fold to use')
    args = parser.parse_args()
    logdir = args.logdir

    main(logdir, batch_size=8)
    
    end_time = time.time()
    print('Total time taken: {:.2f} min'.format((end_time - begin_time) / 60))