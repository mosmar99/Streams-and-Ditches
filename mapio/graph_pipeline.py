import os
import math
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from unet import UNet, UNetDataset
from torch.utils.data import DataLoader
import graph_processing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
import tifffile
import time
import tqdm

def read_file_list(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def reduce_features_pca(x, n_components=64):
    """
    x: np.ndarray of shape [B, C, H, W]
    returns: np.ndarray of shape [B, n_components, H, W]
    """
    B, C, H, W = x.shape
    x_reshaped = np.transpose(x, (0, 2, 3, 1)).reshape(B, H*W, C)
    compressed_list = []
    for b in range(B):
        pca = PCA(n_components=n_components)
        compressed = pca.fit_transform(x_reshaped[b])
        compressed_list.append(compressed)
    compressed_stacked = np.stack(compressed_list)
    compressed_output = compressed_stacked.reshape(B, H, W, n_components)
    compressed_output = np.transpose(compressed_output, (0, 3, 1, 2))
    return compressed_output

def fit_pca(train_loader, best_model_path, n_components=4):
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
    
    return pca_x9, pca_x7, pca_u7

def apply_pca_transform(pca, features):
    B, C, H, W = features.shape
    feats_reshaped = np.transpose(features, (0, 2, 3, 1)).reshape(B * H * W, C)  # [B*H*W, C]

    # Apply PCA transform
    transformed = pca.transform(feats_reshaped)            # [B*H*W, n_components]
    transformed = transformed.reshape(B, H, W, -1)         # [B, H, W, n_components]
    transformed = np.transpose(transformed, (0, 3, 1, 2))  # [B, n_components, H, W]

    return transformed


def main(logdir, epochs=42, batch_size=42):
    train_fnames_path = './data/05m_folds_mapio/r1/f1/train.dat'
    test_fnames_path = './data/05m_folds_mapio/r1/f1/test.dat'

    train_files = read_file_list(train_fnames_path)
    test_files = read_file_list(test_fnames_path)

    all_files = train_files + test_files

    image_folder = './data/05m_chips/slope/'
    label_folder = './data/05m_chips/labels/'

    train_dataset = UNetDataset(all_files, image_folder, label_folder)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=15, pin_memory=True)

    # 20250513_161035
    # instantiate the model
    test_model = UNet().to(device)
    best_model_path = 'logs/unet_model_ckpt1.pth'
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

    pca_x9, pca_x7, pca_u7 = fit_pca(train_loader, best_model_path, n_components=4)

    # iterate over the train dataset
    print('Iterating over the train dataset...')
    for i, (batch_images, batch_labels, img_file_name) in tqdm.tqdm(enumerate(train_loader)):
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

        deep_pca_x9 = apply_pca_transform(pca_x9, intermediate["x9"])
        deep_pca_x7 = apply_pca_transform(pca_x7, intermediate["x7"])
        deep_pca_u7 = apply_pca_transform(pca_u7, intermediate["u7"])

        graphs = [graph_processing.image_to_graph(argmax_pred_cpu[i],
                                                  pred_cpu[i],
                                                  labes_cpu[i],
                                                  deep_pca_x9[i],
                                                  deep_pca_x7[i],
                                                  deep_pca_u7[i],
                                                  batch_images_cpu[i]) for i in range(pred_cpu.shape[0])]

        for j, (nodes, connections, node_mask) in enumerate(graphs):
            # print(nodes[:,:2].shape)
            # plt.figure(figsize=(8, 6))
            # plt.imshow(argmax_pred_cpu[j], vmin=0, vmax=2)

            # color_map = {0:"#ffffff",
            #           1:"#00BFFF",
            #           2:"#32CD32"}
            
            # colors = [color_map[i] for i in nodes[:,-1]]

            # plt.scatter(nodes[:,2], nodes[:,1], c=colors, s=3)
            # graph_processing.plot_graph_edges(plt, nodes[:,:3], connections)
            # plt.show()

            # fig, ax = plt.subplots(1,2,figsize=(8, 6))
            # ax[0].imshow(argmax_pred_cpu[j], vmin=0, vmax=2)
            # graph_processing.plot_graph_edges(ax[0], nodes[:,:3], connections)
            # ax[0].scatter(nodes[:,2], nodes[:,1], c=colors, s=3)

            # ax[1].imshow(labes_cpu[j], vmin=0, vmax=2)
            # graph_processing.plot_graph_edges(ax[1], nodes[:,:3], connections)
            # ax[1].scatter(nodes[:,2], nodes[:,1], c=colors, s=3)
            # plt.show()

            deep_fmt = ('%.7f',)*12
            node_fmt = ('%d', '%d', '%d', '%.7f', '%.7f', '%.7f', '%.7f', '%.7f', '%.7f', '%.7f', '%.7f', *deep_fmt, '%d')
            if nodes.shape[0] != 0:
                np.savetxt(os.path.join(graph_dir, f"{img_file_name[j]}.nodes"), nodes, delimiter=",", fmt=node_fmt)
                np.savetxt(os.path.join(graph_dir, f"{img_file_name[j]}.edges"), connections, delimiter=",", fmt='%d')    
                np.savez_compressed(os.path.join(node_mask_dir, f"{img_file_name[j]}.npz"), image=node_mask, unet_pred=argmax_pred_cpu[j], image_name=img_file_name[j])
        # save the predictions
        # for j in range(pred.shape[0]):
        #     pred_image = pred[j].cpu().numpy().astype('uint8')
        #     pred_pil = Image.fromarray(pred_image, mode='L')
        #     pred_pil.save(os.path.join(pred_dir, f'pred_{i * batch_size + j}.png'))

if __name__ == '__main__':
    import time
    import argparse
    begin_time = time.time()

    # check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get the arguments
    parser = argparse.ArgumentParser(description='UNet Training')
    parser.add_argument('--logdir', type=str, default='logs', help='Directory to save logs')
    args = parser.parse_args()
    logdir = args.logdir

    main(logdir, epochs=100, batch_size=8)
    
    end_time = time.time()
    print('Total time taken: {:.2f} min'.format((end_time - begin_time) / 60))