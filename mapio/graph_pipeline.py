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
import time

def read_file_list(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def main(logdir, epochs=42, batch_size=42):
    # read file names   
    train_fnames_path = './data/mapio_folds/f1/train_files.dat'
    train_files = read_file_list(train_fnames_path)

    image_folder = './data/05m_chips/slope/'
    label_folder = './data/05m_chips/labels/'

    train_dataset = UNetDataset(train_files, image_folder, label_folder)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=15, pin_memory=True)

    # 20250513_161035
    # instantiate the model
    test_model = UNet().to(device)
    best_model_path = 'logs/m1/20250515_134712/unet_model_ckpt.pth'
    test_model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_model.eval()

    intermediate = {}
    def get_features(name):
        def hook(model, input, output):
            intermediate[name] = output.detach().cpu().numpy()
        return hook
        
    test_model.down_conv5[-1].register_forward_hook(get_features("x9"))

    # create a directory to save the predictions
    pred_dir = os.path.join(logdir, 'predictions')
    graph_dir = os.path.join(logdir, 'graphs')
    # os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    # iterate over the train dataset
    print('Iterating over the train dataset...')
    for i, (batch_images, batch_labels) in enumerate(train_loader):
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

        graphs = [graph_processing.image_to_graph(argmax_pred_cpu[i], pred_cpu[i], labes_cpu[i], intermediate["x9"][i]) for i in range(pred_cpu.shape[0])]

        for j, (nodes, connections) in enumerate(graphs):
            deep_fmt = ('%.4f',)*512
            node_fmt = ('%d', '%d', '%d', '%.4f', '%.4f', '%.4f', *deep_fmt, '%d')
            if nodes.shape[0] != 0:
                np.savetxt(os.path.join(graph_dir, f"{i * batch_size + j}.nodes"), nodes, delimiter=",", fmt=node_fmt)
                np.savetxt(os.path.join(graph_dir, f"{i * batch_size + j}.edges"), connections, delimiter=",", fmt='%d')    

        # save the predictions
        # for j in range(pred.shape[0]):
        #     pred_image = pred[j].cpu().numpy().astype('uint8')
        #     pred_pil = Image.fromarray(pred_image, mode='L')
        #     pred_pil.save(os.path.join(pred_dir, f'pred_{i * batch_size + j}.png'))

    print('All predictions saved to', pred_dir)

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