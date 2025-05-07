import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix
from thop import profile

from LGFIN import LGFIN as model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))
    data_transform = {
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }
    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    test_dataset = datasets.ImageFolder(root="/media/datasets/data/test", transform=data_transform["test"])
    test_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=nw)
    print("Using {} images for testing.".format(test_num))
    net = model(num_classes=num_classes)  
    net.to(device)

    net_weight_path = "/media/model.pth"
    # Clear unused GPU cache
    torch.cuda.empty_cache()
    
    # Load model with specified map_location
    state_dict = torch.load(net_weight_path, map_location=device)
    net.load_state_dict(state_dict, strict=False)
    print("Using model:", net_weight_path)

    # Calculate the number of parameters and FLOPs
    test_input = torch.randn(1, 3, 224, 224).to(device)  # Example input tensor
    macs, params = profile(net, inputs=(test_input,), verbose=False)
    print(f'Number of parameters: {params / 1e6:.3f} million')
    print(f'FLOPs: {macs / 1e9:.3f} billion')

    net.eval()
    
    # Initialize metrics
    acc = 0.0  # Accumulate accurate number / epoch
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for test_images, test_labels in test_bar:
            outputs = net(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            
            # Collect predictions and labels for AUC-ROC
            all_preds.extend(predict_y.cpu().numpy())
            all_labels.extend(test_labels.cpu().numpy())

            # Update accuracy
            acc += (predict_y == test_labels.to(device)).sum().item()

    # Convert to numpy arrays for further processing
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate Test Accuracy
    test_accurate = acc / test_num
    print(f'Test Accuracy: {test_accurate:.4f}')
    # Calculate Precision, Sensitivity, Specificity, F1-score
    cm = confusion_matrix(all_labels, all_preds)
    TP = cm[1, 1]  # Assuming class 1 is the positive class
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    recall = sensitivity  # Recall is the same as Sensitivity
    # Print all calculated metrics
    print('Overall Accuracy (OA): {:.3f}'.format(test_accurate))
    print('Precision: {:.3f}'.format(precision))
    print('Sensitivity (Recall): {:.3f}'.format(sensitivity))
    print('Specificity: {:.3f}'.format(specificity))
    print('F1-score: {:.3f}'.format(f1_score))
    print('Finished test')

if __name__ == '__main__':
    main()
