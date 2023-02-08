from torchmetrics import Dice 
from torchmetrics.classification import MulticlassJaccardIndex
import pandas as pd

import torch
from torch.utils.data import DataLoader

from TedSeg import *
from cofo import *
from functools import partial


def get_validation_augmentation(img_size):
    train_transform = [
        albu.Resize(img_size, img_size)
    ]
    return albu.Compose(train_transform)


def test():
    # Set arguments
    IMG_SIZE = 224
    EMB_SIZE = 256
    PATH = "/content/drive/My Drive/Master_Thesis/CoFo/weights/CVC_to_ETIS/kvasir-endo_cofo.pth"
    path_base_target = "/content/drive/My Drive/Master_Thesis/datasets/PolypDatasets/ETIS-LaribPolypDB"
    path_csv_target_test = "/content/drive/My Drive/Master_Thesis/datasets/PolypDatasets/ETIS-LaribPolypDB/csv/test.csv"
    
    # Set and load model
    encoder_args = {
        "pretrained":False,           
    }

    model = CoFoUnet(emb_size=EMB_SIZE, encoder_args=encoder_args)

    model = torch.load(PATH)
    model.eval()

    # Set metrics
    seg_metric = Dice()
    seg_metric_IoU = MulticlassJaccardIndex()


    test = pd.read_csv(path_csv_target_test)
    test["image"] = [os.path.join(path_base_target, 'images', f"{index}.png") for index in test["image_id"]]
    test["mask"] = [os.path.join(path_base_target, 'masks', f"{index}.png") for index in test["image_id"]]
    test_ds = SegmentationDataset(test, get_validation_augmentation, IMG_SIZE, normalize=False)
    test_loader = DataLoader(test_ds, 4, num_workers=4)

    for xb, yb in test_loader:
            xb = xb.cuda()
            yb = yb.cuda()
            pred = model(xb)
            
            seg_metric.update(pred, yb.to(dtype=torch.uint8))
            seg_metric_IoU.update(pred, yb.to(dtype=torch.uint8))

            # dice = batch_dice_score(pred, yb)
            # dice = dice.detach().cpu().numpy()
            # dices.append(dice)
            # dice_score = np.concatenate(dices, axis=0)
            # b.comment = f'{dice_score.mean():.2f}'
    # return dice_score.mean()

    print("Dice Score: ", seg_metric.compute())
    
    print("IoU Score: ", seg_metric_IoU.compute())