import pytorch_lightning as pl

from models.FDA import FDA_first_train

from utils.data import DataModuleSegmentation

from argparse import ArgumentParser, Namespace

import torch

from torchvision.utils import save_image

import os

def create_pseudo_labels_from_list_of_predictions(predictions_list, path_to_pseudo_label_folder):
    """
    Creates segmentation masks (pseudo labels) from predictions. For FDA, we create a segmentation mask from the mean of M model (M: number of segmentation models)

    predictions_list: EITHER list of torch.tensor containing the model's class prediction of each pixel in the image OR list of M lists of torch.tensor (this is the case if
    we use the mean of M model predicitions)

    """
    counter_for_img_name = 0
    if type(predictions_list[0]) == list:
    
        M = len(predictions_list)
        meaned_pred_list = []
        ordered_model_pred_list = []
        for i in range(len(predictions_list[0])):
            same_model_pred_list = []
            for j, _ in enumerate(predictions_list):
                model_pred = predictions_list[j][i]
                same_model_pred_list.append(model_pred)
            ordered_model_pred_list.append(same_model_pred_list)

        for prediction_on_same_img in ordered_model_pred_list:
            v = torch.sum(torch.stack(prediction_on_same_img), dim=0)
            v = v * (1/M)
            meaned_pred_list.append(v)

        print(len(meaned_pred_list))
        for prediction in meaned_pred_list:
            pred_sigmoid = torch.sigmoid(prediction, out=None) 
            seg_mask_mini_batch = torch.round(pred_sigmoid)
            for seg_mask_id in range(seg_mask_mini_batch.shape[0]):
                counter_for_img_name+=1
                print(counter_for_img_name)
                save_image(seg_mask_mini_batch[seg_mask_id], os.path.join(path_to_pseudo_label_folder, f"{counter_for_img_name}.png"))

    else:
        
        for prediction in predictions_list:
            pred_sigmoid = torch.sigmoid(prediction, out=None) 
            seg_mask_mini_batch = torch.round(pred_sigmoid)
            for seg_mask_id in range(seg_mask_mini_batch.shape[0]):
                counter_for_img_name+=1
                print(counter_for_img_name)
                save_image(seg_mask_mini_batch[seg_mask_id], os.path.join(path_to_pseudo_label_folder, f"{counter_for_img_name}.png"))

def main(hparams):

    model = FDA_first_train.load_from_checkpoint(hparams.path_to_checkpoint)

    # call after training
    trainer = pl.Trainer(accelerator='gpu', devices=1)

    predict_dataloader = DataModuleSegmentation(path_to_predict=hparams.path_to_predict_imgs_all, load_size=256)

    predictions = trainer.predict(model, dataloaders=predict_dataloader)

    # predictions_list = [predictions,predictions,predictions]

    create_pseudo_labels_from_list_of_predictions(predictions, hparams.path_to_pseudo_label_folder)

if __name__ == "__main__":

    parser = ArgumentParser(add_help=False)
    parser.add_argument("--path_to_predict_imgs_all", type=str)
    parser.add_argument("--path_to_checkpoint", type=str)
    parser.add_argument("--path_to_pseudo_label_folder", type=str)
    hparams = parser.parse_args()

    main(hparams)