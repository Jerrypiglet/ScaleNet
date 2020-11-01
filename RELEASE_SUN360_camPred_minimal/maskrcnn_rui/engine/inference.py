# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.utils.comm import is_main_process, get_world_size
from maskrcnn_benchmark.utils.comm import all_gather
from maskrcnn_benchmark.utils.comm import synchronize
from maskrcnn_benchmark.utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug


def compute_on_dataset(model, data_loader, device, bbox_aug, timer=None):
    model.eval()
    results_dict_cpu = {}
    results_dict_gpu = {}
    cpu_device = torch.device("cpu")
    for idx, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            if bbox_aug:
                output = im_detect_bbox_aug(model, images, device)
            else:
                output = model(images.to(device))
            if timer:
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output_cpu = [o.to(cpu_device) for o in output]
        results_dict_cpu.update(
            {img_id: result for img_id, result in zip(image_ids, output_cpu)}
        )
        # print('compute_on_dataset id', idx, results_dict_cpu.keys())
        # results_dict_gpu.update(
        #     {img_id: result for img_id, result in zip(image_ids, output)}
        # )
        print('compute_on_dataset id', idx, image_ids[0], output[0].bbox.device)
        # torch.cuda.synchronize()

    print('==============results_dict_cpu.keys()=', results_dict_cpu.keys())
    return results_dict_cpu


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu, return_dict=False, only_gather=False):
    if _dict_to_list is None:
        return
    if get_world_size()==1:
        return predictions_per_gpu
    all_predictions = all_gather(predictions_per_gpu)
    if only_gather:
        return all_predictions
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    if return_dict:
        return predictions

    return _dict_to_list(predictions)

def _dict_to_list(predictions):
    if predictions is None:
        return
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    # if len(image_ids) != image_ids[-1] + 1:
    #     logger = logging.getLogger("maskrcnn_benchmark.inference")
    #     logger.warning(
    #         "Number of images that were gathered from multiple processes is not "
    #         "a contiguous set. Some images might be missing from the evaluation"
    #     )
    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions



def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        bbox_aug=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, bbox_aug, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    print('>>>>>>==============results_dict_cpu.keys()=', len(predictions.keys()), predictions.keys())

    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    print('>>>>>><<<<<<<<<<<==============results_dict_cpu.keys()=', len(predictions))
    print(predictions[0])

    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
