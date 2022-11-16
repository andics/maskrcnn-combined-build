# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
import datetime
import logging
import time
import utils_gen.layer_utils as layer_utils

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

try:
    from apex import amp
    use_amp = True
except ImportError:
    print('Use APEX for multi-precision via apex.amp')
    use_amp = False

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    use_amp,
    cfg,
    dllogger, #Used only for logging Loss, iterations ect.
    distributed,
    tensorboard_writer,
    iters_per_epoch,
    per_iter_end_callback_fn=None,
):

    from training.my_train_net import test_model

    freeze_layers_check_frequency = 20
    tensorboard_update_step_frequency = 20

    #-----Logging-----
    dllogger.log(step="PARAMETER", data={"train_start": True})
    meters = MetricLogger(delimiter="  ")
    #'logger' is used for genral purpose logging. It takes strings.
    logger = logging.getLogger(__name__)
    #-----------------

    max_iter = len(data_loader)
    start_iter = arguments["iteration"]

    model.train()
    start_training_time = time.time()
    end = time.time()

    if distributed:
        model_base_prefix = "model.module."
    else:
        model_base_prefix = "model."

    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        if iteration % freeze_layers_check_frequency == 0:
            layer_utils.check_layer_freeze_pattern(model_base_prefix, iteration, cfg, model, optimizer)

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)


        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        if use_amp:        
            with amp.scale_loss(losses, optimizer) as scaled_losses:
                scaled_losses.backward()
        else:
            losses.backward()

        if not cfg.SOLVER.ACCUMULATE_GRAD:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        else:
            if (iteration + 1) % cfg.SOLVER.ACCUMULATE_STEPS == 0:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data.div_(cfg.SOLVER.ACCUMULATE_STEPS)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        #-----WRITE-LOGS-----
        if iteration % 20 == 0 or iteration == max_iter:
            log_data = {"learning_rate":optimizer.param_groups[0]["lr"]}
            log_data.update(meters.get_dict())
            log_data.update({"eta":eta_string, "memory": torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 })
            dllogger.log(step=(iteration,), data=log_data)
        #---------------------

        #-----WRITE-TENSORBOARD-----
        if iteration % tensorboard_update_step_frequency == 0:
            tensorboard_writer.add_scalar('Loss/Average', meters.loss.avg, iteration)
            tensorboard_writer.add_scalar('Loss/Mask', meters.loss_mask.avg, iteration)
            tensorboard_writer.add_scalar('Loss/Box_reg', meters.loss_box_reg.avg, iteration)
            if cfg.TENSORBOARD.TARGET_LAYERS is not []:
                for layer in cfg.TENSORBOARD.TARGET_LAYERS:
                    layer_weight = eval(model_base_prefix + layer + ".weight.data.cpu().numpy().flat[0]")
                    tensorboard_writer.add_scalar('Weight/' + layer, layer_weight, iteration)
        #----------------------------

        if cfg.SAVE_CHECKPOINT:
            if iteration % checkpoint_period == 0:
                checkpointer.save("model_{:07d}".format(iteration), **arguments)

                if cfg.SOLVER.TEST_EVERY_CHECKPOINT:
                    logger.info("Model reached {:07d} checkpoint iterations! Evaluating its performance".format(iteration))
                    results = test_model(cfg=cfg, model=model, distributed=distributed, iters_per_epoch=iters_per_epoch, dllogger=dllogger, current_iterations=iteration)
                    tensorboard_writer.close()
                    # Add a break because training can't continue anyway due to the model being turned to evaluation mode
                    break

        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

        # per-epoch work (testing)
        if per_iter_end_callback_fn is not None:
            early_exit = per_iter_end_callback_fn(iteration=iteration)
            if early_exit:
                tensorboard_writer.close()
                break


    tensorboard_writer.close()
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    dllogger.log(step=tuple(), data={"e2e_train_time": total_training_time,
                                                   "train_perf_fps": max_iter * cfg.SOLVER.IMS_PER_BATCH / total_training_time})
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info(
    "Total training time: {} ({:.4f} s / it)".format(
        total_time_str, total_training_time / (max_iter)
        )
    )

