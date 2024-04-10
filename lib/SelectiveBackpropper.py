import backproppers
import calculators
import fp_selectors
import loggers
import selectors
import time
import torch
import torch.nn as nn
import trainer as trainer

start_time_seconds = time.time()

class SelectiveBackpropper:

    def __init__(self,
                 model,
                 optimizer,
                 prob_pow,
                 batch_size,
                 lr_sched,
                 num_classes,
                 num_training_images,
                 forwardlr,
                 strategy,
                 kath_oversampling_rate,
                 calculator="relative",
                 fp_selector_type="alwayson",
                 staleness=2,
                 spline_y1=None,
                 spline_y2=None,
                 spline_y3=None):

        ## Hardcoded params
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert device == "cuda"
        self.num_training_images = num_training_images
        num_images_to_prime = self.num_training_images

        log_interval = 1
        bias_batch_log_interval = 1000
        sampling_min = 0
        sampling_max = 1
        max_history_len = 1024
        prob_loss_fn = nn.CrossEntropyLoss
        loss_fn = nn.CrossEntropyLoss
        sample_size = 0 # only needed for kath, topk, lowk

        # Params for resuming from checkpoint
        start_epoch = 0
        start_num_backpropped = 0
        start_num_skipped = 0
        kath_oversampling_rate = 4

        self.selector = None
        self.fp_selector = None
        self.bias_logger = None
        if strategy == "kath":
            self.selector = None
            self.backpropper = backproppers.SamplingBackpropper(device,
                                                                model,
                                                                optimizer,
                                                                loss_fn)
            self.trainer = trainer.KathTrainer(device,
                                               model,
                                               self.backpropper,
                                               batch_size,
                                               int(batch_size * kath_oversampling_rate),
                                               loss_fn,
                                               lr_schedule=lr_sched,
                                               forwardlr=forwardlr)
        elif strategy == "nofilter":
            self.backpropper = backproppers.SamplingBackpropper(device,
                                                                model,
                                                                optimizer,
                                                                loss_fn)
            self.trainer = trainer.NoFilterTrainer(device,
                                                   model,
                                                   self.backpropper,
                                                   batch_size,
                                                   loss_fn,
                                                   lr_schedule=lr_sched,
                                                   forwardlr=forwardlr)
        elif strategy == "logbias":
            probability_calculator = calculators.get_probability_calculator(calculator,
                                                                            device,
                                                                            prob_loss_fn,
                                                                            sampling_min,
                                                                            sampling_max,
                                                                            num_classes,
                                                                            max_history_len,
                                                                            prob_pow,
                                                                            spline_y1,
                                                                            spline_y2,
                                                                            spline_y3)
            self.selector = selectors.get_selector("sampling",
                                                   probability_calculator,
                                                   num_images_to_prime,
                                                   sample_size)
            self.fp_selector = fp_selectors.get_selector("alwayson",
                                                         num_images_to_prime,
                                                         staleness=staleness)

            self.backpropper = backproppers.GradientAndSelectivityLoggingBackpropper(device,
                                                                                     model,
                                                                                     optimizer,
                                                                                     loss_fn,
                                                                                     10,
                                                                                     bias_batch_log_interval)
            self.trainer = trainer.MemoizedTrainer(device,
                                                   model,
                                                   self.selector,
                                                   self.fp_selector,
                                                   self.backpropper,
                                                   batch_size,
                                                   loss_fn,
                                                   lr_schedule=lr_sched,
                                                   forwardlr=forwardlr)


            self.bias_logger = loggers.BiasByEpochLogger("/tmp",
                                                    "test",
                                                    bias_batch_log_interval)
            self.trainer.on_backward_pass(self.bias_logger.handle_backward_batch)

        else:
            probability_calculator = calculators.get_probability_calculator(calculator,
                                                                            device,
                                                                            prob_loss_fn,
                                                                            sampling_min,
                                                                            sampling_max,
                                                                            num_classes,
                                                                            max_history_len,
                                                                            prob_pow,
                                                                            spline_y1,
                                                                            spline_y2,
                                                                            spline_y3)
            self.selector = selectors.get_selector("sampling",
                                                   probability_calculator,
                                                   num_images_to_prime,
                                                   sample_size)

            self.fp_selector = fp_selectors.get_selector(fp_selector_type,
                                                         num_images_to_prime,
                                                         staleness=staleness)

            self.backpropper = backproppers.SamplingBackpropper(device,
                                                                model,
                                                                optimizer,
                                                                loss_fn)

            self.trainer = trainer.MemoizedTrainer(device,
                                                   model,
                                                   self.selector,
                                                   self.fp_selector,
                                                   self.backpropper,
                                                   batch_size,
                                                   loss_fn,
                                                   lr_schedule=lr_sched,
                                                   forwardlr=forwardlr)

        self.logger = loggers.Logger(log_interval = log_interval,
                                     epoch=start_epoch,
                                     num_backpropped=start_num_backpropped,
                                     num_skipped=start_num_skipped,
                                     start_time_seconds = start_time_seconds)

        self.trainer.on_backward_pass(self.logger.handle_backward_batch)
        self.trainer.on_forward_pass(self.logger.handle_forward_batch)

    def next_epoch(self):
        self.logger.next_epoch()
        if self.bias_logger:
            self.bias_logger.next_epoch()

    def next_partition(self):
        if self.selector is not None:
            self.selector.next_partition(self.num_training_images)
        if self.fp_selector is not None:
            self.fp_selector.next_partition(self.num_training_images)
        self.logger.next_partition()
