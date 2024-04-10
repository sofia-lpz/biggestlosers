import argparse
import json
import os
import subprocess


def set_experiment_default_args(parser):
    parser.add_argument('--expname', '-e', default="tmp", type=str, help='experiment name')
    parser.add_argument('--strategy', '-s', default="nofilter", type=str, help='nofilter, sb, kath')
    parser.add_argument('--prob-strategy', '-p', default="relative", type=str, help='relative')
    parser.add_argument('--fp-prob-strategy', '-fp', default="alwayson", type=str, help='alwayson')
    parser.add_argument('--std-mult', default=1, type=float, help='std mult for fp hist calculator')
    parser.add_argument('--beta', default=3, type=float, help='beta on relative')
    parser.add_argument('--dataset', '-d', default="cifar10", type=str, help='mnist, cifar10, svhn, imagenet')
    parser.add_argument('--network', '-n', default="mobilenetv2", type=str, help='network architecture')
    parser.add_argument('--batch-size', '-b', default=128, type=int, help='batch size')
    parser.add_argument('--forward-batch-size', '-fb', default=128, type=int, help='batch size')
    parser.add_argument('--nolog', '-nl', dest='nolog', action='store_true',
                        help='turn off extra logging')
    parser.add_argument('--selector', dest='selector', default="sampling",
                        help='Select strategyy from {sampling, deterministic, topk}')
    parser.add_argument('--profile', dest='profile', action='store_true',
                        help='turn profiling on')
    parser.add_argument('--noaugment', '-na', dest='noaugment', action='store_true',
                        help='Turn augmentation off')

    # Mutually exclusive with LR sched opts
    parser.add_argument('--static-lr', '-slr', default=None, type=float, help='Overrides LR sched opts')

    # LR sched opts
    parser.add_argument('--gradual', '-g', dest='gradual', action='store_true',
                        help='is learning rate gradual')
    parser.add_argument('--fast', '-f', dest='fast', action='store_true',
                        help='is learning rate accelerated')
    parser.add_argument('--forwardlr', dest='forwardlr', action='store_true',
                        help='learning rate schedule is based on forward props')

    parser.add_argument('--num-trials', default=1, type=int, help='number of trials')
    parser.add_argument('--start-epoch', "-se", default=1, type=float, help='SB start epoch')
    parser.add_argument('--src-dir', default="./", type=str, help='/path/to/pytorch-cifar')
    parser.add_argument('--dst-dir', default="/proj/BigLearning/ahjiang/output/", type=str, help='/path/to/dst/dir')

    parser.add_argument('--kath-strategy', default="biased", type=str, help='Katharopoulos18')
    parser.add_argument('--static-selectivity', default=4, type=int, help='Scale of superset')

    return parser

def get_lr_sched_path(src_dir, dataset, gradual, fast):
    filename = "lrsched_{}".format(dataset)
    if gradual:
        filename += "_{}".format("gradual")
    else:
        filename += "_{}".format("step")
    if fast:
        filename += "_{}".format("fast")
    path = os.path.join(src_dir, "data/config/neurips", filename)
    return path

def get_max_num_backprops(lr_filename, profile):
    #if profile:
    #    num_profile_backprops = 2000000
    #    print("[WARNING] Profiling turned on. Overriding max_num_backprops to {}".format(num_profile_backprops))
    #    return num_profile_backprops
    with open(lr_filename) as f:
        data = json.load(f)
    last_lr_jump = max([int(k) for k in data.keys()])
    return int(last_lr_jump * 1.4)

def get_sampling_min():
    return 0

def get_sample_size(batch_size, static_selectivity, selector, strategy):
    if strategy == "kath":
        return batch_size * static_selectivity
    elif selector == "topk":
        return batch_size / static_selectivity
    else:
        return -1

def get_decay():
    return 0.0005

class Seeder():
    def __init__(self):
        self.seed = 1336

    def get_seed(self):
        self.seed += 1
        return self.seed

def get_max_history_length():
    return 1024

def get_output_dirs(dst_dir):
    pickles_dir = os.path.join(dst_dir, "pickles")
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    if not os.path.exists(pickles_dir):
        os.mkdir(pickles_dir)
    return dst_dir, pickles_dir

def get_output_files(sb_selector,
                     dataset,
                     net,
                     sampling_min,
                     batch_size,
                     max_history_length,
                     decay,
                     trial,
                     seed,
                     strategy,
                     kath_strategy,
                     static_sample_size):

    if strategy == "kath":
        sb_selector = "kath-{}".format(kath_strategy)
        max_history_length = static_sample_size
    if strategy == "nofilter":
        sb_selector = "nofilter"
    if sb_selector == "topk":
        max_history_length = static_sample_size

    output_file = "{}_{}_{}_{}_{}_{}_{}_trial{}_seed{}_v4".format(sb_selector,
                                                                  dataset,
                                                                  net,
                                                                  sampling_min,
                                                                  batch_size,
                                                                  max_history_length,
                                                                  decay,
                                                                  trial,
                                                                  seed)

    pickle_file = "{}_{}_{}_{}_{}_{}_{}_trial{}_seed{}".format(sb_selector,
                                                               dataset,
                                                               net,
                                                               sampling_min,
                                                               batch_size,
                                                               max_history_length,
                                                               decay,
                                                               trial,
                                                               seed)
    return output_file, pickle_file

def get_experiment_dirs(dst_dir, dataset, expname):
    output_dir = os.path.join(dst_dir, dataset, expname)
    pickles_dir = os.path.join(output_dir, "pickles")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(pickles_dir):
        os.mkdir(pickles_dir)
    return output_dir, pickles_dir

def get_imagenet_datadir():
    return "/proj/BigLearning/ahjiang/datasets/imagenet-data"

def get_nolog(nolog, profile):
    return nolog or profile

def get_start_epoch(start_epoch, profile):
    #if profile:
    #    print("[WARNING] Profiling turned on. Overriding start_epoch to 0")
    #    return 0
    #else:
    return start_epoch


def main(args):
    seeder = Seeder()
    src_dir = os.path.abspath(args.src_dir)
    lr_sched_path = get_lr_sched_path(src_dir, args.dataset, args.gradual, args.fast)
    if not os.path.isfile(lr_sched_path):
        print("{} is not a file").format(lr_sched_path)
        exit()
    max_num_backprops = get_max_num_backprops(lr_sched_path, args.profile)
    sampling_min = get_sampling_min()
    decay = get_decay()
    output_dir, pickles_dir = get_experiment_dirs(args.dst_dir, args.dataset, args.expname)
    max_history_length = get_max_history_length()
    static_sample_size = get_sample_size(args.batch_size, args.static_selectivity, args.selector, args.strategy)
    start_epoch = get_start_epoch(args.start_epoch, args.profile)
    assert(args.strategy in ["nofilter", "sb", "kath"])

    for trial in range(1, args.num_trials+1):
        seed = seeder.get_seed()
        output_file, pickle_file = get_output_files(args.selector,
                                                    args.dataset,
                                                    args.network,
                                                    sampling_min,
                                                    args.batch_size,
                                                    max_history_length,
                                                    decay,
                                                    trial,
                                                    seed,
                                                    args.strategy,
                                                    args.kath_strategy,
                                                    static_sample_size)
        if args.profile:
            cmd = "python -m cProfile -o profs/{}.prof main.py ".format(args.expname)
        else:
            cmd = "python main.py "
        cmd += "--prob-strategy={} ".format(args.prob_strategy)
        cmd += "--fp-prob-strategy={} ".format(args.fp_prob_strategy)
        cmd += "--prob-pow={} ".format(args.beta)
        cmd += "--max-history-len={} ".format(max_history_length)
        cmd += "--dataset={} ".format(args.dataset)
        cmd += "--prob-loss-fn={} ".format("cross")
        cmd += "--sb-start-epoch={} ".format(start_epoch)
        cmd += "--sb-strategy={} ".format(args.selector)
        cmd += "--net={} ".format(args.network)
        cmd += "--batch-size={} ".format(args.batch_size)
        cmd += "--forward-batch-size={} ".format(args.forward_batch_size)
        cmd += "--decay={} ".format(decay)
        cmd += "--max-num-backprops={} ".format(max_num_backprops)
        cmd += "--pickle-dir={} ".format(pickles_dir)
        cmd += "--pickle-prefix={} ".format(pickle_file)
        cmd += "--sampling-min={} ".format(sampling_min)
        cmd += "--seed={} ".format(seed)
        cmd += "--std-multiplier={} ".format(args.std_mult)
        if args.static_lr is None:
            cmd += "--lr-sched={} ".format(lr_sched_path)
        else:
            if args.gradual or args.fast:
                print("[Warning] Using StaticLR, overridding -g, -f")
            cmd += "--lr={} ".format(args.static_lr)

        if get_nolog(args.nolog, args.profile):
            cmd += "--no-logging "

        if args.forwardlr:
            cmd += "--forwardlr "

        if args.dataset == "imagenet":
            cmd += "--datadir={} ".format(get_imagenet_datadir())

        if args.selector == "topk":
            cmd += "--sample-size={} ".format(static_sample_size)

        if args.strategy == "nofilter":
            print("[Warning] Using NoFilter, overridding prob-strategy, beta, selector")
            cmd += "--nofilter "

        if args.strategy == "kath":
            if args.selector == "topk":
                print("[Warning] Running Kath, overridding --selector")
            assert(args.kath_strategy in ["reweighted", "biased", "baseline"])
            cmd += "--kath "
            cmd += "--kath-strategy={} ".format(args.kath_strategy)
            cmd += "--sample-size={} ".format(static_sample_size)

        if not args.noaugment:
            cmd += "--augment "

        cmd = cmd.strip()

        output_path = os.path.join(output_dir, output_file)
        print("========================================================================")
        print(cmd)
        print("------------------------------------------------------------------------")
        print(output_path)

        with open(os.path.join(pickles_dir, output_file) + "_cmd", "w+") as f:
            f.write(cmd)

        cmd_list = cmd.split(" ")
        with open(output_path, "w+") as f:
            subprocess.call(cmd_list, stdout=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser = set_experiment_default_args(parser)
    args = parser.parse_args()
    main(args)
