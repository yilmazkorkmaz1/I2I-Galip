import torch
import numpy as np
import os
import sys
import json
import open_clip
import torchvision.transforms as transforms
import copy
import argparse
import random
from utils import util
from utils import misc
from networks import unet as unet_decoder
from networks import stylegan_disc
from data import datasets
from training_engine import training_engine


def get_args_parser():

    parser = argparse.ArgumentParser('I2I-Galip', add_help=False)

    parser.add_argument('--gpu', default=0,type=int)
    parser.add_argument('--seed', default=40,type=int)
    parser.add_argument('--exp_name', default="trial", type=str)   
    parser.add_argument('--mixed_precision', action='store_true', default=False)
    parser.add_argument('--result_dir', default="./results/", type=str)
    parser.add_argument('--dataset', default="ixi", type=str)
    parser.add_argument('--domain_1', default="T1", type=str)
    parser.add_argument('--domain_2', default="T2", type=str)
    parser.add_argument('--multi_domain', action='store_true', default=False)
    parser.add_argument('--batch_size', default=4, type=int)    
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--augmentation', default="none", type=str, choices=["flip", "flip_and_norm", "none"])


    ## Optimizer hyperparameters
    parser.add_argument('--generator_lr', type=float, default=2e-4)
    parser.add_argument('--discriminator_lr', type=float, default=2e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--lr_decay_linear', action='store_true', default=True)
    parser.add_argument('--lr_decay_spike', action='store_true', default=False)
    parser.add_argument('--zero_init', action='store_true', default=False)
    parser.add_argument('--ema_decay', default=0.99, type=float)
    parser.add_argument('--optimizer', default="adam", type=str)
    parser.add_argument('--beta1', default=0.5, type=float)
    parser.add_argument('--beta2', default=0.99, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)
    parser.add_argument('--accum_iter_generator', default=1, type=int)
    parser.add_argument('--accum_iter_discriminator', default=1, type=int)
    parser.add_argument('--grad_clip_generator', default=None, type=float)
    parser.add_argument('--grad_clip_discriminator', default=None, type=float)

    ## Loss hyperparameters
    parser.add_argument('--lamda_cycle', default=10.0,type=float)
    parser.add_argument('--lamda_clip', default=1.0,type=float)
    parser.add_argument('--lamda_encode', default=1.0,type=float)
    parser.add_argument('--lamda_identity', default=1.0,type=float)
    parser.add_argument('--lamda_adv', default=1.0,type=float)
    parser.add_argument('--lamda_patch', default=1.0,type=float)
    parser.add_argument('--patch_loss', action='store_true', default=False)
    parser.add_argument('--new_clip_loss', action='store_true', default=False)
    parser.add_argument('--encoding_loss_type', default="COS",type=str)
    parser.add_argument('--cls_token_nums', type=str, default="-1,-2,-3,-4")



    ## Generator hyperparameters
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--out_channels', default=1, type=int)
    parser.add_argument('--model_channels', default=32, type=int)
    parser.add_argument('--attention_resolutions', default="8,16,32", type=str)
    parser.add_argument('--num_res_blocks', default=2, type=int)
    parser.add_argument('--channel_mult', default="1,1,2,2,4,4", type=str)
    parser.add_argument('--num_head_channels', default=32, type=int)
    parser.add_argument('--use_spatial_transformer', action='store_true', default=True)
    parser.add_argument('--transformer_depth', default=1, type=int)
    parser.add_argument('--context_dim', default=768, type=int)
    parser.add_argument('--num_classes', default=3, type=int)
    parser.add_argument('--use_scale_shift_norm', action='store_true', default=False)
    parser.add_argument('--resblock_updown', action='store_true', default=True)
    parser.add_argument('--num_heads', default=-1, type=int)
    parser.add_argument('--norm_type', default="groupnorm", type=str)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--mid_res_blocks', default=9, type=int)
    parser.add_argument('--mid_att_freq', default=2, type=int)



    ## Discriminator parameters
    parser.add_argument('--disc_loss_dim', default="2,5,8,11", type=str)
    parser.add_argument('--disc_class', default=3, type=int)


    return parser




def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)
    args.result_dir += args.exp_name
    print("result_dir: ", args.result_dir)
    if args.multi_domain:
        args.num_classes = 3
        args.disc_class = 3
    args.cls_token_nums = [int(item) for item in args.cls_token_nums.split(',')]
    args.disc_loss_dim = [int(item) for item in args.disc_loss_dim.split(',')]
    args.channel_mult = [int(item) for item in args.channel_mult.split(',')]
    args.attention_resolutions = [int(item) for item in args.attention_resolutions.split(',')]


    
    if not os.path.isdir(args.result_dir):
        os.makedirs(args.result_dir)

    with open(os.path.join(args.result_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        for key, value in vars(args).items():
            f.write(json.dumps({key:value}) + "\n")

    clip_model, _, domain_1_preprocess = open_clip.create_model_and_transforms("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    clip_model.text.output_tokens = True 

    domain_2_preprocess = domain_1_preprocess

    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    
    domain_1 = args.domain_1 #"T1"
    domain_2 = args.domain_2 #"T2"
    dataset = args.dataset #"ixi"
    batch_size = args.batch_size #1

    print("Augmentation types: " + args.augmentation)
    if args.augmentation == "flip":
        augmentations = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)])
    elif args.augmentation == "flip_and_norm" and args.dataset == "ixi":
        augmentations = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.Normalize(mean=0.102588885, std=0.19143428)])
    else:
        augmentations = None

    if args.multi_domain:
        data_loader_train_1, data_loader_train_2 = datasets.prepare_datasets_cyclegan_multi(dataset, batch_size, augmentations)
    else:
        data_loader_train_1, data_loader_train_2 = datasets.prepare_datasets_cyclegan_single(domain_1, domain_2, dataset, batch_size, augmentations)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    clip_model.to(device).requires_grad_(False)
    clip_model.eval()

    network_params = {
        "image_size": args.image_size,
        "in_channels": args.in_channels,
        "out_channels": args.out_channels,
        "model_channels": args.model_channels,
        "attention_resolutions": args.attention_resolutions,
        "num_res_blocks": args.num_res_blocks,
        "channel_mult":args.channel_mult,
        "num_head_channels": args.num_head_channels,
        "use_spatial_transformer": args.use_spatial_transformer,
        "transformer_depth": args.transformer_depth,
        "context_dim": args.context_dim,
        "num_classes": args.num_classes,
        "use_scale_shift_norm":args.use_scale_shift_norm,
        "resblock_updown": args.resblock_updown,
        "num_heads": args.num_heads,
        "use_fp16": args.mixed_precision,
        "norm_type": args.norm_type,
        "mid_res_blocks": args.mid_res_blocks,
        "mid_att_freq":args.mid_att_freq,
    }
    generator = unet_decoder.UNetModel(**network_params)


    with open(os.path.join(args.result_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        sys.stdout = f
        print(generator)
        sys.stdout = sys.__stdout__

    f.close()
    generator.to(device)
    generator.train()
 
    if not args.zero_init:
        generator.init_weights()

    loss_fn = torch.nn.L1Loss()
    mse = torch.nn.MSELoss()

    ema_model = util.ExponentialMovingAverage(copy.deepcopy(generator), args.ema_decay)
    discriminator = stylegan_disc.ProjectedDiscriminator(num_of_disc_feats=len(args.disc_loss_dim), num_of_classes=args.disc_class)
    discriminator = discriminator.to(device)

    log_writer = None
    total_params = sum(p.numel() for p in generator.parameters())
    print("total number of parameters in generator: ", total_params)
    total_params = sum(p.numel() for p in discriminator.parameters())
    print("total number of parameters in discriminator: ", total_params)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(generator.parameters(), lr=args.generator_lr, betas=(args.beta1,args.beta2))
        optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=args.discriminator_lr, betas=(args.beta1,args.beta2))

    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(generator.parameters(), lr=args.generator_lr, betas=(args.beta1,args.beta2), weight_decay=args.weight_decay)
        optimizer_disc = torch.optim.AdamW(discriminator.parameters(), lr=args.discriminator_lr, betas=(args.beta1,args.beta2), weight_decay=args.weight_decay)

    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(generator.parameters(), lr=args.generator_lr)
        optimizer_disc = torch.optim.SGD(discriminator.parameters(), lr=args.discriminator_lr)
        
    for epoch in range(1, args.epochs):  
        if args.lr_decay_linear:
            util.adjust_learning_rate_linear(optimizer, epoch, args.epochs, initial_lr=args.generator_lr)
            util.adjust_learning_rate_linear(optimizer_disc, epoch, args.epochs, initial_lr=args.discriminator_lr)
        elif args.lr_decay_spike:
            util.adjust_learning_rate_spike(optimizer, epoch, args.warmup_epochs, args.generator_lr, args.min_lr, args.epochs)
            util.adjust_learning_rate_spike(optimizer_disc, epoch, args.warmup_epochs, args.discriminator_lr, args.min_lr, args.epochs)
        train_stats = training_engine.train_one_epoch(generator=generator,
                        clip_model= clip_model,
                        discriminator= discriminator,
                        ema_model= ema_model,
                        optimizer= optimizer,
                        optimizer_disc= optimizer_disc,  
                        domain_2_preprocess=domain_2_preprocess,
                        domain_1_preprocess=domain_1_preprocess,
                        data_loader_train_1=data_loader_train_1,
                        data_loader_train_2=data_loader_train_2,
                        tokenizer=tokenizer,
                        device= device,
                        epoch= epoch,
                        loss_fn=loss_fn,
                        mse=mse,
                        log_writer=log_writer,
                        args=args)
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}        
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.result_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        generator.eval()
        os.makedirs(args.result_dir + "/val/" + str(epoch), exist_ok=True)
        if args.multi_domain:
            if args.dataset == "ixi":
                for (domain_1, domain_2) in [["T1","T2"], ["T1","PD"], ["T2", "PD"]]:
                    args.domain_1=domain_1; args.domain_2=domain_2
                    val_stats = training_engine.validation_loop(generator, ema_model.model, clip_model, epoch, device, tokenizer, args=args)
                    log_stats  = {**{k: v for k, v in val_stats.items()},
                                    'epoch': epoch,}
                    print(log_stats)
                    if misc.is_main_process():
                        if log_writer is not None:
                            log_writer.flush()
                        with open(os.path.join(args.result_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                            f.write(json.dumps(log_stats) + "\n")
            elif args.dataset == "brats":
                for (domain_1, domain_2) in [["T1","T2"], ["T1","Flair"], ["T2", "Flair"]]:
                    args.domain_1=domain_1; args.domain_2=domain_2
                    val_stats = training_engine.validation_loop(generator, ema_model.model, clip_model, epoch, device, tokenizer, args=args)
                    log_stats  = {**{k: v for k, v in val_stats.items()},
                                    'epoch': epoch,}
                    print(log_stats)
                    if misc.is_main_process():
                        if log_writer is not None:
                            log_writer.flush()
                        with open(os.path.join(args.result_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                            f.write(json.dumps(log_stats) + "\n") 
        else:
                val_stats = training_engine.validation_loop(generator, ema_model.model, clip_model, epoch, device, tokenizer, args=args)
                log_stats  = {**{k: v for k, v in val_stats.items()},
                                'epoch': epoch,}
                print(log_stats)
                if misc.is_main_process():
                    if log_writer is not None:
                        log_writer.flush()
                    with open(os.path.join(args.result_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n") 

        generator.train()

        if epoch % 5 == 0 or epoch + 1 == args.epochs:
            torch.save(generator.state_dict(), args.result_dir + "/epoch_" + str(epoch))
            torch.save(ema_model.model.state_dict(), args.result_dir + "/epoch_ema" + str(epoch))



if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    print("SEED IS: ", args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    main(args)