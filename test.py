import torch
import numpy as np
import argparse
from networks import unet
from utils import psnr
from data import datasets
import os
import open_clip

def get_args_parser():
    parser = argparse.ArgumentParser('I2I-Galip', add_help=False)
    parser.add_argument('--dataset', default="ixi", type=str)
    parser.add_argument('--domain_1', default="T1", type=str)
    parser.add_argument('--domain_2', default="T2", type=str)
    parser.add_argument('--gpu', default=0,type=int)
    parser.add_argument('--network_pkl', default="", type=str)
    parser.add_argument('--multi_domain', action='store_true', default=False)

    parser.add_argument('--network_type', default="unet", type=str)
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
    parser.add_argument('--context_dim', default=512, type=int)
    parser.add_argument('--num_classes', default=3, type=int)
    parser.add_argument('--use_scale_shift_norm', action='store_true', default=False)
    parser.add_argument('--resblock_updown', action='store_true', default=True)
    parser.add_argument('--num_heads', default=-1, type=int)
    parser.add_argument('--norm_type', default="groupnorm", type=str)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--mid_res_blocks', default=9, type=int)
    parser.add_argument('--mid_att_freq', default=2, type=int)
    return parser


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)

    args.channel_mult = [int(item) for item in args.channel_mult.split(',')]
    args.attention_resolutions = [int(item) for item in args.attention_resolutions.split(',')]
        
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
        "use_fp16": False,
        "norm_type": args.norm_type,
        "mid_res_blocks": args.mid_res_blocks,
        "mid_att_freq":args.mid_att_freq,
    }
    generator = unet.UNetModel(**network_params)
    generator.load_state_dict(torch.load(args.network_pkl))
    generator.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    generator.to(device)

    clip_model, _, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    clip_model.to(device)
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    print(test_loop(generator, clip_model, device, tokenizer, args))


@torch.no_grad()
def test_loop(ema_model, clip_model, device, tokenizer, args):
    if args.multi_domain:
        keyword = "multi"
    else:
        keyword = "single"
    result_dir_1 = "./images/" + keyword + "/" + args.dataset + "/" + args.domain_1 + "_to_" + args.domain_2 
    result_dir_2 = "./images/" + keyword + "/" + args.dataset + "/" + args.domain_2 + "_to_" + args.domain_1

    os.makedirs(result_dir_1, exist_ok=True)
    os.makedirs(result_dir_2, exist_ok=True)

    ema_model.eval()
    dataset_test = datasets.ImageDataset("test", args.dataset, args.domain_1, args.domain_2)

    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    test_batch_size = 1
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=test_batch_size,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    if args.dataset == "ct_to_t1" or args.dataset == "ct_to_t2":
        if args.domain_1 == "ct":
            y_domain_1 = torch.zeros([1]).to(device).long()
        elif args.domain_1 == "mri":
            y_domain_1 = torch.ones([1]).to(device).long()
        if args.domain_2 == "ct":
            y_domain_2 = torch.zeros([1]).to(device).long()
        elif args.domain_2 == "mri":
            y_domain_2 = torch.ones([1]).to(device).long()
        
        label_domain_2 =  "this is a pelvic " + args.domain_2 + " image"
        label_domain_1 =  "this is a pelvic " + args.domain_1 + " image"
    else:
        if args.domain_1 == "T1":
            y_domain_1 = torch.zeros([1]).to(device).long()
        elif args.domain_1 == "T2":
            y_domain_1 = torch.ones([1]).to(device).long()
        elif args.domain_1 == "PD" or args.domain_1 == "Flair":
            y_domain_1 = (torch.ones([1])*2).to(device).long()

        if args.domain_2 == "T1":
            y_domain_2 = torch.zeros([1]).to(device).long()
        elif args.domain_2 == "T2":
            y_domain_2 = torch.ones([1]).to(device).long()
        elif args.domain_2 == "PD" or args.domain_2 == "Flair":
            y_domain_2 = (torch.ones([1])*2).to(device).long()


        label_domain_2 =  "this MRI is " + args.domain_2 + "-weighted"
        label_domain_1 =  "this MRI is " + args.domain_1 + "-weighted"

    psnr_domain_1_ema = []
    psnr_domain_2_ema = []
    ssim_domain_1_ema = []
    ssim_domain_2_ema = []

    for i, data in enumerate(data_loader_test):
        input_image_domain_1, input_image_domain_2 = data
        input_image_domain_1 = input_image_domain_1.to(device)
        input_image_domain_2 = input_image_domain_2.to(device)

        _, clip_text_encoding_domain_2_tokens = clip_model.encode_text(tokenizer(label_domain_2, context_length=256).to(device), False)
        _, clip_text_encoding_domain_1_tokens = clip_model.encode_text(tokenizer(label_domain_1, context_length=256).to(device), False)

        synthesis_domain_2_ema,_ = ema_model(x=input_image_domain_1, context=clip_text_encoding_domain_2_tokens,  y=y_domain_2) 
        synthesis_domain_1_ema,_ = ema_model(x=input_image_domain_2, context=clip_text_encoding_domain_1_tokens,  y=y_domain_1) 

        input_domain_1_np = input_image_domain_1.detach().cpu().numpy() 
        input_domain_2_np = input_image_domain_2.detach().cpu().numpy() 

        output_domain_2_np = synthesis_domain_2_ema.detach().cpu().numpy() 
        output_domain_1_np = synthesis_domain_1_ema.detach().cpu().numpy() 


        psnr_domain_1_ema.append(psnr.compute_psnr(output_domain_1_np[0,0], input_domain_1_np[0,0]))
        psnr_domain_2_ema.append(psnr.compute_psnr(output_domain_2_np[0,0], input_domain_2_np[0,0]))
        ssim_domain_1_ema.append(psnr.compute_ssim(output_domain_1_np[0,0], input_domain_1_np[0,0]))
        ssim_domain_2_ema.append(psnr.compute_ssim(output_domain_2_np[0,0], input_domain_2_np[0,0]))

        np.save(result_dir_1 + "/" + str(i) + ".npy", output_domain_1_np[0,0])
        np.save(result_dir_2 + "/" + str(i) + ".npy", output_domain_2_np[0,0])


    return {'domain_1': args.domain_1, 'domain_2':args.domain_2, 'psnr_domain_1_ema_test': np.nanmean(psnr_domain_1_ema), 'psnr_domain_2_ema_test':np.nanmean(psnr_domain_2_ema), 'ssim_domain_1_ema_test':np.nanmean(ssim_domain_1_ema), 'ssim_domain_2_ema_test':np.nanmean(ssim_domain_2_ema)}


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)    


