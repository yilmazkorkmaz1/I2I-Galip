import torch
import torch.nn as nn
from utils import misc
import numpy as np
import matplotlib.pyplot as plt
from utils import psnr
from data.datasets import ImageDataset
from torchvision import transforms
import torch.nn.functional as F


def train_one_epoch(
    generator: nn.Module,
    clip_model: nn.Module,
    discriminator: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim,
    optimizer_disc: torch.optim,  
    device: torch.device,
    epoch: int,
    domain_2_preprocess,
    domain_1_preprocess,
    data_loader_train_1,
    data_loader_train_2,
    tokenizer,
    loss_fn,
    mse,
    log_writer,
    args,
):

    batch_size = args.batch_size
    accum_iter_generator = args.accum_iter_generator
    disc_loss_dim = args.disc_loss_dim
    lamda_cycle = args.lamda_cycle
    lamda_clip = args.lamda_clip
    lamda_encode = args.lamda_encode
    lamda_adv = args.lamda_adv
    lamda_identity = args.lamda_identity
    print_freq = args.print_freq
    encoding_loss_type = args.encoding_loss_type
    cls_token_nums = args.cls_token_nums
    patch_loss = args.patch_loss

    if encoding_loss_type == "COS":
        cosine_loss = torch.nn.CosineEmbeddingLoss()
        target_cos_label=torch.ones(batch_size).to(device)

    metric_logger = misc.MetricLogger(delimiter="  ")

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    print("tokenizer: ", tokenizer)
    clip_model.requires_grad_(False)

    header = 'Epoch: [{}]'.format(epoch)

    for i, ((input_image_domain_1, label_domain_1, y_domain_1), (input_image_domain_2, label_domain_2, y_domain_2)) in enumerate(metric_logger.log_every(list(zip(data_loader_train_1, data_loader_train_2)), print_freq, header)):
        
        input_image_domain_1 = input_image_domain_1.to(device)
        input_image_domain_2 = input_image_domain_2.to(device)
   
        y_domain_1 = y_domain_1.to(device)
        y_domain_2 = y_domain_2.to(device)

        # CLIP TEXT ENCODINGS
        clip_text_encoding_domain_2, clip_text_encoding_domain_2_tokens = clip_model.encode_text(tokenizer(label_domain_2, context_length=256).to(device), False)
        clip_text_encoding_domain_1, clip_text_encoding_domain_1_tokens = clip_model.encode_text(tokenizer(label_domain_1, context_length=256).to(device), False)
        clip_text_encoding_domain_2 = F.normalize(clip_text_encoding_domain_2, dim=-1).unsqueeze(1)
        clip_text_encoding_domain_1 = F.normalize(clip_text_encoding_domain_1, dim=-1).unsqueeze(1)


        # GENERATE IMAGES
        synthesis_domain_2,_ = generator(x=input_image_domain_1, context=clip_text_encoding_domain_2_tokens, y=y_domain_2) 
        recon_domain_1,_ = generator(x=synthesis_domain_2, context=clip_text_encoding_domain_1_tokens, y=y_domain_1) 
        synthesis_domain_1,_ = generator(x=input_image_domain_2, context=clip_text_encoding_domain_1_tokens, y=y_domain_1) 
        recon_domain_2,_ = generator(x=synthesis_domain_1, context=clip_text_encoding_domain_2_tokens, y=y_domain_2) 

        # CLIP LOSS
        clip_output_image_encoding_synthesis_domain_2 = clip_model.encode_image(domain_2_preprocess(torch.tile(synthesis_domain_2, (1,3,1,1))), True)
        clip_output_image_encoding_synthesis_domain_1 = clip_model.encode_image(domain_1_preprocess(torch.tile(synthesis_domain_1, (1,3,1,1))), True)

        clip_loss_domain_2  = -1.0 * torch.diagonal(clip_output_image_encoding_synthesis_domain_2 @ clip_text_encoding_domain_2[:,0].T)
        clip_loss_domain_1  = -1.0 * torch.diagonal(clip_output_image_encoding_synthesis_domain_1 @ clip_text_encoding_domain_1[:,0].T)

        # CLIP ENCODING LOSS
        a = list(clip_model.visual.get_intermediate_features_clip(domain_1_preprocess(torch.tile(input_image_domain_1, (1,3,1,1))),list(np.arange(0,12)),return_class_token=True,norm=True))
        cls_domain_1_input = torch.stack([a[cls_token_nums[0]][1],a[cls_token_nums[1]][1],a[cls_token_nums[2]][1],a[cls_token_nums[3]][1]], dim=1)
        clip_intermediate_original_domain_1_ = torch.stack([a[dim][0] for dim in list(np.arange(0,12))], dim=1)

        a = list(clip_model.visual.get_intermediate_features_clip(domain_2_preprocess(torch.tile(input_image_domain_2, (1,3,1,1))),list(np.arange(0,12)),return_class_token=True,norm=True))
        cls_domain_2_input = torch.stack([a[cls_token_nums[0]][1],a[cls_token_nums[1]][1],a[cls_token_nums[2]][1],a[cls_token_nums[3]][1]], dim=1)
        clip_intermediate_original_domain_2_ = torch.stack([a[dim][0] for dim in list(np.arange(0,12))], dim=1)

        a = list(clip_model.visual.get_intermediate_features_clip(domain_1_preprocess(torch.tile(synthesis_domain_1, (1,3,1,1))),list(np.arange(0,12)),return_class_token=True,norm=True))
        cls_domain_1_syn = torch.stack([a[cls_token_nums[0]][1],a[cls_token_nums[1]][1],a[cls_token_nums[2]][1],a[cls_token_nums[3]][1]], dim=1)
        clip_output_image_intermediate_synthesis_domain_1_ = torch.stack([a[dim][0] for dim in list(np.arange(0,12))], dim=1)

        a = list(clip_model.visual.get_intermediate_features_clip(domain_2_preprocess(torch.tile(synthesis_domain_2, (1,3,1,1))),list(np.arange(0,12)),return_class_token=True,norm=True))
        cls_domain_2_syn = torch.stack([a[cls_token_nums[0]][1],a[cls_token_nums[1]][1],a[cls_token_nums[2]][1],a[cls_token_nums[3]][1]], dim=1)
        clip_output_image_intermediate_synthesis_domain_2_ = torch.stack([a[dim][0] for dim in list(np.arange(0,12))], dim=1)

        if encoding_loss_type == "L1":
            clip_encoding_loss_domain_1 = loss_fn(cls_domain_1_syn,cls_domain_1_input)
            clip_encoding_loss_domain_2 = loss_fn(cls_domain_2_syn,cls_domain_2_input) 
        elif encoding_loss_type == "MSE":
            clip_encoding_loss_domain_1 = mse(cls_domain_1_syn,cls_domain_1_input)
            clip_encoding_loss_domain_2 = mse(cls_domain_2_syn,cls_domain_2_input) 
        elif encoding_loss_type == "COS":
            clip_encoding_loss_domain_1 = cosine_loss(cls_domain_1_syn.view(cls_domain_1_syn.shape[0],-1).contiguous(),cls_domain_1_input.view(cls_domain_1_syn.shape[0],-1).contiguous(),target_cos_label)
            clip_encoding_loss_domain_2 = cosine_loss(cls_domain_2_syn.view(cls_domain_1_syn.shape[0],-1).contiguous(),cls_domain_2_input.view(cls_domain_1_syn.shape[0],-1).contiguous(),target_cos_label) 
    
        # CYCLE LOSS
        cycle_loss_domain_1 = loss_fn(recon_domain_1, input_image_domain_1)
        cycle_loss_domain_2 = loss_fn(recon_domain_2, input_image_domain_2)

        # IDENTITY LOSS
        gen_identity_loss_domain_1, _ = generator(x=input_image_domain_1, context=clip_text_encoding_domain_1_tokens, y=y_domain_1) 
        gen_identity_loss_domain_2, _ = generator(x=input_image_domain_2, context=clip_text_encoding_domain_2_tokens, y=y_domain_2) 
        identity_loss_domain_1 = loss_fn(input_image_domain_1, gen_identity_loss_domain_1)
        identity_loss_domain_2 = loss_fn(input_image_domain_2, gen_identity_loss_domain_2)

        ## DISCRIMINATOR ##
        clip_intermediate_original_domain_1 = clip_intermediate_original_domain_1_[:,disc_loss_dim].transpose(2,3).contiguous()
        clip_intermediate_original_domain_2 = clip_intermediate_original_domain_2_[:,disc_loss_dim].transpose(2,3).contiguous()
        clip_output_image_intermediate_synthesis_domain_1 = clip_output_image_intermediate_synthesis_domain_1_[:,disc_loss_dim].transpose(2,3).contiguous()
        clip_output_image_intermediate_synthesis_domain_2 = clip_output_image_intermediate_synthesis_domain_2_[:,disc_loss_dim].transpose(2,3).contiguous()

        gen_logits_domain_1 = discriminator(clip_output_image_intermediate_synthesis_domain_1.detach(), clip_text_encoding_domain_1.detach(), class_label=y_domain_1)
        gen_logits_domain_2 = discriminator(clip_output_image_intermediate_synthesis_domain_2.detach(), clip_text_encoding_domain_2.detach(), class_label=y_domain_2)
        loss_gen_domain_1 = mse(torch.zeros_like(gen_logits_domain_1), gen_logits_domain_1) 
        loss_gen_domain_2 = mse(torch.zeros_like(gen_logits_domain_2), gen_logits_domain_2) 
        loss_gen = loss_gen_domain_1 + loss_gen_domain_2
        real_logits_domain_1 = discriminator(clip_intermediate_original_domain_1.detach(), clip_text_encoding_domain_1.detach(), class_label=y_domain_1)
        real_logits_domain_2 = discriminator(clip_intermediate_original_domain_2.detach(), clip_text_encoding_domain_2.detach(), class_label=y_domain_2)
        loss_real_domain_1 = mse(torch.ones_like(real_logits_domain_1), real_logits_domain_1)
        loss_real_domain_2 = mse(torch.ones_like(real_logits_domain_2), real_logits_domain_2)
        loss_real = loss_real_domain_1 + loss_real_domain_2
        disc_loss = loss_gen + loss_real

        # ADVERSARIAL LOSS
        gen_logits_domain_1_generator = discriminator(clip_output_image_intermediate_synthesis_domain_1, clip_text_encoding_domain_1, class_label=y_domain_1)
        gen_logits_domain_2_generator = discriminator(clip_output_image_intermediate_synthesis_domain_2, clip_text_encoding_domain_2, class_label=y_domain_2)        
        domain_1_loss_adv = mse(torch.ones_like(gen_logits_domain_1_generator), gen_logits_domain_1_generator) 
        domain_2_loss_adv = mse(torch.ones_like(gen_logits_domain_2_generator), gen_logits_domain_2_generator) 

        # LOSS CALCULATION
        loss = (lamda_cycle * (cycle_loss_domain_1.mean()  + cycle_loss_domain_2.mean()) 
        + lamda_clip * (clip_loss_domain_1.mean() + clip_loss_domain_2.mean()) 
        + lamda_encode * (clip_encoding_loss_domain_1.mean() + clip_encoding_loss_domain_2.mean())
        + lamda_adv * (domain_1_loss_adv.mean() + domain_2_loss_adv.mean())
        + lamda_identity * (identity_loss_domain_1.mean() + identity_loss_domain_2.mean()))


        discriminator.requires_grad_(False)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ema_model.update(generator)

        discriminator.requires_grad_(True)
        optimizer_disc.zero_grad(set_to_none=True)
        disc_loss.backward()
        optimizer_disc.step()

       # torch.cuda.synchronize()
        metric_logger.update(generator_loss=loss.item())
        metric_logger.update(disc_loss=disc_loss.item())
        metric_logger.update(cycle_loss_domain_1=cycle_loss_domain_1.item())
        metric_logger.update(cycle_loss_domain_2=cycle_loss_domain_2.item())
        metric_logger.update(clip_loss_domain_1=clip_loss_domain_1.mean().item())
        metric_logger.update(clip_loss_domain_2=clip_loss_domain_2.mean().item())
        metric_logger.update(clip_encoding_loss_domain_1=clip_encoding_loss_domain_1.mean().item())
        metric_logger.update(clip_encoding_loss_domain_2=clip_encoding_loss_domain_2.mean().item())
        metric_logger.update(domain_1_loss_adv=domain_1_loss_adv.mean().item())
        metric_logger.update(domain_2_loss_adv=domain_2_loss_adv.mean().item())
        metric_logger.update(identity_loss_domain_1=identity_loss_domain_1.mean().item())
        metric_logger.update(identity_loss_domain_2=identity_loss_domain_2.mean().item())
      

        loss_value_reduce_generator = misc.all_reduce_mean(loss.item())
        loss_value_reduce_discriminator = misc.all_reduce_mean(disc_loss.item())

        if log_writer is not None and (i + 1) % accum_iter_generator == 0:
            log_writer.add_scalar('generator_loss', loss_value_reduce_generator, epoch)
            log_writer.add_scalar('discriminator_loss', loss_value_reduce_discriminator, epoch)

    # gather the stats from all processes
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def validation_loop(generator, ema_model, clip_model, epoch, device, tokenizer, args):
    
    if args.augmentation == "flip_and_norm":
        transform = transforms.Normalize(mean=0.102588885, std=0.19143428)
    else:
        transform = None

    dataset_val = ImageDataset("val",args.dataset, args.domain_1, args.domain_2, transform=transform)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    val_batch_size = 1
    data_loader_validation = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=val_batch_size,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    result_dir = args.result_dir

    if args.dataset == "ct_to_t1" or args.dataset == "ct_to_t2":
        if args.domain_1 == "ct":
            y_domain_1 = torch.zeros([1]).to(device).long()
        elif args.domain_1 == "mri":
            y_domain_1 = torch.ones([1]).to(device).long()
        if args.domain_2 == "ct":
            y_domain_2 = torch.zeros([1]).to(device).long()
        elif args.domain_2 == "mri":
            y_domain_2 = torch.ones([1]).to(device).long()
        
        label_domain_2 =  "this is pelvic " + args.domain_2 
        label_domain_1 =  "this is pelvic " + args.domain_1 
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
    ssim_domain_1_ema = []
    psnr_domain_2_ema = []
    ssim_domain_2_ema = []

    psnr_domain_1 = []
    ssim_domain_1 = []
    psnr_domain_2 = []
    ssim_domain_2 = []

    for i, data in enumerate(data_loader_validation):
        input_image_domain_1, input_image_domain_2 = data
        input_image_domain_1 = input_image_domain_1.to(device)
        input_image_domain_2 = input_image_domain_2.to(device)

        clip_text_encoding_domain_2, clip_text_encoding_domain_2_tokens = clip_model.encode_text(tokenizer(label_domain_2, context_length=256).to(device), False)
        clip_text_encoding_domain_1, clip_text_encoding_domain_1_tokens = clip_model.encode_text(tokenizer(label_domain_1, context_length=256).to(device), False)
        clip_text_encoding_domain_2 = F.normalize(clip_text_encoding_domain_2, dim=-1).unsqueeze(1)
        clip_text_encoding_domain_1 = F.normalize(clip_text_encoding_domain_1, dim=-1).unsqueeze(1)

        synthesis_domain_2,_ = generator(x=input_image_domain_1, context=clip_text_encoding_domain_2_tokens, y=y_domain_2) 
        synthesis_domain_1,_ = generator(x=input_image_domain_2, context=clip_text_encoding_domain_1_tokens, y=y_domain_1) 

        synthesis_domain_2_ema,_ = ema_model(x=input_image_domain_1, context=clip_text_encoding_domain_2_tokens,  y=y_domain_2) 
        synthesis_domain_1_ema,_ = ema_model(x=input_image_domain_2, context=clip_text_encoding_domain_1_tokens,  y=y_domain_1) 

        input_domain_1_np = input_image_domain_1.detach().cpu().numpy() 
        input_domain_2_np = input_image_domain_2.detach().cpu().numpy()

        output_domain_2_np = synthesis_domain_2.detach().cpu().numpy() 
        output_domain_1_np = synthesis_domain_1.detach().cpu().numpy() 

        error_domain_1 = np.abs(input_domain_1_np - output_domain_1_np) 
        error_domain_2 = np.abs(input_domain_2_np - output_domain_2_np)

        psnr_domain_1.append(psnr.compute_psnr(output_domain_1_np[0,0], input_domain_1_np[0,0]))
        psnr_domain_2.append(psnr.compute_psnr(output_domain_2_np[0,0], input_domain_2_np[0,0]))
        ssim_domain_1.append(psnr.compute_ssim(output_domain_1_np[0,0], input_domain_1_np[0,0]))
        ssim_domain_2.append(psnr.compute_ssim(output_domain_2_np[0,0], input_domain_2_np[0,0]))

        if i == 0:
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,12),dpi=100, layout='tight', sharex=True, sharey=True)
            fig.suptitle("epoch: " + str(epoch))
            ax[0,0].imshow(input_domain_1_np[0,0], cmap="gray")
            ax[0,0].set_title("input")
            ax[0,1].imshow(output_domain_2_np[0,0], cmap="gray")
            ax[0,1].set_title("synthesized")
            ax[1,1].imshow(input_domain_2_np[0,0], cmap="gray")
            ax[1,1].set_title("target")
            ax[1,0].imshow(error_domain_2[0,0], cmap="gray")
            ax[1,0].set_title("error")
            ax[0,0].set_axis_off() 
            ax[0,1].set_axis_off()       
            ax[1,0].set_axis_off()     
            ax[1,1].set_axis_off()       
    
            fig.savefig(result_dir + "/val/" + str(epoch) +  "/" +  "from_" + args.domain_1 + "_to_" + args.domain_2 + "_domain_2.png")
            plt.close("all")

            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,12),dpi=100, layout='tight', sharex=True, sharey=True)
            fig.suptitle("epoch: " + str(epoch))
            ax[0,0].imshow(input_domain_2_np[0,0], cmap="gray")
            ax[0,0].set_title("input")
            ax[0,1].imshow(output_domain_1_np[0,0], cmap="gray")
            ax[0,1].set_title("synthesized")
            ax[1,1].imshow(input_domain_1_np[0,0], cmap="gray")
            ax[1,1].set_title("target")
            ax[1,0].imshow(error_domain_1[0,0], cmap="gray")
            ax[1,0].set_title("error")
            ax[0,0].set_axis_off() 
            ax[0,1].set_axis_off()       
            ax[1,0].set_axis_off()     
            ax[1,1].set_axis_off()       
    
            fig.savefig(result_dir + "/val/" + str(epoch) +  "/" +  "from_" + args.domain_2 + "_to_" + args.domain_1 + "_domain_1.png")
            plt.close("all")

        output_domain_2_np = synthesis_domain_2_ema.detach().cpu().numpy()
        output_domain_1_np = synthesis_domain_1_ema.detach().cpu().numpy() 

        error_domain_1 = np.abs(input_domain_1_np - output_domain_1_np)
        error_domain_2 = np.abs(input_domain_2_np - output_domain_2_np)

        psnr_domain_1_ema.append(psnr.compute_psnr(output_domain_1_np[0,0], input_domain_1_np[0,0]))
        psnr_domain_2_ema.append(psnr.compute_psnr(output_domain_2_np[0,0], input_domain_2_np[0,0]))
        ssim_domain_1_ema.append(psnr.compute_ssim(output_domain_1_np[0,0], input_domain_1_np[0,0]))
        ssim_domain_2_ema.append(psnr.compute_ssim(output_domain_2_np[0,0], input_domain_2_np[0,0]))

       
        if i == 0:
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,12),dpi=100, layout='tight', sharex=True, sharey=True)
            fig.suptitle("EMA epoch: " + str(epoch))
            ax[0,0].imshow(input_domain_1_np[0,0], cmap="gray")
            ax[0,0].set_title("input")
            ax[0,1].imshow(output_domain_2_np[0,0], cmap="gray")
            ax[0,1].set_title("synthesized")
            ax[1,1].imshow(input_domain_2_np[0,0], cmap="gray")
            ax[1,1].set_title("target")
            ax[1,0].imshow(error_domain_2[0,0], cmap="gray")
            ax[1,0].set_title("error")
            ax[0,0].set_axis_off() 
            ax[0,1].set_axis_off()       
            ax[1,0].set_axis_off()     
            ax[1,1].set_axis_off()        
    
            fig.savefig(result_dir + "/val/" + str(epoch) +  "/" +  "from_" + args.domain_1 + "_to_" + args.domain_2 + "_domain_2_ema.png")
            plt.close("all")

            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,12),dpi=100, layout='tight', sharex=True, sharey=True)
            fig.suptitle("EMA epoch: " + str(epoch))
            ax[0,0].imshow(input_domain_2_np[0,0], cmap="gray")
            ax[0,0].set_title("input")
            ax[0,1].imshow(output_domain_1_np[0,0], cmap="gray")
            ax[0,1].set_title("synthesized")
            ax[1,1].imshow(input_domain_1_np[0,0], cmap="gray")
            ax[1,1].set_title("target")
            ax[1,0].imshow(error_domain_1[0,0], cmap="gray")
            ax[1,0].set_title("error")
            ax[0,0].set_axis_off() 
            ax[0,1].set_axis_off()       
            ax[1,0].set_axis_off()     
            ax[1,1].set_axis_off()       
    
            fig.savefig(result_dir + "/val/" + str(epoch) +  "/" +  "from_" + args.domain_2 + "_to_" + args.domain_1 + "_domain_1_ema.png")
            plt.close("all")
    

    return {'domain_1': args.domain_1, 'domain_2':args.domain_2, 'psnr_domain_1': np.nanmean(psnr_domain_1), 'psnr_domain_2':np.nanmean(psnr_domain_2),'ssim_domain_1':np.nanmean(ssim_domain_1),'ssim_domain_2':np.nanmean(ssim_domain_2)
            ,'psnr_domain_1_ema': np.nanmean(psnr_domain_1_ema), 'psnr_domain_2_ema':np.nanmean(psnr_domain_2_ema),'ssim_domain_1_ema':np.nanmean(ssim_domain_1_ema),'ssim_domain_2_ema':np.nanmean(ssim_domain_2_ema)}