import os
import argparse
import torch as th
import torch.nn.functional as F
import time
import conf_mgt
from utils import yamlread
from guided_diffusion import dist_util
import numpy as np

try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
)

def toU8(sample):
    if sample is None:
        return sample
    #sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample

def main(conf: conf_mgt.Default_Conf):
    print("Start", conf['name'])

    print(conf.data['eval'][conf.get_default_eval_name()]["paths"])

    device = dist_util.dev("mps" if th.backends.mps.is_available() else "cpu") #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #device = dist_util.dev(conf.get('device'))


    # 🔹 Load Model
    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    show_progress = conf.show_progress

    if conf.classifier_scale > 0 and conf.classifier_path:
        print("loading classifier...")
        classifier = create_classifier(
            **select_args(conf, classifier_defaults().keys()))
        classifier.load_state_dict(
            dist_util.load_state_dict(os.path.expanduser(conf.classifier_path), map_location="cpu")
        )
        classifier.to(device)
        if conf.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

        def cond_fn(x, t, y=None, gt=None, **kwargs):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return th.autograd.grad(selected.sum(), x_in)[0] * conf.classifier_scale
    else:
        cond_fn = None

    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None
        return model(x, t, y if conf.class_cond else None, gt=gt)

    print("sampling...")

    dset = 'eval'
    eval_name = conf.get_default_eval_name()

    # 🔹 Handle .npy data instead of loading images
    gt_path = conf.data[dset][eval_name]["gt_path"]
    mask_path = conf.data[dset][eval_name]["mask_path"]

    print(f"Loading ground truth from {gt_path} and masks from {mask_path}...")

    # ✅ Load Spectral Patches (Ground Truth)
    patches = np.load(gt_path)  # Shape: (N, 256, 256)

    # ✅ Load Masks
    masks = np.load(mask_path) if mask_path else np.ones_like(patches)  # Default to null mask

    # 🔹 Convert to PyTorch tensors
    patches = th.tensor(patches, dtype=th.float32).unsqueeze(1).to(device)  # (N, 1, 256, 256)
    masks = th.tensor(masks, dtype=th.float32).unsqueeze(1).to(device)  # (N, 1, 256, 256)

    batch_size = patches.shape[0]

    # Create model kwargs
    model_kwargs = {"gt": patches, "gt_keep_mask": masks}

    if conf.cond_y is not None:
        classes = th.ones(batch_size, dtype=th.long, device=device)
        model_kwargs["y"] = classes * conf.cond_y
    else:
        classes = th.randint(low=0, high=NUM_CLASSES, size=(batch_size,), device=device)
        model_kwargs["y"] = classes

    # 🔹 Select Sampling Function
    sample_fn = diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop

    # 🔹 Run RePaint (Denoising/Inpainting)
    result = sample_fn(
        model_fn,
        (batch_size, 1, conf.image_size, conf.image_size),  # Changed from (B, 3, H, W) to (B, 1, H, W)
        clip_denoised=conf.clip_denoised,
        model_kwargs=model_kwargs,
        cond_fn=cond_fn,
        device=device,
        progress=show_progress,
        return_all=True,
        conf=conf
    )

    # Convert Results
    #srs = toU8(result['sample'])
    #gts = toU8(result['gt'])
    #lrs = toU8(result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) *
    #           th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask')))
    #gt_keep_masks = toU8((model_kwargs.get('gt_keep_mask') * 2 - 1))

    # 🔹 Save Results
    #conf.eval_imswrite(
    #    srs=srs, gts=gts, lrs=lrs, gt_keep_masks=gt_keep_masks,
    #    img_names=[f"patch_{i}.png" for i in range(batch_size)],
    #    dset=dset, name=eval_name, verify_same=False)

    # Ensure the paths exist
    os.makedirs(conf.data[dset][eval_name]["paths"]["srs"], exist_ok=True)
    os.makedirs(conf.data[dset][eval_name]["paths"]["gts"], exist_ok=True)
    os.makedirs(conf.data[dset][eval_name]["paths"]["lrs"], exist_ok=True)
    os.makedirs(conf.data[dset][eval_name]["paths"]["gt_keep_masks"], exist_ok=True)

    # Save the results as .npy files
    np.save(os.path.join(conf.data[dset][eval_name]["paths"]["srs"], f"{eval_name}_srs.npy"), result["sample"].cpu().numpy())
    
    np.save(os.path.join(conf.data[dset][eval_name]["paths"]["gts"], f"{eval_name}_gts.npy"), result["gt"].cpu().numpy())
    
    np.save(os.path.join(conf.data[dset][eval_name]["paths"]["lrs"], f"{eval_name}_lrs.npy"), 
            (result.get("gt") * model_kwargs.get("gt_keep_mask") + (-1) * 
            th.ones_like(result.get("gt")) * (1 - model_kwargs.get("gt_keep_mask"))).cpu().numpy())
    
    np.save(os.path.join(conf.data[dset][eval_name]["paths"]["gt_keep_masks"], f"{eval_name}_gt_keep_masks.npy"), 
            ((model_kwargs.get("gt_keep_mask") * 2 - 1)).cpu().numpy())

    #print("✅ Results successfully saved as .npy files!")

    '''
    # Define the correct save path
    log_dir = "/Users/saviojr/Documents/Thesis/Code/RePaint_npy/log/spectra_example"

    # Ensure all subdirectories exist
    os.makedirs(os.path.join(log_dir, "inpainted"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "gt_masked"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "gt_keep_mask"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "gt"), exist_ok=True)

    # Save .npy files in the correct locations
    np.save(os.path.join(log_dir, "inpainted", f"{eval_name}_srs.npy"), result["sample"].cpu().numpy())
    np.save(os.path.join(log_dir, "gt", f"{eval_name}_gts.npy"), result["gt"].cpu().numpy())
    np.save(os.path.join(log_dir, "gt_masked", f"{eval_name}_lrs.npy"), (result["gt"] * model_kwargs["gt_keep_mask"]).cpu().numpy())
    np.save(os.path.join(log_dir, "gt_keep_mask", f"{eval_name}_gt_masks.npy"), model_kwargs["gt_keep_mask"].cpu().numpy())
    '''
    
    print("Sampling complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default=None)
    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.get('conf_path')))
    main(conf_arg)