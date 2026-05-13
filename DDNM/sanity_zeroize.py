"""Sanity: load new SPNN ckpt and check A/Ap behavior with/without zeroize."""
import argparse, os, sys, types, torch, numpy as np

THIS = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, THIS)
sys.path.insert(0, os.path.abspath(os.path.join(THIS, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(THIS, "..", "centernet_ref")))

from guided_diffusion.diffusion_copy_copy import (
    Diffusion, zeroize_y_outside_predicted_peaks)
from datasets.voc import VOCValForDDNM


class Cfg(types.SimpleNamespace):
    pass

def main():
    import sys
    ckpt_arg = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/assaf.sh/projects/SPNN/centernet_ref/ckpt/spnn_zeroize_v1/checkpoint.t7"
    args = Cfg(
        detector_ckpt=ckpt_arg,
        detector_num_classes=20, detector_deep_det_head=True,
        detector_deep_head_hidden=128, detector_two_block_head=False,
        detector_head_mode="affine", detector_head_mix_type="householder",
        detector_head_mix_reflections=0,
        detector_hmap_init_scale=0.01, detector_hmap_init_bias=-2.19,
        no_hmap_scale=True, no_hmap_bias=True, internal_head_affine=True,
    )
    config = Cfg(data=Cfg(image_size=256, channels=3))
    cfg_obj = Cfg(model=Cfg(), data=Cfg(image_size=256, channels=3),
                  diffusion=Cfg(beta_schedule="linear", beta_start=0.0001,
                                beta_end=0.02, num_diffusion_timesteps=1000))
    cfg_obj.device = torch.device("cuda")

    runner = Diffusion.__new__(Diffusion)
    runner.config = cfg_obj
    runner.device = cfg_obj.device
    runner.args = Cfg(exp="exp")

    classifier, A, Ap = runner._build_detection_A_Ap(args)

    ds = VOCValForDDNM("/home/assaf.sh/projects/SPNN/centernet_ref/data", image_size=256)
    x01, _ = ds[0]
    x = (x01.unsqueeze(0).to(runner.device) * 2.0 - 1.0)
    print(f"x range=[{x.min():.3f}, {x.max():.3f}]")

    with torch.no_grad():
        y = A(x)
        print(f"y_clean   range=[{y.min():.3f}, {y.max():.3f}]")
        x_rec = Ap(y)
        print(f"  Ap(y)              range=[{x_rec.min():.3f}, {x_rec.max():.3f}]  rel_err={(x-x_rec).norm()/x.norm():.4f}")
        for tk in (1, 100):
            yz = zeroize_y_outside_predicted_peaks(y, nc=20, top_k=tk, score_thresh=0.1)
            xz = Ap(yz)
            nz = int((yz != 0).any(1).sum())
            print(f"  Ap(B(y) k={tk:3d})    range=[{xz.min():.3f}, {xz.max():.3f}]  rel_err={(x-xz).norm()/x.norm():.4f}  cells={nz}")
            # Round-trip in y space: how well does A;Ap;B reproduce B(y)?
            y_rt = A(xz)
            yz_rt = zeroize_y_outside_predicted_peaks(y_rt, nc=20, top_k=tk, score_thresh=0.1)
            print(f"      A(Ap(B(y)))  range=[{y_rt.min():.3f}, {y_rt.max():.3f}]  vs y range=[{y.min():.3f}, {y.max():.3f}]")
            print(f"      ||y_rt-y||/||y||={(y-y_rt).norm()/y.norm():.4f}  ||B(y_rt)-B(y)||/||B(y)||={(yz_rt-yz).norm()/(yz.norm()+1e-8):.4f}")
        # Now test with NOISY x — what step 0 of DDNM actually feeds.
        torch.manual_seed(0)
        x_noise = torch.randn_like(x).clamp(-1, 1)  # rough proxy for x0_t at very high t
        y_n = A(x_noise)
        print(f"\ny_noise   range=[{y_n.min():.3f}, {y_n.max():.3f}]")
        for tk in (1, 100):
            yz_n = zeroize_y_outside_predicted_peaks(y_n, nc=20, top_k=tk, score_thresh=0.1)
            xz_n = Ap(yz_n)
            nz = int((yz_n != 0).any(1).sum())
            print(f"  Ap(B(y_noise) k={tk:3d})  range=[{xz_n.min():.3f}, {xz_n.max():.3f}]  cells={nz}")

        # Test with HUGE x0_t (matches early-sampling x0_t scale ~200)
        for scale in (3.0, 10.0, 50.0, 200.0):
            x_big = scale * torch.randn_like(x)
            y_b = A(x_big)
            yz_b = zeroize_y_outside_predicted_peaks(y_b, nc=20, top_k=100, score_thresh=0.1)
            xz_b = Ap(yz_b)
            nz = int((yz_b != 0).any(1).sum())
            print(f"  scale={scale:5.1f}: y range=[{y_b.min():.1f}, {y_b.max():.1f}]  Ap(B(y)) range=[{xz_b.min():.2f}, {xz_b.max():.2f}]  cells={nz}")

        # Simulate the BP step: x0_t_hat = Ap(y_cur + lambda*(y_tar - y_proj))
        # with everything sane
        x_cur = x_noise  # noisy
        y_cur = A(x_cur)
        yz_cur = zeroize_y_outside_predicted_peaks(y_cur, nc=20, top_k=100, score_thresh=0.1)
        x_hat_cur = Ap(yz_cur)
        # target = the clean image
        yz_target = zeroize_y_outside_predicted_peaks(y, nc=20, top_k=100, score_thresh=0.1)
        x_hat_target = Ap(yz_target)
        # NLBP step: y_final = y_cur + lambda * (A(Ap(B(y))) - A(Ap(B(y_cur))))
        y_tar = A(x_hat_target)
        y_proj = A(x_hat_cur)
        for lam in (0.1, 0.5, 1.0):
            y_final = y_cur + lam * (y_tar - y_proj)
            x_final = Ap(y_final)
            print(f"  BP step (lam={lam}): y_final range=[{y_final.min():.1f}, {y_final.max():.1f}]  Ap(y_final) range=[{x_final.min():.1f}, {x_final.max():.1f}]")

if __name__ == "__main__":
    main()
