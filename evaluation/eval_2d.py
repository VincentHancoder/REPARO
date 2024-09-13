import os
import numpy as np
import cv2
import lpips
import torch
import clip
from PIL import Image
from glob import glob
import re
import json
from tqdm import tqdm
from skimage.metrics import structural_similarity as get_ssim
from skimage.metrics import peak_signal_noise_ratio as get_psnr
import cv2
import argparse
from utils import prepare_mesh, get_view_images, check_dir

class Eval2D():
    def __init__(self, clip_model_path="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # load clip model
        print("Loading clip model...")
        self.clip_model, self.preprocess_clip = clip.load(clip_model_path, device=self.device)
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        # load lpips model
        self.lpips_model = lpips.LPIPS(net='alex').to(self.device)

    @torch.no_grad()
    def get_clip_feat(self, img):
        img = self.preprocess_clip(img).unsqueeze(0).to(self.device)
        feat = self.clip_model.encode_image(img)
        return feat
    
    @torch.no_grad()
    def get_clip_score(self, img_one, img_two):
        feat_one = self.get_clip_feat(img_one)
        feat_two = self.get_clip_feat(img_two)
        clip_score = self.cosine_similarity(feat_one, feat_two)
        return clip_score.cpu().detach().item()

    @torch.no_grad()
    def get_lpips_score(self, img_path_one, img_path_two):
        img1_tensor = lpips.im2tensor(lpips.load_image(img_path_one)).to(self.device)
        img2_tensor = lpips.im2tensor(lpips.load_image(img_path_two)).to(self.device)
        lpips_score = self.lpips_model.forward(img1_tensor, img2_tensor)
        return lpips_score.cpu().detach().item()
    
    def get_psnr_ssim(self, img_one, img_two):
        img_one = cv2.cvtColor(np.array(img_one), cv2.COLOR_RGB2BGR)
        img_two = cv2.cvtColor(np.array(img_two), cv2.COLOR_RGB2BGR)
        img_one_gray = cv2.cvtColor(img_one, cv2.COLOR_BGR2GRAY)
        img_two_gray = cv2.cvtColor(img_two, cv2.COLOR_BGR2GRAY)
        psnr = get_psnr(img_one_gray, img_two_gray)
        ssim = get_ssim(img_one_gray, img_two_gray)
        return psnr, ssim
    
    def __call__(self, img_path_one, img_path_two):
        img_one = Image.open(img_path_one).convert('RGB')
        img_two = Image.open(img_path_two).convert('RGB')
        clip_score = self.get_clip_score(img_one, img_two)
        psnr, ssim = self.get_psnr_ssim(img_one, img_two)
        lpips_score = self.get_lpips_score(img_path_one, img_path_two)
        return {"clip_score": clip_score, "psnr": psnr, "ssim": ssim, "lpips_score": lpips_score}
    
def eval_single(pred_path, tgt_path, evaluator=None):
    """
    Loading images of different views from predict and target folders, and evaluating them one by one.
    """
    tgt_views = sorted(glob(tgt_path+"/*.png"))
    pred_views = sorted(glob(pred_path+"/*.png"))
    clip_scores, psnrs, ssims, lpips_scores = [], [], [], []
    for tgt_view_path, pred_view_path in zip(tgt_views, pred_views):
        assert tgt_view_path.split("/")[-1] == pred_view_path.split("/")[-1] # check name
        res = evaluator(tgt_view_path, pred_view_path)
        clip_scores.append(res["clip_score"])
        psnrs.append(res["psnr"])
        ssims.append(res["ssim"])
        lpips_scores.append(res["lpips_score"])
    ### summary result
    result = {
        'clip_score': np.mean(clip_scores),
        'psnr': np.mean(psnrs),
        'ssim': np.mean(ssims),
        'lpips': np.mean(lpips_scores)
        }
    return result

def eval(
        config, 
        targets_dir="data/GSO_gt_mod",
        target_material_dir="data/GSO_gt",
        save_dir="results/eval", 
        ):
    check_dir(save_dir)
    ### build a dict to save all info
    meta_info = dict()
    # NOTE: not safe
    config = json.load(open(config, 'r'))
    data_info_list = json.load(open(config['data_info_file']))

    for data_info_item in data_info_list:
        name = data_info_item["name"]
        tgt_name = name[4:] if 'GSO' in name else name  
        meta_info[name] = {
            "name": name, 
            "tgt_name": tgt_name,
            "pred_save_dir": os.path.join(config["save_dir"], config["task_name"], data_info_item["name"]),
            "target_dir": os.path.join(targets_dir, tgt_name),
            "target_material_path": os.path.join(target_material_dir, tgt_name, "materials/textures/texture.png"),
            "save_dir": os.path.join(save_dir, config['task_name'], name), # save dir for each mesh
            }

    ### scale normalize, save to one folder, get different view images
    print("Preparing predicted and target meshes...")
    for key, info in tqdm(meta_info.items()):
        meshobj_save_dir=os.path.join(info['save_dir'], "mesh")
        check_dir(meshobj_save_dir)

        prepare_mesh(
            mesh_pred_path=os.path.join(info["pred_save_dir"], f"000/scene_final_{config['method'][0]}/0.obj"), # NOTE: not safe
            mesh_tgt_path=os.path.join(info["target_dir"], f"{info['tgt_name']}.obj"),
            meshobj_save_dir=meshobj_save_dir)

        tgt_view_img_save_dir = os.path.join(info['save_dir'], "views", "target")
        pred_view_img_save_dir = os.path.join(info['save_dir'], "views", "predict")
        # get view images of target images
        get_view_images(obj_file_path=os.path.join(meshobj_save_dir, "mesh_tgt.obj"), texture_file_path=info["target_material_path"], output_folder=tgt_view_img_save_dir)
        # get view images of predicted images
        get_view_images(obj_file_path=os.path.join(meshobj_save_dir, "mesh_pred.obj"), texture_file_path=os.path.join(meshobj_save_dir, "material_0.png"), output_folder=pred_view_img_save_dir)
        # update meta_info
        info["mesh_save_dir"]=meshobj_save_dir
        info["tgt_view_img_save_dir"]=tgt_view_img_save_dir
        info["pred_view_img_save_dir"]=pred_view_img_save_dir
    
    ### eval 
    evaluator = Eval2D()
    print("Evaluating....")
    clip_score_list, psnr_list, ssim_list, lpips_list = [], [], [], []
    for key, info in tqdm(meta_info.items()):
        result = eval_single(pred_path=info["pred_view_img_save_dir"], tgt_path=info["tgt_view_img_save_dir"], evaluator=evaluator)
        info.update(result)
        clip_score_list.append(result.get("clip_score"))
        psnr_list.append(result.get("psnr"))
        ssim_list.append(result.get("ssim"))
        lpips_list.append(result.get("lpips"))
    
    formatted_output = f"@clip_score: {np.mean(clip_score_list)}\n@psnr: {np.mean(psnr_list)}\n@ssim:{np.mean(ssim_list)}\n@lpips: {np.mean(lpips_list)}"
    print(formatted_output)

    ### save
    metric_save_path = os.path.join(save_dir, config['task_name'], "results.txt")
    info_save_path = os.path.join(save_dir, config['task_name'], "meta_info.json")

    with open(metric_save_path, 'w') as w:
        w.write(formatted_output)
    
    json_obj = json.dumps(meta_info, indent=1)
    with open(info_save_path, 'w') as w:
        w.write(json_obj)
    
    print(f"Results have been saved to {metric_save_path} and {info_save_path}")
    print("Done!")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/furniture_rgbxy.json")
    parser.add_argument("--targets_dir", default="data/GSO_gt_mod")
    parser.add_argument("--target_material_dir", default="data/GSO_gt")
    parser.add_argument("--save_dir", default="results/eval")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    eval(
        config=args.config,
        targets_dir=args.targets_dir,
        target_material_dir=args.target_material_dir,
        save_dir=args.save_dir,
        )
