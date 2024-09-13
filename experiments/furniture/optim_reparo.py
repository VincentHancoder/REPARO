# adding depth in point proxy

import random
from tqdm.std import tqdm
import numpy as np
import torch
from pytorch3d.renderer import Materials
from pytorch3d.io import load_objs_as_meshes, IO
from pytorch3d.structures import join_meshes_as_scene
from pytorch3d.transforms import euler_angles_to_matrix

import sys,os,json,time,copy
sys.path.append(".")
from experiments.common import *
from experiments.logger import Logger
from core.LossFunction_v2 import PointLossFunction, SemanticLossFunction, DepthLossFunction, PointLossFunction6D
from core.NvDiffRastRenderer import NVDiffRastFullRenderer
from PIL import Image
from pathlib import Path

import argparse

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

import logging
from experiments.utils import setup_logging
from torch.utils.tensorboard import SummaryWriter

from evaluation.utils import check_dir

class Furniture:
    def __init__(self) -> None:
        self.meshes=[]
        self.origin_trans = []
        
        self.optim_mesh = []
        self.optim_trans=[]
        pass
    
    def load_furniture_scene(self, device, task, filepath):
        files = os.listdir(filepath)
        ply_mesh_loader = IO()
        scene={"origin_meshes":[],"optim":[],"sensors":[],"meshes":[],"origin_sensors":[],"material":[]}
        for file in files:
            fullpath = os.path.join(filepath,file)
            if file.endswith(".obj"):
                mesh = load_objs_as_meshes([fullpath],device=device)
            elif file.endswith(".ply"):
                mesh = ply_mesh_loader.load_mesh(fullpath).to(device)
            else:
                continue
            trans_x = torch.zeros(1,requires_grad=True)
            trans_y = torch.zeros(1,requires_grad=True)
            trans_z = torch.zeros(1,requires_grad=True)
            rot_y = torch.zeros(1,requires_grad=True)
            optim_trans={}
            optim_trans.update({"translation_x":trans_x})
            optim_trans.update({"translation_y":trans_y})
            optim_trans.update({"translation_z":trans_z})
            optim_trans.update({"rotation_y":rot_y})
            scene["optim"]=scene["optim"]+[trans_x,trans_z,trans_y,rot_y]
            self.meshes.append({"model":mesh,"optim":True,"trans":{},"optim_trans":optim_trans})
        
        cam = np.loadtxt(os.path.join(filepath,"camera.txt")).tolist()
        poss = []
        cens=[]
        for i in range(0,len(cam),2):
            pos = [cam[i+0][0],cam[i+0][2],-cam[i+0][1]]
            rot = [cam[i+1][0],cam[i+1][2],cam[i+1][1]]
            v = torch.tensor([0,-1,0]).float()
            rot = torch.tensor(rot).float()
            dir = torch.matmul(v,euler_angles_to_matrix(rot,convention="XYZ"))
            dir[2]*=-1
            center = torch.tensor(pos).float()+dir
            poss.append(pos)
            cens.append(center.tolist())

        task["view"].update({"num":len(poss),"direction":"manual","position":poss,"center":cens})
        return scene, task
    
    def gen_mesh(self,optim):
        tmp = []
        for mesh_info in self.meshes:
            src_mesh = mesh_info["model"].clone()
            if optim==True:
                trans_x = mesh_info["optim_trans"]["translation_x"]
                trans_y = mesh_info["optim_trans"]["translation_y"]
                trans_z = mesh_info["optim_trans"]["translation_z"]
                rot_y = mesh_info["optim_trans"]["rotation_y"]
                # import pdb;pdb.set_trace()
                rot = torch.cat([torch.tensor([0.0]),torch.tensor([0.0]),torch.tensor([0.0])]).to(device)
                optim_trans = torch.cat([trans_x,trans_y,trans_z]).to(device)
                optim_rot = euler_angles_to_matrix(rot,convention="XYZ")
                with torch.no_grad():
                    rot_center = torch.mean(src_mesh.verts_list()[0],axis=0)
                src_mesh.offset_verts_(-rot_center)
                src_mesh.transform_verts_(optim_rot)
                src_mesh.offset_verts_(rot_center)
                src_mesh.offset_verts_(optim_trans)
            tmp.append(src_mesh)
        return join_meshes_as_scene(tmp)


def run(task_name, task):

    bbox = task["bbox"]
    show = task.get("show", False)
    debug = task.get("debug", False)
    methods = task.get("method", ["origin", "ours"])
    resolution = task["resolution"]
    settings = task["setting"]
    folder_path = task["folder_path"]
    check_dir(folder_path)

    # set logger
    setup_logging(log_file=os.path.join(folder_path, "record.log"), level=logging.INFO)
    tensorboard_writer = SummaryWriter(log_dir=folder_path)
    # record config
    logging.info("="*10 + "config" + "="*10)
    for k, v in task.items():
        logging.info(f"{k}:{v}")
    logging.info("="*10 + "config" + "="*10)

    Logger.init(exp_name=task_name, folder_name=folder_path,show=show, debug=debug, path="results/")
    # import pdb;pdb.set_trace()
    #Logger.save_config(task)
    #Logger.save_file("./PointRenderer/NvDiffRastRenderer.py")
    #Logger.save_file("./PointRenderer/LossFunction.py")

    material = Materials(device=device, ambient_color=((0.5, 0.5, 0.5),),specular_color=((0.2,0.2,0.2),),diffuse_color=((0.7,0.7,0.7),))

    testnum=0
    with torch.no_grad():
        model = Furniture()
        #
        gt_scene,task = model.load_furniture_scene(device, task, task["gt_scene_file"])# gt_scene 
        # import pdb;pdb.set_trace()
        sensors, torch_cameras, torch_lights = config_view_light(task, device)
        num_views = len(torch_cameras)
        gt_mesh = model.gen_mesh(optim=False)
        setup_seed(testnum)
        gt_scene["material"]=material
        gt_scene["meshes"]= [{"model": gt_mesh}]
        gt_scene["sensors"]= sensors
        gt_path = task["gt_path"]

        renderer = NVDiffRastFullRenderer(device=device, settings=task.get(
            "renderer"), resolution=resolution)
        
        torch_gt_renderer, torch_sil_renderer, torch_soft_renderer, torch_point_renderer = get_pytorch3d_renderer(
            task.get("renderer").get("background", True), faces_per_pixel=4, resolution=resolution[0], persp=True)
        
        gt_img = renderer.render(gt_scene, DcDt=False)  
    
        def tensor_to_list(value):
            if isinstance(value, torch.Tensor):
                return value.tolist()  
            elif isinstance(value, dict):
                return {k: tensor_to_list(v) for k, v in value.items()}  
            elif isinstance(value, list):
                return [tensor_to_list(v) for v in value]  
            else:
                return value 
        
        
        ###
        Logger.save_scene(gt_scene,name="%03d/scene_gt"%(testnum))
        
        image_gt= Image.open(gt_path).convert('RGB').resize(resolution) #[256,256,3]
        #import pdb;pdb.set_trace()
        image_array = np.array(image_gt)
        tensor_gt = image_array
        tensor_gt = tensor_gt / 255.0 # image 
        gt_img_torch = torch.tensor(tensor_gt).to("cuda:0")
        gt_img_torch = torch.flip(gt_img_torch,dims=[0])
        gt_img_torch = [gt_img_torch]
        
        gt_sil_torch = [torch_sil_renderer(gt_scene["meshes"][0]["model"], cameras=torch_cameras[i], lights=torch_lights[i], materials = material)[0] for i in range(num_views)]

        for i in range(num_views):
            Logger.save_img("%03d_gtimg_torch_%d.png" % ( 
                testnum,i), gt_img_torch[i].cpu().numpy(), flip=False)
            # Logger.save_img("%03d_gtimg_nv_%d.png" % ( 
            #     testnum,i), gt_img["images"][i].cpu().numpy(), flip=True)
    
    for method in methods:
        setup_seed(testnum)
        model = Furniture()
        scene, task = model.load_furniture_scene(device, task, task["optim_scene_file"])
        sensors, torch_cameras, torch_lights = config_view_light(task, device) 
        # TODO: sensors  
        num_views = len(torch_cameras)
        setup_seed(testnum)
        scene["material"]=material
        scene["sensors"]= sensors
        optimizer_type = settings.get("optimizer","Adam")

        gamma = settings["decay"]
        lr = settings["learning_rate"]
        if settings["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(scene["optim"], lr=lr)
        else:
            optimizer = torch.optim.SGD(
                scene["optim"], lr=lr, momentum=0.9)
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=gamma, last_epoch=-1)
        
        view_per_iter = min(num_views, task["setting"]["view_per_iter"])

        #if method in ["our","finetune"]:
        #    pbar = tqdm(range(settings["Niter"]))
        #elif method[:9]=="pytorch3d":
        #    pbar = tqdm(range(int(settings["Niter"]*2.5)))
        #else:
        #    pbar = tqdm(range(settings["Niter"]*5))

        pbar = tqdm(range(settings["Niter"]))
        if task.get("loss", {}).get("enable_depth_proxy", False):
            loss_func = PointLossFunction6D(
                debug=debug, 
                resolution=task["resolution"], 
                settings = task["matching"], 
                device=device, 
                renderer=renderer, # render NVDiffRastRender
                num_views=num_views,
                logger=Logger,
                depth_model_path="./dpt-dinov2-base-kitti")
        else:
            loss_func = PointLossFunction(
                debug=debug, 
                resolution=task["resolution"], 
                settings = task["matching"], 
                device=device, 
                renderer=renderer, # render NVDiffRastRender
                num_views=num_views,
                logger=Logger)    
        loss_weight = {}
        loss_weight["drot_loss"] = task.get("loss", {}).get("drot_loss_weight", 1.0)
        
        semantic_loss_func = None
        if task.get("loss", {}).get("enable_clip_loss", False) or task.get("loss", {}).get("enable_dino_loss", False):
            semantic_loss_func = SemanticLossFunction(args=task, device=device)
        depth_loss_func = None
        if task.get("loss", {}).get("enable_depth_loss", False):
            depth_loss_func = DepthLossFunction(args=task, model_path="./dpt-dinov2-base-kitti", device=device)

        start_time = time.time()
        for iter in pbar:
            optim_mesh = model.gen_mesh(True) #  
            scene["meshes"]=[{"model": optim_mesh}]
            if method==methods[0] and iter==0:
                Logger.save_scene(scene,name="%03d/scene_init"%(testnum))
                with torch.no_grad():
                    render_res = renderer.render(scene, DcDt=False)
                    for i in range(num_views):
                        Logger.save_img("%03d/init_img_nv_%d.png" % (testnum,i), render_res["images"][i].cpu().numpy(), flip=True)
                   
            loss_all = torch.tensor(0.0, device=device)
            for j in range(view_per_iter):
                select_view = np.random.randint(0,num_views)
                if method == "our_torch":
                    # w/o point proxy
                    #render_res = renderer.render(scene["meshes"][0]["model"], cameras=torch_cameras[select_view], lights=torch_lights[select_view], materials = material)
                    #import pdb;pdb.set_trace()
                    loss_record = {}
                    render_res = renderer.render(scene,view=select_view,DcDt=False)
                    drot_loss = loss_func.get_loss(render_res, gt_img_torch[select_view][...,:3],view=0)
                    loss_record["drot_loss"] = drot_loss
                    # NOTE: In the first iteration, render_res["images"].requires_grad=False
                    if semantic_loss_func is not None:
                        semantic_loss = semantic_loss_func(render_image=render_res["images"], ref_image=gt_img_torch[select_view], iter=iter)
                        loss_record.update(semantic_loss)
                    if depth_loss_func is not None:
                        depth_loss = depth_loss_func(render_image=render_res["images"], ref_image=gt_img_torch[select_view], iter=iter)
                        loss_record.update(depth_loss)
                    
                    loss = 0.0
                    formatted_loss = f"iter: {iter}"
                    for key, value in loss_record.items():
                        value = value * loss_weight.get(key, 1.0)
                        loss += value
                        formatted_loss += " @{}: {:.6f}".format(key, value.item())
                        tensorboard_writer.add_scalar(key, value.item(), iter)
                    formatted_loss += " @total loss: {:.6f}".format(loss.item())
                    tensorboard_writer.add_scalar(key, loss.item(), iter)
                    logging.info(formatted_loss)
                
                if method == "our":
                    
                    # import pdb;pdb.set_trace()
                    loss_record = {}
                    drot_loss, render_res = loss_func(
                        gt_img_torch, iteration=iter, scene=scene, view=select_view)
                    loss_record["drot_loss"] = drot_loss
                    # NOTE: In the first iteration, render_res["images"].requires_grad=False
                    if semantic_loss_func is not None:
                        semantic_loss = semantic_loss_func(render_image=render_res["images"], ref_image=gt_img_torch[select_view], iter=iter)
                        loss_record.update(semantic_loss)
                    if depth_loss_func is not None:
                        depth_loss = depth_loss_func(render_image=render_res["images"], ref_image=gt_img_torch[select_view], iter=iter)
                        loss_record.update(depth_loss)
                    
                    loss = 0.0
                    formatted_loss = f"iter: {iter}"
                    for key, value in loss_record.items():
                        value = value * loss_weight.get(key, 1.0)
                        loss += value
                        formatted_loss += " @{}: {:.6f}".format(key, value.item())
                        tensorboard_writer.add_scalar(key, value.item(), iter)
                    formatted_loss += " @total loss: {:.6f}".format(loss.item())
                    tensorboard_writer.add_scalar(key, loss.item(), iter)
                    logging.info(formatted_loss)
                loss_all+=loss
                with torch.no_grad():
                    render_res = renderer.render(scene, DcDt=False)
                    for i in range(num_views):
                        Logger.add_image("%03d_render_%s_%d"%(testnum,method,i),render_res["images"][i],flip=True)
                    #cv2.waitKey(0)
            
            optimizer.zero_grad()
            loss_all/=view_per_iter
            loss_all.backward()
            optimizer.step()
            if iter<settings["Niter"]:
                scheduler.step()
            
            MAE = torch.mean(torch.abs(gt_mesh.verts_list()[0]-optim_mesh.verts_list()[0])).item()
            pbar.set_description("%d %s MAE:%.4f" % (testnum, method, MAE))
            Logger.add_scalar("%03d_render_%s_mae" % (testnum, method), MAE, iter)
        
        end_time = time.time()   
        render_res = renderer.render(scene, DcDt=False)
        log_final(scene, gt_img, render_res,testnum,method,Logger,num_views,gt_mesh)
        Logger.clean()
        Logger.show_metric()
    Logger.exit()

def main(args):
    ### parse config
    config = json.load(open(args.config, 'r'))
    task_name = config["task_name"]
    _data_info_list = json.load(open(config['data_info_file']))
    # select according to chunk-id
    data_info_list = _data_info_list[args.chunk_id::args.chunk_num]
    if args.case_name is not None:
        data_info_list = [x for x in _data_info_list if x['name']==args.case_name]
    
    for data_info_item in data_info_list:
        # prepare config for each sample
        data_info_item["optim_scene_file"] = os.path.join(config["data_dir"], data_info_item["optim_scene_file"])
        data_info_item["gt_path"] = os.path.join(config["data_dir"], data_info_item["gt_path"])
        data_info_item["gt_scene_file"] = os.path.join(config["data_dir"], data_info_item["gt_scene_file"])
        data_info_item["folder_path"] = os.path.join(config["save_dir"], config["task_name"], data_info_item["name"]) # here data_info_item["name"]  is the case name, for example, GSO_CHILDREN_BEDROOM_CLASSIC
        config_item = {**config, **data_info_item} 
        run(task_name=task_name, task=config_item)
    
    return None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/furniture_rgbxy.json")
    parser.add_argument("--chunk_id", default=0, type=int)
    parser.add_argument("--chunk_num", default=1, type=int)
    parser.add_argument("--case_name", default=None, type=str, help="used for special case in meta_info.json")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)