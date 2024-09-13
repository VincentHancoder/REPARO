


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
from core.LossFunction import PointLossFunction
from core.NvDiffRastRenderer import NVDiffRastFullRenderer
from PIL import Image
from pathlib import Path

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

class Furniture:
    def __init__(self) -> None:
        self.meshes=[]
        self.origin_trans = []
        
        self.optim_mesh = []
        self.optim_trans=[]
        pass
    
    #这是得到真值的函数
    def load_furniture_scene(self, device, task, filepath):
        files = os.listdir(filepath)
        ply_mesh_loader = IO()#定义一个.ply格式的mesh读取器
        scene={"origin_meshes":[],"optim":[],"sensors":[],"meshes":[],"origin_sensors":[],"material":[]}
        for file in files:
            fullpath = os.path.join(filepath,file)
            if file.endswith(".obj"):
                mesh = load_objs_as_meshes([fullpath],device=device)
            elif file.endswith(".ply"):
                mesh = ply_mesh_loader.load_mesh(fullpath).to(device)
            else:
                continue
            # fullpath = os.path.join(filepath,file)
            # mesh = load_objs_as_meshes([fullpath],device=device)
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
        # 这有一步读取目前scene中meshes的操作
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
    
if __name__ == "__main__":
    
    argv = sys.argv
    task_name = argv[0].split("/")[1]
    config_path = os.path.join(argv[0].split("/")[0],task_name,sys.argv[1]+".json")
    task = json.load(open(config_path))
    debug = True

    bbox = task["bbox"]
    show = task.get("show", False)
    debug = task.get("debug", False)
    methods = task.get("method", ["origin", "ours"])
    resolution = task["resolution"]
    settings = task["setting"]
    folder_path = task["folder_path"]

    Logger.init(exp_name=task_name, folder_name=folder_path,show=show, debug=debug, path="results/")
    # import pdb;pdb.set_trace()
    #Logger.save_config(task)
    #Logger.save_file("./PointRenderer/NvDiffRastRenderer.py")
    #Logger.save_file("./PointRenderer/LossFunction.py")

    material = Materials(device=device, ambient_color=((0.5, 0.5, 0.5),),specular_color=((0.2,0.2,0.2),),diffuse_color=((0.7,0.7,0.7),))

    testnum=0
    with torch.no_grad():
        model = Furniture()
        #导入obj格式的操作主要在这里实现,load_furniture_scene调用了load_objs_as_meshes方法
        #所以极有可能问题出在load_objs_as_meshes这个地方
        #
        gt_scene,task = model.load_furniture_scene(device, task, task["gt_scene_file"])#保存gt_scene的地方
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
            "renderer"), resolution=resolution)#render就是NVDiffrastFullRender
        
        torch_gt_renderer, torch_sil_renderer, torch_soft_renderer, torch_point_renderer = get_pytorch3d_renderer(
            task.get("renderer").get("background", True), faces_per_pixel=4, resolution=resolution[0], persp=True)
        
        # gt_img = renderer.render(gt_scene,DcDt=False) 
        # # 将数据中gt文件夹下的gt读取得到一个场景文件，然后将场景渲染得到一张图片gt_img
        # # 这里的 gt image 是倒着的，是不是认为这是 成像平面 ？ （相机的成像是倒的）
        
        # def debug_gt_img(inputs):
        #     image = inputs["images"].cpu().detach().numpy() 
        #     image = (image * 255.0).round().astype(np.uint8)
        #     img = Image.fromarray(image[0], 'RGB')
        #     img.save(f'{os.path.join("debug", Path(config_path).name)}.png')
        
        # # debug
        # if debug:
        #      debug_gt_img(gt_img)
        # 
        
        #这张图片就是用来去指导initial场景的摆放的，那这张渲染图像是不是可以替换为输入图像呢？？？
        # for i in range(num_views):
        #     show_img(gt_img["images"][i],title=str(i),flip=True)
        
        ###将gt_img保存出成为json文件
        def tensor_to_list(value):
            if isinstance(value, torch.Tensor):
                return value.tolist()  # 转换Tensor为列表
            elif isinstance(value, dict):
                return {k: tensor_to_list(v) for k, v in value.items()}  # 递归处理字典
            elif isinstance(value, list):
                return [tensor_to_list(v) for v in value]  # 递归处理列表
            else:
                return value  # 不是Tensor、列表或字典，返回原值

        # 转换整个结构
        #gt_img_list = tensor_to_list(gt_img)
        #print(gt_img_list)
        # with open('record.json', 'w') as json_file:
        #     json.dump(gt_img_list, json_file, indent=4)
        
        
        ###
        Logger.save_scene(gt_scene,name="%03d/scene_gt"%(testnum))#在这个地方将scene_gt保存起来？？？不过这个任务不是图片导向的吗
        #确实是图片导向的，这里并不涉及训练过程，只是使用光流法做推理，基于图片与空间中物体来进行迭代
        # import pdb;pdb.set_trace()
        #渲染出来的
        # gt_img_torch = [torch_gt_renderer(gt_scene["meshes"][0]["model"], cameras=torch_cameras[i], lights=torch_lights[i], materials = material)[0] for i in range(num_views)]
        # gt_img_torch = [torch.flip(gt_img_torch[0],dims=[0])]#XJK写的gt_img_torch要在这里上下翻折一下才可以，debug模式下可以看到这样翻折后的guidance图才是对的
        # ###这里将gt_img_torch保存出来看效果，如果要是效果还是输入图片其实就可以拿这张图片去计算loss
        # gt_image_array = gt_img_torch[0][:,:,0:3].cpu().numpy()
        # gt_image_array = (gt_image_array * 255).round().astype(np.uint8)#这里要乘以255证明渲染出来的gt_img_torch是经过归一化的
        # image_gt_img_torch = Image.fromarray(gt_image_array, 'RGB')
        # image_gt_img_torch.save('gt_img_torch.png')#这里将图片保存之后出来看就是输入的那张图片0_gt.png
        ###
        
        image_gt= Image.open(gt_path).resize(resolution)#[256,256,3]
        #import pdb;pdb.set_trace()
        # 将PIL图像转换为NumPy数组
        image_array = np.array(image_gt)

        # 创建一个256x256x4的数组，用于存放新的带有额外通道的图像
        # 初始化前三个通道为image_array，最后一个通道填充为0
        # tensor_gt = np.zeros((256, 256, 4), dtype=np.float32)
        #tensor_gt[:, :, :3] = image_array#这会让后面的tensor_gt的最后一个维度全部是0
        tensor_gt = image_array
        tensor_gt = tensor_gt / 255.0 #对读取进来的image数组进行归一化
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
        # TODO: sensors 是什么？
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
        loss_func = PointLossFunction(
            debug=debug, 
            resolution=task["resolution"], 
            settings = task["matching"], 
            device=device, 
            renderer=renderer, #这里输给render的参数就是NVDiffRastRender
            num_views=num_views,
            logger=Logger)
        
        start_time = time.time()
        for iter in pbar:
            optim_mesh = model.gen_mesh(True) # 加入可优化的东西，但是没有看懂
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
                    #render_res = renderer.render(scene["meshes"][0]["model"], cameras=torch_cameras[select_view], lights=torch_lights[select_view], materials = material)
                    #import pdb;pdb.set_trace()
                    render_res = renderer.render(scene,view=select_view,DcDt=False)
                    loss = loss_func.get_loss(render_res, gt_img_torch[select_view][...,:3],view=0)
                    print(f"OUR_TORCH_LOSS:{loss}")
                if method == "our":
                    # import pdb;pdb.set_trace()
                    loss, render_res = loss_func(
                        gt_img_torch, iteration=iter, scene=scene, view=select_view)
                    print(f"OUR_LOSS:{loss}")
                elif method == "nvdiffrast":
                    render_res = renderer.render(scene, DcDt=True, view=select_view)
                    loss = torch.mean((render_res["images"][0]-gt_img["images"][select_view])**2)
                elif method == "random":
                    x = random.randint(0, 1)
                    if x == 0:
                        loss, render_res = loss_func(gt_img, iteration=iter, scene=scene, view=select_view)
                    else:
                        render_res = renderer.render(scene, DcDt=True, view=select_view)
                        loss = torch.mean((render_res["images"][0]-gt_img["images"][select_view])**2)
                elif method == "finetune":
                    if iter < settings["Niter"]*0.75:
                        loss, render_res = loss_func(gt_img, iteration=iter, scene=scene, view=select_view)
                    else:
                        if iter==int(settings["Niter"]*0.75):
                            optimizer = torch.optim.SGD(scene["optim"], lr=lr*(gamma**iter))
                        render_res = renderer.render(scene, DcDt=True, view=select_view)
                        loss = torch.mean((render_res["images"][0]-gt_img["images"][select_view])**2)
                elif method=="pytorch3d_rgb":
                    render_res = torch_soft_renderer(scene["meshes"][0]["model"], cameras=torch_cameras[select_view], lights=torch_lights[select_view], materials = material)
                    loss = torch.mean((render_res[0,...,:3]-gt_img_torch[select_view][...,:3])**2)
                elif method=="pytorch3d_sil_rgb":
                    render_res = torch_soft_renderer(scene["meshes"][0]["model"], cameras=torch_cameras[select_view], lights=torch_lights[select_view], materials = material)
                    loss = torch.mean((render_res[0,...,:3]-gt_img_torch[select_view][...,:3])**2)
                    loss += torch.mean((render_res[0,...,3]-gt_sil_torch[select_view][...,3])**2)
                elif method=="pytorch3d_sil":
                    render_res = torch_soft_renderer(scene["meshes"][0]["model"], cameras=torch_cameras[select_view], lights=torch_lights[select_view], materials = material)
                    loss = torch.mean((render_res[0,...,3]-gt_sil_torch[select_view][...,3])**2)
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
        # log_final(scene,render_res,testnum,method,Logger,num_views,gt_mesh)
        Logger.clean()
        Logger.show_metric()
    Logger.exit()
