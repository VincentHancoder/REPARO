
import random,math
import xmltodict as xd
import numpy as np
import cv2
import torch
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    BlendParams,
    TexturesVertex,
    OpenGLOrthographicCameras,
    camera_position_from_spherical_angles,
    get_world_to_view_transform,
    look_at_rotation
)
from pytorch3d.io import load_objs_as_meshes,IO
from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix
from pytorch3d.utils import ico_sphere
from pytorch3d.structures import join_meshes_as_scene
from pytorch3d.loss import (
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

from ignite.metrics import PSNR,SSIM
from ignite.engine import Engine
import lpips
import glm
from core.PyTorch3DRenderer import PyTorch3DFullRenderer

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

mesh_loader = IO()

def eval_step(engine, batch):
    return batch
default_evaluator = Engine(eval_step)

psnr=PSNR(data_range=1.0)
ssim = SSIM(data_range=1.0)
psnr.attach(default_evaluator, 'psnr')
ssim.attach(default_evaluator, 'ssim')
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores

        
def get_pytorch3d_renderer(bg, faces_per_pixel, resolution, sigma=1e-4, persp=False):

    if bg == "white":
        bg_cuda = torch.ones((1, resolution, resolution, 3)).cuda()
    elif bg == "black":
        bg_cuda = torch.zeros((1, resolution, resolution, 3)).cuda()
    elif bg == "random":
        bg_cuda = torch.rand((1, resolution, resolution, 3)).cuda()
    else:
        try:
            bg = cv2.imread(bg)
            bg = cv2.cvtColor(bg,cv2.COLOR_BGR2RGB)
            bg = cv2.resize(bg, (resolution, resolution))*1.0/255.0
            bg_cuda = torch.tensor(bg).cuda().reshape(
                (1, resolution, resolution, 3)).float()
        except:
            bg_cuda = None
    if bg_cuda==None:
        blend_param=BlendParams()
    else:
        blend_param = BlendParams(background_color=bg_cuda)
    
    raster_settings = RasterizationSettings(
        image_size=resolution,
        blur_radius=0.0,
        faces_per_pixel=1,
        perspective_correct=persp
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            device=device,
            blend_params=blend_param
        )
    )

    raster_settings_silhouette = RasterizationSettings(
        image_size=resolution,
        blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
        faces_per_pixel=faces_per_pixel,
        perspective_correct=persp
    )

    renderer_silhouette = MeshRenderer(
        rasterizer=MeshRasterizer(
            raster_settings=raster_settings_silhouette
        ),
        shader=SoftSilhouetteShader()
    )

    raster_settings_soft = RasterizationSettings(
        image_size=resolution,
        blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
        faces_per_pixel=faces_per_pixel,
        perspective_correct=persp
    )

    renderer_soft = MeshRenderer(
        rasterizer=MeshRasterizer(
            raster_settings=raster_settings_soft
        ),
        shader=SoftPhongShader(
            device=device,
            blend_params=blend_param
        )
    )

    settings={"resolution":resolution,"sigma":0,"faces_per_pixel":1,"persp":persp,"blend_param":blend_param}
    torch_point_renderer = PyTorch3DFullRenderer(device=device,settings=settings)

    return renderer, renderer_silhouette, renderer_soft, torch_point_renderer #第一个render就是torch_gt_render用来得到gt_img_torch的

def config_view_light(task, device):
    view = task.get("view", None)
    width, height = task.get("resolution", [256, 256])
    if view == None:
        return None, None

    num_views = view.get("num", 10)
    center = view.get("center", [0, 0, 0])
    dist = view.get("dist", 10)
    direction = view.get("direction", "auto")
    camera_type = view.get("type", "perspective")
    fov = view.get("fov", 60.0)
    znear = view.get("znear", 0.01)
    zfar = view.get("zfar", 100.0)
    if direction == "auto":
        elev = torch.linspace(0, 360, num_views+1)[:-1]
        azim = torch.linspace(-180, 180, num_views+1)[:-1]
        center = torch.tensor(center, device=device)[None]
        R, T = look_at_view_transform(dist, elev, azim, device=device, at=center) 
        # This function returns a rotation and translation matrix to apply the ‘Look At’ transformation from world -> view coordinates [0].
        position = center + camera_position_from_spherical_angles(
            distance=dist, elevation=elev, azimuth=azim, device=device)
    elif direction == "random":
        elev = torch.rand(num_views)*360
        azim = torch.rand(num_views)*360-180
        center = torch.tensor(center, device=device)
        R, T = look_at_view_transform(dist, elev, azim, device=device, at=center[None])
        position = center+camera_position_from_spherical_angles(
            distance=dist, elevation=elev, azimuth=azim, device=device)
    elif direction == "square":
        center = torch.tensor(view.get("center"), device=device)[None]
        position = torch.tensor([[0.0,0.0,dist],[0.0,0.0,-dist],[0.0,dist,0.0],[0.0,-dist,0.0],[dist,0.0,0.0],[-dist,0.0,0.0]], device=device)+center
        position = position+torch.rand_like(position,device=device)*0.1
        R = look_at_rotation(position, center, device=device)
        T = -torch.bmm(R.transpose(1, 2), position[:, :, None])[:, :, 0]
    elif direction=="manual":
        position = torch.tensor(view.get("position"), device=device)
        center = torch.tensor(view.get("center"), device=device)
        if len(center.shape)==1:
            center=center[None]
        R = look_at_rotation(position, center, device=device) 
        # NOTE: the rotation from world coordinates -> view coordinates.
        # 为什么是这样？
        T = -torch.bmm(R.transpose(1, 2), position[:, :, None])[:, :, 0]
    else:
        print("not support view")
        exit()

    if camera_type == "perspective":
        perspective = torch.tensor(np.array(glm.perspective(
            glm.radians(fov), 1.0, znear, zfar)), device=device)
    
    else:
        max_y = view.get("max_y", 1.0)
        min_y = view.get("min_y", -1.0)
        max_x = view.get("max_x", 1.0)
        min_x = view.get("min_x", -1.0)
        scale_xyz = view.get("scale_xyz", ((1.0, 1.0, 1.0),))
        camera = OpenGLOrthographicCameras(
            znear=0.0, zfar=10, bottom=min_y, top=max_y, left=min_x, right=max_x, scale_xyz=scale_xyz)
        perspective = camera.get_projection_transform(
        ).get_matrix().transpose(1, 2).to(device)

    camera_matrix = get_world_to_view_transform(
        R=R, T=T).get_matrix().transpose(1, 2).to(device)
    vp = torch.matmul(perspective, camera_matrix).to(device)
    sensors = [{"position": position, "resolution": (
        width, height), "matrix": vp, "camera_matrix": camera_matrix, "perspective_matrix": perspective,"center":center, "init_position":position.clone(),"init_center":center.clone()}]

    R[:, :, 0] *= -1
    R[:, :, 2] *= -1
    T = -torch.bmm(R.transpose(1, 2), position[:, :, None])[:, :, 0]
    if camera_type == "perspective":
        target_cameras = [OpenGLPerspectiveCameras(
            device=device, R=R[None, i, ...], T=T[None, i, ...], znear=znear, zfar=zfar, fov=fov)for i in range(num_views)]
    else:
        target_cameras=[OpenGLOrthographicCameras(
            device=device, R=R[None, i, ...], T=T[None, i, ...], znear=znear, zfar=zfar, bottom=min_y, top=max_y, left=min_x, right=max_x) for i in range(num_views)]
    camera_pos_list = [target_camera.get_camera_center()
                       for target_camera in target_cameras]
    light_power = task.get("renderer").get("light_power", 1.0)
    lights = [PointLights(device=device, location=camera_pos_list[i].cpu().tolist(), ambient_color=[[1.0, 1.0, 1.0]], diffuse_color=[[light_power, light_power, light_power]], specular_color=[[light_power, light_power, light_power]])
              for i in range(num_views)]
    return sensors, target_cameras, lights

def config_scene(scene, task, bbox=1, device="", simple=True, optim=True):
    if not optim:
        scene["meshes"] = scene["origin_meshes"]
        scene["sensors"] = scene["origin_sensors"]
        return scene
    optimize_setting = task.get("optimize_parameter",{})
    scene["optim"] = []
    for item in optimize_setting:
        optims={}
        if item.get("envlight")==True:
            envlight_opt = torch.zeros((6,512,512,3),device=device)+0.5
            envlight_opt.requires_grad=True
            scene["envlight"]=envlight_opt
            scene["optim"].append(envlight_opt)
            continue
        if item.get("material")==True:
            envlight_opt = torch.zeros((6,512,512,3),device=device)+0.5
            envlight_opt.requires_grad=True
            scene["envlight"]=envlight_opt
            scene["optim"].append(envlight_opt)
            diffuse_color = torch.rand((1,3),device=device,requires_grad=True)
            specular_color = torch.rand((1,3),device=device,requires_grad=True)
            scene["optim"].append(diffuse_color)
            scene["optim"].append(specular_color)
            scene["material"]=Materials(device=device, ambient_color=((0.1, 0.1, 0.1),),diffuse_color=diffuse_color, specular_color=specular_color, shininess=64).to(device)
            continue
        view_info = item.get("view",None)
        if view_info!=None:# optimize view
            if view_info=="manual":
                position = torch.tensor(item.get("position"),device = device, requires_grad=True)
                center = torch.tensor(item.get("center"),device = device, requires_grad=True)
                optims.update({"camera_position":position})
                optims.update({"camera_center":center})
                scene["origin_sensors"][0].update({"optim_position":position,"optim_center":center})
                scene["optim"]=scene["optim"]+list(optims.values())
            elif view_info=="disturb":
                pos_range = item.get("pos_range",1.0)
                center_range = item.get("center_range",0.5)
                position = torch.rand(scene["origin_sensors"][0]["init_position"].shape,device=device)*pos_range+scene["origin_sensors"][0]["init_position"].clone().detach()-pos_range/2
                center = torch.rand(scene["origin_sensors"][0]["init_position"].shape,device=device)*center_range+scene["origin_sensors"][0]["init_center"].repeat(position.shape[0],1).clone().detach()-center_range/2
                position.requires_grad=True
                center.requires_grad=True
                optims.update({"camera_position":position})
                optims.update({"camera_center":center})
                scene["origin_sensors"][0].update({"optim_position":position,"optim_center":center})
            scene["optim"]+=list(optims.values())
            continue
        model_id = item.get("meshId")
        if item["init_model"]=="sphere":
            src_mesh = ico_sphere(6, device=device)
        elif item["init_model"]=="origin":
            src_mesh = scene["origin_meshes"][model_id]["model"].clone()
        else:
            if item["init_model"].endswith(".obj"):#如果mesh文件是以.obj格式结尾的就走这个
                src_mesh = load_objs_as_meshes([item["init_model"]], device=device).scale_verts(1.0/bbox)
            else:#如果mesh文件不是以obj格式结尾比如.ply就走这个分支
                from pytorch3d.io import IO
                mesh_loader = IO()
                src_mesh = mesh_loader.load_mesh(item["init_model"]).to(device)
        translations = {}
        trans = item.get("init_global_translation",[0.,0.,0.])
        sca = item.get("init_global_scale",[1.0,1.0,1.0])
        rot = item.get("init_global_rotation",[0.,0.,0.])
        rot_c = item.get("init_global_rotation_center")
        if item.get("optim_global_scale",False)==True:
            if sca=="random":
                sca = torch.rand(3).to(device)+0.5
            else:
                sca = torch.tensor(sca).to(device)
            sca.requires_grad=True
            optims.update({"scale":sca})
            translations.update({"scale":sca})
        else:
            sca=torch.tensor(sca,device=device)
            src_mesh.scale_verts_(sca)
        
        if item.get("optim_global_rotation",False)==True:
            if rot=="random":
                rot = torch.rand(3).to(device)*math.pi/2
            else:
                rot = torch.tensor(rot).to(device)
            rot.requires_grad=True
            if rot_c==None:
                rot_c = torch.mean(src_mesh.verts_list()[0],axis=0)
            else:
                rot_c = torch.tensor(rot_c, device=device)
            translations.update({"rotation":{"rad":rot,"center":rot_c}})
            optims.update({"rotation":rot})
        
        if item.get("optim_global_translation",False)==True:
            if trans=="random":
                trans=torch.rand(3).to(device)*item.get("translation_range",1.0)*2-item.get("translation_range",1.0)
            else:
                trans=torch.tensor(trans,device=device)
            trans.requires_grad=True
            optims.update({"translation":trans})
            translations.update({"translation":trans})
        else:
            trans=torch.tensor(trans,device=device)
            src_mesh.offset_verts_(trans)
        
        if item.get("optim_vertex_position",False)==True:
            verts_shape = src_mesh.verts_packed().shape
            vertex_trans = torch.full(verts_shape, 0.0, device=device, requires_grad=True)
            optims["vertex_trans"] = vertex_trans
            translations["vertex_trans"] = vertex_trans

        if item.get("optim_vertex_color",False)==True and simple:
            verts_shape = src_mesh.verts_packed().shape
            if simple:
                verts_color = torch.full([1, verts_shape[0], 3], 0.5, device=device, requires_grad=True)
            else:
                verts_color = src_mesh.texture.verts_feature
                verts_color.requires_grad=True
            optims["vertex_color"] = verts_color
            translations["vertex_color"] = verts_color

        if item.get("optim_texture_map",False)==True and simple:
            if item.get("init_texture_map")=="origin":
                tex = scene["origin_meshes"][model_id]["model"].textures._maps_padded.clone().to(device)
            elif len(item.get("init_texture_map",[]))==3:
                tex = torch.zeros_like(scene["origin_meshes"][model_id]["model"].textures._maps_padded,device=device)+torch.tensor(item.get("init_texture_map"),device=device)[None]
            else:
                tex = torch.zeros_like(scene["origin_meshes"][model_id]["model"].textures._maps_padded,device=device)+0.5
            tex.requires_grad=True
            optims["color_map"] = tex
            translations["color_map"] = tex
            
        scene["optim"]=scene["optim"]+list(optims.values())
        scene["origin_meshes"][model_id]["trans"]=translations
        scene["origin_meshes"][model_id]["init_model"]=src_mesh
    
    for mesh_idx,mesh_info in enumerate(scene["origin_meshes"]):
        if mesh_info.get("init_model")==None:
            mesh_info["init_model"]=mesh_info["model"].clone()
    
    return scene

def get_current_scene(scene, device):
    if scene["origin_sensors"][0].get("optim_position",None)!=None:
        position = scene["origin_sensors"][0].get("optim_position")
        center = scene["origin_sensors"][0].get("optim_center")
        if center==None:
            center=scene["origin_sensors"][0].get("center")
        R = look_at_rotation(position,center,device=device)
        T = -torch.bmm(R.transpose(1, 2), position[:, :, None])[:, :, 0]
        camera_matrix = get_world_to_view_transform(R=R,T=T).get_matrix().transpose(1,2)
        vp = torch.matmul(scene["origin_sensors"][0].get("perspective_matrix"), camera_matrix).to(device)
        scene["sensors"] = [{"position":position,"center":center,"camera_matrix":camera_matrix,"matrix":vp,"perspective_matrix":scene["origin_sensors"][0].get("perspective_matrix")}]
    else:
        scene["sensors"] = scene["origin_sensors"]
    meshes=[]
    for mesh_idx,mesh_info in enumerate(scene["origin_meshes"]):
        optims = mesh_info.get("trans",{})
        if mesh_info.get("init_model")!=None:
            mesh = mesh_info["init_model"].clone()
        else:
            mesh = mesh_info["model"].clone()
        for k,v in optims.items():
            if k=="scale":
                mesh.verts_list()[0]*=v
            if k=="rotation":
                rot_c = v["center"]
                rot_r = v["rad"]
                #rot_c = torch.mean(mesh.verts_list()[0],axis=0).detach()
                rot_m = euler_angles_to_matrix(rot_r,convention="XYZ")
                mesh.offset_verts_(-rot_c)
                mesh.transform_verts_(rot_m)
                mesh.offset_verts_(rot_c)

            if k=="translation":
                mesh.offset_verts_(v)         

            if k=="vertex_trans":
                mesh.offset_verts_(v)
            if k=="vertex_color":
                mesh.textures = TexturesVertex(verts_features=v) 
            
            if k=="color_map":
                mesh.textures._maps_padded = v
                mesh.textures._maps_list = None
        meshes.append(mesh)
    
    scene["meshes"]=[{"model":join_meshes_as_scene(meshes)}]
    return scene 


def update_mesh_shape_prior_losses(mesh):
    return mesh_edge_loss(mesh)+0.01*mesh_normal_consistency(mesh)+mesh_laplacian_smoothing(mesh, method="uniform")*0.1

def getDict(d, l, defaults = None, ty = float):
    return [ty(d.get(i,defaults)) for i in l]

def load_transform_matrix(transforms):
    scale = transforms.get("scale",dict())
    sx,sy,sz = getDict(scale, ['@x','@y','@z'],1)
    rotate = transforms.get("rotate",dict())
    rx,ry,rz = getDict(rotate, ['@x','@y','@z'],1)
    angle = float(rotate.get('@angle',0))
    translate = transforms.get("translate",dict())
    tx,ty,tz = getDict(translate, ['@x','@y','@z'],0)
    rm = glm.rotate(glm.radians(angle), glm.vec3(rx,ry,rz)) # creates a 4x4 rotation matrix
    sm = glm.mat4(sx,sy,sz,1)
    tm = glm.translate(glm.mat4(), glm.vec3(tx,ty,tz))
    matrix = np.array(tm @ rm @ sm)
    matrix = torch.tensor(matrix)   
    return matrix  

def load_scene(scene_file, device):
    meshes=[]
    if scene_file.endswith("obj"):
        meshes.append(load_objs_as_meshes([scene_file],device=device))
        return meshes
    elif scene_file.endswith("ply"):
        meshes.append(mesh_loader.load_mesh(scene_file).to(device))
        return meshes
    with open(scene_file,'r') as f:
        contents = f.read()
        d = xd.parse(contents,process_namespaces=True)
        scene = d.get("scene")
        shapes = scene.get("shape",[])
        if type(shapes) is not list:
            shapes = [shapes]
        for shape in shapes:
            infos = shape.get("string")
            filename = infos.get('@value')
            transforms = shape.get("transform",dict())
            transform_matrix = load_transform_matrix(transforms).to(device)
            if filename.endswith("obj"):
                mesh = load_objs_as_meshes([filename],device=device)
            elif filename.endswith("ply"):
                mesh = mesh_loader.load_mesh(filename).to(device)
            else:
                raise ValueError('mesh format (%s) is not supported, please use obj/ply'%(filename))
            mesh.transform_verts_(transform_matrix)
            mesh.scale_verts_(1.0)
            meshes.append(mesh)
    return meshes



def log_final(scene, gt_img, render_res, testnum, method, Logger, num_views, gt_mesh=None):
    with torch.no_grad():    
        Logger.save_scene(scene, name="%03d/scene_final_%s"%(testnum,method))        
        for i in range(num_views):
            Logger.save_img("%03d/img_final_%s_%d.png" %(testnum, method, i), render_res["images"][i].cpu().numpy(), flip=True)
        #Logger.add_image(name="%03d/render_%s" % (testnum, method), content=render_res["images"][0], flip=True, type="video", step=iter)
        state = default_evaluator.run([[render_res["images"].permute(0,3,1,2).clamp_(0.0,1.0), gt_img["images"].permute(0,3,1,2).clamp_(0.0,1.0)]])
        psnr_img = state.metrics['psnr']
        ssim_img = state.metrics['ssim']
        rmse_img = (torch.mean((render_res["images"]-gt_img["images"])**2)).item()**0.5
        lpips_img = torch.mean(loss_fn_alex(render_res["images"].permute(0,3,1,2).clamp_(0.0,1.0).cpu(),gt_img["images"].permute(0,3,1,2).clamp_(0.0,1.0).cpu())).item()
        if gt_mesh==None:
            mae = torch.mean(torch.abs(scene["meshes"][0]["model"].verts_list()[0]-scene["origin_meshes"][0]["init_model"].verts_list()[0])).item()
        else:
            mae = torch.mean(torch.abs(scene["meshes"][0]["model"].verts_list()[0]-gt_mesh.verts_list()[0])).item()
        #pbar.set_description("%d %s weight:%.4f psnr:%.4f ssim:%.4f rmse:%.4f lpips:%.4f" % (testnum, method, err_weight, psnr_img, ssim_img, rmse_img, lpips_img))
        #scene["optim"][2].clamp_(0.1,2.0)
        #scene["optim"][0].clamp_(0.0,1.0)
        Logger.add_scalar("metric_psnr_%s" %(method), psnr_img, testnum)
        Logger.add_scalar("metric_ssim_%s" %(method), ssim_img, testnum)
        Logger.add_scalar("metric_rmse_%s" %(method), rmse_img, testnum)
        Logger.add_scalar("metric_lpips_%s" %(method), lpips_img, testnum)
        Logger.add_scalar("metric_mae_%s" %(method), mae, testnum)
        Logger.add_metric("metric_psnr_%s"%(method), psnr_img)
        Logger.add_metric("metric_ssim_%s"%(method), ssim_img)
        Logger.add_metric("metric_rmse_%s"%(method), rmse_img)
        Logger.add_metric("metric_mae_%s"%(method), mae) 
        Logger.add_metric("metric_lpips_%s"%(method), lpips_img)   
        for param_id,param in enumerate(scene["optim"]):  
            Logger.save_npy("%03d/param_%d_%s.npy"%(testnum, param_id, method),param.detach().cpu().numpy())

def show_img(img, stop=False, title="",flip=True,writer=None):
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    img = img.astype(np.float32)
    if img.shape[-1]>=3:
        img = img[...,:3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(512,512),interpolation=cv2.INTER_NEAREST)
    if flip:
        img = cv2.flip(img, 0)
    #cv2.imshow("show_img_"+title,img)
    if writer!=None:
        img = (np.clip(img,0,1)*255).astype(np.uint8)
        writer.write(img)
    if stop:
        cv2.waitKey(0)
    else:
        cv2.waitKey(1)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True