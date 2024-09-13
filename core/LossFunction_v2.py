import torch
import numpy as np
from torch.nn import Module
from geomloss import SamplesLoss
from sklearn.neighbors import NearestNeighbors
import clip

class PointLossFunction_ablation(Module):
    def __init__(self, resolution, renderer, device, settings, debug, num_views, logger):
        super().__init__()
        self.num_views = num_views
        
        self.match_weight = settings.get("matching_weight",1.0)
        self.matchings_record=[0 for i in range(num_views)]
        self.matchings = [[] for i in range(num_views)]
        self.rasts = [[] for i in range(num_views)]
        self.rgb_weight = [self.match_weight for i in range(num_views)]
        self.matching_interval = settings.get("matching_interval",0)
        self.renderer = renderer
        self.device = device
        self.resolution = resolution[0]
        self.debug=debug
        self.logger = logger
        self.step = -1
        #Matcher setting
        self.matcher_type=settings.get("matcher","Sinkhorn")
        self.matcher = None
        self.loss = SamplesLoss("sinkhorn", blur=0.01)

        #normal image grid, used for pixel position completion
        x = torch.linspace(0, 1, self.resolution)
        y = torch.linspace(0, 1, self.resolution)
        pos = torch.meshgrid(x, y)
        self.pos = torch.cat([pos[1][..., None], pos[0][..., None]], dim=2).to(self.device)[None,...].repeat(num_views,1,1,1)
        self.pos_np = self.pos[0].clone().cpu().numpy().reshape(-1,2)

        # param
        self.enable_pixel_matching = settings.get('enable_pixel_matching', True)
        self.enable_xy = settings.get("enable_xy", True)

    
    def visualize_point(self, res, title, view):#(N,5) (r,g,b,x,y)
        res = res.detach().cpu().numpy()
        X = res[...,3:]
        #need install sklearn
        nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(self.pos_np)
        distances = np.exp(-distances*self.resolution)
        img = np.sum(res[indices,:3]*distances[...,None],axis = 1)
        img = img/np.sum(distances,axis = 1)[...,None]
        img = img.reshape(self.resolution, self.resolution, 3)
        self.logger.add_image(title+"_"+str(view), img, self.step)

    #unused currently
    def rgb_match_weight(self, view=0):
        return self.rgb_weight[view]

    def match_Sinkhorn(self, haspos, render_point_5d, gt_rgb, view):
        h,w = render_point_5d.shape[1:3]
        target_point_5d = torch.zeros((haspos.shape[0], h, w, 5), device=self.device)
        target_point_5d[..., :3] = torch.clamp(gt_rgb,0,1)
        target_point_5d[..., 3:] = render_point_5d[...,3:].clone().detach()
        if not self.enable_pixel_matching:
            return target_point_5d
        target_point_5d = target_point_5d.reshape(-1, h*w, 5)
        render_point_5d_match = render_point_5d.clone().reshape(-1,h*w,5)
        render_point_5d_match.clamp_(0.0,1.0)
        render_point_5d_match[...,:3] *= self.rgb_match_weight(view)
        target_point_5d[...,:3] = target_point_5d[...,:3]*self.rgb_match_weight(view)
        pointloss = self.loss(render_point_5d_match, target_point_5d)*self.resolution*self.resolution
        [g] = torch.autograd.grad(torch.sum(pointloss), [render_point_5d_match]) 
        g[...,:3]/=self.rgb_match_weight(view)
        return (render_point_5d-g.reshape(-1,h,w,5)).detach()
    
    def get_loss(self, render_res, gt_rgb, view):
        haspos = render_res["msk"]
        render_pos = (render_res["pos"]+1.0)/2.0
        render_rgb = render_res["images"][...,:3]#[1,512,512,4]-->[1,512,512,3]
        render_pos[haspos==False]=self.pos[view:view+1][haspos==False].clone()
        render_point_5d = torch.cat([render_rgb, render_pos], dim=-1)
        match_point_5d = self.match_Sinkhorn(haspos, render_point_5d, gt_rgb, view)
        if not self.enable_xy:
            disp = match_point_5d[..., :3] - render_point_5d[...,:3]
        else:
            disp = match_point_5d-render_point_5d
        loss = torch.mean(disp**2)
        return loss
    
    def forward(self, gt, iteration=-1, scene=None, view=0):
        self.step=iteration

        new_match = ((self.matchings_record[view] % (self.matching_interval+1))==0)

        if new_match:
            # import pdb;pdb.set_trace()
            render_res = self.renderer.render(scene, view=view, DcDt=False)#False
            self.rasts[view] = render_res["rasts"]
        else:
            # import pdb;pdb.set_trace()
            render_res = self.renderer.render(scene, rasts_list = self.rasts[view], view=view, DcDt=False)#False
        
        #import pdb;pdb.set_trace()
        self.matchings_record[view] += 1
        haspos = render_res["msk"]
        render_pos = (render_res["pos"]+1.0)/2.0
        render_rgb = render_res["images"]
        render_pos[haspos==False]=self.pos[view:view+1][haspos==False].clone()
        render_point_5d = torch.cat([render_rgb, render_pos], dim=-1)
        gt_rgb=gt[view:view+1][0][...,:3].unsqueeze(0) # shape=(1,256,256,3)
        if new_match:
            if self.matcher_type=="Sinkhorn":
                self.matchings[view] = self.match_Sinkhorn(haspos, render_point_5d, gt_rgb, view)

        match_point_5d = self.matchings[view]
        disp = match_point_5d-render_point_5d
        loss = torch.mean(disp**2)

        if self.debug:
            self.visualize_point(match_point_5d.reshape(-1,5),title="match",view=view)
        
        return loss, render_res

class PointLossFunction(Module):
    def __init__(self, resolution, renderer, device, settings, debug, num_views, logger):
        super().__init__()
        self.num_views = num_views
        
        self.match_weight = settings.get("matching_weight",1.0)
        self.matchings_record=[0 for i in range(num_views)]
        self.matchings = [[] for i in range(num_views)]
        self.rasts = [[] for i in range(num_views)]
        self.rgb_weight = [self.match_weight for i in range(num_views)]
        self.matching_interval = settings.get("matching_interval",0)
        self.renderer = renderer
        self.device = device
        self.resolution = resolution[0]
        self.debug=debug
        self.logger = logger
        self.step = -1
        #Matcher setting
        self.matcher_type=settings.get("matcher","Sinkhorn")
        self.matcher = None
        self.loss = SamplesLoss("sinkhorn", blur=0.01)

        #normal image grid, used for pixel position completion
        x = torch.linspace(0, 1, self.resolution)
        y = torch.linspace(0, 1, self.resolution)
        pos = torch.meshgrid(x, y)
        self.pos = torch.cat([pos[1][..., None], pos[0][..., None]], dim=2).to(self.device)[None,...].repeat(num_views,1,1,1)
        self.pos_np = self.pos[0].clone().cpu().numpy().reshape(-1,2)
    
    def visualize_point(self, res, title, view):#(N,5) (r,g,b,x,y)
        res = res.detach().cpu().numpy()
        X = res[...,3:]
        #need install sklearn
        nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(self.pos_np)
        distances = np.exp(-distances*self.resolution)
        img = np.sum(res[indices,:3]*distances[...,None],axis = 1)
        img = img/np.sum(distances,axis = 1)[...,None]
        img = img.reshape(self.resolution, self.resolution, 3)
        self.logger.add_image(title+"_"+str(view), img, self.step)

    #unused currently
    def rgb_match_weight(self, view=0):
        return self.rgb_weight[view]

    def match_Sinkhorn(self, haspos, render_point_5d, gt_rgb, view):
        h,w = render_point_5d.shape[1:3]
        target_point_5d = torch.zeros((haspos.shape[0], h, w, 5), device=self.device)
        target_point_5d[..., :3] = torch.clamp(gt_rgb,0,1)
        target_point_5d[..., 3:] = render_point_5d[...,3:].clone().detach()
        target_point_5d = target_point_5d.reshape(-1, h*w, 5)
        render_point_5d_match = render_point_5d.clone().reshape(-1,h*w,5)
        render_point_5d_match.clamp_(0.0,1.0)
        render_point_5d_match[...,:3] *= self.rgb_match_weight(view)
        target_point_5d[...,:3] = target_point_5d[...,:3]*self.rgb_match_weight(view)
        pointloss = self.loss(render_point_5d_match, target_point_5d)*self.resolution*self.resolution
        [g] = torch.autograd.grad(torch.sum(pointloss), [render_point_5d_match])
        g[...,:3]/=self.rgb_match_weight(view)
        return (render_point_5d-g.reshape(-1,h,w,5)).detach()
    

    def get_loss(self, render_res, gt_rgb, view):
        haspos = render_res["msk"]
        render_pos = (render_res["pos"]+1.0)/2.0
        render_rgb = render_res["images"][...,:3]#[1,512,512,4]-->[1,512,512,3]
        render_pos[haspos==False]=self.pos[view:view+1][haspos==False].clone()
        render_point_5d = torch.cat([render_rgb, render_pos], dim=-1)
        match_point_5d = self.match_Sinkhorn(haspos, render_point_5d, gt_rgb, view)
        disp = match_point_5d-render_point_5d
        loss = torch.mean(disp**2)
        return loss
    
    def forward(self, gt, iteration=-1, scene=None, view=0):
        self.step=iteration

        new_match = ((self.matchings_record[view] % (self.matching_interval+1))==0)

        if new_match:
            # import pdb;pdb.set_trace()
            render_res = self.renderer.render(scene, view=view, DcDt=False)#False
            self.rasts[view] = render_res["rasts"]
        else:
            # import pdb;pdb.set_trace()
            render_res = self.renderer.render(scene, rasts_list = self.rasts[view], view=view, DcDt=False)#False
        
        #import pdb;pdb.set_trace()
        self.matchings_record[view] += 1
        haspos = render_res["msk"]
        render_pos = (render_res["pos"]+1.0)/2.0
        render_rgb = render_res["images"]
        render_pos[haspos==False]=self.pos[view:view+1][haspos==False].clone()
        render_point_5d = torch.cat([render_rgb, render_pos], dim=-1)
        gt_rgb=gt[view:view+1][0][...,:3].unsqueeze(0) # shape=(1,256,256,3)
        #import pdb;pdb.set_trace()
        if new_match:
            if self.matcher_type=="Sinkhorn":
                self.matchings[view] = self.match_Sinkhorn(haspos, render_point_5d, gt_rgb, view)

        match_point_5d = self.matchings[view]
        disp = match_point_5d-render_point_5d
        loss = torch.mean(disp**2)

        if self.debug:
            self.visualize_point(match_point_5d.reshape(-1,5),title="match",view=view)
        
        return loss, render_res

class PointLossFunction6D(Module):
    def __init__(self, resolution, renderer, device, settings, debug, num_views, logger, depth_model_path="facebook/dpt-dinov2-base-kitti"):
        super().__init__()
        self.num_views = num_views
        
        self.match_weight = settings.get("matching_weight",1.0)
        self.matchings_record=[0 for i in range(num_views)]
        self.matchings = [[] for i in range(num_views)]
        self.rasts = [[] for i in range(num_views)]
        self.rgb_weight = [self.match_weight for i in range(num_views)]
        self.matching_interval = settings.get("matching_interval",0)
        self.renderer = renderer
        self.device = device
        self.resolution = resolution[0]
        self.debug=debug
        self.logger = logger
        self.step = -1
        #Matcher setting
        self.matcher_type=settings.get("matcher","Sinkhorn")
        self.matcher = None
        self.loss = SamplesLoss("sinkhorn", blur=0.01)

        #normal image grid, used for pixel position completion
        x = torch.linspace(0, 1, self.resolution)
        y = torch.linspace(0, 1, self.resolution)
        pos = torch.meshgrid(x, y)
        self.pos = torch.cat([pos[1][..., None], pos[0][..., None]], dim=2).to(self.device)[None,...].repeat(num_views,1,1,1)
        self.pos_np = self.pos[0].clone().cpu().numpy().reshape(-1,2)

        # depth model
        self.depth_model = DPTForDepthEstimation.from_pretrained(depth_model_path).to(self.device)
        self.preprocess_depth = AutoImageProcessor.from_pretrained(depth_model_path)
        self._freeze_model(self.depth_model)
        self.depth_model.to(self.device)
        self.transform_depth = Compose([
            lambda x:255.0 * x[:3], 
            Normalize(mean=(123.675, 116.28, 103.53),
                        std=(58.395, 57.12, 57.375))])
        self.depth_tgt = None
        
    def get_depth_map(self, image):
        image = image.permute(0, 3, 1, 2).contiguous()
        padded_image = F.pad(image, (5,5,5,5), mode='constant', value=0) # padding to fit with the size_divisor=14
        depth_map = self.depth_model(pixel_values=padded_image).predicted_depth
        depth_map = F.interpolate(depth_map.unsqueeze(1), size=image.shape[2:], mode='bicubic', align_corners=False)
        # normalize
        min_value, max_value = depth_map.min(), depth_map.max()
        normalized_depth_map = (depth_map - min_value) / (max_value - min_value)
        return normalized_depth_map.permute(0, 2, 3, 1) # (B, 1, H, W) -> (B, H, W, 1)
    
    def _freeze_model(self, model):
        # freeze_clip_model
        for param in model.parameters():
            param.requires_grad = False

    def visualize_point(self, res, title, view):#(N,6) (r,g,b,d,x,y)
        res = res.detach().cpu().numpy()
        X = res[...,4:]
        #need install sklearn
        nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(self.pos_np)
        distances = np.exp(-distances*self.resolution)
        img = np.sum(res[indices,:3]*distances[...,None],axis = 1)
        img = img/np.sum(distances,axis = 1)[...,None]
        img = img.reshape(self.resolution, self.resolution, 3)
        self.logger.add_image(title+"_"+str(view), img, self.step)

    #unused currently
    def rgb_match_weight(self, view=0):
        return self.rgb_weight[view]

    def match_Sinkhorn(self, haspos, render_point, gt_rgbd, view):
        h, w = render_point.shape[1:3]
        target_point = torch.zeros((haspos.shape[0], h, w, render_point.shape[-1]), device=self.device)
        gt_size = gt_rgbd.size(-1)
        target_point[..., :gt_size] = torch.clamp(gt_rgbd, 0.0, 1.0)
        target_point[..., gt_size:] = render_point[..., gt_size:].clone().detach()
        target_point = target_point.reshape(-1, h*w, render_point.size(-1))
        render_point_match = render_point.clone().reshape(-1,h*w,render_point.size(-1))
        render_point_match = torch.clamp(render_point_match, 0.0, 1.0)
        render_point_match[...,:gt_size] *= self.rgb_match_weight(view)
        target_point[...,:gt_size] = target_point[...,:gt_size]*self.rgb_match_weight(view)
        pointloss = self.loss(render_point_match, target_point)*self.resolution*self.resolution
        [g] = torch.autograd.grad(torch.sum(pointloss), [render_point_match]) 
        g[..., :gt_size]/=self.rgb_match_weight(view)
        return (render_point-g.reshape(-1,h,w,render_point.size(-1))).detach()
    
    def get_loss(self, render_res, gt_rgb, view):
        haspos = render_res["msk"]
        render_pos = (render_res["pos"]+1.0)/2.0
        render_rgb = render_res["images"][...,:3]#[1,512,512,4]-->[1,512,512,3]
        render_pos[haspos==False]=self.pos[view:view+1][haspos==False].clone()
        gt_rgb = gt_rgb[None]
        if self.depth_tgt is None:
            self.depth_tgt = self.get_depth_map(image=gt_rgb)
        render_depth = self.get_depth_map(image=render_rgb)# shape = (B, H, W, 1)
        render_point_6d = torch.cat([render_rgb, render_depth, render_pos], dim=-1) # shape=(B,H,W,6)
        gt_rgbd = torch.cat([gt_rgb, self.depth_tgt], dim=-1)
        match_point_6d = self.match_Sinkhorn(haspos, render_point_6d, gt_rgbd, view)
        disp = match_point_6d-render_point_6d
        loss = torch.mean(disp**2)
        return loss
    
    def forward(self, gt, iteration=-1, scene=None, view=0):
        self.step=iteration

        new_match = ((self.matchings_record[view] % (self.matching_interval+1))==0)

        if new_match:
            # import pdb;pdb.set_trace()
            render_res = self.renderer.render(scene, view=view, DcDt=False)#False
            self.rasts[view] = render_res["rasts"]
        else:
            # import pdb;pdb.set_trace()
            render_res = self.renderer.render(scene, rasts_list = self.rasts[view], view=view, DcDt=False)#False
        
        #import pdb;pdb.set_trace()
        self.matchings_record[view] += 1
        haspos = render_res["msk"]
        render_pos = (render_res["pos"]+1.0)/2.0
        render_rgb = render_res["images"]
        render_pos[haspos==False]=self.pos[view:view+1][haspos==False].clone()
        
        gt_rgb=gt[view:view+1][0][...,:3].unsqueeze(0) # shape=(1,256,256,3)

        ### get depth map
        # gt_rgb, shape=(B, H, W, C)
        # render_rgb, shape=(B, H, W, C)
        if self.depth_tgt is None:
            self.depth_tgt = self.get_depth_map(image=gt_rgb)
        render_depth = self.get_depth_map(image=render_rgb)# shape = (B, H, W, 1)
        render_point_6d = torch.cat([render_rgb, render_depth, render_pos], dim=-1) # shape=(B,H,W,6)
        gt_rgbd = torch.cat([gt_rgb, self.depth_tgt], dim=-1)

        if new_match:
            if self.matcher_type=="Sinkhorn":
                self.matchings[view] = self.match_Sinkhorn(haspos, render_point_6d, gt_rgbd, view)

        match_point_6d = self.matchings[view]
        disp = match_point_6d-render_point_6d
        loss = torch.mean(disp**2)

        if self.debug:
            self.visualize_point(match_point_6d.reshape(-1,6),title="match",view=view)
        
        return loss, render_res



from torchvision.transforms import Compose, Normalize 
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel, AutoImageProcessor, AutoModel, DPTForDepthEstimation

class SemanticLossFunction():
    def __init__(self, args, clip_model_path="openai/clip-vit-base-patch32", dino_model_path="facebook/dinov2-base", device="cpu"):
        self.args = args
        self.device = device
        self.enable_clip_loss = args["loss"].get("enable_clip_loss", False)
        self.enable_dino_loss = args["loss"].get("enable_dino_loss", False)
        if self.enable_clip_loss:
            self.clip_vision_model = CLIPVisionModel.from_pretrained(pretrained_model_name_or_path=clip_model_path)
            # self.preprocess_clip = CLIPProcessor.from_pretrained(pretrained_model_name_or_path="openai/clip-vit-base-patch32")
            # self.clip_model, self.preprocess_clip = clip.load(clip_model_path, device=self.device)
            self._freeze_model(self.clip_vision_model)
            self.clip_vision_model.to(self.device)
            # preprocess for tensor
            # self.transform_clip = Compose([Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        if self.enable_dino_loss:
            self.dino_model = AutoModel.from_pretrained(dino_model_path)
            # self.preprocess_dino = AutoImageProcessor.from_pretrained(dino_model_path)
            self._freeze_model(self.dino_model)
            self.dino_model.to(self.device)
            # self.transform_dino = Compose([Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) 
            # dino is the same as imagenet, ref: https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/data/transforms.py#L46

        # loss weight
        self.clip_loss_weight = args["loss"].get("clip_loss_weight", 0.0)
        self.dino_loss_weight = args["loss"].get("dino_loss_weight", 0.0)

        # warm_up
        self.warmup_iter = args["loss"].get("warmup_iter", 1)

        # initialize target
        self.feat_ref_clip = None
        self.feat_ref_dino = None
    
    def _freeze_model(self, model):
        # freeze_clip_model
        for param in model.parameters():
            param.requires_grad = False

    def clip_transform(self, image):
        """
        transforms.ToTensor() 会将 PIL.Image 转换到 [0, 1] 
        """
        image = image.permute(0, 3, 1, 2).contiguous()
        image = F.interpolate(image, size=(224, 224), mode='bilinear')
        return Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(image) # the same as the clip model

    def dino_transform(self, image):
        """
        transforms.ToTensor() 会将 PIL.Image 转换到 [0, 1] 
        """
        image = image.permute(0, 3, 1, 2).contiguous()
        image = F.interpolate(image, size=(224, 224), mode='bilinear')
        return Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image) # the same as the dino model

    def get_clip_loss(self, render_image, ref_image):
        if self.feat_ref_clip is None:
            ref_image = self.clip_transform(ref_image[None])
            self.feat_ref_clip = self.clip_vision_model(ref_image, output_hidden_states=True).last_hidden_state
        render_image = self.clip_transform(render_image)
        out_render = self.clip_vision_model(pixel_values=render_image, output_hidden_states=True)
        feat_render = out_render.last_hidden_state
        if self.args["loss"].get("feat_loss_mse", False):
            loss = F.mse_loss(feat_render, self.feat_ref_clip.detach())
        else:
            loss = 1.0 - F.cosine_similarity(feat_render, self.feat_ref_clip, dim=-1)
        return torch.mean(loss)

    def get_dino_loss(self, render_image, ref_image):
        if self.feat_ref_dino is None:
            ref_image = self.dino_transform(ref_image[None])
            self.feat_ref_dino = self.dino_model(ref_image, output_hidden_states=True).last_hidden_state
        render_image = self.dino_transform(render_image)
        out_render = self.dino_model(pixel_values=render_image, output_hidden_states=True)
        feat_render = out_render.last_hidden_state
        if self.args["loss"].get("feat_loss_mse", False):
            loss = F.mse_loss(feat_render, self.feat_ref_dino.detach())
        else:
            loss = 1.0 - F.cosine_similarity(feat_render, self.feat_ref_dino, dim=-1)
        return torch.mean(loss)

    def __call__(self, render_image, ref_image, iter):
        """
        Args:
            render_image (Tensor): the image rendered from 3D asserts, shape=(B, H, W, C)
            ref_image (tensor): shape=(H, W, C) 
        NOTE: render_image and ref_image should have the same transform operation
        """
        loss = {}
        # get loss
        if self.enable_clip_loss:
            clip_loss_weight = min(iter / self.warmup_iter, 1.0) * self.clip_loss_weight
            loss["clip_loss"] = self.get_clip_loss(render_image, ref_image) * clip_loss_weight
        if self.enable_dino_loss:
            dino_loss_weight = min(iter / self.warmup_iter, 1.0) * self.dino_loss_weight
            loss["dino_loss"] = self.get_dino_loss(render_image, ref_image) * dino_loss_weight
        return loss

class DepthLossFunction():
    def __init__(self, args, model_path="facebook/dpt-dinov2-base-kitti", device="cpu"):
        self.device = device
        self.enable_depth_loss = args["loss"].get("enable_depth_loss", False)
        if self.enable_depth_loss:
            self.depth_model = DPTForDepthEstimation.from_pretrained(model_path).to(self.device)
            self.preprocess_depth = AutoImageProcessor.from_pretrained(model_path)
            self._freeze_model(self.depth_model)
            self.depth_model.to(self.device)
            self.transform_depth = Compose([
                lambda x:255.0 * x[:3], 
                Normalize(mean=(123.675, 116.28, 103.53),
                          std=(58.395, 57.12, 57.375))])
        # loss weight
        self.depth_loss_weight = args["loss"].get("depth_loss_weight", 0.0)
        # initialize target
        self.depth_ref = None

    def _freeze_model(self, model):
        # freeze_clip_model
        for param in model.parameters():
            param.requires_grad = False

    def depth_transform(self, image):
        image = image.permute(0, 3, 1, 2).contiguous()
        image = F.interpolate(image, size=(384, 384), mode='bilinear') # dpt will do a center padding, padding size is 4
        image = F.pad(image, (4,4,4,4), mode='constant', value=0)
        return self.transform_depth(image)
    
    def get_depth_loss(self, render_image, ref_image):
        if self.depth_ref is None:
            ref_image = self.depth_transform(ref_image[None])
            self.depth_ref = self.depth_model(pixel_values=ref_image).predicted_depth
        render_image = self.depth_transform(render_image)
        depth_render = self.depth_model(pixel_values=render_image).predicted_depth
        loss = torch.abs(depth_render-self.depth_ref)
        return torch.mean(loss)
    
    def __call__(self, render_image, ref_image, iter):
        loss = {}
        if self.enable_depth_loss:
            loss["depth_loss"] = self.get_depth_loss(render_image, ref_image)
        return loss
