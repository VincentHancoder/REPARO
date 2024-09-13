import torch
import numpy as np
from torch.nn import Module
from geomloss import SamplesLoss
from sklearn.neighbors import NearestNeighbors

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
        #！5d数据的自动梯度回传，这里面进行计算的是已经经过映射得到的rgb与xy值了，
        # 所以点代理的映射过程应该不在这，而是在render过程(mesh->pixel)，也就是这个映射过程有可能是用了别人写好的
        [g] = torch.autograd.grad(torch.sum(pointloss), [render_point_5d_match])
        g[...,:3]/=self.rgb_match_weight(view)
        return (render_point_5d-g.reshape(-1,h,w,5)).detach()
    #our_torch方法计算场景渲染图像与单张真值图像的梯度定义，reder_res{"images": ;"msk": ;"pos": ;"rasts"}
    def get_loss(self, render_res, gt_rgb, view):
        haspos = render_res["msk"]
        render_pos = (render_res["pos"]+1.0)/2.0
        render_rgb = render_res["images"][...,:3]#[1,512,512,4]-->[1,512,512,3] 像素梯度
        #这里的render_res其实就是每个epoch中场景渲染出来的图像，这里的操作也可以说明渲染出来的字典类型的images对应的内容也是最后通道数为4
        render_pos[haspos==False]=self.pos[view:view+1][haspos==False].clone()
        render_point_5d = torch.cat([render_rgb, render_pos], dim=-1)#这个地方的5d就是对应原文的5d-rgbxy，所以rgbxy就是把rgb与pos给连接起来
        match_point_5d = self.match_Sinkhorn(haspos, render_point_5d, gt_rgb, view)#这里的Sinkhorn就是最优传输算法，用于建模5d的真值与渲染的之间的loss
        disp = match_point_5d-render_point_5d
        loss = torch.mean(disp**2)
        return loss#最后返回的是5d数据用最优传输算法建模的loss
    
    def forward(self, gt, iteration=-1, scene=None, view=0):
        self.step=iteration

        new_match = ((self.matchings_record[view] % (self.matching_interval+1))==0)

        if new_match:
            render_res = self.renderer.render(scene, view=view, DcDt=False)
            self.rasts[view] = render_res["rasts"]
        else:
            render_res = self.renderer.render(scene, rasts_list = self.rasts[view], view=view, DcDt=False)
        
        self.matchings_record[view] += 1
        haspos = render_res["msk"]
        render_pos = (render_res["pos"]+1.0)/2.0
        render_rgb = render_res["images"]
        render_pos[haspos==False]=self.pos[view:view+1][haspos==False].clone()
        render_point_5d = torch.cat([render_rgb, render_pos], dim=-1)
        #gt_rgb=gt[view:view+1][0][...,:3].unsqueeze(0)
        #import pdb;pdb.set_trace()
        gt_rgb=gt["images"][view:view+1]#只取了gt_img的['images']
        if new_match:
            if self.matcher_type=="Sinkhorn":
                self.matchings[view] = self.match_Sinkhorn(haspos, render_point_5d, gt_rgb, view)

        match_point_5d = self.matchings[view]
        disp = match_point_5d-render_point_5d
        loss = torch.mean(disp**2)

        if self.debug:
            self.visualize_point(match_point_5d.reshape(-1,5),title="match",view=view)
        
        return loss, render_res
            

            

