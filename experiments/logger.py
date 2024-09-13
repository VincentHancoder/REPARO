import cv2
import datetime
import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.io import save_obj
from shutil import copyfile

class Loggers:
    loggers = {}
    metrics = {}
    exp_name = ""
    path = ""
    time=""
    folder_name=""
    vis=True
    def __init__(self) -> None:
        pass

    def init(self, exp_name="experiment",path="/mnt/HDD/exp_log", folder_name="",show=False, debug=False):
        self.time = f'{datetime.datetime.now():%d-%b-%y-%H-%M-%S}'#每个文件夹命名是时间的地方
        self.exp_name=exp_name
        self.prefix = path
        #self.path = os.path.join(self.prefix, self.exp_name, self.folder_name)
        self.path=folder_name
        self.show = show
        self.debug=debug
        os.makedirs(self.path,exist_ok=True)
        self.core_logger = SummaryWriter(log_dir = self.path)

    def add_image(self, name, content = None, step=0, type = "video", flip=True):
        if torch.is_tensor(content):
            content = content.detach().cpu().numpy()
        
        if flip:
            content = cv2.flip(content, 0)

        if len(content.shape) == 3:
            content = content[..., :3]
        
        content = np.clip(content,0,1)
        content = (content*255).astype(np.uint8)

        if content.shape[-1]==3:
            content = cv2.cvtColor(content, cv2.COLOR_BGR2RGB)
        
        content = cv2.resize(content,(512,512))    
        if type == "video":
            if name not in self.loggers:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                color = True
                if len(content.shape) == 2 or content.shape[2] == 1:
                    color = False
                self.loggers[name] = cv2.VideoWriter(
                    os.path.join(self.path, name.replace(" ", "_") + ".mp4"),
                    fourcc, 60.0,
                    (content.shape[1], content.shape[0]),
                    color
                )
            writer = self.loggers[name]
            writer.write(content)
        
        if self.show:
            try:
                content = cv2.resize(content,(512,512))
                cv2.imshow(name,content)
                cv2.waitKey(1)
            except:
                print("visualization failed. if you don't have a screen, don't turn show option on")
        if len(content.shape)==2:
            content = content[...,None]
        
        #self.core_logger.add_image(name, content, dataformats = "HWC", global_step = step)
    
    def save_config(self, task):
        f = open(os.path.join(self.path, "config.py"), 'w')
        f.write(str(task))
        f.close()
    
    def save_file(self, file):
        copyfile(file, os.path.join(self.path, file.split("/")[-1]))
    
    
    def save_txt(self, name, arr):
        np.savetxt(os.path.join(self.path,name),arr)
    
    def save_npy(self, name, arr):
        np.save(os.path.join(self.path,name),arr)
    
    def save_img(self, name, img, flip=True):
        img = img.astype(np.float32)
        img = img[...,:3]
        if img.shape[-1]==3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if flip:
            img = cv2.flip(img, 0)
        
        img = (np.clip(img,0,1)*255).astype(np.uint8)
        img = cv2.resize(img,(512,512),interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(self.path,name),img)

    def step(self, stop = False):
        if self.debug or stop:
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)
        return
            
    def mark(self):
        open(os.path.join(self.path, "success"), 'w')
    
    def clean(self):
        for k, v in self.loggers.items():
            v.release()
        cv2.destroyAllWindows()
        self.loggers={}
    
    def exit(self):
        print("exit")
        self.core_logger.close()
        self.clean()
        for k,v in self.metrics.items():
            np.savetxt(os.path.join(self.path,k+".txt"),np.array(v))

    def show_metric(self):
        for k,v in self.metrics.items():
            print(k,np.average(np.array(v)))
        
    def ret_metric(self):
        return self.metrics
        
    def add_scalar(self, label: str, x, y):
        self.core_logger.add_scalar(label, x, y)

    def add_metric(self, label, x):
        if label not in self.metrics:
            self.metrics[label]=[]
        self.metrics[label].append(x)

    def add_video(self, label: str, vid: torch.Tensor):
        """ The shape of vid should be (T, H, W, 3)
        """
        self.core_logger.add_video(label, vid.reshape([1, *vid.size()]))

    def add_mesh(self, label: str, vertice, color, face, step):
        """ The shape of these tensors should be (N, 3)
        """
        self.core_logger.add_mesh(
            label,
            vertice.reshape([1, *vertice.size()]) if vertice is not None else None,
            color.reshape([1, *color.size()]) if color is not None else None,
            face.reshape([1, *face.size()]) if face is not None else None,
            global_step = step
        )
    
    def save_scene(self,scene,name):
        path = os.path.join(self.path,name)
        os.makedirs(path,exist_ok=True)
        #return
        for id,meshinfo in enumerate(scene["meshes"]):
            verts,faces = meshinfo["model"].get_mesh_verts_faces(0)
            try:
                verts_uv = meshinfo["model"].textures.verts_uvs_list()[0]
                faces_uv = meshinfo["model"].textures.faces_uvs_list()[0]
                tex = meshinfo["model"].textures.maps_padded()[0]
                save_obj(os.path.join(path,str(id)+".obj"),verts,faces,verts_uvs=verts_uv,faces_uvs=faces_uv,texture_map=tex)
            except:
                print("no texture save")
                save_obj(os.path.join(path,str(id)+".obj"),verts,faces)



Logger = Loggers()