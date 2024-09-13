import torch
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRendererWithFragments,
    MeshRasterizer,
    SoftPhongShader,
    SoftPhongShader,
)
#Not recommend, please use NvDiffRastRenderer.
class PyTorch3DFullRenderer:
    def __init__(self, device, settings):
        raster_settings = RasterizationSettings(
            image_size=settings["resolution"],
            blur_radius=settings["sigma"],
            faces_per_pixel=settings["faces_per_pixel"],
            perspective_correct=settings["persp"]
        )
        #MeshRendererWithFragments provides us with barycoords.
        self.renderer  = MeshRendererWithFragments( 
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=device,
                blend_params=settings["blend_param"]
            )
        )

    def render_point(self, mesh, image_fragments):#渲染点代理的功能函数，输入是mesh
        # face_ids:list of each view's faces
        pix_to_face = image_fragments.pix_to_face.detach()[:, :, :, 0:1]
        bary_coords = image_fragments.bary_coords[:, :, :, 0:1, :]
        faces_list = mesh.faces_list()[0]
        #verts_list = image_fragments.proj_mesh.verts_list()[0][None, ...]
        msk = (pix_to_face[..., 0] != -1).detach()
        info = pix_to_face[msk].detach()
        tri_id = faces_list[info[..., 0].long()].detach()
        uvw = bary_coords[msk].detach().permute(1, 0, 2).squeeze(0)
        b,h,w = pix_to_face.shape[:-1]
        has_pos = torch.zeros((b,h,w), device=pix_to_face.device)
        has_pos[msk] = 1
        #points = verts_list[0,tri_id.long(), :]  # (B,N,3,3)
        #pos_3d = points.permute(2, 0, 1)*uvw
        #pos_3d = pos_3d.permute(1, 2, 0)
        #pos_2d = -torch.sum(pos_3d[..., :2], dim=1)
        pos_map = torch.zeros((b,h,w,2), device=pix_to_face.device)
        #pos_map[msk] = pos_2d
        return has_pos, pos_map

    def render(self, mesh, cameras, lights, materials):
        images, image_fragments = self.renderer(mesh, cameras=cameras, lights=lights, materials=materials)
        #import pdb;pdb.set_trace()
        msk, pos = self.render_point(mesh, image_fragments)
        res = {"images":images,
                "msk":msk,
                "pos": pos,#含有位置信息的pos就是xy，所以这里其实就是从mesh上的点代理映射得到rgb与xy的渲染函数
                "rasts": image_fragments
            }
        return res