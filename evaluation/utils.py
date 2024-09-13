import os
import vtk
import subprocess
import numpy as np
import trimesh
from xvfbwrapper import Xvfb


def activate_virtual_framebuffer():
    '''
    Activates a virtual (headless) framebuffer for rendering 3D
    scenes via VTK.

    Most critically, this function is useful when this code is being run
    in a Dockerized notebook, or over a server without X forwarding.

    * Requires the following packages:
      * `sudo apt-get install libgl1-mesa-dev xvfb`
    '''

    vtk.OFFSCREEN = True
    os.environ['DISPLAY']=':99.0'

    commands = ['Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &',
                'sleep 3',
                'exec "$@"']

    for command in commands:
        subprocess.call(command,shell=True)


def prepare_mesh(mesh_pred_path, mesh_tgt_path, meshobj_save_dir):
    # load mesh
    mesh_pred = trimesh.load_mesh(mesh_pred_path)
    mesh_tgt = trimesh.load_mesh(mesh_tgt_path)

    # scale 这一部份的代码就是(1)将3D模型进行中心化(2)对模型进行尺度归一化
    #尺度归一化已在gt摆放时做好，这一部份可以不用执行归一化操作
    # pred
    # rescale = max(mesh_pred.extents)
    tform = [
        -(mesh_pred.bounds[1][i] + mesh_pred.bounds[0][i])/2.
        for i in range(3)
    ]
    matrix = np.eye(4)
    matrix[:3, 3] = tform
    mesh_pred.apply_transform(matrix)
    matrix = np.eye(4)
    # matrix[:3, :3] /= rescale
    mesh_pred.apply_transform(matrix)
    
    # tgt
    # pdb.set_trace()
    # rescale = max(mesh_tgt.extents)
    tform = [
        -(mesh_tgt.bounds[1][i] + mesh_tgt.bounds[0][i])/2.
        for i in range(3)
    ]
    matrix = np.eye(4)
    matrix[:3, 3] = tform
    mesh_tgt.apply_transform(matrix)
    matrix = np.eye(4)
    # matrix[:3, :3] /= rescale
    mesh_tgt.apply_transform(matrix)

    
    # import numpy as np

    # 假设mesh_pred和mesh_tgt已经被正确加载

    # # 计算两个mesh的最大延展，并决定共同的缩放因子
    # max_extent_pred = max(mesh_pred.extents) / 2.
    # max_extent_tgt = max(mesh_tgt.extents) / 2.
    # common_rescale = max(max_extent_pred, max_extent_tgt)

    # # 对mesh_pred进行尺度归一化
    # tform_pred = [-(mesh_pred.bounds[1][i] + mesh_pred.bounds[0][i])/2. for i in range(3)]
    # matrix_pred = np.eye(4)
    # matrix_pred[:3, 3] = tform_pred
    # mesh_pred.apply_transform(matrix_pred)
    # matrix_pred = np.eye(4)
    # matrix_pred[:3, :3] /= (max_extent_pred / common_rescale)
    # mesh_pred.apply_transform(matrix_pred)

    # # 对mesh_tgt进行尺度归一化
    # tform_tgt = [-(mesh_tgt.bounds[1][i] + mesh_tgt.bounds[0][i])/2. for i in range(3)]
    # matrix_tgt = np.eye(4)
    # matrix_tgt[:3, 3] = tform_tgt
    # mesh_tgt.apply_transform(matrix_tgt)
    # matrix_tgt = np.eye(4)
    # matrix_tgt[:3, :3] /= (max_extent_tgt / common_rescale)
    # mesh_tgt.apply_transform(matrix_tgt)

    # viz
    meshobj_pred_path = f'{meshobj_save_dir}/mesh_pred.obj'  # 这是文件路径
    meshobj_tgt_path = f'{meshobj_save_dir}/mesh_tgt.obj'  # 这是文件路径

    # 为 mesh_pred 和 mesh_tgt 的父目录创建目录
    # 注意，我们只为文件的目录调用 os.makedirs，而不是文件路径
    if not os.path.exists(meshobj_save_dir):
        os.makedirs(meshobj_save_dir)

    # 现在可以安全地导出 mesh_pred 和 mesh_tgt 了，因为它们的父目录已经存在
    e = mesh_pred.export(meshobj_pred_path)
    e = mesh_tgt.export(meshobj_tgt_path)


def render_mesh(obj_file_path, texture_file_path, output_folder):
    
    # 从文件加载 OBJ 模型
    # obj_file_path = 'C:/Users/86136/Desktop/GS_mesh/mesh/mesh.obj'
    reader = vtk.vtkOBJReader()
    reader.SetFileName(obj_file_path)

    # 创建一个纹理映射器
    texture_mapper = vtk.vtkPolyDataMapper()
    texture_mapper.SetInputConnection(reader.GetOutputPort())

    # 创建一个纹理
    texture = vtk.vtkTexture()
    # texture_file_path = 'C:/Users/86136/Desktop/GS_mesh/mesh/albedo.png'

    texture_reader = vtk.vtkPNGReader()
    texture_reader.SetFileName(texture_file_path)
    texture.SetInputConnection(texture_reader.GetOutputPort())

    # 创建一个带有纹理的 Actor
    actor = vtk.vtkActor()
    actor.SetMapper(texture_mapper)
    actor.SetTexture(texture)

    # 创建渲染器、渲染窗口、交互工具
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)

    # 创建一个通用渲染窗口交互工具
    render_window_interactor = vtk.vtkGenericRenderWindowInteractor()

    # 设置渲染窗口的大小
    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.SetSize(800, 800)

    # 设置保存图片的渲染窗口
    render_window_offscreen = vtk.vtkRenderWindow()
    render_window_offscreen.SetOffScreenRendering(1)
    render_window_offscreen.SetSize(800, 800)

    render_window_interactor.SetRenderWindow(render_window)

    # 将渲染器添加到窗口
    render_window.AddRenderer(renderer)

    center = reader.GetOutput().GetCenter()
    print(center)


    elevations = [0, 0, -30, -15, -10, -5, -5, 0, 0, 0, 0, 5, 5, 10, 15, 30, 0, 0]

    for i, elevation in enumerate(elevations):
        azimuth = i * 360 / len(elevations)  # 在360度中均匀变化的Azimuth

        camera = renderer.GetActiveCamera()
        camera.SetPosition(center[0], center[1], center[2] + 8)
        camera.SetFocalPoint(center[0], center[1], center[2])
        camera.Azimuth(azimuth)
        camera.Elevation(elevation)

        # 渲染并保存图片
        render_window.Render()

        # 使用 Offscreen 渲染
        render_window_offscreen.AddRenderer(renderer)
        render_window_offscreen.Render()

        image_filter = vtk.vtkWindowToImageFilter()
        image_filter.SetInput(render_window_offscreen)
        image_filter.Update()

        writer = vtk.vtkPNGWriter()
        output_image_path = f'{output_folder}/view_{i}.png'
        writer.SetFileName(output_image_path)
        writer.SetInputConnection(image_filter.GetOutputPort())
        writer.Write()

        print(f'Image saved to {output_image_path}')
    

def get_view_images(obj_file_path, texture_file_path, output_folder):
    assert check_file_exists(obj_file_path) and check_file_exists(texture_file_path)
    check_dir(output_folder)
    vdisplay = Xvfb(width=1920, height=1080)
    vdisplay.start()
    render_mesh(obj_file_path, texture_file_path, output_folder)
    vdisplay.stop()


def check_file_exists(file):
    if os.path.exists(file):
        return True
    else:
        return False
    
def check_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)