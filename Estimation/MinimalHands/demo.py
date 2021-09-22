import cv2
import torch
from manopth import manolayer
from model.detnet import detnet
from utils import func, bone, AIK, smoother
import numpy as np
import matplotlib.pyplot as plt
from utils import vis
from op_pso import PSO
import open3d

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
_mano_root = 'mano/models'

# 检查模型参数是否齐全，并加载
module = detnet().to(device)
print('load model start')
check_point = torch.load('my_results/checkpoints/ckp_detnet_68.pth', map_location=device)
model_state = module.state_dict()
state = {}
for k, v in check_point.items():
    if k in model_state:
        state[k] = v
    else:
        print(k, ' is NOT in current model')
model_state.update(state)
module.load_state_dict(model_state)
print('load model finished')
pose, shape = func.initiate("zero")                 # 1x48, 1x10的全0tensor
pre_useful_bone_len = np.zeros((1, 15))             # 1x15
pose0 = torch.eye(3).repeat(1, 16, 1, 1)            # 1x16x3x3, 3x3的对角阵，扩充到16通道

mano = manolayer.ManoLayer(flat_hand_mean=True,
                           side="right",
                           mano_root=_mano_root,
                           use_pca=False,
                           root_rot_mode='rotmat',
                           joint_rot_mode='rotmat')
print('start opencv')
point_fliter = smoother.OneEuroFilter(4.0, 0.0)     # 一欧滤波器
mesh_fliter = smoother.OneEuroFilter(4.0, 0.0)      # http://www.lifl.fr/~casiez/1euro/InteractiveDemo/
shape_fliter = smoother.OneEuroFilter(4.0, 0.0)     # 可以较好的追踪动态信号，延迟较低，但对震颤的抗性较差
cap = cv2.VideoCapture(0)
print('opencv finished')
flag = 1
plt.ion()
f = plt.figure()

fliter_ax = f.add_subplot(111, projection='3d')     # 一张子图，3d格式
plt.show()
view_mat = np.array([[1.0, 0.0, 0.0],
                     [0.0, -1.0, 0],
                     [0.0, 0, -1.0]])
mesh = open3d.geometry.TriangleMesh()
hand_verts, j3d_recon = mano(pose0, shape.float())
mesh.triangles = open3d.utility.Vector3iVector(mano.th_faces)
hand_verts = hand_verts.clone().detach().cpu().numpy()[0]
mesh.vertices = open3d.utility.Vector3dVector(hand_verts)
viewer = open3d.visualization.Visualizer()
viewer.create_window(width=480, height=480, window_name='mesh')
viewer.add_geometry(mesh)
viewer.update_renderer()

print('start pose estimate')

pre_uv = None
shape_time = 0
opt_shape = None
shape_flag = True
while (cap.isOpened()):
    ret_flag, img = cap.read()
    input = np.flip(img.copy(), -1)                                 # 镜像翻转
    k = cv2.waitKey(1) & 0xFF                                       # 按q退出，防bug
    if input.shape[0] > input.shape[1]:                             # 0高1宽
        margin = (input.shape[0] - input.shape[1]) // 2             # 取图像正中间的一个正方形
        input = input[margin:-margin]
    else:
        margin = (input.shape[1] - input.shape[0]) // 2
        input = input[:, margin:-margin]
    img = input.copy()
    img = np.flip(img, -1)
    cv2.imshow("Capture_Test", img)                                 # 摄像头采集画面,720x720
    input = cv2.resize(input, (128, 128))
    input = torch.tensor(input.transpose([2, 0, 1]), dtype=torch.float, device=device)  # hwc -> chw
    input = func.normalize(input, [0.5, 0.5, 0.5], [1, 1, 1])
    result = module(input.unsqueeze(0))
    # 预处理完成，[b, c, h, w]，result为预测结果，module是DetNet
    # for key in result.keys():
    #     print(key)
    # for key, value in result.items():
    #     if key == 'xyz':
    #         print(value)
    # print(result['xyz'])
    # h_map     [1, 21, 32, 32]
    # d_map     [1, 21, 3, 32, 32]
    # l_map     [1, 21, 3, 32, 32]
    # delta     [1, 21, 3]
    # xyz       [1, 21, 3]
    # uv        [1, 21, 2]                  hmap 32x32，uv就是这里的index

    pre_joints = result['xyz'].squeeze(0)
    now_uv = result['uv'].clone().detach().cpu().numpy()[0, 0]
    now_uv = now_uv.astype(np.float)
    trans = np.zeros((1, 3))
    trans[0, 0:2] = now_uv - 16.0       # uv按16归一化，map 0-32
    trans = trans / 16.0
    new_tran = np.array([[trans[0, 1], trans[0, 0], trans[0, 2]]])
    pre_joints = pre_joints.clone().detach().cpu().numpy()

    flited_joints = point_fliter.process(pre_joints)

    #####################################
    # xyz通过手部原点和单位长建系的坐标系下，永远是相对位置，不能映射回原图
    # img = img.copy()
    # for j in flited_joints:
    #     img = cv2.circle(img, (int((j[0]+1)*img.shape[0]/2), int((j[1]+1)*img.shape[1]/2)), 5, (0,255,255), -1)
    # cv2.imshow("Test", img)
    img = img.copy()
    curr_uv = result['uv'].clone().detach().cpu().numpy()[0]
    for j in curr_uv:
        img = cv2.circle(img, (int(j[1]*img.shape[1]/32), int(j[0]*img.shape[0]/32)), 5, (0,255,255), -1)
    cv2.imshow("Test", img)

    curr_xyz = result['xyz'].clone().detach().cpu().numpy()[0]
    # print(curr_xyz[9])
    ref_z = curr_xyz[0][2]
    x1, y1 = int(curr_uv[0][1]*img.shape[1]/32), int(curr_uv[0][0]*img.shape[0]/32)
    x2, y2 = int(curr_uv[9][1]*img.shape[1]/32), int(curr_uv[9][0]*img.shape[0]/32)
    import numpy as np
    e = np.sqrt(np.square(x1-x2) + np.square(y1-y2))
    for xyz in curr_xyz:
        xyz[2] = (xyz[2] - ref_z) * e
    print(curr_xyz)
    #####################################

    fliter_ax.cla()

    filted_ax = vis.plot3d(flited_joints + new_tran, fliter_ax)
    pre_useful_bone_len = bone.caculate_length(pre_joints, label="useful")

    NGEN = 100
    popsize = 100
    low = np.zeros((1, 10)) - 3.0
    up = np.zeros((1, 10)) + 3.0
    parameters = [NGEN, popsize, low, up]
    pso = PSO(parameters, pre_useful_bone_len.reshape((1, 15)),_mano_root)
    pso.main()
    opt_shape = pso.ng_best
    opt_shape = shape_fliter.process(opt_shape)

    opt_tensor_shape = torch.tensor(opt_shape, dtype=torch.float)
    _, j3d_p0_ops = mano(pose0, opt_tensor_shape)
    template = j3d_p0_ops.cpu().numpy().squeeze(0) / 1000.0  # template, m 21*3
    ratio = np.linalg.norm(template[9] - template[0]) / np.linalg.norm(pre_joints[9] - pre_joints[0])
    j3d_pre_process = pre_joints * ratio  # template, m
    j3d_pre_process = j3d_pre_process - j3d_pre_process[0] + template[0]
    pose_R = AIK.adaptive_IK(template, j3d_pre_process)
    pose_R = torch.from_numpy(pose_R).float()
    #  reconstruction
    hand_verts, j3d_recon = mano(pose_R, opt_tensor_shape.float())
    mesh.triangles = open3d.utility.Vector3iVector(mano.th_faces)
    hand_verts = hand_verts.clone().detach().cpu().numpy()[0]
    hand_verts = mesh_fliter.process(hand_verts)
    hand_verts = np.matmul(view_mat, hand_verts.T).T
    hand_verts[:, 0] = hand_verts[:, 0] - 50
    hand_verts[:, 1] = hand_verts[:, 1] - 50
    mesh_tran = np.array([[-new_tran[0, 0], new_tran[0, 1], new_tran[0, 2]]])
    hand_verts = hand_verts - 100 * mesh_tran

    mesh.vertices = open3d.utility.Vector3dVector(hand_verts)
    mesh.paint_uniform_color([228 / 255, 178 / 255, 148 / 255])
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    viewer.update_geometry(mesh)
    viewer.poll_events()
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
