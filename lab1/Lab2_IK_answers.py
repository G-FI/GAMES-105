import numpy as np
from scipy.spatial.transform import Rotation as R


def build_tree(parent):
    children = [np.array([], dtype=np.int32) for p in parent]
    for i in range(0, len(parent)):
        if parent[i] == -1:
            continue
        children[parent[i]] = np.concatenate((children[parent[i]], [i]))
    return children


#input: 原来的orientation，新施加的rotation， children数组，parent_idx，表示这次更新的parent的索引
#输出，更新了子节点之后的orientation
def update_child_orientation(joint_orientations, r, children, parent_idx):

    if len(children[parent_idx]) == 0:
        return joint_orientations

    for child_idx in children[parent_idx]:
        joint_orientations[child_idx] = (
             r * R.from_quat(joint_orientations[child_idx])).as_quat()
        #更新当前子节点的子节点
        joint_orientations = update_child_orientation( joint_orientations, r,
                                                      children, child_idx)
        
    return joint_orientations


    

def update_child_postion(meta_data, joint_positions, joint_orientations,
                         children, parent_idx, below_root):
    if len(children[parent_idx]) == 0:
        return joint_positions

    for child_idx in children[parent_idx]:
        if below_root:
            #parent节点于Root_joint之下，或者就是Root_jonit节点，其之下的节点都使用正常方式更新
            initial_offset = meta_data.joint_initial_position[child_idx] - meta_data.joint_initial_position[parent_idx]
            joint_positions[child_idx] = joint_positions[parent_idx] + R.from_quat(joint_orientations[parent_idx]).apply(initial_offset)
            joint_positions =  update_child_postion(meta_data, joint_positions, joint_orientations, children, child_idx, True)
        else:
            #这一层的节点是位于Root_joint之上，或就是Root_joint,使用第二种方式更新
            initial_offset = meta_data.joint_initial_position[child_idx] - meta_data.joint_initial_position[parent_idx]
            joint_positions[child_idx] = joint_positions[parent_idx] + R.from_quat(joint_orientations[child_idx]).apply(initial_offset)

            if child_idx == 0:
                #此节点为Root_joint节点，其子节点位置按照正常方式更新
                joint_positions =  update_child_postion(meta_data, joint_positions, joint_orientations, children, child_idx, True)
            else:
                #还未到达Root_joint节点
                joint_positions = update_child_postion(meta_data, joint_positions, joint_orientations, children, child_idx, False)



            

    return joint_positions


def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations,
                             target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    '''CCD方法
        path 中是路径索引，通过索引获取,从path的最后一个θ开始，进行循环
        直接修改position和orientation
    ''' 
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    parent = meta_data.joint_parent

    if path2[-1] == 0:  #过Root_joint节点
        parent[path2[0]] = -1  #更换新的根节点
        for i in range(1, len(path2)):
            parent[path2[i]] = path[i - 1]

    children = build_tree(parent)
    end_pos = joint_positions[path[-1]]



    error_bound = 0.001
    max_iteration = 10
    below_root = True

    for k in range(max_iteration):
        #迭代开始的节点都是位于Root_joint之下的
        below_root = True
        for i in range(len(path) - 2, 0, -1):  #len(path)-2表示端点的父节点
            error = np.dot(target_pose - end_pos, target_pose - end_pos)
            if error <= error_bound:
                return joint_positions, joint_orientations
           
            cur_idx = path[i]

            end_pos = joint_positions[path[-1]]
            pi = joint_positions[cur_idx]

            vec1 = end_pos - pi  #旋转之前的offset向量
            vec2 = target_pose - pi

            vec1_norm = np.sqrt((vec1 * vec1).sum())
            vec2_norm = np.sqrt((vec2 * vec2).sum())

            rotate_vec = np.cross(vec1, vec2)
            u = rotate_vec / np.sqrt((rotate_vec * rotate_vec).sum())
            #计算旋转角
            theta = np.arccos((vec1 * vec2).sum() / (vec1_norm * vec2_norm))

            u *= theta
            r = R.from_rotvec(u)
            #joint_orientations[i] = r * joint_orientations[i] # 相当于给原来的旋转再加上一个旋转r

            #给当前节点旋转后，给所有子节点施加相同的旋转
            if below_root: #是Root_joint时已经允许旋转自身。
                joint_orientations[cur_idx] = (
                r * R.from_quat(joint_orientations[cur_idx])).as_quat()

            joint_orientations = update_child_orientation(joint_orientations, r, children, cur_idx)

            joint_positions = update_child_postion(meta_data, joint_positions,
                                                   joint_orientations, children, cur_idx, below_root)

            #一开始肯定不经过root，当把root 进行旋转，移动之后，后续的节点都是经过root的
            if below_root == True and cur_idx == 0:
                below_root = False

  
    return joint_positions, joint_orientations


def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations,
                             relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """

    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    parent = meta_data.joint_parent
    root_idx = path[0]
    if path2[-1] == 0:  #过Root_joint节点
        parent[path2[0]] = -1  #更换新的根节点
        for i in range(1, len(path2)):
            parent[path2[i]] = path[i - 1]

    children = build_tree(parent)

    root_pos = joint_positions[path[0]]
    target_pos = np.array([root_pos[0] + relative_x, target_height, relative_z + root_pos[2] ])
    end_pos = joint_positions[path[-1]]

    error_bound = 0.0001
    max_iteration = 10
    below_root = True


    for _ in range(max_iteration):
        below_root = True
        for i in range(len(path) - 2, -1, -1):  #len(path)-2表示端点的父节点
            error = np.dot(target_pos - end_pos, target_pos - end_pos)
            if error <= error_bound:
                return joint_positions, joint_orientations
           
            cur_idx = path[i]

            end_pos = joint_positions[path[-1]]
            pi = joint_positions[cur_idx]

            vec1 = end_pos - pi  #旋转之前的offset向量
            vec2 = target_pos - pi

            vec1_norm = np.sqrt((vec1 * vec1).sum())
            vec2_norm = np.sqrt((vec2 * vec2).sum())

            rotate_vec = np.cross(vec1, vec2)
            u = rotate_vec / np.sqrt((rotate_vec * rotate_vec).sum())
            #计算旋转角
            theta = np.arccos((vec1 * vec2).sum() / (vec1_norm * vec2_norm))

            u *= theta
            r = R.from_rotvec(u)
            #joint_orientations[i] = r * joint_orientations[i] # 相当于给原来的旋转再加上一个旋转r

            #给当前节点旋转后，给所有子节点施加相同的旋转
            if below_root: #是Root_joint时已经允许旋转自身。
                joint_orientations[cur_idx] = (
                r * R.from_quat(joint_orientations[cur_idx])).as_quat()

            joint_orientations = update_child_orientation(joint_orientations, r, children, cur_idx)

            joint_positions = update_child_postion(meta_data, joint_positions,
                                                   joint_orientations, children, cur_idx, below_root)

            #一开始肯定不经过root，当把root 进行旋转，移动之后，后续的节点都是经过root的
            if below_root == True and cur_idx == 0:
                below_root = False

  


    return joint_positions, joint_orientations


def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations,
                             left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """

    return joint_positions, joint_orientations


if __name__ == '__main__':
    parent = [-1, 0, 1, 2, 0, 4, 6, 7]
    res = build_tree(parent)
    print(res)
