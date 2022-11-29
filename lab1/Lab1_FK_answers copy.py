import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    #读取BVH文件Frame time之后的数据到motion_data中
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def recursive_construct_test(lines, begin):
    joint_name = []
    joint_parent = []
    joint_offset = np.array([[]])
    # if(begin >= end):
    #     return joint_name, joint_parent, joint_offset
    if(lines[begin].split()[0]== 'JOINT'):
        name = lines[begin].split()[1]
        joint_name = [name]
        joint_parent = [-1]
        off0 = lines[begin+2].split()[1]
        off1 = lines[begin+2].split()[2]
        off2 = lines[begin+2].split()[3]
        joint_offset = np.array([[off0, off1, off2]])        

    while(lines[begin].split()[0] == 'JOINT'):

        block_end, j_name, j_parent, j_offset = recursive_construct_test(lines, begin+4)
        
        joint_name = joint_name + j_name
        joint_offset = np.concatenate((joint_offset, j_offset), axis=0)

        off = len(joint_name)
        length = len(j_parent)
        #子节点的parent都往后移off
        for i in range(1, len(j_parent)):
            j_parent[i] += off
        
        joint_parent = joint_parent + j_parent

        begin = block_end + 1
        if(lines[begin+1].split()[0] == 'JOINT'):
            begin += 1
        

    if(lines[begin].split()[0] == 'End'):
        name = lines[begin-4].split()[1]+'_end'
        joint_name.append(name)
        off0 = lines[begin+2].split()[1]
        off1 = lines[begin+2].split()[2]
        off2 = lines[begin+2].split()[3]
        
        joint_parent = [-1]
        joint_offset = np.array([[off0, off1, off2]])

        begin += 3    

    return begin, joint_name, joint_parent, joint_offset


def rescursive_construct(lines, idx):
    #递归开始，必然以 { 开始
    assert(lines[idx].split()[0]=='{')
    
    joint_name = []
    joint_parent = [-1]
    off0 = lines[idx+1].split()[1]
    off1 = lines[idx+1].split()[2]
    off2 = lines[idx+1].split()[3]
    joint_offset = np.array([[off0, off1, off2]], dtype=np.float64)     

    #添加当前层信息
    if(lines[idx-1].split()[0] == 'ROOT'):
        joint_name = [lines[idx-1].split()[1]]
    elif(lines[idx-1].split()[0] == 'JOINT'):
        joint_name = [lines[idx-1].split()[1]]
    elif(lines[idx - 1].split()[0] == 'End'):
        joint_name = [lines[idx-5].split()[1] + '_end']
    
    idx += 1

    
    
    while( idx < len(lines) and (not lines[idx].split()[0]=='{' and  not lines[idx].split()[0]=='}')):
        idx += 1
    
    if(idx >= len(lines)):
        return idx, joint_name

    #如果有子节点，递归
    while(lines[idx].split()[0] == '{'):
        #idx 为子节点 } 的位置
        idx, child_name, child_parent, child_offset = rescursive_construct(lines, idx)
        assert(lines[idx].split()[0] == '}')
        
        joint_name += child_name
        joint_offset = np.concatenate((joint_offset, child_offset), axis=0)

        off = len(joint_parent)
        length = len(child_parent)

        #子节点的父节点的索引都是位于0位置
        child_parent[0] = 0
        for i in range(1, length):
            child_parent[i] += off
        joint_parent += child_parent

        #如果还有其他
        if(lines[idx+2].split()[0] == '{'):
            #有多个子节点
            idx += 2
        else:
            idx += 1
            assert(lines[idx].split()[0] == '}')
            break
    
    #该节点递归结束，必然以 } 结尾
    assert(lines[idx].split()[0]=='}') 
    return idx, joint_name, joint_parent, joint_offset
 
   
    

def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent= []
    joint_offset = np.array([[]], dtype=np.float64)
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        _, joint_name, joint_parent, joint_offset = rescursive_construct(lines, 2)

    return joint_name, joint_parent, joint_offset


def quantriteion(q1, q2):
    w = [q1[0] * q2[0] + (q1[1:4] * q2[1:4]).sum()]
    v = q1[0] * q2[1:4] + q2[0] * q1[1:4] - cross(q1[1:4], q2[1:4])
    w = np.concatenate((w, v), axis=0)
    return w
def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """

    '''
        joint_name 是bvh文件中从上到下按顺序输入的，所以每个joint_name和data中对应的数据顺序是相同的
        joint_name为root时对应6个数据，否则为3个数据
        data中前六个数据为全局的位置

        1. 获取frame_id对应的data
        2. 根据data计算各个关节的全局位置
        3. 根据data计算各个关节的全区朝向
    '''
    data = motion_data[frame_id]
    euler_angles = []
    
    #Origin Postion
    origin = data[0:3]

    idx = 3
    for name in joint_name:
        if name.endswith('_end'):
            euler_angles.append(np.zeros((3,), dtype=np.float64))
        else:
            euler_angles.append(data[idx: idx + 3])
            idx += 3

    #Rotations
    rotations = []
    tmp_i = 0
    for i in range(0, len(joint_name)):
        r = R.from_euler('XYZ', euler_angles[i], degrees=True).as_quat()
        rotations.append(r)
        # #如果时页节点，euler_angeles中没有对应的channel项，设置它的rotation为单位四元数，不旋转
        # if(joint_name[i].endswith('_end')):
        #     r = R.from_euler('XYZ', [0.0, 0.0, 0.0], degrees=True).as_quat()
        #     rotations.append(r)
        # else:
        #     r = R.from_euler('XYZ', euler_angles[tmp_i], degrees=True).as_quat()
        #     rotations.append(r)
        #     tmp_i += 1
                   
    joint_orientations = [rotations[0]]
    
    #计算Orientations
    for i in range(1, len(joint_name)):
        parent_idx = joint_parent[i]
        parent_orientation = joint_orientations[parent_idx]
        #O(i)= O(i-1) * Ri
        o = (R.from_quat(parent_orientation) * R.from_quat(rotations[i])).as_quat()
        joint_orientations.append(o)

    #根节点的位置
    joint_positions =[origin + R.from_quat(joint_orientations[0]).apply(joint_offset[0])]

    #计算Positions
    for i in range(1, len(joint_name)):
        parent_idx = joint_parent[i]
        parent_position = joint_positions[parent_idx]
        # P(i) = P(i-1) + Q(i-1) offset(i)
        pos = parent_position + R.from_quat(joint_orientations[parent_idx]).apply(joint_offset[i])
        joint_positions.append(pos)

    joint_positions = np.array(joint_positions)
    joint_orientations = np.array(joint_orientations)

    return joint_positions, joint_orientations
def part2_forward_kinematics_test(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """

    '''
        joint_name 是bvh文件中从上到下按顺序输入的，所以每个joint_name和data中对应的数据顺序是相同的
        joint_name为root时对应6个数据，否则为3个数据
        data中前六个数据为全局的位置

        1. 获取frame_id对应的data
        2. 根据data计算各个关节的全局位置
        3. 根据data计算各个关节的全区朝向
    '''
    data = motion_data[frame_id]
    euler_angles = []
    
    #Origin Postion
    origin = data[0:3]

    idx = 3
    for name in joint_name:
        if not name.endswith('_end'):
            euler_angles.append(data[idx: idx + 3])
            idx += 3      
            

    #Rotations
    rotations = []
    # for i in range(0, len(joint_name)):
    #     r = R.from_euler('XYZ', euler_angles[i], degrees=True).as_quat()
    #     rotations.append(r)
    for euler_angle in euler_angles:
        r = R.from_euler('XYZ', euler_angle, degrees=True).as_quat()
        rotations.append(r)

    joint_orientations = [rotations[0]]

    #Rotations是和 joint_name顺序一致
    #计算Orientations
    end_num = 0
    for i in range(1, len(joint_name)):
        if joint_name[i].endswith('_end'):#end节点不用旋转，跳过
            end_num += 1
            continue 

        parent_idx = joint_parent[i]
        parent_orientation = joint_orientations[parent_idx-end_num]
        #O(i)= O(i-1) * Ri
        o = (R.from_quat(parent_orientation) * R.from_quat(rotations[i - end_num])).as_quat()
        joint_orientations.append(o)

    #根节点的位置
    joint_positions =[origin]

    #计算Positions
    end_num = 0
    for i in range(1, len(joint_name)):
        if joint_name[i].endswith('_end'):#end节点不用旋转，跳过
            end_num += 1
        parent_idx = joint_parent[i]
        parent_position = joint_positions[parent_idx]
        # P(i) = P(i-1) + Q(i-1) offset(i)
        pos = parent_position + R.from_quat(joint_orientations[parent_idx-end_num]).apply(joint_offset[i])
        joint_positions.append(pos)

    joint_positions = np.array(joint_positions)
    joint_orientations = np.array(joint_orientations)

    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    a_data = load_motion_data(A_pose_bvh_path)
    t_names, _, _ = part1_calculate_T_pose(T_pose_bvh_path)
    a_names,_, _ = part1_calculate_T_pose(A_pose_bvh_path)

    motion_data = a_data[:, 0:3]


    for t_name in t_names:
        if t_name.endswith('_end'):
            continue

        end_num = 0
        for i in range(0, len(a_names)):
            if a_names[i].endswith('_end'):
                end_num += 1
                continue
            if a_names[i] == t_name:
                idx = 3 + 3*(i - end_num)
                
                #只有lShoulder/rShoulder两个关节是有旋转的，Rt = Ra * Q(A->T)T
                # lShoulder/rShoulder的子节点由于 Q(A->T) 与Qpi(A->T)相等因此 Rt = Qpi(A->T) * Ra * Q(A->T)T=Ra
                if t_name == 'lShoulder':#lShoulder
                    Q = R.from_euler('XYZ',[0,0,-45], degrees=True)
                    QT = Q.inv()
                    ra = R.from_euler('XYZ', a_data[:, idx:idx+3], degrees=True)
                    rt = ra * QT
                    rt = rt.as_euler('XYZ')
                    motion_data = np.hstack((motion_data, rt))
                elif t_name == 'rShoulder':#rShoulder
                    Q = R.from_euler('XYZ',[0,0,45], degrees=True)
                    QT = Q.inv()
                    ra = R.from_euler('XYZ', a_data[:, idx:idx+3], degrees=True)
                    rt= ra * QT
                    rt = rt.as_euler('XYZ')
                    motion_data = np.hstack((motion_data, rt))
                else:
                    motion_data = np.hstack((motion_data, a_data[:, idx:idx+3]))

                
                #motion_data = np.concatenate((motion_data, a_data[:, idx:idx+3]), axis=1)
                #motion_data.append(a_data[:, idx: idx+3])
    
    return  motion_data


def part3_retarget_func_test(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    a_data = load_motion_data(A_pose_bvh_path)
    t_names, _, _ = part1_calculate_T_pose(T_pose_bvh_path)
    a_names,_, _ = part1_calculate_T_pose(A_pose_bvh_path)

    motion_data = a_data[:, 0:3]


    for t_name in t_names:
        if t_name.endswith('_end'):
            continue

        end_num = 0
        for i in range(0, len(a_names)):
            if a_names[i].endswith('_end'):
                end_num += 1
                continue
            if a_names[i] == t_name:
                idx = 3 + 3*(i - end_num)
                # if i == 15 or i == 16 or i == 17 or i == 18 or i ==19:
                #     r = R.from_euler('XYZ',[0.0,0.0,45],degrees=True)
                #     motion_data = np.hstack((motion_data, r.apply(a_data[:, idx:idx+3]) ))
                # elif i == 20 or i == 21 or i == 22 or i == 23 or i ==24:
                #     r = R.from_euler('XYZ',[0.0,0.0,-45],degrees=True)
                #     motion_data = np.hstack((motion_data, r.apply(a_data[:, idx:idx+3]) ))

                # else:
                if i ==16:
                    r = R.from_euler('XYZ',[0,0,45], degrees=True)
                    motion_data = np.hstack((motion_data, r.apply(a_data[:, idx:idx+3])))
                motion_data = np.hstack((motion_data, a_data[:, idx:idx+3]))

                
                #motion_data = np.concatenate((motion_data, a_data[:, idx:idx+3]), axis=1)
                #motion_data.append(a_data[:, idx: idx+3])
    
    return  motion_data

    # motion_data = None
    # return motion_data

if __name__ == '__main__':
    # with open("data/walk60.bvh", 'r') as f:
    #     lines = f.readlines()
    #     idx, joint_name, joint_parent, joint_off = rescursive_construct(lines, 2)
    #     for i in range(0, len(joint_name)):
    #         print(f'{i}, {joint_name[i]}, {joint_parent[i]}, {joint_off[i]}')

    # bvh_file_path = "data/walk60.bvh" 
    # joint_name, joint_parent, joint_offset = part1_calculate_T_pose(bvh_file_path)
    # motion_data = load_motion_data(bvh_file_path)
    # joint_positions, joint_orientations = part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, 0)

    part3_retarget_func('data/walk60.bvh', 'data/A_pose_run.bvh')