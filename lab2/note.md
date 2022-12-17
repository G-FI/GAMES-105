1. lerp/slerp 对旋转进行插值时有震荡产生
    答：两个quaternion夹角大于90°，导致插值出来的值不合理
    解决：cos判断quaternoin夹角，大于90°时将一个quat按原点对称，使得夹角小于90°
2. task2当前只是对每一帧根据未来的位置desired_pos来与motions进行比较计算cost，选择cost最小的动作
    问题：但是刚开始desired_pos就不为零，本来因该idel，但是当cost计算后匹配到了run，然后即使没有输入也会一直run
3. 当前实现：

```python
    cost =0
    for motion in motions:
        for frame in frame:
            cost = compute_cost(current_pose, candidate_pose) #candidate pose是motion的第frame帧
            #更新cost,记录candidate motion 和 candidate frame
    #如果motion和正在播放的motion是同一个 and frame 相近不做更新
    #否则进行插值过度 TODO
    
```
问题：帧率奇低，因为每一帧都遍历所有帧
    动作也不好，可能的问题是cost函数定义问题