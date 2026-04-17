from numpy import *
import numpy as np
from numpy.linalg import inv

'''
一个3D激光雷达单元正在扫描一个近似平面的表面，
返回距离、仰角和方位角测量值。为了估计
表面的参数方程（作为平面），我们需要
找到一组最佳拟合测量值的参数。

sph_to_cat 和 estimate_params 函数，
分别将激光雷达测量值转换到笛卡尔坐标系
并估计平面参数。我们假设
测量噪声可以忽略不计。
'''


def sph_to_cart(epsilon, alpha, r):
  """
  将传感器读数转换到传感器坐标系中的笛卡尔坐标。
  epsilon 和 alpha 的值以弧度给出,r 以米为单位。
  epsilon 是仰角,alpha 是方位角（即在 x,y 平面内）。
  """
  p = np.zeros(3)  # 位置向量

  # 转换为角度
  # 似乎不需要
  #epsilon = epsilon / 180 * pi
  #alpha = alpha / 180 * pi


  p[0]=r*np.cos(alpha)*np.cos(epsilon)
  p[1]=r*np.sin(alpha)*np.cos(epsilon)
  p[2]=r*np.sin(epsilon)

  return p

def estimate_params(P):
  """
  从笛卡尔坐标系中的传感器读数估计参数。
  P 矩阵中的每一行包含一个3D点测量；
  矩阵 P 的大小为 n x 3（对于 n 个点）。格式为：

  P = [[x1, y1, z1],
       [x2, y2, z2], ...]

  其中所有坐标值以米为单位。需要三个参数
  来拟合平面，即 a、b 和 c，根据方程

  z = a + bx + cy

  函数应将参数作为大小为三的 NumPy 数组返回，
  顺序为 [a, b, c]。
  """

  P = array(P)

  param_est = zeros(3)

  ones_row = matrix(ones(len(P)))  # 行数
  ones_column = ones_row.T
  
  x_row = (P[:,0])
  x_column = matrix(x_row).T
  y_row = (P[:,1])
  y_column = matrix(y_row).T

  z_row = (P[:,2])
  z_column = matrix(z_row).T

  A = hstack(((ones_column),(x_column),(y_column)))

  B = z_column


  param_set = linalg.inv(A.T.dot(A)).dot(A.T).dot(B)

  param_est[0] = param_set[0,0]
  param_est[1] = param_set[1,0]
  param_est[2] = param_set[2,0]

  return param_est



P = [[1, 2, 3], [10, 3, 4], [3, 4, 15], [3, 4, 15]]
print("点测量值：")
print(P)
x = estimate_params(P)
print("估计的平面参数：")
print(x)
