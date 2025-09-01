import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def start_config():
    plt.style.use('seaborn-v0_8-whitegrid')  # 使用白色网格背景

    plt.rcParams['font.sans-serif'] = ['SimSun', 'Microsoft YaHei', 'Times New Roman']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 15
    plt.rcParams['axes.unicode_minus'] = False

start_config()

def plot_solar_angles(phi_deg=30, n=80, t=14, beta=30, gamma=45, filename="solar_angles_demo.png"):  # phi_deg: 观测点纬度（°），n: 年内第 n 天，t: 真太阳时（h）
    # 转换为弧度
    phi = np.radians(phi_deg)
    delta = np.radians(23.45 * np.sin(2 * np.pi / 365 * (284 + n)))  # 赤纬角 δ
    omega = np.radians(15 * (t - 12))  # 时角 ω

    # 计算太阳高度角 h 和方位角 α
    h = np.arcsin(np.sin(delta) * np.sin(phi) + np.cos(delta) * np.cos(phi) * np.cos(omega))
    num = np.sin(delta) * np.cos(phi) - np.cos(delta) * np.sin(phi) * np.cos(omega)
    alpha = np.arccos(num / np.cos(h))  # 方位角 α

    # 面板法线方向
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)
    nx = np.sin(beta_rad) * np.sin(gamma_rad)
    ny = np.sin(beta_rad) * np.cos(gamma_rad)
    nz = np.cos(beta_rad)

    # 设置图形
    fig = plt.figure(figsize=(18, 6))

    # --------- 图1：地球球体 + 赤道、观测点、赤纬、时角 ---------
    ax1 = fig.add_subplot(131, projection='3d')
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(u.size), np.cos(v))
    ax1.plot_surface(x, y, z, color='lightblue', alpha=0.6)

    # 绘制赤道（z=0 平面与球体交线）
    eq_u = np.linspace(0, 2 * np.pi, 200)
    eq_x = np.cos(eq_u)
    eq_y = np.sin(eq_u)
    eq_z = np.zeros_like(eq_u)
    ax1.plot(eq_x, eq_y, eq_z, color='red', linewidth=1.5, label='赤道 (0°纬线)')

    # 标记观测点
    px, py, pz = np.cos(phi), 0, np.sin(phi)
    ax1.scatter(px, py, pz, color='black', s=50)
    ax1.text(px, py, pz, f'观测点 (φ={phi_deg}°)', color='black')

    # 赤纬方向（δ）
    sun1 = np.array([np.cos(delta), 0, np.sin(delta)])
    ax1.quiver(0, 0, 0, *sun1, color='orange', length=1.2, normalize=True)
    ax1.text(*(sun1*1.3), f'赤纬 δ={np.degrees(delta):.1f}°', color='orange')

    # 时角方向（ω）
    sun2 = np.array([np.cos(delta)*np.cos(omega), np.cos(delta)*np.sin(omega), np.sin(delta)])
    ax1.quiver(0, 0, 0, *sun2, color='gold', length=1.2, normalize=True)
    ax1.text(*(sun2*1.3), f'时角 ω={np.degrees(omega):.1f}°', color='gold')

    ax1.set_title('地球球体示意：赤道、观测点、赤纬 δ 与时角 ω')
    ax1.legend(loc='upper right')
    ax1.set_box_aspect([1,1,1])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    # 公式与说明
    ax1.text2D(0.05, 0.95, r"$\delta(n)=23.45\cdot\sin\left(\frac{2\pi}{365}\cdot(284+n)\right)$", transform=ax1.transAxes, fontsize=12, color='darkred', va='top')
    ax1.text2D(0.05, 0.90, r"$\omega(t)=15\cdot(t-12)$", transform=ax1.transAxes, fontsize=12, color='darkred', va='top')
    ax1.text2D(0.05, 0.80, "赤纬角δ、时角ω的几何意义", transform=ax1.transAxes, fontsize=11, color='black', va='top')

    # --------- 图2：太阳高度角 h 推导剖面示意 ---------
    ax2 = fig.add_subplot(132)
    # 只画 y-z 平面，y为南北，z为垂直
    # 地面（y轴）
    ax2.plot([-1, 1], [0, 0], color='gray', linewidth=2)
    ax2.text(1.05, 0, '地面', color='gray', va='center', fontsize=14)
    # 竖直方向（z轴）
    ax2.plot([0, 0], [0, 1], color='black', linewidth=2)
    ax2.text(0, 1.05, '竖直', color='black', ha='center', fontsize=14)
    # 太阳光线
    y_sun = np.cos(h)
    z_sun = np.sin(h)
    ax2.arrow(0, 0, y_sun*0.9, z_sun*0.9, head_width=0.05, head_length=0.08, fc='orange', ec='orange', linewidth=2, length_includes_head=True)
    ax2.text(y_sun*1.05, z_sun*1.05, '太阳光线', color='orange', fontsize=14)
    # 圆弧标注高度角 h
    arc_theta = np.linspace(0, h, 50)
    arc_r = 0.3
    arc_y = arc_r * np.cos(arc_theta)
    arc_z = arc_r * np.sin(arc_theta)
    ax2.plot(arc_y, arc_z, color='blue')
    ax2.text(arc_r*0.7, arc_r*0.2, r'$h$', color='blue', fontsize=18)
    # 辅助虚线
    ax2.plot([0, y_sun], [0, 0], color='gray', linestyle='--')
    ax2.text(y_sun*0.5, -0.05, '地面投影', color='gray', ha='center', fontsize=13)
    # 观测点
    ax2.scatter(0, 0, color='black', s=40)
    ax2.text(0, -0.1, '观测点', color='black', ha='center', fontsize=14)
    # 设置坐标
    ax2.set_xlim(-0.2, 1.2)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_aspect('equal')
    ax2.set_xlabel('南北 (y)', fontsize=15)
    ax2.set_ylabel('垂直 (z)', fontsize=15)
    ax2.set_title('太阳高度角 h 的几何意义（侧视剖面）', fontsize=18)
    # 公式与说明
    ax2.text(0.05, 0.95, r"$h(t)=\arcsin(\sin\phi\cdot\sin\delta+\cos\phi\cdot\cos\delta\cdot\cos\omega)$", transform=ax2.transAxes, fontsize=15, color='darkblue', va='top')
    ax2.text(0.05, 0.80, "太阳高度角h的空间几何意义", transform=ax2.transAxes, fontsize=13, color='black', va='top')

    # --------- 图3：太阳光线与面板法线夹角 θ ---------
    ax3 = fig.add_subplot(133, projection='3d')
    # 面板法线
    n_vec = np.array([nx, ny, nz])
    ax3.quiver(0, 0, 0, *n_vec, length=1.2, normalize=True)
    ax3.text(*(n_vec*1.3), '面板法线', color='blue', fontsize=14)
    # 太阳光线同上
    s_vec = np.array([np.cos(h)*np.sin(alpha), np.cos(h)*np.cos(alpha), np.sin(h)])
    ax3.quiver(0, 0, 0, *s_vec, length=1.2, normalize=True)
    ax3.text(*(s_vec*1.3), '太阳光线', color='orange', fontsize=14)

    # 增加面板投影面积可视化（以半透明平面表示）
    # 面板中心在原点，法线为n_vec，面积单位1
    panel_size = 0.5
    # 构造面板平面
    panel_corners = np.array([
        [-panel_size, -panel_size, 0],
        [ panel_size, -panel_size, 0],
        [ panel_size,  panel_size, 0],
        [-panel_size,  panel_size, 0],
        [-panel_size, -panel_size, 0]
    ])
    # 旋转到法线方向
    # 默认面板法线为z轴，旋转到n_vec
    z_axis = np.array([0,0,1])
    axis = np.cross(z_axis, n_vec)
    if np.linalg.norm(axis) > 1e-6:
        angle = np.arccos(np.dot(z_axis, n_vec)/np.linalg.norm(n_vec))
        rot = R.from_rotvec(axis/np.linalg.norm(axis)*angle)
        panel_corners_rot = rot.apply(panel_corners)
    else:
        panel_corners_rot = panel_corners
    ax3.plot(panel_corners_rot[:,0], panel_corners_rot[:,1], panel_corners_rot[:,2], color='blue', alpha=0.3)
    ax3.text(*(n_vec*0.7), '有效投影面积', color='blue', fontsize=13)

    # 标注夹角 θ
    cos_theta = np.sin(h)*np.cos(beta_rad) + np.cos(h)*np.sin(beta_rad)*np.cos(alpha-gamma_rad)
    theta = np.arccos(cos_theta)
    ax3.text(0,0,0, f'θ={np.degrees(theta):.1f}°', color='black', fontsize=15)

    ax3.set_title('太阳光线与面板法线夹角 θ 示意', fontsize=18)
    ax3.set_xlim(-1,1)
    ax3.set_ylim(-1,1)
    ax3.set_zlim(0,1)
    ax3.set_xlabel('x', fontsize=15)
    ax3.set_ylabel('y', fontsize=15)
    ax3.set_zlabel('z', fontsize=15)
    ax3.legend([])
    # 公式与说明
    ax3.text2D(0.05, 0.95, r"$G(t)=G_{sc}\cdot\cos\theta(t)$", transform=ax3.transAxes, fontsize=15, color='darkgreen', va='top')
    ax3.text2D(0.05, 0.90, r"$\cos\theta(t)=\sin h\cdot\cos\beta+\cos h\cdot\sin\beta\cdot\cos(\alpha-\gamma)$", transform=ax3.transAxes, fontsize=15, color='darkgreen', va='top')
    ax3.text2D(0.05, 0.80, r"单位面积太阳能板接受的太阳辐射强度", transform=ax3.transAxes, fontsize=13, color='black', va='top')
    ax3.text2D(0.05, 0.65, r"$P_{th}(t)=\eta\cdot A\cdot G(t)$", transform=ax3.transAxes, fontsize=15, color='purple', va='top')
    ax3.text2D(0.05, 0.60, r"理论发电功率", transform=ax3.transAxes, fontsize=13, color='black', va='top')
    ax3.text2D(0.05, 0.50, r"$\Delta P(t)=\frac{P_{tr}(t)-P_{th}(t)}{P_{th}(t)}$", transform=ax3.transAxes, fontsize=15, color='brown', va='top')
    ax3.text2D(0.05, 0.45, r"实际与理论输出偏差", transform=ax3.transAxes, fontsize=13, color='black', va='top')

    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.show()

def plot_earth_sphere(phi_deg=30, n=80, t=14, save_path="earth_declination_hour.png"):
    phi = np.radians(phi_deg)
    delta = np.radians(23.45 * np.sin(2 * np.pi / 365 * (284 + n)))
    omega = np.radians(15 * (t - 12))
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111, projection='3d')
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(u.size), np.cos(v))
    ax1.plot_surface(x, y, z, color='lightblue', alpha=0.6)
    eq_u = np.linspace(0, 2 * np.pi, 200)
    eq_x = np.cos(eq_u)
    eq_y = np.sin(eq_u)
    eq_z = np.zeros_like(eq_u)
    ax1.plot(eq_x, eq_y, eq_z, color='red', linewidth=2, label='赤道 (0°纬线)')
    px, py, pz = np.cos(phi), 0, np.sin(phi)
    ax1.scatter(px, py, pz, color='black', s=60)
    ax1.text(px, py, pz, f'观测点 (φ={phi_deg}°)', color='black', fontsize=14)
    sun1 = np.array([np.cos(delta), 0, np.sin(delta)])
    ax1.quiver(0, 0, 0, *sun1, color='orange', length=1.2, normalize=True)
    ax1.text(*(sun1*1.3), f'赤纬 δ={np.degrees(delta):.1f}°', color='orange', fontsize=14)
    sun2 = np.array([np.cos(delta)*np.cos(omega), np.cos(delta)*np.sin(omega), np.sin(delta)])
    ax1.quiver(0, 0, 0, *sun2, color='gold', length=1.2, normalize=True)
    ax1.text(*(sun2*1.3), f'时角 ω={np.degrees(omega):.1f}°', color='gold', fontsize=14)
    ax1.set_title('地球球体示意：赤道、观测点、赤纬 δ 与时角 ω', fontsize=18)
    ax1.legend(loc='upper right', fontsize=13)
    ax1.set_box_aspect([1,1,1])
    ax1.set_xlabel('X', fontsize=15)
    ax1.set_ylabel('Y', fontsize=15)
    ax1.set_zlabel('Z', fontsize=15)
    ax1.text2D(0.05, 0.95, r"$\delta(n)=23.45\cdot\sin\left(\frac{2\pi}{365}\cdot(284+n)\right)$", transform=ax1.transAxes, fontsize=15, color='darkred', va='top')
    ax1.text2D(0.05, 0.90, r"$\omega(t)=15\cdot(t-12)$", transform=ax1.transAxes, fontsize=15, color='darkred', va='top')
    ax1.text2D(0.05, 0.80, "赤纬角δ、时角ω的几何意义", transform=ax1.transAxes, fontsize=13, color='black', va='top')
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()

def plot_elevation_section(phi_deg=30, n=80, t=14, save_path="elevation_azimuth.png"):
    phi = np.radians(phi_deg)
    delta = np.radians(23.45 * np.sin(2 * np.pi / 365 * (284 + n)))
    omega = np.radians(15 * (t - 12))
    h = np.arcsin(np.sin(delta) * np.sin(phi) + np.cos(delta) * np.cos(phi) * np.cos(omega))
    fig = plt.figure(figsize=(7, 7))
    ax2 = fig.add_subplot(111)
    ax2.plot([-1, 1], [0, 0], color='gray', linewidth=2)
    ax2.text(1.05, 0, '地面', color='gray', va='center', fontsize=14)
    ax2.plot([0, 0], [0, 1], color='black', linewidth=2)
    ax2.text(0, 1.05, '竖直', color='black', ha='center', fontsize=14)
    y_sun = np.cos(h)
    z_sun = np.sin(h)
    ax2.arrow(0, 0, y_sun*0.9, z_sun*0.9, head_width=0.05, head_length=0.08, fc='orange', ec='orange', linewidth=2, length_includes_head=True)
    ax2.text(y_sun*1.05, z_sun*1.05, '太阳光线', color='orange', fontsize=14)
    arc_theta = np.linspace(0, h, 50)
    arc_r = 0.3
    arc_y = arc_r * np.cos(arc_theta)
    arc_z = arc_r * np.sin(arc_theta)
    ax2.plot(arc_y, arc_z, color='blue')
    ax2.text(arc_r*0.7, arc_r*0.2, r'$h$', color='blue', fontsize=18)
    ax2.plot([0, y_sun], [0, 0], color='gray', linestyle='--')
    ax2.text(y_sun*0.5, -0.05, '地面投影', color='gray', ha='center', fontsize=13)
    ax2.scatter(0, 0, color='black', s=40)
    ax2.text(0, -0.1, '观测点', color='black', ha='center', fontsize=14)
    ax2.set_xlim(-0.2, 1.2)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_aspect('equal')
    ax2.set_xlabel('南北 (y)', fontsize=15)
    ax2.set_ylabel('垂直 (z)', fontsize=15)
    ax2.set_title('太阳高度角 h 的几何意义（侧视剖面）', fontsize=18)
    ax2.text(0.05, 0.95, r"$h(t)=\arcsin(\sin\phi\cdot\sin\delta+\cos\phi\cdot\cos\delta\cdot\cos\omega)$", transform=ax2.transAxes, fontsize=15, color='darkblue', va='top')
    ax2.text(0.05, 0.80, "太阳高度角h的空间几何意义", transform=ax2.transAxes, fontsize=13, color='black', va='top')
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()

def plot_panel_theta(phi_deg=30, n=80, t=14, beta=30, gamma=45, save_path="power_output.png"):
    phi = np.radians(phi_deg)
    delta = np.radians(23.45 * np.sin(2 * np.pi / 365 * (284 + n)))
    omega = np.radians(15 * (t - 12))
    h = np.arcsin(np.sin(delta) * np.sin(phi) + np.cos(delta) * np.cos(phi) * np.cos(omega))
    num = np.sin(delta) * np.cos(phi) - np.cos(delta) * np.sin(phi) * np.cos(omega)
    alpha = np.arccos(num / np.cos(h))
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)
    nx = np.sin(beta_rad) * np.sin(gamma_rad)
    ny = np.sin(beta_rad) * np.cos(gamma_rad)
    nz = np.cos(beta_rad)
    n_vec = np.array([nx, ny, nz])
    s_vec = np.array([np.cos(h)*np.sin(alpha), np.cos(h)*np.cos(alpha), np.sin(h)])
    fig = plt.figure(figsize=(7, 7))
    ax3 = fig.add_subplot(111, projection='3d')
    ax3.quiver(0, 0, 0, *n_vec, length=1.2, normalize=True)
    ax3.text(*(n_vec*1.3), '面板法线', color='blue', fontsize=14)
    ax3.quiver(0, 0, 0, *s_vec, length=1.2, normalize=True)
    ax3.text(*(s_vec*1.3), '太阳光线', color='orange', fontsize=14)
    panel_size = 0.5
    panel_corners = np.array([
        [-panel_size, -panel_size, 0],
        [ panel_size, -panel_size, 0],
        [ panel_size,  panel_size, 0],
        [-panel_size,  panel_size, 0],
        [-panel_size, -panel_size, 0]
    ])
    z_axis = np.array([0,0,1])
    axis = np.cross(z_axis, n_vec)
    if np.linalg.norm(axis) > 1e-6:
        angle = np.arccos(np.dot(z_axis, n_vec)/np.linalg.norm(n_vec))
        rot = R.from_rotvec(axis/np.linalg.norm(axis)*angle)
        panel_corners_rot = rot.apply(panel_corners)
    else:
        panel_corners_rot = panel_corners
    ax3.plot(panel_corners_rot[:,0], panel_corners_rot[:,1], panel_corners_rot[:,2], color='blue', alpha=0.3)
    ax3.text(*(n_vec*0.7), '有效投影面积', color='blue', fontsize=13)
    cos_theta = np.sin(h)*np.cos(beta_rad) + np.cos(h)*np.sin(beta_rad)*np.cos(alpha-gamma_rad)
    theta = np.arccos(cos_theta)
    ax3.text(0,0,0, f'θ={np.degrees(theta):.1f}°', color='black', fontsize=15)
    ax3.set_title('太阳光线与面板法线夹角 θ 示意', fontsize=18)
    ax3.set_xlim(-1,1)
    ax3.set_ylim(-1,1)
    ax3.set_zlim(0,1)
    ax3.set_xlabel('x', fontsize=15)
    ax3.set_ylabel('y', fontsize=15)
    ax3.set_zlabel('z', fontsize=15)
    ax3.legend([])
    ax3.text2D(0.05, 0.95, r"$G(t)=G_{sc}\cdot\cos\theta(t)$", transform=ax3.transAxes, fontsize=15, color='darkgreen', va='top')
    ax3.text2D(0.05, 0.90, r"$\cos\theta(t)=\sin h\cdot\cos\beta+\cos h\cdot\sin\beta\cdot\cos(\alpha-\gamma)$", transform=ax3.transAxes, fontsize=15, color='darkgreen', va='top')
    ax3.text2D(0.05, 0.80, r"单位面积太阳能板接受的太阳辐射强度", transform=ax3.transAxes, fontsize=13, color='black', va='top')
    ax3.text2D(0.05, 0.65, r"$P_{th}(t)=\eta\cdot A\cdot G(t)$", transform=ax3.transAxes, fontsize=15, color='purple', va='top')
    ax3.text2D(0.05, 0.60, r"理论发电功率", transform=ax3.transAxes, fontsize=13, color='black', va='top')
    ax3.text2D(0.05, 0.50, r"$\Delta P(t)=\frac{P_{tr}(t)-P_{th}(t)}{P_{th}(t)}$", transform=ax3.transAxes, fontsize=15, color='brown', va='top')
    ax3.text2D(0.05, 0.45, r"实际与理论输出偏差", transform=ax3.transAxes, fontsize=13, color='black', va='top')
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()

class SolarPowerAnalysisAT:
    def __init__(self, data_path, output_dir='AT分析结果'):
        self.data_path = data_path
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 奥地利（以维也纳为例）地理参数
        self.station_latitude = 48.2082
        self.station_longitude = 16.3738        
        self.station_capacity = 1000  # 2MW，单位统一为MW
        self.panel_efficiency = 0.17  # 组件效率
        self.system_losses = 0.10
        self.panel_tilt = 38  # 奥地利最佳倾角约为纬度，推荐35-40°
        self.panel_azimuth = 180
        self.load_data()

    def load_data(self):
        print("加载AT相关数据...")
        df = pd.read_csv(self.data_path)
        # 只保留AT相关列
        at_cols = [c for c in df.columns if c.startswith('AT_')]
        at_cols += ['utc_timestamp', 'cet_cest_timestamp']
        self.df = df[at_cols].copy()
        self.df['datetime'] = pd.to_datetime(self.df['utc_timestamp'])
        self.df['year'] = self.df['datetime'].dt.year
        self.df['month'] = self.df['datetime'].dt.month
        self.df['day'] = self.df['datetime'].dt.day
        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['minute'] = self.df['datetime'].dt.minute
        self.df['day_of_year'] = self.df['datetime'].dt.dayofyear
        self.df['season'] = self.df['month'].apply(self.get_season)
        self.df['solar_time'] = self.df['hour'] + self.df['minute'] / 60
        print(f"AT数据加载完成，共{len(self.df)}条")

    def get_season(self, month):
        if month in [12, 1, 2]:
            return '冬季'
        elif month in [3, 4, 5]:
            return '春季'
        elif month in [6, 7, 8]:
            return '夏季'
        else:
            return '秋季'

    def solar_position(self, day_of_year, solar_time):
        declination = np.radians(23.45) * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        hour_angle = np.radians(15 * (solar_time - 12))
        lat_rad = np.radians(self.station_latitude)
        elevation = np.arcsin(np.sin(declination) * np.sin(lat_rad) +
                              np.cos(declination) * np.cos(lat_rad) * np.cos(hour_angle))
        azimuth = np.arctan2(np.sin(hour_angle),
                             np.cos(hour_angle) * np.sin(lat_rad) -
                             np.tan(declination) * np.cos(lat_rad))
        return elevation, azimuth

    def calculate_theoretical_power(self):
        print("计算理论可发功率...")
        # 1. 计算太阳位置
        elevations, azimuths = self.solar_position(self.df['day_of_year'], self.df['solar_time'])
        self.df['sun_elevation'] = np.degrees(elevations)
        self.df['sun_azimuth'] = np.degrees(azimuths)

        # 2. 计算入射角
        panel_tilt_rad = np.radians(self.panel_tilt)
        panel_azimuth_rad = np.radians(self.panel_azimuth)
        
        cos_incidence = (np.sin(elevations) * np.cos(panel_tilt_rad) +
                        np.cos(elevations) * np.sin(panel_tilt_rad) *
                        np.cos(azimuths - panel_azimuth_rad))
        cos_incidence = np.maximum(cos_incidence, 0)

        # 3. 辐照度计算（考虑直射和散射）
        if 'AT_irradiance' in self.df.columns and self.df['AT_irradiance'].sum() > 0:
            # 使用实测数据
            ghi = self.df['AT_irradiance']  # 全球水平辐照度
            # 估算散射辐照比例（简化模型）
            diffuse_ratio = np.where(
                self.df['sun_elevation'] > 0,
                0.3 + 0.7 * (1 - cos_incidence),  # 阴天时散射比例更大
                0.4  # 日出日落时散射占主导
            )
            
            direct_irradiance = ghi * (1 - diffuse_ratio)  # 直射辐照
            diffuse_irradiance = ghi * diffuse_ratio       # 散射辐照
        else:            # 使用改进的理论模型
            base_irradiance = 1200  # 增加晴空辐照度基准值（W/m²）
            
            # 根据太阳高度角动态调整直射和散射比例
            sun_elevation_deg = self.df['sun_elevation']
            
            # 直射比例随太阳高度角变化（高度角越大，直射比例越大）
            direct_ratio = np.clip(0.5 + sun_elevation_deg/90 * 0.4, 0, 0.9)
            
            # 散射比例随高度角变化（清晨和傍晚散射比例较小）
            diffuse_ratio = np.where(
                sun_elevation_deg > 0,
                (1 - direct_ratio) * np.clip(sun_elevation_deg/40, 0, 1),  # 白天
                0.1 * np.clip((sun_elevation_deg + 6)/6, 0, 1)  # 晨昏时段
            )
            
            # 计算直射和散射辐照度
            direct_irradiance = np.where(
                sun_elevation_deg > 0,
                base_irradiance * direct_ratio * cos_incidence,
                0
            )
            
            diffuse_irradiance = np.where(
                sun_elevation_deg > -6,
                base_irradiance * diffuse_ratio * (1 + cos_incidence)/2,
                0
            )

        # 4. 温度修正
        temp_coefficient = -0.0045
        standard_temp = 25
        if 'AT_temperature' in self.df.columns and self.df['AT_temperature'].sum() > 0:
            temp_factor = 1 + temp_coefficient * (self.df['AT_temperature'] - standard_temp)
        else:
            # 根据月份估算温度影响
            avg_monthly_temp = {
                12:-1, 1:-1, 2:1,    # 冬季
                3:5, 4:10, 5:15,     # 春季
                6:18, 7:20, 8:20,    # 夏季
                9:15, 10:10, 11:5    # 秋季
            }
            self.df['est_temp'] = self.df['month'].map(avg_monthly_temp)
            temp_factor = 1 + temp_coefficient * (self.df['est_temp'] - standard_temp)

        # 5. 总辐照度和功率计算
        self.df['tilted_irradiance'] = direct_irradiance + diffuse_irradiance

        # 计算面板面积（使用安装容量反推）
        panel_area = self.station_capacity * 1_000_000 / (self.panel_efficiency * 1_000)  # m2
        
        # 计算理论功率
        self.df['theoretical_power'] = (
            self.df['tilted_irradiance'] * panel_area * self.panel_efficiency * 
            temp_factor * (1 - self.system_losses) / 1_000_000  # 转换为MW
        )
        # 限制最大功率
        self.df['theoretical_power'] = np.minimum(self.df['theoretical_power'], self.station_capacity)
        
        print("理论功率计算完成")

    def analyze_and_plot(self):
        # 长周期
        monthly = self.df.groupby('month').agg({'AT_solar_generation_actual': 'mean', 'theoretical_power': 'mean'}).rename(columns={'AT_solar_generation_actual':'实际功率', 'theoretical_power':'理论功率'})
        plt.figure(figsize=(8,5))
        monthly.plot(kind='bar')
        plt.title('奥地利光伏月均功率')
        plt.ylabel('功率 (MW)')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/AT_月均功率.png', dpi=300)
        plt.close()
        # 短周期
        hourly = self.df.groupby('hour').agg({'AT_solar_generation_actual': 'mean', 'theoretical_power': 'mean'}).rename(columns={'AT_solar_generation_actual':'实际功率', 'theoretical_power':'理论功率'})
        plt.figure(figsize=(8,5))
        hourly.plot()
        plt.title('奥地利光伏日内平均功率')
        plt.ylabel('功率 (MW)')
        plt.xlabel('小时')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/AT_日内平均功率.png', dpi=300)
        plt.close()        # 偏差分析
        self.df['power_deviation'] = self.df['AT_solar_generation_actual'] - self.df['theoretical_power']
        self.df['power_relative_deviation'] = np.where(
            self.df['theoretical_power'] > 0,
            (self.df['AT_solar_generation_actual'] - self.df['theoretical_power']) / self.df['theoretical_power'] * 100,
            0
        )
        
        # 筛选有效数据（去除夜间和异常值）
        valid_data = self.df[
            (self.df['theoretical_power'] > 0.1) & 
            (self.df['power_relative_deviation'].between(-200, 200))
        ]
        
        # 创建组合图：小提琴图+箱线图
        plt.figure(figsize=(12, 6))
        
        # 按季节绘制小提琴图
        sns.violinplot(data=valid_data, x='season', y='power_relative_deviation',
                      inner='box', color='lightblue')
        
        # 添加均值点
        season_means = valid_data.groupby('season')['power_relative_deviation'].mean()
        plt.scatter(range(len(season_means)), season_means, color='red', marker='o', 
                   s=100, label='季节平均值')
        
        # 美化图形
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        plt.title('奥地利光伏功率相对偏差分布（按季节）', fontsize=12)
        plt.xlabel('季节', fontsize=10)
        plt.ylabel('相对偏差 (%)', fontsize=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/AT_功率相对偏差季节箱线图.png', dpi=300)
        plt.close()
        
        # 创建按小时分布的热力图
        plt.figure(figsize=(12, 6))
        pivot_table = valid_data.pivot_table(
            values='power_relative_deviation', 
            index='hour',
            columns='season',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_table, cmap='RdYlBu_r', center=0, annot=True, fmt='.1f',
                   cbar_kws={'label': '相对偏差 (%)'})
        
        plt.title('日内-季节相对偏差分布热力图', fontsize=12)
        plt.xlabel('季节', fontsize=10)
        plt.ylabel('小时', fontsize=10)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/AT_功率相对偏差分布.png', dpi=300)
        plt.close()
        print('分析与可视化完成！')

    def run_all(self):
        self.calculate_theoretical_power()
        self.analyze_and_plot()

if __name__ == '__main__':
    start_config()
    plot_earth_sphere()
    plot_elevation_section()
    plot_panel_theta()
    analyzer = SolarPowerAnalysisAT('../数据集/Q1.csv')
    analyzer.run_all()
