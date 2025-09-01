import math
import heapq
from typing import Dict, List, Tuple, Set, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import datetime
import time
import threading  # 新增导入
import random  # 新增导入
from pylab import mpl
 
# 设置中文显示字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 地球半径（米），用于计算经纬度之间的距离
EARTH_RADIUS = 6371000

# 游船相关参数
BOAT_SPEED = 5.0  # 游船速度，单位：米/秒 (约18公里/小时)
BOAT_SCHEDULE_START = 8  # 游船开始时间（小时）
BOAT_SCHEDULE_END = 16   # 游船结束时间（小时）
BOAT_WAIT_TIME = 600     # 游船等待时间，单位：秒 (10分钟)
BOAT_INTERVAL = 3600     # 游船发车间隔，单位：秒 (1小时)

# 观光车相关参数
TOUR_BUS_SPEED = 8.0  # 观光车速度，单位：米/秒 (约28.8公里/小时)
TOUR_BUS_SCHEDULE_START = 8  # 观光车开始时间（小时）
TOUR_BUS_SCHEDULE_END = 16   # 观光车结束时间（小时）
TOUR_BUS_DEPARTURE_INTERVAL = 600  # 观光车发车间隔，单位：秒 (10分钟)
TOUR_BUS_STOP_TIME = 180  # 观光车每站停靠时间，单位：秒 (3分钟)
TOUR_BUS_BASE_FARE = 5  # 观光车基础票价
TOUR_BUS_PER_STOP_FARE = 0  # 观光车每站加价
TOUR_BUS_COUNT = 10  # 观光车数量

# 定义观光车路线 - 使用字典存储多条路线
TOUR_BUS_ROUTES = {
    1: ["入口", "花园", "湖泊", "观景台", "餐厅", "博物馆", "出口"],
    2: ["入口", "花园", "观景台", "博物馆", "出口"],
    # 可以添加更多路线
}

# 全局变量存储观光车时间表
tour_bus_schedule = {}

# 添加全局变量缓存景点间距离
attraction_distances_cache = None


class Node:
    """节点类，表示景点或道路转弯点"""
    def __init__(self, node_id: str, lat: float, lon: float, is_attraction: bool = False):
        self.id = node_id
        self.lat = lat
        self.lon = lon
        self.is_attraction = is_attraction  # 是否为景点（而非道路转弯点）
    
    def __str__(self):
        return f"Node({self.id}, {self.lat}, {self.lon}, {self.is_attraction})"
    
class Road:
    """道路类，表示两个节点之间的连接"""
    def __init__(self, road_id: str, node1_id: str, node2_id: str, 
                 length: float, density: float = 0.0, is_blocked: bool = False):
        self.id = road_id
        self.node1_id = node1_id
        self.node2_id = node2_id
        self.length = length  # 道路长度（米）
        self.density = density  # 游客密度（0-1之间的值）
        self.is_blocked = is_blocked  # 是否被湖水隔开
    
    def __str__(self):
        return f"Road({self.id}, {self.node1_id}-{self.node2_id}, {self.length}m, density: {self.density}, blocked: {self.is_blocked})"
    
class ScenicAreaGraph:
    """景区图类，包含所有节点和道路"""
    def __init__(self):
        self.nodes: Dict[str, Node] = {}  # 节点ID到节点的映射
        self.roads: Dict[str, Road] = {}  # 道路ID到道路的映射
        self.adjacency_list: Dict[str, List[Tuple[str, str]]] = {}  # 邻接表：节点ID -> [(相邻节点ID, 道路ID)]
    
    def add_node(self, node: Node):
        """添加节点"""
        self.nodes[node.id] = node
        if node.id not in self.adjacency_list:
            self.adjacency_list[node.id] = []
    
    def add_road(self, road: Road):
        """添加道路"""
        self.roads[road.id] = road
        
        # 更新邻接表
        if road.node1_id in self.adjacency_list and road.node2_id in self.nodes:
            self.adjacency_list[road.node1_id].append((road.node2_id, road.id))
        
        if road.node2_id in self.adjacency_list and road.node1_id in self.nodes:
            self.adjacency_list[road.node2_id].append((road.node1_id, road.id))
    
    def get_neighbors(self, node_id: str) -> List[Tuple[str, str]]:
        """获取节点的所有邻居节点及连接的道路ID"""
        return self.adjacency_list.get(node_id, [])
    
    def get_road(self, road_id: str) -> Optional[Road]:
        """根据道路ID获取道路"""
        return self.roads.get(road_id)
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """根据节点ID获取节点"""
        return self.nodes.get(node_id)

def calculate_attraction_distances():
    """
    计算所有景点之间的实际距离（通过道路网络），使用缓存
    
    返回:
        字典，键为(景点1, 景点2)，值为最短距离
    """
    global attraction_distances_cache
    
    # 如果已经有缓存，直接返回
    if attraction_distances_cache is not None:
        return attraction_distances_cache
    
    distances = {}
    attractions = [node_id for node_id, node in graph.nodes.items() if node.is_attraction]
    
    # 对每对景点计算最短路径距离
    for i, start_id in enumerate(attractions):
        # 使用Dijkstra算法计算从当前景点到所有其他景点的最短距离
        open_set = []
        heapq.heappush(open_set, (0, start_id))
        
        g_score = {start_id: 0}
        
        while open_set:
            current_dist, current_id = heapq.heappop(open_set)
            
            # 如果是景点，记录距离
            if current_id in attractions and current_id != start_id:
                distances[(start_id, current_id)] = current_dist
            
            # 探索邻居
            for neighbor_id, road_id in graph.get_neighbors(current_id):
                road = graph.get_road(road_id)
                if not road or road.is_blocked:
                    continue
                
                # 计算到邻居的距离
                tentative_g_score = g_score.get(current_id, float('inf')) + road.length
                
                # 如果找到更短路径，更新
                if tentative_g_score < g_score.get(neighbor_id, float('inf')):
                    g_score[neighbor_id] = tentative_g_score
                    heapq.heappush(open_set, (tentative_g_score, neighbor_id))
    
    # 缓存结果
    attraction_distances_cache = distances
    return distances

# 计算观光车时间表（考虑实际路径距离）
def calculate_tour_bus_schedule():
    """
    计算观光车时间表（考虑实际路径距离）
    
    返回:
        字典，键为(路线ID, 站点名称)，值为观光车到达时间的列表
    """
    global tour_bus_schedule
    schedule = {}
    
    # 计算所有景点之间的实际距离
    attraction_distances = calculate_attraction_distances()
    
    # 计算每条路线的时间表
    for route_id, route in TOUR_BUS_ROUTES.items():
        # 计算路线中相邻站点之间的实际距离
        route_distances = []
        for i in range(len(route) - 1):
            from_stop = route[i]
            to_stop = route[i + 1]
            
            # 获取两个景点之间的实际距离
            distance = attraction_distances.get((from_stop, to_stop), float('inf'))
            if distance == float('inf'):
                # 如果直接距离不存在，尝试反向
                distance = attraction_distances.get((to_stop, from_stop), float('inf'))
            
            route_distances.append(distance)
        
        # 计算该路线的车辆数量（平均分配）
        buses_per_route = TOUR_BUS_COUNT // len(TOUR_BUS_ROUTES)
        
        # 为每辆车计算时间表
        for bus_index in range(buses_per_route):
            # 计算发车时间（从8:00开始，每隔10分钟发一辆）
            departure_minutes = bus_index * 10
            hours = TOUR_BUS_SCHEDULE_START + departure_minutes // 60
            minutes = departure_minutes % 60
            
            # 确保时间在营业时间内
            if hours < TOUR_BUS_SCHEDULE_END:
                departure_time = datetime.datetime.combine(
                    datetime.date.today(), 
                    datetime.time(hours, minutes)
                )
                
                current_time = departure_time
                direction = 1  # 1表示正向，-1表示反向
                
                # 模拟车辆运行直到结束时间
                while current_time.time() < datetime.time(TOUR_BUS_SCHEDULE_END, 0):
                    # 正向行驶
                    if direction == 1:
                        for i in range(len(route)):
                            stop = route[i]
                            key = (route_id, stop)
                            
                            if key not in schedule:
                                schedule[key] = []
                            
                            # 记录到达时间（第一站就是发车时间）
                            if i > 0:
                                # 计算行驶时间 = 距离 / 速度
                                travel_time = route_distances[i-1] / TOUR_BUS_SPEED
                                current_time += datetime.timedelta(seconds=travel_time)
                            
                            # 检查是否超过结束时间
                            if current_time.time() >= datetime.time(TOUR_BUS_SCHEDULE_END, 0):
                                break
                                
                            schedule[key].append(current_time)
                            
                            # 添加停靠时间
                            current_time += datetime.timedelta(seconds=TOUR_BUS_STOP_TIME)
                            
                            # 检查是否超过结束时间
                            if current_time.time() >= datetime.time(TOUR_BUS_SCHEDULE_END, 0):
                                break
                    
                    # 反向行驶
                    else:
                        for i in range(len(route)-1, -1, -1):
                            stop = route[i]
                            key = (route_id, stop)
                            
                            if key not in schedule:
                                schedule[key] = []
                            
                            # 记录到达时间
                            if i < len(route) - 1:
                                # 计算行驶时间 = 距离 / 速度
                                travel_time = route_distances[i] / TOUR_BUS_SPEED
                                current_time += datetime.timedelta(seconds=travel_time)
                            
                            # 检查是否超过结束时间
                            if current_time.time() >= datetime.time(TOUR_BUS_SCHEDULE_END, 0):
                                break
                                
                            schedule[key].append(current_time)
                            
                            # 添加停靠时间
                            current_time += datetime.timedelta(seconds=TOUR_BUS_STOP_TIME)
                            
                            # 检查是否超过结束时间
                            if current_time.time() >= datetime.time(TOUR_BUS_SCHEDULE_END, 0):
                                break
                    
                    # 切换方向
                    direction *= -1
                    
                    # 检查是否超过结束时间
                    if current_time.time() >= datetime.time(TOUR_BUS_SCHEDULE_END, 0):
                        break
    
    # 对每个站点的时间进行排序
    for key in schedule:
        schedule[key].sort()
    
    tour_bus_schedule = schedule
    return schedule

# 获取下一班观光车信息
def get_next_tour_bus(stop_id: str, current_time: datetime.datetime = None, route_id: int = None) -> Tuple[datetime.datetime, int, int, int]:
    """
    获取下一班观光车的信息
    
    参数:
        stop_id: 站点ID
        current_time: 当前时间，如果为None则使用系统当前时间
        route_id: 指定路线ID，如果为None则返回所有路线中最快的一班
    
    返回:
        (下一班观光车到达时间, 需要等待的秒数, 票价, 路线ID)
        如果没有观光车，返回(None, -1, 0, 0)
    """
    if current_time is None:
        current_time = datetime.datetime.now()
    
    best_time = None
    best_wait = float('inf')
    best_fare = 0
    best_route = 0
    
    # 查找所有经过该站点的路线
    for key, times in tour_bus_schedule.items():
        r_id, stop = key
        if stop == stop_id and (route_id is None or r_id == route_id):
            # 找到下一班车
            for bus_time in times:
                if bus_time > current_time:
                    wait_time = (bus_time - current_time).total_seconds()
                    
                    # 计算票价（基础票价 + 每站加价）
                    fare = TOUR_BUS_BASE_FARE
                    
                    # 确定站点在路线中的位置
                    if stop_id in TOUR_BUS_ROUTES[r_id]:
                        fare += TOUR_BUS_PER_STOP_FARE * TOUR_BUS_ROUTES[r_id].index(stop_id)
                    
                    # 如果这是更早的班次，更新最佳选择
                    if wait_time < best_wait:
                        best_time = bus_time
                        best_wait = wait_time
                        best_fare = fare
                        best_route = r_id
                    
                    break  # 只取第一班符合条件的车
    
    if best_time is None:
        return None, -1, 0, 0
    
    return best_time, best_wait, best_fare, best_route

# 计算乘坐观光车到下一站的时间（考虑实际路径距离）
def calculate_tour_bus_time(from_stop: str, to_stop: str, board_time: datetime.datetime, route_id: int) -> Tuple[float, float]:
    """
    计算乘坐观光车从一站到另一站的时间和票价（考虑实际路径距离）
    
    参数:
        from_stop: 起始站点
        to_stop: 目标站点
        board_time: 上车时间
        route_id: 路线ID
    
    返回:
        (旅行时间(秒), 票价)
    """
    # 检查路线是否存在
    if route_id not in TOUR_BUS_ROUTES:
        return float('inf'), 0
    
    route = TOUR_BUS_ROUTES[route_id]
    
    # 检查站点是否在路线上
    if from_stop not in route or to_stop not in route:
        return float('inf'), 0
    
    # 计算所有景点之间的实际距离
    attraction_distances = calculate_attraction_distances()
    
    # 检查站点顺序
    from_index = route.index(from_stop)
    to_index = route.index(to_stop)
    
    # 计算正向行驶时间
    if from_index < to_index:
        travel_time = 0
        for i in range(from_index, to_index):
            # 获取两个景点之间的实际距离
            distance = attraction_distances.get((route[i], route[i+1]), float('inf'))
            if distance == float('inf'):
                # 如果直接距离不存在，尝试反向
                distance = attraction_distances.get((route[i+1], route[i]), float('inf'))
            
            travel_time += distance / TOUR_BUS_SPEED
            
            # 添加停靠时间（除了起始站）
            if i > from_index:
                travel_time += TOUR_BUS_STOP_TIME
        
        # 计算票价
        fare = TOUR_BUS_BASE_FARE + TOUR_BUS_PER_STOP_FARE * (to_index - from_index)
        
        return travel_time, fare
    
    # 计算反向行驶时间
    elif from_index > to_index:
        travel_time = 0
        for i in range(from_index, to_index, -1):
            # 获取两个景点之间的实际距离
            distance = attraction_distances.get((route[i], route[i-1]), float('inf'))
            if distance == float('inf'):
                # 如果直接距离不存在，尝试反向
                distance = attraction_distances.get((route[i-1], route[i]), float('inf'))
            
            travel_time += distance / TOUR_BUS_SPEED
            
            # 添加停靠时间（除了起始站）
            if i < from_index:
                travel_time += TOUR_BUS_STOP_TIME
        
        # 计算票价
        fare = TOUR_BUS_BASE_FARE + TOUR_BUS_PER_STOP_FARE * (from_index - to_index)
        
        return travel_time, fare
    
    # 同一站点
    else:
        return 0, 0

#游船部分
def get_next_boat_time(current_time: datetime.datetime = None) -> Tuple[datetime.datetime, int]:
    """
    获取下一班游船的时间和等待时间
    
    参数:
        current_time: 当前时间，如果为None则使用系统当前时间
    
    返回:
        (下一班游船时间, 需要等待的秒数)
        如果当前时间已超过最后一班船，返回(None, -1)
    """
    if current_time is None:
        current_time = datetime.datetime.now()
    
    # 获取今天的日期
    today = current_time.date()
    
    # 创建游船时间表
    boat_times = []
    for hour in range(BOAT_SCHEDULE_START, BOAT_SCHEDULE_END + 1):
        boat_time = datetime.datetime.combine(today, datetime.time(hour, 0))
        boat_times.append(boat_time)
    
    # 找到下一班船
    for boat_time in boat_times:
        # 船会等待10分钟
        departure_time = boat_time + datetime.timedelta(seconds=BOAT_WAIT_TIME)
        
        # 如果当前时间在船到达和离开之间
        if current_time < departure_time:
            wait_time = max(0, (boat_time - current_time).total_seconds())
            return boat_time, wait_time
    
    # 如果所有船都已离开
    return None, -1

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算两个经纬度坐标之间的哈弗辛距离（米）"""
    # 将角度转换为弧度
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    # 哈弗辛公式
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return EARTH_RADIUS * c

def calculate_speed(density: float, base_speed: float = 1.4) -> float:
    """
    根据游客密度计算实际步行速度
    
    参数:
        density: 游客密度 (0-1)
        base_speed: 基础步行速度 (米/秒)，默认1.4m/s (约5公里/小时)
    
    返回:
        实际步行速度 (米/秒)
    """
    # 简单模型：密度越高，速度越慢
    # 当密度为0时，速度为base_speed
    # 当密度为1时，速度降低为base_speed的0.3倍
    return base_speed * (1 - 0.7 * density)

def heuristic(graph: ScenicAreaGraph, node1_id: str, node2_id: str, cost_type: str) -> float:
    """
    启发式函数：估计从节点1到节点2的代价
    
    参数:
        graph: 景区图
        node1_id: 起始节点ID
        node2_id: 目标节点ID
        cost_type: 代价类型，"distance"或"time"
    
    返回:
        估计代价
    """
    node1 = graph.get_node(node1_id)
    node2 = graph.get_node(node2_id)
    
    if not node1 or not node2:
        return float('inf')
    
    # 计算直线距离
    distance = haversine_distance(node1.lat, node1.lon, node2.lat, node2.lon)
    
    if cost_type == "distance":
        return distance
    else:  # time
        # 使用最大速度（密度为0时的速度）来估计时间
        max_speed = calculate_speed(0)
        return distance / max_speed if max_speed > 0 else float('inf')

def reconstruct_path(came_from: Dict[str, Tuple[str, str]], current_id: str) -> List[str]:
    """
    从came_from字典重构路径
    
    参数:
        came_from: 记录每个节点的前驱节点和道路的字典
        current_id: 终点节点ID
    
    返回:
        路径节点ID列表
    """
    path = [current_id]
    while current_id in came_from:
        current_id, _ = came_from[current_id]
        path.append(current_id)
    path.reverse()
    return path


def a_star_shortest_path(graph: ScenicAreaGraph, start_id: str, end_id: str) -> Tuple[List[str], float]:
    """
    使用A*算法寻找最短路径（基于距离）
    
    参数:
        graph: 景区图
        start_id: 起点节点ID
        end_id: 终点节点ID
    
    返回:
        (路径节点ID列表, 总距离) 或 ([], float('inf')) 如果找不到路径
    """
    return a_star(graph, start_id, end_id, cost_type="distance")

def a_star_fastest_path(graph: ScenicAreaGraph, start_id: str, end_id: str, 
                       start_time: datetime.datetime = None, use_tour_bus: bool = False) -> Tuple[List[str], float, List[Tuple[str, datetime.datetime]], float]:
    """
    使用A*算法寻找最快路径（基于时间，考虑游客密度、游船和观光车）
    
    参数:
        graph: 景区图
        start_id: 起点节点ID
        end_id: 终点节点ID
        start_time: 出发时间，如果为None则使用当前时间
        use_tour_bus: 是否使用观光车
    
    返回:
        (路径节点ID列表, 总时间, 交通信息列表, 总票价) 
        交通信息格式: (道路ID或"TOUR_BUS", 乘车时间)
    """
    return a_star(graph, start_id, end_id, cost_type="time", start_time=start_time, use_tour_bus=use_tour_bus)

def a_star(graph: ScenicAreaGraph, start_id: str, end_id: str, cost_type: str = "distance", 
          start_time: datetime.datetime = None, use_tour_bus: bool = False) -> Tuple[List[str], float, List[Tuple[str, datetime.datetime]], float]:
    """
    A*算法实现
    
    参数:
        graph: 景区图
        start_id: 起点节点ID
        end_id: 终点节点ID
        cost_type: 代价类型，"distance"或"time"
        start_time: 出发时间，仅对cost_type="time"有效
        use_tour_bus: 是否使用观光车
    
    返回:
        (路径节点ID列表, 总代价, 交通信息列表, 总票价) 或 ([], float('inf'), [], 0) 如果找不到路径
        交通信息格式: (道路ID或"TOUR_BUS", 乘车时间)
    """
    # 验证起点和终点是否为景点
    start_node = graph.get_node(start_id)
    end_node = graph.get_node(end_id)
    if not start_node or not end_node or not start_node.is_attraction or not end_node.is_attraction:
        return [], float('inf'), [], 0
    
    # 对于最短路径，不需要考虑时间
    if cost_type == "distance":
        # 开放列表：存储待探索的节点 (f_score, node_id)
        open_set = []
        heapq.heappush(open_set, (0, start_id))
        
        # 从起点到每个节点的实际代价
        g_score = {start_id: 0}
        
        # 从起点经过每个节点到终点的估计代价
        f_score = {start_id: heuristic(graph, start_id, end_id, cost_type)}
        
        # 记录路径
        came_from = {}
        
        while open_set:
            # 获取f_score最小的节点
            current_f, current_id = heapq.heappop(open_set)
            
            # 如果到达终点，重构路径
            if current_id == end_id:
                path = reconstruct_path(came_from, current_id)
                total_cost = g_score[current_id]
                return path, total_cost, [], 0
            
            # 探索当前节点的所有邻居
            for neighbor_id, road_id in graph.get_neighbors(current_id):
                road = graph.get_road(road_id)
                if not road:
                    continue
                
                # 计算从当前节点到邻居节点的代价
                if cost_type == "distance":
                    cost = road.length
                else:  # time
                    speed = calculate_speed(road.density)
                    cost = road.length / speed if speed > 0 else float('inf')
                
                # 计算从起点到邻居节点的总代价
                tentative_g_score = g_score.get(current_id, float('inf')) + cost
                
                # 如果找到更优路径，更新信息
                if tentative_g_score < g_score.get(neighbor_id, float('inf')):
                    came_from[neighbor_id] = (current_id, road_id)
                    g_score[neighbor_id] = tentative_g_score
                    f_score[neighbor_id] = tentative_g_score + heuristic(graph, neighbor_id, end_id, cost_type)
                    
                    # 如果邻居节点不在开放列表中，加入开放列表
                    if neighbor_id not in [node_id for _, node_id in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor_id], neighbor_id))
        
        return [], float('inf'), [], 0
    
    else:  # 最快路径，考虑游船和观光车
        # 设置出发时间
        if start_time is None:
            start_time = datetime.datetime.now()
        
        # 开放列表：存储待探索的节点 (f_score, node_id, current_time, transport_info, total_fare)
        open_set = []
        heapq.heappush(open_set, (0, start_id, start_time, [], 0))
        
        # 从起点到每个节点的实际代价
        g_score = {start_id: 0}
        
        # 从起点经过每个节点到终点的估计代价
        f_score = {start_id: heuristic(graph, start_id, end_id, cost_type)}
        
        # 记录路径和到达时间
        came_from = {}
        arrival_time = {start_id: start_time}
        
        while open_set:
            # 获取f_score最小的节点
            current_f, current_id, current_time, current_transport_info, current_fare = heapq.heappop(open_set)
            
            # 如果到达终点，重构路径
            if current_id == end_id:
                path = reconstruct_path(came_from, current_id)
                total_time = (current_time - start_time).total_seconds()
                return path, total_time, current_transport_info, current_fare
            
            # 探索当前节点的所有邻居
            for neighbor_id, road_id in graph.get_neighbors(current_id):
                road = graph.get_road(road_id)
                if not road:
                    continue
                
                # 计算从当前节点到邻居节点的代价
                if road.is_blocked:
                    # 处理游船道路
                    next_boat_time, wait_time = get_next_boat_time(current_time)
                    
                    # 如果没有船了，跳过这条路
                    if next_boat_time is None:
                        continue
                    
                    # 计算航行时间
                    sail_time = road.length / BOAT_SPEED
                    
                    # 总时间 = 等待时间 + 航行时间
                    cost = wait_time + sail_time
                    
                    # 到达邻居节点的时间
                    neighbor_time = current_time + datetime.timedelta(seconds=cost)
                    
                    # 更新交通信息
                    new_transport_info = current_transport_info + [(road_id, next_boat_time)]
                    
                    # 票价不变
                    new_fare = current_fare
                else:
                    # 普通道路
                    speed = calculate_speed(road.density)
                    cost = road.length / speed if speed > 0 else float('inf')
                    
                    # 到达邻居节点的时间
                    neighbor_time = current_time + datetime.timedelta(seconds=cost)
                    
                    # 交通信息不变
                    new_transport_info = current_transport_info
                    
                    # 票价不变
                    new_fare = current_fare
                
                # 计算从起点到邻居节点的总代价
                tentative_g_score = g_score.get(current_id, float('inf')) + cost
                
                # 如果找到更优路径，更新信息
                if tentative_g_score < g_score.get(neighbor_id, float('inf')):
                    came_from[neighbor_id] = (current_id, road_id)
                    g_score[neighbor_id] = tentative_g_score
                    arrival_time[neighbor_id] = neighbor_time
                    f_score[neighbor_id] = tentative_g_score + heuristic(graph, neighbor_id, end_id, cost_type)
                    
                    # 在A*算法中，修改观光车处理部分
                    # 如果使用观光车，考虑乘坐观光车的选项
                    if use_tour_bus and current_id in [stop for (_, stop) in tour_bus_schedule.keys()]:
                        # 获取所有可能的路线
                        possible_routes = set()
                        for (r_id, stop) in tour_bus_schedule.keys():
                            if stop == current_id:
                                possible_routes.add(r_id)
                        
                        # 考虑每条路线
                        for r_id in possible_routes:
                            # 获取下一班观光车信息
                            next_bus_time, wait_time, fare_to_next, bus_route_id = get_next_tour_bus(current_id, current_time, r_id)
                            
                            if next_bus_time is not None:
                                # 考虑乘坐观光车到所有可能的站点
                                for possible_stop in TOUR_BUS_ROUTES[r_id]:
                                    if possible_stop == current_id:
                                        continue
                                    
                                    # 计算乘坐观光车的时间和票价
                                    bus_time, bus_fare = calculate_tour_bus_time(current_id, possible_stop, next_bus_time, r_id)
                                    
                                    if bus_time < float('inf'):
                                        # 总时间 = 等待时间 + 乘车时间
                                        total_bus_time = wait_time + bus_time
                                        
                                        # 到达目标站点的时间
                                        bus_arrival_time = current_time + datetime.timedelta(seconds=total_bus_time)
                                        
                                        # 更新交通信息
                                        bus_transport_info = current_transport_info + [(f"TOUR_BUS_{r_id}", next_bus_time)]
                                        
                                        # 更新票价
                                        bus_total_fare = current_fare + bus_fare
                                        
                                        # 计算总代价
                                        bus_tentative_g_score = g_score.get(current_id, float('inf')) + total_bus_time
                                        
                                        # 如果找到更优路径，更新信息
                                        if bus_tentative_g_score < g_score.get(possible_stop, float('inf')):
                                            came_from[possible_stop] = (current_id, f"TOUR_BUS_{r_id}")
                                            g_score[possible_stop] = bus_tentative_g_score
                                            arrival_time[possible_stop] = bus_arrival_time
                                            f_score[possible_stop] = bus_tentative_g_score + heuristic(graph, possible_stop, end_id, cost_type)
                                            
                                            # 加入开放列表
                                            heapq.heappush(open_set, (f_score[possible_stop], possible_stop, bus_arrival_time, bus_transport_info, bus_total_fare))
                    
                    # 如果邻居节点不在开放列表中，加入开放列表
                    found = False
                    for i, (f, nid, t, ti, tf) in enumerate(open_set):
                        if nid == neighbor_id:
                            found = True
                            if tentative_g_score < g_score.get(neighbor_id, float('inf')):
                                open_set[i] = (f_score[neighbor_id], neighbor_id, neighbor_time, new_transport_info, new_fare)
                                heapq.heapify(open_set)
                            break
                    
                    if not found:
                        heapq.heappush(open_set, (f_score[neighbor_id], neighbor_id, neighbor_time, new_transport_info, new_fare))
        
        return [], float('inf'), [], 0

def print_path(graph: ScenicAreaGraph, path: List[str], total_cost: float, cost_type: str, 
              transport_info: List[Tuple[str, datetime.datetime]] = None, total_fare: float = 0):
    """
    打印路径详情
    
    参数:
        graph: 景区图
        path: 路径节点ID列表
        total_cost: 总代价
        cost_type: 代价类型，"distance"或"time"
        transport_info: 交通信息列表
        total_fare: 总票价
    """
    if not path:
        print("无法找到路径")
        return
    
    print(f"路径找到! 总{'距离' if cost_type == 'distance' else '时间'}: {total_cost:.2f}{'米' if cost_type == 'distance' else '秒'}")
    
    if total_fare > 0:
        print(f"总票价: {total_fare:.2f}元")
    
    # 打印交通信息
    if transport_info:
        print("交通信息:")
        for transport_id, transport_time in transport_info:
            if transport_id.startswith("TOUR_BUS_"):
                route_id = int(transport_id.split("_")[2])
                print(f"  乘坐{route_id}号线观光车, 发车时间: {transport_time.strftime('%H:%M:%S')}")
            else:
                road = graph.get_road(transport_id)
                if road and road.is_blocked:
                    print(f"  乘坐游船通过道路 {transport_id} ({road.node1_id}-{road.node2_id}), 发船时间: {transport_time.strftime('%H:%M:%S')}")
    
    print("路径详情:")
    
    for i in range(len(path) - 1):
        node1_id = path[i]
        node2_id = path[i + 1]
        
        # 查找连接两个节点的道路
        road = None
        road_id = None
        for neighbor_id, r_id in graph.get_neighbors(node1_id):
            if neighbor_id == node2_id:
                road = graph.get_road(r_id)
                road_id = r_id
                break
        
        if road:
            node1 = graph.get_node(node1_id)
            node2 = graph.get_node(node2_id)
            is_attraction1 = "景点" if node1.is_attraction else "转弯点"
            is_attraction2 = "景点" if node2.is_attraction else "转弯点"
            
            if cost_type == "distance":
                print(f"  {node1_id} ({is_attraction1}) -> {node2_id} ({is_attraction2}): {road.length:.2f}米")
            else:
                if road.is_blocked:
                    print(f"  {node1_id} ({is_attraction1}) -> {node2_id} ({is_attraction2}): 乘船 {road.length:.2f}米 (船速: {BOAT_SPEED:.2f}米/秒)")
                else:
                    speed = calculate_speed(road.density)
                    time_cost = road.length / speed if speed > 0 else float('inf')
                    print(f"  {node1_id} ({is_attraction1}) -> {node2_id} ({is_attraction2}): {time_cost:.2f}秒 (密度: {road.density:.2f}, 速度: {speed:.2f}米/秒)")
        
        # 检查是否是观光车路段
        elif any(transport_id.startswith("TOUR_BUS") for transport_id, _ in transport_info if transport_id.startswith("TOUR_BUS")):
            # 这是一个观光车路段
            route_id = int([tid for tid, _ in transport_info if tid.startswith("TOUR_BUS")][0].split("_")[2])
            print(f"  {node1_id} -> {node2_id}: 乘坐{route_id}号线观光车 (车速: {TOUR_BUS_SPEED:.2f}米/秒)")

def visualize_graph(graph: ScenicAreaGraph, shortest_path: List[str] = None, fastest_path: List[str] = None):
    """
    可视化景区图和路径
    
    参数:
        graph: 景区图
        shortest_path: 最短路径节点ID列表
        fastest_path: 最快路径节点ID列表
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制所有道路
    for road_id, road in graph.roads.items():
        node1 = graph.get_node(road.node1_id)
        node2 = graph.get_node(road.node2_id)
        
        if node1 and node2:
            # 根据道路是否被阻断选择不同的线型和颜色
            if road.is_blocked:
                ax.plot([node1.lon, node2.lon], [node1.lat, node2.lat], 
                        'b--', alpha=0.5, linewidth=2, label='游船路线' if road_id == list(graph.roads.keys())[0] else "")
            else:
                # 根据密度调整颜色
                color = plt.cm.RdYlGn_r(road.density)  # 密度越高颜色越红
                ax.plot([node1.lon, node2.lon], [node1.lat, node2.lat], 
                        color=color, alpha=0.5, linewidth=2)
    
    # 绘制观光车路线
    colors = ['m', 'c', 'y', 'k']  # 不同路线的颜色
    for route_id, route in TOUR_BUS_ROUTES.items():
        color = colors[(route_id - 1) % len(colors)]
        for i in range(len(route) - 1):
            node1 = graph.get_node(route[i])
            node2 = graph.get_node(route[i + 1])
            if node1 and node2:
                ax.plot([node1.lon, node2.lon], [node1.lat, node2.lat], 
                        color=color, alpha=0.7, linewidth=3, 
                        label=f'观光车路线{route_id}' if i == 0 else "")
    
    # 绘制所有节点
    attraction_lons = []
    attraction_lats = []
    attraction_labels = []
    
    turning_point_lons = []
    turning_point_lats = []
    
    dock_lons = []  # 码头经度
    dock_lats = []  # 码头纬度
    
    for node_id, node in graph.nodes.items():
        if node.is_attraction:
            attraction_lons.append(node.lon)
            attraction_lats.append(node.lat)
            attraction_labels.append(node_id)
        elif "码头" in node_id:
            dock_lons.append(node.lon)
            dock_lats.append(node.lat)
        else:
            turning_point_lons.append(node.lon)
            turning_point_lats.append(node.lat)
    
    # 绘制转弯点（小灰点）
    ax.scatter(turning_point_lons, turning_point_lats, c='gray', s=20, alpha=0.5, label='转弯点')
    
    # 绘制码头（黄色点）
    ax.scatter(dock_lons, dock_lats, c='yellow', s=80, alpha=0.7, label='码头')
    
    # 绘制景点（大点，带标签）
    ax.scatter(attraction_lons, attraction_lats, c='blue', s=100, alpha=0.7, label='景点')
    
    # 添加景点标签
    for i, label in enumerate(attraction_labels):
        ax.annotate(label, (attraction_lons[i], attraction_lats[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=12)
    
    # 绘制最短路径（如果提供）
    if shortest_path and len(shortest_path) > 1:
        path_lons = []
        path_lats = []
        for node_id in shortest_path:
            node = graph.get_node(node_id)
            if node:
                path_lons.append(node.lon)
                path_lats.append(node.lat)
        
        ax.plot(path_lons, path_lats, 'r-', linewidth=3, label='最短路径')
    
    # 绘制最快路径（如果提供）
    if fastest_path and len(fastest_path) > 1:
        path_lons = []
        path_lats = []
        for node_id in fastest_path:
            node = graph.get_node(node_id)
            if node:
                path_lons.append(node.lon)
                path_lats.append(node.lat)
        
        ax.plot(path_lons, path_lats, 'g-', linewidth=3, label='最快路径')
    
    # 添加颜色条表示密度
    norm = Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('游客密度 (0-1)')
    
    # 添加图例和标题
    ax.legend(loc='best')
    ax.set_title('景区地图和路线规划')
    ax.set_xlabel('经度')
    ax.set_ylabel('纬度')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 示例用法
def main():
    # 创建景区图
    global graph
    graph = ScenicAreaGraph()
    
    # 添加景点节点
    attractions = [
        Node("入口", 31.2300, 121.4700, True),
        Node("花园", 31.2320, 121.4750, True),
        Node("湖泊", 31.2350, 121.4800, True),
        Node("观景台", 31.2380, 121.4850, True),
        Node("餐厅", 31.2400, 121.4900, True),
        Node("博物馆", 31.2420, 121.4950, True),
        Node("出口", 31.2450, 121.5000, True),
        Node("湖心岛", 31.2370, 121.4830, True),  # 新增湖心岛景点
    ]
    
    # 添加道路转弯点
    turning_points = [
        Node("T1", 31.2310, 121.4720, False),
        Node("T2", 31.2330, 121.4770, False),
        Node("T3", 31.2360, 121.4820, False),
        Node("T4", 31.2390, 121.4870, False),
        Node("T5", 31.2410, 121.4920, False),
        Node("T6", 31.2430, 121.4970, False),
        Node("码头1", 31.2345, 121.4810, False),  # 新增码头
        Node("码头2", 31.2375, 121.4840, False),  # 新增码头
    ]
    
    # 添加所有节点
    for node in attractions + turning_points:
        graph.add_node(node)
    
    # 添加道路
    roads = [
        Road("R1", "入口", "T1", 200, 0.1, False),
        Road("R2", "T1", "花园", 150, 0.3, False),
        Road("R3", "花园", "T2", 100, 0.5, False),
        Road("R4", "T2", "T3", 180, 0.2, False),
        Road("R5", "T3", "湖泊", 120, 0.4, False),
        Road("R6", "湖泊", "T4", 160, 0.6, False),
        Road("R7", "T4", "观景台", 140, 0.7, False),
        Road("R8", "观景台", "T5", 130, 0.8, False),
        Road("R9", "T5", "餐厅", 110, 0.9, False),
        Road("R10", "餐厅", "T6", 170, 0.5, False),
        Road("R11", "T6", "博物馆", 150, 0.3, False),
        Road("R12", "博物馆", "出口", 190, 0.2, False),
        Road("R13", "T2", "T5", 300, 0.4, False),  # 捷径
        
        # 新增游船道路
        Road("B1", "湖泊", "码头1", 50, 0.0, False),
        Road("B2", "码头1", "码头2", 400, 0.0, True),  # 被湖水隔开，需要游船
        Road("B3", "码头2", "湖心岛", 50, 0.0, False),
        Road("B4", "湖心岛", "观景台", 300, 0.0, True),  # 被湖水隔开，需要游船
    ]
    
    for road in roads:
        graph.add_road(road)
    
    # 计算观光车时间表
    calculate_tour_bus_schedule()
    
    # 打印观光车时间表示例
    print("=== 观光车时间表示例 ===")
    for (route_id, stop), times in list(tour_bus_schedule.items())[:5]:  # 只打印前5个
        print(f"路线{route_id} - {stop}: {[t.strftime('%H:%M:%S') for t in times[:3]]}...")  # 只打印前3个时间
    
    # 测试不使用观光车的最快路径
    print("\n=== 不使用观光车的最快路径 (基于时间，考虑游客密度和游船) ===")
    start, end = "入口", "湖心岛"
    test_time = datetime.datetime.now().replace(hour=9, minute=0, second=0)
    
    fastest_path, total_time, transport_info, total_fare = a_star_fastest_path(graph, start, end, test_time, False)
    print_path(graph, fastest_path, total_time, "time", transport_info, total_fare)
    
    # 测试使用观光车的最快路径
    print("\n=== 使用观光车的最快路径 (基于时间，考虑游客密度、游船和观光车) ===")
    fastest_path_bus, total_time_bus, transport_info_bus, total_fare_bus = a_star_fastest_path(graph, start, end, test_time, True)
    print_path(graph, fastest_path_bus, total_time_bus, "time", transport_info_bus, total_fare_bus)
    
    # 比较两种方案
    print("\n=== 方案比较 ===")
    print(f"不使用观光车: 时间 {total_time:.2f}秒, 票价 {total_fare:.2f}元")
    print(f"使用观光车: 时间 {total_time_bus:.2f}秒, 票价 {total_fare_bus:.2f}元")
    
    if total_time_bus < total_time:
        time_saved = total_time - total_time_bus
        print(f"使用观光车可以节省 {time_saved:.2f}秒, 但需要多支付 {total_fare_bus - total_fare:.2f}元")
    else:
        print("不使用观光车更快")
    
    # 可视化
    visualize_graph(graph, None, fastest_path_bus)

if __name__ == "__main__":
    main()