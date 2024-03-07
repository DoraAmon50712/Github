def isBetween(ax,ay,bx,by,cx,cy):
    crossproduct = (cy - ay) * (bx - ax) - (cx - ax) * (by - ay)
    epsilon=0.000001
    # compare versus epsilon for floating point values, or != 0 if using integers
    if abs(crossproduct) > epsilon:
        return False

    dotproduct = (cy - ay) * (bx - ax) - (cx - ax) * (by - ay)
    if dotproduct < 0:
        return False

    squaredlengthba = (cy - ay) * (bx - ax) - (cx - ax) * (by - ay)
    if dotproduct > squaredlengthba:
        return False

    return True
ax,ay = (539,723)
bx,by = (93,476)
cx,cy = (96,478)
print(isBetween(ax,ay,bx,by,cx,cy))

def calculate_speed(last_x, new_y, fps, distance):
    """
    計算速度的函數
    :param last_x: 上一幀中車輛的 y 座標
    :param new_y: 當前幀中車輛的 y 座標
    :param fps: 幀率
    :param distance: 距離
    :return: 車速
    """
    speed = abs(new_y - last_x) * fps * distance / 1000 / 1000  # 公里/小時
    return speed

def is_point_on_line(x,y,x1,y1,x2,y2):
    A = y2-y1
    B = x1 - x2
    C = x2*y1- x1*y2

    return  abs(A*x+B*y+C) < 1e-9
x = 3
y = 2
x1 = 1
y1 = 1
x2 = 5
y2 = 3
print(is_point_on_line(x,y,x1,y1,x2,y2))
