import itertools
#Vars
origin = {'x':0, 'y':0}

def slope(origin, target):
    if target['x'] == origin['x']:
        return 0
    else:
        m = (target['y'] - origin['y']) / (target['x'] - origin['x'])
        return m

def line_eqn(origin, target):
    x = origin['x']
    y = origin['y']
    c = -(slope(origin, target)*x - y)
    c = y - (slope(origin, target)*x)
    #return 'y = ' + str(slope(target)) + 'x + ' + str(c)
    m = slope(origin, target)
    return {'m':m, 'c':c}

def get_y(x, slope, c):
    # y = mx + c    
    y = (slope*x) + c
    return y

def get_x(y, slope, c):     
    #x = (y-c)/m
    if slope == 0:
        c = 0   #vertical lines never intersect with y-axis
    if slope == 0:
        slope = 1   #Do NOT divide by zero
    x = (y - c)/slope
    return x

def get_points(origin, target):
    coord_list = []
    #Step along x-axis
    for i in range(origin['x'], target['x']+1):     
        eqn = line_eqn(origin, target)
        y = get_y(i, eqn['m'], eqn['c'])        
        coord_list.append([i, y])

    #Step along y-axis
    for i in range(origin['y'], target['y']+1):
        eqn = line_eqn(origin, target)
        x = get_x(i, eqn['m'], eqn['c'])
        coord_list.append([x, i])

    #return unique list     
    return list(map(lambda x: [int(x[0]), int(x[1])],
                    [k for k, _ in itertools.groupby(sorted(coord_list))]))

origin = {'x':93, 'y':476}
target = {'x':539, 'y':723}



def remove_duplicates(lst):
    # 展開二維陣列成一維
    flattened_lst = [tuple(x) for x in lst]
    # 用 set 去除重複值
    unique_lst = list(set(flattened_lst))
    # 將展開的一維陣列轉回原本的二維陣列形式
    result = [list(x) for x in unique_lst]
    return sorted(result)

print(remove_duplicates(get_points(origin, target)))

def is_in_list(coord_list, target_coord):
    for coord in coord_list:
        if coord[0] == target_coord[0] and coord[1] == target_coord[1]:
            return True
    return False


