
def slope(x1, y1, x2, y2):
    if (x2 - x1 != 0):
        return (float)(y2 - y1) / (x2 - x1)
    return sys.maxint
def intercept(x1,y1,x2,y2):
    y_intercept=y1- (slope(x1, y1, x2, y2)*x1)
    return y_intercept

def pointIsOnLine(x1, y1,x2, y2,x,y):
    if (y == ((slope(x1, y1, x2, y2) * x) + intercept(x1,y1,x2,y2))):
        return True;

    return False;

# driver code
x1 = 36
y1 = 497
x2 = 187
y2 = 580
x=94
y=491
print("Slope is:", slope(x1, y1, x2, y2))
print("intercept:",intercept(x1,y1,x2,y2))
print("pointison",pointIsOnLine(x1, y1,x2, y2,x,y))