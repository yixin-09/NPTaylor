import math

def TwoSum(a,b):
    x = a + b
    z = x - a
    y = (a-(x-z))+(b-z)
    return x,y

def Split(a):
    z = a * (134217728.0 + 1.0)
    x = z - (z - a)
    y = a - x
    return x,y


def TwoPro(a,b):
    x = a * b
    ah,al = Split(a)
    bh,bl = Split(b)
    y = al * bl - (((x-ah*bh)-al*bh)-ah*bl)
    return x,y
