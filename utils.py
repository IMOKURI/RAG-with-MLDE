import math
import time


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%d分 %d秒" % (m, s)


def time_since(since):
    now = time.time()
    s = now - since
    return f"処理時間: {as_minutes(s)}"
