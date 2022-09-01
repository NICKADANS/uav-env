# --------------------------------------------------------
# 兴趣点类
# --------------------------------------------------------

class PoI:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.done = 0  # 是否被采集

    def to_list(self):
        return [self.x, self.y, self.done]