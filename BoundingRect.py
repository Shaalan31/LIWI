class BoundingRect:
    def __init__(self, height, width, rect):
        self.height = height
        self.width = width
        self.rect = rect

    @property
    def x(self):
        return self.rect
