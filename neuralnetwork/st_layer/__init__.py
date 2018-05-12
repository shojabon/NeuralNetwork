class st_layer:

    def __init__(self, size, f_type, stabilizer):
        self.size = size
        self.type = f_type
        self.stabilizer = stabilizer

    def get_size(self):
        return self.size

    def get_type(self):
        return self.type

    def get_stabilizer(self):
        return self.stabilizer