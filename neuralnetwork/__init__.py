class Stemoid:

    def __init__(self, model):
        self.model = model

    def predict(self, input_data):
        return self.model.predict(input_data)

    def get_loss(self, input_data, lable):
        return self.model.get_loss(input_data, lable)

    def get_shape(self):
        return self.model.get_shape

    def get_accuracy(self, input_data, lable):
        return self.model.get_accuracy(input_data, lable)

    def learn(self, optimizer, input_data, lable):
        gradient = self.model.get_gradient(input_data, lable)
        optimizer.update(self.model.weights, gradient)


