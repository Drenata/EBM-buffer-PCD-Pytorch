import torch


class SampleReplayBuffer:
    def __init__(self, batch_size, buffer_length=10000, data_size=(3, 32, 32)):
        self.buffer_length = buffer_length
        self.data_size = data_size
        self.buffer = torch.rand((self.buffer_length,) + self.data_size)
        self.index = 0
        self.batch_size = batch_size
        self.cpu = torch.device("cpu")
        self.gpu = torch.device("cuda:0")

    def sample(self):
        indices = torch.randint(low=0, high=self.buffer_length, size=(self.batch_size,))
        return self.buffer[indices].to(self.gpu, non_blocking=True)

    def add_sample(self, x):
        if self.index + self.batch_size >= self.buffer_length:
            end = self.buffer_length - self.index
            self.buffer[self.index : self.index + end] = x[:end].to(
                self.cpu, non_blocking=True
            )

            start = self.batch_size - end
            self.buffer[:start] = x[end:].to(self.cpu, non_blocking=True)
        else:
            self.buffer[self.index : self.index + self.batch_size] = x.to(
                self.cpu, non_blocking=True
            )

        self.index = (self.index + self.batch_size) % self.buffer_length
