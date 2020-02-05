import math

import tensorboardX
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from net import StandardBNCNN, StandardCNN
from sample_replay_buffer import SampleReplayBuffer


class SGLDTrainer:
    def __init__(self):
        self.batch_size = 128
        self.epochs = 500
        self.num_workers = 3
        self.smoothness_scale = 1.0

        transform = transforms.Compose([transforms.ToTensor()])
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
        self.testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        self.device = torch.device("cuda:0")
        self.model = StandardCNN()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, betas=[0.0, 0.999])
        self.step = 0

        self.buffer_sample_rate = 0.95
        self.data_size = (3, 32, 32)
        self.sample_replay_buffer = SampleReplayBuffer(
            self.batch_size, data_size=self.data_size
        )

        self.dynamics_steps = 60
        self.step_size = 10
        self.noise_scale = 0.005

        self.writer = tensorboardX.SummaryWriter()

    def train(self):

        for epoch in range(self.epochs):
            average_loss = 0.0
            for i, (x, _) in enumerate(self.trainloader, 0):
                output = self.process_batch(x)
                loss = output["loss"].item()
                average_loss += (loss - average_loss) / (i + 1)
                print(
                    f"[{i+1}/{len(self.trainloader)}] -- loss {loss:.3f} -- avg. loss {average_loss:.3f}",
                    end="       \r",
                )

                if self.step % 50 == 0:
                    self.log({"average_loss": average_loss, **output})

                self.step += 1

            print(f"\nEpoch {epoch+1} started...")
            torch.save(self.model.state_dict(), f"./model{epoch}.pt")

    def process_batch(self, x):

        x = x.to(self.device, non_blocking=True)

        sample = self.get_sample()

        self.optimizer.zero_grad()
        losses = self.compute_loss(positive_examples=x, negative_examples=sample)
        losses["loss"].backward()
        self.optimizer.step()

        return {"positive_examples": x, "negative_examples": sample, **losses}

    def compute_loss(self, positive_examples, negative_examples):

        positive_energy = self.model(positive_examples)
        negative_energy = self.model(negative_examples)

        maximum_likelihood_loss = positive_energy - negative_energy
        smoothness_loss = (
            positive_energy ** 2 + negative_energy ** 2
        ) * self.smoothness_scale
        total_loss = maximum_likelihood_loss + smoothness_loss

        return {
            "loss": total_loss.mean(),
            "positive_energy": positive_energy,
            "negative_energy": negative_energy,
            "maximum_likelihood_loss": maximum_likelihood_loss,
            "smoothness_loss": smoothness_loss,
        }

    def get_sample(self):
        sample = self.get_initial_sample()

        #self.model.eval()

        sample_optimizer = torch.optim.SGD([sample.requires_grad_()], lr=self.step_size)
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(sample_optimizer, max_lr=50.0, total_steps=self.dynamics_steps)

        for i in range(self.dynamics_steps):
            sample_optimizer.zero_grad()
            energy = self.model(sample)
            energy.backward(torch.ones_like(energy))
            torch.nn.utils.clip_grad_value_(sample, 0.01)
            sample_optimizer.step()
            #scheduler.step()
            noise = torch.empty_like(sample, dtype=torch.float32).normal_(
                0, self.noise_scale
            )
            with torch.no_grad():
                #sample.clamp_(0, 1)
                sample.add_(noise)
            if self.step % 50 == 0:
                print(energy.detach().mean(), torch.std_mean(sample.detach()))

        sample = sample.detach()
        self.sample_replay_buffer.add_sample(sample)
        #self.model.train()

        return sample

    def get_initial_sample(self):
        if torch.rand(1) > self.buffer_sample_rate:
            return torch.rand((self.batch_size,) + self.data_size, device=self.device)
        else:
            return self.sample_replay_buffer.sample()

    def log(self, data):

        self.writer.add_scalar("average loss", data["average_loss"], self.step)
        self.writer.add_scalar("batch loss", data["loss"], self.step)

        std, mean = torch.std_mean(data["positive_energy"])
        self.writer.add_scalar("positive energy mean", mean, self.step)
        self.writer.add_scalar("positive energy std", std, self.step)

        std, mean = torch.std_mean(data["negative_energy"])
        self.writer.add_scalar("negative energy mean", mean, self.step)
        self.writer.add_scalar("negative energy std", std, self.step)

        self.writer.add_scalar(
            "smoothness loss", data["smoothness_loss"].mean(), self.step
        )

        self.writer.add_images(
            "positive examples", data["positive_examples"], self.step
        )
        self.writer.add_images(
            "negative examples", data["negative_examples"], self.step
        )


if __name__ == "__main__":
    trainer = SGLDTrainer()
    trainer.train()
