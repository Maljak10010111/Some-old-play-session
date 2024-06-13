import torch
from torchvision import datasets, transforms

if __name__ == '__main__':
    # transformation for cifar10
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL images to tensors
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.2154, 0.229))  # Normalize for CIFAR-10
    ])

    # loading cifar10
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # getting tensors for inputs and corresponding labels
    def get_inputs_labels(dataset):
        inputs, labels = [], []
        for image, label in dataset:
            inputs.append(image)
            labels.append(label)
        return torch.stack(inputs), torch.tensor(labels)


    train_inputs, train_labels = get_inputs_labels(trainset)
    test_inputs, test_labels = get_inputs_labels(testset)
