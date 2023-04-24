import torch

def testDeconvolution():
    # Input
    input = torch.randn(1, 1, 3, 3)
    # Deconvolution
    deconv = torch.nn.ConvTranspose2d(1, 1, 3, stride=1, padding=1)
    # Output
    output = deconv(input)
    print(output)
    return

def testConvolution():
    # Input
    input = torch.randn(1, 1, 3, 3)
    # Convolution
    conv = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
    # Output
    output = conv(input)
    print(output)
    return

def testFullyConnected():
    # Input
    input = torch.randn(1, 1)
    # Fully Connected
    fc = torch.nn.Linear(1, 1)
    # Output
    output = fc(input)
    print(output)
    return

def testPooling():
    # Input
    input = torch.randn(1, 1, 3, 3)
    # Pooling
    pool = torch.nn.MaxPool2d(3, stride=1, padding=1)
    # Output
    output = pool(input)
    print(output)
    return

def test():
    print("Testing")
    print("Deconvolution")
    testDeconvolution()
    print("Convolution")
    testConvolution()
    print("Fully Connected")
    testFullyConnected()
    print("Pooling")
    testPooling()
    return

test()
