import torchvision.datasets as dset


def main():
    dataset = dset.ImageFolder(root='imgs/lfw')
    print(len(dataset))
    # print(dataset[:5]) -- not working
    print(type(dataset.imgs))
    print(len(dataset.imgs))
    print(dataset.imgs[:5])
    

if __name__ == '__main__':
    main()
