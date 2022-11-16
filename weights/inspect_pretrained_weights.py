import torch

def main():
    f = "https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-101.pkl"

    loaded_weights = torch.load(f, map_location=torch.device("cpu"))

    return

if __name__=="__main__":
    main()