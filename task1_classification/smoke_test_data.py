from dataloaders import get_pneumoniamnist_loaders

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_pneumoniamnist_loaders(
        data_root="./data",
        batch_size=64,
        num_workers=2
    )

    x, y = next(iter(train_loader))
    print("batch x:", x.shape, x.dtype, x.min().item(), x.max().item())
    print("batch y:", y.shape, y.dtype, y[:10].view(-1).tolist())

    print("num train batches:", len(train_loader))
    print("num val batches:", len(val_loader))
    print("num test batches:", len(test_loader))
