from torch.utils.data import DataLoader


def get_dataloader(dataset, cfg, transform):
    train_dataset = dataset(root=cfg.data_root, train=False, 
                                     download=True, transform=transform)
    test_dataset = dataset(root=cfg.data_root, train=False, 
                                    download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, 
                              shuffle=True, num_workers=cfg.num_workers,)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, 
                             shuffle=False, num_workers=cfg.num_workers,
                             drop_last=True)

    return train_loader, test_loader
