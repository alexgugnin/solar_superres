if __name__ == '__main__':
    
    import torch
    from pathlib import Path
    from torch.utils.data import DataLoader
    from torch import nn
    from PIL import Image
    from torchvision import transforms
    import time
    from solarDataset import SolarDataset
    from customUnet import Unet2D
    from train_and_valid_steps import train_step, valid_step
    from normalizations.maxTransform import MaxNorm
    from normalizations.compression import ArcsinCompression, HyperbTangNorm, LogTransform


    # Setup path to data folder
    data_path = Path("/mnt/ceph/users/ogugnin/171_tensors_only/")

    '''
    Initialising train, validation and test paths. Split:
    Jpeg: 3193 + 347 + 288
    FITS: 5224(-3 missing data) + 581
    '''
    resol = 128
    train_dir = data_path / "train"
    valid_dir = data_path / "validation"
    #test_dir = data_path / "test"
    #temp_dir = data_path / "temp"

    #DEFINING GENERAL TRANSFORM (for jpeg inputs also totensor) 
    
    #Normalization coefs are different for every resolution
    #For 64x
    '''
    general_transforms = transforms.Compose([
        transforms.Normalize(mean=[166.80511, 221.54995, 77.97677],
                             std=[158.16921, 195.92302, 83.03113])
    ])
    '''
    #For 128x
    '''
    general_transforms = transforms.Compose([
        transforms.Normalize(mean=[166.80272437858724, 221.55462074754726, 77.97706133759476],
                             std=[158.825408299861, 196.41179470392694, 83.21854877978049])
    ])
    '''
    general_transforms = transforms.Compose([
        transforms.Normalize(mean=[166.80272437858724, 166.80272437858724, 166.80272437858724],
                             std=[158.825408299861, 158.825408299861, 158.825408299861])
    ])
    '''
    #For 256x
    
    general_transforms = transforms.Compose([
        transforms.Normalize(mean=[166.8967504236785, 221.6779214974206, 78.02248786538688],
                             std=[159.20750361849758, 196.72687920578056, 83.33513308905512])
    ])
    
    general_transforms = transforms.Compose([
        transforms.Normalize(mean=[166.8967504236785, 166.8967504236785, 166.8967504236785],
                             std=[159.20750361849758, 159.20750361849758, 159.20750361849758])
    ])
    '''
    '''
    general_transforms = transforms.Compose([
        MaxNorm(max_val = 9656.9769)
    ])
    '''
    '''
    comp_factor = 0.1
    general_transforms = transforms.Compose([
        ArcsinCompression(div_factor = comp_factor)
    ])
    '''
    
    '''
    #Hyperbolic Tangent Normalization (Tanh Estimator)
    general_transforms = transforms.Compose([
        HyperbTangNorm(mean=[166.8967504236785, 221.6779214974206, 78.02248786538688],
                             std=[159.20750361849758, 196.72687920578056, 83.33513308905512])
    ])
    '''
    '''
    general_transforms = transforms.Compose([
        LogTransform()
    ])
    '''
    train_data = SolarDataset(targ_dir=train_dir, res = resol, transform = general_transforms)
    valid_data = SolarDataset(targ_dir=valid_dir, res = resol, transform = general_transforms)
    #test_data = SolarDataset(targ_dir=test_dir, transform = general_transforms)
    #temp_data = SolarDataset(targ_dir=temp_dir, transform = general_transforms)

    BATCH_SIZE = 2

    train_dataloader = DataLoader(dataset=train_data,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_data,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False)
    '''
    test_dataloader = DataLoader(dataset=test_data,
                                        batch_size=1,
                                        shuffle=False)
    '''                
    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    torch.manual_seed(42)

    #model = Unet2D(in_channels = 1, out_channels = 1).to(device)
    model = Unet2D(in_channels = 3, out_channels = 3).to(device)
    epochs = 30
    train_progress = []
    validation_progress = []

    start_time = time.time()
    loss_func = nn.MSELoss()
    lr = 1e-3
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    start_width = 128
    blocks = 3
    min_val = 0.1
    for epoch in range(epochs):
        #print(f"Epoch {epoch}\n")

        train_progress.append(
            train_step(data_loader = train_dataloader,
                    model = model,
                    loss_func = loss_func,
                    optimizer = optimizer,
                    device=device
                    )
            )

        validation_progress.append(
            valid_step(data_loader = valid_dataloader,
                    model = model,
                    loss_func = loss_func,
                    device=device
                    )
        )
        scheduler.step(validation_progress[-1])
        if validation_progress[-1] <= min_val:
            torch.save(model.state_dict(), f"solarResolution{resol}_{epoch}_only171_defnorm.pth")
            print(f"Saved loss is {validation_progress[-1]}")
            print(f"lr is {optimizer.state_dict}")
            min_val = validation_progress[-1]
    
    #Saving Final Result
    end_time = time.time()
    print(f"Total training time is {end_time - start_time} s.")

    torch.save(model.state_dict(), f"solarResolution{resol}_{epochs}epochs(final)_only171_defnorm.pth")
    with open(f'losses{resol}_{epochs}ep_only171_defnorm.txt', 'w') as outf:
        for (trainL, validL) in zip(train_progress, validation_progress):
            outf.write(f"{(trainL, validL)}\n")