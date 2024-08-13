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
    data_path = Path(...)

    '''
    Initialising train, validation and test paths.
    Split: 5224(-3 missing data) + 581
    '''
    resol = 128
    train_dir = data_path / "train"
    valid_dir = data_path / "validation"

    #DEFINING GENERAL TRANSFORM 
    
    #Normalization coefs are different for every resolution
    general_transforms_64 = transforms.Compose([
                            transforms.Normalize(mean=[166.80511, 166.80511, 166.80511],
                                                 std=[158.16921, 158.16921, 158.16921])
                            ])

    inv_transf_64 = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/158.16921, 1/158.16921, 1/158.16921 ]),
                                        transforms.Normalize(mean = [ -166.80511, -166.80511, -166.80511 ],
                                                     std = [ 1., 1., 1. ]),
                                        ])

    general_transforms_128 = transforms.Compose([
                            transforms.Normalize(mean=[166.80272437858724, 166.80272437858724, 166.80272437858724],
                                                 std=[158.825408299861, 158.825408299861, 158.825408299861])
                            ])

    inv_transf_128 = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/158.825408299861, 1/158.825408299861, 1/158.825408299861 ]),
                                         transforms.Normalize(mean = [ -166.80272437858724, -166.80272437858724, -166.80272437858724 ],
                                                     std = [ 1., 1., 1. ]),
                                        ])

    general_transforms_256 = transforms.Compose([
                            transforms.Normalize(mean=[166.8967504236785, 166.8967504236785, 166.8967504236785],
                                                 std=[159.20750361849758, 159.20750361849758, 159.20750361849758])
                                        ])

    inv_transf_256 = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/159.20750361849758, 1/159.20750361849758, 1/159.20750361849758 ]),
                                         transforms.Normalize(mean = [ -166.8967504236785, -166.8967504236785, -166.8967504236785 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

    train_data = SolarDataset(targ_dir=train_dir, res = resol, transform = general_transforms)
    valid_data = SolarDataset(targ_dir=valid_dir, res = resol, transform = general_transforms)

    BATCH_SIZE = 4

    train_dataloader = DataLoader(dataset=train_data,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_data,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False)
               
    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    torch.manual_seed(42)

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
            torch.save(model.state_dict(), f"...")
            print(f"Saved loss is {validation_progress[-1]}")
            print(f"lr is {optimizer.state_dict}")
            min_val = validation_progress[-1]
    
    #Saving Final Result
    end_time = time.time()
    print(f"Total training time is {end_time - start_time} s.")

    torch.save(model.state_dict(), f"...")
    with open(f'...', 'w') as outf:
        for (trainL, validL) in zip(train_progress, validation_progress):
            outf.write(f"{(trainL, validL)}\n")