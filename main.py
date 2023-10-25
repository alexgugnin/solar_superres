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
    torch.manual_seed(42)

    ''' HYPERPARAMS '''
    BATCH_SIZE = 4
    EPOCHS = 100
    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ''' Setup path to data folder '''
    data_path = Path("data/solar_superres256/")

    ''' Initialising train and validation paths '''
    train_dir = data_path / "train"
    valid_dir = data_path / "validation"

    ''' Defining a general transformation '''
    general_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    ''' Creating training and validation dataloaders '''
    train_data = SolarDataset(targ_dir=train_dir, transform = general_transforms)
    valid_data = SolarDataset(targ_dir=valid_dir, transform = general_transforms)

    train_dataloader = DataLoader(dataset=train_data,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True)
    valid_dataloader = DataLoader(dataset=train_data,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False)

    ''' Initialising the model '''
    model = Unet2D(in_channels = 3, out_channels = 3).to(device)

    ''' Model's training '''
    train_progress = []
    validation_progress = []
    start_time = time.time()
    for epoch in range(EPOCHS):

        train_progress.append(
            train_step(data_loader = train_dataloader,
                    model = model,
                    loss_func = nn.MSELoss(),
                    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4),
                    device=device
                    )
            )

        validation_progress.append(
            valid_step(data_loader = valid_dataloader,
                    model = model,
                    loss_func = nn.MSELoss(),
                    device=device
                    )
        )

    #Saving Final Result
    end_time = time.time()
    print(f"Total training time is {end_time - start_time} s.")
    torch.save(model.state_dict(), f"solarResolution_{EPOCHS}EPOCHS(final).pth")
    with open(f'losses_{EPOCHS}ep.txt', 'w') as outf:
        for (trainL, validL) in zip(train_progress, validation_progress):
            outf.write(f"{(trainL, validL)}\n")
