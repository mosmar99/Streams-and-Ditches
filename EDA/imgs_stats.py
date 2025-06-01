im

def load_and_process_data():
    def read_file_list(file_path):
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    test_fnames_path = './data/mapio_folds/f1/test_files.dat'
    train_fnames_path = './data/mapio_folds/f1/train_files.dat'
    val_fnames_path = './data/mapio_folds/f1/val_files.dat'
    print("Beginning to load data...")
    train_files = read_file_list(train_fnames_path)
    test_files = read_file_list(test_fnames_path)
    val_files = read_file_list(val_fnames_path)

    image_folder = './data/05m_chips/slope/'
    label_folder = './data/05m_chips/labels/'
    train_dataset = UNetDataset(train_files, image_folder, label_folder)
    test_dataset = UNetDataset(test_files, image_folder, label_folder)
    val_dataset = UNetDataset(val_files, image_folder, label_folder)
    print(f"Created datasets with {len(train_dataset)} train, {len(test_dataset)} test, {len(val_dataset)} val samples.")

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=15, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=15, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=15, pin_memory=True)
    print(f"Created DataLoaders with batch size {batch_size}")

    return train_loader, test_loader, val_loader, train_dataset, test_dataset, val_dataset

train_loader, test_loader, val_loader, train_dataset, test_dataset, val_dataset = load_and_process_data()
