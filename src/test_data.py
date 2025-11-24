from data_loader import create_datasets

train_ds, val_ds, test_ds = create_datasets(
    r"C:\Users\PC-STYLE\Desktop\SOD-Project\data\ECSSD\train",
    r"C:\Users\PC-STYLE\Desktop\SOD-Project\data\ECSSD\train_mask"
)

for images, masks in train_ds.take(1):
    print("Images batch:", images.shape)
    print("Masks batch:", masks.shape)
