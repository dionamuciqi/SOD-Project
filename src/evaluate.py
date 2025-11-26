import tensorflow as tf
import matplotlib.pyplot as plt
from data_loader import create_datasets

model = tf.keras.models.load_model("sod_model.h5", compile=False)

def iou(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype("float32")
    inter = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - inter
    return float((inter + 1e-6) / (union + 1e-6))

def precision(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype("float32")
    TP = (y_true * y_pred).sum()
    FP = ((1 - y_true) * y_pred).sum()
    return float((TP + 1e-6) / (TP + FP + 1e-6))

def recall(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype("float32")
    TP = (y_true * y_pred).sum()
    FN = (y_true * (1 - y_pred)).sum()
    return float((TP + 1e-6) / (TP + FN + 1e-6))

def f1(prec, rec):
    return float(2 * (prec * rec) / (prec + rec + 1e-6))

_, _, test_ds = create_datasets(
    r"C:\Users\PC-STYLE\Desktop\SOD-Project\data\ECSSD\train",
    r"C:\Users\PC-STYLE\Desktop\SOD-Project\data\ECSSD\train_mask"
)

iou_total = prec_total = rec_total = f1_total = 0
n = 0

for img, mask in test_ds:
    pred = model.predict(img)
    i = iou(mask.numpy(), pred)
    p = precision(mask.numpy(), pred)
    r = recall(mask.numpy(), pred)
    f = f1(p, r)

    iou_total += i
    prec_total += p
    rec_total += r
    f1_total += f
    n += 1

print("FINAL METRICS ")
print("IoU:", iou_total / n)
print("Precision:", prec_total / n)
print("Recall:", rec_total / n)
print("F1 Score:", f1_total / n)

# Visualization
for img, mask in test_ds.take(1):
    pred = model.predict(img)[0]
    bin_pred = (pred > 0.5).astype("float32")

    plt.figure(figsize=(12,4))
    plt.subplot(1,4,1); plt.imshow(img[0]); plt.title("Input"); plt.axis("off")
    plt.subplot(1,4,2); plt.imshow(mask[0,...,0], cmap="gray"); plt.title("GT"); plt.axis("off")
    plt.subplot(1,4,3); plt.imshow(pred[...,0], cmap="gray"); plt.title("Prediction"); plt.axis("off")
    plt.subplot(1,4,4); plt.imshow(img[0]); plt.imshow(bin_pred[...,0], cmap="Reds", alpha=0.5); plt.title("Overlay"); plt.axis("off")
    plt.show()
