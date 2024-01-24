import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


#import metric from yggdrasil
os.system("/home/yannh/Documents/uni/phd/synch_metrics.sh")




metric_path ="/home/yannh/Documents/uni/phd/river_ml/metrics/tiramisu_htira-bclf_regular_trainingset_kepler_v16rho1_5to20/tiramisu_htira-bclf_regular.pkl"
metric_path ="/home/yannh/Documents/uni/phd/river_ml/metrics/tiramisu_mtira-bclf_regular_trainingset_kepler_v16rho1_5to20/tiramisu_mtira-bclf_regular.pkl"
metric_path ="/home/yannh/Documents/uni/phd/river_ml/metrics/tiramisu_stira-bclf_regular_trainingset_kepler_v16rho1_5to20/tiramisu_stira-bclf_regular.pkl"

with open(metric_path, "rb") as f:
    metrics = pd.read_pickle(f)

n_metric = len(metrics)


data = { 
    "epoch": np.empty(n_metric),
    "epoch_completion_ratio": np.empty(n_metric),
    "n_batch_eval": np.empty(n_metric), 
    "train_loss": np.empty(n_metric),
    "val_loss": np.empty(n_metric),
    "train_img_cm": np.empty((n_metric, 3,3)),
    "train_pixel_cm": np.empty((n_metric, 3,3)),
    "val_img_cm": np.empty((n_metric, 3,3)),
    "val_pixel_cm": np.empty((n_metric, 3,3)),
    "train_img_cm2d": np.empty((n_metric, 2,2)),
    "train_pixel_cm2d": np.empty((n_metric, 2,2)),
    "val_img_cm2d": np.empty((n_metric, 2,2)),
    "val_pixel_cm2d": np.empty((n_metric, 2,2))
}


cm_names = [
    "train_img", 
    "train_pixel",
    "val_img",
    "val_pixel"
    ]

metric_names = [
    "accuracy",
    "recall",
    "precision"
]


class_ids = {
    "noise": 0,
    "resonance": 1,
    "trojan": 2
}

class_ids_2d = {
    "noise": 0,
    "all": 1
}

for cm_name in cm_names:
    for metric_name in metric_names:
        for class_id in class_ids.keys():
            if metric_name == "accuracy":
                data.update({f"{cm_name}_{metric_name}" : np.empty(n_metric)})
            else:
                data.update({f"{cm_name}_{class_id}_{metric_name}" : np.empty(n_metric)})
        for class_id in class_ids_2d.keys():
            if metric_name == "accuracy":
                data.update({f"{cm_name}2d_{metric_name}" : np.empty(n_metric)})
            else:
                data.update({f"{cm_name}2d_{class_id}_{metric_name}" : np.empty(n_metric)})
                
                
    
    
    
def reduce_to_2classes(cm):
    new_cm = np.empty((2,2), dtype=int)
    new_cm[0,0] = cm[0,0]
    new_cm[1,1] = cm[1:,1:].sum()
    new_cm[0,1] = cm[0,1:].sum()
    new_cm[1,0] = cm[1:,0].sum()
    return new_cm

def get_accuracy(cm):
    return np.diag(cm).sum()/cm.sum()
    
def get_recall(cm, class_id):
    return cm[class_id,class_id]/cm[class_id,:].sum()

def get_precision(cm, class_id):
    return cm[class_id,class_id]/cm[:,class_id].sum()

def sliding_mean(data, window_size=10):
    result = np.empty(data.size - window_size)
    for i in range(data.size - window_size):
        result[i] = np.mean(data[i:i+window_size])
    return result

def slope(data):
    return (np.roll(data,1,0) - data)[1:]


def plot_data_and_slope(metric, window_size, skip_first):
    fig, ax = plt.subplots(4, 1, figsize=figsize_four)
    if "_" in metric:
        class_name, metric_name = metric.split("_")
    else:
        class_name, metric_name = "NONE", metric
    fig.suptitle(metric_name)
    d1 = data[f"train_img2d_{metric}"]
    d2 = data[f"val_img2d_{metric}"] 
    d3 = data[f"train_pixel2d_{metric}"] 
    d4 = data[f"val_pixel2d_{metric}"]
    ax[0].plot(epoch, d1, label="training")
    ax[0].plot(epoch, d2, label="validation")
    ax[0].set(xlabel="epoch", ylabel=metric_name, title=f"img {metric_name} of class {class_name}")
    ax[1].plot(epoch, d3, label="training")
    ax[1].plot(epoch, d4, label="validation")
    ax[1].set(xlabel="epoch", ylabel=metric_name, title=f"pixel {metric_name} of class {class_name}")

    ax[2].plot(epoch[1+skip_first:-window_size], slope(sliding_mean(d1, window_size=window_size))[skip_first:], label="training")
    ax[2].plot(epoch[1+skip_first:-window_size], slope(sliding_mean(d2, window_size=window_size))[skip_first:], label="validation")
    ax[2].set(xlabel="epoch", ylabel=f"{metric_name} slope (smoothed)", title=f"img {metric_name} slope plot of class {class_name}")
    ax[3].plot(epoch[1+skip_first:-window_size], slope(sliding_mean(d3, window_size=window_size))[skip_first:], label="training")
    ax[3].plot(epoch[1+skip_first:-window_size], slope(sliding_mean(d4, window_size=window_size))[skip_first:], label="validation")
    ax[3].set(xlabel="epoch", ylabel=f"{metric_name} slope (smoothed)", title=f"pixel {metric_name} slope plot of class {class_name}")

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()
    plt.tight_layout()
    plt.show(block=False)




for i_m, m in enumerate(metrics):
    for k, v in m.items():
        data[k][i_m] = v   
        
    for name in cm_names:
        cm = m[f"{name}_cm"]
        cm2d = reduce_to_2classes(cm)
        data[f"{name}_cm2d"][i_m] = cm2d
        data[f"{name}_accuracy"][i_m] = get_accuracy(cm)
        data[f"{name}2d_accuracy"][i_m] = get_accuracy(cm2d)
        for k,v in class_ids.items():
            data[f"{name}_{k}_recall"][i_m] = get_recall(cm, v)
            data[f"{name}_{k}_precision"][i_m] = get_precision(cm, v)
        for k, v in class_ids_2d.items():
            data[f"{name}2d_{k}_recall"][i_m] = get_recall(cm2d, v)
            data[f"{name}2d_{k}_precision"][i_m] = get_precision(cm2d, v)


epoch = data["epoch"] + data["epoch_completion_ratio"]


figsize_single= (8, 3)
figsize_double= (8, 6)
figsize_four = (8, 9)

# loss plot
fig, ax = plt.subplots(1, 1, figsize=figsize_single)
fig.suptitle("loss")
ax.plot(epoch, data["train_loss"], label="training")
ax.plot(epoch, data["val_loss"], label="validation")
ax.set(xlabel="epoch", ylabel="loss", title="loss plot")
ax.legend()
plt.tight_layout()
plt.show(block=False)




# accuracy plot
window_size = 20
skip_first = 10

plot_data_and_slope("accuracy", window_size, skip_first)

plot_data_and_slope("all_recall", window_size, skip_first)

plot_data_and_slope("all_precision", window_size, skip_first)

# fig, ax = plt.subplots(4, 1, figsize=figsize_four)
# fig.suptitle("accuracy")
# ax[0].plot(epoch, data["train_img2d_accuracy"], label="training")
# ax[0].plot(epoch, data["val_img2d_accuracy"], label="validation")
# ax[0].set(xlabel="epoch", ylabel="accuracy", title="img accuracy plot noise vs all")
# ax[1].plot(epoch, data["train_pixel2d_accuracy"], label="training")
# ax[1].plot(epoch, data["val_pixel2d_accuracy"], label="validation")
# ax[1].set(xlabel="epoch", ylabel="accuracy", title="pixel accuracy plot noise vs all")

# ax[2].plot(epoch[1+skip_first:-window_size], slope(sliding_mean(data["train_img2d_accuracy"], window_size=window_size))[skip_first:], label="training")
# ax[2].plot(epoch[1+skip_first:-window_size], slope(sliding_mean(data["val_img2d_accuracy"], window_size=window_size))[skip_first:], label="validation")
# ax[2].set(xlabel="epoch", ylabel="accuracy slope (smoothed)", title="img accuracy slope plot noise vs all")
# ax[3].plot(epoch[1+skip_first:-window_size], slope(sliding_mean(data["train_pixel2d_accuracy"], window_size=window_size))[skip_first:], label="training")
# ax[3].plot(epoch[1+skip_first:-window_size], slope(sliding_mean(data["val_pixel2d_accuracy"], window_size=window_size))[skip_first:], label="validation")
# ax[3].set(xlabel="epoch", ylabel="accuracy slope (smoothed)", title="pixel accuracy slope plot noise vs all")

# ax[0].legend()
# ax[1].legend()
# ax[2].legend()
# ax[3].legend()
# plt.tight_layout()
# plt.show(block=False)

# all class recall
fig, ax = plt.subplots(2, 1, figsize=figsize_double)
fig.suptitle("recall 'all'")
ax[0].plot(epoch, data["train_img2d_all_recall"], label="training")
ax[0].plot(epoch, data["val_img2d_all_recall"], label="validation")
ax[0].set(xlabel="epoch", ylabel="recall class 'all'", title="img recall plot class 'all'")
ax[1].plot(epoch, data["train_pixel2d_all_recall"], label="training")
ax[1].plot(epoch, data["val_pixel2d_all_recall"], label="validation")
ax[1].set(xlabel="epoch", ylabel="recall class 'all'", title="pixel recall plot class 'all'")
ax[0].legend()
ax[1].legend()
plt.tight_layout()
plt.show(block=True)


