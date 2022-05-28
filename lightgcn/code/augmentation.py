import register, world
from register import dataset
import utils
import torch
import numpy as np

def save_txt(data_dict, file_name):
    out = []
    for key in data_dict.keys():
        out.append(data_dict[key])
        out[-1].insert(0, key)
        
    textfile = open("augmentation/{}.txt".format(file_name), "w")
    for element in out:
        s = ' '.join(element)
        textfile.write(s + "\n")
    textfile.close()

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
weight_file = utils.getFileName()
Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
world.cprint(f"loaded model weights from {weight_file}")

# weak users
weak_num = 25
weak_users = [i for i in range(len(dataset.allPos)) if len(dataset.allPos[i]) <= weak_num]
allPos = dataset.getUserPosItems(weak_users)
batch_users_gpu = torch.Tensor(weak_users).long()
batch_users_gpu = batch_users_gpu.to(world.device)
rating = Recmodel.getUsersRating(batch_users_gpu)

exclude_index = []
exclude_items = []
for range_i, items in enumerate(allPos):
    exclude_index.extend([range_i] * len(items))
    exclude_items.extend(items)
rating[exclude_index, exclude_items] = -(1<<10)
rating = rating.detach().numpy() 
rating_index = np.argwhere(rating>0.99)
augmentation_dict = {}
for u, i in rating_index:
    user = str(weak_users[u])
    item = str(i)
    if augmentation_dict.get(user):
        augmentation_dict[user].append(item)
    else:
        augmentation_dict[user] = [item]

min = 9999
max = 0
for k in augmentation_dict.keys():
    count = len(augmentation_dict[k])
    if  count > max:
        max = count
    if count < min:
        min = count
print("max number: {}, min number: {}".format(max, min))
save_txt(augmentation_dict, "aug_25_99")

