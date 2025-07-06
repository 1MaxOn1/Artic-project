from models.baseline import ModelArtic
from data.data_prepare import MakeTimeSeriesPerWeek
from data.dataset_arctic import IceDataset
from torch.utils.data import DataLoader
from utils.visualization import Visualize

temp_data = MakeTimeSeriesPerWeek(r'D:\Arctic_project\dataset\osisaf')
temp_data.load()
dataset_train = IceDataset(temp_data.x_train, temp_data.y_train)
dataset_val = IceDataset(temp_data.x_val, temp_data.y_val)
dataset_test = IceDataset(temp_data.x_test, temp_data.y_test)

dataloader_train = DataLoader(dataset_train, batch_size=8)
dataloader_val = DataLoader(dataset_val, batch_size=8)
dataloader_test = DataLoader(dataset_test, batch_size=8)

model = ModelArtic(c_in=48, c_out=24, num_layers=5)
# model.train_loop(dataloader_train=dataloader_train, dataloader_eval=dataloader_val)

model.load(r'D:\Arctic_project\model_weights\bets_model.pth')
model.test(dataloader_test=dataloader_test)

pred = model.prediction(temp_data.x_test[0])
y = temp_data.y_test[0]

vis = Visualize(num_weeks=5)
vis.visualize(pred, y)

# прогноз для 1 сепредины и конца
# графики прогноза на каждую неделю
# создать ветку с базовыми архитектурами файл с гибкой моделью гибкий размер изображения, количество каналов.