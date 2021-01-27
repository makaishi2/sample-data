# 共通関数定義

# テスト用
SAMPLE = 'abc123'

import pip, site, importlib
def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])

install('japanize_matplotlib')
install('torchviz')
importlib.reload(site) 

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torchviz import make_dot
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets

# 損失関数値計算用
def eval_loss(loader, device, net, criterion):
  
  # DataLoaderから最初の1セットを取得する
  images, labels = next(iter(loader))

  # デバイスの割り当て
  inputs = images.to(device)
  labels = labels.to(device)

  # 予測値の計算
  outputs = net(inputs)

  #  損失値の計算
  loss = criterion(outputs, labels)

  return loss
  
# 学習用関数
def fit(net, optimizer, criterion, device, num_epochs, train_loader, test_loader, history):

  base_epochs = len(history)
  
  for epoch in range(base_epochs, num_epochs+base_epochs):
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0

    #訓練フェーズ
    net.train()
    count = 0

    for inputs, labels in train_loader:
      count += len(labels)
      inputs = inputs.to(device)
      labels = labels.to(device)

      # 勾配の初期化
      optimizer.zero_grad()

      # 順伝搬計算
      outputs = net(inputs)

      # 誤差計算
      loss = criterion(outputs, labels)
      train_loss += loss.item()

      # 勾配計算
      loss.backward()

      # 重み変更
      optimizer.step()

      #予測値算出
      predicted = torch.max(outputs, 1)[1]

      #正解件数算出
      train_acc += (predicted == labels).sum()

    # 訓練データに対する損失と精度の計算
    avg_train_loss = train_loss / count
    avg_train_acc = train_acc / count

    #予測フェーズ
    net.eval()
    count = 0

    for inputs, labels in test_loader:
      with torch.no_grad():
        count += len(labels)

        inputs = inputs.to(device)
        labels = labels.to(device)

        # 順伝搬計算
        outputs = net(inputs)

        # 誤差計算
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        #予測値算出
        predicted = torch.max(outputs, 1)[1]

        #正解件数算出
        val_acc += (predicted == labels).sum()

      # 検証データに対する損失と精度の計算
      avg_val_loss = val_loss / count
      avg_val_acc = val_acc / count
    
    print (f'Epoch [{(epoch+1)}/{num_epochs+base_epochs}], loss: {avg_train_loss:.5f} acc: {avg_train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}')
    item = np.array([epoch+1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc])
    history = np.vstack((history, item))
  return history

# 学習ログ解析

def evaluate_history(history):
  #損失関数値と精度の確認
  print(f'初期状態: 損失関数: {history[0,3]:.5f} 精度: {history[0,4]:.5f}') 
  print(f'最終状態: 損失関数: {history[-1,3]:.5f} 精度: {history[-1,4]:.5f}' )

  num_epochs = len(history)
  unit = num_epochs / 10

  # 学習曲線の表示 (損失関数)
  plt.plot(history[:,0], history[:,1], 'b', label='訓練')
  plt.plot(history[:,0], history[:,3], 'k', label='検証')
  plt.xticks(np.arange(0,num_epochs+1, unit))
  plt.xlabel('繰り返し回数')
  plt.ylabel('損失関数値')
  plt.title('学習曲線(損失関数)')
  plt.legend()
  plt.show()

  # 学習曲線の表示 (精度)
  plt.plot(history[:,0], history[:,2], 'b', label='訓練')
  plt.plot(history[:,0], history[:,4], 'k', label='検証')
  plt.xticks(np.arange(0,num_epochs+1,unit))
  plt.xlabel('繰り返し回数')
  plt.ylabel('精度')
  plt.title('学習曲線(精度)')
  plt.legend()
  plt.show()
  
# 予測結果表示
def show_predict_result(net, loader, classes, plt):

  # DataLoaderから最初の1セットを取得する
  images, labels = next(iter(loader))

  # デバイスの割り当て
  inputs = images.to(device)
  labels = labels.to(device)

  # 予測値の計算
  outputs = net(inputs)
  predicted = torch.max(outputs,1)[1]
  images = images.to('cpu')

  # 最初の100個の表示
  plt.figure(figsize=(15, 20))
  for i in range(100):
    ax = plt.subplot(10, 10, i + 1)
    image = images[i].numpy()
    label_name = classes[labels[i]]
    predicted_name = classes[predicted[i]]
    img = np.transpose(image, (1, 2, 0))
    img2 = (img + 1)/2 
    plt.imshow(img2)
    if label_name == predicted_name:
      c = 'k'
    else:
      c = 'b'
    ax.set_title(label_name + ':' + predicted_name, c=c)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()  
