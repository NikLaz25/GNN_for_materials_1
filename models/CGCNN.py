from __future__ import print_function, division

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    Определяет слой сверточной нейронной сети для обработки атомных признаков и их соседей.
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters:
        ----------
        atom_fea_len (int): Number of atom hidden features. Размерность вектора признаков атома
        nbr_fea_len (int): Number of bond features. Размерность вектора признаков соседей
        """
        super(ConvLayer, self).__init__()

        self.atom_fea_len = atom_fea_len # Размерность вектора признаков атома
        self.nbr_fea_len = nbr_fea_len # Размерность вектора признаков соседей
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len) # Полносвязный слой для объединения признаков атома и его соседей
        
        self.sigmoid = nn.Sigmoid() # Функция активации сигмоид для фильтрации признаков       
        self.softplus1 = nn.Softplus() # Функция активации softplus для обработки признаков

        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len) # Нормализация батча для стабилизации обучения
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len) # Нормализация батча для выходных признаков
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass  Прямой проход данных через слой.

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape  # N - число атомов, M - число соседей для каждого атома
        
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :] # Извлечение признаков соседних атомов по индексам
        
        # Объединение признаков центрального атома, его соседей и их взаимодействий
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2) 
        total_gated_fea = self.fc_full(total_nbr_fea) # Пропуск через полносвязный слой
        
        # Применение нормализации и активации
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2) # Разделение на две части
        nbr_filter = self.sigmoid(nbr_filter) # Фильтр признаков
        nbr_core = self.softplus1(nbr_core) # Основные признаки
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed) # Финальная активация и обновление признаков атома
        return out


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    Определяет всю архитектуру сверточной нейронной сети на графах для предсказания свойств кристаллов.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False):
        """
        Initialize CrystalGraphConvNet. Инициализация сети.

        Parameters
        ----------
        orig_atom_fea_len: int
          Number of atom features in the input. Исходная размерность вектора признаков атома.
        nbr_fea_len: int
          Number of bond features. Количество элементов связи.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers. Количество скрытых элементов атома в сверточных слоях
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        
        # Линейный слой для преобразования исходных признаков атома в скрытое пространство
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        # Создание списка сверточных слоев
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        # Линейный слой для преобразования после свертки перед полносвязными слоями
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        
        # Создание полносвязных слоев и функций активации для них
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        # Выходной слой: для классификации - 2 нейрона, для регрессии - 1 нейрон
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        
        # Если задача классификации, добавляем слой Dropout для регуляризации
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass  Прямой проход данных через модель.

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx   Отображение из crystal idx в atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        # Преобразование исходных признаков атомов в эмбеддинги
        atom_fea = self.embedding(atom_fea)

        # Последовательное применение сверточных слоев
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        # Агрегация признаков атомов для каждого кристалла
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        
        # Преобразование после свертки перед полносвязными слоями
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        
        # Если задача классификации, применяем Dropout перед выходным слоем
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        
        # Последовательное применение полносвязных слоев с функцией активации Softplus
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        
        # Выходной слой для получения предсказаний
        out = self.fc_out(crys_fea)
        
        # Если задача классификации, применяем логарифмическую софтмакс-функцию для получения лог-вероятностей классов
        if self.classification:
            out = self.logsoftmax(out)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features / Агрегация (пулинг) признаков атомов для каждого кристалла.

        N: Total number of atoms in the batch 
        N0: Total number of crystals in the batch 

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        
        # Инициализация списка для хранения агрегированных признаков каждого кристалла
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        # Объединение списка в тензор
        return torch.cat(summed_fea, dim=0)
