# Модуль с графовой нейронной сетью в виде class GCN(torch.nn.Module)

# librories Torch
import torch
import torch.nn as nn
from torch.nn import Linear

# librories Torch geometric
from torch_geometric.nn import PointTransformerConv, global_mean_pool

class GCN(torch.nn.Module):
    """
    Графовая нейронная сеть (GNN) для предсказания свойств материалов.
    Сочетает структурную информацию (атомы и их связи) с параметрами кристаллической решётки.

    Параметры:
        hyperparameters (dict): Словарь с гиперпараметрами модели, включая:
            - hidden_embeding (int): Размер скрытого векторного пространства.
        dataset
    """
    def __init__(self, hyperparameters, dataset):
        super(GCN, self).__init__()
        torch.manual_seed(12345)  # Фиксируем случайное начальное состояние для воспроизводимости

        # 1. Инициализация гиперпараметров
        self.hidden_embeding = hyperparameters['hidden_embeding']
        self.dataset = dataset

        # 2. Слои для обработки структурных данных (атомы и их связи)
        # PointTransformerConv - графовый слой, учитывающий позиции атомов
        self.structural_embedding = PointTransformerConv(
            in_channels=self.dataset.num_node_features,  # Количество признаков узла (например, 5)
            out_channels=self.hidden_embeding       # Размер выходного эмбеддинга
        )

        # 3. Слой для обработки параметров решётки
        # Линейный слой преобразует 9 параметров решётки (3x3 матрица) в скрытое пространство
        self.lattice_embedding = nn.Linear(9, self.hidden_embeding)

        # 4. Скрытые полносвязные слои для комбинирования признаков
        self.hidden_layers = nn.Sequential(
            # Первый слой: объединяет структурные и решёточные признаки
            Linear(self.hidden_embeding * 2, self.hidden_embeding * 3),
            nn.ReLU(inplace=True),  # Функция активации
            # Второй слой: сжатие признаков
            Linear(self.hidden_embeding * 3, self.hidden_embeding * 2),
            nn.ReLU(inplace=True),
            # Третий слой: подготовка к выходному слою
            Linear(self.hidden_embeding * 2, self.hidden_embeding),
            nn.ReLU(inplace=True)
        )

        # 5. Выходной слой (регрессия)
        self.out = Linear(self.hidden_embeding, 1)  # Предсказывает 1 значение (например, энергию)

    def forward(self, data):
        """
        Проход данных через модель.
        
        Параметры:
            data (torch_geometric.data.Data): Граф с полями:
                - x: Признаки узлов [num_nodes, num_features]
                - pos: Координаты атомов [num_nodes, 3]
                - edge_index: Индексы связей [2, num_edges]
                - lattice: Параметры решётки [batch_size, 9]
                - batch: Вектор батча [num_nodes]
                
        Возвращает:
            torch.Tensor: Предсказанное значение [batch_size, 1]
        """
        # 1. Структурный эмбеддинг (атомы + связи)
        # PointTransformerConv учитывает позиции атомов (data.pos) и связи (data.edge_index)
        x = self.structural_embedding(data.x, data.pos, data.edge_index)  # [num_nodes, hidden_embeding]

        # 2. Усреднение по графу (pooling)
        # Преобразует эмбеддинги узлов в один вектор на граф
        x = global_mean_pool(x, data.batch)  # [batch_size, hidden_embeding]

        # 3. Эмбеддинг решётки
        l = self.lattice_embedding(data.lattice)  # [batch_size, hidden_embeding]

        # 4. Объединение структурных и решёточных признаков
        x = torch.hstack([x, l])  # [batch_size, hidden_embeding * 2]

        # 5. Обработка через скрытые слои
        x = self.hidden_layers(x)  # [batch_size, hidden_embeding]

        # 6. Выходной слой
        x = self.out(x)  # [batch_size, 1]

        return x