# author: 
# contact: ycfrude@163.com
# datetime:2022/4/1 3:47 PM
import numpy as np


class ShuffleAmongSegments:
    """
    原始方法为shuffle within segments
    论文的shuffle是发生在不同的segment内部的，这里支持segment之间的顺序调换
    基本思路是：
        当shuffle rate 为<=0时，纯随机
        当shuffle rate >0 时， 采用 泊松分布 决定每个位置的偏移量， 当两个次序得到的值一样时，保证原始的相对位置。
    """

    def __init__(self, shuffle_rate=-1, augment_count=2):
        # 后续可以考虑对label数量缺少的类别单独数据增强
        assert shuffle_rate <= 1
        self.shuffle_rate = shuffle_rate
        self.augment_count = augment_count

    def get_segments_cut(self, query, labels):
        segment_position = []
        for i, label in enumerate(labels):
            if label.startswith("B"):
                segment_position.append(i)
            if label == "O" and (i == 0 or labels[i - 1] != "O"):
                segment_position.append(i)
        segment_position.append(len(query))
        query_result = [query[segment_position[i]:segment_position[i + 1]] for i in range(len(segment_position) - 1)]
        labels_result = [labels[segment_position[i]:segment_position[i + 1]] for i in range(len(segment_position) - 1)]
        return query_result, labels_result

    def shuffle_query_among_segments(self, query, labels):
        query_result, labels_result = self.get_segments_cut(query, labels)
        if self.shuffle_rate <= 0:
            new_order = np.arange(len(query_result))
            np.random.shuffle(new_order)
        else:
            shuffle_lambda = self.shuffle_rate * len(query_result)
            new_order = np.random.poisson(shuffle_lambda, size=len(query_result)) + np.arange(len(query_result))
            new_order = np.argsort(new_order)
        query_result = [xi for i in new_order for xi in query_result[i]]
        labels_result = [xi for i in new_order for xi in labels_result[i]]
        return query_result, labels_result

    def augment_data(self, data):
        new_data = []
        for example in data:
            times = np.random.poisson(self.augment_count)
            query = example['query']
            labels = example['labels']
            for i in range(times):
                new_query, new_labels = self.shuffle_query_among_segments(query, labels)
                new_data.append({'query': new_query, 'labels': new_labels})
        return data + new_data

    def __call__(self, data):
        return self.augment_data(data)
