from sklearn.model_selection import train_test_split

from src.config import Config
from src.lightning_module.models import ShuffleAmongSegments
from src.lightning_module.pl_ner import *
from src.utils import file_reading, file_writing

augment = ShuffleAmongSegments(-1, 0.5)

if __name__ == "__main__":
    for i in range(10,20):
        predict_path = "/opt/wekj/aimeng_huawei_cloud/pretrain/ner/datasets/JDNER/test.txt"
        train_path = "train_data/train.txt"
        # dev_path = "/opt/wekj/aimeng_huawei_cloud/pretrain/ner/datasets/JDNER/dev.txt"
        # dev_data = file_reading(dev_path)
        train_data = file_reading(train_path)
        label2id = FeatureConverter.generate_label2id()
        # print(f"len(train_data) = {len(train_data)}\ntrain_data = {train_data}")
        id2label = {v:k for k,v in label2id.items()}
        train_data, dev_data = train_test_split(train_data, test_size=0.1, shuffle=True)
        train_data = augment(train_data)
        predict_data = file_reading(predict_path)
        print(predict_data[0])
        config = Config()
        train(train_data, dev_data, config, label2id, id2label, gpus=[1])
        predict_data = predict(predict_data, config)
        file_writing(predict_data, f"/opt/wekj/aimeng_huawei_cloud/pretrain/ner/datasets/JDNER/test_{i}.txt")


        import pandas as pd
        unlabeled_path = "/opt/wekj/aimeng_huawei_cloud/pretrain/train_data/unlabeled_train_data.txt"
        unlabeled_data = []
        with open(unlabeled_path, 'r') as f:
            unlabeled_data = f.read().splitlines()
            unlabeled_data = [{"query":[xi if xi not in (""," ") else "[SPACE]" for xi in x]} for x in unlabeled_data]
        print(f"len unlabeled data = {len(unlabeled_data)}")
        # print(unlabeled_data)
        unlabeled_data = predict(unlabeled_data, config)
        unlabeled_data = pd.DataFrame(unlabeled_data)
        unlabeled_data.to_csv(f"/opt/wekj/aimeng_huawei_cloud/pretrain/ner/datasets/JDNER/unlabeled_{i}.csv", index=None)