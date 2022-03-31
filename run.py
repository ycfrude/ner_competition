from src.lightning_module.pl_ner import *
from src.utils import file_reading, file_writing
from src.lightning_module.config import Config
from sklearn.model_selection import train_test_split


if __name__ == "__main__": 
    predict_path = "/opt/wekj/aimeng_huawei_cloud/pretrain/ner/datasets/JDNER/test.txt"
    train_path = "/opt/wekj/aimeng_huawei_cloud/pretrain/train_data/train.txt"
    # dev_path = "/opt/wekj/aimeng_huawei_cloud/pretrain/ner/datasets/JDNER/dev.txt"
    # dev_data = file_reading(dev_path)
    train_data = file_reading(train_path)
    label2id = FeatureConverter.generate_label2id()
    # print(f"len(train_data) = {len(train_data)}\ntrain_data = {train_data}")
    id2label = {v:k for k,v in label2id.items()}
    train_data, dev_data = train_test_split(train_data, test_size=0.1, shuffle=True)
    predict_data = file_reading(predict_path)
    print(predict_data[0])
    config = Config()
    train(train_data, dev_data, config, label2id, id2label, gpus=[1])
    predict_data = predict(predict_data, config)
    file_writing(predict_data, "/opt/wekj/aimeng_huawei_cloud/pretrain/ner/datasets/JDNER/test_04.txt")


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
    unlabeled_data.to_csv("/opt/wekj/aimeng_huawei_cloud/pretrain/ner/datasets/JDNER/unlabeled_04.csv", index=None)