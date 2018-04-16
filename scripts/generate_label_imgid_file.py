import os
import json

DIR = "../data/train"
LabelStartIndexFile = "../data/train_label_v2.txt"

def parseLabelInfoToFile():
    dir_path = DIR
    file_name = LabelStartIndexFile
    train_map = {}

    file_list = os.listdir(dir_path)
    print(len(file_list))
    for img_name in file_list:
        try:
            item_map = {};
            result = img_name.split("_")
            img_id = int(result[0])
            label = str((result[1][:-4]))
            if train_map.get(label) is None:
                print(label, img_id)
                item_map["start"] = img_id;
                item_map["end"] = img_id;
            else:
                item_map = train_map[label];
                if int(item_map["start"]) > img_id:
                    item_map["start"] = str(img_id)
                if int(item_map["end"]) < img_id:
                    item_map["end"] = str(img_id)
            train_map[label] = item_map
        except Exception as e:
            print(e)

    fd = open(file_name, "w")
    fd.write(json.dumps(train_map))
    # 关闭文件
    fd.close()

#读写文件,文件格式{label：start_index}
parseLabelInfoToFile()

