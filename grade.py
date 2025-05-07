import os
import shutil
import re
import numpy as np
import Levenshtein

std_result_dir = "standard_result"
result_dir = "result"

def get_file_list(path):
    """
    Get a list of all files in the given directory and its subdirectories.
    """
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return [f for f in file_list if "html" in f]

def levenshtein_accuracy(ref: str, hyp: str) -> float:
    dist = Levenshtein.distance(ref, hyp)
    accuracy = 1 - dist / max(len(ref), 1)  # 防止除以0
    return accuracy


files = get_file_list(result_dir)
files.sort()
results = []
for f in files:
    f_std = f.replace(result_dir, std_result_dir)
    with open(f, "r", encoding="utf-8") as file:
        content = file.read()
        table = re.findall(r'<table.*?>(.*?)</table>', content, re.DOTALL)[0]
    with open(f_std, "r", encoding="utf-8") as file:
        content_std = file.read()
        table_std = re.findall(r'<table.*?>(.*?)</table>', content_std, re.DOTALL)[0]
    # 提取出表格中的文本内容，分为list
    table_list = re.findall(r'<td.*?>(.*?)</td>', table, re.DOTALL)
    table_list_std = re.findall(r'<td.*?>(.*?)</td>', table_std, re.DOTALL)
    table_list = [c.replace("<br/>", "").replace("<br>", "").replace("&nbsp;", " ").replace(" ", "") for c in table_list]
    table_list_std = [c.replace("<br/>", "").replace("<br>", "").replace("&nbsp;", " ").replace(" ", "") for c in table_list_std]
    # 纯文字相似度
    table_content = '\n'.join([line for line in table_list if line!=""])
    table_content_std = '\n'.join([line for line in table_list_std if line!=""])
    with open(f.replace("html", "txt"), "w", encoding="utf-8") as file:
        file.write(table_content)
    with open(f_std.replace("html", "txt"), "w", encoding="utf-8") as file:
        file.write(table_content_std)
    # 计算准确率
    maxacc = levenshtein_accuracy(table_content_std, table_content)
    # while pos:=table_content.find("\n") != -1:
    #     table_content = table_content[pos+1:]
    #     accuracy = levenshtein_accuracy(table_content_std, table_content)
    #     if accuracy > maxacc:
    #         maxacc = accuracy
    print(f"File: {f}, Accuracy: {maxacc:.2%}", " " if maxacc>0.9 else "↓", end=" \t")
    # 单元格准确率
    hit = 0
    empty = 0
    for i in range(len(table_list_std)):
        c = table_list_std[i]
        if c == "":
            empty += 1
            continue
        if c in table_list:
            hit += 1
            index = table_list.index(c)
            table_list[index] = ""
            table_list_std[i] = ""
    if len(table_list_std) - empty > 0:
        hit_rate = (hit / (len(table_list_std) - empty))
    else:
        hit_rate = 0
    print("Cell Hit Rate: ", f"{hit_rate:.2%}", end="\t")
    # 去除空行
    table_list_left = "".join([c for c in table_list if c != ""])
    table_list_std_left = "".join([c for c in table_list_std if c != ""])
    # print(table_list_left)
    # print(table_list_std_left)
    leftacc = levenshtein_accuracy(table_list_std_left, table_list_left)
    if leftacc > 0:
        print("Remained Part Accuracy: ", f"{leftacc:.2%}")
    else:
        print("Remained Part Accuracy: ", "NaN")
    results += [(f, maxacc, hit_rate, leftacc)]
import csv
with open("result.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["File", "Max Accuracy", "Cell Hit Rate", "Remained Part Accuracy"])
    for result in results:
        writer.writerow(result)
