import cv2,time,warnings,torch,os,xlsxwriter
import numpy as np
from paddleocr import PaddleOCR
from collections import Counter
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from torch import  nn
from sklearn.cluster import DBSCAN
warnings.filterwarnings("ignore")
to=time.time()



def predict_with_efficientnet_b0(image):
    # set CPU
    device = torch.device("cpu")
    # input shape
    img_height = 32
    img_width = 32
    image = Image.fromarray(image)
    # 圖像預處理
    preprocess = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])

    # 預處理後的圖像
    input_tensor = preprocess(image)
    input_tensor = input_tensor.unsqueeze(0)  # 增加一個batch維度
    input_tensor = input_tensor.to(device)


    # load model
    # efficientnet_b0 = models.efficientnet_b1(pretrained=False)
    # num_ftrs = efficientnet_b0.classifier[1].in_features
    #
    # # Modify the classifier
    # efficientnet_b0.classifier[1] = nn.Linear(num_ftrs, 11)
    # model = efficientnet_b0.to(device)
    #
    # # 加載最優模型
    # model_dir = "../saved_models"
    # model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model_numb1.pth'), map_location=device))
    #
    # model.eval()  # 設置模型為評估模式
    # class_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'none']
    #
    # # 預測
    # with torch.no_grad():
    #     output = model(input_tensor)
    #     _, pred = torch.max(output, 1)
    #     predicted_class = class_names[pred.item()]
    from ultralytics import YOLO
    model = YOLO('/home/kingargroo/YOLOVISION/runs/classify/train2/weights/best.pt')
    results = model(image)
    predicted_class=results[0].probs.top1
    if predicted_class== 10:
        predicted_class='none'
    return predicted_class





#加載paddle ocr模型
paddleocr = PaddleOCR(lang='ch', show_log=False,
                      det_model_dir='.paddleocr\\whl\\det\\ch\\ch_PP-OCRv4_det_infer',
                      rec_model_dir='.paddleocr\\whl\\rec\\ch\\ch_PP-OCRv4_rec_infer') 
warnings.filterwarnings("ignore", message="Since the angle classifier is not initialized")

# 讀取圖像
image_path = "tabletest3.jpg"
image = cv2.imread(image_path)
assert image is not None, f"Image read error: Unable to read image from {image_path}"
imgh,imgw=image.shape[0:2]


image_copy=image.copy()
image_copy=cv2.cvtColor(image_copy,cv2.COLOR_BGR2GRAY)

# Apply binary thresholding
_, binary_image = cv2.threshold(image_copy, 150, 255, cv2.THRESH_BINARY_INV)

# Define a horizontal kernel for detecting horizontal lines
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
horizontal_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horizontal_kernel)

# Define a vertical kernel for detecting vertical lines
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
vertical_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, vertical_kernel)

kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
vertical_lines = cv2.morphologyEx(vertical_lines, cv2.MORPH_DILATE, kernel_v)
height, width = vertical_lines.shape

for col in range(width):
    # 當前列的所有像素值
    column_pixels = vertical_lines[:, col]


    count_255 = np.sum(column_pixels == 255)

    if count_255 < 200:
        vertical_lines[:, col] = 0

# Combine detected horizontal and vertical lines
table_lines = cv2.add(horizontal_lines, vertical_lines)

# 膨脹操作來使線條變得粗糙
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
table_lines_dilated = cv2.morphologyEx(table_lines, cv2.MORPH_DILATE, kernel)
# Invert the image to get the lines in black
final_image = cv2.bitwise_not(table_lines_dilated)

cv2.imwrite("struct2.jpg",final_image)

# 尋找輪廓
contours, _ = cv2.findContours(final_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

box_data = []
for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
        # 計算矩形的邊界框
        x, y, w, h = cv2.boundingRect(approx)
        if   w<=0.3*imgw and  h<=0.3*imgh:
            xc,yc=x+w/2,y+h/2

            box_data.append([xc, yc, w-3, h-3, 1])
assert len(box_data)!=0,"finding rectangles fail"
#nms
box_data = np.array(box_data)
indices = cv2.dnn.NMSBoxes(box_data[:, :4].tolist(), box_data[:, 4].tolist(), score_threshold=0.5, nms_threshold=0.45)

data_dict={}
coordinates_dict={}
ys=[]
count=0
for i in indices:

    xc, yc, w, h =box_data[i][0],box_data[i][1],box_data[i][2],box_data[i][3]
    x,y,w,h=int(xc-w/2),int(yc-h/2),int(w),int(h)
     # 裁剪矩形區域
    cropped_image = image[y:y+h, x:x+w]


    #paddle OCR 識別
    result = paddleocr.ocr(cropped_image)
    try:
        result_str=result[0][0][1][0]
        recognize_score=result[0][0][1][1]
        if len(result_str)==1 and recognize_score<0.995:
            result_str=predict_with_efficientnet_b0(cropped_image)
    except:
        result_str=predict_with_efficientnet_b0(cropped_image)

    #result_str="none"
    data_=[[xc,yc,w,h],result_str]
    data_dict[i]=data_

    coordinates_dict[i]=yc
    ys.append(yc)

    # 保存裁剪的圖像
    #cv2.imwrite(f'../try/rectangle17_{count}.jpg', cropped_image)
    count+=1

    # 隨機生成顏色
    color = (0, 0,255)
    cv2.circle(image,(int(xc),int(yc)),3,color,3,1)

cv2.imwrite("result_table2.jpg",image)



# 使用 DBSCAN 對y進行聚類，找出屬於同一行的內容
sorted_dict = dict(sorted(coordinates_dict.items(), key=lambda item: item[1]))
lines=[]
sorted_y_list=list(sorted_dict.values())
index_list=list(sorted_dict.keys())
num_list=[]
current_line = []
current_line_data = []
lines_data=[]
ws=[]
# eps:一個閾值來決定什麼樣的差距算是同一行
#min_samples: 一個類中最少要有幾個樣本
dbscan = DBSCAN(eps=3, min_samples=5)
labels = dbscan.fit_predict(np.array(sorted_y_list).reshape((-1,1)))
print(labels)
# 根據聚類結果儲存行內容
lines_dict = {}
for i, label in enumerate(labels):
    # -1 表示噪聲點
    #assert label > -1,"error, label must bigger than -1"
    if label not in lines_dict:
        #新的類別
        lines_dict[label] = []
        if i!=0:
            #一行結束
        #按照x排序，恢復同一行不同內容的順序
            current_line=sorted(current_line,key=lambda x: x[0][0])
            lines.append(current_line)
            current_line_data=[data[1] for data in current_line]
            lines_data.append(current_line_data)
            num_list.append(len(current_line))
            #儲存寬度
            ws.append([sample[0][2] for sample in current_line ])
            #臨時變量清空
            current_line=[]
            current_line_data = []


    lines_dict[label].append(sorted_y_list[i])
    index=index_list[i]
    current_line.append(data_dict[index])


current_line=sorted(current_line,key=lambda x: x[0][0])
lines.append(current_line)
current_line_data=[data[1] for data in current_line]
lines_data.append(current_line_data)
num_list.append(len(current_line))
ws.append([sample[0][2] for sample in current_line ])


# 查找元素屬於哪個範圍
def find_range_index(ranges, a):
    for i, (low, high) in enumerate(ranges):
        if low <= a and a <= high:
            return i   # 返回0開始的索引
    return -1  # 如果不屬於任何範圍，返回-1

max_num=int(max(num_list))
min_num=int(min(num_list))

# 用於儲存每個元素的標籤（tag）
return_list = []
ranges=[]
# 遍歷每一行
for row_idx,line in enumerate(lines):
    current_tag = 0  # 起始標籤
    tagged_row = []  # 當前行的標籤
    col_idx = 0  # 列索引

    while col_idx < len(line):
        if row_idx == 0:  # 第一行，直接分配標籤，並以此爲基準
            tagged_row.append(current_tag)
            current_tag += 1
            #加入區間範圍
            ranges.append([line[col_idx][0][0]-line[col_idx][0][2]/2,line[col_idx][0][0]+line[col_idx][0][2]/2])
        else:
            #利用x座標判斷當前元素的tag
            current_x=line[col_idx][0][0]
            current_tag=find_range_index(ranges, current_x)
            #assert current_tag !=-1, "current_tag==-1, error"
            tagged_row.append(current_tag)
        col_idx += 1
    #一行結束
    return_list.append(tagged_row)




data=lines_data
sign_index=return_list

# 統計每個小列表中各個元素的個數
count_per_list = [Counter(sublist) for sublist in sign_index]
# 找出每個元素的最大出現次數
all_keys = set().union(*[counter.keys() for counter in count_per_list])
max_counts = {key: max(counter.get(key, 0) for counter in count_per_list) for key in all_keys}

# 計算每一行相同元素的個數與最大數量的差值
differences = []
for counter in count_per_list:
    diff = {key: max_counts[key] - counter.get(key, 0) for key in all_keys}
    differences.append(diff)
"""
列表 1 與最大數量的差值: {0: 0, 1: 2, 2: 0}
列表 2 與最大數量的差值: {0: 0, 1: 2, 2: 0}
列表 3 與最大數量的差值: {0: 0, 1: 0, 2: 0}
"""

# 創建一個新的 Excel 文件
workbook = xlsxwriter.Workbook('output.xlsx')
worksheet = workbook.add_worksheet()

# 定義一個函數來合併並寫入數據
def write_merged_data(row, col, value, merge_count):
    worksheet.merge_range(row, col, row, col + merge_count, value, cell_format)

# 設置格式
cell_format = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'font_size': 16})

for row_idx, row in enumerate(data):
    diff_row=differences[row_idx]
    col_idx = 0

    prev_merge_num=0
    while col_idx < len(row):
        tag=sign_index[row_idx][col_idx]
        if sum(diff_row.values())!=0:
            diff_num=diff_row[tag]
            if diff_num!=0:
                value=data[row_idx][col_idx]
                if value=="none":
                    value=" "
                write_merged_data(row_idx,col_idx+prev_merge_num,value,diff_num)
                prev_merge_num+=diff_num
            else:
                value=data[row_idx][col_idx]
                if value=="none":
                    value=" "
                worksheet.write(row_idx, col_idx+prev_merge_num, value)

        else:

            value=data[row_idx][col_idx]
            if value=="none":
                    value=" "
            worksheet.write(row_idx, col_idx, value)
        col_idx+=1
# 關閉 Excel 文件
workbook.close()

t1=time.time()
print(t1-to)

