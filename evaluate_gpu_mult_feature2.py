import scipy.io
import torch
import numpy as np
import os
import pandas as pd

torch.cuda.set_device(0)

#######################################################################
# Evaluate
def evaluate(qf_street, ql_street, qf_drone, ql_drone, gf, gl, num_drone, num_street):
    # 初始化结果列表
    CMC_all = []
    ap_all = []
    
    # 获取所有唯一标签
    unique_labels = np.unique(ql_street)  # 假设街道和无人机标签一致
    # print(1)
    # 对每个标签进行处理
    for label in unique_labels:
        # 获取当前标签的街道和无人机特征索引
        street_indices = np.where(ql_street == label)[0]
        drone_indices = np.where(ql_drone == label)[0]
        
        # 确保有足够的图片
        if len(street_indices) < 1 or len(drone_indices) < 3:
            print(f"警告: 标签 {label} 的街道图片不足1张或无人机图片不足3张")
            continue
        
        # 选择1张街道照片 (固定选第一张)
        street_idx = street_indices[:num_street]
        # street_feature = qf_street[street_idx].unsqueeze(0)
        street_features = qf_street[street_idx]
        avg_street_features = torch.mean(street_features, dim=0, keepdim=True)
        
        
        # 选择3张无人机照片 (固定选前3张)
        drone_idx = drone_indices[:num_drone]
        drone_features = qf_drone[drone_idx]
        
        # 计算3张无人机照片的平均特征
        avg_drone_feature = torch.mean(drone_features, dim=0, keepdim=True)
        
        # 融合街道和无人机特征 (等权重平均)
        fused_feature = (avg_street_features + avg_drone_feature) / 2
        
        # 计算余弦相似度
        score = torch.mm(gf, fused_feature.t()).squeeze(1).cpu().numpy()
        
        # 排序并计算指标
        index = np.argsort(score)[::-1]  # 从大到小排序
        query_index = np.argwhere(gl == label)
        good_index = query_index
        junk_index = np.argwhere(gl == -1)
        
        ap_tmp, CMC_tmp = compute_mAP(index, good_index, junk_index)
        CMC_all.append(CMC_tmp)
        ap_all.append(ap_tmp)
    
    # 计算平均指标
    if len(CMC_all) > 0:
        CMC = torch.mean(torch.stack(CMC_all).float(), dim=0)
        ap = np.mean(ap_all)
        return CMC, ap
    else:
        return torch.zeros(len(gl)), 0.0

def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:   # if empty
        cmc[0] = -1
        return ap, cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc
def create_tables(metrics_data, output_file='metrics_tables.xlsx'):
    """创建并保存指标表格到Excel文件"""
    # 为每个指标创建DataFrame
    metrics = ['Recall@1', 'Recall@5', 'Recall@10', 'Recall@top1', 'AP']
    drones = sorted({d for d in metrics_data})
    streets = sorted({s for d in metrics_data.values() for s in d})
    
    dfs = {metric: pd.DataFrame(index=drones, columns=streets) for metric in metrics}
    
    # 填充数据
    for d in drones:
        for s in streets:
            if s in metrics_data[d]:
                for metric in metrics:
                    dfs[metric].at[d, s] = metrics_data[d][s][metric]
    
    # 写入Excel文件
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for metric, df in dfs.items():
            df.index.name = 'num_drone'
            df.columns.name = 'num_street'
            df.to_excel(writer, sheet_name=metric)
    
    print(f"表格已生成并保存到 {output_file}")
######################################################################
result = scipy.io.loadmat('pytorch_result.mat')
query_feature_street = torch.FloatTensor(result['query_street_f'])
query_label_street = result['query_street_label'][0]
query_feature_drone = torch.FloatTensor(result['query_drone_f'])
query_label_drone = result['query_drone_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_label = result['gallery_label'][0]
multi = os.path.isfile('multi_query.mat')

if multi:
    m_result = scipy.io.loadmat('multi_query.mat')
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_label = m_result['mquery_label'][0]
    mquery_feature = mquery_feature.cuda()

query_feature_street = query_feature_street.cuda()
query_feature_drone = query_feature_drone.cuda()
gallery_feature = gallery_feature.cuda()

print(query_feature_street.shape)
print(query_feature_drone.shape)
print(gallery_feature.shape)

metrics_data = {}

nums_drone = 7
nums_street = 7
for num_drone in range(nums_drone):
    for num_street in range(nums_street):
        CMC, ap = evaluate(query_feature_street, query_label_street, query_feature_drone, query_label_drone, gallery_feature, 
                           gallery_label, num_drone, num_street)

        # 计算各项指标
        top1_index = min(round(len(gallery_label) * 0.01), len(CMC) - 1)
        
        recall1 = CMC[0] * 100
        recall5 = CMC[4] * 100
        recall10 = CMC[9] * 100
        recall_top1 = CMC[top1_index] * 100
        ap_value = ap * 100
        
        # 保存结果到字典
        if num_drone not in metrics_data:
            metrics_data[num_drone] = {}
            
        metrics_data[num_drone][num_street] = {
            'Recall@1': recall1,
            'Recall@5': recall5,
            'Recall@10': recall10,
            'Recall@top1': recall_top1,
            'AP': ap_value
        }
        
        # 打印结果
        print(f"now is {num_drone}drone and {num_street}street")
        print(f'Recall@1:{recall1:.2f} Recall@5:{recall5:.2f} Recall@10:{recall10:.2f} Recall@top1:{recall_top1:.2f} AP:{ap_value:.2f}')

# 生成并保存表格
create_tables(metrics_data)


# nums_drone = 7
# nums_street = 7
# for num_drone in range(nums_drone):
#     for num_street in range(nums_street):
#         CMC, ap = evaluate(query_feature_street, query_label_street, query_feature_drone, query_label_drone, gallery_feature, 
#                            gallery_label, num_drone, num_street)

#         # print(round(len(gallery_label) * 0.01))
#         print(f"now is {num_drone}drone and {num_street}street")  
#         print('Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f' % (
#             CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, CMC[round(len(gallery_label) * 0.01)] * 100, ap * 100))

# multiple-query evaluation is not used.
# CMC = torch.IntTensor(len(gallery_label)).zero_()
# ap = 0.0
# if multi:
#     for i in range(len(query_label)):
#         mquery_index1 = np.argwhere(mquery_label == query_label[i])
#         mquery_index2 = np.argwhere(mquery_cam == query_cam[i])
#         mquery_index = np.intersect1d(mquery_index1, mquery_index2)
#         mq = torch.mean(mquery_feature[mquery_index, :], dim=0)
#         ap_tmp, CMC_tmp = evaluate(mq, query_label[i], query_cam[i], gallery_feature, gallery_label, gallery_cam)
#         if CMC_tmp[0] == -1:
#             continue
#         CMC = CMC + CMC_tmp
#         ap += ap_tmp
#         # print(i, CMC_tmp[0])
#     CMC = CMC.float()
#     CMC = CMC / len(query_label)  # average CMC
#     print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], ap / len(query_label)))