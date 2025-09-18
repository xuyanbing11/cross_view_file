import scipy.io
import torch
import numpy as np
import os

torch.cuda.set_device(0)

#######################################################################
# Evaluate
def evaluate(qf_street, ql_street, qf_drone, ql_drone, gf, gl):
    # 合并 street 和 drone 的查询特征和标签
    qf = torch.cat((qf_street, qf_drone), dim=0)
    ql = np.concatenate((ql_street, ql_drone))

    unique_labels = np.unique(ql)
    combined_features = []

    # 对相同标签的特征求平均
    for label in unique_labels:
        indices = np.where(ql == label)[0]
        label_features = qf[indices]
        avg_feature = torch.mean(label_features, dim=0)
        combined_features.append(avg_feature)

    combined_features = torch.stack(combined_features)

    # 计算余弦相似度
    combined_features = combined_features.view(-1, 512)
    gf = gf.view(-1, 512)
    score = torch.mm(gf, combined_features.t())
    score = score.cpu()
    score = score.numpy()

    CMC_all = []
    ap_all = []

    for i in range(score.shape[1]):
        # predict index
        index = np.argsort(score[:, i])  # from small to large
        index = index[::-1]

        # good index
        query_index = np.argwhere(gl == unique_labels[i])
        good_index = query_index
        junk_index = np.argwhere(gl == -1)

        ap_tmp, CMC_tmp = compute_mAP(index, good_index, junk_index)
        CMC_all.append(CMC_tmp)
        ap_all.append(ap_tmp)

    CMC_all = torch.stack(CMC_all)
    CMC = torch.mean(CMC_all.float(), dim=0)
    ap = np.mean(ap_all)

    return CMC, ap

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

CMC, ap = evaluate(query_feature_street, query_label_street, query_feature_drone, query_label_drone, gallery_feature, gallery_label)

print(round(len(gallery_label) * 0.01))
print('Recall@1:%.2f Recall@5:%.2f Recall@10:%.2f Recall@top1:%.2f AP:%.2f' % (
    CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, CMC[round(len(gallery_label) * 0.01)] * 100, ap * 100))

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