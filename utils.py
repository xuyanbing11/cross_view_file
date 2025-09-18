import os
import torch
import yaml
from model import two_view_net, three_view_net
from yaml import SafeLoader 
def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1 # count the image number in every class
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        print('no dir: %s'%dirname)
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pth" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name

######################################################################
# Save model
#---------------------------
def save_network(network, dirname, epoch_label):
    if not os.path.isdir('./model/'+dirname):
        os.mkdir('./model/'+dirname)
    if isinstance(epoch_label, int):
        save_filename = 'net_%03d.pth'% epoch_label
    else:
        save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',dirname,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda()


######################################################################
#  Load model for resume
#---------------------------
def load_network(name, opt):
    dirname = os.path.join('./model', name)
    last_model_name = os.path.basename(get_model_list(dirname, 'net'))
    epoch = last_model_name.split('_')[1].split('.')[0]
    
    config_path = os.path.join(dirname, 'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream) or {}  # 确保 config 不为 None
    
    # 为所有可能缺失的键设置默认值（关键修改）
    opt.name = config.get('name', 'base_model')
    opt.data_dir = config.get('data_dir', '/path/to/data')
    opt.train_all = config.get('train_all', False)
    opt.droprate = config.get('droprate', 0.65)
    opt.color_jitter = config.get('color_jitter', False)
    opt.batchsize = config.get('batchsize', 8)
    opt.h = config.get('h', 256)
    opt.w = config.get('w', 256)
    opt.share = config.get('share', False)
    opt.stride = config.get('stride', 2)
    opt.pool = config.get('pool', 'avg')
    opt.erasing_p = config.get('erasing_p', 0.0)
    opt.lr = config.get('lr', 0.01)
    opt.nclasses = config.get('nclasses', 701)  # 根据实际数据集调整默认值
    opt.use_dense = config.get('use_dense', False)
    opt.fp16 = config.get('fp16', False)  # 为 'fp16' 设置默认值 False
    opt.views = config.get('views', 3)
    
    # 其他模型初始化逻辑保持不变
    if opt.use_dense:
        model = ft_net_dense(opt.nclasses, opt.droprate, opt.stride, None, opt.pool)
    elif opt.PCB:
        model = PCB(opt.nclasses)
    elif opt.views == 2:
        model = two_view_net(opt.nclasses, opt.droprate, stride=opt.stride, pool=opt.pool, share_weight=opt.share)
    elif opt.views == 3:
        model = three_view_net(opt.nclasses, opt.droprate, stride=opt.stride, pool=opt.pool, share_weight=opt.share)
    
    # 加载模型权重
    save_path = os.path.join(dirname, f'net_{epoch}.pth')
    print(f'Load the model from {save_path}')
    model.load_state_dict(torch.load(save_path))
    return model, opt, epoch
    
    

def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def update_average(model_tgt, model_src, beta):
    toogle_grad(model_src, False)
    toogle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)

    toogle_grad(model_src, True)

