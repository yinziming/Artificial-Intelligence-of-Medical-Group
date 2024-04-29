import torch

# computes the dice score on two tensors
def dice(y_true, y_pred):
    epsilon = torch.tensor(1e-7, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    sum_p = torch.sum(y_pred, dim=0)
    sum_r = torch.sum(y_true, dim=0)
    sum_pr = torch.sum(y_true * y_pred, dim=0)
    dice_numerator = 2*sum_pr
    dice_denominator = sum_r+sum_p
    dice_score = (dice_numerator + epsilon) / (dice_denominator + epsilon)
    return dice_score


def dice_whole_metric(y_true, y_pred):
    #computes the dice for the whole tumor
    y_true_f = torch.reshape(y_true, shape=(-1, 4))
    y_pred_f = torch.reshape(y_pred, shape=(-1, 4))
    y_whole = torch.sum(y_true_f[:, 1:], dim=1) #所有行，取第一列至最后一列，沿张量1轴求和
    p_whole = torch.sum(y_pred_f[:, 1:], dim=1)
    dice_whole = dice(y_whole, p_whole)
    return dice_whole


def dice_en_metric(y_true, y_pred):
    # computes the dice for the enhancing region
    y_true_f = torch.reshape(y_true, shape=(-1, 4))
    y_pred_f = torch.reshape(y_pred, shape=(-1, 4))
    y_enh = torch.sum(y_true_f[:, 3:], dim=1)  # 所有行，取最后一列
    p_enh = torch.sum(y_pred_f[:, 3:], dim=1)
    dice_en = dice(y_enh, p_enh)
    return dice_en


def dice_core_metric(y_true, y_pred):
    # computes the dice for the core region
    y_true_f = torch.reshape(y_true, shape=(-1, 4))
    y_pred_f = torch.reshape(y_pred, shape=(-1, 4))
    
    y_true_f1 = y_true_f[:, 1:2]
    y_true_f3 = y_true_f[:, 3:]
    y_true_f = torch.cat([y_true_f1, y_true_f3], dim=1)  # 所有行，取第一列,第三列
    y_core = torch.sum(y_true_f, axis=1)
    
    y_pred_f1 = y_pred_f[:, 1:2]
    y_pred_f3 = y_pred_f[:, 3:]
    y_pred_f = torch.cat([y_pred_f1, y_pred_f3], dim=1)    
    p_core = torch.sum(y_pred_f, dim=1)
    dice_core = dice(y_core, p_core)
    return dice_core


def categorical_crossentropy(y_true, y_pred):
    axis = 1
    output = y_pred.clone()
    output /= torch.sum(y_pred, dim=axis, keepdim=True)
    epsilon = torch.tensor(1e-7, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    output = torch.clip(y_pred, epsilon, 1-epsilon)
    loss = -torch.sum(y_true * torch.log(output), dim=axis)
    return loss


def weighted_log_loss(y_true, y_pred):
    axis = 1
    output = y_pred.clone()
    output /= torch.sum(y_pred, dim=axis, keepdims=True)
    epsilon = torch.tensor(1e-7, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    output = torch.clip(y_pred,epsilon, 1-epsilon)

    # y_true_f = torch.reshape(y_true, shape=(-1, 4))
    # sum_t = torch.sum(y_true_f, dim=-2)
    # weights = torch.pow(torch.square(sum_t) + 1e-7, -1).reshape(1, 4, 1, 1, 1)

    weights = torch.tensor([1, 5, 2, 2], device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')).reshape((1,4,1,1,1))
    loss = y_true * torch.log(output) * weights + (1-y_true)*torch.log(1-output) * weights
    loss = -torch.mean(torch.sum(loss, dim=axis))
    return loss


def gen_dice_loss(y_true, y_pred):
    '''
    computes the sum of two losses : generalised dice loss and weighted cross entropy
    '''
    # generalised dice score is calculated as in this paper : https://arxiv.org/pdf/1707.03237
    epsilon = torch.tensor(1e-7, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    y_true_f = torch.reshape(y_true, shape=(-1, 4))
    y_pred_f = torch.reshape(y_pred, shape=(-1, 4))
    sum_p = torch.sum(y_pred_f, dim=-2)
    sum_t = torch.sum(y_true_f, dim=-2)
    sum_pr = torch.sum(y_true_f * y_pred_f, axis=-2)
    weights = torch.pow(torch.square(sum_t) + epsilon, -1)
    generalised_dice_numerator = 2 * torch.sum(weights * sum_pr)
    generalised_dice_denominator = torch.sum(weights * (sum_t + sum_p))
    generalised_dice_score = generalised_dice_numerator / generalised_dice_denominator
    GDL = 1-generalised_dice_score
    del sum_p, sum_t, sum_pr, weights
    return GDL

def sensitivity_specificity_loss(y_true, y_pred):
    epsilon = torch.tensor(1e-7, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    lamda = torch.tensor(0.05, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    y_true_f = torch.reshape(y_true, shape=(-1, 4))
    y_pred_f = torch.reshape(y_pred, shape=(-1, 4))
    err = torch.pow((y_true_f - y_pred_f), 2)
    SS = lamda * (torch.sum(err * y_true_f)) / (torch.sum(y_true_f) + epsilon) + (1-lamda) * (torch.sum(err * (1 - y_true_f))) / (torch.sum((1 - y_true_f)) + epsilon)
    del y_true_f, y_pred_f, err
    return SS