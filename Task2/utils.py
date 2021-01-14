import os
import torch
import config



def load_model(model,load_dir,name):
    if not os.path.exists(load_dir):
        os.makedirs(load_dir)

    model.load_state_dict(torch.load(os.path.join(load_dir,name),
                                     map_location=config.DEVICE))


def save_model(model,save_dir,name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model.state_dict(),os.path.join(save_dir,name))


def variable(func):
    '''
    used as a decorator
    '''
    return func()


def generate_base_anchors(base_size=16,ratios=[0.5,1,2],scales=[8,16,32]):
    '''
    based on github.com/rbgirshick/py-faster-rcnn
    '''
    base_anchor=torch.tensor([[1,1,base_size,base_size]])-1
    ratios=torch.tensor(ratios)
    scales=torch.tensor(scales)
    w,h,center_w,center_h=get_center(base_anchor)
    #process ratio 
    size=w*h
    size_ratios=size/ratios
    anchors_w=torch.round(torch.sqrt(size_ratios))
    anchors_h=torch.round(anchors_w*ratios)
    anchors=make_anchors(anchors_w,anchors_h,center_w,center_h)
    #process scale
    w,h,center_w,center_h=get_center(anchors)
    anchors_w=torch.mul(w.view(-1,1),scales).view(-1)
    anchors_h=torch.mul(h.view(-1,1),scales).view(-1)
    center_w=center_w.repeat(len(scales))
    center_h=center_h.repeat(len(scales))
    anchors=make_anchors(anchors_w,anchors_h,center_w,center_h)
    return anchors+1

    
def get_center(anchor):
    '''
    return the center of anchor
    '''
    w=anchor[:,2]-anchor[:,0]+1
    h=anchor[:,3]-anchor[:,1]+1
    center_w=anchor[:,0]+(w-1)/2
    center_h=anchor[:,1]+(h-1)/2
    return w,h,center_w,center_h


def make_anchors(anchors_w,anchors_h,center_w,center_h):
    '''
    anchors_w,anchors_h:[N,1]
    '''
    w_min=center_w-(anchors_w-1)/2
    h_min=center_h-(anchors_h-1)/2
    w_max=center_w+(anchors_w-1)/2
    h_max=center_h+(anchors_h-1)/2
    anchors=torch.stack([h_min,w_min,h_max,w_max]).permute(1,0)
    return anchors


def generate_anchors(base_anchors,feat_stride=16,height=14,width=14):
    shift_width=torch.tensor(list(range(0,feat_stride*width,feat_stride)))
    shift_height=torch.tensor(list(range(0,feat_stride*height,feat_stride)))
    shift_width,shift_height=torch.meshgrid(shift_width,shift_height)
    f=torch.flatten
    shift=torch.stack((f(shift_height),f(shift_width),
        f(shift_height),f(shift_width)),dim=1)
    shift_dim0=shift.size(0)
    base_anchors_dim0=base_anchors.size(0)
    shift=shift.repeat(base_anchors_dim0,1)
    base_anchors=base_anchors.repeat(shift_dim0,1)
    anchors=shift+base_anchors
    return anchors


def reg2bbox(anchors,bbox_regression):
    anchors_height=anchors[:,2]-anchors[:,0]
    anchors_width=anchors[:,3]-anchors[:,1]
    anchors_center_height=anchors[:,0]+0.5*anchors_height
    anchors_center_width=anchors[:,1]+0.5*anchors_width
    dy=bbox_regression[:,0]
    dx=bbox_regression[:,1]
    dh=bbox_regression[:,2]
    dw=bbox_regression[:,3]
    hat_G_y=dy*anchors_height+anchors_center_height
    hat_G_x=dx*anchors_width+anchors_center_width
    hat_G_h=anchors_height*torch.exp(dh)
    hat_G_w=anchors_width*torch.exp(dw)
    ymin=hat_G_y-0.5*hat_G_h
    xmin=hat_G_x-0.5*hat_G_w
    ymax=hat_G_y+0.5*hat_G_h
    xmax=hat_G_x+0.5*hat_G_w
    bbox=torch.stack([ymin,xmin,ymax,xmax],dim=1)
    return bbox

