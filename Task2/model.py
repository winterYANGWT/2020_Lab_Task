import config
import torch
import torch.nn as nn
import torchvision
import math
import utils



class VGGBlock(nn.Module):
    def __init__(self,in_channel,out_channel,
                 kernel_size=3,stride=1,padding=1):
        super().__init__()
        self.conv=nn.Conv2d(in_channel,out_channel,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)
        self.bn=nn.BatchNorm2d(out_channel)
        self.activation=nn.LeakyReLU(0.2,inplace=True)


    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.activation(x)
        return x



class VGG(nn.Module):
    def __init__(self,in_channel=3,out_channel=512):
        super().__init__()
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.pool_count=0
        self.block1=self.make_layers(in_channel=self.in_channel,out_channel=64,
                                     num_layers=2,pooling=True)
        self.block2=self.make_layers(in_channel=64,out_channel=128,
                                     num_layers=2,pooling=True)
        self.block3=self.make_layers(in_channel=128,out_channel=256,
                                     num_layers=4,pooling=True)
        self.block4=self.make_layers(in_channel=256,out_channel=512,
                                     num_layers=4,pooling=True)
        self.block5=self.make_layers(in_channel=512,out_channel=self.out_channel,
                                     num_layers=4,pooling=False)


    def make_layers(self,in_channel,out_channel,num_layers,pooling):
        assert num_layers>=1,\
        '''
        num_layers should be greater than or equal to 1
        '''
        layers=[]
        layers.append(VGGBlock(in_channel,out_channel))

        for _ in range(1,num_layers):
            layers.append(VGGBlock(out_channel,out_channel))

        if pooling==True:
            self.pool_count+=1
            layers.append(nn.MaxPool2d(kernel_size=2,stride=2))

        return nn.Sequential(*layers)


    def forward(self,x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x=self.block5(x)
        return x



class BackboneNet(nn.Module):
    def __init__(self,in_channel=3,out_channel=512):
        super().__init__()
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.cnn=VGG(in_channel=self.in_channel,out_channel=self.out_channel)
        self.feature_stride=2**self.pool_count


    def forward(self,x):
        features=self.cnn(x)
        return features
        


class Proposal(object):
    def __init__(self,nms_thresh=0.7,img_size=(800,600),min_size=16,
                 num_pre_nms=12000,num_post_nms=2000):
        super().__init__()
        self.nms_thresh=nms_thresh
        self.img_size=img_size
        self.min_size=min_size
        self.num_pre_nms=num_pre_nms
        self.num_post_nms=num_post_nms


    def __call__(self,anchors,rpn_reg_pred,rpn_cls_prob,scale):
        rpn_cls_prob=rpn_cls_prob.squeeze(1)
        #bbox regression
        roi=utils.reg2bbox(anchors,rpn_reg_pred)
        roi=torchvision.ops.clip_boxes_to_image(roi,size=self.img_size)
        #delete small anchor
        min_size=self.min_size*scale
        keep_indices=torchvision.ops.remove_small_boxes(roi,min_size)
        roi=roi[keep_indices,:]
        cls_prob=cls_prob[keep_indices]
        #sort positive anchor and delete some anchor (or not)
        cls_prob,order_indices=torch.sort(cls_prob)

        if num_pre_nms>0:
            order_indices=order_indices[:num_pre_nms]
            cls_prob=cls_prob[:num_pre_nms]

        roi=roi[order_indices,:]

        roi=torchvision.ops.nms(roi,cls_prob,self.nms_thresh)

        if num_post_nms>0:
            roi=roi[:num_post_nms]

        return roi



class RPN(nn.Module):
    def __init__(self,in_channel=512,out_channel=512,feature_stride=16):
        super().__init__()
        self.in_channel=in_channel
        self.out_channel=out_channel
        base_anchors=utils.generate_base_anchors(feature_stride=feature_stride)
        self.anchors=utils.generate_anchors(base_anchors)
        self.num_anchors=self.anchors.size(0)
        self.proposal_layer=Proposal()
        self.rpn_conv=nn.Sequential(nn.Conv2d(in_channel,out_channel,
                                              kernel_size=3,
                                              stride=1,
                                              padding=1),
                                    nn.LeakyReLU(0.2,inplace=True))
        self.cls_conv=nn.Conv2d(in_channel,self.num_anchors,
                                kernel_size=1,
                                stride=1,
                                padding=0)
        self.reg_conv=nn.Conv2d(in_channel,self.num_anchors*4,
                                kernel_size=1,
                                stride=1,
                                padding=0)
        self.softmax=nn.Sigmoid(dim=2)


    def forward(self,features,scales):
        B,C,H,W=features.size()
        A=self.num_anchors
        #rpn_feature:B,C,H,W
        rpn_features=self.rpn_conv(features)
        #rpn_cls_prob:B,H*W*A,1 the probability of a anchor is positive or negative
        rpn_cls_score=self.cls_conv(rpn_features).permute(0,2,3,1)
        rpn_cls_score=rpn_cls_score.contiguous().view(B,-1,1)
        rpn_cls_prob=self.softmax(rpn_cls_score)
        #rpn_bbox_pred:B,H*W*A,4 bounding box regression
        rpn_bbox_pred=self.reg_conv(rpn_features).permute(0,2,3,1)
        rpn_bbox_pred=self.rpn_bbox_pred.contiguous().view(B,-1,4)
        #get region proposal
        rois=[]
        roi_indices=[]

        for b in range(B):
            roi=self.proposal_layer(self.anchors,
                                    rpn_bbox_pred[b],
                                    rpn_cls_prob[b],
                                    scales[b])
            roi_index=b*torch.ones(len(roi))
            rois.append(roi)
            roi_indices.append(roi_index)

        rois=torch.cat(rois,dim=0)
        roi_indices=torch.cat(roi_indices,dim=0)
        return rpn_bbox_pred,rpn_cls_score,rois,roi_indices



class ROIHead(nn.Module):
    def __init__(self,num_classes,roi_size=7):
        super().__init__()
        self.num_class=num_class
        self.roi_size=roi_size
        self.spatial_scale=1/16
        self.roi_pooling=torchvision.ops.RoiPool(output_size=self.roi_size,
                                                 spatial_scale=self.spatial_scale)
        self.fc=nn.Sequential(nn.Linear(49,4096),
                              nn.Dropout(),
                              nn.LeakyReLU(0.2,inplace=True),
                              nn.Linear(4096,4096),
                              nn.Dropout(),
                              nn.LeakyReLU(0.2,inplace=True))
        self.bbox_pred=nn.Linear(4096,self.num_class*4)
        self.cls_score=nn.Linear(4096,self.num_class)
        self.softmax=nn.Softmax(dim=1)


    def forward(self,features,rois,roi_indices):
        indices_and_roi=torch.cat([roi_indices,roi],dim=1)
        indices_and_roi=indices_and_roi[:,0,2,1,4,3].contiguous()
        pooled_roi=self.roi(features,indices_and_roi)
        pooled_roi=torch.flatten(pooded_roi,start_dim=1,end_dim=-1)
        fc=self.fc(pooled_roi)
        bbox_pred=self.bbox_pred(fc)
        cls_score=self.cls_score(fc)
        cls_prob=self.softmax(cls_score)
        return bbox_pred,cls_prob



class FasterRCNN(nn.Module):
    def __init__(self,num_class):
        super().__init__()
        self.num_class=num_class
        self.backbone=BackboneNet()
        self.rpn=RPN()
        self.roi_size=self.backbone.cnn.
        self.roi_head=ROIHead(self.num_class)

    def forward(self,x,scales):
        features=self.backbone(x)
        rpn_bbox_pred,rpn_cls_score,rois,roi_indices=self.rpn(features,scales)
        roi_bbox_pred,roi_cls_prob=self.roi_head(features,rois,roi_indices)
        return roi_cls_prob,roi_bbox_pred,rois,roi_indices


if __name__ == '__main__':
    cnn=VGG()
    rpn=RPN()
    print(cnn)

