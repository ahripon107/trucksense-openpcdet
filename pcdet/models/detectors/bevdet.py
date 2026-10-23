from .bevfusion import BevFusion
from .centerpoint import CenterPoint
from .detector3d_template import Detector3DTemplate


class BEVDet(BevFusion):
    """
        Camera-only BEVDet: image backbone -> neck -> LSS view transform -> BEV encoder -> CenterHead.
    """
    def __init__(self, model_cfg, num_class, dataset):
        Detector3DTemplate.__init__(self, model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_topology = [
            'image_backbone', 'neck', 'vtransform', 'backbone_2d', 'dense_head'
        ]
        self.module_list = self.build_networks()

    def build_vtransform(self, model_info_dict):
        vtransform_module, model_info_dict = super().build_vtransform(model_info_dict)
        model_info_dict['num_bev_features'] = self.model_cfg.VTRANSFORM.OUT_CHANNEL
        return vtransform_module, model_info_dict

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    get_training_loss = CenterPoint.get_training_loss
    post_processing = CenterPoint.post_processing
