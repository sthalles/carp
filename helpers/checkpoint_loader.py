import torch
from collections import OrderedDict

class CheckpointsLoader:

    def __init__(self, pretrained):
        self.pretrained = pretrained
    
    def _load_carp_or_dino(self, checkpoint_key='teacher'):
        state_dict = torch.load(self.pretrained, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        return state_dict

    def _load_moco_v3(self, encoder_name='base_encoder'):
        checkpoint = torch.load(self.pretrained, map_location="cpu")
        linear_keyword = 'fc'
        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith(f'module.{encoder_name}') and not k.startswith(f'module.{encoder_name}.%s' % linear_keyword):
                # remove prefix
                state_dict[k[len(f"module.{encoder_name}."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        return state_dict

    def _load_swav_deepcluster_selav2(self, model):
        state_dict = torch.load(self.pretrained, map_location='cpu')
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # remove prefixe "module."
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        for k, v in model.state_dict().items():
            if k not in list(state_dict):
                print('key "{}" could not be found in provided state dict'.format(k))
            elif state_dict[k].shape != v.shape:
                print('key "{}" is of different shape in model and provided state dict'.format(k))
                state_dict[k] = v
        return state_dict

    def _load_barlowtwins(self):
        state_dict = torch.load(self.pretrained, map_location='cpu')
        return state_dict

    def _load_infomin(self):
        encoder_state_dict = OrderedDict()
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']
        for k, v in state_dict.items():
            k = k.replace('module.', '')
            if 'encoder' in k:
                k = k.replace('encoder.', '')
                encoder_state_dict[k] = v
        return encoder_state_dict

    def _load_triplet(self):
        checkpoint = torch.load(self.pretrained, map_location="cpu")
        state_dict = checkpoint['state_dict']
        return state_dict

    def _load_obow(self):
        checkpoint = torch.load(self.pretrained, map_location="cpu")
        state_dict = checkpoint['network']
        for k in list(state_dict.keys()):
            if k.startswith('fc'):
                del state_dict[k]
        return state_dict

    def _load_pclv2(self):
        checkpoint = torch.load(self.pretrained, map_location="cpu")
        # rename pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        return state_dict
    
    def _load_vicreg(self):
        state_dict = torch.load(self.pretrained, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
            state_dict = {
                key.replace("module.backbone.", ""): value
                for (key, value) in state_dict.items()
            }
        return state_dict

    def load_pretrained(self, model, model_name):
        state_dict = None
        print(f"=> Loading checkpoints for {model_name}.")
        if model_name == 'carp' or model_name == 'dino':
            state_dict = self._load_carp_or_dino()
        elif model_name == 'swav' or model_name == 'deepclusterv2' or model_name == 'selav2':
            state_dict = self._load_swav_deepcluster_selav2(model)
        elif model_name == 'mocov3':
            state_dict = self._load_moco_v3()
        elif model_name == "barlowtwins":
            state_dict = self._load_barlowtwins()
        elif model_name == "infomin":
            state_dict = self._load_infomin()
        elif model_name == 'obow':
            state_dict = self._load_obow()
        elif model_name == 'triplet':
            state_dict = self._load_triplet()
        elif model_name == "vicreg":
            state_dict = self._load_vicreg()
        elif model_name == 'pclv2':
            state_dict = self._load_pclv2()
        if state_dict is None:
            print(f"Model's parameters initialized from scratch.")

        msg = model.load_state_dict(state_dict, strict=False)
        print(msg.missing_keys)
        print(f"=> Checkpoints for [{model_name}] successfully loaded!")
        return model


# def checkpoint_loader(model_name, model, pretrained):


#     elif model_name == "barlowtwins":
#         print(f"Loading {model_name}...")
#         state_dict = torch.load(pretrained, map_location='cpu')
#         msg = model.load_state_dict(state_dict, strict=False)
#     elif model_name == "infomin":
#         print(f"Loading {model_name}...")
#         encoder_state_dict = OrderedDict()
#         checkpoint = torch.load(pretrained, map_location='cpu')
#         state_dict = checkpoint['model']
#         for k, v in state_dict.items():
#             k = k.replace('module.', '')
#             if 'encoder' in k:
#                 k = k.replace('encoder.', '')
#                 encoder_state_dict[k] = v
#         msg = model.load_state_dict(encoder_state_dict, strict=True)
#     elif model_name == 'obow':
#         print(f"Loading {model_name}...")
#         checkpoint = torch.load(pretrained, map_location="cpu")
#         state_dict = checkpoint['network']
#         for k in list(state_dict.keys()):
#             if k.startswith('fc'):
#                 del state_dict[k]
#         msg = model.load_state_dict(checkpoint["network"], strict=False)
#     elif model_name == 'triplet':
#         print(f"Loading {model_name}...")
#         checkpoint = torch.load(pretrained, map_location="cpu")
#         state_dict = checkpoint['state_dict']
#         msg = model.load_state_dict(state_dict, strict=False)
#     elif model_name == "vicreg":
#         print(f"Loading {model_name}...")
#         model = torch.hub.load('facebookresearch/vicreg:main', 'resnet50').cuda()
#         model.fc = torch.nn.Identity()
#         msg = 'Model checkpoints downloaded from the internet.'
#     elif model_name == 'supervised_r50':
#         model = models.resnet50()
#         msg = model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth', model_dir='./'))
#         model.fc = torch.nn.Identity()
#     elif model_name == 'selfcls':
#         print("=> loading checkpoint '{}'".format(pretrained))
#         checkpoint = torch.load(pretrained, map_location="cpu")
#         state_dict = checkpoint['state_dict']
#         print(f"=> checkpoint epoch {checkpoint['epoch']}")
#         for k in list(state_dict.keys()):
#             if k.startswith('module.backbone.'):
#                 # remove prefix
#                 state_dict[k[len("module.backbone."):]] = state_dict[k]
#             del state_dict[k]
#         msg = model.load_state_dict(state_dict, strict=False)
#     elif model_name == 'pclv2':
#         checkpoint = torch.load(pretrained, map_location="cpu")
#         # rename pre-trained keys
#         state_dict = checkpoint['state_dict']
#         for k in list(state_dict.keys()):
#             # retain only encoder_q up to before the embedding layer
#             if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
#                 # remove prefix
#                 state_dict[k[len("module.encoder_q."):]] = state_dict[k]
#             # delete renamed or unused k
#             del state_dict[k]

#         msg = model.load_state_dict(state_dict, strict=False)
#     elif model_name == 'scratch':
#         msg = "Running from scratch."
#     else:
#         msg = "Warning: no checkpoint has been loaded!!!"

#     print(msg)
#     print("=> loaded pre-trained model '{}'".format(pretrained))
#     return model