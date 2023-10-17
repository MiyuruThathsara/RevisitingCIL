import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet,SimpleCosineIncrementalNet,SimpleVitNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from peft import LoraConfig, get_peft_model


num_workers = 8
# batch_size=128

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True)
        self.args=args

    def after_task(self):
        self._known_classes = self._total_classes
    
    def replace_fc(self,trainloader, model, args):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data,label)=batch
                data=data.cuda()
                label=label.cuda()
                embedding=model.convnet(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list=np.unique(self.train_dataset.labels)
        proto_list = []
        for class_index in class_list:
            # print('Replacing...',class_index)
            data_index=(label_list==class_index).nonzero().squeeze(-1)
            embedding=embedding_list[data_index]
            proto=embedding.mean(0)
            self._network.fc.weight.data[class_index]=proto
        return model
    
    def print_trainable_parameters(self, model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )

    def replace_fc_loss(self,trainloader, model, args):
        model = model.train()

        if self._cur_task == 0:
            # for name, module in model.named_children():
            #     print(name, module)

            print("Before LoRA ...")
            self.print_trainable_parameters(model)
            
            config = LoraConfig(
                            r=16,
                            lora_alpha=16,
                            target_modules=["qkv"],
                            lora_dropout=0.1,
                            bias="none",
                            modules_to_save=["fc"],
                        )
            
            lora_model = get_peft_model(model, config)
            print("After LoRA ...")
            self.print_trainable_parameters(lora_model)

            for name, param in lora_model.named_parameters():
                print(f"Layer: {name}, Parameters: {param.numel()}, Shape: {param.shape}")

            # Optimizer Change
            optimizer = optim.SGD(lora_model.parameters(), momentum=0.9, lr=self.args["init_lr"], weight_decay=self.args["weight_decay"])
            # optimizer = optim.Adam(model.fc.parameters(), lr=self.args["init_lr"], weight_decay=self.args["weight_decay"])
            # optimizer = optim.AdamW(model.fc.parameters(), lr=self.args["init_lr"], weight_decay=self.args["weight_decay"])

            criterion = nn.CrossEntropyLoss()
            # criterion = nn.NLLLoss(reduction='mean')  

            # Scheduler Change
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.args["min_lr"])
            # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

            for epoch in range(self.args["tuned_epoch"]):
                for i, batch in enumerate(trainloader):
                    (_,data,label)=batch
                    data=data.cuda()
                    label=label.cuda()

                    optimizer.zero_grad()

                    outputs=lora_model(data)
                    print(outputs["logits"].shape)
                    loss=criterion(outputs["logits"], label.long())

                    loss.backward()
                    optimizer.step()
                scheduler.step()
                
                y_pred, y_true = self._eval_cnn(self.test_loader)
                cnn_accy = self._evaluate(y_pred, y_true)
                print('Epoch : ', epoch, 'Accuracy (CNN): ', cnn_accy["top1"])
            
            model = lora_model

        else:
            optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=self.args["init_lr"], weight_decay=self.args["weight_decay"])

            criterion = nn.CrossEntropyLoss()
            # criterion = nn.NLLLoss(reduction='mean')  

            # Scheduler Change
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.args["min_lr"])
            # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

            self.print_trainable_parameters(model)

            for name, param in model.named_parameters():
                print(f"Layer: {name}, Parameters: {param.numel()}, Shape: {param.shape}")


            for epoch in range(self.args["tuned_epoch"]):
                for i, batch in enumerate(trainloader):
                    (_,data,label)=batch
                    data=data.cuda()
                    label=label.cuda()

                    optimizer.zero_grad()

                    outputs=model(data)
                    print(outputs["logits"].shape)
                    loss=criterion(outputs["logits"], label.long())

                    loss.backward()
                    optimizer.step()
                scheduler.step()
                
                y_pred, y_true = self._eval_cnn(self.test_loader)
                cnn_accy = self._evaluate(y_pred, y_true)
                print('Epoch : ', epoch, 'Accuracy (CNN): ', cnn_accy["top1"])
            
        # else:
        #     embedding_list = []
        #     label_list = []
        #     with torch.no_grad():
        #         for i, batch in enumerate(trainloader):
        #             print(i)
        #             (_,data,label)=batch
        #             data=data.cuda()
        #             label=label.cuda()
        #             embedding=model.convnet(data)
        #             embedding_list.append(embedding.cpu())
        #             label_list.append(label.cpu())
        #     embedding_list = torch.cat(embedding_list, dim=0)
        #     label_list = torch.cat(label_list, dim=0)

        #     class_list=np.unique(self.train_dataset.labels)
        #     proto_list = []
        #     for class_index in class_list:
        #         # print('Replacing...',class_index)
        #         data_index=(label_list==class_index).nonzero().squeeze(-1)
        #         embedding=embedding_list[data_index]
        #         proto=embedding.mean(0)
        #         self._network.fc.original_module.weight.data[class_index]=proto

            # print("Training Started ...") 
            # for epoch in range(2):
            #     for i, batch in enumerate(trainloader):
            #         (_,data,label)=batch
            #         data=data.cuda()
            #         label=label.cuda()

            #         optimizer.zero_grad()

            #         with torch.no_grad():
            #             embedding=model.convnet(data)
            #         outputs=model.fc(embedding)
            #         print(outputs["logits"].shape)
            #         loss=criterion(outputs["logits"], label.long())

            #         loss.backward()
            #         optimizer.step()
            #     # scheduler.step()
                
            #     # cnn_accy, nme_accy = model.eval_task()
            #     y_pred, y_true = self._eval_cnn(self.test_loader)
            #     cnn_accy = self._evaluate(y_pred, y_true)
            #     print('Epoch : ', epoch, 'Accuracy (CNN): ', cnn_accy["top1"])
                
        return model
   
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes, cur_task=self._cur_task)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", )
        self.train_dataset=train_dataset
        self.data_manager=data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet=data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        
        self._network.to(self._device)
        # self.replace_fc(train_loader_for_protonet, self._network, None)
        self.replace_fc_loss(train_loader_for_protonet, self._network, None)

        
    

   