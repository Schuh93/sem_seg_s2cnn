import torch, copy
import numpy as np, pytorch_lightning as pl
from s2cnn import s2_near_identity_grid, so3_near_identity_grid, SO3Convolution, S2Convolution, so3_integrate



class ConvNet(pl.LightningModule):
    def __init__(self, hparams, train_data, test_data):
        super().__init__()
        
        self.hparams = copy.deepcopy(hparams)
        self.train_data = train_data
        self.test_data = test_data
        
        self.channels = self.hparams.channels.copy()
        self.kernels = self.hparams.kernels.copy()
        self.strides = self.hparams.strides.copy()
        self.activation_fn = self.hparams.activation_fn
        self.batch_norm = self.hparams.batch_norm
        self.nodes = self.hparams.nodes.copy()
        
        self.loss_function = torch.nn.CrossEntropyLoss()
        
        assert len(self.channels) == len(self.kernels) == len(self.strides)
        possible_activation_fns = ['ReLU', 'LeakyReLU']
        assert self.activation_fn in possible_activation_fns
        
        module_list = []
        self.channels.insert(0,1)
        
        for i in range(len(self.channels)-1):
            in_ch = self.channels[i]
            out_ch = self.channels[i+1]
            module_list.append(torch.nn.Conv2d(in_ch, out_ch, kernel_size=self.kernels[i], stride=self.strides[i]))
            if self.activation_fn == 'ReLU':
                module_list.append(torch.nn.ReLU())
            elif self.activation_fn == 'LeakyReLU':
                module_list.append(torch.nn.LeakyReLU())
            else:
                raise NotImplementedError(f"Activation function must be in {possible_activation_fns}.")
        
        self.conv = torch.nn.Sequential(*module_list)
        
        
        module_list = []
        
        self.nodes.insert(0,self.channels[-1])
        self.nodes.append(10)
        
        for i in range(len(self.nodes) - 1):
            in_nodes = self.nodes[i]
            out_nodes = self.nodes[i+1]
            if self.batch_norm:
                module_list.append(torch.nn.BatchNorm1d(in_nodes))
            module_list.append(torch.nn.Linear(in_features=in_nodes, out_features=out_nodes))
            if i != (len(self.nodes) - 2):
                if self.activation_fn == 'ReLU':
                    module_list.append(torch.nn.ReLU())
                elif self.activation_fn == 'LeakyReLU':
                    module_list.append(torch.nn.LeakyReLU())
                else:
                    raise NotImplementedError(f"Activation function must be in {possible_activation_fns}.")
                
        self.dense = torch.nn.Sequential(*module_list)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        x = torch.mean(x, dim=2)
        x = self.dense(x)
        return x
    
    def loss(self, x, y_true):
        y_pred = self(x)
        loss = self.loss_function(y_pred, y_true)
        return loss
    
    def correct_predictions(self, x, y_true):
        outputs = self(x)
        _, y_pred = torch.max(outputs, 1)
        correct = (y_pred == y_true).long().sum()
        return correct
    
    def prepare_data(self):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_data,
                                           batch_size=self.hparams.train_batch_size,
                                           shuffle=True, num_workers=self.hparams.num_workers)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.test_data,
                                           batch_size=self.hparams.test_batch_size,
                                           shuffle=False, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.test_data,
                                           batch_size=self.hparams.test_batch_size,
                                           shuffle=False, num_workers=self.hparams.num_workers)
    
    def configure_optimizers(self):
        self._optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                            weight_decay=self.hparams.weight_decay, amsgrad=False)
        
        return {'optimizer': self._optimizer}

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)
        correct = self.correct_predictions(x, y)
        
        logs = {'loss': loss.cpu().item()}
        return {'loss': loss, 'train_correct': correct, 'log': logs}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().cpu().item()
        train_correct = torch.stack([x['train_correct'] for x in outputs]).sum().cpu()
        train_acc = train_correct / len(self.train_data)
        
        logs = {'train_loss': avg_loss, 'train_acc': train_acc}    
        return {'train_loss': avg_loss, 'train_acc': train_acc, 'log': logs}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)
        correct = self.correct_predictions(x, y)
        return {'val_loss': loss, 'val_correct': correct}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean().cpu().item()
        val_correct = torch.stack([x['val_correct'] for x in outputs]).sum().cpu()
        val_acc = val_correct / len(self.test_data)

        logs = {'val_loss': avg_loss, 'val_acc': val_acc}        
        return {'val_loss': avg_loss, 'val_acc': val_acc, 'log': logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)
        correct = self.correct_predictions(x, y)
        return {'test_loss': loss, 'test_correct': correct}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean().cpu().item()
        test_correct = torch.stack([x['test_correct'] for x in outputs]).sum().cpu()
        test_acc = test_correct / len(self.test_data)

        logs = {'test_loss': avg_loss, 'test_acc': test_acc}        
        return {'test_loss': avg_loss, 'test_acc': test_acc, 'log': logs}

    def get_progress_bar_dict(self):
        running_train_loss = self.trainer.running_loss.mean()
        avg_training_loss = running_train_loss.cpu().item() if running_train_loss is not None else float('NaN')
        lr = self.hparams.lr

        tqdm_dict = {
            'loss': '{:.2E}'.format(avg_training_loss),
            'lr': '{:.2E}'.format(lr),
        }

        if self.trainer.truncated_bptt_steps is not None:
            tqdm_dict['split_idx'] = self.trainer.split_idx

        if self.trainer.logger is not None and self.trainer.logger.version is not None:
            tqdm_dict['v_num'] = self.trainer.logger.version

        return tqdm_dict

    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    
    
    
class S2ConvNet(pl.LightningModule):
    def __init__(self, hparams, train_data, test_data):
        super().__init__()
        
        self.hparams = copy.deepcopy(hparams)
        self.train_data = train_data
        self.test_data = test_data
        
        self.channels = self.hparams.channels.copy()
        self.bandlimit = self.hparams.bandlimit.copy()
        self.kernel_max_beta = self.hparams.kernel_max_beta.copy()
        self.activation_fn = self.hparams.activation_fn
        self.batch_norm = self.hparams.batch_norm
        self.nodes = self.hparams.nodes.copy()
        
        self.loss_function = torch.nn.CrossEntropyLoss()
        
        if isinstance(self.kernel_max_beta, float):
            self.kernel_max_beta = [self.kernel_max_beta] * (len(self.channels))

        assert len(self.channels) == len(self.bandlimit) == len(self.kernel_max_beta)
        possible_activation_fns = ['ReLU', 'LeakyReLU']
        assert self.activation_fn in possible_activation_fns
        
        
        grid_s2 = s2_near_identity_grid(max_beta=self.kernel_max_beta[0] * np.pi, n_alpha=6, n_beta=1)
        grids_so3 = [
            so3_near_identity_grid(max_beta=max_beta * np.pi, n_alpha=6, n_beta=1, n_gamma=6) for max_beta in self.kernel_max_beta[1:]
        ]
        
        module_list = []
        self.channels.insert(0,1) # greyscale
        self.bandlimit.insert(0,30) # depends on image size
        
        in_ch = self.channels[0]
        out_ch = self.channels[1]
        b_in = self.bandlimit[0]
        b_out = self.bandlimit[1]
        
        module_list.append(S2Convolution(
            nfeature_in=in_ch, nfeature_out=out_ch, b_in=b_in, b_out=b_out, grid=grid_s2
        ))
        
        if self.activation_fn == 'ReLU':
                module_list.append(torch.nn.ReLU())
        elif self.activation_fn == 'LeakyReLU':
            module_list.append(torch.nn.LeakyReLU())
        else:
            raise NotImplementedError(f"Activation function must be in {possible_activation_fns}.")
        
        for i in range(1, len(self.channels)-1):
            in_ch = self.channels[i]
            out_ch = self.channels[i+1]
            b_in = self.bandlimit[i]
            b_out = self.bandlimit[i+1]
            
            module_list.append(
                SO3Convolution(
                    nfeature_in=in_ch,
                    nfeature_out=out_ch,
                    b_in=b_in,
                    b_out=b_out,
                    grid=grids_so3[i-1],
                )
            )
            
            if self.activation_fn == 'ReLU':
                module_list.append(torch.nn.ReLU())
            elif self.activation_fn == 'LeakyReLU':
                module_list.append(torch.nn.LeakyReLU())
            else:
                raise NotImplementedError(f"Activation function must be in {possible_activation_fns}.")
            
        self.conv = torch.nn.Sequential(*module_list)
        
        
        module_list = []
        
        self.nodes.insert(0,self.channels[-1])
        self.nodes.append(10)
        
        for i in range(len(self.nodes) - 1):
            in_nodes = self.nodes[i]
            out_nodes = self.nodes[i+1]
            if self.batch_norm:
                module_list.append(torch.nn.BatchNorm1d(in_nodes))
            module_list.append(torch.nn.Linear(in_features=in_nodes, out_features=out_nodes))
            if i != (len(self.nodes) - 2):
                if self.activation_fn == 'ReLU':
                    module_list.append(torch.nn.ReLU())
                elif self.activation_fn == 'LeakyReLU':
                    module_list.append(torch.nn.LeakyReLU())
                else:
                    raise NotImplementedError(f"Activation function must be in {possible_activation_fns}.")
                
        self.dense = torch.nn.Sequential(*module_list)
        
    def forward(self, x):
        x = self.conv(x)
        x = so3_integrate(x)
        x = self.dense(x)
        return x
    
    def loss(self, x, y_true):
        y_pred = self(x)
        loss = self.loss_function(y_pred, y_true)
        return loss
    
    def correct_predictions(self, x, y_true):
        outputs = self(x)
        _, y_pred = torch.max(outputs, 1)
        correct = (y_pred == y_true).long().sum()
        return correct
    
    def prepare_data(self):
        pass

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_data,
                                           batch_size=self.hparams.train_batch_size,
                                           shuffle=True, num_workers=self.hparams.num_workers)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.test_data,
                                           batch_size=self.hparams.test_batch_size,
                                           shuffle=False, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.test_data,
                                           batch_size=self.hparams.test_batch_size,
                                           shuffle=False, num_workers=self.hparams.num_workers)
    
    def configure_optimizers(self):
        self._optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                            weight_decay=self.hparams.weight_decay, amsgrad=False)
        
        return {'optimizer': self._optimizer}
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)
        correct = self.correct_predictions(x, y)
        
        logs = {'loss': loss.cpu().item()}
        return {'loss': loss, 'train_correct': correct, 'log': logs}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().cpu().item()
        train_correct = torch.stack([x['train_correct'] for x in outputs]).sum().cpu()
        train_acc = train_correct / len(self.train_data)
        
        logs = {'train_loss': avg_loss, 'train_acc': train_acc}    
        return {'train_loss': avg_loss, 'train_acc': train_acc, 'log': logs}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)
        correct = self.correct_predictions(x, y)
        return {'val_loss': loss, 'val_correct': correct}

    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean().cpu().item()
        test_correct = torch.stack([x['val_correct'] for x in outputs]).sum().cpu()
        test_acc = test_correct / len(self.test_data)

        logs = {'val_loss': avg_loss, 'val_acc': test_acc}        
        return {'val_loss': avg_loss, 'val_acc': test_acc, 'log': logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(x, y)
        correct = self.correct_predictions(x, y)
        return {'test_loss': loss, 'test_correct': correct}
    
    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean().cpu().item()
        test_correct = torch.stack([x['test_correct'] for x in outputs]).sum().cpu()
        test_acc = test_correct / len(self.test_data)

        logs = {'test_loss': avg_loss, 'test_acc': test_acc}        
        return {'test_loss': avg_loss, 'test_acc': test_acc, 'log': logs}
    
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean().cpu().item()
        test_correct = torch.stack([x['test_correct'] for x in outputs]).sum().cpu()
        test_acc = test_correct / len(self.test_data)

    def get_progress_bar_dict(self):
        # call .item() only once but store elements without graphs
        running_train_loss = self.trainer.running_loss.mean()
        avg_training_loss = running_train_loss.cpu().item() if running_train_loss is not None else float('NaN')
        lr = self.hparams.lr

        tqdm_dict = {
            'loss': '{:.2E}'.format(avg_training_loss),
            'lr': '{:.2E}'.format(lr),
        }

        if self.trainer.truncated_bptt_steps is not None:
            tqdm_dict['split_idx'] = self.trainer.split_idx

        if self.trainer.logger is not None and self.trainer.logger.version is not None:
            tqdm_dict['v_num'] = self.trainer.logger.version

        return tqdm_dict
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())