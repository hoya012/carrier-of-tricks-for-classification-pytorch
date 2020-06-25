import torch
from utils import AverageMeter, accuracy
from learning.smoothing import LabelSmoothing
from learning.mixup import MixUpWrapper, NLLMultiLabelSmooth

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, data_loader, epoch, args, result_dict):
        total_loss = 0
        count = 0
        
        losses = AverageMeter()
        top1 = AverageMeter()

        self.model.train()

        if args.mixup > 0.0:
            self.criterion = NLLMultiLabelSmooth(args.label_smooth)
            data_loader = MixUpWrapper(args.num_classes, args.mixup, data_loader)
        elif args.label_smooth > 0.0:
            self.criterion = LabelSmoothing(args.label_smooth)

        for batch_idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            if len(labels.size()) > 1:
                labels = torch.argmax(labels, axis=1)

            prec1, prec3 = accuracy(outputs.data, labels, topk=(1, 3))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.tolist()
            count += labels.size(0)

            if batch_idx % args.log_interval == 0:
                if args.mixup > 0.0:
                    _s = str(len(str(len(data_loader.dataloader.sampler))))
                    ret = [
                    ('epoch: {:0>3} [{: >' + _s + '}/{} ({: >3.0f}%)]').format(epoch, count, len(data_loader.dataloader.sampler),\
                     100 * count / len(data_loader.dataloader.sampler)),
                    'train_loss: {: >4.2e}'.format(total_loss / count),
                    'train_accuracy : {:.2f}%'.format(top1.avg)
                    ]
                else:
                    _s = str(len(str(len(data_loader.sampler))))
                    ret = [
                        ('epoch: {:0>3} [{: >' + _s + '}/{} ({: >3.0f}%)]').format(epoch, count, len(data_loader.sampler), 100 * count / len(data_loader.sampler)),
                        'train_loss: {: >4.2e}'.format(total_loss / count),
                        'train_accuracy : {:.2f}%'.format(top1.avg)
                    ]
                print(', '.join(ret))

        self.scheduler.step()
        result_dict['train_loss'].append(losses.avg)
        result_dict['train_acc'].append(top1.avg)

        return result_dict