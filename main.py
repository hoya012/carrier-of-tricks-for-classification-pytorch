import os, sys
import torch
import torch.nn as nn
import torchvision

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PATH + '/../..')

from option import get_args
from learning.trainer import Trainer
from learning.evaluator import Evaluator
from utils import get_model, make_optimizer, make_scheduler, make_dataloader, plot_learning_curves

def main():
    args = get_args()
    torch.manual_seed(args.seed)

    shape = (224,224,3)    

    """ define dataloader """
    train_loader, valid_loader, test_loader = make_dataloader(args)

    """ define model architecture """
    model = get_model(args, shape, args.num_classes)

    if torch.cuda.device_count() >= 1:
        print('Model pushed to {} GPU(s), type {}.'.format(torch.cuda.device_count(), torch.cuda.get_device_name(0)))
        model = model.cuda() #nn.DataParallel(model).cuda()
    else:
        raise ValueError('CPU training is not supported')

    #print(model)
    #print(iter(train_loader).next())

    """ define loss criterion """
    criterion = nn.CrossEntropyLoss().cuda()

    """ define optimizer """
    optimizer = make_optimizer(args, model)

    """ define learning rate scheduler """
    scheduler = make_scheduler(args, optimizer)
    
    """ define trainer, evaluator, result_dictionary """
    result_dict = {'args':vars(args), 'epoch':[], 'train_loss' : [], 'train_acc' : [], 'val_loss' : [], 'val_acc' : [], 'test_acc':[]}
    trainer = Trainer(model, criterion, optimizer, scheduler)
    evaluator = Evaluator(model, criterion)
    evaluator.save(result_dict)

    """ define training loop """
    for epoch in range(args.epochs):
        result_dict['epoch'] = epoch
        result_dict = trainer.train(train_loader, epoch, args, result_dict)
        result_dict = evaluator.evaluate(valid_loader, epoch, args, result_dict)
        evaluator.save(result_dict)
        plot_learning_curves(result_dict, epoch, args)

    result_dict = evaluator.test(test_loader, args, result_dict)
    evaluator.save(result_dict)
    print(result_dict)

    """ save model checkpoint """
    model.save()

if __name__ == '__main__':
    main()