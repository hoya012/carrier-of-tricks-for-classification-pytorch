import os, sys, time
import numpy as np
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
        model = model.cuda() 
    else:
        raise ValueError('CPU training is not supported')

    """ define loss criterion """
    criterion = nn.CrossEntropyLoss().cuda()

    """ define optimizer """
    optimizer = make_optimizer(args, model)

    """ define learning rate scheduler """
    scheduler = make_scheduler(args, optimizer)

    """ define loss scaler for automatic mixed precision """
    scaler = torch.cuda.amp.GradScaler()

    """ define trainer, evaluator, result_dictionary """
    result_dict = {'args':vars(args), 'epoch':[], 'train_loss' : [], 'train_acc' : [], 'val_loss' : [], 'val_acc' : [], 'test_acc':[]}
    trainer = Trainer(model, criterion, optimizer, scheduler, scaler)
    evaluator = Evaluator(model, criterion)

    train_time_list = []
    valid_time_list = []

    if args.evaluate:
        """ load model checkpoint """
        model.load()
        result_dict = evaluator.test(test_loader, args, result_dict)
    else:
        evaluator.save(result_dict)

        best_val_acc = 0.0
        """ define training loop """
        for epoch in range(args.epochs):
            result_dict['epoch'] = epoch

            torch.cuda.synchronize()
            tic1 = time.time()

            result_dict = trainer.train(train_loader, epoch, args, result_dict)

            torch.cuda.synchronize()
            tic2 = time.time() 
            train_time_list.append(tic2 - tic1)

            torch.cuda.synchronize()
            tic3 = time.time()

            result_dict = evaluator.evaluate(valid_loader, epoch, args, result_dict)

            torch.cuda.synchronize()
            tic4 = time.time() 
            valid_time_list.append(tic4 - tic3)

            if result_dict['val_acc'][-1] > best_val_acc:
                print("{} epoch, best epoch was updated! {}%".format(epoch, result_dict['val_acc'][-1]))
                best_val_acc = result_dict['val_acc'][-1]
                model.save(checkpoint_name='best_model')

            evaluator.save(result_dict)
            plot_learning_curves(result_dict, epoch, args)

        result_dict = evaluator.test(test_loader, args, result_dict)
        evaluator.save(result_dict)

        """ save model checkpoint """
        model.save(checkpoint_name='last_model')

        """ calculate test accuracy using best model """
        model.load(checkpoint_name='best_model')
        result_dict = evaluator.test(test_loader, args, result_dict)
        evaluator.save(result_dict)

    print(result_dict)
    np.savetxt('train_time_amp{}.csv'.format(args.amp), train_time_list, delimiter=',', fmt='%s')
    np.savetxt('valid_time_amp{}.csv'.format(args.amp), valid_time_list, delimiter=',', fmt='%s')

if __name__ == '__main__':
    main()