

# for import sth
from options import My_train_opts
from data import create_dataset
from model import create_model


# begin
if _name_ == '_main_':
  #load the opt
  opt = My_train_opts()
  
  # create dataset,given the options
  dataset = create_dataset(opt)
  
  # load the model
  model = create_model(opt)
  
  # set up for model
  #model.setup(opt)
  
  
  total_iters = 0  # record the num of loop
  
  for epoch in range(opt.epoch_count,opt.niter + opt.niter_decay + 1):    # control the loop by modify the epoch_count
  
  # niter+niter_decay :I don't know
    epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
  
  
    for step,data in enumerate(dataset):
      #train
      total_iters += opt.batch_size
      epoch_iter += opt.batch_size
      model.set_input(data)  # unpack the dataset
      model.optimize_paras()  # 
    if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')  # .save_networks 
            model.save_networks(epoch)  # .save_networks
      
    print('end of epoch %d/%d' % (epoch,opt.niter + opt.niter_decay))
    model.update_learnign_rate()  #.update_learning_rate
      
   # model:.save_networks,.update_learning_rate,set_input,optimize_paras(),setup(),
