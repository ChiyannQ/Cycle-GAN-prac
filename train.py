

# for import sth
from options import My_train_opts
from data import create_dataset
from model import create_model
from util.visualizer import Visualizer

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
  
  # visualizer
  visualizer = Visualizer(opt) 
  
  total_iters = 0  # record the num of loop
  
  for epoch in range(opt.epoch_count,opt.niter + opt.niter_decay + 1):    # control the loop by modify the epoch_count
  
  # niter+niter_decay :I don't know
  
    epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
  
  
    for step,data in enumerate(dataset):
      #train
      total_iters += opt.batch_size
      epoch_iter += opt.batch_size
      model.set_input(data)  # unpack the dataset
      model.optimize_paras()  
      
      if total_iters % opt.disp_freq == 0:
        # disp image on visdom and save it to a HTML file
        save_result = total_iters % opt.update_html_freq == 0
        model.compute_visuals()
        visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        
      if total_iters % opt.loss_freq ==0 :
        # print loss
        losses = model.get_current_losses()
        t_comp = (time.time() - iter_start_time) / opt.batch_size
        visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
        if opt.display_id > 0:
          visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
        
      if total_iters % opt.save_freq == 0:
        # save network
        print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
        save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
        model.save_networks(save_suffix)
      
    
      # cache model every <save_epoch_freq> epochs
    if epoch % opt.save_epoch_freq == 0:              
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')  # .save_networks 
            model.save_networks(epoch)  # .save_networks
      
    print('end of epoch %d/%d' % (epoch,opt.niter + opt.niter_decay))
    #.update_learning_rate
    model.update_learnign_rate()  
      
   # model:.save_networks,.update_learning_rate,set_input,optimize_paras(),setup(),get_current_losses(),compute_visuals(),
  #get_current_visuals()
  # visualizer:
  #opt:
  
