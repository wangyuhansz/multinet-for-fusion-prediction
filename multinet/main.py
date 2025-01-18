import torch
import utils, networks, train, plot


def main():
    args = utils.prepare_args()
    model = networks.get_model(args)

    # Load the lower fidelity model for transfer learning
    if args.mode == 'tl':
        model.load_state_dict(torch.load(utils.get_tl_load_path(args))) 

    # The loading method varies depending on args.mode
    train_data, test_data, val_data, data_range = utils.get_data(args) 

    optimizer = utils.get_optimizer(args.optimizer, model, args.lr)
    loss_fn = utils.get_loss_fn(args.loss_fn)
    device = utils.get_device(args.device)
    save_path, log_path, plot_path = utils.get_save_log_plot_path(args)

    if args.mode == 'train' or args.mode == 'tl':
        train.train_model(model, train_data, test_data, val_data, optimizer, loss_fn, device, 
                          args.epochs, args.val_interval, save_path, log_path)
    
    plot.plot_pre_exp_results(args, test_data, data_range, device, save_path, plot_path, mode='test')
    plot.plot_pre_exp_results(args, val_data, data_range, device, save_path, plot_path, mode='val')


if __name__ == "__main__":
    main()
