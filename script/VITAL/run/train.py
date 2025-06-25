if overwrite:
    
    # ------------------------- ready training for clip -------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config_dict['init_lr'],
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.9,         
        patience=config_dict['patience'],    
        min_lr=1e-20,        
        threshold=1e-4,      
        cooldown=20          
    )
    
    for i in range(config_dict['num_saves']):  
        if config_dict['alpha_init'] is not None:
            alpha_init = config_dict['alpha_init'] # use customized fixed alpha_init
        else:
            alpha_init = None if i == 0 else alpha_tmp # automatically recalibrate alpha after 50 epochs
        
        train_losses_tmp, test_losses_tmp, alpha_tmp = train_vital(model, 
                                                                    train_dataloader,
                                                                    test_dataloader, 
                                                                    optimizer, 
                                                                    scheduler,
                                                                    num_epochs = config_dict['num_epochs'], 
                                                                    target_type = config_dict['clip_target_type'],
                                                                    train_type = config_dict['train_type'],
                                                                    alpha_init = alpha_init,
                                                                    beta = config_dict['beta'],
                                                                    es_patience = config_dict['es_patience'],
                                                                    target_ratio = config_dict['target_ratio']
                                                        )
        
    
        train_losses = train_losses + train_losses_tmp
        test_losses = test_losses + test_losses_tmp
        if i == 0:
            train_losses = train_losses[11:]
            test_losses = test_losses[11:]
        
        # every num_epochs, evaluate the model
        model.eval()
        if config_dict['3d']:
            train_eval_metrics_ts2txt = eval_clip3d_ts2txt(model, evalclipts2txt_train)
            test_eval_metrics_ts2txt = eval_clip3d_ts2txt(model, evalclipts2txt_test)
            train_eval_metrics_txt2ts = eval_clip3d_txt2ts(model, evalcliptxt2ts_train)
            test_eval_metrics_txt2ts = eval_clip3d_txt2ts(model, evalcliptxt2ts_test)

        else:
            train_eval_metrics_ts2txt = eval_clip_ts2txt(model, evalclipts2txt_train)
            test_eval_metrics_ts2txt = eval_clip_ts2txt(model, evalclipts2txt_test)
            train_eval_metrics_txt2ts = eval_clip_txt2ts(model, evalcliptxt2ts_train)
            test_eval_metrics_txt2ts = eval_clip_txt2ts(model, evalcliptxt2ts_test)


        #  ------ save model and losses ------
        train_eval_metrics_ts2txt_list.append(train_eval_metrics_ts2txt)
        test_eval_metrics_ts2txt_list.append(test_eval_metrics_ts2txt)
        eval_dict_ts2txt = {'train_losses': train_losses,
                    'test_losses': test_losses,
                    'train_evals': train_eval_metrics_ts2txt_list,
                    'test_evals': test_eval_metrics_ts2txt_list }
        eval_dict_eng = eng_eval_metrics(eval_dict_ts2txt, binary=True, plot=False)
        print("-" * 70)
        print(f"Metric     |  Training  |  Testing")
        print("-" * 70)
        print(f"F1        |   {eval_dict_eng['train_f1'][-1]:.3f}   |   {eval_dict_eng['test_f1'][-1]:.3f}")
        print(f"Precision |   {eval_dict_eng['train_precision'][-1]:.3f}   |   {eval_dict_eng['test_precision'][-1]:.3f}")
        print(f"Recall    |   {eval_dict_eng['train_recall'][-1]:.3f}   |   {eval_dict_eng['test_recall'][-1]:.3f}")
        print(f"AUROC     |   {eval_dict_eng['train_auroc'][-1]:.3f}   |   {eval_dict_eng['test_auroc'][-1]:.3f}")
        print(f"AUPRC     |   {eval_dict_eng['train_auprc'][-1]:.3f}   |   {eval_dict_eng['test_auprc'][-1]:.3f}")
        print("-" * 70)

        train_eval_metrics_txt2ts_list.append(train_eval_metrics_txt2ts)
        test_eval_metrics_txt2ts_list.append(test_eval_metrics_txt2ts)
        eval_dict_txt2ts = {'train_losses': train_losses,
                    'test_losses': test_losses,
                    'train_evals': train_eval_metrics_txt2ts_list,
                    'test_evals': test_eval_metrics_txt2ts_list }
        eval_dict_eng = eng_eval_metrics(eval_dict_txt2ts, binary=False, plot=False)
        print("-" * 70)
        print(f"Metric     |  Training  |  Testing")
        print("-" * 70)
        print(f"F1        |   {eval_dict_eng['train_f1'][-1]:.3f}   |   {eval_dict_eng['test_f1'][-1]:.3f}")
        print(f"Precision |   {eval_dict_eng['train_precision'][-1]:.3f}   |   {eval_dict_eng['test_precision'][-1]:.3f}")
        print(f"Recall    |   {eval_dict_eng['train_recall'][-1]:.3f}   |   {eval_dict_eng['test_recall'][-1]:.3f}")
        print(f"AUROC     |   {eval_dict_eng['train_auroc'][-1]:.3f}   |   {eval_dict_eng['test_auroc'][-1]:.3f}")
        print(f"AUPRC     |   {eval_dict_eng['train_auprc'][-1]:.3f}   |   {eval_dict_eng['test_auprc'][-1]:.3f}")
        print("-" * 70)

        torch.save(model.state_dict(), config_dict['output_dir']+'/model.pth')
        torch.save(eval_dict_ts2txt, config_dict['output_dir']+'/evals_clip_ts2txt.pth')
        torch.save(eval_dict_txt2ts, config_dict['output_dir']+'/evals_clip_txt2ts.pth')
        
        
        # Eval CLIP
        _ = eng_eval_metrics(eval_dict_ts2txt, binary=False, plot=True, plot_confusion_matrices=True)
        _ = eng_eval_metrics(eval_dict_txt2ts, binary=False, plot=True, plot_confusion_matrices=True)
        # for y_col in config_dict['txt2ts_y_cols']:
        #     try:
        #         text_levels = list(df_test[y_col].unique())
        #         _ = net_emb(df_test, model, config_dict,
        #                     top=100,
        #                     y_col = y_col,
        #                     text_levels = text_levels)
        #         # _ = net_emb_w_text(df_test, model, config_dict,
        #         #                 top=100,
        #         #                 y_col = y_col,
        #         #                 text_levels = text_levels)
        #     except Exception as e:
        #         print(f"Error plot network embedding for {y_col}")
        #         continue
        
        if config_dict['train_type'] != 'clip':
            # Eval VAE
            plot_reconstructions(model, 
                                df=df_test, 
                                config_dict = config_dict, 
                                title="Test Data Reconstructions")
            
            # Eval Generation
            if len(config_dict['txt2ts_y_cols'])<3:
                # viz_generation_conditional(df_train, model, config_dict)
                viz_generation_conditional(df_test, model, config_dict)
        
        
        if len(train_losses_tmp) < config_dict['num_epochs']: # Early stopping
            break
        
else:
    model.eval()
    # eval clip
    eval_dict_ts2txt = torch.load(config_dict['output_dir']+'/evals_clip_ts2txt.pth', map_location=torch.device(device), weights_only=False)
    eval_dict_txt2ts = torch.load(config_dict['output_dir']+'/evals_clip_txt2ts.pth', map_location=torch.device(device), weights_only=False)
    eval_dict_eng = eng_eval_metrics(eval_dict_ts2txt, binary=True, plot=True, plot_confusion_matrices=True)
    eval_dict_eng = eng_eval_metrics(eval_dict_txt2ts, binary=False, plot=True, plot_confusion_matrices=True)
    
    for y_col in config_dict['txt2ts_y_cols']:
        try:
            text_levels = list(df_train[y_col].unique())
            _ = net_emb(df_test, model, config_dict,
                        top=100,
                        y_col = y_col,
                        text_levels = text_levels)
        except Exception as e:
            print(f"Error plot network embedding for {y_col}")
            continue
        
        
    # eval vae
    plot_reconstructions(model, 
                        df = df_test, 
                        config_dict = config_dict, 
                        title="Test Data Reconstructions")
    # Eval Generation
    if len(config_dict['txt2ts_y_cols'])<3:
        # viz_generation_marginal(df_train, model, config_dict)
        viz_generation_conditional(df_train, model, config_dict)
        # viz_generation_marginal(df_test, model, config_dict)
        viz_generation_conditional(df_test, model, config_dict)
    
    train_eval_metrics_ts2txt_list = eval_dict_ts2txt['train_evals']
    test_eval_metrics_ts2txt_list = eval_dict_ts2txt['test_evals']
    train_eval_metrics_txt2ts_list = eval_dict_txt2ts['train_evals']
    test_eval_metrics_txt2ts_list = eval_dict_txt2ts['test_evals']
    train_losses = eval_dict_ts2txt['train_losses']
    test_losses = eval_dict_ts2txt['test_losses']

    