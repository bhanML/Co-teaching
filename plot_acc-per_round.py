import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib.ticker as ticker
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--round', type = str, help = 'dir to save result txt files', default =1)
parser.add_argument('--tau_ratio', type = float, help = 'dir to save result txt files', default =1.0)
args = parser.parse_args()

plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
 
colors={}
colors['coteaching']='royalblue'
colors['decoupling']='black'
colors['forward']='yellow'
colors['forget']='sandybrown'
colors['normal']='crimson'
colors['s_model']='fuchsia'
colors['bootstrap-soft']='saddlebrown'
colors['bootstrap-hard']='saddlebrown'

line_styles={}
line_styles['coteaching']='-'
line_styles['decoupling']='-'
line_styles['forward']='-'
line_styles['forget']='-'
line_styles['normal']='-'
line_styles['s_model']='-'
line_styles['bootstrap-soft']='--'
line_styles['bootstrap-hard']='-'

names={}
names['normal']='Normal'
names['forget']='MentorNet'
names['coteaching']='Co-teaching'
names['decoupling']='Decoupling'
names['forward']='F-correction'
names['s_model']='S-model'
names['bootstrap-soft']='Bootstrap-soft'
names['bootstrap-hard']='Bootstrap-hard'

names_main_body={}
names_main_body['normal']='Normal'
names_main_body['forget']='MentorNet'
names_main_body['coteaching']='Co-teaching'
names_main_body['decoupling']='Decoupling'
names_main_body['forward']='F-correction'
names_main_body['s_model']='S-model'
names_main_body['bootstrap-hard']='Bootstrap'

ROUND='round%s' % args.round

def get_data(dataset, model_type, noise_type, noise_rate,root='results',with_pure_ratio=False):  
    if noise_type=='comple':
        datafile='text_rebuttal/{0}/tau{4}/{1}/{2}/{1}_{2}_{3}.txt'.format(root, dataset, model_type, noise_type, tau_ratio)
    else:
        datafile='{0}/{1}/{2}/{1}_{2}_{3}_{4}.txt'.format(root, dataset, model_type, noise_type, noise_rate, args.tau_ratio)

    if not os.path.exists(datafile):
        None
    else:
        with open(datafile, 'r') as f:
            lines=f.readlines()
            train_sizes=[int(line.split()[0].replace(':','')) for line in lines]
            if model_type=='coteaching':
                train_scores=[float(line.split()[1]) for line in lines]
                train_scores2=[float(line.split()[2]) for line in lines]
                test_scores=[float(line.split()[3]) for line in lines]
                test_scores2=[float(line.split()[4]) for line in lines]
                if with_pure_ratio:
                    pure_ratio=[float(line.split()[5]) for line in lines]
                    pure_ratio2=[float(line.split()[6]) for line in lines]
                    #print train_scores, train_scores2, test_scores, test_scores2
                    return np.array(train_sizes), np.array(train_scores)/100, np.array(train_scores2)/100, np.array(test_scores)/100, np.array(test_scores2)/100, np.array(pure_ratio)/100, np.array(pure_ratio2)/100
                else:
                    return np.array(train_sizes), np.array(train_scores)/100, np.array(train_scores2)/100, np.array(test_scores)/100, np.array(test_scores2)/100
            elif model_type=='decoupling':
                train_scores=[float(line.split()[1]) for line in lines]
                train_scores2=[float(line.split()[2]) for line in lines]
                test_scores=[float(line.split()[3]) for line in lines]
                test_scores2=[float(line.split()[4]) for line in lines]
                if with_pure_ratio:
                    pure_ratio=[float(line.split()[5]) for line in lines]
                    #print train_scores, train_scores2, test_scores, test_scores2
                    return np.array(train_sizes), np.array(train_scores)/100, np.array(train_scores2)/100, np.array(test_scores)/100, np.array(test_scores2)/100, np.array(pure_ratio)/100
                else:
                    return np.array(train_sizes), np.array(train_scores)/100, np.array(train_scores2)/100, np.array(test_scores)/100, np.array(test_scores2)/100
            elif model_type=='forget':	
                train_scores=[float(line.split()[1]) for line in lines]
    	        test_scores=[float(line.split()[2]) for line in lines]
                if with_pure_ratio:
                    pure_ratio=[float(line.split()[3]) for line in lines]
                    #print train_scores, test_scores
                    return np.array(train_sizes), np.array(train_scores)/100, np.array(test_scores)/100, np.array(pure_ratio)/100 
                else:
                    return np.array(train_sizes), np.array(train_scores)/100, np.array(test_scores)/100 
            else:	
                train_scores=[float(line.split()[1]) for line in lines]
    	        test_scores=[float(line.split()[2]) for line in lines]
                #print train_scores, test_scores
                return np.array(train_sizes), np.array(train_scores)/100, np.array(test_scores)/100 

def plot_acc(use_title=False, ylim=None, mode='full',loc="lower right",with_pure_ratio=False,
			dataset='mnist', model_types=['normal','forget','coteaching'], curve_dir='figures/acc/',
			noise_type='symmetric', noise_rate=0.45, with_trainscores=False, with_hline=False, dpi=100):
    save_dir=curve_dir
    save_dir2=curve_dir+'/train_and_test'
    #plt.figure(figsize=(12,10))
    fig, ax = plt.subplots(figsize=(12, 10))
    if with_hline:
        plt.axhline(0.1,linestyle='-.',color='c', linewidth=3)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    if noise_type=='symmetric':
        title='(%s, %s, %s'%(dataset, 'symmetry', str(int(float(noise_rate)*100)))+'%)' 
    if noise_type=='pairflip':
        title='(%s, %s, %s'%(dataset, 'pair', str(int(float(noise_rate)*100)))+'%)' 
    #title='(%s, %s, %s'%(dataset, noise_type, str(int(float(noise_rate)*100)))+'%)' 
    filename='%s-%s-%s-%s'%(dataset, noise_type, str(int(float(noise_rate)*100)), mode)
    print title
    if use_title:
        ax.set_title(title, fontsize=26)
    if ylim is not None:
        plt.ylim(*ylim)
    ax.set_xlabel("Epoch", fontsize=26)
    ax.set_ylabel("Test Accuracy", fontsize=26)

    
    models_data={}
    for model_type in model_types:
	if model_type=='coteaching' or model_type=='decoupling':
            if model_type=='coteaching':
                if with_pure_ratio:
                    train_sizes, train_scores, train_scores2, test_scores1, test_scores2, pure_ratio, pure_ratio2=get_data(dataset=dataset, model_type=model_type, noise_type=noise_type, noise_rate=noise_rate, with_pure_ratio=with_pure_ratio)
                else:
                    train_sizes, train_scores, train_scores2, test_scores1, test_scores2=get_data(dataset=dataset, model_type=model_type, noise_type=noise_type, noise_rate=noise_rate, with_pure_ratio=with_pure_ratio)
            if model_type=='decoupling':
                if with_pure_ratio:
                    train_sizes, train_scores, train_scores2, test_scores1, test_scores2, pure_ratio=get_data(dataset=dataset, model_type=model_type, noise_type=noise_type, noise_rate=noise_rate, with_pure_ratio=with_pure_ratio)
                else:
                    train_sizes, train_scores, train_scores2, test_scores1, test_scores2=get_data(dataset=dataset, model_type=model_type, noise_type=noise_type, noise_rate=noise_rate, with_pure_ratio=with_pure_ratio)

            models_data[model_type]=(train_sizes, train_scores, train_scores2, test_scores1,test_scores2)
	    mean_last = (np.mean(test_scores1[-10:])+np.mean(test_scores2[-10:]))/float(2.0)
	    std_last = (np.std(test_scores1[-10:])+np.std(test_scores2[-10:]))/float(2.0)
	    print dataset, model_type, noise_type, noise_rate, 'mean+std: %s+%s'%(mean_last*100, std_last*100)
            if with_trainscores:
                # plot train acc 1 
                plt.plot(models_data[model_type][0], models_data[model_type][1], 
                         color=colors[model_type],linestyle=line_styles[model_type], marker='o',
                         label=names[model_type]+' train', linewidth=3)
                # plot test acc 1 
                plt.plot(models_data[model_type][0], models_data[model_type][3], 
                     color=colors[model_type],linestyle=line_styles[model_type],
                     label=names[model_type]+' test', linewidth=3)
	    else:
                # plot test acc 1 
                if mode=='part':
                    test_acc_avg=(models_data[model_type][3]+models_data[model_type][4])/float(2)
                    plt.plot(models_data[model_type][0], test_acc_avg, 
                         color=colors[model_type],linestyle=line_styles[model_type],
                         label=names_main_body[model_type], linewidth=3)
                if mode=='full':
                    plt.plot(models_data[model_type][0], models_data[model_type][3], 
                         color=colors[model_type],linestyle=line_styles[model_type],
                         label=names[model_type]+'-1', linewidth=3)
                    plt.plot(models_data[model_type][0], models_data[model_type][4], 
                         color=colors[model_type],linestyle='--',
                         label=names[model_type]+'-2', linewidth=3)
	else:
            train_sizes, train_scores, test_scores=get_data(dataset=dataset, model_type=model_type, noise_type=noise_type, noise_rate=noise_rate, with_pure_ratio=with_pure_ratio)
            if model_type=='forward':
                train_scores = train_scores*100
                test_scores = test_scores*100
            models_data[model_type]=(train_sizes, train_scores, test_scores)
	    mean_last = np.mean(test_scores[-10:])
	    print dataset, model_type, noise_type, noise_rate, mean_last
            if with_trainscores:
                # plot train acc 1 
                plt.plot(models_data[model_type][0], models_data[model_type][1], 
                         color=colors[model_type],linestyle=line_styles[model_type], marker='o',
                         label=names[model_type]+' train', linewidth=3)
                # plot test acc 
                plt.plot(models_data[model_type][0], models_data[model_type][2], 
                         color=colors[model_type],linestyle=line_styles[model_type],
                         label=names[model_type]+' test', linewidth=3)
	    else:
                # plot test acc 
                if mode=='full':
                    plt.plot(models_data[model_type][0], models_data[model_type][2], 
                         color=colors[model_type],linestyle=line_styles[model_type],
                         label=names[model_type], linewidth=3)
                if mode=='part':
                    plt.plot(models_data[model_type][0], models_data[model_type][2], 
                         color=colors[model_type],linestyle=line_styles[model_type],
                         label=names_main_body[model_type], linewidth=3)

    if not os.path.exists(save_dir):
        os.system('mkdir -p %s'%save_dir)
    legend=plt.legend(loc=loc, fontsize=26)
    frame = legend.get_frame() 
    frame.set_alpha(1) 
    frame.set_facecolor('none')
    if with_trainscores:
        plt.savefig('%s/%s.png'%(save_dir2,filename),dpi=dpi)
        plt.savefig('%s/%s.pdf'%(save_dir2,filename),dpi=dpi)
    else:
        plt.savefig('%s/%s.png'%(save_dir,filename),dpi=dpi)
        plt.savefig('%s/%s.pdf'%(save_dir,filename),dpi=dpi)
    return plt

def compare_acc(datasets=['mnist','cifar10'],model_types=['normal', 'forget','coteaching'], noise_rates=[0.45, 0.5], noise_type='symmetric', with_trainscores=False, with_hline=False, ylim=[0.0, 1.0], mode='full', loc="lower right"): #mode = full or part, full for appendix, part for main body
    for dataset in datasets:
	for noise_rate in noise_rates:
            plot_acc(use_title=True, mode=mode, ylim=ylim, 
    	       		dataset=dataset, model_types=model_types, 
    			noise_type=noise_type, noise_rate=noise_rate,loc=loc,
			with_trainscores=with_trainscores, with_hline=with_hline)

if __name__=='__main__':
    # plot test curves noise 0.45 0.5
    model_types_appendix=['coteaching']

    # full plot for appendix
    print '==>ploting full figures'
    compare_acc(datasets=['mnist'],model_types=model_types_appendix, noise_rates=[0.2], noise_type='symmetric', with_trainscores=False, ylim=[0.86], mode='full')
    compare_acc(datasets=['mnist'],model_types=model_types_appendix, noise_rates=[0.45], noise_type='pairflip', with_trainscores=False, ylim=[0.0], mode='full')
    compare_acc(datasets=['mnist'],model_types=model_types_appendix, noise_rates=[0.5], noise_type='symmetric', with_trainscores=False, ylim=[0.55 ], mode='full')

    compare_acc(datasets=['cifar10'],model_types=model_types_appendix, noise_rates=[0.2], noise_type='symmetric', with_trainscores=False, ylim=[0.50, 0.85], mode='full')
    compare_acc(datasets=['cifar10'],model_types=model_types_appendix, noise_rates=[0.45], noise_type='pairflip', with_trainscores=False, ylim=[0.0, 0.80], mode='full')
    compare_acc(datasets=['cifar10'],model_types=model_types_appendix, noise_rates=[0.5,], noise_type='symmetric', with_trainscores=False, ylim=[0.40, 0.80], mode='full')

    compare_acc(datasets=['cifar100'],model_types=model_types_appendix, noise_rates=[0.2], noise_type='symmetric', with_trainscores=False, ylim=[0.0, 0.65], mode='full')
    compare_acc(datasets=['cifar100'],model_types=model_types_appendix, noise_rates=[0.45], noise_type='pairflip', with_trainscores=False, ylim=[0.0, 0.35], mode='full')
    compare_acc(datasets=['cifar100'],model_types=model_types_appendix, noise_rates=[0.5,], noise_type='symmetric', with_trainscores=False, ylim=[0.00, 0.55], mode='full')

