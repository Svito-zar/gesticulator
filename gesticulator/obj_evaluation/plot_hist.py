"""
Plots the experimental results after calculating motion statistics
Expects that calc_distance was run before this script

@author: Taras Kucherenko
"""

import matplotlib.pyplot as plt
import csv
import numpy as np

def read_joint_names(filename):
    with open(filename, 'r') as f:
        org = f.read()
        joint_names = org.split(',')

    return joint_names

def read_csv(filename):

    x=[]
    y=[]
    total_sum = 0
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip the headers
        for row in reader:
            x.append(float(row[0]))
            next_val = float(row[-2]) + float(row[-5])
            y.append(next_val)
            total_sum += next_val

            # Crop on 50
            if float(row[0]) >= 50:
                break

    return np.array(x), np.array(y) / total_sum


plt.rcParams.update({'font.size': 36})

plt.rcParams.update({'pdf.fonttype': 42, 'font.family':'sans'})

type = "vel"

original_filename = 'result/GT/hmd_'+type+'_1.csv'

x,original = read_csv(original_filename)

# Get Full model results
feature_filename = 'result/FullModel/hmd_' + type + '_1.csv'
_, full_model = read_csv(feature_filename)


# Get No Autoregr
_,no_autoregr = read_csv('result/NoAutoregression/hmd_' + type + '_1.csv')

# Get No FiLM
_,no_FiLM = read_csv('result/NoFiLM/hmd_' + type + '_1.csv')

_,no_Vel_pen = read_csv('result/NoVelPenalty/hmd_' + type + '_1.csv')


# Get No Speech
_,no_speech = read_csv('result/NoSpeech/hmd_' + type + '_1.csv')

# Get No Text
_,no_Text = read_csv('result/NoText/hmd_' + type + '_1.csv')

_,no_PCA = read_csv('result/NoPCA/hmd_' + type + '_1.csv')


plt.plot(x,original,linewidth=7.0, label='Ground Truth', color='Purple')

plt.plot(x,full_model , label='Proposed Model',linewidth=7.0)

plot_type = 2

if plot_type == 1:

    plt.plot(x,no_autoregr , label='No Autoregression',linewidth=7.0, color='C6')

    plt.plot(x,no_FiLM , label='No FiLM',linewidth=7.0, color='C1')

    plt.plot(x,no_Vel_pen , label='No Velocity Loss',linewidth=7.0, color='C3')

else:

    plt.plot(x,no_PCA , label='No PCA',linewidth=7.0, color='Blue')

    plt.plot(x, no_Text , label='No Text',linewidth=7.0, color='C2')

    plt.plot(x, no_speech , label='No Speech',linewidth=7.0, color='C5')



plt.xlabel("Velocity ($cm$/$s$)", size=50)
plt.ylabel('Frequency (%)', size=50)

plt.xticks(np.arange(0,51,5), size=50)

leg = plt.legend(prop={'size': 42})

figure = plt.gcf() # get current figure
figure.set_size_inches(23, 13)
plt.savefig("myplot.pdf") #, dpi = 100)