import pulp, argparse, time
import numpy as np
from copy import deepcopy
from pulp import LpStatus
from pulp import value
import math
def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mdp")
    parser.add_argument("--algorithm")
    args = parser.parse_args()

    name = args.mdp
    algo = args.algorithm
    with open(name) as f:
        content = f.readlines()
    name = [content[i][:-1].split() for i in range(len(content))]

    mdp  = {}
    s=[[]]
    for value in name:
        if value[0] == 'transition':
            if 'transition' not in mdp:
                mdp['transition'] = [[[] for a in range(mdp['numActions'])] for s in range(mdp['numStates'])]
            i, j = int(value[1]), int(value[2])
            mdp['transition'][i][j].append((int(value[3]), float(value[4]), float(value[5])))
            temp_list=[i,j,int(value[3]), float(value[4]), float(value[5])]
            s.append(temp_list)
        elif value[0] == 'episodic':
            mdp['type']   = value[0]
        elif value[0] == 'mdptype':
            mdp['type']   = value[1]
        elif value[0] == 'end':
            mdp['end'] = [int(i) for i in value[1:]]
        elif value[0] == 'discount':
            mdp['discount'] = float(value[1])
        else:
            mdp[value[0]] = int(value[1])
    return s,mdp, algo

##################################################################################################

def policyiter(numStates, numActions, transition, discount, tolerance,s):
    state=np.array([i for i in range(numStates)])
    action=np.array([i for i in range(numActions)])
    V=np.zeros(len(state))
    V1=np.ones(len(state))
    p=0
    A=np.zeros(len(action))
    gamma=discount
    Policy=np.zeros(numStates)
    gamma=discount
    count=0
    Policy1=np.ones(len(Policy))
    comparison = Policy == Policy1
    flag=0
    while flag==0:
        V1=deepcopy(V)
        flag=1
        for ss in state:
            Q=[]
            Policy1=deepcopy(Policy)
            for a in action:
                temp=0
                for i in range(len(s)):
                    if s[i][0]==ss and a==s[i][1]:
                        temp=temp+s[i][4]*(s[i][3]+gamma*V1[s[i][2]])
                if temp>V1[ss]:
                    flag=0
                    i=len(s)
                    Policy[ss]=a
                    V[ss]=temp
                    break
    p=0
    V=deepcopy(V1)
    
    return V,Policy1
 


############################################################


s,mdp, algo = init()

numStates  = mdp['numStates']
numActions = mdp['numActions']
transition = mdp['transition']
discount   = mdp['discount']
tolerance  = 1e-12
s.pop(0)
VV,pi = policyiter(numStates, numActions, transition, discount, 1e-9,s)


starti=18
endi=84
x=1
rows=int(math.sqrt(numStates))
path=''
while(starti!=endi):
    if(pi[starti]==0):
        path=path+' '+'N'
        starti=starti-rows
    if(pi[starti]==1):
        path=path+' '+'E'
        starti=starti+1
    if(pi[starti]==2):
        path=path+' '+'S'
        starti=starti+rows
    if(pi[starti]==3):
        path=path+' '+'W'
        starti=starti-1
file1=open("pathfile.txt","w")    
file1. truncate(0)
file1.writelines(path)  
    
    
    
    
    
    
    
    
    