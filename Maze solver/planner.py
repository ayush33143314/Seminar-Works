import pulp, argparse, time
import numpy as np
from copy import deepcopy
from pulp import LpStatus
from pulp import value

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

def linearProgram(numStates, numActions, transition, discount, tolerance,s):
    states=np.array([i for i in range(numStates)])
    actions=np.array([i for i in range(numActions)])
    prob=pulp.LpProblem("ValueFn",pulp.LpMinimize)
    decision_variables=[]
    for state in range(len(states)):
        variable=str('V'+str(state))
        variable=pulp.LpVariable(str(variable))
        decision_variables.append(variable)
    total_cost=0
    for state in range(len(states)):
        total_cost+=decision_variables[state]
    prob+=total_cost
    for state in states:
            for action in actions:
                temp=0
                q=0
                prob += ( decision_variables[state] >= pulp.lpSum([s[i][4]*(s[i][3] + discount*decision_variables[s[i][2]]) for i in range(len(s))  if s[i][0]==state and s[i][1]==action]))
                    

    optimization_result = prob.solve()
    V=np.zeros(numStates)
    gamma=discount

    for v in prob.variables():
        index = int(v.name[1:])
        V[index]=v.varValue
    pi = [0 for i in range(numStates)]
    for state in range(numStates):
        val  = V[state]
        actVal=np.zeros(numActions)
        for a in actions:
            temp1=0
            for i in range(len(s)):
                if s[i][0]==state and a==s[i][1]:
                    temp1=temp1+s[i][4]*(s[i][3]+gamma*V[s[i][2]])
            actVal[a]=temp1
        
        action = (np.abs(np.asarray(actVal) - val)).argmin() 
        pi[state] = action
    return V,pi
        
    
    



#################################################################
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
    while (p<200):
        for ss in state:
                V1=deepcopy(V)
                temp=0
                for i in range(len(s)):
                    if s[i][0]==ss and Policy[ss]==s[i][1]:
                        temp=temp+s[i][4]*(s[i][3]+gamma*V1[s[i][2]])
                V[ss]=temp
        p=p+1
    return V,Policy1
 


############################################################



def iter(numStates, numActions, transition, discount,tolerance ,s):
    state=np.array([i for i in range(numStates)])
    action=np.array([i for i in range(numStates)])
    V=np.zeros(len(state))
    V1=np.ones(len(state))
    p=0
    A=np.zeros(len(action))
    gamma=discount
    while (p<10000):
        for ss in state:
            V1=deepcopy(V)
            for a in action:
                temp=0
                for i in range(len(s)):
                    if s[i][0]==ss and a==s[i][1]:
                        temp=temp+s[i][4]*(s[i][3]+gamma*V1[s[i][2]])
                if temp>V1[ss]:
                    V[ss]=temp
                    A[ss]=a
        p=p+1
    return V,A


s,mdp, algo = init()

numStates  = mdp['numStates']
numActions = mdp['numActions']
transition = mdp['transition']
discount   = mdp['discount']
tolerance  = 1e-12
s.pop(0)
if (algo == 'hpi'):
    V0, pi = policyiter(numStates, numActions, transition, discount, 1e-9,s)
elif (algo == 'lp'):
    V0, pi = linearProgram(numStates, numActions, transition, discount, tolerance,s)
elif (algo == 'vi'):
    V0, pi = iter(numStates, numActions, transition, discount, tolerance,s)

for i in range(len(V0)):
    print('{:.6f}'.format(round(V0[i], 6)) + "\t" + str(int(pi[i])))