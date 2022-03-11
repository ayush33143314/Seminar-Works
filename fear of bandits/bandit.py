import sys
import math
import numpy as np
import argparse
import warnings
import random
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore", category = RuntimeWarning)

algos = ['epsGreedy', 'ucb', 'KLucb', 'thompson']

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance")
    parser.add_argument("--algorithm")
    parser.add_argument("--randomSeed")
    parser.add_argument("--epsilon")
    parser.add_argument("--horizon")
    args = parser.parse_args()

    err  = 0
    Inst = args.instance
    try:
        Rs = int(args.randomSeed)
        Ep = float(args.epsilon)
        Hz = int(args.horizon)
    except:
        print("Conversion issues. Please check your inputs.")
        return (), 1

    An = len(algos)
    for i in range(len(algos)):
        if args.algorithm == algos[i]:
            An = i
            break
    return (Inst, An, Rs, Ep, Hz), 0


arg, err = init()
if err:
    print("Error confronted! Exiting!")
    exit()

def thompson(In, Ep, Rs, Hz):
    random.seed(Rs)
    values = {}
    for arm in In:
        values[arm] = [ [], 0, 0 ] 

    for t in range(Hz):
        betas = [np.random.beta(values[arm][1]+1, values[arm][2]+1) for arm in values]
        mxP   = In[betas.index(max(betas))]
        if np.random.random_sample() < mxP:
            values[mxP][0].append(mxP)
            values[mxP][1] += 1
        else:
            values[mxP][0].append(mxP)
            values[mxP][2] += 1

    optArm = max(float(arm) for arm in values)
    regret = Hz * optArm
    for arm in values:
        regret = regret-np.sum(values[arm][0])
    return round(regret, 2)

    
    
def epsGreedy(In,Ep,Rs,Hz):
    T=Hz
    E=Ep
    t=int(0)
    instances=np.array(In)
    random.seed(Rs)
    a=np.zeros(len(instances))
    regret=float(0)
    reg=np.array([])
    while t<=E*T:
        action=random.randrange(0, len(instances), 1)
        reward=np.random.choice(np.arange(0, 2), p=[1-instances[action],instances[action]])
        a[action]=(a[action]*t+reward)/(t+1)
        t=t+1
        regret=regret-instances[action]+np.max(instances)
        
    while t<=T:
        regret=regret-instances[a.argmax()]+instances[instances.argmax()]
        t=t+1
    return round(regret,2)


def ucb( In, Ep, Rs, Hz):
    T=Hz
    instances=np.array(In)
    arms=len(instances)
    random.seed(Rs)
    t=int(0)
    p=np.zeros(arms)
    ut=np.zeros(arms)
    regret=float(0)
    ucb=np.zeros(arms)
    while t<arms:
        action=t
        reward=np.random.choice(np.arange(0, 2), p=[1-instances[action],instances[action]])
        p[action]=(p[action]*t+reward)/(t+1)
        ut[action]=ut[action]+1
        t=t+1
        regret=regret-instances[action]+np.max(instances)
    while t<=T:
        for i in range(arms):
            ucb[i]=p[i]+math.sqrt(2*math.log(t)/ut[i])
        action=ucb.argmax()
        reward=np.random.choice(np.arange(0, 2), p=[1-instances[action],instances[action]])
        p[action]=(p[action]*t+reward)/(t+1)
        ut[action]=ut[action]+1
        regret=regret-instances[action]+np.max(instances)
        t=t+1
    return round(regret,2)



def KLucb( In, Ep, Rs, Hz):
    def log(num):
        try:
            return math.log(num)
        except:
            return 0

    def ucbCalc( p, u, t):
        
        c     = 3
        tol   = 1.0e-4

        start = p
        end   = 1.0
        mid   = (start + end) / 2.0
        final = (log(t) + c*log(log(t))) / u

        while abs(start - end) > tol:
            if p*log(p/mid) + (1-p)*log((1-p)/(1-mid)) > final:
                end   = mid
            else:
                start = mid
            mid = (start + end) / 2.0
        return mid

    random.seed(Rs)
    picks  = {}
    for arm in In:
        picks[arm] = [1, [int(random.random_sample() < arm)], 0]

    for t in range(len(picks), Hz):
        # print(str(t+1) + " in " + str(Hz), end = "\r")
        for arm in picks:
            picks[arm][2] = np.mean(picks[arm][1])
        ucb = [ucbCalc( picks[arm][2], picks[arm][0], t) for arm in picks]
        mxP = In[ucb.index(max(ucb))]
        picks[mxP][0] += 1
        if random.random_sample() < mxP:
            picks[mxP][1].append(1)
        else:
            picks[mxP][1].append(0)
    optArm = max(float(arm) for arm in picks)
    Hz = sum([picks[arm][0] for arm in picks])
    regret = Hz * optArm
    for arm in picks:
        regret -= np.sum(picks[arm][1])
    return round(regret, 4)



(Inst, An, Rs, Ep, Hz) = arg

with open(Inst) as f:
    content = f.readlines()
In = [float(content[i]) for i in range(len(content))]

if An == 0:
    regret = epsGreedy(In, Ep, Rs, Hz)
elif An == 1:
    regret = ucb( In, Ep, Rs, Hz)
elif An == 2:
    regret = KLucb(In, Ep, Rs, Hz)
elif An == 3:
    regret = thompson(In, Ep, Rs, Hz)
print(str(Inst) + ", " + str(algos[An]) + ", " + str(Rs) + ", " + str(Ep) + ", " + str(Hz) + ", " + str(regret))