import numpy as np
import random 
import pandas as pd
import pulp
import matplotlib.pyplot as plt
import math
def nCr(n,r):
    g = math.factorial
    return int(g(n)/g(r)/g(n-r))

print("Enter n(even):") # Taking Population Size
num = int(input())
Rl = []
f = np.zeros((100,num))
for i in range(1,6):
    for j in range (1,6):
        a = [j,i]
        Rl.append(a)
R = np.array(Rl)
ri = np.random.randint(2,6,num)
ri = ri*1.0
#print("Initial Ri:",ri)   #Initial Rating of the agents
p_ij = np.ones((num,num),dtype=float) * 1/(num-1)
np.fill_diagonal(p_ij, 0, wrap=False) 
#print("Initial probabilty of meeting i to j:\n",p_ij,"\n")  #Initial probabilty of the agents
rg = np.zeros((100,num))
t = 1
s_ij = np.zeros((10000,num)) #It will hold Satisfaction values
dfarr = {} #Contains DataFrame after every of n iterations

while t < 100:
    z = 0
    y = 0
    w_choice = []
    u = np.zeros((num,25)) #Stores payoff for 25 different ratings possible
    idx = np.zeros(num)  #Contains the index of u(i) for which it is maximum
    chk = np.zeros(num)  # Used in Assigning couple
    pij_cpl = np.zeros(num)  #Stores who meets with whom(index)
    ri_u = np.zeros((num,25)) #25 possible rating calculated for each  agent
    pij_u = np.zeros((num,25)) # 25 possible rating calculated for each  agent
    
    for i in range(num):
        j = np.cumsum(p_ij[i]) #Calculates cumulative sum
        w_choice.append(j)
    w_choice = np.array(w_choice)
    b = np.insert(w_choice , 0 , values=0, axis=1) #Adding 0 column in the starting
    #print(b)
    for i in range(num):
        while(chk[i]==0 and y<1000):
            y = y+1
            q = np.random.random()
            if(chk[i]==0):
                for j in range(num):
                    if(q>0 and q<b[i][j+1] and chk[j]==0 and i!=j):
                        pij_cpl[i] = j
                        pij_cpl[j] = i
                        chk[i]=1
                        chk[j]=1
                        break
            else:
                continue
    #print(pij_cpl)           
    mx = np.zeros(num)
    for i in range(num):
        s_ij[t-1][i] = np.random.randint(1,6)
        z = z+1
    #print("Satisfaction:",s_ij[t-1])
    z = np.ones((num,num-1))
    for i in range(num):
        for j in range(25):
            ri_u[i][j] = ((t-1)*ri[i] + R[j][0]/5*R[j][1])/((t-1) + R[j][0]/5) #Possible 25 ratings calculated
            pij_u[i][j] = min(1,p_ij[i][int(pij_cpl[i])]*(R[j][0]+R[j][1])/6)  #Possible 25 Probabilities calculated
            u[i][j] = (s_ij[t-1][i]- s_ij[t-2][i])*(R[j][0]-rg[t-2][i] ) + (ri_u[i][j]-ri[i]) #Putting them in the formula
    #temp1 = np.ones()
    temp2 = np.ones(5)
    #pij_u[i][j]-p_ij[i][int(pij_cpl[i])] Can use this also
    #The below part helps in matrix generation
    for j in range(num):
        p=0
        temp1 = np.reshape(u[j],(5,5))
        for qq in range(5):
            temp2[qq] = max(temp1[qq])
        temp3 = min(temp2)
        lel = np.where(temp3==u[j]) #Index of maximum value in each row
        idx[j] = lel[0][0]
        for i in range(num):
            if(i!=j):
                z[j][p] = i
                p = p+1
    #print("Index chosen:",idx)
    # Dataset of ratings given bt i
    
    for i in range(num):
        f[t-1][i] = ((t-1)*ri[i] + R[int(idx[i].astype(int))][0]/5*R[idx[int(pij_cpl[i])].astype(int)][0])/(t -1+ R[int(idx[i].astype(int))][0]/5) 
        rg[t-1][i] = R[idx[i].astype(int)][0]
    data = {'Iteration':np.ones(num).astype(int)*t,'Agent':np.arange(1,num+1),'Meets with':pij_cpl.astype(int)+1,'Last meeting Satisfaction':s_ij[t-2],'Current meeting Satisfaction':s_ij[t-1],'Rating before':ri,'Rating given':[R[idx[i].astype(int)][0] for i in range(len(idx))],'Rating Received':[R[idx[int(pij_cpl[i])].astype(int)][0] for i in range(len(idx))],'Agg R after meeting':[f[t-1][i] for i in range(len(idx))]}
    df = pd.DataFrame(data)
    dfarr[t-1] = df
    #print(df[['Agent','Meets with','Rating given','Rating Received','Agg R after meeting']])
   
    a = np.zeros((num,nCr(num,2)))
    sm = 0
    v = 0
    b=[1]*num
    for i in range(num-1):
        h=i
        sm = int(sm+num-1-i)
        for j in range(v,sm):
            a[i][j] = 1
        for j in range(v,sm):
            a[h+1][j] = 1
            h = h+1
        v = sm

    for i in range(num):
        ri[i] = f[t-1][i]
        cnt = 0 
        k = np.where(z[i]==pij_cpl[i])
        for j in range(nCr(num,2)):
            if(a[i][j]==1):
                cnt = cnt + 1
            if(cnt==k[0]+1):
                a[i][j]=0
    #print(a) #This is a 6*nCr(num,2) matrix of zeros and ones. Suppose 3 meets 4, then (5+4+1)th column and 3rd row will be zero and same for each agent.       

    pij_solve = {}
    #Creating Problem
    prob = pulp.LpProblem("Probabilty",pulp.LpMaximize)
    
    #Creating LpVariable    
    pij_solve = pulp.LpVariable.dicts("Val",list(range(nCr(num,2))),lowBound = 0,upBound = 1)
    #Objective Function
    prob += pulp.lpSum([0*pij_solve[0] + 0*pij_solve[0]])
    #Entering Constraints
    for i in range(num):
        prob += pulp.lpSum([a[i][j]*pij_solve[j]] for j in range(nCr(num,2))) == 1 - pij_u[i][int((idx[i].astype(int)//5)*5+idx[pij_cpl[i].astype(int)].astype(int)//5)]
    
    #Solving
    prob.solve()
    #Printing if Optimal or not
    #print(pulp.LpStatus[prob.status])
    sol = np.zeros(nCr(num,2)) #It will contain nCr(num,2) unknowns found
    k = 0
    for v in pij_solve:
        v_val = pij_solve[v].varValue
        sol[k] = v_val
        k = k+1
    #print(sol,"\n")
    k = 0
    #Updating probabilities after solving
    for i in range(num):
        for j in range(i,num):
            if(i!=j):
                p_ij[i][j] = sol[k]
                p_ij[j][i] = p_ij[i][j]
                k = k+1
        p_ij[i][int(pij_cpl[i])] = pij_u[i][int((idx[i].astype(int)//5)*5+idx[pij_cpl[i].astype(int)].astype(int)//5)]
        p_ij[int(pij_cpl[i])][i] = p_ij[i][int(pij_cpl[i])]
    #print(p_ij) #Probility after nth iteration

    #jdfkdjn = input("Press Enter to continue:")
    t = t+1
print("Enter Agent number to see the plot:")
n = int(input())
t = []
t2 = []
for i in range(99):
    q = s_ij[i][n-1]
    t.append(q)
    q = rg[i][n-1]
    t2.append(q)
plt.figure(figsize = (30,20))
plt.plot(np.arange(0,99),t,label = "Satisfaction")
plt.plot(np.arange(0,99),t2,label = "Rating given")
plt.xlabel('Number of Iterations')
plt.ylabel('Satsifaction/Rating')
plt.legend()
plt.show()


            
             
