import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


colors = ('b','c','g','m','r','y','k','w')
w = np.array([0,0])
x = np.array(([5,1],[2,4]))
x_norm = preprocessing.normalize(x, norm='l2') #ncreate unit norm for data point
plt.plot(x_norm[0][0],x_norm[0][1], marker = '_', color='k', label='datapoint 1')
plt.plot(x_norm[1][0],x_norm[1][1], marker = '+', color='k', label='datapoint 2')
plt.quiver(0, 0, x_norm[0][0],x_norm[0][1], angles='xy', scale_units='xy', scale=1, color='k')
plt.quiver(0, 0, x_norm[1][0],x_norm[1][1], angles='xy', scale_units='xy', scale=1, color='k')
y = np.array([-1.,1.])
j = 0
k = 0
while j < 3: # 3 als Beispiel
    for i in range(2):
        walt= w
        print("w", i, w)
        if (round(y[i] * np.dot(w,x[i])) >= 1): # >= 1 as example
        else:
            loss = 1 - y[i] * np.dot(w,x[i]) 
        print("loss", i, loss)
        print('PROJECTION PART:', x[i]*np.dot(w,x[i])*1./np.square(np.linalg.norm(x[i])))
        print('Yi*Xi*/||Xi||^2:',x[i]*y[i]*1./np.square(np.linalg.norm(x[i])))
        w = w + (y[i]*loss*x[i])* 1./np.square(np.linalg.norm(x[i]))
        plt.plot([walt[0],w[0]],[walt[1],w[1]], color = colors[k], linestyle ='dotted')
        print("w after", i, w)
        loss = round(y[i] * np.dot(w,x[i])) #(w[0]*x[0][0]+w[1]*x[0][1]) //test that result is 1 and therefor constraint is met
        print("loss after", i, loss)
        plt.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1, color=colors[k], label=('round', k+1, 'datapoint', i+1))
        z = np.array(range(-2, 2))
        u = eval('-z*w[0]/w[1]')
        plt.plot(z, u, color = colors[k])
        plt.legend()
        k = k+1
    j = j+1

plt.xlabel('X1')
plt.ylabel('X2')
plt.axis('equal')
plt.show()
