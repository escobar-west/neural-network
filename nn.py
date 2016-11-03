import numpy as np
import matplotlib.pyplot as plt

numInputs   = 10
hidden     = 1000
numOutputs  = 8

trainLength = 500

eta1 = .0005
eta2 = .0005
eta3 = .0001
lam =  .00001

weights1 = 0.1*np.random.rand(numInputs, hidden) - 0.05
bias1 = 0.1*np.random.rand(hidden) - 0.05
weights2 = 0.1*np.random.rand(hidden, numOutputs) - 0.05
bias2 = 0.1*np.random.rand(numOutputs) - 0.05

error = np.zeros(trainLength)
count = 0

validation = np.floor(trainLength/10)

for trial in range(trainLength):

   target = np.zeros(8) # hot vector, stores the roll number

   roll = np.random.randint(8)
   target[roll] = 1

   if roll == 0: # MM~DD~YYYY
      month = np.random.randint(1,99)
      day = np.random.randint(1,99)
      year = np.random.randint(1,9999)
      inputString = '{:02}~{:02}~{:04}'.format(month, day, year)

   elif roll == 1: # M~DD~YYYY
      month = np.random.randint(1,9)
      day = np.random.randint(1,99)
      year = np.random.randint(1,9999)
      inputString = ' {}~{:02}~{:04}'.format(month, day, year)

   elif roll == 2: #  MM~D~YYYY
      month = np.random.randint(1,99)
      day = np.random.randint(1,9)
      year = np.random.randint(1,9999)
      inputString = ' {:02}~{}~{:04}'.format(month, day, year)

   elif roll == 3: # M~D~YYYY
      month = np.random.randint(1,9)
      day = np.random.randint(1,9)
      year = np.random.randint(1,9999)
      inputString = '  {}~{}~{:04}'.format(month, day, year)

   if roll == 4: # MM~DD~YY
      month = np.random.randint(1,99)
      day = np.random.randint(1,99)
      year = np.random.randint(2000,2099)
      inputString = '  {:02}~{:02}~{:02}'.format(month, day, year%100)

   elif roll == 5: # M~DD~YY
      month = np.random.randint(1,9)
      day = np.random.randint(1,99)
      year = np.random.randint(2000,2099)
      inputString = '   {}~{:02}~{:02}'.format(month, day, year%100)

   elif roll == 6: # MM~D~YY
      month = np.random.randint(1,99)
      day = np.random.randint(1,9)
      year = np.random.randint(2000,2099)
      inputString = '   {:02}~{}~{:02}'.format(month, day, year%100)

   elif roll == 7: # M~D~YY
      month = np.random.randint(1,9)
      day = np.random.randint(1,9)
      year = np.random.randint(2000,2099)
      inputString = '    {}~{}~{:02}'.format(month, day, year%100)

   inputs = np.zeros(numInputs)

   for i in range(numInputs):
      inputs[i] = ord(inputString[i]) - ord('0')

   y = inputs.dot(weights1) + bias1

   z = np.zeros(y.size)
   
   for i in range(z.size): 
      if y[i] >= 0:
            z[i] = y[i]
      else:
         z[i] = np.expm1(y[i])

   w = z.dot(weights2) + bias2

   output = (np.expm1(w)+1)/np.sum(np.expm1(w)+1)

   error[trial] = np.sum( target*np.log1p(output-1) + (1-target)*(1-np.log1p(output-1)) ) + lam*np.sum(weights1*weights1) + lam*np.sum(weights2*weights2)

   delta = (output*(1-target)/(1-output) - target) - np.sum(output*(1-target)/(1-output) - target)*output

   dCdW2 = z.reshape(hidden, 1).dot(delta.reshape(1, numOutputs)) + lam*weights2

   temp = delta.reshape(1, numOutputs).dot(weights2.T)

   for i in range(temp.size):
      if z[i] < 0:
         temp[0][i] = temp[0][i]*(z[i]+1)

   dCdW1 = inputs.reshape(numInputs, 1).dot(temp) + lam*weights1

   weights2 = weights2 - eta2*dCdW2

   weights1 = weights1 - eta1*dCdW1

   bias2 = bias2 - eta3*delta.reshape(numOutputs)

   bias1 = bias1 - eta3*temp.reshape(hidden)

   if np.argmax(output)==np.argmax(target) and trial >= trainLength-validation:
      count = count + 1

   elif trial > trainLength-validation:
      print('incorrect at:', trial, 'target is', target)

      print('output is', output)

      print('guess is', np.argmax(output), 'and result is', np.argmax(target))

print('success rate:', 100*count/validation, '%')

plt.plot(error)

plt.show()

print(dCdW1)

print(dCdW2)

while 1:
   x = input('input your date:\n')

   if x == 'quit':
      print('quitting program')
      exit()

   x = ' '*(10-len(x)) + x

   print(x)

   for i in range(numInputs):
      inputs[i] = ord(x[i]) - ord('0')

   y = inputs.dot(weights1) + bias1

   z = np.zeros(y.size)
   
   for i in range(z.size): 
      if y[i] >= 0:
            z[i] = y[i]
      else:
         z[i] = np.expm1(y[i])

   w = z.dot(weights2) + bias2

   output = (np.expm1(w)+1)/np.sum(np.expm1(w)+1)

   guess = np.argmax(output)

   print('My guess is: ', guess)