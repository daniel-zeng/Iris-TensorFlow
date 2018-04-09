import tensorflow as tf
import csv
import random

numFeat = 4
numClass = 3

########################
###    Graph Model   ###
########################

w = tf.Variable(tf.random_normal([2, 1]), tf.float32) #or w = tf.Variable([.1], tf.float32)
b = tf.Variable(tf.random_normal([4, 1]), tf.float32)
x = tf.placeholder(tf.float32, [None , numFeat])
y = tf.placeholder(tf.int64, [None, numClass])

init = tf.contrib.layers.xavier_initializer()
h1 = tf.layers.dense(inputs=x, units = 20, kernel_initializer=init)
y_pred = tf.layers.dense(inputs=h1, units = numClass, kernel_initializer=init)


loss = tf.nn.softmax_cross_entropy_with_logits(logits = y_pred, labels = y)
loss_val = tf.reduce_mean(loss)

########################
###   Data Loading   ###
########################

xtrain = []
ytrain = []
xtest = []
ytest = []
def oneHotIris(val):
    return {'Iris-setosa': [1, 0, 0], 'Iris-versicolor': [0, 1, 0], 'Iris-virginica': [0, 0, 1]}[val]

with open('iris.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if row:
            addTest = 0 == random.randint(0, 9)
            xadd = xtest if addTest else xtrain
            yadd = ytest if addTest else ytrain
            xadd.append([float(val) for val in row[0:4]])
            yadd.append(oneHotIris(row[4]))

data = {x: xtrain, y: ytrain}
testing = {x: xtest, y: ytest}

########################
###     Training     ###
########################

optimizer = tf.train.AdamOptimizer(0.001)
train = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(300):
    sess.run(train, feed_dict=data)
    if epoch % 10 == 0:
        loss_v_train = sess.run(loss_val, feed_dict=data)
        loss_v_test = sess.run(loss_val, feed_dict=testing)
        print(epoch, "Test:", loss_v_test, "Train:", loss_v_train)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Test accuracy:", sess.run(accuracy, feed_dict=testing))

sess.close()
