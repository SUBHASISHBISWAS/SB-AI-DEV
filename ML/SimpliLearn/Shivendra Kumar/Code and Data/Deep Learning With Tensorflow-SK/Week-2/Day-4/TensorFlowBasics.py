#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 23:01:46 2019

@author: subhasish
"""

import tensorflow as tf

x=tf.Variable([3.0], name='x')

y=tf.Variable([4.0],name='y')

#computational Graph

f= x*x*y +y +2

with tf.Session() as sess:
    #Intilize variable
    x.initializer.run()
    y.initializer.run()
    
    print("Value of X",x.eval())
    print("Value of Y",y.eval())
    
    res=f.eval()
    
    print("Rsult of Computauon Graph",res)