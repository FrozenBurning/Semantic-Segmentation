'''
@Description: 
@Author: Zhaoxi Chen
@Github: https://github.com/FrozenBurning
@Date: 2020-03-14 14:33:27
@LastEditors: Zhaoxi Chen
@LastEditTime: 2020-03-14 14:37:03
'''
import pickle
import sys
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.filen = filename

    def write(self, message):
	    self.terminal.write(message)
	    with open(self.filen,'a') as f:
             f.write(message)

    def flush(self):
	    pass
    
def historian(hist,file_name='TrainHistory.txt'):
    with open(file_name, 'wb') as file_pi:
        pickle.dump(hist, file_pi)
    return
        
