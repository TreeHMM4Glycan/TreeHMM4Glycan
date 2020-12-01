import matplotlib.pyplot as plt
import pickle


a = pickle.load(open('Results/eval-EXP-20201201-013746-model.pkl','rb'))

print(a['Metrics'])