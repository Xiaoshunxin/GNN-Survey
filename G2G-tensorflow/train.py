import pdb
import time
from model import Graph2Gauss
from utils import load_dataset, score_node_classification
from preprocessor import preprocessor

# g = load_dataset('data/cora_ml.npz')
g = preprocessor(domain="computers")
A, X, z = g['A'], g['X'], g['z']

for i in range(5):
    print("##############" + str(i) + "#################")
    start_time = time.time()
    g2g = Graph2Gauss(A=A, X=X, L=64, verbose=True, p_val=0.0, p_test=0.0)
    sess = g2g.train()
    mu, sigma = sess.run([g2g.mu, g2g.sigma])


    accu, p_macro, r_macro, f1_macro, f1_micro = score_node_classification(mu, z, n_repeat=1, norm=False)
    print("accu = {:.2f}".format(accu * 100))
    print("macro_p = {:.2f}".format(p_macro * 100))
    print("macro_r = {:.2f}".format(r_macro * 100))
    print("macro_f1 = {:.2f}".format(f1_macro * 100))
    print("micro_f1 = {:.2f}".format(f1_micro * 100))
    print("Total time elapsed: {:.2f}s".format(time.time() - start_time))