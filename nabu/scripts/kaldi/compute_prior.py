'''@file counpute_prior.py
this script can be used to compute pdf priors for kaldi'''

import os
import sys
import itertools
import numpy as np

traindir = sys.argv[1]

#read the pdffile
with open(os.path.join(traindir, 'pdfs')) as fid:
    pdfs = fid.readlines()
pdfs = [pdf.split()[1:] for pdf in pdfs]
pdfs = list(itertools.chain.from_iterable(pdfs))
pdfs = map(int, pdfs)

#count each pdf occurrence
counts, _ = np.histogram(pdfs, range(max(pdfs)+2))

#normalize the counts to get the priors
prior = counts.astype(np.float32)/counts.sum()

np.save(os.path.join(traindir, 'prior.npy'), prior)
