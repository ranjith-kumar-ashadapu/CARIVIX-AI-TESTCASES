import numpy as np
import numpy.random._pickle as nrp
import numpy.random._mt19937 as mt
print('np version', np.__version__)
print('has MT19937', hasattr(mt, 'MT19937'))
print('MT19937 repr:', mt.MT19937)
print('numpy.random._pickle dir:')
print('\n'.join(dir(nrp)))
print('\nDone')
