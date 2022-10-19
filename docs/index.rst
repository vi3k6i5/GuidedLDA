.. GuidedLDA documentation master file, created by
   sphinx-quickstart on Thu Oct  5 00:28:03 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GuidedLDA's documentation!
=====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. image:: https://readthedocs.org/projects/guidedlda/badge/?version=latest
    :target: http://guidedlda.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://badge.fury.io/py/guidedlda.svg
    :target: https://badge.fury.io/py/guidedlda
    :alt: Package version


``GuidedLDA`` OR ``SeededLDA`` implements latent Dirichlet allocation (LDA) using collapsed Gibbs sampling. ``GuidedLDA`` can be guided by setting some seed words per topic. Which will make the topics converge in that direction.

You can read more about guidedlda in `the documentation <https://guidedlda.readthedocs.io>`_.

Installation
------------

::

    pip install guidedlda

If pip install does not work, then try the next step:

::

    https://github.com/vi3k6i5/GuidedLDA
    cd GuidedLDA
    sh build_dist.sh
    python setup.py sdist
    pip install -e .

If the above step also does not work, please raise an `issue <https://github.com/vi3k6i5/guidedlda/issues>`_ with details of your workstation's OS version, Python version, architecture etc. and I will try my best to fix it ASAP.

Getting started
---------------

``guidedlda.GuidedLDA`` implements latent Dirichlet allocation (LDA). The interface follows
conventions found in scikit-learn_.

`Example Code <https://github.com/vi3k6i5/GuidedLDA/blob/master/examples/example_seeded_lda.py>`_.


The following demonstrates how to inspect a model of a subset of the NYT
news dataset. The input below, ``X``, is a document-term matrix (sparse matrices
are accepted).

.. code-block:: python

    >>> import numpy as np
    >>> import guidedlda
    
    >>> X = guidedlda.datasets.load_data(guidedlda.datasets.NYT)
    >>> vocab = guidedlda.datasets.load_vocab(guidedlda.datasets.NYT)
    >>> word2id = dict((v, idx) for idx, v in enumerate(vocab))
    
    >>> X.shape
    (8447, 3012)
    
    >>> X.sum()
    1221626
    >>> # Normal LDA without seeding
    >>> model = guidedlda.GuidedLDA(n_topics=5, n_iter=100, random_state=7, refresh=20)
    >>> model.fit(X)
    INFO:guidedlda:n_documents: 8447
    INFO:guidedlda:vocab_size: 3012
    INFO:guidedlda:n_words: 1221626
    INFO:guidedlda:n_topics: 5
    INFO:guidedlda:n_iter: 100
    WARNING:guidedlda:all zero column in document-term matrix found
    INFO:guidedlda:<0> log likelihood: -11489265
    INFO:guidedlda:<20> log likelihood: -9844667
    INFO:guidedlda:<40> log likelihood: -9694223
    INFO:guidedlda:<60> log likelihood: -9642506
    INFO:guidedlda:<80> log likelihood: -9617962
    INFO:guidedlda:<99> log likelihood: -9604031
    
    >>> topic_word = model.topic_word_
    >>> n_top_words = 8
    >>> for i, topic_dist in enumerate(topic_word):
    >>>     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    >>>     print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    Topic 0: company percent market business plan pay price increase
    Topic 1: game play team win player season second start
    Topic 2: life child write man school woman father family
    Topic 3: place open small house music turn large play
    Topic 4: official state government political states issue leader case
    
    >>> # Guided LDA with seed topics.
    >>> seed_topic_list = [['game', 'team', 'win', 'player', 'season', 'second', 'victory'],
    >>>                    ['percent', 'company', 'market', 'price', 'sell', 'business', 'stock', 'share'],
    >>>                    ['music', 'write', 'art', 'book', 'world', 'film'],
    >>>                    ['political', 'government', 'leader', 'official', 'state', 'country', 'american','case', 'law', 'police', 'charge', 'officer', 'kill', 'arrest', 'lawyer']]
    
    >>> model = guidedlda.GuidedLDA(n_topics=5, n_iter=100, random_state=7, refresh=20)
    
    >>> seed_topics = {}
    >>> for t_id, st in enumerate(seed_topic_list):
    >>>     for word in st:
    >>>         seed_topics[word2id[word]] = t_id
    
    >>> model.fit(X, seed_topics=seed_topics, seed_confidence=0.15)
    INFO:guidedlda:n_documents: 8447
    INFO:guidedlda:vocab_size: 3012
    INFO:guidedlda:n_words: 1221626
    INFO:guidedlda:n_topics: 5
    INFO:guidedlda:n_iter: 100
    WARNING:guidedlda:all zero column in document-term matrix found
    INFO:guidedlda:<0> log likelihood: -11486362
    INFO:guidedlda:<20> log likelihood: -9767277
    INFO:guidedlda:<40> log likelihood: -9663718
    INFO:guidedlda:<60> log likelihood: -9624150
    INFO:guidedlda:<80> log likelihood: -9601684
    INFO:guidedlda:<99> log likelihood: -9587803
    
    
    >>> n_top_words = 10
    >>> topic_word = model.topic_word_
    >>> for i, topic_dist in enumerate(topic_word):
    >>>     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    >>>     print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    Topic 0: game play team win season player second point start victory
    Topic 1: company percent market price business sell executive pay plan sale
    Topic 2: play life man music place write turn woman old book
    Topic 3: official government state political leader states issue case member country
    Topic 4: school child city program problem student state study family group

The document-topic distributions should be retrived as: ``doc_topic = model.transform(X)``.

.. code-block:: python

    >>> doc_topic = model.transform(X)
    >>> for i in range(9):
    >>>     print("top topic: {} Document: {}".format(doc_topic[i].argmax(), 
                                                      ', '.join(np.array(vocab)[list(reversed(X[i,:].argsort()))[0:5]])))
    top topic: 4 Document: plant, increase, food, increasingly, animal
    top topic: 3 Document: explain, life, country, citizen, nation
    top topic: 2 Document: thing, solve, problem, machine, carry
    top topic: 2 Document: company, authority, opera, artistic, director
    top topic: 3 Document: partner, lawyer, attorney, client, indict
    top topic: 2 Document: roll, place, soon, treat, rating
    top topic: 3 Document: city, drug, program, commission, report
    top topic: 1 Document: company, comic, series, case, executive
    top topic: 3 Document: son, scene, charge, episode, attack

Save the model for production or for running later:

.. code-block:: python

    >>> from six.moves import cPickle as pickle
    >>> # Uncomment next step if you want to lighten the model object
    >>> # This step will delete some matrices inside the model.
    >>> # you will be able to use model.transform(X) the same way as earlier.
    >>> # you wont be able to use model.fit_transform(X_new)
    >>> # model.purge_extra_matrices()
    >>> with open('guidedlda_model.pickle', 'wb') as file_handle:
    >>>     pickle.dump(model, file_handle)
    >>> # load the model for prediction
    >>> with open('guidedlda_model.pickle', 'rb') as file_handle:
    >>>     model = pickle.load(file_handle)
    >>> doc_topic = model.transform(X)


Requirements
------------

Python 2.7 or Python 3.3+ is required. The following packages are required

- numpy_
- pbr_

Caveat
------

``guidedlda`` aims for Guiding LDA. More often then not the topics we get from a LDA model are not to our satisfaction. GuidedLDA can give the topics a nudge in the direction we want it to converge. We have production trained it for half a million documents (We have a big machine). We have run predictions on millions and manually checked topics for thousands (we are satisfied with the results).

If you are working with a very large corpus you may wish to use more sophisticated topic models such as those implemented in hca_ and MALLET_.  hca_ is written entirely in C and MALLET_ is written in Java. Unlike ``guidedlda``, hca_ can use more than one processor at a time. Both MALLET_ and hca_ implement topic models known to be more robust than standard latent Dirichlet allocation.

Notes
-----

Latent Dirichlet allocation is described in `Blei et al. (2003)`_ and `Pritchard
et al. (2000)`_. Inference using collapsed Gibbs sampling is described in
`Griffiths and Steyvers (2004)`_. And Guided LDA is described in `Jagadeesh Jagarlamudi, Hal Daume III and Raghavendra Udupa (2012)`_


Important links
---------------

- Documentation: http://guidedlda.readthedocs.org
- Source code: https://github.com/vi3k6i5/guidedlda/
- Issue tracker: https://github.com/vi3k6i5/guidedlda/issues

Other implementations
---------------------
- scikit-learn_'s `LatentDirichletAllocation <http://scikit-learn.org/dev/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html>`_ (uses online variational inference)
- `gensim <https://pypi.python.org/pypi/gensim>`_ (uses online variational inference)

Credits
-------
I would like to thank the creators of `LDA project <https://github.com/lda-project/lda>`_. I used the code from that LDA project as base to implement GuidedLDA on top of it.

Thanks to : `Allen Riddell <https://twitter.com/ariddell>`_ and `Tim Hopper <https://twitter.com/tdhopper>`_. :)

License
-------

``guidedlda`` is licensed under Version 2.0 of the Mozilla Public License.

.. _Python: http://www.python.org/
.. _scikit-learn: http://scikit-learn.org
.. _hca: http://www.mloss.org/software/view/527/
.. _MALLET: http://mallet.cs.umass.edu/
.. _numpy: http://www.numpy.org/
.. _pbr: https://pypi.python.org/pypi/pbr
.. _Cython: http://cython.org
.. _Blei et al. (2003): http://jmlr.org/papers/v3/blei03a.html
.. _Pritchard et al. (2000): http://www.genetics.org/content/155/2/945.full
.. _Griffiths and Steyvers (2004): http://www.pnas.org/content/101/suppl_1/5228.abstract
.. _Jagadeesh Jagarlamudi, Hal Daume III and Raghavendra Udupa (2012): http://www.umiacs.umd.edu/~jags/pdfs/GuidedLDA.pdf


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
