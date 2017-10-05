PYTHON ?= python
CYTHON ?= cython

cython:
	find guidedlda -name "*.pyx" -exec $(CYTHON) {} \;
