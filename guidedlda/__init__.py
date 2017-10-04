from __future__ import absolute_import, unicode_literals  # noqa

import logging

from guidedlda.guidedlda import GuidedLDA  # noqa
import guidedlda.datasets  # noqa

logging.getLogger('guidedlda').addHandler(logging.NullHandler())
