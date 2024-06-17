# -*- coding: utf-8 -*-
from datasources.classification.cifar import CIFAR10Data as CIFAR10
from datasources.classification.cifar import CIFAR100Data as CIFAR100
from datasources.pose.mouse import MouseData as Mouse
from datasources.pose.flic import FLICData as FLIC
from datasources.pose.lsp import LSPData as LSP

__all__ = ('CIFAR10', 'CIFAR100', 'Mouse', 'FLIC', 'LSP')
