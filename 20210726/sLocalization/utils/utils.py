#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 20:07:40 2021

@author: ines
"""
from models.blankNet import BlankNet
from models.F2018Net import F2018Net
from models.simpleNet import simpleNet

def get_networks(args):
    """
    """
    if args.net == 'F2018Net':
        net = F2018Net()
    elif args.net == 'blankNet':
        net = BlankNet()
    elif args.net == 'simpleNet':
        net = simpleNet()
    return net
