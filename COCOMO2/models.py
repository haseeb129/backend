from django.db import models


def cocomo2(OB, VB, Prod):
    ob = OB
    vb = VB
    prod = Prod
    nop = ((ob)*(100-vb))/100
    eff = nop/prod
    data = [nop, eff]
    return data
