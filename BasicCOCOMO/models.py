from django.db import models
import re

# Create your models here.
# ** indicates ^ (raising powers)
# BasicCOCOMO


def cocomoBasic(kloc):

    dataOrganic = basicOrganic(kloc)
    dataSemiDetatched = basicSemiDetatched(kloc)
    dataEmbedded = basicEmbedded(kloc)
    return[dataOrganic, dataSemiDetatched, dataEmbedded]
    # return {"dataOrganic":dataOrganic,"dataSemiDetatched":dataSemiDetatched,"dataEmbedded":dataEmbedded}


def basicOrganic(kloc):
    a = 2.4
    b = 1.05
    c = 2.5
    d = 0.38
    effort = a*((kloc) ** b)
    devlopmentTime = c*((effort) ** d)
    staffSize = effort/devlopmentTime
    productivity = kloc/effort
    data = [a, b, c, d, effort, devlopmentTime, staffSize, productivity]
    return data


def basicSemiDetatched(kloc):
    a = 3.0
    b = 1.12
    c = 2.5
    d = 0.35
    effort = a*((kloc) ** b)
    devlopmentTime = c*((effort) ** d)
    staffSize = effort/devlopmentTime
    productivity = kloc/effort
    data = [a, b, c, d, effort, devlopmentTime, staffSize, productivity]
    return data


def basicEmbedded(kloc):
    a = 3.6
    b = 1.20
    c = 2.5
    d = 0.32
    effort = a*((kloc) ** b)
    devlopmentTime = c*((effort) ** d)
    staffSize = effort/devlopmentTime
    productivity = kloc/effort
    data = [a, b, c, d, effort, devlopmentTime, staffSize, productivity]
    return data

# Intermediate COComo


def Intermediatecocomo(kloc, EAF):
    dataOrganic = IntermediateOrganic(kloc, EAF)
    dataSemiDetatched = IntermediateSemiDetatched(kloc, EAF)
    dataEmbedded = IntermediateEmbedded(kloc, EAF)
    return[dataOrganic, dataSemiDetatched, dataEmbedded]


def IntermediateOrganic(kloc, EAF):
    a = 3.2
    b = 1.05
    c = 2.5
    d = 0.38
    eaf = EAF
    effort = (a*((kloc) ** b))*eaf
    devlopmentTime = c*((effort) ** d)
    staffSize = effort/devlopmentTime
    productivity = kloc/effort
    data = [a, b, c, d, effort, devlopmentTime, staffSize, productivity]
    return data


def IntermediateSemiDetatched(kloc, EAF):
    a = 3.0
    b = 1.12
    c = 2.5
    d = 0.35
    eaf = EAF
    effort = (a*((kloc) ** b))*eaf
    devlopmentTime = c*((effort) ** d)
    staffSize = effort/devlopmentTime
    productivity = kloc/effort
    data = [a, b, c, d, effort, devlopmentTime, staffSize, productivity]
    return data


def IntermediateEmbedded(kloc, EAF):
    a = 2.8
    b = 1.20
    c = 2.5
    d = 0.32
    eaf = EAF
    effort = (a*((kloc) ** b))*eaf
    devlopmentTime = c*((effort) ** d)
    staffSize = effort/devlopmentTime
    productivity = kloc/effort
    data = [a, b, c, d, effort, devlopmentTime, staffSize, productivity]
    return data


# Detailed COCOMO

def detailedcocomo(kloc, EAF, Name):
    if re.search("organic", Name):
        data = detailedOrganic(kloc, EAF)
    elif re.search("semidetached", Name):
        data = detailedSemiDetatched(kloc, EAF)
    elif re.search("embedded", Name):
        data = detailedEmbedded(kloc, EAF)

    return data


def detailedOrganic(kloc, EAF):
    a = 3.2
    b = 1.05
    c = 2.5
    d = 0.38
    eaf = EAF

    effort = (a*((kloc) ** b))*eaf
    devlopmentTime = c*((effort) ** d)
    data = [effort, devlopmentTime, ]
    return data


def detailedSemiDetatched(kloc, EAF):
    a = 3.0
    b = 1.12
    c = 2.5
    d = 0.35
    eaf = EAF
    effort = (a*((kloc) ** b))*eaf
    devlopmentTime = c*((effort) ** d)
    data = [effort, devlopmentTime, ]
    return data


def detailedEmbedded(kloc, EAF):
    a = 2.8
    b = 1.20
    c = 2.5
    d = 0.32
    eaf = EAF

    effort = (a*((kloc) ** b))*eaf
    devlopmentTime = c*((effort) ** d)
    data = [effort, devlopmentTime, ]
    return data
