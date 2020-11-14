from django.db import models

# Create your models here.
# ** indicates ^ (raising powers)
# BasicCOCOMO


def cocomoBasic(kloc):
    if 2 < kloc <= 50:
        data = basicOrganic(kloc)
    elif 50 < kloc <= 300:
        data = basicSemiDetatched(kloc)
    elif kloc > 300:
        data = basicEmbedded(kloc)
    else:
        data = basicOrganic(kloc)
    return data


def basicOrganic(kloc):
    a = 2.4
    b = 1.05
    c = 2.5
    d = 0.38
    effort = a*((kloc) ** b)
    devlopmentTime = c*((effort) ** d)
    staffSize = effort/devlopmentTime
    productivity = kloc/effort
    data = [effort, devlopmentTime, staffSize, productivity]
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
    data = [effort, devlopmentTime, staffSize, productivity]
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
    data = [effort, devlopmentTime, staffSize, productivity]
    return data

# Intermediate COComo


def Intermediatecocomo(kloc, EAF):
    if 2 < kloc <= 50:
        data = IntermediateOrganic(kloc, EAF)
    elif 50 < kloc <= 300:
        data = IntermediateSemiDetatched(kloc, EAF)
    elif kloc > 300:
        data = IntermediateEmbedded(kloc, EAF)
    else:
        data = IntermediateOrganic(kloc, EAF)
    return data


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
    data = [effort, devlopmentTime, staffSize, productivity]
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
    data = [effort, devlopmentTime, staffSize, productivity]
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
    data = [effort, devlopmentTime, staffSize, productivity]
    return data


# Detailed COCOMO

def detailedcocomo(kloc, EAF, U, T):
    if 2 < kloc <= 50:
        data = detailedOrganic(kloc, EAF, U, T)
    elif 50 < kloc <= 300:
        data = detailedSemiDetatched(kloc, EAF, U, T)
    elif kloc > 300:
        data = detailedEmbedded(kloc, EAF, U, T)
    else:
        data = detailedOrganic(kloc, EAF, U, T)
    return data


def detailedOrganic(kloc, EAF, U, T):
    a = 3.2
    b = 1.05
    c = 2.5
    d = 0.38
    eaf = EAF
    u = U
    t = T
    effort = (a*((kloc) ** b))*eaf
    totalEffort = u*effort
    devlopmentTime = c*((effort) ** d)
    totalDevolopmentTime = t*devlopmentTime
    staffSize = effort/devlopmentTime
    productivity = kloc/effort
    data = [effort, totalEffort, devlopmentTime,
            totalDevolopmentTime, staffSize, productivity]
    return data


def detailedSemiDetatched(kloc, EAF, U, T):
    a = 3.0
    b = 1.12
    c = 2.5
    d = 0.35
    eaf = EAF
    u = U
    t = T
    effort = (a*((kloc) ** b))*eaf
    totalEffort = u*effort
    devlopmentTime = c*((effort) ** d)
    totalDevolopmentTime = t*devlopmentTime
    staffSize = effort/devlopmentTime
    productivity = kloc/effort
    data = [effort, totalEffort, devlopmentTime,
            totalDevolopmentTime, staffSize, productivity]
    return data


def detailedEmbedded(kloc, EAF, U, T):
    a = 2.8
    b = 1.20
    c = 2.5
    d = 0.32
    eaf = EAF
    u = U
    t = T
    effort = (a*((kloc) ** b))*eaf
    totalEffort = u*effort
    devlopmentTime = c*((effort) ** d)
    totalDevolopmentTime = t*devlopmentTime
    staffSize = effort/devlopmentTime
    productivity = kloc/effort
    data = [effort, totalEffort, devlopmentTime,
            totalDevolopmentTime, staffSize, productivity]
    return data
