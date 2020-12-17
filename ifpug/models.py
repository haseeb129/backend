from django.db import models


def internalLogicFiles(det,ret):
    det=det
    ret=ret
    low=0
    average=0
    high=0
    if(det<=19 and ret<=5):
        low=1
    if(det<=50 and ret<=1):
        low=1
    if(det>50 and ret <=1):
        average=1
    if((det>19 and det <=50) and (ret>1 and ret<=5)):
        average=1
    if(det<=19 and ret>5):
        average=1
    if(det>50 and ret >1 ):
        high=1
    if(det>19 and ret >5 ):
        high=1
    
    data={'low':low,'average': average,'high': high}
    return data


def externalInterfaceFiles(det,ret):
    det=det
    ret=ret
    low=0
    average=0
    high=0
    if(det<=19 and ret<=5):
        low=1
    if(det<=50 and ret<=1):
        low=1
    if(det>50 and ret <=1):
        average=1
    if((det>19 and det <=50) and (ret>1 and ret<=5)):
        average=1
    if(det<=19 and ret>5):
        average=1
    if(det>50 and ret >1 ):
        high=1
    if(det>19 and ret >5 ):
        high=1
    
    data={'low':low,'average': average,'high': high}
    return data    


def externalInputs(det,ftr):
    det=det
    ftr=ftr
    low=0
    average=0
    high=0
    if(det<=15 and ftr<=1):
        low=1
    if(det<=4 and ftr<=2):
        low=1
    if(det>15 and ftr <=1):
        average=1
    if((det>4 and det <=15) and (ftr>1 and ftr<=2)):
        average=1
    if(det<=4 and ftr>2):
        average=1
    if(det>4 and ftr >2 ):
        high=1
    if(det>15 and ftr >1 ):
        high=1
    
    data={'low':low,'average': average,'high': high}
    return data    


def externalOutputs(det,ftr):
    det=det
    ftr=ftr
    low=0
    average=0
    high=0
    if(det<=19 and ftr<=1):
        low=1
    if(det<=5 and ftr<=3):
        low=1
    if(det>19 and ftr <=1):
        average=1
    if((det>5 and det <=19) and (ftr>1 and ftr<=3)):
        average=1
    if(det<=5 and ftr>3):
        average=1
    if(det>5 and ftr >3 ):
        high=1
    if(det>19 and ftr >1 ):
        high=1
    
    data={'low':low,'average': average,'high': high}
    return data


def externalQueries(inDET,inFTR,det,ftr):
    outdet=det
    outftr=ftr
    inDET=inDET
    inFTR=inFTR
    low=0
    average=0
    high=0
    incomplexity='Null'
    complexity='Null'
    if(inDET<=15 and inFTR<=1):
        incomplexity='low'
    if(inDET<=4 and inFTR<=2):
        incomplexity='low'
    if(inDET>15 and inFTR<=1):
        incomplexity='average'
    if((inDET>4 and inDET<=15) and (inFTR>1 and inFTR<=2)):
        incomplexity='average'
    if(inDET<=4 and inFTR>2):
        incomplexity='average'
    if(inDET>4 and inFTR>2):
        incomplexity='high'
    if(inDET>15 and inFTR>1):
        incomplexity='high'
    ######################################
    if(det<=19 and ftr<=1):
        complexity='low'
    if(det<=5 and ftr<=3):
        complexity='low'
    if(det>19 and ftr<=1):
        complexity='average'
    if((det>5 and det<=19) and (ftr>1 and ftr<=3)):
        complexity='average'
    if(det<=5 and ftr>3):
        complexity='average'
    if(det>5 and ftr>3):
        complexity='high'
    if(det>19 and ftr>1):
        complexity='high'
    if(incomplexity=='low' and complexity=='low'):
        low=1
    if(incomplexity=='high' and complexity=='high'):
        high=1
    if(incomplexity=='Null' and complexity=='Null'):
        average=1
    data={'low':low,'average': average,'high': high}
    return data


def unadjustedFP(lowILF,avgILF,highILF,lowEIF,avgEIF,highEIF,lowEI,avgEI,highEI,lowEO,avgEO,highEO,lowEQ,avgEQ,highEQ):
    lowILF  =lowILF
    avgILF  =avgILF
    highILF =highILF
    lowEIF  =lowEIF
    avgEIF  =avgEIF
    highEIF =highEIF
    lowEI   =lowEI
    avgEI   =avgEI
    highEI  =highEI
    lowEO   =lowEO
    avgEO   =avgEO
    highEO  =highEO
    lowEQ   =lowEQ
    avgEQ   =avgEQ
    highEQ  =highEQ
######################
    ilfLowFP    = lowILF * 7
    ilfAvgFP    = avgILF * 10
    ilfHighFP   = highILF* 15
    eifLowFP    = lowEIF * 5
    eifAvgFP    = avgEIF * 7
    eifHighFP   = highEIF* 10
    eiLowFP     = lowEI  * 3
    eiAvgFP     = avgEI  * 4
    eiHighFP    = highEI * 6
    eoLowFP     = lowEO  * 4
    eoAvgFP     = avgEO  * 5
    eoHighFP    = highEO * 7
    eqLowFP     = lowEQ  * 3
    eqAvgvFP    = avgEQ  * 4
    eqHighFP    = highEQ * 6
    totalUnadjustedFP=(ilfLowFP+ilfAvgFP+ilfHighFP)+(eifLowFP+eifAvgFP+eifHighFP)+(eiLowFP+eiAvgFP+eiHighFP)+(eoLowFP+eoAvgFP+eoHighFP)+(eqLowFP+eqAvgvFP+eqHighFP)
    percentILF_FP=(ilfLowFP+ilfAvgFP+ilfHighFP)/totalUnadjustedFP
    percentEIF_FP=(eifLowFP+eifAvgFP+eifHighFP)/totalUnadjustedFP
    percentEI_FP=(eiLowFP+eiAvgFP+eiHighFP)/totalUnadjustedFP
    percentEO_FP=(eoLowFP+eoAvgFP+eoHighFP)/totalUnadjustedFP
    percentEQ_FP=(eqLowFP+eqAvgvFP+eqHighFP)/totalUnadjustedFP
    percentTotalUnadjustedFP=percentILF_FP+ percentEIF_FP+ percentEI_FP+ percentEO_FP+ percentEQ_FP
    data=[totalUnadjustedFP,percentTotalUnadjustedFP]
    return data


def VAf(datacCommunications,dataProcessing,performance,heavilyUsedConfiguration,transactionRates,onlineDataEntry,designForEndUserEfficiency,onlineUpdate,complexProcessing,useableInOtherApp,installationEase,operationalEase,multipleSites,facilitateChange):
    datacCommunications     =    datacCommunications
    dataProcessing          =    dataProcessing
    performance             =    performance
    heavilyUsedConfiguration=    heavilyUsedConfiguration
    transactionRates        =    transactionRates
    onlineDataEntry         =    onlineDataEntry
    designForEndUserEfficiency=    designForEndUserEfficiency
    onlineUpdate            =    onlineUpdate
    complexProcessing       =    complexProcessing
    useableInOtherApp       =    useableInOtherApp
    installationEase        =    installationEase
    operationalEase         =    operationalEase
    multipleSites           =    multipleSites
    facilitateChange        =    facilitateChange
    totalDegreeOfInfluence=datacCommunications+dataProcessing+performance+heavilyUsedConfiguration+transactionRates+onlineDataEntry+designForEndUserEfficiency+onlineUpdate+complexProcessing+useableInOtherApp+installationEase+operationalEase+multipleSites+facilitateChange
    vaf=((totalDegreeOfInfluence*0.01)+0.65)
    data=[totalDegreeOfInfluence, vaf]
    return data