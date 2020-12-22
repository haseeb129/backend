from django.http import JsonResponse
from rest_framework.decorators import api_view
# from analytics.signals import object_viewed_signal
from rest_framework import response, decorators, permissions, status
from .models import internalLogicFiles as ilf, externalInterfaceFiles as eif, externalInputs as ei
from .models import externalOutputs as eo, externalQueries as eq, VAf as VAF, unadjustedFP as uFP
from rest_framework.response import Response
# Create your views here.



@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def ifpug(request):

    #1. Identify the project or application being counted.
    # customerName=request.data['Customer Name']
    # projectName=request.data['Project Name']
    # projectCode=request.data['Project Code']
    # analyst=request.data['Analyst']
    # date=request.data['Date']
    # print("request.data",request.data)
    customerName='Customer Name'
    projectName='Project Name'
    projectCode='Project Code'
    analyst='Analyst'
    date='Date'

    #2. List and analyze each of the components of the application.
    #2a. Internal Logical Files (ILFs)
    ilf_summary=[]
    ilf_loop=request.data['ILF']
    for i in ilf_loop:
        if not i:
            break
        lofILF=i['fileName']
        ilfNotes=i['notesAssumptions']
        ILF_DET=int(i['det'])
        ILF_RET=int(i['ret'])
        ilf_summary.append(ilf(ILF_DET,ILF_RET))
    low_count=0
    avg_count=0
    high_count=0 
    for i in ilf_summary:
        if(i['low']==1):
            low_count+=1
        if(i['average']==1):
            avg_count+=1
        if(i['high']==1):
            high_count+=1
    ilf_summary_count=[low_count,avg_count,high_count]
  
    
    #2b. External Interface Files (EIFs)
    eif_summary=[]
    eif_loop=request.data['EIF']
    for e in eif_loop:
        if not e:
            break
        lofEIF=e['fileName']
        eifNotes=e['notesAssumptions']
        EIF_DET=int(e['det'])
        EIF_RET=int(e['ret'])
        eif_summary.append(eif(EIF_DET,EIF_RET))
    low_count=0
    avg_count=0
    high_count=0 
    for i in eif_summary:
        if(i['low']==1):
            low_count+=1
        if(i['average']==1):
            avg_count+=1
        if(i['high']==1):
            high_count+=1
    eif_summary_count=[low_count,avg_count,high_count]
 
    #2c. External Inputs (EIs)
    ei_summary=[]
    ei_loop=request.data['EI']
    for e in ei_loop:
        if not e:
            break
        lofEI=e['fileName']
        eiNotes=e['notesAssumptions']
        EI_DET=int(e['det'])
        EI_FTR=int(e['ftr'])
        ei_summary.append(ei(EI_DET,EI_FTR))
    low_count=0
    avg_count=0
    high_count=0 
    for i in ei_summary:
        if(i['low']==1):
            low_count+=1
        if(i['average']==1):
            avg_count+=1
        if(i['high']==1):
            high_count+=1
    ei_summary_count=[low_count,avg_count,high_count]


    # #2d. External Outputs (EOs)
    eo_summary=[]
    eo_loop=request.data['EO']
    for e in eo_loop:
        if not e:
            break
        lofEO=e['fileName']
        eoNotes=e['notesAssumptions']
        EO_DET=int(e['det'])
        EO_FTR=int(e['ftr'])
        eo_summary.append(eo(EO_DET,EO_FTR))
    low_count=0
    avg_count=0
    high_count=0 
    for i in eo_summary:
        if(i['low']==1):
            low_count+=1
        if(i['average']==1):
            avg_count+=1
        if(i['high']==1):
            high_count+=1
    eo_summary_count=[low_count,avg_count,high_count]
  

    # #2e. External Queries (EQs)
    eq_summary=[]
    eq_loop=request.data['EQ']
    for e in eq_loop:
        if not e:
            break
        lofEQ=e['queries']
        eqNotes=e['notesAssumptions']
        inEQ_DET=int(e['detInput'])
        inEQ_FTR=int(e['ftrInput'])
        outEQ_DET=int(e['detOutput'])
        outEQ_FTR=int(e['ftrOutput'])
        eq_summary.append(eq(inEQ_DET,inEQ_FTR,outEQ_DET,outEQ_FTR))
    low_count=0
    avg_count=0
    high_count=0 
    for i in eq_summary:
        if(i['low']==1):
            low_count+=1
        if(i['average']==1):
            avg_count+=1
        if(i['high']==1):
            high_count+=1
    eq_summary_count=[low_count,avg_count,high_count]
 

    #3. Review the Unadjusted Function Point Count.
    lowILF  =ilf_summary_count[0]
    avgILF  =ilf_summary_count[1]
    highILF =ilf_summary_count[2]
    lowEIF  =eif_summary_count[0]
    avgEIF  =eif_summary_count[1]
    highEIF =eif_summary_count[2]
    lowEI   =ei_summary_count[0]
    avgEI   =ei_summary_count[1]
    highEI  =ei_summary_count[2]
    lowEO   =eo_summary_count[0]
    avgEO   =eo_summary_count[1]
    highEO  =eo_summary_count[2]
    lowEQ   =eq_summary_count[0]
    avgEQ   =eq_summary_count[1]
    highEQ  =eq_summary_count[2]
    uFPdata=uFP(lowILF,avgILF,highILF,lowEIF,avgEIF,highEIF,lowEI,avgEI,highEI,lowEO,avgEO,highEO,lowEQ,avgEQ,highEQ)
    unadjustedFP=uFPdata[0]
    percentuFP=uFPdata[1]

    #4. Calculate the Value Adjustment Factor.
    #should not be greater than five
    datacCommunications     =int(request.data['Data Communications'])
    dataProcessing          =int(request.data['Distributed Processing'])
    performance             =int(request.data['Performance'])
    heavilyUsedConfiguration=int(request.data['Heavily Used Configuration'])
    transactionRates        =int(request.data['Transaction Rates'])
    onlineDataEntry         =int(request.data['Online Data Entry'])
    designForEndUserEfficiency=int(request.data['Design for End User Efficiency'])
    onlineUpdate            =int(request.data['Online Update'])
    complexProcessing       =int(request.data['Complex Processing'])
    useableInOtherApp       =int(request.data['Usable in Other Applications'])
    installationEase        =int(request.data['Installation Ease'])
    operationalEase         =int(request.data['Operational Ease'])
    multipleSites           =int(request.data['Multiple Sites'])
    facilitateChange        =int(request.data['Facilitate Change'])
    ValAdjFac=VAF(datacCommunications,dataProcessing,performance,heavilyUsedConfiguration,transactionRates,onlineDataEntry,designForEndUserEfficiency,onlineUpdate,complexProcessing,useableInOtherApp,installationEase,operationalEase,multipleSites,facilitateChange)
    TDI=ValAdjFac[0]
    ValAF=ValAdjFac[1]

    # #unadjustedFP
    # #ValAF
    AFPC=unadjustedFP*ValAF
    calibrationFactor=int(request.data['calibration factor'])
    TFPM=AFPC*calibrationFactor
    deliveryRate=int(request.data['delivery rate'])
    daysPerMonth=int(request.data['day per month'])

    HighLvlEffEst=(TFPM/deliveryRate)*daysPerMonth
    
    newData={
        'Customer Name':customerName,
        'Project Name':projectName,
        'Project Code':projectCode,
        'Analyst':analyst,
        'Date':date,

        'UNADJUSTED FUNCTION POINT COUNT (FP)':
        [
            {'name':'Internal Logical Files (ILFs)','complexity':'low',     'count':ilf_summary_count[0], 'weight':'7', 'function point':ilf_summary_count[0]*7},
            {'name':'Internal Logical Files (ILFs)','complexity':'average', 'count':ilf_summary_count[1], 'weight':'10', 'function point':ilf_summary_count[1]*10},
            {'name':'Internal Logical Files (ILFs)','complexity':'high',    'count':ilf_summary_count[2], 'weight':'15', 'function point':ilf_summary_count[2]*15},
            {'name':'External Interface Files (EIFs)','complexity':'low',     'count':eif_summary_count[0], 'weight':'5', 'function point':eif_summary_count[0]*5},
            {'name':'External Interface Files (EIFs)','complexity':'average', 'count':eif_summary_count[1], 'weight':'7', 'function point':eif_summary_count[1]*7},
            {'name':'External Interface Files (EIFs)','complexity':'high',    'count':eif_summary_count[2], 'weight':'10', 'function point':eif_summary_count[2]*10},
            {'name':'External Inputs (EIs)','complexity':'low',     'count':ei_summary_count[0], 'weight':'3', 'function point':ei_summary_count[0]*3},
            {'name':'External Inputs (EIs)','complexity':'average', 'count':ei_summary_count[1], 'weight':'4', 'function point':ei_summary_count[1]*4},
            {'name':'External Inputs (EIs)','complexity':'high',    'count':ei_summary_count[2], 'weight':'6', 'function point':ei_summary_count[2]*6},
            {'name':'External Outputs (EOs)','complexity':'low',     'count':eo_summary_count[0], 'weight':'4', 'function point':eo_summary_count[0]*4},
            {'name':'External Outputs (EOs)','complexity':'average', 'count':eo_summary_count[1], 'weight':'5', 'function point':eo_summary_count[1]*5},
            {'name':'External Outputs (EOs)','complexity':'high',    'count':eo_summary_count[2], 'weight':'7', 'function point':eo_summary_count[2]*7},
            {'name':'External Queries (EQs)','complexity':'low',     'count':eq_summary_count[0], 'weight':'3', 'function point':eq_summary_count[0]*3},
            {'name':'External Queries (EQs)','complexity':'average', 'count':eq_summary_count[1], 'weight':'4', 'function point':eq_summary_count[1]*4},
            {'name':'External Queries (EQs)','complexity':'high',    'count':eq_summary_count[2], 'weight':'6', 'function point':eq_summary_count[2]*6}

        ],         
        
        "Internal Logical Files (ILFs)":(( (ilf_summary_count[0]*7) + (ilf_summary_count[1]*10) + (ilf_summary_count[2]*15) )/unadjustedFP)*100,
        "External Interface Files (EIFs)":(( (eif_summary_count[0]*5) + (eif_summary_count[1]*7) + (eif_summary_count[2]*10) )/unadjustedFP)*100,
        "External Inputs (EIs)":(( (ei_summary_count[0]*3) + (ei_summary_count[1]*4) + (ei_summary_count[2]*6) )/unadjustedFP)*100,
        "External Outputs (EOs)":(( (eo_summary_count[0]*4) + (eo_summary_count[1]*5) + (eo_summary_count[2]*7) )/unadjustedFP)*100,
        "External Queries (EQs)":(( (eq_summary_count[0]*3) + (eq_summary_count[1]*4) + (eq_summary_count[2]*6) ) /unadjustedFP)*100,

        'Total Unadjusted Function Point Count':unadjustedFP,
        'Total Degree of Influence (TDI)':TDI,
        'Value Adjustment Factor (VAF)':ValAF,
        "Adjusted Function Point Count (AFP)":AFPC,
        'calibration factor':calibrationFactor,
        "Total Function Point Measure (TFP)":TFPM,
        "Delivery Rate (DR) in FPs/person month":deliveryRate,
        "Days per person-month (DPM)":daysPerMonth,
        "High Level Effort Estimate (in person-days)":HighLvlEffEst
    }

    
    # object_viewed_signal.send(
    #     newData.__class__, instance=newData, request=request)
    return JsonResponse(newData, safe=False)
