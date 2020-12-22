from django.http import JsonResponse
from rest_framework.decorators import api_view
# from analytics.signals import object_viewed_signal
from rest_framework import response, decorators, permissions, status
# Create your views here.

@decorators.api_view(["POST"])
@decorators.permission_classes([permissions.AllowAny])
def COSMIC(request):
    userInputs=     int(request.data['userInputs'])
    userOutputs=    int(request.data['userOutputs'])
    userInquiries=  int(request.data['userInquiries'])
    files=          int(request.data['files'])
    extInterface=   int(request.data['externalInterface'])
    inputChoice=    request.data['inputChoice']
    outputChoice=    request.data['outputChoice']
    inquiryChoice=  request.data['inquiryChoice']
    fileChoice=     request.data['fileChoice']
    externalChoice= request.data['externalChoice']

    if(inputChoice=='simple'):
        userInputs*=3
    if(inputChoice=='average'):
        userInputs*=4
    if(inputChoice=='complex'):
        userInputs*=6
    if(outputChoice=='simple'):
        userOutputs*=4
    if(outputChoice=='average'):
        userOutputs*=5
    if(outputChoice=='complex'):
        userOutputs*=7
    if(inquiryChoice=='simple'):
        userInquiries*=3
    if(inquiryChoice=='average'):
        userInquiries*=4
    if(inquiryChoice=='complex'):
        userInquiries*=6
    if(fileChoice=='simple'):
        files*=7
    if(fileChoice=='average'):
        files*=10
    if(fileChoice=='complex'):
        files*=15
    if(externalChoice=='simple'):
        extInterface*=5
    if(externalChoice=='average'):
        extInterface*=7
    if(externalChoice=='complex'):
        extInterface*=10

    count=userInputs+userOutputs+userInquiries+files+extInterface

    backupRecovery=     int(request.data['Does the system require reliable backup and recovery?'])
    communication=      int(request.data['Are data communications required?'])
    distProcessing=     int(request.data['Are there distributed processing functions?'])
    performance=        int(request.data['Is performance critical?'])
    operationEnv=       int(request.data['Will the system run in an existing, heavily utilized operational environment?'])
    dataEntry=          int(request.data['Does the system require on-line data entry?'])
    inputTranscript=    int(request.data['Does the on-line data entry require the input transaction to be built over multiple screens or operations?'])
    filesUpdated=       int(request.data['Are the master files updated on-line?'])
    complexity=         int(request.data['Are the inputs, outputs, files, or inquiries complex?'])
    processComplex=     int(request.data['Is the internal processing complex?'])
    reuseable=          int(request.data['Is the code designed to be reusable?'])
    conversion=         int(request.data['Are conversion and installation included in the design?'])
    multiInstall=       int(request.data['Is the system designed for multiple installations in different organizations?'])
    easeOfUse=          int(request.data['Is the application designed to facilitate change and ease of use by the user?'])

    sum=backupRecovery+communication+distProcessing+performance+operationEnv+dataEntry+inputTranscript+filesUpdated+complexity+processComplex+reuseable+conversion+multiInstall+easeOfUse

    FPM=count*(0.65+0.01*sum)

    newData={
            'userInputs':     userInputs,
            'userOutputs':    userOutputs,
            'userInquiries':  userInquiries,
            'files':          files,
            'extInterface':   extInterface,
            'count total':    count,
            ' sum of Fi':     sum,
            'Funtion Point Metric':FPM  

    }
    # object_viewed_signal.send(
    #     newData.__class__, instance=newData, request=request)
    return JsonResponse(newData, safe=False)