from django.db import models


class projectapi(models.Model):
    
    Normalized_Work_Effort = models.FloatField()
    Summary_Work_Effort = models.FloatField()
    Normalised_Work_Effort_Level_1 = models.FloatField()
    Effort_Unphased = models.FloatField()
    Adjusted_Function_Points = models.FloatField()
    Functional_Size = models.FloatField()
    Added_count = models.FloatField()
    Input_count = models.FloatField()
    Max_Team_Size = models.FloatField()
    Speed_of_Delivery = models.FloatField()
    Development_Type_New_Development = models.FloatField()
    Language_Type_3GL = models.FloatField()
