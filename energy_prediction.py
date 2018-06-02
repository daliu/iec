from __future__ import division

from matplotlib.pyplot import step, xlim, ylim, show
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
import pytz

import numpy as np
import pandas as pd
# xbos clients
from xbos import get_client
from xbos.services.hod import HodClient
from xbos.services.mdal import *

from house import IEC # Prediction model. 
import matplotlib.pyplot as plt


now = datetime.utcnow().replace(tzinfo=pytz.timezone("UTC")).astimezone(
    tz=pytz.timezone("America/Los_Angeles"))
print now
start = now.strftime('%Y-%m-%d %H:%M:%S %Z') #period when the data can be collectd
end = '2018-05-05 00:00:00 PST'
WINDOW = '15min'

# data clients. 

# To get the client we usually need a client for BOSSWAVE our decentralized operating system.
# Easiest way to get it is by using get_client() which you import from xbos. Other ways include entity files. 
# https://github.com/SoftwareDefinedBuildings/XBOS. 
# To use xbos make sure to get an entity file from Thanos and to get a executable file which 
# connects you to the system. Also, make sure to set the entity file in your bash_profile with
# export BW2_DEFAULT_ENTITY=path to .ent file

# The MDAL client gets the data from our database. The query to get the data is illustrated by,
# buidling_meteres_query_mdal and lighting_meter_query_mdal.
# Documentation: https://docs.xbos.io/mdal.html#using and https://github.com/gtfierro/mdal <- better
mdal = MDALClient("xbos/mdal", client=get_client())
# HODClient gets the uuid for data. This uses brick which is a language built on SPARQL.
# Can be trick to use.
# To try your own queries go to: corbusier.cs.berkeley.edu:47808. And try the queries we set up below.
# Documentation: for brick: brickschema.org/structure/
# If you need queries, it's best to ask either Thanos or Daniel. 
hod = HodClient("xbos/hod") #IDs of thermostat

# temporal parameters
SITE = "ciee" #building name 

# Brick queries -- graphical rep of building, queries of where to get data
building_meters_query = """SELECT ?meter ?meter_uuid FROM %s WHERE {
    ?meter rdf:type brick:Building_Electric_Meter .
    ?meter bf:uuid ?meter_uuid .
};"""
thermostat_state_query = """SELECT ?tstat ?status_uuid FROM %s WHERE {
    ?tstat rdf:type brick:Thermostat_Status .
    ?tstat bf:uuid ?status_uuid .
};"""
lighting_state_query = """SELECT ?lighting ?state_uuid FROM %s WHERE {
    ?light rdf:type brick:Lighting_State .
    ?light bf:uuid ?state_uuid
};"""
lighting_meter_query = """SELECT ?lighting ?meter_uuid FROM %s WHERE {
    ?meter rdf:type brick:Electric_Meter .
    ?lighting rdf:type brick:Lighting_System .
    ?lighting bf:hasPoint ?meter .
    ?meter bf:uuid ?meter_uuid
};"""

building_meters_query_mdal = {
    "Composition": ["meter", "tstat_state"],  # defined under "Variables"
    "Selectors": [MEAN, MAX],
    "Variables": [
        {
            "Name": "meter",
            "Definition": building_meters_query % SITE, # NOTE: Mdal queries the uuids by itself. it is better practice for now to do that manually by calling hod.do_query(your_query)
            "Units": "kW"
        },
        {
            "Name": "tstat_state",
            "Definition": thermostat_state_query % SITE,
        }
    ],
    "Time": {
        "T0": start, "T1": end,
        "WindowSize": WINDOW,
        "Aligned": True,
    }
}

resp = mdal.do_query(building_meters_query_mdal)   
df = resp['df']

demand = "4d6e251a-48e1-3bc0-907d-7d5440c34bb9" # hard coded


lighting_meter_query_mdal = {
    "Composition": ["lighting"],
    "Selectors": [MEAN],
    "Variables": [
        {
            "Name": "lighting",
            "Definition": lighting_meter_query % SITE,
            "Units": "kW"
        },
    ],
    "Time": {
        "T0": start, "T1": end,
        "WindowSize": WINDOW,
        "Aligned": True,
    }
}
# queries the data from the database with mdal
resp = mdal.do_query(lighting_meter_query_mdal, timeout=200)
lighting_df = resp['df']

# We are using a similarity based approach to predict. This means, that we build a similarity 
# measure to see how similar past days were to the day we are currently experiencing and taking
# a sort of weighted average. This is similar to k-nearest-neighbors. 
# Other approaches can also be tried, like finding different features by which to measure similarity
# or some other ML technique all together (e.g. neural nets or some regression)

# TODO Currently the big issue we are facing is that we are trying to subtract
# the variable consumption from the overall building consumption. This means that for our case
# we want to subtract the HVAC (heating) consumption from the building consumption, since that is 
# something we control and it doesn't make sense to learn it.
# Now, the issue is finding out what the exact consumption for heating and cooling is which Marco is doing and should
# be followed up with.  


# TODO find out the right values. Marco is working on this
heating_consume = .3  # in kW. 
cooling_consume = 5.  # kW
meter = df.columns[0]
all_but_meter = df.columns[1:]

# amount to subtract for heating, cooling
# TODO some values become negative after the following operation, which should not happen. Marco is checking the data.
h = (df[all_but_meter] == 1).apply(sum, axis=1) * heating_consume
c = (df[all_but_meter] == 2).apply(sum, axis=1) * cooling_consume

meterdata = df[meter] - h - c

# NOTE: the following is some data manipulation and setting up the right time zone.
# Followed by, predicting the data with the IEC model. 

# unit conversion
meterdata = meterdata / (1000 * 60)
# print meterdata
print lighting_df.describe()
meterdata = pd.DataFrame.from_records({'House Consumption': meterdata})
print meterdata.describe()
print meterdata['House Consumption']

meterdata = meterdata.tz_convert(pytz.timezone("America/Los_Angeles"))  # /??????????????????????????????????
yesterday = now - timedelta(hours=500)

# Prediction happening here and should be looked at.
print("yesterday meterdata ------------------------------------- " + str(yesterday))
print(meterdata[:yesterday])
prediction = IEC(meterdata[:yesterday], prediction_window=12 * 60).predict(["STLF"])
# prices.index = data.index
# prices['US'] = Dollar/Kwh

index = np.arange(12 * 60)
plt.plot(index, prediction[["STLF"]], label="Energy Prediction")
plt.plot(index, meterdata[["House Consumption"]][-12 * 60:], label="Ground Truth")
plt.xlabel('Predictive horizon (Minutes)')
plt.ylabel(r'KWh')
plt.legend()
plt.show()
# data * prices = posa plirwses
