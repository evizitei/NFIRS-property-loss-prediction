# NFIRS property loss prediction

See project_report.md for project overview

### Requirements
*make sure to pip install these all before running anything*

tensorflow (https://www.tensorflow.org/)
dbfread (https://dbfread.readthedocs.io)
dataset (https://dataset.readthedocs.io/en/latest/)

### Exploring Data files

you can download the data for this project from

http://www.fema.gov/media-library-data/20130726-2126-31471-8394/nfirs_2011_120612.zip

And then to use any of the following commands
you can unzip that file into the "data" folder
in this project directory (which is gitignored)

All the data files are in .dbf format, so
you need dbfread in order to consume them.

```
from dbfread import DBF
incidents_table = DBF('data/basicincident.dbf')
len(incidents_table) # -> 2,311,716

# this will be a great deal of output
# but it's useful for seeing how records are structured
for record in table:
    print(record)
```

the basicincident.dbf file is the most interesting one, and it's
records look like this:

```
OrderedDict([
  (u'STATE', u'AK'),
  (u'FDID', u'13425'),
  (u'INC_DATE', 3292011),
  (u'INC_NO', u'0000051'),
  (u'EXP_NO', 0),
  (u'VERSION', u'5.0'),
  (u'DEPT_STA', u'8'),
  (u'INC_TYPE', u'461'),
  (u'ADD_WILD', u'N'),
  (u'AID', u'1'),
  (u'ALARM', 32920110944),
  (u'ARRIVAL', 32920110950),
  (u'INC_CONT', 32920111004),
  (u'LU_CLEAR', 32920111016),
  (u'SHIFT', u''),
  (u'ALARMS', u'0'),
  (u'DISTRICT', u''),
  (u'ACT_TAK1', u'86'),
  (u'ACT_TAK2', u''),
  (u'ACT_TAK3', u''),
  (u'APP_MOD', u'Y'),
  (u'SUP_APP', 0),
  (u'EMS_APP', 0),
  (u'OTH_APP', 4),
  (u'SUP_PER', 0),
  (u'EMS_PER', 0),
  (u'OTH_PER', 5),
  (u'RESOU_AID', u'Y'),
  (u'PROP_LOSS', 18000),
  (u'CONT_LOSS', 0),
  (u'PROP_VAL', 18000),
  (u'CONT_VAL', 0),
  (u'FF_DEATH', 0),
  (u'OTH_DEATH', 0),
  (u'FF_INJ', 0),
  (u'OTH_INJ', 0),
  (u'DET_ALERT', u'U'),
  (u'HAZ_REL', u'N'),
  (u'MIXED_USE', u'NN'),
  (u'PROP_USE', u'419'),
  (u'CENSUS', u'')])
```

They use numerical codes a lot here so the data is going to need
some cleaning.  Additionally, it's hard to explore
in dbf format, so it seems worth shoving the relevant files
 into sqlite tables with indexes for easier querying.
There's a useful helper script for doing this
(make sure your sqlite folder exists):

`./bin/data_to_sqlite`

it will take some time to run, since there are a few
million basic incidents and 600k fire incidents,
and it's processing them iteratively.  This takes
on the order of 3 hours, and so the final dataset
will be provided as a google drive file here:

*_include file link at some point_*

exploring the data in a sql database is a little easier:

```
import dataset

db = dataset.connect('sqlite:///./sqlite/incidents.sqlite')
basic_table = db['basic_incidents']
fire_table = db['fire_incidents']

# return one row from each table
basic_table.find_one()
fire_table.find_one()

# find by column values:
basic_table.find(FDID=u'11100', INC_NO=u'391')
```
