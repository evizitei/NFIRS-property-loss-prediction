from datetime import datetime

def intfix(val):
    if val is None:
        return -1
    return int(val)

expensive_loss_states = ["AK", "AR", "IL", "IN", "KY", "MN", "MS", "NY", "PA", "TN", "WV"]
cheap_loss_states = ["DE", "HI", "ME","NV"]

def state_expense_level(state):
    if state in expensive_loss_states:
        return 2
    if state in cheap_loss_states:
        return 0
    return 1


def map_inc_type(input_type):
    leader = int(str(input_type)[0:2])
    if leader in [11, 12, 13, 14, 15, 17]:
        return leader
    else:
        return 10 # other categories are similarly low damage

def aid_level(input_aid):
    if input_aid == "N":
        return 0
    aid_int = int(input_aid)
    if aid_int in [1,3]:
        return 1
    if aid_int in [2,4]:
        return 2
    return 3

def normalize_ts(ts):
    string = str(ts)
    if len(string) == 11:
        string = "0" + string
    return string

def map_ts_to_hour(ts):
    return int(normalize_ts(ts)[8:10])

def map_ts_to_month(ts):
    return int(normalize_ts(ts)[0:2])

def map_ts_to_year(ts):
    return int(normalize_ts(ts)[4:8])

def map_ts_to_day_of_month(ts):
    return int(normalize_ts(ts)[2:4])

def map_ts_to_hour(ts):
    return int(normalize_ts(ts)[8:10])

def map_ts_to_minute(ts):
    return int(normalize_ts(ts)[10:12])

def ts_to_dt(ts):
    return datetime(map_ts_to_year(ts), map_ts_to_month(ts), map_ts_to_day_of_month(ts),
             map_ts_to_hour(ts), map_ts_to_minute(ts))


def hour_group(alarm):
    hour = map_ts_to_hour(alarm)
    if hour > 6 and hour <= 14:
        return 0 # morning
    if hour > 17 and hour <= 22:
        return 0 # evening, same as morn
    if hour > 14 and hour <= 17:
        return 1 # afternoon
    if hour <= 5 or hour >= 23:
        return 3 # overnigh
    return 0

def controlled_time(alarm, controlled):
    try:
        return (ts_to_dt(controlled) - ts_to_dt(alarm)).seconds
    except ValueError:
        return 0

def detector_flag(det_state):
    if det_state == "2":
        return 1
    return 0

def hazmat_group(hazrel):
    if hazrel is None or hazrel in ["N", ""]:
        return 0
    val = int(hazrel)
    if val in [1,4,7]:
        return 1
    if val in [2,3]:
        return 2
    if val in [6,8]:
        return 3
    if val == 5:
        return 4
    return 0

def mixed_use_group(mu):
    if mu is None or mu in ["", "00", "NN"]:
        return 0
    if mu in ["33", "63"]:
        return 1
    if mu in ["10", "20", "51", "53"]:
        return 2
    if mu in ["58", "59", "60"]:
        return 3
    if mu in ["40", "65"]:
        return 4
    return 0

def property_use_group(pu):
    if pu is None or pu in ["NNN", "UUU"]:
        return 0
    if pu in ["2", "3","9"]:
        return 1
    if pu in ["0", "1", "5"]:
        return 2
    if pu in ["7", "8"]:
        return 3
    if pu in ["4", "6"]:
        return 4
    return 0

def area_origin_group(ao):
    if ao is None or ao in ["UU", "uu", ""]:
        return 0
    if ao in ["8", "9"]:
        return 1
    if ao in ["0","3","5", "6"]:
        return 2
    if ao in ["2", "4", "7"]:
        return 3
    if ao in ["1"]:
        return 4
    return 0

def heat_source_group(hs):
    if hs is None or hs in ["UU", "uu", ""]:
        return 0
    if hs in ["0","1","4","5","6"]:
        return 1
    if hs in ["7","8","9"]:
        return 2
    return 0

def ignition_group(ci):
    if ci is None or ci in ["","U","u"]:
        return 0
    if ci in ["1","2","3"]:
        return 1
    if ci in ["4"]:
        return 2
    if ci in ["0", "5"]:
        return 3
    return 0

def fire_spread_group(fs):
    if fs is None or fs == "":
        return 0
    return int(fs)

def structure_type_group(st):
    if st is None or st in [""]:
        return 0
    val = int(st)
    if val in [5,6]:
        return 1
    if val in [7,8]:
        return 2
    if val in [3]:
        return 3
    if val in [0,2,4]:
        return 4
    if val in [1]:
        return 5
    return 0

def structure_status_group(ss):
    if ss is None or ss in ["", "U"]:
        return 0
    val = int(ss)
    if val in [0,3,7]:
        return 1
    if val in [2]:
        return 2
    if val in [1,4,6]:
        return 3
    if val in [5]:
        return 4
    return 0

def aes_system_group(aes):
    if aes is None or aes in ["", "U"]:
        return 0
    if aes in ["N"]:
        return 1
    if aes in ["1"]:
        return 2
    if aes in ["2"]:
        return 3
    return 0

def one_hot_encode(out_rec, in_rec, field, count):
    val = in_rec[field]
    for i in range(count):
        out_field = field + "_" + str(i)
        if i == val:
            out_rec[out_field] = 1
        else:
            out_rec[out_field] = 0

def one_hot_encode_incident_type(out_rec, in_rec):
    mapping = {
        10: 0,
        11: 1,
        12: 2,
        13: 3,
        14: 4,
        15: 5,
        17: 6,
    }
    val = mapping[in_rec["INC_TYPE"]]
    for i in range(7):
        out_field = "INC_TYPE_" + str(i)
        if i == val:
            out_rec[out_field] = 1
        else:
            out_rec[out_field] = 0

def feature_scale(out_rec, in_rec, field, min_val, max_val):
    val = in_rec[field]
    out_val = (val - min_val)/max_val
    out_rec[field] = out_val

def fix_val(rec, field, median):
    if rec[field] <= 0:
        rec[field] = median

def fill_in_missing_values(rec):
    fix_val(rec, "CONTROLLED_TIME", 1080)
    fix_val(rec, "CLEAR_TIME", 3300)
    fix_val(rec, "SUPPRESSION_APPARATUS", 3)
    fix_val(rec, "SUPPRESSION_PERSONNEL", 8)
    fix_val(rec, "SQUARE_FEET", 1200)
    # PROP_VAL Median > 1000 => 20000
    if rec["PROP_VALUE"] <= 0:
        rec["PROP_VALUE"] = 20000
    return rec

def normalize_features(clean_rec, normalized_rec):
    one_hot_encode(normalized_rec, clean_rec, 'STATE_EXPENSE', 3)
    one_hot_encode_incident_type(normalized_rec, clean_rec)
    one_hot_encode(normalized_rec, clean_rec, 'AID', 4)
    one_hot_encode(normalized_rec, clean_rec, 'HOUR_GROUP', 4)
    feature_scale(normalized_rec, clean_rec, 'CONTROLLED_TIME', 60.0, 11280.0)
    feature_scale(normalized_rec, clean_rec, 'CLEAR_TIME', 60.0, 20160.0)
    feature_scale(normalized_rec, clean_rec, 'SUPPRESSION_APPARATUS', 1.0, 301.0)
    feature_scale(normalized_rec, clean_rec, 'SUPPRESSION_PERSONNEL', 1.0, 67.0)
    feature_scale(normalized_rec, clean_rec, 'PROP_VALUE', 1.0, 23733600.0)
    normalized_rec["DETECTOR_FLAG"] = clean_rec["DETECTOR_FLAG"]
    one_hot_encode(normalized_rec, clean_rec, 'HAZMAT_RELEASE', 5)
    one_hot_encode(normalized_rec, clean_rec, 'MIXED_USE', 5)
    normalized_rec["HEAT_SOURCE"] = clean_rec["HEAT_SOURCE"]
    one_hot_encode(normalized_rec, clean_rec, 'IGNITION', 4)
    one_hot_encode(normalized_rec, clean_rec, 'FIRE_SPREAD', 6)
    one_hot_encode(normalized_rec, clean_rec, 'STRUCTURE_TYPE', 6)
    one_hot_encode(normalized_rec, clean_rec, 'STRUCTURE_STATUS', 5)
    feature_scale(normalized_rec, clean_rec, 'SQUARE_FEET', 1.0, 1690000.0)
    one_hot_encode(normalized_rec, clean_rec, 'AES_SYSTEM', 4)
    return normalized_rec
