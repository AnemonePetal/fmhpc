from datetime import datetime, timedelta

def first_hours(time_range_list,hours=2):
    result = []
    for time_range in time_range_list:
        start_time = datetime.strptime(time_range[0], "%Y-%m-%d %H:%M:%S")
        end_time = start_time + timedelta(hours=hours)
        result.append((start_time.strftime("%Y-%m-%d %H:%M:%S"), end_time.strftime("%Y-%m-%d %H:%M:%S")))
    return result

def last_hours(time_range_list, hours=1):
    result = []
    for time_range in time_range_list:
        end_time = datetime.strptime(time_range[1], "%Y-%m-%d %H:%M:%S")
        start_time = end_time - timedelta(hours=hours)
        result.append((start_time.strftime("%Y-%m-%d %H:%M:%S"), end_time.strftime("%Y-%m-%d %H:%M:%S")))
    return result

def shift_hours_range(time_range_list, hours=1,direction=1):
    result = []
    for time_range in time_range_list:
        start_time = datetime.strptime(time_range[0], "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(time_range[1], "%Y-%m-%d %H:%M:%S")
        if direction == 1:
            start_time = start_time + timedelta(hours=hours)
            end_time = end_time + timedelta(hours=hours)
        else:
            start_time = start_time - timedelta(hours=hours)
            end_time = end_time - timedelta(hours=hours)
        result.append((start_time.strftime("%Y-%m-%d %H:%M:%S"), end_time.strftime("%Y-%m-%d %H:%M:%S")))
    return result

def shift_time_range(time_range_list, timedelta, direction=1):
    result = []
    for time_range in time_range_list:
        start_time = datetime.strptime(time_range[0], "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(time_range[1], "%Y-%m-%d %H:%M:%S")
        if direction == 1:
            start_time = start_time + timedelta
            end_time = end_time + timedelta
        else:
            start_time = start_time - timedelta
            end_time = end_time - timedelta
        result.append((start_time.strftime("%Y-%m-%d %H:%M:%S"), end_time.strftime("%Y-%m-%d %H:%M:%S")))
    return result

def shift_hours_str(time_str, hours=1,direction=1):
    time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    if direction == 1:
        time = time + timedelta(hours=hours)
    else:
        time = time - timedelta(hours=hours)
    return time.strftime("%Y-%m-%d %H:%M:%S")

def shift_time_str(time_str, timedelta, direction=1):
    time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
    if direction == 1:
        time = time + timedelta
    else:
        time = time - timedelta
    return time.strftime("%Y-%m-%d %H:%M:%S")

def str2time(time_str):
    return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")

if __name__ == "__main__":
    time_range_list = [
        ('2023-05-23 06:00:00', '2023-05-23 17:00:00')
    ]
    print(last_hours(first_hours(time_range_list)))
