import re

def rename_feat(feature):
    if 'temp' in feature or 'power' in feature:
        return feature
    feature = feature.replace("node_memory_", "mem:")
    feature = feature.replace("node_disk_", "disk:")
    feature = re.sub(r"^(?!mem:|disk:)", "cpu:", feature)
    feature = feature.replace("(cumsum)", " (cumsum)")
    feature = feature.replace("(diff)", " (diff)")
    feature = feature.replace("_", " ")

    return feature
