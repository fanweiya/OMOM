def parent(number):
    masterlabel={}
    for i,value in enumerate([i.split('---')[number] for i in label]):
            if value=='zheng-chang':
                masterlabel[i] = 1
            else:
                masterlabel[i] = 0
    return masterlabel

