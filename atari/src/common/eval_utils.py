# calculate dead ratio
def get_param_cnt_ratio(cnt_dict, param_dict, post_fix='', activation=False, name_idx=3):
    param_cnt_ratio = {}

    _cnt_dict, _param_dict = {}, {}
    for layer_name, _ in cnt_dict.items():
        # when computing activation output, only consider the output after ReLU
        if activation:
            name_idx = 4
            if 'ReLU' not in layer_name.split('.')[2]:
                continue

        if '.'.join(layer_name.split('.')[:1]) in _cnt_dict:
            _cnt_dict['.'.join(layer_name.split('.')[:1])] += cnt_dict[layer_name]
            _param_dict['.'.join(layer_name.split('.')[:1])] += param_dict[layer_name]
        
        if '.'.join(layer_name.split('.')[:2]) in _cnt_dict:
            _cnt_dict['.'.join(layer_name.split('.')[:2])] += cnt_dict[layer_name]
            _param_dict['.'.join(layer_name.split('.')[:2])] += param_dict[layer_name]
        
        if '.'.join(layer_name.split('.')[:name_idx]) in _cnt_dict:
            _cnt_dict['.'.join(layer_name.split('.')[:name_idx])] += cnt_dict[layer_name]
            _param_dict['.'.join(layer_name.split('.')[:name_idx])] += param_dict[layer_name]
        
        if '.'.join(layer_name.split('.')[:1]) not in _cnt_dict:
            _cnt_dict['.'.join(layer_name.split('.')[:1])] = cnt_dict[layer_name]
            _param_dict['.'.join(layer_name.split('.')[:1])] = param_dict[layer_name]
        
        if '.'.join(layer_name.split('.')[:2]) not in _cnt_dict:
            _cnt_dict['.'.join(layer_name.split('.')[:2])] = cnt_dict[layer_name]
            _param_dict['.'.join(layer_name.split('.')[:2])] = param_dict[layer_name]
        
        if '.'.join(layer_name.split('.')[:name_idx]) not in _cnt_dict:
            _cnt_dict['.'.join(layer_name.split('.')[:name_idx])] = cnt_dict[layer_name]
            _param_dict['.'.join(layer_name.split('.')[:name_idx])] = param_dict[layer_name]

    for k, v in _cnt_dict.items():
        num_cnt = v
        num_param = _param_dict[k] 

        param_cnt_ratio [k + '_' + post_fix] = num_cnt / num_param

    if ('backbone' in _cnt_dict) and ('policy' in _cnt_dict):
        param_cnt_ratio['total' + '_' + post_fix] = (
            (_cnt_dict['backbone'] + _cnt_dict['policy']) / (_param_dict['backbone'] + _param_dict['policy'])
        )
        
    return param_cnt_ratio
