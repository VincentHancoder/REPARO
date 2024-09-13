# Evaluation details
注意：目前只能通过 config.json 来获取保存路径，之后可以修改保存路径为，`results/furniture-xxx/case-name`，这样就能够通过`furniture-xxx`定位整组实验的路径。
**确定方式之后要对此代码进行修改**
```
config_path_list = [x for x in config_path_list if "all" in x]
```
## Usage
```
python eval_2d.py  --config "config/furniture_rgbxy.json" \ 
    --targets_dir "data/GSO_gt_mod" \
    --target_material_dir "data/GSO_gt" \
    --save_dir "results/eval"
```
* `config`: it includes necessary info
* `save_dir`: the results will be saved to `save_dir/save_name` with the below structure (`save_name` equals `task_name` in config file.):
```
--save_dir
    --save_name
        --BEDROOM_CLASSIC
            --mesh
            --views
                --predict
                    --view_0.png
                    ...
                --target
                    --view_0.png
                    ...
        ...
```
After evaluation, the information will be saved to `save_dir/save_name/meta_info.json`, and the summary results will be saved to `save_dir/save_name/results.txt`
